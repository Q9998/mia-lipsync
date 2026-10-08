#!/usr/bin/env python3
"""Mia 自建口型同步服务：Wav2Lip + GFPGAN，异步任务接口

链路：Wav2Lip 生成口型 -> GFPGAN 逐帧人脸修复（paste_back，只动人脸）
      -> ffmpeg 把原音频 mux 回来。GFPGAN 是画质/嘴部修复的关键一步，
      之前版本缺了它，直接导致"方块模糊嘴"。
"""
import os, subprocess, json, uuid, base64, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

WORKDIR = "/tmp/lipsync"
os.makedirs(WORKDIR, exist_ok=True)
JOBS = {}
# 完成文件的保留时长（秒）：供分片下载，超时由清道夫删除
RESULT_TTL = 7200


def _reaper():
    """清道夫：删除过期的结果文件"""
    import time
    while True:
        time.sleep(600)
        now = time.time()
        for jid, job in list(JOBS.items()):
            fp = job.get("file")
            if job.get("status") == "done" and fp and os.path.exists(fp):
                if now - os.path.getmtime(fp) > RESULT_TTL:
                    try:
                        os.remove(fp)
                    except OSError:
                        pass
                    job.pop("file", None)


threading.Thread(target=_reaper, daemon=True).start()

WAV2LIP_CKPT = "/opt/wav2lip/checkpoints/wav2lip_gan.pth"
GFPGAN_CKPT = "/opt/gfpgan/experiments/pretrained_models/GFPGANv1.4.pth"
GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"

_gfpgan_restorer = None
_restorer_lock = threading.Lock()


def get_restorer():
    """懒加载 GFPGAN（首次调用时初始化；权重缺失则现场下载）"""
    global _gfpgan_restorer
    if _gfpgan_restorer is None:
        with _restorer_lock:
            if _gfpgan_restorer is None:
                if not os.path.exists(GFPGAN_CKPT) or os.path.getsize(GFPGAN_CKPT) < 100_000_000:
                    os.makedirs(os.path.dirname(GFPGAN_CKPT), exist_ok=True)
                    print("downloading GFPGAN weights...", flush=True)
                    subprocess.run(["curl", "-fSL", GFPGAN_URL, "-o", GFPGAN_CKPT],
                                   check=True, timeout=900)
                from gfpgan import GFPGANer
                _gfpgan_restorer = GFPGANer(
                    model_path=GFPGAN_CKPT,
                    upscale=1, arch="clean", channel_multiplier=2,
                    bg_upsampler=None)
                print("GFPGAN restorer ready", flush=True)
    return _gfpgan_restorer


def gfpgan_enhance(src_mp4, dst_mp4):
    """逐帧 GFPGAN 人脸修复，完成后把原音频 mux 回来。返回处理的帧数。"""
    import cv2
    restorer = get_restorer()
    cap = cv2.VideoCapture(src_mp4)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp_v = dst_mp4.replace(".mp4", "-nov.mp4")
    vw = cv2.VideoWriter(tmp_v, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            _, _, restored = restorer.enhance(
                frame, has_aligned=False, only_center_face=False, paste_back=True)
        except Exception:
            restored = None
        vw.write(restored if restored is not None else frame)
        n += 1
    cap.release()
    vw.release()
    if n == 0:
        raise RuntimeError("gfpgan: no frames read from " + src_mp4)
    cmd = ["ffmpeg", "-y", "-i", tmp_v, "-i", src_mp4,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18",
           "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0?", "-shortest", dst_mp4]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    try:
        os.remove(tmp_v)
    except OSError:
        pass
    if r.returncode != 0 or not os.path.exists(dst_mp4):
        raise RuntimeError("ffmpeg mux failed: " + r.stderr[-500:])
    return n


def run_job(job_id, face_b64, audio_b64):
    JOBS[job_id] = {"status": "processing"}
    face_path = os.path.join(WORKDIR, f"face-{job_id}.mp4")
    audio_path = os.path.join(WORKDIR, f"audio-{job_id}.wav")
    out_path = os.path.join(WORKDIR, f"out-{job_id}.mp4")
    tmp = out_path.replace(".mp4", "-wl.mp4")
    try:
        with open(face_path, "wb") as f:
            f.write(base64.b64decode(face_b64))
        with open(audio_path, "wb") as f:
            f.write(base64.b64decode(audio_b64))
        # 1) Wav2Lip 口型生成
        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True：防 A4000 上人脸检测时显存碎片 OOM
        wl_env = dict(os.environ)
        wl_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        cmd = ["python3", "/opt/wav2lip/inference.py",
               "--checkpoint_path", WAV2LIP_CKPT,
               "--face", face_path, "--audio", audio_path, "--outfile", tmp]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=wl_env)
        if r.returncode != 0:
            JOBS[job_id] = {"status": "failed", "error": "wav2lip: " + r.stderr[-1000:]}
            return
        # 2) GFPGAN 人脸修复（修嘴部模糊/方块嘴，保画质）——之前缺的就是这一步
        try:
            n = gfpgan_enhance(tmp, out_path)
        except Exception as e:
            JOBS[job_id] = {"status": "failed", "error": "gfpgan: " + str(e)[:1000]}
            return
        JOBS[job_id] = {"status": "done", "file": out_path,
                        "size": os.path.getsize(out_path),
                        "frames_enhanced": n}
    except Exception as e:
        JOBS[job_id] = {"status": "failed", "error": str(e)}
    finally:
        # 成功时保留 out_path 供分片下载（清道夫超时清理）；失败/临时文件删掉
        for p in (face_path, audio_path, tmp):
            try:
                os.remove(p)
            except OSError:
                pass
        if JOBS.get(job_id, {}).get("status") != "done":
            try:
                os.remove(out_path)
            except OSError:
                pass


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path).path
        q = urlparse(self.path).query
        if p == "/health":
            self._json({
                "status": "ok",
                "engines": ["wav2lip", "gfpgan"],
                "wav2lip_checkpoint": os.path.exists(WAV2LIP_CKPT),
                "gfpgan_weights": os.path.exists(GFPGAN_CKPT),
            })
        elif p == "/gpu":
            # GPU 自检：供外部验证 CUDA 是否真正可用
            try:
                import torch
                self._json({
                    "cuda_available": torch.cuda.is_available(),
                    "torch_cuda_version": torch.version.cuda,
                    "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                })
            except Exception as e:
                self._json({"cuda_available": False, "error": str(e)})
        elif p.startswith("/dl/"):
            # 分片下载：/dl/<job_id>?off=0&len=1572864，返回 raw 二进制
            from urllib.parse import parse_qs
            jid = p.split("/")[-1]
            job = JOBS.get(jid, {})
            fp = job.get("file")
            if job.get("status") != "done" or not fp or not os.path.exists(fp):
                self._json({"error": "not ready"}, 404)
                return
            qs = parse_qs(q)
            try:
                off = int(qs.get("off", ["0"])[0])
                ln = int(qs.get("len", ["1572864"])[0])
            except ValueError:
                self._json({"error": "bad range"}, 400)
                return
            ln = min(ln, 4 * 1024 * 1024)
            with open(fp, "rb") as f:
                f.seek(off)
                data = f.read(ln)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif p.startswith("/status/"):
            jid = p.split("/")[-1]
            self._json(JOBS.get(jid, {"status": "unknown"}))
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/lipsync":
            self._json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length))
        jid = uuid.uuid4().hex[:8]
        t = threading.Thread(target=run_job, args=(jid, req["face_b64"], req["audio_b64"]), daemon=True)
        t.start()
        self._json({"status": "accepted", "job_id": jid})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"listening on {port}", flush=True)
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
