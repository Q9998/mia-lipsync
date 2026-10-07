#!/usr/bin/env python3
"""Mia 自建口型同步服务：Wav2Lip + GFPGAN，HTTP 接口（支持 base64 上传/下载）"""
import os, subprocess, json, uuid, base64
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

WORKDIR = "/tmp/lipsync"
os.makedirs(WORKDIR, exist_ok=True)

class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            # 检查 checkpoint 是否存在
            ckpt = "/opt/wav2lip/checkpoints/wav2lip_gan.pth"
            self._json({
                "status": "ok",
                "engines": ["wav2lip", "gfpgan"],
                "checkpoint": os.path.exists(ckpt),
            })
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/lipsync":
            self._json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length))

        job = uuid.uuid4().hex[:8]
        face_path = os.path.join(WORKDIR, f"face-{job}.mp4")
        audio_path = os.path.join(WORKDIR, f"audio-{job}.wav")
        out_path = os.path.join(WORKDIR, f"out-{job}.mp4")

        try:
            # 保存上传的文件
            with open(face_path, "wb") as f:
                f.write(base64.b64decode(req["face_b64"]))
            with open(audio_path, "wb") as f:
                f.write(base64.b64decode(req["audio_b64"]))
        except Exception as e:
            self._json({"error": f"decode failed: {e}"}, 400)
            return

        # 1. Wav2Lip
        tmp = out_path.replace(".mp4", "-wav2lip.mp4")
        ckpt = "/opt/wav2lip/checkpoints/wav2lip_gan.pth"
        if not os.path.exists(ckpt):
            self._json({"error": "checkpoint missing"}, 500)
            return
        cmd = [
            "python3", "/opt/wav2lip/inference.py",
            "--checkpoint_path", ckpt,
            "--face", face_path, "--audio", audio_path, "--outfile", tmp,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            self._json({"error": "wav2lip timeout"}, 500)
            return
        if r.returncode != 0:
            self._json({"error": "wav2lip failed", "log": r.stderr[-2000:]}, 500)
            return

        # 2. 输出（GFPGAN 可选后处理，暂跳过）
        os.rename(tmp, out_path)
        with open(out_path, "rb") as f:
            out_b64 = base64.b64encode(f.read()).decode()

        # 清理
        for p in (face_path, audio_path, out_path):
            try: os.remove(p)
            except: pass

        self._json({"status": "done", "out_b64": out_b64})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"listening on {port}", flush=True)
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
