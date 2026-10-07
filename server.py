#!/usr/bin/env python3
"""Mia 自建口型同步服务：Wav2Lip + GFPGAN，异步任务接口"""
import os, subprocess, json, uuid, base64, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

WORKDIR = "/tmp/lipsync"
os.makedirs(WORKDIR, exist_ok=True)
JOBS = {}

def run_job(job_id, face_b64, audio_b64):
    JOBS[job_id] = {"status": "processing"}
    face_path = os.path.join(WORKDIR, f"face-{job_id}.mp4")
    audio_path = os.path.join(WORKDIR, f"audio-{job_id}.wav")
    out_path = os.path.join(WORKDIR, f"out-{job_id}.mp4")
    try:
        with open(face_path, "wb") as f:
            f.write(base64.b64decode(face_b64))
        with open(audio_path, "wb") as f:
            f.write(base64.b64decode(audio_b64))
        ckpt = "/opt/wav2lip/checkpoints/wav2lip_gan.pth"
        tmp = out_path.replace(".mp4", "-wl.mp4")
        cmd = ["python3", "/opt/wav2lip/inference.py",
               "--checkpoint_path", ckpt,
               "--face", face_path, "--audio", audio_path, "--outfile", tmp]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if r.returncode != 0:
            JOBS[job_id] = {"status": "failed", "error": r.stderr[-1000:]}
            return
        os.rename(tmp, out_path)
        with open(out_path, "rb") as f:
            JOBS[job_id] = {"status": "done", "out_b64": base64.b64encode(f.read()).decode()}
    except Exception as e:
        JOBS[job_id] = {"status": "failed", "error": str(e)}
    finally:
        for p in (face_path, audio_path, out_path):
            try: os.remove(p)
            except: pass

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
        if p == "/health":
            ckpt = "/opt/wav2lip/checkpoints/wav2lip_gan.pth"
            self._json({"status": "ok", "engines": ["wav2lip"], "checkpoint": os.path.exists(ckpt)})
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
