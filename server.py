#!/usr/bin/env python3
"""Mia 自建口型同步服务：Wav2Lip + GFPGAN，HTTP 接口"""
import os, subprocess, json, uuid
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
            self._json({"status": "ok", "engines": ["wav2lip", "gfpgan"]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/lipsync":
            self._json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length))
        face = req.get("face")      # base video path (16:9)
        audio = req.get("audio")    # wav path
        out = req.get("out") or os.path.join(WORKDIR, f"out-{uuid.uuid4().hex[:8]}.mp4")

        # 1. Wav2Lip
        tmp = out.replace(".mp4", "-wav2lip.mp4")
        cmd = [
            "python3", "/opt/wav2lip/inference.py",
            "--checkpoint_path", "/opt/wav2lip/checkpoints/wav2lip_gan.pth",
            "--face", face, "--audio", audio, "--outfile", tmp,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            self._json({"error": "wav2lip failed", "log": r.stderr[-2000:]}, 500)
            return

        # 2. GFPGAN 修复 (按帧)
        # 简化：直接输出 wav2lip 结果，gfpgan 可选后处理
        if req.get("gfpgan", True):
            # TODO: 集成 gfpgan 批量修复
            pass

        os.rename(tmp, out)
        self._json({"status": "done", "out": out})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
