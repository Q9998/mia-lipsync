# Mia 口型同步自建方案

## 目标
完全自控的 Wav2Lip + GFPGAN 流水线，不依赖任何第三方镜像。

## 架构
```
[RunPod GPU] ← 我们的 Dockerfile（CUDA 11.8 + Wav2Lip + GFPGAN）
     ↓ HTTP :8000
[lipsync server.py] → /lipsync {face, audio} → 输出口型视频
```

## 构建
```bash
cd ~/workspace/goals/ai/lipsync
docker build -t mia-lipsync:v1 .
# 推送到 Docker Hub（需 Q 的账号）或直接用 tar 包传到 RunPod
```

## 使用
```bash
# RunPod 用自定义镜像 mia-lipsync:v1 创建 Pod
# POST http://<pod>:8000/lipsync
# {"face": "/path/to/base-16x9.mp4", "audio": "/path/to/seg.wav", "out": "/path/to/out.mp4"}
```

## 成本
- 自建镜像：一次性构建，免费
- RunPod GPU：A4000 $0.17/hr，按需开关
- 18 条约 30 分钟 ≈ $0.09/次

## 状态
- [x] Dockerfile
- [x] server.py
- [ ] 构建测试
- [ ] 18 条 16:9 生产
