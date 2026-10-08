# Mia lip-sync 自建镜像：Wav2Lip + GFPGAN，完全自控不依赖第三方
# CUDA 12.1：匹配 RunPod 宿主机驱动（R550+/CUDA 12.4），根治 GPU 不可用问题
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV FLICKIES_ENABLE_NONCOMMERCIAL=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git wget curl ffmpeg libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Wav2Lip
RUN git clone https://github.com/Rudrabha/Wav2Lip.git wav2lip
WORKDIR /opt/wav2lip
# PyTorch with CUDA 12.1 support (NOT cpu-only!) — torch 2.5.x 是最后一个带 cu121 wheel 的版本
RUN pip3 install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir numpy opencv-python librosa numba scipy
# 构建时打印 torch CUDA 信息（构建机无 GPU 时 cuda_available=false，属正常）
RUN python3 -c "import torch; print('cuda_available=', torch.cuda.is_available(), 'version=', torch.version.cuda)" || true
# Patch for modern librosa API (mel() args are keyword-only now)
RUN sed -i 's/librosa.filters.mel(hp.sample_rate, hp.n_fft,/librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,/' audio.py && grep -n "librosa.filters.mel" audio.py
# Patch for torch>=2.5: torch.load defaults to weights_only=True, breaks Wav2Lip checkpoint
RUN sed -i 's/torch\.load(checkpoint_path)/torch.load(checkpoint_path, weights_only=False)/' inference.py && grep -n "weights_only" inference.py

# Wav2Lip checkpoint (~1GB) — 必须下载成功且非空，否则构建失败
RUN mkdir -p checkpoints && \
    (curl -fSL "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth" -o checkpoints/wav2lip_gan.pth || \
     curl -fSL "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth" -o checkpoints/wav2lip_gan.pth || \
     curl -fSL "https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth" -o checkpoints/wav2lip_gan.pth) && \
    [ $(stat -c%s checkpoints/wav2lip_gan.pth) -gt 100000000 ] && \
    echo "checkpoint OK: $(stat -c%s checkpoints/wav2lip_gan.pth) bytes"

# GFPGAN
WORKDIR /opt
RUN git clone https://github.com/TencentARC/GFPGAN.git gfpgan
WORKDIR /opt/gfpgan
RUN pip3 install --no-cache-dir -r requirements.txt 2>/dev/null || pip3 install --no-cache-dir basicsr facexlib gfpgan

# GFPGAN v1.4 权重（~350MB）：构建时下载 + 非空校验，server.py 启动时不再依赖现场下载
RUN mkdir -p experiments/pretrained_models && \
    curl -fSL "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" \
      -o experiments/pretrained_models/GFPGANv1.4.pth && \
    [ $(stat -c%s experiments/pretrained_models/GFPGANv1.4.pth) -gt 100000000 ] && \
    echo "GFPGAN weights OK: $(stat -c%s experiments/pretrained_models/GFPGANv1.4.pth) bytes"

WORKDIR /opt/wav2lip
EXPOSE 8000

# 健康检查 + 任务接口
COPY server.py /opt/wav2lip/server.py
CMD ["python3", "server.py"]
