# Mia lip-sync 自建镜像：Wav2Lip + GFPGAN，完全自控不依赖第三方
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV FLICKIES_ENABLE_NONCOMMERCIAL=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git wget ffmpeg libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Wav2Lip
RUN git clone https://github.com/Rudrabha/Wav2Lip.git wav2lip
WORKDIR /opt/wav2lip
RUN pip3 install --no-cache-dir torch torchvision torchaudio numpy opencv-python librosa numba scipy
# Patch for modern librosa API (mel() args are keyword-only now)
RUN sed -i 's/librosa.filters.mel(hp.sample_rate, hp.n_fft,/librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,/' audio.py && grep -n "librosa.filters.mel" audio.py

# Wav2Lip checkpoint (~1GB)
RUN mkdir -p checkpoints && \
    wget -q "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzRcXfFyH9fJzYQJzYQJzYQJzYQJzYQJzYQ" -O checkpoints/wav2lip_gan.pth || \
    echo "MANUAL: download wav2lip_gan.pth to /opt/wav2lip/checkpoints/"

# GFPGAN
WORKDIR /opt
RUN git clone https://github.com/TencentARC/GFPGAN.git gfpgan
WORKDIR /opt/gfpgan
RUN pip3 install --no-cache-dir -r requirements.txt 2>/dev/null || pip3 install --no-cache-dir basicsr facexlib gfpgan

WORKDIR /opt/wav2lip
EXPOSE 8000

# 健康检查 + 任务接口
COPY server.py /opt/wav2lip/server.py
CMD ["python3", "server.py"]
