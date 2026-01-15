FROM dustynv/l4t-pytorch:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST=8.7
ENV CUDA_HOME=/usr/local/cuda
ENV USE_NINJA=1
ENV MAX_JOBS=4

RUN apt-get update && apt-get install -y \
    git cmake ninja-build \
    libopenblas-dev libomp-dev \
    python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel packaging
RUN pip install poetry

# -----------------------------
# Build xFormers v0.0.33 (compatible with PyTorch 2.4)
# -----------------------------
WORKDIR /opt
RUN git clone --recursive https://github.com/facebookresearch/xformers.git
WORKDIR /opt/xformers
RUN git checkout v0.0.33
RUN sed -i '/torch/d' requirements.txt

RUN pip install ninja && \
    pip install -v --no-build-isolation .

WORKDIR /workspace

