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

# -----------------------------
# Fix Jetson pip index (JP6)
# -----------------------------
RUN mkdir -p /etc/pip && \
    printf "[global]\nindex-url = https://pypi.jetson-ai-lab.io/jp6/cu126\nextra-index-url = https://pypi.ngc.nvidia.com\ntrusted-host = pypi.jetson-ai-lab.io\n" \
    > /etc/pip.conf

RUN pip install --upgrade pip setuptools wheel packaging

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

# -----------------------------
# Install HuggingFace Transformers (pure Python)
# -----------------------------
RUN pip install --no-build-isolation \
    "transformers<4.42" \
    tokenizers \
    sentencepiece \
    safetensors \
    accelerate

WORKDIR /workspace

