FROM dustynv/l4t-pytorch:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST=8.7
ENV CUDA_HOME=/usr/local/cuda
ENV USE_NINJA=1
ENV MAX_JOBS=4

# Override NVIDIA's baked-in pip env vars
ENV PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126
ENV PIP_EXTRA_INDEX_URL=https://pypi.ngc.nvidia.com
ENV PIP_TRUSTED_HOST=pypi.jetson-ai-lab.io

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

RUN pip install --upgrade pip setuptools wheel packaging ninja

# -----------------------------
# Install downgraded numpy
# -----------------------------
RUN python3 -m pip install --upgrade --force-reinstall --no-cache-dir "numpy<2"

# -----------------------------
# Build xFormers v0.0.28 (compatible with PyTorch 2.4)
# -----------------------------
WORKDIR /opt
RUN git clone --recursive https://github.com/facebookresearch/xformers.git
WORKDIR /opt/xformers
RUN git checkout v0.0.28
RUN git submodule update --init --recursive
RUN sed -i '/torch/d' requirements.txt

RUN pip install ninja && \
    pip install -v --no-build-isolation .

# -----------------------------
# Build Triton 3.0.0
# ---------v--------------------
WORKDIR /opt
RUN git clone --recursive https://github.com/triton-lang/triton.git
WORKDIR /opt/triton
RUN git checkout v3.0.0
RUN git submodule update --init --recursive

RUN pip install cmake
RUN python3 install.py

# -----------------------------
# Install HuggingFace Transformers and others
# -----------------------------
RUN pip install --no-build-isolation \
    "transformers>=4.55,<5.0" \
    tokenizers \
    sentencepiece \
    safetensors \
    accelerate \
    triton

WORKDIR /workspace

# -----------------------------
# Install jv + requirements
# -----------------------------
RUN git clone --recursive https://github.com/Jedi-Vision/jedi-vision-nano-code.git
WORKDIR /workspace/jedi-vision-nano-code
RUN pip install .
RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/torchvision/d' requirements.txt && \
    sed -i '/transformers/d' requirements.txt && \
    sed -i '/numpy/d' requirements.txt
RUN pip install -r requirements.txt