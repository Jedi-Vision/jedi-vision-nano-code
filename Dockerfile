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

RUN pip install ninja cmake wheel
RUN pip install -e python --no-build-isolation

# -----------------------------
# Install HuggingFace Transformers and others
# -----------------------------
RUN pip install --no-build-isolation \
    "transformers>=4.55,<5.0" \
    tokenizers \
    sentencepiece \
    safetensors \
    accelerate

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
    sed -i '/numpy/d' requirements.txt && \
    sed -i '/opencv-python/d' requirements.txt
RUN pip install -r requirements.txt
RUN bash get_weights.sh

# -----------------------------
# Install GStreamer
# -----------------------------
RUN apt-get update && \
    apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav && \
    apt-get install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev

# -----------------------------
# Ensure OpenCV built with GStreamer
# -----------------------------
RUN pip3 uninstall opencv-python -y
RUN apt-get update && apt-get install -y \
    python3-opencv \
    libopencv-dev

# -----------------------------
# Download example videos
# -----------------------------

# Fetch video
RUN curl -L https://oregonstate.box.com/shared/static/3p1ohmn4tm6ytwccnp3tdvtbiybk4c53.mp4 -o examples/videos/sidewalk_pov.mp4
RUN curl -L https://oregonstate.box.com/shared/static/neyzpi2f42knbavdvdcvcpdir1zqm3o7.mov -o examples/videos/two_people.mov