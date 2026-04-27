FROM dustynv/l4t-pytorch:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST=8.7
ENV CUDA_HOME=/usr/local/cuda
ENV USE_NINJA=1
ENV MAX_JOBS=4
ENV VCPKG_ROOT=/opt/vcpkg

# Override NVIDIA's baked-in pip env vars
ENV PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126
ENV PIP_EXTRA_INDEX_URL=https://pypi.ngc.nvidia.com
ENV PIP_TRUSTED_HOST=pypi.jetson-ai-lab.io

RUN apt-get update && apt-get install -y \
    build-essential \
    git ninja-build pkg-config \
    curl jq \
    libopenblas-dev libomp-dev \
    python3-dev python3-pip

# -----------------------------
# Fix Jetson pip index (JP6)
# -----------------------------
RUN mkdir -p /etc/pip && \
    printf "[global]\nindex-url = https://pypi.jetson-ai-lab.io/jp6/cu126\nextra-index-url = https://pypi.ngc.nvidia.com\ntrusted-host = pypi.jetson-ai-lab.io\n" \
    > /etc/pip.conf

RUN pip install --upgrade pip setuptools wheel packaging ninja
RUN pip install "cmake==4.1.2"
RUN cmake --version

# -----------------------------
# Install downgraded numpy
# -----------------------------
RUN python3 -m pip install --upgrade --force-reinstall "numpy<2"

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

RUN pip install ninja wheel
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

# Keep the spatial-audio toolchain late so edits there do not invalidate
# the heavier xFormers/Triton layers above.
# -----------------------------
# Install spatial-audio toolchain
# -----------------------------
RUN apt-get update && apt-get install -y \
    libzmq3-dev \
    portaudio19-dev libportaudio2 \
    libasound2-plugins alsa-utils pulseaudio-utils \
    libglfw3-dev libxinerama-dev libxcursor-dev \
    xorg-dev libglu1-mesa-dev
WORKDIR /opt
RUN git clone --depth 1 https://github.com/microsoft/vcpkg.git "${VCPKG_ROOT}" && \
    "${VCPKG_ROOT}/bootstrap-vcpkg.sh" && \
    "${VCPKG_ROOT}/vcpkg" version

WORKDIR /workspace/jedi-vision-nano-code/src/spatial-audio
# Validate the spatial-audio toolchain without paying for a full build here.
RUN cmake --preset vcpkg
WORKDIR /workspace/jedi-vision-nano-code
RUN pip install .
RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/torchvision/d' requirements.txt && \
    sed -i '/transformers/d' requirements.txt && \
    sed -i '/numpy/d' requirements.txt
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

# -----------------------------
# Install NVIDIA VPI
# -----------------------------

# Install packages required by add-apt-repository
RUN apt-get update && \
    apt-get install -y gnupg software-properties-common
 
# Add Jetson public APT repository
RUN apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc && \
    add-apt-repository 'deb https://repo.download.nvidia.com/jetson/common r36.4 main'
 
# Install VPI depedencies
RUN apt-get update && apt-get install -y libnpp-12-6 libcufft-12-6 cuda-cudart-12-6 libegl1-mesa
 
# Add CUDA packages to library path
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12-6/targets/aarch64-linux/lib/
 
# This is a temporary workaround required to install pva-allow-2 in docker which will not be necessary next release
# RUN apt-get install -y pva-allow-2 || true && rm /var/lib/dpkg/info/pva-allow-2.post* && dpkg --configure pva-allow-2
 
# Install VPI
RUN apt-get install -y libnvvpi3 vpi3-dev vpi3-samples
RUN apt install python3.10-vpi3