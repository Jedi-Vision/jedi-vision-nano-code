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
    sed -i '/numpy/d' requirements.txt
RUN pip install -r requirements.txt

RUN curl $'https://dl.boxcloud.com/d/1/b1\041X6GsnOefyCso5CmP_Xi1hxXPwPY8oSSvZqnAnqT8mDNA5kyVtOTvDUDp4xrpNue298CiSVMxXfBEKdyAjhLaoM964sopZYju0P2WRLpP7eaM2I-tX5mbun_l7cydyOfERyvu-QIhWwIRxv_6mV5M4YWnVhy-cE2ulUyz6B_VFOtA1MHb2BMFA_pyJGi59-g1Jsj-vuvMjGmnAeP2FowKAMrk_luex_6pYymruxnzP1_-HwNJVQbjMtbc3bwSUM1wcH8e81CL4W-46ZKn8Y0EMFavmdTJLdKisCUP3FZ3LwVbfTsGWmjGS8RBrjAIh7ONB-9cZ9dgumO7GOcyRWWPdSqc19nYPJHiK1HFfu0jYkgESCWex8qTC_CzOc_-sCzyRZHdn-utxbOfn1TBLO7HkqAzpuJb4J7QUnWCQlRPU-KHTf2wy1vFHi0E7N76YxP7jjCLiFQ9h_SmzmQps-OHe4HYfY2I81QWaVel4J_PZu-g-XqsQI6avr3SClU7EsAtXtuuwH1qztMNVS57aIhRtRNfhb2g0rs6n4w_GbsovTprqMbdutToqnVHwd6AWa2V7H1Jzq22kdZH-RlUgz2C8eClr7N7j-HAdB002UUTHNPjEljpv5n6XFiDXAzC79XvjCmjxWVdlvwAPVU26jbuZDJi37N9FxWFIqTkCzifnppLUVc5DMbzT8qFUSy0WnWSM1zG1Kg3fWvNxwZ9zqOVKtxokM0LVKSp7MlXk8Q9rFuKUP6vhztkMk4S7qOChw8bRFzjag-BGP79NoEqjELpqONCSVy_-T0Hym7oxSHaS4KA_98D7o3n5RZVvXmlA9lqPjqUVjH85m1sBxCtVzNI_EyDcFKZR4LYblUfzT9ft2OqaPmr0g9LDr0ouDl9UbT8bh5eZlq1odZtLl7taNsZ1VPM5lqTZmf6wHGsM4qIBptzKwb1MrWldXTBP6VlYSwQCUaI60J6wBoSJD0Nsj1WLg6Qk_hmo7QpsKYoqi8L6ZIjZ1_c_7OByhvgNBEHJIqiVx5klkHK6GlGh1mRd_JnjRaq5p4ApfP0rSCGleHbcS4w5CohigKyGKX-3BTtuosjVunq5o-zWe9mbRVu411IkkQUPJv3lOMrvPfsXxa538wvy5D4ykzA6q0x7eIbgLuBfpFB9zIcDc3DB2XW5Ieu8FAtyIK5Wnf8zJWCKzvU0A7Tu7kkC5b6mKUTj57cbpD_3d_4oH7lZ2OAu-lAZ7hBd3axSgA2fMiMwBXjtdfY0HdpWVTkCeNEx4JP_AL2JiG1BlipyjVStUQS28SzjBaJDJW184IRsAws78hbS6aXEAxukeSED520SI5SZGVHb4KKkos7_qVglHYOwQIl818UNOlKPCxmNb_g5sMwII4RomBJ-yR7pMNqnNOl31LSVh-ghvQKkg1q8TWYfHka2QqtbdBeCW1dTVvoDKupg54YeFst252MYJL8Ez0QTvXU0I1xTIzT4ygds_kXasUuZEgarUqjo-_pGGmcINHrjlcguXohgLMM5A../download' \
  --compressed \
  --output example/videos/sidewalk_pov.mp4 \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:146.0) Gecko/20100101 Firefox/146.0' \
  -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' \
  -H 'Accept-Language: en-US,en;q=0.5' \
  -H 'Accept-Encoding: gzip, deflate, br, zstd' \
  -H 'Referer: https://oregonstate.app.box.com/' \
  -H 'Alt-Used: dl.boxcloud.com' \
  -H 'Connection: keep-alive' \
  -H 'Cookie: b=069282ef78579a8f9c36bb01686877a8c19241655af6cb6d19c30dba82ffc2da' \
  -H 'Upgrade-Insecure-Requests: 1' \
  -H 'Sec-Fetch-Dest: iframe' \
  -H 'Sec-Fetch-Mode: navigate' \
  -H 'Sec-Fetch-Site: cross-site' \
  -H 'Sec-Fetch-User: ?1' \
  -H 'Priority: u=4'

RUN bash get_weights.sh