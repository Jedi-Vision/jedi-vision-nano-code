# Jedi Vision Project Code for Jetson Nano

## Contains
* Scene representation conversion
  * Monocular depth model
* Environment representation
  * YOLO model inference
* Object representation
  * Grounded object tracking model
* Audio representation
  * Python bindings to Steam Audio

## Installation

This installation assumes that you have a Python version between `3.11` and `3.14` installed on your computer.

1. First off, make you have at least [Python 3.10](https://www.python.org/downloads/) installed.

2. Then clone the repository.

    via HTTP
    ```bash
    git clone --recursive https://github.com/Jedi-Vision/jedi-vision-nano-code.git
    ```

    via SSH
    ```bash
    git clone --recursive git@github.com:Jedi-Vision/jedi-vision-nano-code.git
    ```

    **If running via a docker container please look at the [next section](#nvidia-jetson-orin-nano--docker-container-usage-instructions) for further installation instructions.**

3. Create a virtual environment and activate

    ```python
    python -m venv .venv && source .venv/bin/activate
    ```

4. Navigate to the project folder and install the dependencies

    ```bash
    pip install -r requirements.txt
    ```

5. Then install the `jv` library.

    ```bash
    pip install .
    ```

6. (Optional) If you are running into a *"No module named jv"* error, try installing our `jv` library in editable mode.

    ```bash
    pip install -e .
    ```

## NVIDIA Jetson Orin Nano / Docker Container Usage Instructions

Since building everything from scratch takes more memory than is available on the Jetson, even through swap memory. We build upon an existing l4t-pytorch image, and manually build xFormers and some other dependencies we needed.

### Installation
Run the command

```bash
sudo docker build -t jp61-orin-xformers:latest .
```

### Running

Run using the following:
```bash
sudo docker run -it --rm \
    --runtime nvidia \
    --network host \
    -v $PWD:/workspace/jv \
    jp61-orin-xformers
```

It is also possible to run this command from `start_container.sh`.

### Alternative: Jetson Containers

To use this software package on a Jetson Orin Nano, we utilized the [jetson-containers](https://github.com/dusty-nv/jetson-containers) library from Dustin Franklin.

We have included a fork of the repository with a Jedi-Vision specific PyTorch container which includes all the needed dependencies and Poetry setup.

***This will probably fail due to lack of memory in building PyTorch.***

#### Installation

To install on a Jetson Orin Nano with Jetpack 6.2 (nvidia-l4t-core 36.4.7) with Docker container run the following:

```bash
# navigate to repo
cd jedi-vision-nano-code

# install the container tools
bash src/jetson-containers/install.sh
```

##### Docker Default Runtime

If you're going to be building containers, you need to set Docker's `default-runtime` to `nvidia`, so that the NVCC compiler and GPU are available during `docker build` operations.  Add `"default-runtime": "nvidia"` to your `/etc/docker/daemon.json` configuration file before attempting to build the containers:

``` json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },

    "default-runtime": "nvidia"
}
```

Then restart the Docker service, or reboot your system before proceeding:

```bash
$ sudo systemctl restart docker
```

You can then confirm the changes by looking under `docker info`

```bash
$ sudo docker info | grep 'Default Runtime'
Default Runtime: nvidia
```

##### Build Container

```bash
# Build the PyTorch container with specific CUDA version (12.6)
CUDA_VERSION=12.6 jetson-containers build --name jv-pytorch-container pytorch
```

Then install all the dependencies:

```bash
# Start docker container
bash start_container.sh

# Install dependencies
cd workspace/jv
POETRY_VIRTUALENVS_CREATE=false poetry install
```

#### Usage

Use the provided `start_container.sh`

```bash
jetson-containers run -v ./jedi-vision-nano-code:/workspace/jv \
    jv-pytorch-container:r36.4.tegra-aarch64-cu126-22.04-python
```

Or run the container directly with `jetson-containers` and link the existing repository to a volume inside the container.
```bash
cd ../  # navigate to outside of repo folder
jetson-containers run -v ./jedi-vision-nano-code:/workspace/jv \
    jv-pytorch-container
```

The command `-v ./jedi-vision-nano-code:/workspace/jv` links the repository folder `jedi-vision-nano-code` to a folder `workspace/jv`
