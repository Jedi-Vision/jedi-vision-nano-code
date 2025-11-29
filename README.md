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

1. First off, make you have [Poetry](https://python-poetry.org/docs/) installed.

2. Then clone the repository.

    via HTTP
    ```bash
    git clone https://github.com/Jedi-Vision/jedi-vision-nano-code.git
    ```

    via SSH
    ```bash
    git clone git@github.com:Jedi-Vision/jedi-vision-nano-code.git
    ```
    
    Also make sure to initialize the submodules,
    ```bash
    cd jedi-vision-nano-code
    git submodule update --init --recursive
    ```

    **If running via a docker container please look at the [next section](#nvidia-jetson-orin-nano--docker-container-usage-instructions) for further installation instructions.**

3. (Optional) If you'd like poetry to automatically create a virtual environment *in the project* run the following.

    ```bash
    poetry config virtualenvs.in-project true
    ```

4. Navigate to the project folder and install the dependencies

    Then run:
    ```bash
    poetry install
    ```

5. Activate the virtual environment

    Run the following to get the activation command for the Poetry virtual environment.
    ```bash
    poetry env activate
    ```

    If you configured Poetry to install the virtual environment in the project you can just run:
    ```bash
    source .venv/bin/activate
    ```

6. (Optional) If you are running into a *"No module named jv"* error, try installing our `jv` library in editable mode.

    ```bash
    pip install -e .
    ```

## NVIDIA Jetson Orin Nano / Docker Container Usage Instructions

To use this software package on a Jetson Orin Nano, we utilized the [jetson-containers](https://github.com/dusty-nv/jetson-containers) library from Dustin Franklin.

We have included a fork of the repository with a Jedi-Vision specific PyTorch container which includes all the needed dependencies and Poetry setup.

### Installation

To install on a Jetson Orin Nano with Jetpack 6.2 (nvidia-l4t-core 36.4.7) with Docker container run the following:

```bash
# navigate to repo
cd jedi-vision-nano-code

# install the container tools
bash src/jetson-containers/install.sh
```

#### Docker Default Runtime

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

#### Build Container

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

### Usage

Use the provided `start_container.sh`

```bash
bash start_container.sh
```

Or run the container directly with `jetson-containers` and link the existing repository to a volume inside the container.
```bash
cd ../  # navigate to outside of repo folder
jetson-containers run -v ./jedi-vision-nano-code:/workspace/jv jv-pytorch-container
```

The command `-v ./jedi-vision-nano-code:/workspace/jv` links the repository folder `jedi-vision-nano-code` to a folder `workspace/jv`
