# Jedi Vision Project Code

## Contains
* Scene representation conversion
  * Monocular depth model
* Environment representation
  * YOLO model inference
* Object representation
  * Grounded object tracking model
* Audio representation
  * Python bindings to Steam Audio
* Serialization
  * Custom serialization of object information and bindings for IPC client/server connection

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

## Running

We make use of modifiable `.yaml` files for easy configuration of parameters (see `/config/default.yaml` and `/src/jv/driver/driver.py` for more information on possible parameters).

To run, just input the following:
```bash
python main.py

    usage: main.py [-h] [-c CONFIG] [-a [ARGS ...]]

    options:
    -h, --help            show this help message and exit
    -c CONFIG, --config CONFIG
                            Configuration file containing arguments for Driver.
    -a [ARGS ...], --args [ARGS ...]
                            Arguments to replace any config field in 'key:value' style.
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
    --privileged \  # lowkey sketchy but should be fine
    -v $PWD:/workspace/jv \
    -v /tmp/:/tmp/ \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    -v /dev:/dev \
    -e DISPLAY=$DISPLAY \
    -e PULSE_SERVER=unix:/tmp/pulse/native \
    -v /run/user/1000/pulse/native:/tmp/pulse/native \
    -v "$HOME/.config/pulse/cookie:/root/.config/pulse/cookie:ro" \
    -v /etc/machine-id:/etc/machine-id:ro \
    jp61-orin-xformers
```

(It is also possible to run this command from `start_container.sh`.)

The image sets `PULSE_SERVER=unix:/tmp/pulse/native` and configures ALSA's default
device to use PulseAudio. With the host socket and cookie mounted as shown above,
PulseAudio-aware apps connect to the host server directly, and ALSA-default apps route
through PulseAudio as well.

Then start the pipeline:

```bash
python3 main.py
```

You'll likely need to configure the Driver arguments in `main.py`.


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

## Jetson Orin Nano Power Mode

On the Jetson, there are different power modes available that result in better computational power. However, this is at the cost of higher temperatures and more powerdraw.

To check the current power mode run
```bash
sudo nvpmodel -q
```

To change the power mode you can run the following
```bash
sudo nvpmodel -m $MODE
```
With `$MODE` being a number 0-2.

The modes are as follows
```
0: NV Power Mode: 15W
1: NV Power Mode: 25W
2: NV Power Mode: MAXN_SUPER
```

It is recommended for the best inference speeds to run with the `MAXN_SUPER` power mode.

## Connecting Camera via CSI

In order to configure the CSI connector pin's to support a camera connection, launch Jetson-IO:

```bash
sudo /opt/nvidia/jetson-io/jetson-io.py
```

Then, select 
1. `Configure Jetson Nano CSI Connector`
2. `Configure for compatible hardware`

From there you can select from the available options, or [custom configurations](https://docs.nvidia.com/jetson/archives/r35.3.1/DeveloperGuide/text/SD/CameraDevelopment/SensorSoftwareDriverProgramming.html#sd-cameradevelopment-sensorsoftwaredriverprogramming-kernelconfiguration) if you are adding a non-natively supported device with custom drivers. Then,

1. `Save pin changes`
2. `Save and reboot to reconfigure pins`

After rebooting, run the following to ensure that you can see the available cameras:

```bash
ls /dev/video*
```

NOTE: Using the Argus API for GStreamer to stream directly from the camera CSI requires that X11 is disabled, i.e. if it connects to an X11 socket for display it will crash. Therein, although you can still access over SSH with X11 enabled, ensure that you add `DEVICE=:0` before running anything that uses the GStreamer backend. This is only a concern when *accessing a Jetson Nano over SSH*, if you are locally accessing you shouldn't run into problems.

### Calibrating Camera

Camera's typically have distortion, and due to that they need to be calibrated to determine the extrinsic and intrinsic parameters, which can then be used to undistort an image. Read [this](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) for more information.
