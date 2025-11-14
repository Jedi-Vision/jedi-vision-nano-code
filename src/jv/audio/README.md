# Audio

Since the STEAM audio library runs in C, we use sockets to communicate object information between the models and the spatial audio generator.

## ZeroMQ Installation

We utilize the ZeroMQ concurrent networking library. It has bindings in both C and Python, which make it well suited for our usecase. For more information about the library, see the [user guide](https://zguide.zeromq.org/).

### C Library

Please see the [czmq](https://github.com/zeromq/czmq) GitHub page for installation steps.

### Python Library

`pyzmq` should already be installed via the Poetry project, but if you are running this separately it can be easily installed with the command below.

```bash
pip install pyzmq
```

## Usage

Please see `example/` for example usage for a server in C, and a client in Python. The Makefile is set up for Apple Silicon, so you may need to tinker it for compilation on your unique device.