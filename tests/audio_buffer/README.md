# Audio Buffer Testing

The following folder contains important testing code for validating functionality of the Audio Buffer class. The main file to be run is `run_audio_buffer.py`, which runs the `AudioBuffer` client class in a loop sending fake test data across a ZeroMQ socket.

## Server Options
Depending on the serialization option, you need to test with different servers. All servers include a Makefile, but this is not guaranteed to work on anything but Apple Silicon devices.

### Protobuf
This is the Protobuf server that deserializes the Protobuf data format. This server requires the `pbtools` submodule.

### Struct
This decodes the byte information into C struct data.