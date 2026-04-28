#!/usr/bin/env bash

set -euo pipefail

DOCKER_ARGS=(
    sudo docker run -it --rm
    --runtime nvidia
    --network host
    --privileged
    -v "$PWD:/workspace/jv"
    -v /tmp/:/tmp/
    -v /tmp/argus_socket:/tmp/argus_socket
    -v "$HOME/.Xauthority:/root/.Xauthority:rw"
    -v /dev:/dev
    -e "DISPLAY=${DISPLAY:-}"
    -e PULSE_SERVER=unix:/tmp/pulse/native
    -v /run/user/1000/pulse/native:/tmp/pulse/native
    -v "$HOME/.config/pulse/cookie:/root/.config/pulse/cookie:ro"
    -v /etc/machine-id:/etc/machine-id:ro
    --device nvidia.com/gpu=all 
    --device nvidia.com/pva=all 
    -it
)

echo "Routing container audio through host PulseAudio (unix:/tmp/pulse/native)."

DOCKER_ARGS+=(jp61-orin-xformers)

"${DOCKER_ARGS[@]}"

# Have privileged is lowkey sketchy but should be fine
