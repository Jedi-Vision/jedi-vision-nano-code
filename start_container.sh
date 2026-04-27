#!/usr/bin/env bash

set -euo pipefail

HOST_UID="${HOST_UID:-$(id -u)}"
HOST_PULSE_DIR="${HOST_PULSE_DIR:-/run/user/${HOST_UID}/pulse}"
HOST_PULSE_COOKIE="${HOST_PULSE_COOKIE:-${PULSE_COOKIE:-$HOME/.config/pulse/cookie}}"
HOST_PULSE_SERVER="${PULSE_SERVER:-}"
AUDIO_ROUTE="none"

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
    -v /etc/machine-id:/etc/machine-id:ro
    --device nvidia.com/gpu=all 
    --device nvidia.com/pva=all 
    -it
)

if [[ -z "${HOST_PULSE_SERVER}" && -S "${HOST_PULSE_DIR}/native" ]]; then
    HOST_PULSE_SERVER="unix:${HOST_PULSE_DIR}/native"
fi

if [[ -n "${HOST_PULSE_SERVER}" ]]; then
    DOCKER_ARGS+=(-e "PULSE_SERVER=${HOST_PULSE_SERVER}")
    AUDIO_ROUTE="pulse"
fi

if [[ "${HOST_PULSE_SERVER}" == unix:* ]]; then
    pulse_socket_path="${HOST_PULSE_SERVER#unix:}"
    pulse_mount_dir="$(dirname "${pulse_socket_path}")"
    xdg_runtime_dir="$(dirname "${pulse_mount_dir}")"

    DOCKER_ARGS+=(
        -e "XDG_RUNTIME_DIR=${xdg_runtime_dir}"
        -v "${pulse_mount_dir}:${pulse_mount_dir}"
    )
fi

if [[ -f "${HOST_PULSE_COOKIE}" ]]; then
    DOCKER_ARGS+=(-v "${HOST_PULSE_COOKIE}:/root/.config/pulse/cookie:ro")
fi

if [[ -e /dev/snd ]]; then
    DOCKER_ARGS+=(--device /dev/snd --group-add audio)

    if [[ "${AUDIO_ROUTE}" == "none" ]]; then
        AUDIO_ROUTE="alsa"
    fi
fi

if [[ "${AUDIO_ROUTE}" == "pulse" ]]; then
    echo "Routing container audio through host PulseAudio (${HOST_PULSE_SERVER})."
elif [[ "${AUDIO_ROUTE}" == "alsa" ]]; then
    echo "Routing container audio through host ALSA devices (/dev/snd)."
else
    echo "Warning: no host audio route detected. Set PULSE_SERVER or expose /dev/snd for audio output." >&2
fi

DOCKER_ARGS+=(jp61-orin-xformers)

"${DOCKER_ARGS[@]}"

# Have privileged is lowkey sketchy but should be fine
