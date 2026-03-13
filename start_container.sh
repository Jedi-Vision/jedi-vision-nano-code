HOST_UID=$(id -u)
HOST_PULSE_DIR="/run/user/${HOST_UID}/pulse"
PULSE_AUDIO_ARGS=""
PULSE_COOKIE_ARGS=""

if [ -S "${HOST_PULSE_DIR}/native" ]; then
    PULSE_AUDIO_ARGS="-e XDG_RUNTIME_DIR=/run/user/${HOST_UID} -e PULSE_SERVER=unix:${HOST_PULSE_DIR}/native -v ${HOST_PULSE_DIR}:${HOST_PULSE_DIR}"
fi

if [ -f "$HOME/.config/pulse/cookie" ]; then
    PULSE_COOKIE_ARGS="-v $HOME/.config/pulse/cookie:/root/.config/pulse/cookie:ro"
fi

sudo docker run -it --rm \
    --runtime nvidia \
    --network host \
    --privileged \
    -v $PWD:/workspace/jv \
    -v /tmp/:/tmp/ \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    -v /dev:/dev \
    --device /dev/snd \
    --group-add audio \
    $PULSE_AUDIO_ARGS \
    $PULSE_COOKIE_ARGS \
    -e DISPLAY=$DISPLAY \
    -v /etc/machine-id:/etc/machine-id:ro \
    jp61-orin-xformers

# Have privileged is lowkey sketchy but should be fine