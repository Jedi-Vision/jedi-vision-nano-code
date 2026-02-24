sudo docker run -it --rm \
    --runtime nvidia \
    --network host \
    --privileged \
    -v $PWD:/workspace/jv \
    -v /tmp/:/tmp/ \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    -v /dev:/dev \
    -e DISPLAY=$DISPLAY \
    -v /etc/machine-id:/etc/machine-id:ro \
    jp61-orin-xformers

# Have privileged is lowkey sketchy but should be fine