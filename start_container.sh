sudo docker run -it --rm \
    --runtime nvidia \
    --network host \
    -v $PWD:/workspace/jv \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    -e DISPLAY=$DISPLAY \
    jp61-orin-xformers