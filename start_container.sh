sudo docker run -it --rm \
    --runtime nvidia \
    --network host \
    -v $PWD:/workspace/jv \
    jp61-orin-xformers