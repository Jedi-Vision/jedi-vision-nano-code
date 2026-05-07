#!/usr/bin/env bash

set -e

echo "Starting app..."

python3 main.py -c config/nano.yaml -a output_to:socket &>> /workspace/jv/logs/detect.log &

./src/spatial-audio/build/jsa-live-3d \
    --ipc ipc:///tmp/jv/audio/0.sock \
    --audio-buffer-ms 120 \
    --stream-timeout-ms 60 \
    --audio-azimuth-scale 3 \
    --audio-azimuth-max-deg 90 \
    --tone-min-gap-ms 200 \
    --source-mode tones &>> /workspace/jv/logs/audio.log &

exec "$@"