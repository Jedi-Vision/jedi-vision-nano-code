#!/usr/bin/env bash
set -Eeuo pipefail

mkdir -p /workspace/jv/logs

if [[ -z "${DISPLAY:-}" ]]; then
    unset DISPLAY
fi

if [[ "${1:-}" == "visualize" ]]; then
    echo "Starting app in visualize mode..."

    DISPLAY=:0 ./src/spatial-audio/build/jsa-visual-monitor \
        --ipc ipc:///tmp/jv/audio/0.sock \
        --forward-ipc ipc:///tmp/jv/audio/1.sock \
        >> /workspace/jv/logs/audio.log 2>&1 &

    DISPLAY=:0 ./src/spatial-audio/build/jsa-live-3d \
        --ipc ipc:///tmp/jv/audio/1.sock \
        --audio-buffer-ms 120 \
        --max-interp-window-ms 25 \
        --stale-frame-drop-ms 200 \
        --audio-azimuth-scale 2.75 \
        --audio-azimuth-max-deg 90 \
        --tone-min-gap-ms 200 \
        --source-mode tones \
        --hrtf default \
        >> /workspace/jv/logs/audio.log 2>&1 &

    PID1=$!

    DISPLAY=:0 python3 main.py \
    -c config/nano.yaml \
    -a output_to:socket \
    >> /workspace/jv/logs/detect.log 2>&1 &

    PID2=$!

else
    echo "Starting app..."

    ./src/spatial-audio/build/jsa-live-3d \
        --ipc ipc:///tmp/jv/audio/0.sock \
        --audio-buffer-ms 120 \
        --stream-timeout-ms 60 \
        --audio-azimuth-scale 3 \
        --audio-azimuth-max-deg 90 \
        --tone-min-gap-ms 200 \
        --source-mode tones \
        >> /workspace/jv/logs/audio.log 2>&1 &

    PID1=$!

    python3 main.py \
    -c config/nano.yaml \
    -a output_to:socket \
    >> /workspace/jv/logs/detect.log 2>&1 &

    PID2=$!
fi

echo "Processes started:"
echo "audio PID:   $PID1"
echo "main.py PID: $PID2"

# Forward signals properly
trap 'kill -TERM $PID1 $PID2 2>/dev/null' SIGINT SIGTERM

# Wait for either process to exit
wait -n

STATUS=$?

echo "A process exited with status $STATUS"

# Kill remaining process
kill -TERM $PID1 $PID2 2>/dev/null || true

wait || true

exit $STATUS