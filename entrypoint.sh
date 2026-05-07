#!/usr/bin/env bash
set -Eeuo pipefail

mkdir -p /workspace/jv/logs

if [[ -z "${DISPLAY:-}" ]]; then
    unset DISPLAY
fi

echo "Starting app..."

python3 main.py \
    -c config/nano.yaml \
    -a output_to:socket \
    >> /workspace/jv/logs/detect.log 2>&1 &

PID1=$!

./src/spatial-audio/build/jsa-live-3d \
    --ipc ipc:///tmp/jv/audio/0.sock \
    --audio-buffer-ms 120 \
    --stream-timeout-ms 60 \
    --audio-azimuth-scale 3 \
    --audio-azimuth-max-deg 90 \
    --tone-min-gap-ms 200 \
    --source-mode tones \
    >> /workspace/jv/logs/audio.log 2>&1 &

PID2=$!

echo "Processes started:"
echo "main.py PID: $PID1"
echo "audio PID:   $PID2"

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