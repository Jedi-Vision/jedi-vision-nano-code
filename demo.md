# Demo Startup

## Steps

1. Start the display manager:

   ```bash
   sudo systemctl start gdm
   ```

2. Start the Bluetooth service:

   ```bash
   sudo systemctl start bluetooth.service
   ```

3. Open the terminal on the GUI and close it.

4. Start the docker container

   ```bash
   bash start_container.sh
   ```

5. In the docker shell, pull the latest code changes

   ```bash
   git pull
   ```

6. Build the spatial audio (on the host, outside docker)

   ```bash
   cd src/spatial-audio
   cmake --preset vcpkg
   cmake --build build -j
   cd ../..
   ```

7. Start the object detection with output to socket (Note: Keep running this command until it works, irratic behaviour of the code)

   ```bash
   python3 main.py -c config/nano.yaml -a output_to:socket
   ```

8. Start the spatial audio visual monitor

   ```bash
   ./src/spatial-audio/build/jsa-visual-monitor \
     --ipc ipc:///tmp/jv/audio/0.sock \
     --forward-ipc ipc:///tmp/jv/audio/1.sock
   ```

9. Start the spatial audio

   ```bash
    ./src/spatial-audio/build/jsa-live-3d \
    --ipc ipc:///tmp/jv/audio/1.sock \
    --audio-buffer-ms 120 \
    --max-interp-window-ms 25 \
    --stale-frame-drop-ms 200 \
    --audio-azimuth-scale 2.75 \
    --audio-azimuth-max-deg 90 \
    --tone-min-gap-ms 200 \
    --source-mode tones
   ```

   ```bash
    ./src/spatial-audio/build/jsa-live-3d \
    --ipc ipc:///tmp/jv/audio/1.sock \
    --audio-buffer-ms 120 \
    --max-interp-window-ms 25 \

    --audio-azimuth-scale 2.75 \
    --audio-azimuth-max-deg 90 \
    --source-mode songs
   ```
