"""
Test script for FrameBuffer — monocular and binocular modes.

Usage:
    Monocular (default webcam):
        python tests/test_frame_buffer.py --mode mono

    Binocular (CSI cameras via GStreamer):
        python tests/test_frame_buffer.py --mode bino

    With options:
        python tests/test_frame_buffer.py --mode mono --camera 0 --frame-rate 30 --frame-skip 0 --buffer-size 10

Press 'q' to quit.
"""

import argparse
import cv2
import numpy as np
import time

from jv.camera import FrameBuffer


def test_monocular(camera_index, frame_rate, frame_skip, buffer_size, warmup):
    """Test monocular camera capture and display."""
    print("=== Monocular FrameBuffer Test ===")
    print(f"  Camera index : {camera_index}")
    print(f"  Frame rate   : {frame_rate}")
    print(f"  Frame skip   : {frame_skip}")
    print(f"  Buffer size  : {buffer_size}")
    print(f"  Warmup frames: {warmup}")
    print()

    fb = FrameBuffer(
        size=buffer_size,
        camera_index=camera_index,
        warmup_frames=warmup,
        frame_skip=frame_skip,
        frame_rate=frame_rate,
        binocular=False,
    )

    fb.start()
    print("FrameBuffer started (monocular). Press 'q' to quit.")

    frame_count = 0
    fps_start = time.time()
    display_fps = 0.0

    try:
        while True:
            result = fb.get()
            if result is None:
                continue

            frame, frame_number, timestamp_ms = result
            frame_count += 1

            # Calculate display FPS every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start
                display_fps = 30 / elapsed if elapsed > 0 else 0.0
                fps_start = time.time()

            # Overlay info
            info = f"Frame #{frame_number} | ts={timestamp_ms:.1f}ms | FPS={display_fps:.1f}"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Monocular", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print(f"Total frames displayed: {frame_count}")
        fb.stop()
        cv2.destroyAllWindows()


def test_binocular(frame_rate, frame_skip, buffer_size, warmup, gstreamer_kwargs):
    """Test binocular camera capture and display (CSI cameras via GStreamer)."""
    print("=== Binocular FrameBuffer Test ===")
    print(f"  Frame rate       : {frame_rate}")
    print(f"  Frame skip       : {frame_skip}")
    print(f"  Buffer size      : {buffer_size}")
    print(f"  Warmup frames    : {warmup}")
    print(f"  GStreamer kwargs  : {gstreamer_kwargs}")
    print()

    fb = FrameBuffer(
        size=buffer_size,
        warmup_frames=warmup,
        frame_skip=frame_skip,
        frame_rate=frame_rate,
        left_sensor_id=1,
        right_sensor_id=0,
        binocular=True,
        gstreamer_kwargs=gstreamer_kwargs,
    )

    fb.start()
    print("FrameBuffer started (binocular). Press 'q' to quit.")

    frame_count = 0
    fps_start = time.time()
    display_fps = 0.0

    try:
        while True:
            result = fb.get()
            if result is None:
                continue

            (frame_left, frame_right), frame_number, timestamp_ms = result
            frame_count += 1

            # Calculate display FPS every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start
                display_fps = 30 / elapsed if elapsed > 0 else 0.0
                fps_start = time.time()

            # Overlay info on both frames
            info = f"Frame #{frame_number} | ts={timestamp_ms:.1f}ms | FPS={display_fps:.1f}"
            cv2.putText(frame_left, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame_left, "LEFT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame_right, "RIGHT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Stack frames side by side for display
            combined = np.hstack((frame_left, frame_right))
            cv2.imshow("Binocular (Left | Right)", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print(f"Total frames displayed: {frame_count}")
        fb.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Test FrameBuffer monocular / binocular capture")
    parser.add_argument("--mode", type=str, choices=["mono", "bino"], default="mono",
                        help="Camera mode: 'mono' for monocular, 'bino' for binocular (default: mono)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index for monocular mode (default: 0)")
    parser.add_argument("--frame-rate", type=int, default=30,
                        help="Frame rate (default: 30)")
    parser.add_argument("--frame-skip", type=int, default=0,
                        help="Number of frames to skip between captures (default: 0)")
    parser.add_argument("--buffer-size", type=int, default=1,
                        help="Max frames in the queue (default: 10)")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Number of warmup frames before buffering (default: 10)")

    # GStreamer options (binocular only)
    parser.add_argument("--capture-width", type=int, default=1280,
                        help="GStreamer capture width (default: 1280)")
    parser.add_argument("--capture-height", type=int, default=720,
                        help="GStreamer capture height (default: 720)")
    parser.add_argument("--display-width", type=int, default=640,
                        help="GStreamer display width (default: 640)")
    parser.add_argument("--display-height", type=int, default=360,
                        help="GStreamer display height (default: 360)")
    parser.add_argument("--sensor-mode", type=int, default=4,
                        help="GStreamer sensor mode (default: 4)")
    parser.add_argument("--flip-method", type=int, default=0,
                        help="GStreamer flip method (default: 0)")

    args = parser.parse_args()

    if args.mode == "mono":
        test_monocular(
            camera_index=args.camera,
            frame_rate=args.frame_rate,
            frame_skip=args.frame_skip,
            buffer_size=args.buffer_size,
            warmup=args.warmup,
        )
    else:
        gstreamer_kwargs = {
            "sensor_mode": args.sensor_mode,
            "capture_width": args.capture_width,
            "capture_height": args.capture_height,
            "display_width": args.display_width,
            "display_height": args.display_height,
            "flip_method": args.flip_method,
        }
        test_binocular(
            frame_rate=args.frame_rate,
            frame_skip=args.frame_skip,
            buffer_size=args.buffer_size,
            warmup=args.warmup,
            gstreamer_kwargs=gstreamer_kwargs,
        )


if __name__ == "__main__":
    main()
