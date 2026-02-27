import argparse
import cv2
import numpy as np
from jv.camera import FrameBuffer


def get_video_writer(filename, frame_shape, fps):
    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(filename, fourcc, fps, (width, height))


def record_binocular(frame_rate, frame_skip, buffer_size, warmup, gstreamer_kwargs, left_path, right_path, max_frames=None):
    print("=== Binocular FrameBuffer Record ===")
    print(f"  Frame rate       : {frame_rate}")
    print(f"  Frame skip       : {frame_skip}")
    print(f"  Buffer size      : {buffer_size}")
    print(f"  Warmup frames    : {warmup}")
    print(f"  GStreamer kwargs : {gstreamer_kwargs}")
    print(f"  Left video path  : {left_path}")
    print(f"  Right video path : {right_path}")
    print()

    fb = FrameBuffer(
        size=buffer_size,
        warmup_frames=warmup,
        frame_skip=frame_skip,
        frame_rate=frame_rate,
        binocular=True,
        gstreamer_kwargs=gstreamer_kwargs,
    )

    fb.start()
    print("FrameBuffer started (binocular). Press 'q' to quit.")

    left_writer = None
    right_writer = None
    frame_count = 0

    try:
        while True:
            result = fb.get()
            if result is None:
                continue

            (frame_left, frame_right), frame_number, timestamp_ms = result

            if left_writer is None or right_writer is None:
                left_writer = get_video_writer(left_path, frame_left.shape, frame_rate)
                right_writer = get_video_writer(right_path, frame_right.shape, frame_rate)

            left_writer.write(frame_left)
            right_writer.write(frame_right)
            frame_count += 1

            # Optionally display for feedback
            combined = np.hstack((frame_left, frame_right))
            cv2.imshow("Recording (Left | Right)", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if max_frames is not None and frame_count >= max_frames:
                print(f"Reached max_frames={max_frames}")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print(f"Total frames recorded: {frame_count}")
        fb.stop()
        if left_writer:
            left_writer.release()
        if right_writer:
            right_writer.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Record stereo videos from FrameBuffer")
    parser.add_argument("--frame-rate", type=int, default=30, help="Frame rate (default: 30)")
    parser.add_argument("--frame-skip", type=int, default=0, help="Number of frames to skip between captures (default: 0)")
    parser.add_argument("--buffer-size", type=int, default=1, help="Max frames in the queue (default: 1)")
    parser.add_argument("--warmup", type=int, default=30, help="Number of warmup frames before buffering (default: 30)")
    parser.add_argument("--capture-width", type=int, default=1280, help="GStreamer capture width (default: 1280)")
    parser.add_argument("--capture-height", type=int, default=720, help="GStreamer capture height (default: 720)")
    parser.add_argument("--display-width", type=int, default=1280, help="GStreamer display width (default: 1280)")
    parser.add_argument("--display-height", type=int, default=720, help="GStreamer display height (default: 720)")
    parser.add_argument("--sensor-mode", type=int, default=3, help="GStreamer sensor mode (default: 3)")
    parser.add_argument("--flip-method", type=int, default=2, help="GStreamer flip method (default: 0)")
    parser.add_argument("--left-path", type=str, default="left.mp4", help="Output path for left video")
    parser.add_argument("--right-path", type=str, default="right.mp4", help="Output path for right video")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to record (default: unlimited)")

    args = parser.parse_args()

    gstreamer_kwargs = {
        "sensor_mode": args.sensor_mode,
        "capture_width": args.capture_width,
        "capture_height": args.capture_height,
        "display_width": args.display_width,
        "display_height": args.display_height,
        "framerate": args.frame_rate,
        "flip_method": args.flip_method,
    }

    record_binocular(
        frame_rate=args.frame_rate,
        frame_skip=args.frame_skip,
        buffer_size=args.buffer_size,
        warmup=args.warmup,
        gstreamer_kwargs=gstreamer_kwargs,
        left_path=args.left_path,
        right_path=args.right_path,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
