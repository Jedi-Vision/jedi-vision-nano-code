import cv2
from queue import Queue, Empty
import threading
import time


class FrameBuffer:

    def __init__(
        self,
        size: int = 0,
        camera_index: int | str = 0,
        warmup_frames: int = 30,
        frame_skip: int = 2,
        frame_rate: int = 30,
    ):
        """
        A one-way frame buffer for reading from a Camera stream using cv2.

        Automatically starts capturing on instantiation.

        Args:
            size (int, optional): Maximum number of frames to store in the queue.
            camera_index (int | str, optional): Index of the camera to capture from (default is 0).
            warmup_frames (int, optional): Number of frames to run without adding to queue.
            frame_skip (int, optional): Number of frames to skip over (to decrease throughput)
            frame_rate (int, optional): Frame rate of video (default 30fps), will sleep to ensure
                that video frames are not added to queue faster than frame rate of video.
        """
        self.camera_index = camera_index
        self.max_size = size
        self.q = Queue(size)
        self.capture = cv2.VideoCapture(camera_index)
        self.frame_count = 0
        self.frame_skip = frame_skip
        self.warmup_frames = warmup_frames
        self.frame_rate = frame_rate
        self.running = False
        self.thread = None

    def start(self):
        """Starts the frame capturing thread."""
        if not self.capture.isOpened():
            raise RuntimeError("Failed to open camera.")
        self.running = True
        self.thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the frame capturing thread and releases the camera."""
        self.q.join()
        self.running = False
        if self.thread is not None:
            self.thread.join()
        self.capture.release()

    def _capture_frames(self):
        """Worker for capturing frames from the camera and storing them in the queue."""
        while self.running:
            ret, frame = self.capture.read()
            self.frame_count += 1

            # Warmup
            if self.frame_count < self.warmup_frames:
                continue

            # Skip frame
            if self.frame_count % self.frame_skip != 0:
                continue

            # Ensure that throughput does not exceed video framerate
            time.sleep(1/self.frame_rate)

            if not ret:
                break
            if self.q.full():
                self.q.get(timeout=0.001)  # Remove the oldest frame to make space
            self.q.put(frame, timeout=0.001)

    def get(self):
        """
        Retrieves the next frame from the queue.

        :return: The next frame, or None if the queue is empty.
        """
        try:
            frame = self.q.get(timeout=0.001)
            self.q.task_done()
            return frame
        except Empty as e:
            print(f"Frame buffer empty: {e}")
            return None

# Example usage:
# if __name__ == "__main__":
#     fb = FrameBuffer(size=10, camera_index=0)
#     fb.start()
#     try:
#         while True:
#             frame = fb.get()
#             if frame is not None:
#                 cv2.imshow("Frame", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#     finally:
#         fb.stop()
#         cv2.destroyAllWindows()
