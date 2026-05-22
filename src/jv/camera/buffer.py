import cv2
from queue import Queue, Empty
import threading
import time
import logging


class FrameBuffer:

    def __init__(
        self,
        size: int = 0,
        camera_index: int | str = 0,
        left_sensor_id: int | str = 0,
        right_sensor_id: int | str | None = 1,
        warmup_frames: int = 30,
        frame_skip: int = 2,
        frame_rate: int = 30,
        binocular: bool = False,
        use_gstreamer: bool = False,
        gstreamer_kwargs: dict = {}
    ):
        """
        A one-way frame buffer for reading from a Camera stream using cv2.

        Automatically starts capturing on instantiation. Stores a frame with it's
        video timestamp in ms (calculated according to FPS), and frame number.

        Depending on whether use_binocular is true or not, a frame can include two one or two
        images. When use_binocular is enabled, two separate _capture_frames threads are
        ran for both camera's. Using binocular accesses the camera's using Nvidia's
        GStreamer pipeline, which allows for quick direct access to the camera's through
        the camera serial interface (CSI).

        Args:
            size (int, optional): Maximum number of frames to store in the queue.
            camera_index (int | str, optional): Index of the camera to capture from (default is 0).
            left_sensor_id (int | str, optional): Sensor ID for left camera (default is 0). Can also pass
                video file path of left binocular image to simulate binocular video stream.
            right_sensor_id (int | str, optional): Sensor ID for right camera (default is 1). Can also pass
                video file path of left binocular image to simulate binocular video stream.
            warmup_frames (int, optional): Number of frames to run without adding to queue.
            frame_skip (int, optional): Number of frames to skip over (to decrease throughput)
            frame_rate (int, optional): Frame rate of video (default 30fps), will sleep to ensure
                that video frames are not added to queue faster than frame rate of video.
            binocular (bool, optional): Whether to use binocular camera's. Set's up two
                camera streams and returns frame from both camera on get().
            gstreamer_args (dict, optional): GStreamer kwargs. Arguments include:
                - sensor_mode (int, optional): Camera sensor mode (default is 3).
                - capture_width (int, optional): Width of the captured video (default is 1280).
                - capture_height (int, optional): Height of the captured video (default is 720).
                - display_width (int, optional): Width of the displayed video (default is 1280).
                - display_height (int, optional): Height of the displayed video (default is 720).
                - framerate (int, optional): Frame rate of the video (default is 30).
                - flip_method (int, optional): Flip method for the video (default is 0).
        """
        self.camera_index = camera_index
        self.max_size = size
        self.q = Queue(size)
        self.frame_count = 0
        self.frame_skip = frame_skip
        self.warmup_frames = warmup_frames
        self.frame_rate = frame_rate
        self.running = False
        self.thread = None
        self.binocular = binocular
        self.use_gstreamer = use_gstreamer
        self.gstreamer_kwargs = gstreamer_kwargs
        self.left_sensor_id = left_sensor_id
        self.right_sensor_id = right_sensor_id
        self.split_bino = False

    def start(self):
        """Starts the frame capturing thread(s)."""
        if self.binocular:
            if self.right_sensor_id is None:
                self.split_bino = True
                if self.use_gstreamer and not isinstance(self.left_sensor_id, str):
                    self.capture_left = cv2.VideoCapture(
                        gstreamer_pipeline(
                            sensor_id=self.left_sensor_id,
                            framerate=self.frame_rate,
                            **self.gstreamer_kwargs
                        ),
                        cv2.CAP_GSTREAMER
                    )
                else:
                    self.capture_left = cv2.VideoCapture(self.left_sensor_id)
            else:
                if isinstance(self.left_sensor_id, str) or isinstance(self.right_sensor_id, str):
                    self.capture_left = cv2.VideoCapture(self.left_sensor_id)
                    self.capture_right = cv2.VideoCapture(self.right_sensor_id)
                    self.use_gstreamer = False
                else:
                    if self.use_gstreamer:
                        self.capture_left = cv2.VideoCapture(
                            gstreamer_pipeline(
                                sensor_id=self.left_sensor_id,
                                framerate=self.frame_rate,
                                **self.gstreamer_kwargs
                            ),
                            cv2.CAP_GSTREAMER
                        )
                        self.capture_right = cv2.VideoCapture(
                            gstreamer_pipeline(
                                sensor_id=self.right_sensor_id,
                                framerate=self.frame_rate,
                                **self.gstreamer_kwargs
                            ),
                            cv2.CAP_GSTREAMER
                        )
                    else:
                        self.capture_left = cv2.VideoCapture(self.left_sensor_id)
                        self.capture_right = cv2.VideoCapture(self.right_sensor_id)

            if self.split_bino:
                if not self.capture_left.isOpened():
                    raise RuntimeError("Failed to open camera.")
            elif not self.capture_left.isOpened() or not self.capture_right.isOpened():
                raise RuntimeError("Failed to open one or both cameras.")

            self.running = True
            self.thread = threading.Thread(target=self._capture_frames_binocular, daemon=True)
            self.thread.start()
        else:
            if self.use_gstreamer:
                self.capture = cv2.VideoCapture(
                    gstreamer_pipeline(
                        sensor_id=int(self.camera_index),
                        framerate=self.frame_rate,
                        **self.gstreamer_kwargs
                    ),
                    cv2.CAP_GSTREAMER
                )
            else:
                self.capture = cv2.VideoCapture(self.camera_index)
            if not self.capture.isOpened():
                raise RuntimeError("Failed to open camera.")
            self.running = True
            self.thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.thread.start()

    def stop(self):
        """Stops the frame capturing thread(s) and releases the camera(s)."""
        self.running = False
        if self.binocular:
            if self.thread is not None:
                self.thread.join()
            if self.capture_left is not None:
                self.capture_left.release()
            try:
                if self.capture_right is not None:
                    self.capture_right.release()
            except AttributeError:
                pass
        else:
            if self.thread is not None:
                self.thread.join()
            if self.capture is not None:
                self.capture.release()

    def _capture_frames_binocular(self):
        """Worker for capturing frames from both cameras and storing them in the queue."""
        frame_count = 0

        while self.running:
            ret_left, frame_left = self.capture_left.read()
            if self.split_bino:
                ret_right, frame_right = ret_left, None
            else:
                ret_right, frame_right = self.capture_right.read()
            frame_count += 1
            timestamp_ms = time.time() * 1000

            # Warmup
            if frame_count < self.warmup_frames:
                continue

            # Skip frame
            if self.frame_skip != 0 and frame_count % self.frame_skip != 0:
                continue

            if not ret_left or not ret_right:
                break

            if self.split_bino and frame_left is not None:
                mid = frame_left.shape[1] // 2
                frame_right = frame_left[:, mid:]
                frame_left = frame_left[:, :mid]

            if self.q.full():
                self.q.get(timeout=0.001)  # Remove the oldest frame to make space

            # If not using GStreamer, there is no downscaling, so we need to
            # manually downscale the image before adding to queue.
            # if split bino, we need to halve the display_width for scaling
            if not self.use_gstreamer:
                if frame_left is not None and frame_right is not None:
                    [frame_left, frame_right] = map(
                        lambda f: cv2.resize(
                            f,
                            dsize=(
                                self.gstreamer_kwargs.get("display_width", 1280) // 2 if self.split_bino
                                else self.gstreamer_kwargs.get("display_width", 1280),
                                self.gstreamer_kwargs.get("display_height", 720)),
                            interpolation=cv2.INTER_AREA
                        ),
                        [frame_left, frame_right]
                    )

            # Add both frames to the queue
            self.q.put(((frame_left, frame_right), frame_count, timestamp_ms), timeout=0.001)

            # Ensure that throughput does not exceed video framerate
            # time.sleep(1 / self.frame_rate)

    def _capture_frames(self):
        """Worker for capturing frames from a single camera and storing them in the queue."""
        frame_count = 0

        while self.running:
            ret, frame = self.capture.read()
            frame_count += 1
            timestamp_ms = time.time() * 1000

            # Warmup
            if frame_count < self.warmup_frames:
                continue

            # Skip frame
            if self.frame_skip != 0 and frame_count % self.frame_skip != 0:
                continue

            if not ret:
                break

            if self.q.full():
                self.q.get(timeout=0.001)  # Remove the oldest frame to make space

            if frame is not None and not self.use_gstreamer:
                frame = cv2.resize(
                    frame,
                    (self.gstreamer_kwargs.get("display_width", 1280),
                        self.gstreamer_kwargs.get("display_height", 720))
                )

            # Add frame to the queue
            self.q.put(((frame,), frame_count, timestamp_ms), timeout=0.001)

            # Ensure that throughput does not exceed video framerate
            # time.sleep(1 / self.frame_rate)

    def get(self):
        """
        Retrieves the next frame from the queue.

        :return: The next frame(s), along with timestamp info and frame number,
                 or None if the queue is empty.
        """
        try:
            frame = self.q.get(timeout=0.001)
            self.q.task_done()
            return frame
        except Empty:
            return None


def gstreamer_pipeline(
    sensor_id: int = 0,
    sensor_mode: int = 4,  # (1280x720, 59.9999 fps)
    capture_width: int = 1280,
    capture_height: int = 720,
    display_width: int = 1280,
    display_height: int = 720,
    framerate: int = 30,
    flip_method: int = 0,
):
    """
    Pulled from:
    https://github.com/asujaykk/Stereo-Camera-Depth-Estimation-And-3D-visulaization-on-jetson-nano/blob/main/Camera/jetsonCam.py

    Credit to akhil_kk.
    """

    return (
        "nvarguscamerasrc sensor_id=%d sensor_mode=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip_method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink"
        % (
            sensor_id,
            sensor_mode,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

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
