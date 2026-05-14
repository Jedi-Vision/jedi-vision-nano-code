from calibrate import stereoCalibrateCamera
from jv.camera import FrameBuffer
import cv2

BOX_SIZE = 50  # mm
GRID_SIZE = (9, 9)  # width times height, internal corners not true pattern size!

gstreamer_kwargs = {
    "sensor_mode": 4,
    "capture_width": 2560,
    "capture_height": 720,
    "display_width": 2560,
    "display_height": 720,
    "flip_method": 2,
}

fb = FrameBuffer(
        size=1,
        warmup_frames=0,
        frame_skip=0,
        frame_rate=60,
        binocular=True,
        left_sensor_id=0,
        right_sensor_id=None,
        gstreamer_kwargs=gstreamer_kwargs,
    )

stereoCalibrateCamera(
    fb,
    "elp",
    chessboard_box_size=BOX_SIZE,
    chessboard_grid_size=GRID_SIZE,
    number_of_frames=100
)
