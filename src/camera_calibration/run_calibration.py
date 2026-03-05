from calibrate import stereoCalibrateCamera
from jv.camera import FrameBuffer

BOX_SIZE = 25  # mm
GRID_SIZE = (10, 7)  # width times height, internal corners not true pattern size!

gstreamer_kwargs = {
    "sensor_mode": 4,
    "capture_width": 1280,
    "capture_height": 720,
    "display_width": 1280,
    "display_height": 720,
    "flip_method": 2,
}

fb = FrameBuffer(
        size=1,
        warmup_frames=0,
        frame_skip=0,
        frame_rate=60,
        binocular=True,
        left_sensor_id=1,
        right_sensor_id=0,
        gstreamer_kwargs=gstreamer_kwargs,
    )

stereoCalibrateCamera(
    fb,
    "imx219-83",
    chessboard_box_size=BOX_SIZE,
    chessboard_grid_size=GRID_SIZE
)
