from calibrate import stereoCalibrateCamera
from jv.camera import FrameBuffer

BOX_SIZE = 25  # mm
GRID_SIZE = (11, 8)  # width times height

gstreamer_kwargs = {
    "sensor_mode": 3,
    "capture_width": 1280,
    "capture_height": 720,
    "display_width": 1280,
    "display_height": 720,
    "framerate": 30,
    "flip_method": 2,
}

fb = FrameBuffer(
        size=1,
        warmup_frames=0,
        frame_skip=0,
        frame_rate=30,
        binocular=True,
        gstreamer_kwargs=gstreamer_kwargs,
    )

stereoCalibrateCamera(
    fb,
    "imx219-83",
    chessboard_box_size=BOX_SIZE,
    chessboard_grid_size=GRID_SIZE
)
