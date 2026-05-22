from calibrate import stereoCalibrateCamera
import argparse
import cv2
from pathlib import Path

BOX_SIZE = 25  # mm
GRID_SIZE = (10, 7)  # width times height, internal corners not true pattern size!

gstreamer_kwargs = {
    "sensor_mode": 4,
    "capture_width": 2560,
    "capture_height": 720,
    "display_width": 2560,
    "display_height": 720,
    "flip_method": 2,
}


def load_capture_folder(capture_folder):
    capture_folder = Path(capture_folder)
    left_paths = sorted((capture_folder / "left").glob("*"))
    right_paths = sorted((capture_folder / "right").glob("*"))

    if len(left_paths) != len(right_paths):
        raise ValueError("capture folder must contain the same number of left and right images")
    if len(left_paths) == 0:
        raise ValueError("capture folder must contain images in left/ and right/")

    img_list_c1 = [cv2.imread(str(path)) for path in left_paths]
    img_list_c2 = [cv2.imread(str(path)) for path in right_paths]

    unreadable = [
        str(path)
        for path, image in zip(left_paths + right_paths, img_list_c1 + img_list_c2)
        if image is None
    ]
    if unreadable:
        raise ValueError(f"could not read image: {unreadable[0]}")

    return img_list_c1, img_list_c2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-folder", help="Folder containing left/ and right/ calibration images")
    args = parser.parse_args()

    if args.capture_folder:
        img_list_c1, img_list_c2 = load_capture_folder(args.capture_folder)
        stereoCalibrateCamera(
            None,
            "elp",
            chessboard_box_size=BOX_SIZE,
            chessboard_grid_size=GRID_SIZE,
            number_of_frames=100,
            img_list_c1=img_list_c1,
            img_list_c2=img_list_c2,
        )
    else:
        from jv.camera import FrameBuffer

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


if __name__ == "__main__":
    main()
