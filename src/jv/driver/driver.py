from jv.audio import ObjectBuffer
from jv.representation import YoloEnvironmentRepresentationModel
from jv.scene.video_depth_anything_code.video_depth_stream import VideoDepthAnything, MODEL_CONFIGS
from jv.camera import FrameBuffer
from jv.representation.data import ObjectRepData
from typing import Literal
import torch
import cv2
import time
from functools import wraps


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {(end - start) * 1000:.4f} ms")
        return result
    return wrapper


class Driver:

    def __init__(
        self,
        device: Literal["cpu", "mps", "cuda"],
        output_to: Literal["socket", "file", "none"] = "socket",
        object_model_name: str = "yolo11",
        vda_model_name: str = "vits",
        chkpts_folder: str = "./checkpoints",
        retain_frames: int = 30,
        object_buffer_size: int = 0,
        frame_buffer_size: int = 0,
        warmup_frames: int = 30,
        camera_index: str | int = 0,
        frame_skip: int = 0,
        frame_rate: int = 30,
        show_det: bool = False,
        depth: bool = True,
        metric: bool = False
    ) -> None:
        """
        Initializes the driver with specified configuration for device, output, models, buffers, and video processing.

        Args:
            device (Literal["cpu", "mps", "cuda"]): Device to run models on.
            output_to (Literal["socket", "file", "none"]): Output destination for object buffer.
            object_model_name (str, optional): Name of the object detection model. Defaults to "yolo11".
            vda_model_name (str, optional): Name of the video depth estimation model. Defaults to "vits".
            chkpts_folder (str, optional): Path to the folder containing model checkpoints. Defaults to "./checkpoints".
            retain_frames (int, optional): Number of frames to retain in environment model. Defaults to 30.
            object_buffer_size (int, optional): Size of the object buffer. Defaults to 0.
            frame_buffer_size (int, optional): Size of the frame buffer. Defaults to 0.
            warmup_frames (int, optional): Number of warmup frames for frame buffer. Defaults to 30.
            camera_index (str | int, optional): Index or identifier for the camera source. Defaults to 0.
            frame_skip (int, optional): Number of frames to skip between processing. Defaults to 0.
            frame_rate (int, optional): Frame rate for video capture. Defaults to 30.
            show_det (bool, optional): Whether to display detection results. Defaults to False.
            depth (bool, optional): Whether to enable depth estimation. Defaults to True.
            metric (bool, optional): Whether to use metric depth estimation model. Defaults to False.
        """

        self.frame_buffer = FrameBuffer(
            size=frame_buffer_size,
            camera_index=camera_index,
            warmup_frames=warmup_frames,
            frame_skip=frame_skip,
            frame_rate=frame_rate
        )

        self.env_model = YoloEnvironmentRepresentationModel(
            model_name=object_model_name,
            device=device,
            retain_frames=retain_frames,
        )

        self.object_buffer = ObjectBuffer(size=object_buffer_size, output_to=output_to)

        self.depth = depth
        self.scene_model = VideoDepthAnything(**MODEL_CONFIGS[vda_model_name])
        checkpoint_name = 'metric_video_depth_anything' if metric else 'video_depth_anything'
        self.scene_model.load_state_dict(
            torch.load(
                f'{chkpts_folder}/{checkpoint_name}_{vda_model_name}.pth',
                map_location='cpu',
                weights_only=True
            ),
            strict=True
        )
        self.scene_model = self.scene_model.to(torch.device(device)).eval()

        self.device = device
        self.show_det = show_det

    @timeit
    def model_run(self, frame, frame_number, timestamp_ms):

        objects = self.env_model.run(frame, show_det=self.show_det)

        if self.depth:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for VDA
            depth = self.scene_model.infer_video_depth_one(
                frame,
                input_size=518,
                device=self.device,
                fp32=True  # fp32=False causes a black output and NaN values in model
            )

            # Add depth information to objects
            depth = torch.tensor(depth)
            for obj in objects:
                obj.depth = depth[int(obj.y_2d)][int(obj.x_2d)].item()

            # Visualize depth map if wanted
            if self.show_det:
                colormap = self.scene_model.colormap
                # Normalize
                color_depth = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8)*255).to(torch.uint8)
                color_depth = colormap[color_depth]
                cv2.imshow("msg.mask", color_depth)
                cv2.waitKey(1)

        return ObjectRepData(
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
            objects=objects
        )

    def run(self):

        print("Starting...")

        if self.object_buffer is not None:
            self.object_buffer.start()
        self.frame_buffer.start()

        frame_count = 0

        while True:

            frame = self.frame_buffer.get()
            if frame is None:
                continue
            frame_count += 1
            msg = self.model_run(*frame)
            self.object_buffer.put(msg)
