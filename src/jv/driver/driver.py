from jv.audio import ObjectBuffer
from jv.representation import YoloEnvironmentRepresentationModel
from jv.scene.video_depth_anything_code.video_depth_stream import VideoDepthAnything, MODEL_CONFIGS
from jv.camera import FrameBuffer
from typing import Literal
import torch
import cv2


class Driver:

    def __init__(
        self,
        device: Literal["cpu", "mps", "cuda"],
        object_model_name: str = "yolo11",
        vda_model_name: str = "vits",
        chkpts_folder: str = "./checkpoints",
        retain_frames: int = 30,
        object_buffer_size: int = 0,
        frame_buffer_size: int = 0,
        warmup_frames: int = 30,
        camera_index: str | int = 0,
        frame_skip: int = 2,
        frame_rate: int = 30,
        show_det: bool = False,
        metric: bool = False
    ) -> None:

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
        self.object_buffer = ObjectBuffer(size=object_buffer_size)
        self.scene_model = VideoDepthAnything(**MODEL_CONFIGS[vda_model_name])
        checkpoint_name = 'metric_video_depth_anything' if metric else 'video_depth_anything'
        self.scene_model.load_state_dict(
            torch.load(
                f'{chkpts_folder}/{checkpoint_name}_{vda_model_name}.pth',
                map_location='cpu'
            ),
            strict=True
        )
        self.scene_model = self.scene_model.to(torch.device(device)).eval()
        self.device = device
        self.show_det = show_det

    def model_run(self, frame):

        msg = self.env_model.run(frame, show_det=self.show_det)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for VDA
        depth = self.scene_model.infer_video_depth_one(
            frame,
            input_size=518,
            device=self.device,
            fp32=True  # fp32=False causes a black output and NaN values in model
        )

        # Add depth mask to msg and convert to meters
        depth = torch.tensor(depth)
        msg.mask = depth

        # Visualize depth map if wanted
        if self.show_det:
            colormap = self.scene_model.colormap
            # Normalize
            color_depth = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8)*255).to(torch.uint8)
            color_depth = colormap[color_depth]
            cv2.imshow("msg.mask", color_depth)
            cv2.waitKey(1)

        return msg

    def run(self):

        print("Starting...")
        self.object_buffer.start()
        self.frame_buffer.start()

        while True:

            frame = self.frame_buffer.get()
            if frame is None:
                continue
            msg = self.model_run(frame)
            self.object_buffer.put(msg)
