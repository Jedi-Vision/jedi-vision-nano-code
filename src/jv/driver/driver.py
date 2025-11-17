from jv.audio import ObjectBuffer
from jv.representation import YoloEnvironmentRepresentationModel
# from jv.scene.video_depth_anything_code.video_depth_anything.video_depth_stream import VideoDepthAnything
from jv.camera import FrameBuffer
from typing import Literal
# import torch


class Driver:

    def __init__(
        self,
        object_model_name: str,
        device: Literal["cpu", "mps", "cuda"],
        retain_frames: int = 30,
        object_buffer_size: int = 0,
        frame_buffer_size: int = 0,
        warmup_frames: int = 30,
        camera_index: str | int = 0,
        frame_skip: int = 2,
        frame_rate: int = 30
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
            retain_frames=retain_frames
        )
        self.object_buffer = ObjectBuffer(size=object_buffer_size)
        # self.scene_model = VideoDepthAnything(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        self.device = device
        self.depth_maps = []

    def model_run(self, frame):

        msg = self.env_model.run(frame)
        # msg.mask = torch.mul(msg.mask, depth)  # apply binary mask to depth mask

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
            self.scene_model.infer_video_depth_one(frame, input_size=518, device=self.device, fp32=True)
