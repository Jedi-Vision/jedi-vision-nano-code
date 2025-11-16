from jv.audio import ObjectBuffer
from jv.representation import YoloEnvironmentRepresentationModel
from jv.scene.video_depth_anything.video_depth_anything.video_depth_stream import VideoDepthAnything
from jv.camera import FrameBuffer
from typing import Literal


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
        frame_skip: int = 2
    ) -> None:

        self.frame_buffer = FrameBuffer(
            size=frame_buffer_size,
            camera_index=camera_index,
            warmup_frames=warmup_frames,
            frame_skip=frame_skip
        )
        
        self.env_model = YoloEnvironmentRepresentationModel(
            model_name=object_model_name,
            device=device,
            retain_frames=retain_frames
        )
        self.object_buffer = ObjectBuffer(size=object_buffer_size)
        self.scene_model = VideoDepthAnything()
        self.device = device
        self.depth_maps = []

    def model_run(self, frame):

        env = self.env_model.run(frame)
        # scene = self.scene_model.run(frame)
        # env.mask = torch.mul(msg.mask, scene)

        return env

    def run(self):

        print("Starting...")
        self.object_buffer.start()
        self.frame_buffer.start()

        while True:

            frame = self.frame_buffer.get()
            if frame is None:
                continue
            msg = self.env_model.run(frame)
            self.object_buffer.put(msg)
            depth = self.scene_model.infer_video_depth_one(
                        frame, 
                        input_size=len(frame[1]), 
                        device=self.device, 
                        fp32=True
                    )
            self.depth_maps.append(depth)
