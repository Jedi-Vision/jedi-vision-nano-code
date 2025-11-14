from jv.audio import ObjectBuffer
from jv.representation import YoloEnvironmentRepresentationModel
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
    ) -> None:

        self.frame_buffer = FrameBuffer(size=frame_buffer_size)
        self.object_buffer = ObjectBuffer(size=object_buffer_size)
        self.env_model = YoloEnvironmentRepresentationModel(
            model_name=object_model_name,
            device=device,
            retain_frames=retain_frames
        )
        self.scene_model = None
        self.warmup_frames = warmup_frames

    def model_run(self, frame):

        env = self.env_model.run(frame)
        # scene = self.scene_model.run(frame)
        # env.mask = torch.mul(msg.mask, scene)

        return env

    def run(self):

        print("Starting frame and object buffer.")
        self.frame_buffer.start()
        self.object_buffer.start()

        print("Starting run loop.")
        while True:

            if self.frame_buffer.frame_count < self.warmup_frames:
                continue

            frame = self.frame_buffer.get()
            msg = self.env_model.run(frame)
            self.object_buffer.put(msg)
