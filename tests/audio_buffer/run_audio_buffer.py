from jv.audio import AudioBuffer
from jv.representation import ObjectRepData, ObjectXYCoordData
import time
import torch


buffer = AudioBuffer()

while True:
    print("Adding to queue...")
    buffer.queue_message(ObjectRepData(
        object_coordinates=[ObjectXYCoordData(
            object_id=1,
            label="car",
            x=0,
            y=0
        )],
        mask=torch.ones((512, 512))
    ))
    time.sleep(0.1)
