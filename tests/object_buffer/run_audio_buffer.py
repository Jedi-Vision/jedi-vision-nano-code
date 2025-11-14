from jv.audio import ObjectBuffer
from jv.representation import ObjectRepData, ObjectXYCoordData
import time
import torch


buffer = ObjectBuffer()
buffer.start()

while True:
    print("Adding to queue...")
    buffer.put(ObjectRepData(
        object_coordinates=[ObjectXYCoordData(
            object_id=1,
            label="car",
            x=0,
            y=0
        )],
        mask=torch.ones((512, 512))
    ))
    time.sleep(0.1)

buffer.stop()