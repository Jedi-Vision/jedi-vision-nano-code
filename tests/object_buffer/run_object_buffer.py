from jv.audio import ObjectBuffer
from jv.representation import ObjectRepData, ObjectCoordData
import time


buffer = ObjectBuffer()
buffer.start()


i = 0
while True:
    print("Adding to queue...")
    buffer.put(ObjectRepData(
        objects=[ObjectCoordData(
            id=i,
            label=-1,
            x_2d=0.,
            y_2d=0.,
            depth=0.
        )],
        frame_number=i,
        timestamp_ms=(1/30)*i
    ))

    i += 1
    time.sleep(0.01)

buffer.stop()
