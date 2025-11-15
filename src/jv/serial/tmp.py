from .serial import serialize_dataclass
from ..representation.data import ObjectRepData, ObjectXYCoordData
import torch

x = ObjectRepData(
        object_coordinates=[
            ObjectXYCoordData(
                object_id=1,
                label=5,
                x=20.,
                y=40.
            ),
            ObjectXYCoordData(
                object_id=2,
                label=10,
                x=25.,
                y=35.
            ),
        ],
        mask=torch.ones((512, 512))
    )
ser_x = serialize_dataclass(x)

print(', '.join(f'0x{b:02x}' for b in ser_x))  # Output: 0x00 0x01 0x02 0xff 0xfe
