from .serial import serialize_dataclass
from ..representation.data import ObjectRepData, ObjectCoordData

x = ObjectRepData(
        objects=[
            ObjectCoordData(
                id=1,
                label=5,
                x_2d=20.,
                y_2d=40.,
                depth=0.
            ),
            ObjectCoordData(
                id=2,
                label=10,
                x_2d=25.,
                y_2d=35.,
                depth=0.
            ),
        ],
        frame_number=10,
        timestamp_ms=0.03333
    )
ser_x = serialize_dataclass(x)

print(', '.join(f'0x{b:02x}' for b in ser_x))
