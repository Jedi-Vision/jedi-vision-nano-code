from jv.serial import serialize_dataclass
from jv.representation.data import ObjectRepData, ObjectCoordData


def test_serialize_dataclass():
    x = ObjectRepData(
        timestamp_ms=-1,
        frame_number=-1,
        objects=[
            ObjectCoordData(
                id=1,
                label=1,
                x_2d=1,
                y_2d=1,
                depth=0
            )
        ],
    )
    ser_x = serialize_dataclass(x)

    assert isinstance(ser_x, bytes)
