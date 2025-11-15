from jv.serial import serialize_dataclass
from jv.representation.data import ObjectRepData, ObjectXYCoordData
import torch


# Add your test functions below
def test_serialize_dataclass():
    x = ObjectRepData(
        object_coordinates=[
            ObjectXYCoordData(
                object_id=1,
                label="car",
                x=1,
                y=1
            )
        ],
        mask=torch.ones((512, 512))
    )
    ser_x = serialize_dataclass(x)

    assert isinstance(ser_x, bytes)
