"""
File: data.py
Author: Colin Pannikkat
Date: Fall 2025
Description: This file defines data representations for object coordinate data and object
             representation data. It includes classes for storing object coordinates with
             associated metadata and a mask tensor.
"""

from dataclasses import dataclass, fields, is_dataclass
import torch


@dataclass
class ObjectCoordData:
    id: int
    label: int
    x_2d: float  # Normalized [0, 1]
    y_2d: float  # Normalized [0, 1]
    depth: float  # Depth in meters


@dataclass
class ObjectRepData:
    frame_number: int
    timestamp_ms: float
    objects: list[ObjectCoordData]

    def to_dict(self):
        return {
            "frame_number": self.frame_number,
            "timestamp_ms": self.timestamp_ms,
            "objects": [
                {
                    "id": obj.id,
                    "x_2d": obj.x_2d,
                    "y_2d": obj.y_2d,
                    "depth": obj.depth,
                }
                for obj in self.objects
            ],
        }

    def to_protobuf(self, proto_cls):
        """
        Convers the ObjectRepData instance to a Protobuf class.
        """
        return dataclass_to_proto(self, proto_cls)


def dataclass_to_proto(dc, proto_cls):
    """
    Source: ChatGPT, thank the robot overlords above
    """
    msg = proto_cls()
    for field in fields(dc):
        value = getattr(dc, field.name)

        if is_dataclass(value):
            # Nested message
            sub_msg = dataclass_to_proto(value, getattr(proto_cls, field.name).__class__)
            getattr(msg, field.name).CopyFrom(sub_msg)

        elif isinstance(value, list):
            # Repeated fields (maybe nested)
            lst = getattr(msg, field.name)
            for item in value:
                if is_dataclass(item):
                    sub_msg = getattr(msg, field.name).add()
                    sub_msg.CopyFrom(
                        dataclass_to_proto(item, sub_msg.__class__)
                    )
                else:
                    lst.append(item)
        elif isinstance(value, torch.Tensor):
            setattr(msg, field.name, value.numpy().tobytes())
        else:
            setattr(msg, field.name, value)

    return msg
