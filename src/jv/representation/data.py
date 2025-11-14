"""
File: data.py
Author: Colin Pannikkat
Date: Fall 2025
Description: This file defines data representations for object coordinate data and object
             representation data. It includes classes for storing object coordinates with
             associated metadata and a mask tensor.
"""

from typing import Union, List
from dataclasses import dataclass, fields, is_dataclass
import torch


@dataclass
class ObjectXYCoordData:
    object_id: Union[int, None]
    label: Union[str, None]
    x: float
    y: float


@dataclass
class ObjectRepData:
    object_coordinates: List[ObjectXYCoordData]
    mask: torch.Tensor

    def to_dict(self):
        """
        Converts the ObjectRepData instance to a python dictionary.
        """
        return {
            "object_coordinates": [
                {
                    "object_id": coord.object_id,
                    "label": coord.label,
                    "x": coord.x,
                    "y": coord.y,
                }
                for coord in self.object_coordinates
            ],
            "mask": self.mask.tolist(),
        }

    def to_protobuf(self, proto_cls):
        """
        Convers the ObjectRepData instance to a Protobuf class.
        """
        return dataclass_to_proto(self, proto_cls)


def dataclass_to_proto(dc, proto_cls):
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
