"""
File: data.py
Author: Colin Pannikkat
Date: Fall 2025
Description: This file defines data representations for object coordinate data and object 
             representation data. It includes classes for storing object coordinates with 
             associated metadata and a mask tensor.
"""

from typing import Union, List
from dataclasses import dataclass
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
