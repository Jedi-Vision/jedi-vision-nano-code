"""
File: abstract.py
Author: Colin Pannikkat
Date: Fall 2025
Description: This file defines the AbstractModelClass, an abstract base class for implementing
             machine learning models. It enforces the implementation of methods for running
             the model, preprocessing inputs, performing forward passes, and postprocessing outputs.
"""

from abc import ABC, abstractmethod
import torch
from .data import ObjectRepData
from .exceptions import InvalidTorchDeviceException


class AbstractModelClass(ABC):
    def __init__(self, model: torch.nn.Module, device: str):
        self.model = model

        match device:
            case "mps":
                if torch.backends.mps.is_available():
                    self.device = torch.device(device)
                    print("Setting device to MPS.")
                else:
                    raise InvalidTorchDeviceException(device)
            case "cuda":
                if torch.cuda.is_available():
                    self.device = torch.device(device)
                    print("Setting device to CUDA.")
                else:
                    raise InvalidTorchDeviceException(device)
            case "cpu":
                self.device = torch.device(device)
                print("Setting device to CPU.")
            case _:
                raise InvalidTorchDeviceException(device)

    @abstractmethod
    def run(self, input, device: str) -> ObjectRepData:
        pass

    @abstractmethod
    def preprocess(self, input) -> ...:
        pass

    @abstractmethod
    def forward(self, input) -> ...:
        pass

    @abstractmethod
    def postprocess(self, out) -> ObjectRepData:
        pass
