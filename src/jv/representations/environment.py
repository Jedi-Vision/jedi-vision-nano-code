"""
File: environment.py
Author: Colin Pannikkat
Date: Fall 2025
Description: This file contains the implementation of the SegFormerEnvironmentRepresentationModelClass,
             which is a model class for environment representation using SegFormer.
"""

from .abstract import AbstractModelClass
import torch
from PIL.Image import Image
from .data import ObjectRepData
from transformers import (
    BatchFeature,
    SegformerImageProcessorFast,
    SegformerForSemanticSegmentation,
)
from transformers.modeling_outputs import SemanticSegmenterOutput
import cv2
import numpy as np
from .ade_palette import ade_palette


SEG_MODEL_ZOO = {
    "ade-small": "nvidia/segformer-b0-finetuned-ade-512-512",
}

CLASS_TO_CONF = {

}


class SegFormerEnvironmentRepresentationModelClass(AbstractModelClass):
    def __init__(self, seg_model_name: str, k: int, device: str):
        super().__init__(self._setup_model(seg_model_name), device)
        self.model.to(self.device)
        self.k = k

    def _setup_model(self, seg_model_name) -> torch.nn.Module:

        model = SEG_MODEL_ZOO[seg_model_name]
        self.processor = SegformerImageProcessorFast.from_pretrained(model)
        model = SegformerForSemanticSegmentation.from_pretrained(model)

        return model

    def run(self, input: Image, device: str) -> ObjectRepData:
        batch_feature = self.preprocess(input, device=device)
        output = self.forward(batch_feature, device=device)
        return self.postprocess(output, device=device, input=input)

    def preprocess(self, input: Image, **kwargs) -> BatchFeature:
        inputs = self.processor(images=input, return_tensors="pt")
        inputs.to(self.device)
        return inputs

    def forward(self, input, **kwargs) -> SemanticSegmenterOutput:
        return self.model(**input)

    def postprocess(self, out: SemanticSegmenterOutput, **kwargs) -> ObjectRepData:
        logits = out.logits
        input: Image | None = kwargs.get("input", None)

        if input is None or logits is None:
            raise Exception("Fatal error on model run.")

        original_size = (input.size[1], input.size[0])

        seg_map = torch.softmax(logits.squeeze(0), dim=0).detach().cpu().numpy()
        epsilon = 0.5
        arg_seg_map = np.argmax(seg_map, axis=0)
        confidence = np.max(seg_map, axis=0)

        # TODO: diff conf per class
        arg_seg_map[confidence < epsilon] = 0
        seg_map_resized = cv2.resize(arg_seg_map, original_size, interpolation=cv2.INTER_NEAREST)

        # Overlay segmentation map on the original frame
        # TODO: add label
        palette = np.array(ade_palette(), dtype=np.uint8)
        color_mask = palette[seg_map_resized]

        blended = cv2.addWeighted(input, 0.7, color_mask, 0.3, 0)

        cv2.imshow("Segmentation", blended)
        # Exit on 'q' key press
        while not (cv2.waitKey(1) & 0xFF == ord('q')):
            pass

        cv2.destroyAllWindows()

        # TODO: Add output conversion from segmask
        return ObjectRepData([], mask=torch.zeros_like(input))
