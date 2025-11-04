"""
File: environment.py
Author: Colin Pannikkat
Date: Fall 2025
Description: This file contains the implementation of the SegFormerEnvironmentRepresentationModelClass,
             which is a model class for environment representation using SegFormer.
"""

from .abstract import AbstractModelClass
import torch
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
from .ade_id import ADE_ID_TO_LABEL


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

    def run(self, input: torch.Tensor) -> ObjectRepData:
        batch_feature = self.preprocess(input)
        output = self.process(batch_feature)
        return self.postprocess(output, input=input)

    def preprocess(self, input: torch.Tensor, **kwargs) -> BatchFeature:
        inputs = self.processor(images=input, return_tensors="pt")
        inputs.to(self.device)
        return inputs

    def process(self, input: BatchFeature, **kwargs) -> SemanticSegmenterOutput:
        return self.model(**input)

    def postprocess(
        self,
        out: SemanticSegmenterOutput,
        **kwargs
    ) -> ObjectRepData:

        logits = out.logits

        input: torch.Tensor | None = kwargs.get("input", None)

        if input is None or logits is None:
            raise Exception("Fatal error on model run.")

        # TODO: diff conf per class
        # TODO: Add output conversion from segmask
        return ObjectRepData([], mask=torch.zeros_like(input))

    def postprocess_to_image(
        self,
        out: SemanticSegmenterOutput,
        frame: np.ndarray,
        add_text_labels: bool = False,
        epsilon: float = 0.5
    ) -> np.ndarray:

        logits = out.logits

        if logits is None:
            raise Exception("Fatal error on model run.")

        # Resize
        target_size = (frame.shape[0], frame.shape[1])
        upsampled_logits = torch.nn.functional.interpolate(
            input=logits,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        # Get segmentation map labels
        # TODO: Add per class prob exclusion
        pred_labels = torch.argmax(upsampled_logits, dim=1)[0].cpu().numpy()

        # Resize, create color mask, and apply to original frame
        palette = np.array(ade_palette(), dtype=np.uint8)
        color_mask = palette[pred_labels]
        masked = cv2.addWeighted(frame, 1, color_mask, 0.5, 0)

        # Apply text label based on centroid mask parts
        if add_text_labels:
            label_overlay = masked.copy()
            for cls_id in np.unique(pred_labels):
                if cls_id >= len(ADE_ID_TO_LABEL):
                    continue
                mask = (pred_labels == cls_id).astype(np.uint8)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

                for i in range(num_labels):
                    if stats[i, cv2.CC_STAT_AREA] < 500:
                        continue
                    cx, cy = centroids[i]
                    cx, cy = int(cx), int(cy)
                    cv2.putText(
                        img=label_overlay,
                        text=ADE_ID_TO_LABEL[str(cls_id)],
                        org=(cx, cy),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 255, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA
                    )
            return label_overlay

        return masked
