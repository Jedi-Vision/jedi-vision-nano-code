"""
File: environment.py
Author: Colin Pannikkat
Date: Fall 2025
Description: This file contains the implementation of the SegFormerEnvironmentRepresentationModelClass,
             which is a model class for environment representation using SegFormer.
"""

# PyTorch stuff
import torch
import torch.nn.functional as F

# Local imports
from .data import ObjectRepData, ObjectXYCoordData
from .ade_utils import ade_palette, ADE_ID_TO_LABEL
from .abstract import AbstractModelClass

# Model imports
from transformers import (
    BatchFeature,
    SegformerImageProcessorFast,
    SegformerForSemanticSegmentation,
)
from ultralytics.models import YOLO
from ultralytics.engine.results import Results
from transformers.modeling_outputs import SemanticSegmenterOutput
import cv2
import numpy as np

# System stuff
from functools import reduce
from collections import defaultdict


SEG_MODEL_ZOO = {
    "ade-b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "ade-b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "ade-b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "ade-b3": "nvidia/segformer-b3-finetuned-ade-512-512",
    "ade-b4": "nvidia/segformer-b4-finetuned-ade-512-512"
}

YOLO_MODEL_ZOO = {
    "yolo11": "yolo11n.pt"
}

CLASS_TO_CONF = {

}


def mode_pool2d(mask, kernel_size, stride):
    """
    Performs mode pooling on a 2D class label mask.
    For each window, the most frequent label is selected.

    Args:
        mask (torch.Tensor): [H, W] tensor with integer class labels
        kernel_size (int): pooling window size
        stride (int): stride of pooling window

    Returns:
        torch.Tensor: pooled mask of shape [H_out, W_out]
    """
    # Add batch and channel dimensions
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # Use unfold to extract sliding windows
    patches = F.unfold(mask.float(), kernel_size=kernel_size, stride=stride)  # [1, K*K, L]
    patches = patches.long()  # convert back to int for mode computation

    # Compute mode along each column (i.e. per patch)
    mode_vals, _ = torch.mode(patches, dim=1)  # [1, L]

    # Reshape to 2D
    H_out = (mask.shape[2] - kernel_size) // stride + 1
    W_out = (mask.shape[3] - kernel_size) // stride + 1
    return mode_vals.view(H_out, W_out)


class SegFormerEnvironmentRepresentationModel(AbstractModelClass):
    def __init__(self, seg_model_name: str, k: int, device: str):
        super().__init__(self._setup_model(seg_model_name), device)
        self.model.to(self.device)
        self.k = k

    def _setup_model(self, seg_model_name) -> torch.nn.Module:
        model = SEG_MODEL_ZOO[seg_model_name]
        self.processor = SegformerImageProcessorFast.from_pretrained(model)
        model = SegformerForSemanticSegmentation.from_pretrained(model)

        return model

    def run(self, input: torch.Tensor, **kwargs) -> ObjectRepData:
        batch_feature = self.preprocess(input, **kwargs)
        output = self.process(batch_feature, **kwargs)
        return self.postprocess(output, input=input, **kwargs)

    def preprocess(self, input: torch.Tensor, **kwargs) -> BatchFeature:
        inputs = self.processor(images=input, return_tensors="pt")
        inputs.to(self.device)
        return inputs

    def process(self, input: BatchFeature, **kwargs) -> SemanticSegmenterOutput:
        return self.model(**input)

    def postprocess(
        self,
        out: SemanticSegmenterOutput,
        epsilon: float = 0.5,
        include: list = [12, 20],  # person and car only
        **kwargs
    ) -> ObjectRepData:

        logits = out.logits

        input: torch.Tensor | None = kwargs.get("input", None)

        if input is None or logits is None:
            raise Exception("Fatal error on model run, no input frame provided.")

        # Resize output segmentation map
        target_size = (input.shape[0], input.shape[1])
        upsampled_logits = torch.nn.functional.interpolate(
            input=logits,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        # TODO: diff conf per class
        label_prob = torch.softmax(upsampled_logits, dim=1)[0].detach().cpu()
        (pred_label_prob, pred_label) = torch.max(label_prob, dim=0)
        pred_label[pred_label_prob < epsilon] = 0

        mask = torch.zeros((pred_label.shape[0], pred_label.shape[1]), dtype=torch.float16)
        exclude_mask = reduce(
            lambda x, y: torch.logical_and(x, (pred_label != y)),
            include,
            pred_label
        )
        include_mask = reduce(
            lambda x, y: torch.logical_or(x, (pred_label == y)),
            include[1:],
            pred_label == include[0]
        )
        mask[exclude_mask] = 0
        mask[include_mask] = pred_label[include_mask].to(torch.float16)
        blocked_mask = mode_pool2d(mask, kernel_size=self.k, stride=self.k)
        mask[include_mask] = 1

        object_coordinates = []

        (height, width) = blocked_mask.shape

        # Iterate through block mask and pull every associated object
        for w in range(1, width + 1):
            for h in range(1, height + 1):
                if blocked_mask[h-1][w-1] != 0:
                    x = self.k*w - 1
                    y = self.k*h - 1
                    # label = ADE_ID_TO_LABEL[str(int(blocked_mask[h-1][w-1].item()))]
                    object_coordinates.append(
                        ObjectXYCoordData(
                            label=int(blocked_mask[h-1][w-1].item()),
                            x=x,
                            y=y
                        )
                    )
                    # # Add a star or object onto the mask at (x, y)
                    # cv2.drawMarker(
                    #     mask.numpy(),  # Convert mask to numpy for cv2 operations
                    #     position=(x, y),
                    #     color=(1),  # Use a value of 1 for the mask
                    #     markerType=cv2.MARKER_STAR,
                    #     markerSize=10,
                    #     thickness=1,
                    #     line_type=cv2.LINE_AA
                    # )
                    # # Add text label near the object
                    # cv2.putText(
                    #     img=mask.numpy(),  # Convert mask to numpy for cv2 operations
                    #     text=label,
                    #     org=(x, y - 10),  # Position text slightly above the marker
                    #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    #     fontScale=0.5,
                    #     color=(1),  # Use a value of 1 for the mask
                    #     thickness=1,
                    #     lineType=cv2.LINE_AA
                    # )

        return ObjectRepData(object_coordinates=object_coordinates, mask=mask)

    def postprocess_to_image(
        self,
        out: SemanticSegmenterOutput,
        frame: np.ndarray,
        add_text_labels: bool = False,
        epsilon: float = 0.5
    ) -> np.ndarray:

        logits = out.logits

        if logits is None:
            raise Exception("Fatal error on model run, no logits provided.")

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

        # Create color mask and apply to original frame
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
                    if stats[i, cv2.CC_STAT_AREA] < 200:
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


class YoloEnvironmentRepresentationModel(AbstractModelClass):

    def __init__(self, model_name: str, device: str, retain_frames: int = 30):

        self.model: YOLO  # fix type errors
        super().__init__(self._setup_model(model_name), device)

        self.track_history = defaultdict(lambda: [])
        self.retain_frames = retain_frames

    def _setup_model(self, yolo_model_name) -> YOLO:
        return YOLO(YOLO_MODEL_ZOO[yolo_model_name])

    def run(self, input, **kwargs) -> ObjectRepData:
        out = self.process(input, **kwargs)
        return self.postprocess(out, input=input, **kwargs)

    def preprocess(self, input) -> ...:
        pass

    def process(self, input, **kwargs) -> Results:
        return self.model.track(input, persist=True, device=self.device)[0]

    def postprocess(self, out: Results, **kwargs) -> ObjectRepData:

        input: torch.Tensor | None = kwargs.get("input", None)
        show_det: bool = kwargs.get("show_det", False)

        if input is None:
            raise Exception("Fatal error on model run, no input frame provided.")

        mask = torch.zeros((input.shape[0], input.shape[1]), dtype=torch.uint8)

        if out.boxes and out.boxes.is_track:

            if show_det:
                frame = out.plot()
                cv2.imshow("track", frame)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    pass

            object_coordinates = []
            boxes = out.boxes.xywh.cpu() if out.boxes.xywh is torch.Tensor else out.boxes.xywh
            object_ids = out.boxes.id.int().cpu().tolist() if isinstance(out.boxes.id, torch.Tensor) else \
                (out.boxes.id.astype(int).tolist() if isinstance(out.boxes.id, np.ndarray) else None)
            labels = out.boxes.cls.int().cpu().tolist() if isinstance(out.boxes.cls, torch.Tensor) else \
                (out.boxes.cls.astype(int).tolist() if isinstance(out.boxes.cls, np.ndarray) else None)

            if object_ids is None or labels is None:
                raise Exception("Fatal error on model run, no labels or object id's found.")

            for box, object_id, label_id in zip(boxes, object_ids, labels):
                x, y, _, _ = box
                track = self.track_history[object_id]
                track.append((float(x), float(y)))

                if len(track) > self.retain_frames:  # retain track for only 30 frames
                    track.pop(0)

                object_coordinates.append(
                    ObjectXYCoordData(
                        object_id=object_id,
                        label=label_id,
                        x=float(x),
                        y=float(y)
                    )
                )

                mask[int(y)][int(x)] = 1  # add pixel to mask

            return ObjectRepData(object_coordinates, mask=mask)

        return ObjectRepData([], mask=mask)

    def postprocess_to_image(
        self,
        out: Results
    ) -> np.ndarray:

        # Get the boxes and track IDs
        if out.boxes and out.boxes.is_track:
            boxes = out.boxes.xywh.cpu() if out.boxes.xywh is torch.Tensor else out.boxes.xywh
            object_ids = out.boxes.id.int().cpu().tolist() if isinstance(out.boxes.id, torch.Tensor) else \
                (out.boxes.id.astype(int).tolist() if isinstance(out.boxes.id, np.ndarray) else None)

            if object_ids is None:
                raise Exception("Fatal error on model run, no labels or object id's found.")

            # Visualize the result on the frame
            frame = out.plot()

            # Plot the tracks
            for box, object_id in zip(boxes, object_ids):
                x, y, _, _ = box
                track = self.track_history[object_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 30 tracks for 30 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            return frame
        else:
            raise Exception("Error on inference")
