from typing import Literal
from numpy.typing import NDArray
from ..representation.data import Object2DCoordData, ObjectCoordData
import numpy as np


def sample(
    obj: Object2DCoordData,
    depth: NDArray,
    sample_method: Literal['gauss', 'center', 'min', 'max', 'mean', 'median'],
    **kwargs
):
    """
    Sample 3D coordinates from a depth map based on a 2D object bounding box.
    This function extracts depth information from a region of interest (ROI) defined by
    a 2D object's bounding box and computes 3D coordinates using the specified sampling method.
    Args:
        obj (Object2DCoordData): A 2D object containing bounding box coordinates (x1, y1, x2, y2),
            id, and label information.
        depth (NDArray): A 3D array representing the reprojection map with shape (height, width, 3),
            where the last dimension contains [x, y, z] coordinates.
        sample_method (Literal['gauss', 'center', 'min', 'max', 'mean', 'median']): The method
            used to sample coordinates from the ROI.
            - 'gauss': Gaussian-weighted averaging of valid depth values
            - 'center': Sample from the center of the ROI
            - 'min': Use minimum depth value
            - 'max': Use maximum depth value
            - 'mean': Use mean of depth values
            - 'median': Use median of depth values
        **kwargs: Additional keyword arguments.
            - 'depth_conv' (float): Conversion factor applied to depth values. Default is 0.001.
    Returns:
        ObjectCoordData: An object containing the sampled 3D coordinates with fields:
            - id: The object identifier from the input
            - label: The object label from the input
            - x: The sampled x-coordinate (scaled by depth_conv)
            - y: The sampled y-coordinate (scaled by depth_conv)
            - depth: The sampled depth/z-coordinate (scaled by depth_conv)
    Raises:
        NotImplementedError: If sample_method is not 'gauss' or is an unrecognized value.
    Notes:
        For the 'gauss' method, zero and non-finite depth values are excluded from calculations.
        If no valid depth values exist in the ROI, returns zeros for all coordinates.
        The Gaussian weights are computed based on distance from the bounding box center,
        with sigma set to 1/4 of the smallest ROI dimension (minimum 1.0).
    """

    match sample_method:
        # Gaussian weighted averaging of depth estimate based on bounding box
        # region of interest
        case "gauss":

            DEPTH_CONV = kwargs.get("depth_conv", 0.001)

            x1, y1, x2, y2 = map(int, (obj.x1, obj.y1, obj.x2, obj.y2))

            # Extract the region of interest from the reprojection map (depth)
            roi = depth[y1:y2, x1:x2]
            z_values = roi[..., 2]

            # Exclude zero or invalid values when computing depths
            valid_mask = (z_values > 0) & np.isfinite(z_values)

            # If there is a valid mask of non-zero depths for the given prediction
            # then we take a Gaussian-weighted average to derive the x and y
            # coordinates of the object, and get the minimum (non-zero) depth
            # value
            if valid_mask.any():
                min_z = np.min(z_values[valid_mask])

                h, w = roi.shape[:2]
                cx, cy = w / 2.0, h / 2.0
                yy, xx = np.mgrid[:h, :w]

                # Weights derived from Gaussian kernel based on distance from bounding box center
                sigma = max(min(h, w) / 4.0, 1.0)
                weights = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))

                x_values = roi[..., 0]
                y_values = roi[..., 1]

                avg_x = np.average(x_values[valid_mask], weights=weights[valid_mask])
                avg_y = np.average(y_values[valid_mask], weights=weights[valid_mask])
            else:
                min_z, avg_x, avg_y = 0.0, 0.0, 0.0

            return ObjectCoordData(
                id=obj.id,
                label=obj.label,
                x=float(avg_x * DEPTH_CONV),
                y=float(avg_y * DEPTH_CONV),
                depth=float(min_z * DEPTH_CONV)
            )

        case "center":
            DEPTH_CONV = kwargs.get("depth_conv", 0.001)
            x1, y1, x2, y2 = map(int, (obj.x1, obj.y1, obj.x2, obj.y2))
            roi = depth[y1:y2, x1:x2]

            cy, cx = roi.shape[0] // 2, roi.shape[1] // 2
            center_coord = roi[cy, cx]

            return ObjectCoordData(
                id=obj.id,
                label=obj.label,
                x=float(center_coord[0] * DEPTH_CONV),
                y=float(center_coord[1] * DEPTH_CONV),
                depth=float(center_coord[2] * DEPTH_CONV)
            )

        case "min":
            DEPTH_CONV = kwargs.get("depth_conv", 0.001)
            x1, y1, x2, y2 = map(int, (obj.x1, obj.y1, obj.x2, obj.y2))
            roi = depth[y1:y2, x1:x2]
            z_values = roi[..., 2]

            valid_mask = (z_values > 0) & np.isfinite(z_values)
            if valid_mask.any():
                masked_z = z_values.copy()
                masked_z[~valid_mask] = np.inf
                min_idx = np.unravel_index(np.argmin(masked_z), z_values.shape)
                coord = roi[min_idx]
            else:
                coord = np.array([0.0, 0.0, 0.0])

            return ObjectCoordData(
                id=obj.id,
                label=obj.label,
                x=float(coord[0] * DEPTH_CONV),
                y=float(coord[1] * DEPTH_CONV),
                depth=float(coord[2] * DEPTH_CONV)
            )

        case "max":
            DEPTH_CONV = kwargs.get("depth_conv", 0.001)
            x1, y1, x2, y2 = map(int, (obj.x1, obj.y1, obj.x2, obj.y2))
            roi = depth[y1:y2, x1:x2]
            z_values = roi[..., 2]

            valid_mask = (z_values > 0) & np.isfinite(z_values)
            if valid_mask.any():
                masked_z = z_values.copy()
                masked_z[~valid_mask] = -np.inf
                max_idx = np.unravel_index(np.argmax(masked_z), z_values.shape)
                coord = roi[max_idx]
            else:
                coord = np.array([0.0, 0.0, 0.0])

            return ObjectCoordData(
                id=obj.id,
                label=obj.label,
                x=float(coord[0] * DEPTH_CONV),
                y=float(coord[1] * DEPTH_CONV),
                depth=float(coord[2] * DEPTH_CONV)
            )

        case "mean":
            DEPTH_CONV = kwargs.get("depth_conv", 0.001)
            x1, y1, x2, y2 = map(int, (obj.x1, obj.y1, obj.x2, obj.y2))
            roi = depth[y1:y2, x1:x2]
            z_values = roi[..., 2]

            valid_mask = (z_values > 0) & np.isfinite(z_values)
            if valid_mask.any():
                avg_x = np.mean(roi[..., 0][valid_mask])
                avg_y = np.mean(roi[..., 1][valid_mask])
                avg_z = np.mean(z_values[valid_mask])
            else:
                avg_x, avg_y, avg_z = 0.0, 0.0, 0.0

            return ObjectCoordData(
                id=obj.id,
                label=obj.label,
                x=float(avg_x * DEPTH_CONV),
                y=float(avg_y * DEPTH_CONV),
                depth=float(avg_z * DEPTH_CONV)
            )

        case "median":
            DEPTH_CONV = kwargs.get("depth_conv", 0.001)
            x1, y1, x2, y2 = map(int, (obj.x1, obj.y1, obj.x2, obj.y2))
            roi = depth[y1:y2, x1:x2]
            z_values = roi[..., 2]

            valid_mask = (z_values > 0) & np.isfinite(z_values)
            if valid_mask.any():
                med_x = np.median(roi[..., 0][valid_mask])
                med_y = np.median(roi[..., 1][valid_mask])
                med_z = np.median(z_values[valid_mask])
            else:
                med_x, med_y, med_z = 0.0, 0.0, 0.0

            return ObjectCoordData(
                id=obj.id,
                label=obj.label,
                x=float(med_x * DEPTH_CONV),
                y=float(med_y * DEPTH_CONV),
                depth=float(med_z * DEPTH_CONV)
            )

        case _:
            raise NotImplementedError
