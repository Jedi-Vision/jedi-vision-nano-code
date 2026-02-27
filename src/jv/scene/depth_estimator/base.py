from abc import ABC, abstractmethod
import numpy as np
import cv2


class StereoDepthEstimatorBase(ABC):
    def __init__(self, max_depth: float = 10000.):
        """
        Base class for stereo depth estimators.
        Args:
            max_depth: Maximum depth value (in mm) to keep. Points beyond this are zeroed out.
        """
        self.max_depth = max_depth

    @abstractmethod
    def calc_disparity(self, frame: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Compute the disparity map from a pair of stereo images.

        Args:
            frame: Tuple of (left_img, right_img), both as np.ndarray.

        Returns:
            Disparity map as np.ndarray (float32).
        """
        pass

    def __call__(self, frame, Q):
        return self.run(frame, Q)

    def myReprojectImageTo3D(self, disparity: np.ndarray, Q: np.ndarray):
        """
        Projects a disparity map to 3D space using a reprojection matrix, with additional handling for invalid
        or extreme depth values.

        Args:
            disparity (np.ndarray): A 2D array representing the disparity map.
            Q (np.ndarray): A 4x4 reprojection matrix used to convert disparity values to 3D coordinates.

        Returns:
            np.ndarray: A 3D array of shape (H, W, 3) containing the reprojected 3D points in float32 format.

        Notes:
            - Depth values are clipped to a maximum specified by self.max_depth.
            - Points with invalid disparity (<= 0) are set to zero.
            - Division by very small W values is avoided to prevent numerical instability.
            - **About 5-7x slower than using cv2.reprojectImageTo3D3**
        """
        h, w = disparity.shape
        i, j = np.indices((h, w))
        homog = np.stack([j, i, disparity, np.ones_like(disparity)], axis=-1)
        homog = homog.reshape(-1, 4).T
        points = Q @ homog
        W = points[3]
        valid = disparity.flatten() > 0
        points[:, valid] /= W[valid]
        points[:, ~valid] = 0
        Z = points[2]
        points[:, Z > self.max_depth] = self.max_depth
        points = points[:3].T.reshape(h, w, 3)
        return points.astype(np.float32)

    def reprojectImageTo3D(self, disparity: np.ndarray, Q: np.ndarray):
        """
        Projects a disparity map to 3D space using a reprojection matrix, with additional handling for invalid
        or extreme depth values.

        Args:
            disparity (np.ndarray): A 2D array representing the disparity map.
            Q (np.ndarray): A 4x4 reprojection matrix used to convert disparity values to 3D coordinates.

        Returns:
            np.ndarray: A 3D array of shape (H, W, 3) containing the reprojected 3D points in float32 format.

        Notes:
            - Depth values are clipped to a maximum specified by self.max_depth.
            - Points with invalid disparity (<= 0) are set to zero.
        """
        points = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
        points[points[:, :, 2] > self.max_depth] = 0
        return points.astype(np.float32)

    def run(self, frame: tuple[np.ndarray, np.ndarray], Q: np.ndarray):
        disparity = self.calc_disparity(frame)
        points_3D = self.reprojectImageTo3D(disparity, Q)
        points_3D[np.isinf(points_3D)] = 0
        return points_3D, disparity
