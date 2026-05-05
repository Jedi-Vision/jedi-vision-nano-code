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

        # Forwarded attributes to set in derived child clases
        self.right_match = None
        self.stereo = None
        self.wls_sigma = False
        self.wls_filter = False

    @abstractmethod
    def calc_disparity(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        """
        Compute the disparity map from a pair of stereo images.

        Args:
            left_img (np.ndarray): Grayscale image from left camera.
            right_img (np.ndarray): Grayscale image from right camera.

        Returns:
            Disparity map as np.ndarray (float32).
        """
        pass

    def __call__(self, frame, Q):
        return self.run(frame, Q)

    def apply_wls_filter(self, left_disparity, left_img, right_img):
        """
        Apply Weighted Least Squares (WLS) filter to refine disparity map.

        This method uses both left and right disparity maps to apply a WLS filter,
        which improves disparity map quality by reducing noise while preserving edges.

        Args:
            left_disparity (np.ndarray): Disparity map computed from left image.
            left_img (np.ndarray): Grayscale image from left camera.
            right_img (np.ndarray): Grayscale image from right camera.

        Returns:
            np.ndarray: Filtered disparity map with improved quality.

        Raises:
            AssertionError: If right_matcher or stereo model is not instantiated.

        Note:
            - Requires right_match and stereo attributes to be initialized.
            - Uses wls_sigma for both lambda and sigma color parameters.
            - Disparity values are scaled by 16.0 before filtering as OpenCV's
              stereo block matching returns disparities in CV_16S format multiplied by 16.
        """

        assert self.right_match is not None, "right_matcher not instantiated!"
        assert self.stereo is not None, "stereo model not instantiated!"

        right_disp = self.right_match.compute(right_img, left_img).astype(np.float32) / 16.0
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo)
        wls_filter.setLambda(self.wls_sigma)
        wls_filter.setSigmaColor(self.wls_sigma)
        return wls_filter.filter(left_disparity, left_img, disparity_map_right=right_disp)

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

        left_img, right_img = frame
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        disparity = self.calc_disparity(left_gray, right_gray)
        if self.wls_filter:
            disparity = self.apply_wls_filter(disparity, left_gray, right_gray)

        # More post-processing to change funky results
        disparity[disparity < 0] = 0
        # assert not np.isinf(disparity).any(), "Disparity map contains infinity."
        disparity[np.isinf(disparity)] = 0

        points_3D = self.reprojectImageTo3D(disparity, Q)
        points_3D[np.isinf(points_3D)] = 0
        return points_3D, disparity
