import numpy as np
import vpi
from .base import StereoDepthEstimatorBase
import cv2


class VPIDepthEstimator(StereoDepthEstimatorBase):
    """
    Drop-in replacement depth estimator using NVIDIA VPI stereo disparity.

    Assumes:
    - Input images are ALREADY rectified
    - Input images are numpy arrays (H, W) or (H, W, 3)
    """

    def __init__(
        self,
        num_disparities: int = 16 * 6,
        block_size: int = 3,
        min_disparity: int = 0,
        P1: int | None = None,
        P2: int | None = None,
        disp12_max_diff: int = 1,
        pre_filter_cap: int = 0,
        uniqueness_ratio: int = 5,
        speckle_window_size: int = 100,
        speckle_range: int = 2,
        max_depth: float = 10000.
    ):
        self.max_depth = max_depth
        self.num_disparities = num_disparities
        assert num_disparities > 0 and num_disparities % 16 == 0, \
            "num_disparities must be > 0 and divisible by 16"
        assert block_size > 0 and block_size % 2 == 1, \
            "block_size must be an odd number >= 1"
        IMG_CHANNELS = 1
        # P1: penalty on disparity change by +/- 1 between neighbor pixels
        if P1 is None:
            P1 = 8 * IMG_CHANNELS * block_size**2
        # P2: penalty on disparity change by >1 between neighbor pixels, must be > P1
        if P2 is None:
            P2 = 32 * IMG_CHANNELS * block_size**2
        assert P2 > P1, "P2 must be greater than P1"

        # Performance: persist backend context
        self._backend_ctx = vpi.Backend(vpi.Backend.CUDA)

    def calc_disparity(self, left_img, right_img):
        """
        Main entry point (matches repo convention)

        Args:
            left_img, right_img: np.ndarray

        Returns:
            np.ndarray (depth in meters OR disparity)
        """

        # Convert to grayscale if needed
        if left_img.ndim == 3:
            left_img = self._to_gray(left_img)
        if right_img.ndim == 3:
            right_img = self._to_gray(right_img)

        with self._backend_ctx:
            left = vpi.asimage(left_img)
            right = vpi.asimage(right_img)

            # Ensure correct format
            left = left.convert(vpi.Format.U8)
            right = right.convert(vpi.Format.U8)

            disparity = vpi.stereodisp(
                left,
                right,
                window=5,
                maxdisp=self.num_disparities
            )

            disp_np = disparity.cpu().astype(np.float32) * (1. / 16.)

        return disp_np

    def _to_gray(self, img):
        return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
