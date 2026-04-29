import numpy as np
import vpi
from .base import StereoDepthEstimatorBase


class VPIDepthEstimator(StereoDepthEstimatorBase):
    """
    Drop-in replacement depth estimator using NVIDIA VPI stereo disparity.

    Assumes:
    - Input images are ALREADY rectified
    - Input images are numpy arrays (H, W) or (H, W, 3)
    """

    def __init__(self, config):
        self.max_disparity = config.get("num_disparities", 16*6)

        # Output control
        self.return_depth = config.get("return_depth", True)

        # Backend selection
        backend_str = config.get("backend", "cuda").upper()
        self.backend = getattr(vpi.Backend, backend_str)

        # Performance: persist backend context
        self._backend_ctx = vpi.Backend(self.backend)

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
                maxdisp=self.max_disparity
            )

            disp_np = disparity.cpu().astype(np.float64) * (1. / 16.)

        return disp_np

    def _to_gray(self, img):
        return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
