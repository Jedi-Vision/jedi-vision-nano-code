import cv2
import numpy as np
from .base import StereoDepthEstimatorBase


class BMStereoDepthEstimator(StereoDepthEstimatorBase):
    """
    Stereo depth estimator (along with 3D points) using block matching.

    Supports both Semi-Global Block Matching (SGBM) and simple Block Matching (BM).
    SGBM is more accurate but slower; BM is faster but less robust.

    Args:
        num_disparities: Maximum disparity minus minimum disparity. The value is always greater
            than zero. Must be divisible by 16. Determines the resolution of your stereo / depth map.
        block_size: Matched block size. It must be an odd number >=1. Normally, it should be
            somewhere in the 3 to 11 range. For StereoBM, must be odd and >= 5.
        min_disparity: Minimum possible disparity value. Normally, it is zero but sometimes
            rectification algorithms can shift images, so this parameter needs to be adjusted
            accordingly.
        pre_filter_cap: Truncation value for the prefiltered image pixels. The algorithm first
            computes x-derivative at each pixel and clips its value by [-preFilterCap,
            preFilterCap] interval. The result values are passed to the Birchfield-Tomasi pixel
            cost function.
        uniqueness_ratio: Margin in percentage by which the best (minimum) computed cost function
            value should "win" the second best value to consider the found match correct.
            Normally, a value within the 5-15 range is good enough.
        speckle_window_size: Maximum size of smooth disparity regions to consider their noise
            speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it
            somewhere in the 50-200 range.
        speckle_range: Maximum disparity variation within each connected component. If you do
            speckle filtering, set the parameter to a positive value, it will be implicitly
            multiplied by 16. Normally, 1 or 2 is good enough.
        max_depth: Maximum depth value (in mm) to keep. Points beyond this are zeroed out.
    """
    def __init__(
        self,
        num_disparities: int = 16 * 6,
        block_size: int = 11,
        min_disparity: int = 0,
        pre_filter_cap: int = 1,
        uniqueness_ratio: int = 5,
        speckle_window_size: int = 100,
        speckle_range: int = 2,
        max_depth: float = 10000.,
    ):
        super().__init__(max_depth)
        assert num_disparities > 0 and num_disparities % 16 == 0, \
            "num_disparities must be > 0 and divisible by 16"
        assert block_size >= 5 and block_size % 2 == 1, \
            "block_size must be odd and >= 5 for StereoBM"
        self.stereo = cv2.StereoBM.create(
            numDisparities=num_disparities,
            blockSize=block_size,
        )
        self.stereo.setMinDisparity(min_disparity)
        self.stereo.setPreFilterCap(pre_filter_cap)
        self.stereo.setUniquenessRatio(uniqueness_ratio)
        self.stereo.setSpeckleWindowSize(speckle_window_size)
        self.stereo.setSpeckleRange(speckle_range)

    def calc_disparity(self, frame: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Compute the disparity map from a pair of stereo images.

        Args:
            frame: Tuple of (left_img, right_img), both as np.ndarray.

        Returns:
            Disparity map as np.ndarray (float32).
        """
        left_img, right_img = frame
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        disparity[disparity < 0] = 0
        assert not np.isinf(disparity).any(), "Disparity map contains infinity."
        return disparity


class SGBMStereoDepthEstimator(BMStereoDepthEstimator):
    """
    Stereo depth estimator using Semi-Global Block Matching (SGBM).

    Args:
        num_disparities: Maximum disparity minus minimum disparity. The value is always greater
            than zero. Must be divisible by 16. Determines the resolution of your stereo / depth map.
        block_size: Matched block size. It must be an odd number >=1. Normally, it should be
            somewhere in the 3 to 11 range.
        min_disparity: Minimum possible disparity value. Normally, it is zero but sometimes
            rectification algorithms can shift images, so this parameter needs to be adjusted
            accordingly.
        P1: The first parameter controlling the disparity smoothness. P1 is the
            penalty on the disparity change by plus or minus 1 between neighbor pixels.
            Defaults to 8 * 1 * block_size^2.
        P2: The second parameter controlling the disparity smoothness. The larger
            the values are, the smoother the disparity is. P2 is the penalty on the disparity
            change by more than 1 between neighbor pixels. The algorithm requires P2 > P1.
            Defaults to 32 * 1 * block_size^2.
        disp12_max_diff: Maximum allowed difference (in integer pixel units) in
            the left-right disparity check. Set it to a non-positive value to disable the check.
        pre_filter_cap: Truncation value for the prefiltered image pixels. The algorithm first
            computes x-derivative at each pixel and clips its value by [-preFilterCap,
            preFilterCap] interval. The result values are passed to the Birchfield-Tomasi pixel
            cost function.
        uniqueness_ratio: Margin in percentage by which the best (minimum) computed cost function
            value should "win" the second best value to consider the found match correct.
            Normally, a value within the 5-15 range is good enough.
        speckle_window_size: Maximum size of smooth disparity regions to consider their noise
            speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it
            somewhere in the 50-200 range.
        speckle_range: Maximum disparity variation within each connected component. If you do
            speckle filtering, set the parameter to a positive value, it will be implicitly
            multiplied by 16. Normally, 1 or 2 is good enough.
        max_depth: Maximum depth value (in mm) to keep. Points beyond this are zeroed out.
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
        max_depth: float = 10000.,
    ):
        self.max_depth = max_depth
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
        self.stereo = cv2.StereoSGBM.create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=P1,
            P2=P2,
            disp12MaxDiff=disp12_max_diff,
            preFilterCap=pre_filter_cap,
            uniquenessRatio=uniqueness_ratio,
            speckleWindowSize=speckle_window_size,
            speckleRange=speckle_range,
        )
