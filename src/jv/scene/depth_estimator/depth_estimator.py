import cv2
import numpy as np


class DepthEstimator:
    def __init__(
        self,
        num_disparities: int = 16 * 6,
        block_size: int = 11,
        min_disparity: int = 0
    ):
        """
        Stereo depth estimator using block matching.

        Args:
            num_disparities: Must be divisible by 16
            block_size: Odd number >= 5
            min_disparity: Usually 0
        """

        self.stereo = cv2.StereoBM_create(
            numDisparities=num_disparities,
            blockSize=block_size
        )

        self.min_disparity = min_disparity
        self.baseline = 0.06
        self.focal_length = 700

    def __call__(self, frame):
        """
        Computes depth from binocular frames.

        Args:
            frame: ((left_img, right_img), frame_number, timestamp)

        Returns:
            depth map (H x W numpy array)
        """

        left_img, right_img = frame

        # Convert to grayscale
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        # Compute disparity
        disparity = self.stereo.compute(left_gray, right_gray)

        # Normalize disparity
        disparity = disparity.astype(np.float32) / 16.0

        # Replace invalid disparities
        disparity[disparity <= 0] = 0.1

        depth = (self.focal_length * self.baseline) / disparity

        return depth