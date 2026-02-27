import cv2
import numpy as np


class StereoDepthEstimator:
    def __init__(
        self,
        num_disparities: int = 16 * 6,
        block_size: int = 11,
        min_disparity: float = 0.,
    ):
        """
        Stereo depth estimator (along with 3D points) using block matching.

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

    def __call__(self, frame, Q):
        return self.run(frame, Q)

    def calc_disparity(self, frame: tuple[np.ndarray, np.ndarray]) -> np.ndarray:

        left_img, right_img = frame

        # Convert to grayscale
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        # Compute disparity
        disparity = self.stereo.compute(left_gray, right_gray)

        # Disparity is scaled by 16 (fixed-point) in OpenCV StereoBM, so unscale it
        disparity = disparity.astype(np.float32) / 16.0

        # Replace invalid disparities
        disparity[disparity <= self.min_disparity] = self.min_disparity

        assert not np.isinf(disparity).any(), "Disparity map contains infinity."

        return disparity

    def run(self, frame: tuple[np.ndarray, np.ndarray], Q: np.ndarray):
        """
        Computes real world (X,Y,depth) coordinates from binocular frames, each
        coordinate corresponds to a (h,w) coordinate pair in left frame.

        Args:
            frame: (left_img, right_img)
            Q: Disparity-to-depth mapping matrix from rectification.

        Returns:
            3D point array (H x W x (X, Y, Z) numpy array)
        """

        disparity = self.calc_disparity(frame)
        points_3D = cv2.reprojectImageTo3D(disparity, Q)

        return points_3D
