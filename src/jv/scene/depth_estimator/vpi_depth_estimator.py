import numpy as np
import vpi

class VPIDepthEstimator:
    def __init__(
        self,
        baseline,           # meters
        focal_length_px,    # pixels
        max_disparity=64,
        backend=vpi.Backend.CUDA
    ):
        self.baseline = baseline
        self.focal_length = focal_length_px
        self.max_disparity = max_disparity
        self.backend = backend

    def estimate(self, left_img, right_img):
        """
        left_img, right_img: numpy uint8 grayscale or RGB
        returns: depth map (float32, meters)
        """

        # Ensure grayscale
        if left_img.ndim == 3:
            left_img = self._to_gray(left_img)
        if right_img.ndim == 3:
            right_img = self._to_gray(right_img)

        with vpi.Backend(self.backend):
            # Wrap numpy → VPI
            left = vpi.asimage(left_img)
            right = vpi.asimage(right_img)

            # Convert to required format
            left = left.convert(vpi.Format.U8)
            right = right.convert(vpi.Format.U8)

            # Stereo disparity
            disparity = vpi.stereodisp(
                left,
                right,
                window=5,
                maxdisp=self.max_disparity
            )

            # Convert back to numpy
            disp_np = disparity.cpu()

        # Convert disparity → depth
        depth = self._disparity_to_depth(disp_np)

        return depth

    def _disparity_to_depth(self, disparity):
        disparity = disparity.astype(np.float32)

        # Avoid divide by zero
        disparity[disparity == 0] = np.nan

        depth = (self.focal_length * self.baseline) / disparity
        return depth

    def _to_gray(self, img):
        return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
