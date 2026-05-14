import cv2
import numpy as np


class Rectifier:

    def __init__(
        self,
        calibration_data,
        img_size,
        cap_size,
        split_bino: bool = False
    ):

        self._init_rectify(
            *calibration_data,
            img_size=img_size,
            cap_size=cap_size,
            split_bino=split_bino
        )

    def _init_rectify(self, mtx1, dist1, mtx2, dist2, R, T, img_size, cap_size, split_bino: bool = False):
        """
        Initializes stereo rectification for a pair of cameras.

        Computes the rectification transforms and projection matrices for two cameras using their
        intrinsic and extrinsic parameters. Also generates the undistort and rectify maps for both
        left and right images.

        Args:
            mtx1 (np.ndarray): Intrinsic matrix of the left camera.
            dist1 (np.ndarray): Distortion coefficients of the left camera.
            mtx2 (np.ndarray): Intrinsic matrix of the right camera.
            dist2 (np.ndarray): Distortion coefficients of the right camera.
            R (np.ndarray): Rotation matrix between the coordinate systems of the two cameras.
            T (np.ndarray): Translation vector between the coordinate systems of the two cameras.
            img_size (tuple): Size of the scaled displayed images as (width, height).
            cap_size (tuple): Size of the captured images as (width, height).

        Sets:
            self.R1 (np.ndarray): Rectification transform for the left camera.
            self.R2 (np.ndarray): Rectification transform for the right camera.
            self.P1 (np.ndarray): Projection matrix in the new (rectified) coordinate systems for the left camera.
            self.P2 (np.ndarray): Projection matrix in the new (rectified) coordinate systems for the right camera.
            self.Q (np.ndarray): Disparity-to-depth mapping matrix.
            self.roi1 (tuple): Region of interest in the rectified left image.
            self.roi2 (tuple): Region of interest in the rectified right image.
            self.map1_left (np.ndarray): First output map for cv2.remap for the left image.
            self.map2_left (np.ndarray): Second output map for cv2.remap for the left image.
            self.map1_right (np.ndarray): First output map for cv2.remap for the right image.
            self.map2_right (np.ndarray): Second output map for cv2.remap for the right image.
        """

        # Check if downscaled image and get new instrinsic camera parameters
        # with the scaling
        if img_size != cap_size:

            def scale_camera_matrix(mtx, scale):
                mtx_scaled = mtx.copy()
                mtx_scaled[0, 0] *= scale  # fx
                mtx_scaled[1, 1] *= scale  # fy
                mtx_scaled[0, 2] *= scale  # cx
                mtx_scaled[1, 2] *= scale  # cy
                return mtx_scaled

            scale_width = img_size[0] / cap_size[0]
            scale_height = img_size[1] / cap_size[1]
            assert scale_width == scale_height, "Width and height must be scaled equally."
            scale = scale_width

            [mtx1, mtx2] = map(lambda x: scale_camera_matrix(x, scale), [mtx1, mtx2])

        # If binocular image is single and needs to be split, the passed in img_size from
        # the camera is 2x the width of the actual left and right images, thus resize
        if split_bino:
            img_size = (img_size[0] // 2, img_size[1])

        # Compute rectification transforms for stereo cameras
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx1, dist1, mtx2, dist2, img_size,
            R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        self.R1 = R1
        self.R2 = R2
        self.P1 = P1
        self.P2 = P2
        self.Q = Q
        self.roi1 = roi1
        self.roi2 = roi2

        # Load rectification maps
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            mtx1, dist1, R1, P1, img_size, cv2.CV_16SC2
        )
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            mtx2, dist2, R2, P2, img_size, cv2.CV_16SC2
        )

    def rectify(self, frame: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Undistorts and rectifies an image pair (frame) according to rectification maps"""
        left_img, right_img = frame
        rectified_left = cv2.remap(left_img, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(right_img, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        return (rectified_left, rectified_right)
