from jv.audio import ObjectBuffer
from jv.representation import YoloObjectRepresentationModel
from jv.scene import VideoDepthAnything, MODEL_CONFIGS
from jv.scene.depth_estimator import SGBMStereoDepthEstimator, BMStereoDepthEstimator
try:
    from jv.scene.depth_estimator import VPIDepthEstimator
except ImportError:
    print("VPI is not installed, skipping import...")
from jv.scene import sample, kalman
from jv.rectification import Rectifier
from jv.camera import FrameBuffer
from jv.representation.data import ObjectRepData, ObjectCoordData, Object2DCoordData
from typing import Literal
import torch
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


from jv.management import SystemManagement, log_block
sys_mgmt = SystemManagement()


DEPTH_CONV = 0.001  # 1mm per 0.001 meter for stereo depth conversion


class Driver:

    def __init__(
        self,
        device: Literal["cpu", "mps", "cuda"],
        output_to: Literal["socket", "file", "none"] = "socket",
        serial_type: Literal["struct", "protobuf"] = "struct",
        object_model_name: str = "yolo11",
        det_whitelist: set = {0},
        vda_model_name: str = "vits",
        chkpts_folder: str = "./checkpoints",
        retain_frames: int = 30,
        object_buffer_size: int = 0,
        frame_buffer_size: int = 0,
        warmup_frames: int = 30,
        camera_index: str | int = 0,
        frame_skip: int = 0,
        frame_rate: int = 30,
        show_det: bool = False,
        binocular: bool = True,
        use_gstreamer: bool = True,
        depth: bool = True,
        metric: bool = False,
        multi_object: bool = False,
        use_kalman: bool = False,
        gstreamer_kwargs: dict = {},
        object_kwargs: dict = {},
        depth_kwargs: dict = {},
    ) -> None:
        """
        Initializes the driver with specified configuration for device, output, models, buffers, and video processing.

        Args:
            device (Literal["cpu", "mps", "cuda"]): Device to run models on.
            output_to (Literal["socket", "file", "none"]): Output destination for object buffer.
            object_model_name (str, optional): Name of the object detection model. Defaults to "yolo11".
            det_whitelist (set, optional): Object detection label whitelist.
            vda_model_name (str, optional): Name of the video depth estimation model. Defaults to "vits".
            chkpts_folder (str, optional): Path to the folder containing model checkpoints. Defaults to "./checkpoints".
            retain_frames (int, optional): Number of frames to retain in environment model. Defaults to 30.
            object_buffer_size (int, optional): Size of the object buffer. Defaults to 0.
            frame_buffer_size (int, optional): Size of the frame buffer. Defaults to 0.
            warmup_frames (int, optional): Number of warmup frames for frame buffer. Defaults to 30.
            camera_index (str | int, optional): Index or identifier for the camera source if monocular. Defaults to 0.
            binocular (bool, optional): Indicates whether camera setup is binocular. Uses GStreamer pipeline to access
                cameras.
            frame_skip (int, optional): Number of frames to skip between processing. Defaults to 0.
            frame_rate (int, optional): Frame rate for video capture. Defaults to 30.
            show_det (bool, optional): Whether to display detection results. Defaults to False.
            depth (bool, optional): Whether to enable depth estimation. Defaults to True.
            metric (bool, optional): Whether to use metric depth estimation model. Defaults to False.
            multi_object (bool, optional): Whether to allow detection of multiple objects. If disabled the closest
                object will be sent.

        Kwargs:
            gstreamer_kwargs (dict, optional): GStreamer kwargs. See buffer.py for more information.
            object_kwargs (dict, optional): Object detection kwargs. See object.py for more information.
            depth_kwargs (dict, optional): Depth kwargs. See video_depth_anything_code and depth_estimator
                for more information.
        """

        # Type coercion
        if isinstance(det_whitelist, list):
            det_whitelist = set(det_whitelist)
        self.det_whitelist = det_whitelist

        self.frame_buffer = FrameBuffer(
            size=frame_buffer_size,
            camera_index=camera_index,
            left_sensor_id=depth_kwargs.get("left_sensor_id", 0),
            right_sensor_id=depth_kwargs.get("right_sensor_id", None),
            warmup_frames=warmup_frames,
            frame_skip=frame_skip,
            frame_rate=frame_rate,
            binocular=binocular,
            use_gstreamer=use_gstreamer,
            gstreamer_kwargs=gstreamer_kwargs
        )
        if depth_kwargs.get("right_sensor_id", None) is None:
            self.split_bino = True
        else:
            self.split_bino = False

        self.env_model = YoloObjectRepresentationModel(
            model_name=object_model_name,
            device=device,
            retain_frames=retain_frames,
        )

        self.object_buffer = ObjectBuffer(
            size=object_buffer_size,
            output_to=output_to,
            serial_type=serial_type
        )

        self.depth = depth
        self.binocular = binocular
        self.multi_object = multi_object
        self.use_kalman = use_kalman

        # Binocular depth
        if self.binocular:
            def load_calibration_data(filename):
                fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
                mtx1 = fs.getNode("mtx1").mat()
                dist1 = fs.getNode("dist1").mat()
                mtx2 = fs.getNode("mtx2").mat()
                dist2 = fs.getNode("dist2").mat()
                R = fs.getNode("R").mat()
                T = fs.getNode("T").mat()
                fs.release()
                return mtx1, dist1, mtx2, dist2, R, T
            match depth_kwargs.get('method', "sgbm"):
                case "sgbm":
                    self.depth_estimator = SGBMStereoDepthEstimator(
                        num_disparities=depth_kwargs.get('num_disparities', 16*6),
                        block_size=depth_kwargs.get('block_size', 5),
                        min_disparity=depth_kwargs.get('min_disparity', 0),
                        P1=depth_kwargs.get('P1', None),
                        P2=depth_kwargs.get('P2', None),
                        disp12_max_diff=depth_kwargs.get('disp12_max_diff', 1),
                        pre_filter_cap=depth_kwargs.get('pre_filter_cap', 0),
                        uniqueness_ratio=depth_kwargs.get('uniqueness_ratio', 5),
                        speckle_window_size=depth_kwargs.get('speckle_window_size', 100),
                        speckle_range=depth_kwargs.get('speckle_range', 32),
                        max_depth=depth_kwargs.get('max_depth', 10000),
                        wls_filter=depth_kwargs.get('wls_filter', False),
                        wls_lambda=depth_kwargs.get('wls_lambda', 8000),
                        wls_sigma=depth_kwargs.get('wls_sigma', 1.5),
                        med_filter=depth_kwargs.get('med_filter', False),
                        ksize=depth_kwargs.get('ksize', 3),
                    )
                case "bm":
                    self.depth_estimator = BMStereoDepthEstimator(
                        num_disparities=depth_kwargs.get('num_disparities', 16*6),
                        block_size=depth_kwargs.get('block_size', 5),
                        min_disparity=depth_kwargs.get('min_disparity', 0),
                        pre_filter_cap=depth_kwargs.get('pre_filter_cap', 1) if
                        depth_kwargs.get('pre_filter_cap', 1) > 0 else 1,
                        uniqueness_ratio=depth_kwargs.get('uniqueness_ratio', 5),
                        speckle_window_size=depth_kwargs.get('speckle_window_size', 100),
                        speckle_range=depth_kwargs.get('speckle_range', 32),
                        max_depth=depth_kwargs.get('max_depth', 10000),
                        wls_filter=depth_kwargs.get('wls_filter', False),
                        wls_lambda=depth_kwargs.get('wls_lambda', 8000),
                        wls_sigma=depth_kwargs.get('wls_sigma', 1.5),
                        med_filter=depth_kwargs.get('med_filter', False),
                        ksize=depth_kwargs.get('ksize', 3),
                    )
                case "vpi":
                    self.depth_estimator = VPIDepthEstimator(
                        num_disparities=depth_kwargs.get('num_disparities', 16*6),
                        block_size=depth_kwargs.get('block_size', 5),
                        min_disparity=depth_kwargs.get('min_disparity', 0),
                        pre_filter_cap=depth_kwargs.get('pre_filter_cap', 1) if
                        depth_kwargs.get('pre_filter_cap', 1) > 0 else 1,
                        uniqueness_ratio=depth_kwargs.get('uniqueness_ratio', 5),
                        speckle_window_size=depth_kwargs.get('speckle_window_size', 100),
                        speckle_range=depth_kwargs.get('speckle_range', 32),
                        max_depth=depth_kwargs.get('max_depth', 10000),
                    )

            # Handle binocular split logic for ensuring that capture/display width is divided by two
            # for the individual frames
            self.rectifier = Rectifier(
                calibration_data=load_calibration_data(depth_kwargs.get("calibration_data", "camera_calibration.yaml")),
                img_size=(gstreamer_kwargs['display_width'] // 2 if self.split_bino is False
                          else gstreamer_kwargs['display_width'], gstreamer_kwargs['display_height']),
                cap_size=(gstreamer_kwargs['capture_width'] // 2 if self.split_bino is False
                          else gstreamer_kwargs['capture_width'], gstreamer_kwargs['capture_height']),
                split_bino=self.split_bino
            )
        else:
            self.scene_model = VideoDepthAnything(**MODEL_CONFIGS[vda_model_name])
            checkpoint_name = 'metric_video_depth_anything' if metric else 'video_depth_anything'
            self.scene_model.load_state_dict(
                torch.load(
                    f'{chkpts_folder}/{checkpoint_name}_{vda_model_name}.pth',
                    map_location='cpu',
                    weights_only=True
                ),
                strict=True
            )
            self.scene_model = self.scene_model.to(torch.device(device)).eval()

        self.device = device
        self.show_det = show_det

        if self.show_det:  # check if display and if not set to false
            try:
                cv2.imshow("test", np.zeros((1, 1, 3), dtype=np.uint8))
                cv2.destroyWindow("test")
            except cv2.error:
                self.show_det = False

        self.object_kwargs = object_kwargs
        self.depth_kwargs = depth_kwargs

        self.kalman_states: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # Store kalman filter state

        # Log system configuration at startup
        sys_mgmt.updateSetting("device", device)
        sys_mgmt.updateSetting("frame_rate", frame_rate)
        sys_mgmt.updateSetting("depth_enabled", depth)
        sys_mgmt.logMetric("driver.initialized", True)

    @log_block("model_inference")
    def model_run(self, frame, frame_number, timestamp_ms):
        # Log frame processing
        sys_mgmt.logMetric("frame.number", frame_number)
        sys_mgmt.logMetric("frame.timestamp_ms", timestamp_ms)

        # If we are using binocular images for depth, we need to ensure that detection
        # is done on rectified image pairs, so that we can then map the depth calculated
        # for rectified images to the corresponding objects given the rectified (x,y)
        # coordinates

        if self.binocular:
            frame = self.rectifier.rectify(frame)

        objects = self.env_model.run(
            frame[0],
            show_det=self.show_det,
            det_whitelist=self.det_whitelist,
            **self.object_kwargs
        )

        # Log object detection results
        sys_mgmt.logMetric("objects.detected_count", len(objects))

        if self.depth:
            if self.binocular:
                # If we are using binocular depth estimation, then we need to convert the
                # (x, y, d) triplet into real world (X, Y, Z) coordinates according to the
                # 'Q' matrix given by rectifification.
                depth, disparity = self.depth_estimator.run(
                    frame,
                    self.rectifier.Q
                )
                assert not torch.isinf(torch.tensor(depth)).any(), "Infinity found in depth map."

                # For each object from the 2d (x,y) coordinate we get the 3d (X,Y,Z)
                # convert to meters as well
                def two_to_three(obj: Object2DCoordData):  # type: ignore

                    object = sample(
                        obj,
                        depth,
                        sample_method=self.depth_kwargs.get("sample_method", "gauss"),
                        depth_conv=DEPTH_CONV
                    )

                    if self.use_kalman:
                        object, state = kalman(object, self.kalman_states.get(object.id))
                        self.kalman_states[object.id] = state

                    return object

            else:  # Use monocular depth model
                frame = cv2.cvtColor(frame[0], cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for VDA
                depth = self.scene_model.infer_video_depth_one(
                    frame,
                    input_size=518,
                    device=self.device,
                    **self.depth_kwargs
                )  # fp32=False causes a black output and NaN values in model

                # Add depth information to objects
                def two_to_three(obj: Object2DCoordData):
                    x, y = (int((obj.x1 + obj.x2) / 2), int((obj.y1 + obj.y2) / 2))

                    return ObjectCoordData(
                        id=obj.id,
                        label=obj.label,
                        x=float(x),
                        y=float(y),
                        depth=float(depth[y][x])
                    )

            objects = list(map(two_to_three, objects))

            # If not multi_object, ensure that only closest object is detected
            if objects and not self.multi_object:
                objects = [min(objects, key=lambda obj: (not np.isfinite(obj.depth), obj.depth))]
                # Should only be one object, set id manually to 1
                objects[0].id = 1
                print(f"Object.x: {objects[0].x}, Object.y: {objects[0].y}, Object.depth: {objects[0].depth}, Object.id: {objects[0].id}")

            # Log depth statistics
            sys_mgmt.logMetric("depth.min", depth.min().item())  # TODO Review these metrics
            sys_mgmt.logMetric("depth.max", depth.max().item())
            sys_mgmt.logMetric("depth.mean", depth.mean().item())

            # Visualize depth map if wanted
            if self.show_det:
                if self.binocular:
                    # Visualize binocular depth map
                    depth_map = depth[..., 2]
                    norm_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore
                    color_depth = cv2.applyColorMap(norm_depth.astype('uint8'), cv2.COLORMAP_JET)
                    cv2.imshow("binocular_depth", color_depth)
                    cv2.waitKey(1)
                    # Visualize disparity map
                    norm_disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore
                    color_disparity = cv2.applyColorMap(norm_disparity.astype('uint8'), cv2.COLORMAP_JET)
                    cv2.imshow("binocular_disparity", color_disparity)
                    cv2.waitKey(1)
                    # visualize_results(frame[0], frame[1], disparity, depth)
                else:
                    colormap = self.scene_model.colormap
                    # Normalize
                    color_depth = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8)*255).astype(np.uint8)
                    color_depth = colormap[color_depth]
                    cv2.imshow("msg.mask", color_depth)
                    cv2.waitKey(1)
        else:
            # Convert Object2DCoordData to ObjectCoordData with dummy depth if needed
            objects = [
                ObjectCoordData(
                    id=obj.id,
                    label=obj.label,
                    x=(obj.x1 + obj.x2) / 2,
                    y=(obj.y1 + obj.y2) / 2,
                    depth=0.0
                ) for obj in objects
            ]

        return ObjectRepData(
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
            objects=objects
        )

    @log_block("driver_main_loop")
    def run(self):

        print("Starting...")

        if self.object_buffer is not None:
            self.object_buffer.start()
        self.frame_buffer.start()

        frame_count = 0
        start_time = time.time()

        while True:

            try:
                while True:
                    frame = self.frame_buffer.get()
                    if frame is None:
                        continue
                    frame_count += 1

                    # Calculate and log FPS every 30 frames
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        fps = 30 / elapsed
                        sys_mgmt.logMetric("driver.fps", fps)
                        sys_mgmt.tokensPerSec = fps  # Update interface property
                        start_time = time.time()

                    msg = self.model_run(*frame)
                    self.object_buffer.put(msg)

                    # Record frame group processed
                    sys_mgmt.recordFrameGroupProcessed(f"frame_{frame_count}")

            except KeyboardInterrupt:
                print("Interrupted by user (SIGINT). Exiting...")

                # Log shutdown metrics
                sys_mgmt.logMetric("driver.total_frames_processed", frame_count)
                sys_mgmt.logMetric("driver.shutdown", True)

                print("Terminating frame buffer...")
                self.frame_buffer.stop()
                print("Terminating object buffer...")
                self.object_buffer.stop()

                exit(0)


def visualize_results(
    left_img: np.ndarray,
    right_img: np.ndarray,
    disparity: np.ndarray,
    points_3D: np.ndarray,
):
    """Show left/right pair, disparity with colorbar, depth with colorbar,
    and a 3-D point cloud.

    Note: relative depth (1/d) and metric depth (f·B/d) differ only by the
    constant factor f·B.  After normalisation to a colour range they look
    *identical*, so we show only one depth panel — with a colorbar carrying
    actual values so you can read real numbers off it.
    """

    min_disparity = 0.0
    valid_disp = disparity > min_disparity

    # ── 1. Disparity (float, actual values) ───────────────────────────────
    disp_vis = disparity.copy().astype(np.float64)
    disp_vis[~valid_disp] = np.nan
    if valid_disp.any():
        p_lo, p_hi = np.percentile(disparity[valid_disp], [2, 98])
        disp_vis = np.clip(disp_vis, p_lo, p_hi)
    else:
        p_lo, p_hi = 0, 1

    # ── 2. Depth (float, actual values) ───────────────────────────────────
    depth_z = np.abs(points_3D[..., 2].copy())
    depth_vis = depth_z.copy().astype(np.float64)
    depth_vis[~valid_disp] = np.nan
    valid_depth = (depth_z > 0) & valid_disp
    if valid_depth.any():
        d_lo, d_hi = np.percentile(depth_z[valid_depth], [2, 98])
        depth_vis = np.clip(depth_vis, d_lo, d_hi)
    else:
        d_lo, d_hi = 0, 1

    # ── 3. Matplotlib figure ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: left image
    axes[0, 0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Left image")
    axes[0, 0].axis("off")

    # Top-right: right image
    axes[0, 1].imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Right image")
    axes[0, 1].axis("off")

    # Bottom-left: disparity with colorbar (actual pixel values)
    im_disp = axes[1, 0].imshow(disp_vis, cmap="jet", vmin=p_lo, vmax=p_hi)
    axes[1, 0].set_title("Disparity  (pixels)")
    axes[1, 0].axis("off")
    fig.colorbar(im_disp, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Bottom-right: depth with colorbar (actual world-unit values)
    im_depth = axes[1, 1].imshow(depth_vis, cmap="magma", vmin=d_lo, vmax=d_hi)
    axes[1, 1].set_title("depth")
    axes[1, 1].axis("off")
    fig.colorbar(im_depth, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
