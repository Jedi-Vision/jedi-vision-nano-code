from jv.audio import ObjectBuffer
from jv.representation import YoloObjectRepresentationModel
from jv.scene import VideoDepthAnything, MODEL_CONFIGS
from jv.scene import StereoDepthEstimator
from jv.rectification import Rectifier
from jv.camera import FrameBuffer
from jv.representation.data import ObjectRepData
from typing import Literal
import torch
import cv2
import time

from jv.management import SystemManagement, log_block
sys_mgmt = SystemManagement()


DEPTH_CONV = (1/0.001)  # 1mm per 0.001 meter for stereo depth conversion


class Driver:

    def __init__(
        self,
        device: Literal["cpu", "mps", "cuda"],
        output_to: Literal["socket", "file", "none"] = "socket",
        serial_type: Literal["struct", "protobuf"] = "struct",
        object_model_name: str = "yolo11",
        vda_model_name: str = "vits",
        chkpts_folder: str = "./checkpoints",
        retain_frames: int = 30,
        object_buffer_size: int = 0,
        frame_buffer_size: int = 0,
        warmup_frames: int = 30,
        camera_index: str | int = 0,
        binocular: bool = True,
        frame_skip: int = 0,
        frame_rate: int = 30,
        show_det: bool = False,
        depth: bool = True,
        metric: bool = False,
        use_sgbm: bool = True,
        num_disparities: int = 16 * 6,
        block_size: int = 11,
        min_disparity: int = 0,
        P1: int | None = None,
        P2: int | None = None,
        disp12_max_diff: int = 1,
        pre_filter_cap: int = 0,
        uniqueness_ratio: int = 5,
        speckle_window_size: int = 100,
        speckle_range: int = 2,
        max_depth: float = 100.,
        calibration_data: str = "camera_calibration.yaml",
        gstreamer_kwargs: dict = {}
    ) -> None:
        """
        Initializes the driver with specified configuration for device, output, models, buffers, and video processing.

        Args:
            device (Literal["cpu", "mps", "cuda"]): Device to run models on.
            output_to (Literal["socket", "file", "none"]): Output destination for object buffer.
            object_model_name (str, optional): Name of the object detection model. Defaults to "yolo11".
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
            use_sgbm (bool, optional): If True, use StereoSGBM (more accurate, slower). If False,
                use StereoBM (faster, simpler). Defaults to True.
            num_disparities: Maximum disparity minus minimum disparity. The value is always greater
                than zero. Must be divisible by 16. Determines the resolution of your stereo / depth map.
            block_size: Matched block size. It must be an odd number >=1. Normally, it should be
                somewhere in the 3 to 11 range.
            min_disparity: Minimum possible disparity value. Normally, it is zero but sometimes
                rectification algorithms can shift images, so this parameter needs to be adjusted
                accordingly.
            P1: The first parameter controlling the disparity smoothness. P1 is the penalty on the
                disparity change by plus or minus 1 between neighbor pixels. Defaults to
                8 * 1 * block_size^2.
            P2: The second parameter controlling the disparity smoothness. The larger the values are,
                the smoother the disparity is. P2 is the penalty on the disparity change by more than
                1 between neighbor pixels. The algorithm requires P2 > P1. Defaults to
                32 * 1 * block_size^2.
            disp12_max_diff: Maximum allowed difference (in integer pixel units) in the left-right
                disparity check. Set it to a non-positive value to disable the check.
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
            gstreamer_args (dict, optional): GStreamer kwargs. See buffer.py for more information.
        """

        self.frame_buffer = FrameBuffer(
            size=frame_buffer_size,
            camera_index=camera_index,
            warmup_frames=warmup_frames,
            frame_skip=frame_skip,
            frame_rate=frame_rate,
            binocular=binocular,
            gstreamer_kwargs=gstreamer_kwargs
        )

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

        # Monocular depth
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

        # Binocular depth
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
        self.depth_estimator = StereoDepthEstimator(
            use_sgbm=use_sgbm,
            num_disparities=num_disparities,
            block_size=block_size,
            min_disparity=min_disparity,
            P1=P1,
            P2=P2,
            disp12_max_diff=disp12_max_diff,
            pre_filter_cap=pre_filter_cap,
            uniqueness_ratio=uniqueness_ratio,
            speckle_window_size=speckle_window_size,
            speckle_range=speckle_range,
            max_depth=max_depth
        )
        if self.binocular:
            self.rectifier = Rectifier(
                calibration_data=load_calibration_data(calibration_data),
                img_size=(gstreamer_kwargs['display_height'], gstreamer_kwargs['display_width']),
            )

        self.device = device
        self.show_det = show_det

        # Log system configuration at startup
        sys_mgmt.updateSetting("device", device)
        sys_mgmt.updateSetting("frame_rate", frame_rate)
        sys_mgmt.updateSetting("depth_enabled", depth)
        sys_mgmt.logMetric("driver.initialized", True)

    @log_block("model_inference")
    def model_run(self, frame, frame_number, timestamp_ms, object_kwargs, depth_kwargs):
        # Log frame processing
        sys_mgmt.logMetric("frame.number", frame_number)
        sys_mgmt.logMetric("frame.timestamp_ms", timestamp_ms)

        # If we are using binocular images for depth, we need to ensure that detection
        # is done on rectified image pairs, so that we can then map the depth calculated
        # for rectified images to the corresponding objects given the rectified (x,y)
        # coordinates
        if self.binocular:
            frame = self.rectifier.rectify(frame)

        objects = self.env_model.run(frame[0], show_det=self.show_det, **object_kwargs)

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
                for obj in objects:
                    x, y = obj.x_2d, obj.y_2d
                    obj.depth = float(depth[y][x][2]) * DEPTH_CONV
                    obj.x_3d, obj.y_3d = float(depth[y][x][0]) * DEPTH_CONV, float(depth[y][x][1]) * DEPTH_CONV

            else:  # Use monocular depth model
                frame = cv2.cvtColor(frame[0], cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for VDA
                depth = self.scene_model.infer_video_depth_one(
                    frame,
                    input_size=518,
                    device=self.device,
                    **depth_kwargs
                )  # fp32=False causes a black output and NaN values in model

                # Add depth information to objects
                for obj in objects:
                    obj.depth = float(depth[obj.y_2d][obj.x_2d])

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
                else:
                    colormap = self.scene_model.colormap
                    # Normalize
                    color_depth = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8)*255).to(torch.uint8)
                    color_depth = colormap[color_depth]
                    cv2.imshow("msg.mask", color_depth)
                    cv2.waitKey(1)

        return ObjectRepData(
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
            objects=objects
        )

    @log_block("driver_main_loop")
    def run(self, object_kwargs: dict = {}, depth_kwargs: dict = {}):

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

                    msg = self.model_run(
                        *frame,
                        object_kwargs=object_kwargs,
                        depth_kwargs=depth_kwargs
                    )
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
