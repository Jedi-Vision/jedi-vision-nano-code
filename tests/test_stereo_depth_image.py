"""
Test script for StereoDepthEstimator.

Loads a left and right image pair with a known disparity, computes stereo
depth via the jv.scene.StereoDepthEstimator, and visualises the disparity
map, depth map, and a 3-D point cloud.

Usage
-----
# Use the built-in synthetic stereo pair (no files needed):
    python tests/test_stereo_depth.py

# Change the simulated scene depth (disparity is derived from f, B, Z):
    python tests/test_stereo_depth.py --target-depth 500

# Supply your own images + calibration (images are rectified automatically):
    python tests/test_stereo_depth.py \
        --left path/to/left.png --right path/to/right.png \
        --calibration camera_calibration.yaml

# Supply images without calibration but with known camera parameters:
    python tests/test_stereo_depth.py \
        --left left.png --right right.png \
        --focal-length 1050 --baseline 120

# Optionally override stereo matcher parameters:
    python tests/test_stereo_depth.py --num-disparities 128 --block-size 15
"""

import sys
import os
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Make sure the jv package is importable from the repo root
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from jv.scene import StereoDepthEstimator  # noqa: E402
from jv.rectification import Rectifier  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_Q_matrix(
    focal_length: float,
    cx: float,
    cy: float,
    baseline: float,
) -> np.ndarray:
    """
    Build the 4×4 disparity‑to‑depth (Q) matrix used by
    ``cv2.reprojectImageTo3D`` and ``StereoDepthEstimator.reprojectImageTo3D``.

    Parameters
    ----------
    focal_length : float
        Focal length in pixels (assumed same for both cameras).
    cx, cy : float
        Principal point of the **left** camera in pixels.
    baseline : float
        Distance between the two camera centres (in chosen world units, e.g. mm).
    """
    Q = np.array([
        [1, 0,  0,              -cx],
        [0, 1,  0,              -cy],
        [0, 0,  0,     focal_length],
        [0, 0,  1.0 / baseline,   0],
    ], dtype=np.float64)
    return Q


def load_calibration_data(calibration_path: str):
    """
    Load stereo calibration parameters from an OpenCV YAML file.

    Returns
    -------
    tuple
        (mtx1, dist1, mtx2, dist2, R, T) -- suitable for passing straight
        into ``Rectifier(calibration_data=..., img_size=...)``.
    """
    fs = cv2.FileStorage(calibration_path, cv2.FILE_STORAGE_READ)
    mtx1 = fs.getNode("mtx1").mat()
    dist1 = fs.getNode("dist1").mat()
    mtx2 = fs.getNode("mtx2").mat()
    dist2 = fs.getNode("dist2").mat()
    R = fs.getNode("R").mat()
    T = fs.getNode("T").mat()
    fs.release()
    return mtx1, dist1, mtx2, dist2, R, T


def generate_synthetic_stereo_pair(
    width: int = 640,
    height: int = 480,
    focal_length: float = 500.0,
    baseline: float = 60.0,
    target_depth: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Create a synthetic stereo pair by drawing random shapes on an image and
    horizontally shifting it to simulate a right camera view.

    The pixel shift (disparity) is derived from the supplied camera geometry
    so that the images are physically consistent with the Q matrix built from
    the same ``focal_length`` and ``baseline``::

        disparity = focal_length * baseline / target_depth

    Parameters
    ----------
    width, height : int
        Image dimensions in pixels.
    focal_length : float
        Focal length in pixels (same value used to build Q).
    baseline : float
        Stereo baseline in the same world units as ``target_depth``.
    target_depth : float
        Depth of the synthetic scene (same world units as ``baseline``).

    Returns
    -------
    left, right : np.ndarray
        BGR image pair.
    shift_px : int
        The integer disparity (horizontal pixel shift) applied.
    """
    # Derive the shift from the stereo geometry:  d = f * B / Z
    shift_px = max(1, int(round(focal_length * baseline / target_depth)))

    # Create a base left image with geometric features
    left = np.zeros((height, width, 3), dtype=np.uint8)

    # Background gradient (gives texture for block matching)
    for y in range(height):
        left[y, :] = (
            int(50 + 150 * y / height),
            int(100 + 100 * (1 - y / height)),
            int(80 + 120 * y / height),
        )

    rng = np.random.RandomState(42)

    # Add random rectangles
    for _ in range(15):
        x1 = rng.randint(50, width - 150)
        y1 = rng.randint(50, height - 150)
        w = rng.randint(40, 140)
        h = rng.randint(40, 140)
        color = tuple(int(c) for c in rng.randint(0, 255, 3))
        cv2.rectangle(left, (x1, y1), (x1 + w, y1 + h), color, -1)

    # Add circles
    for _ in range(10):
        cx = rng.randint(60, width - 60)
        cy = rng.randint(60, height - 60)
        r = rng.randint(15, 60)
        color = tuple(int(c) for c in rng.randint(0, 255, 3))
        cv2.circle(left, (cx, cy), r, color, -1)

    # Add text for extra texture
    cv2.putText(left, "STEREO TEST", (width // 4, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    # Simulate right image by shifting left image to the left (positive disparity)
    right = np.zeros_like(left)
    if shift_px > 0:
        right[:, :width - shift_px] = left[:, shift_px:]
    else:
        right = left.copy()

    return left, right, shift_px


def compute_depth_direct(
    disparity: np.ndarray,
    focal_length: float,
    baseline: float,
    min_disparity: float = 0.0,
) -> np.ndarray:
    """
    Compute depth directly from disparity using the simple closed‑form
    relationship::

        Z = f * B / d

    This assumes a perfectly rectified stereo pair with identical, aligned
    cameras (no principal‑point offset, no rotation).

    Parameters
    ----------
    disparity : np.ndarray
        H×W disparity map (pixels).
    focal_length : float
        Focal length in pixels.
    baseline : float
        Stereo baseline in world units.
    min_disparity : float
        Disparities at or below this value are treated as invalid.

    Returns
    -------
    depth : np.ndarray
        H×W depth map in the same world units as ``baseline``.
        Invalid pixels are set to 0.
    """
    depth = np.zeros_like(disparity, dtype=np.float64)
    valid = disparity > min_disparity
    depth[valid] = (focal_length * baseline) / disparity[valid]
    return depth


def compare_depth_methods(
    disparity: np.ndarray,
    q_depth: np.ndarray,
    direct_depth: np.ndarray,
    min_disparity: float = 0.0,
):
    """
    Print statistics and plot a side‑by‑side comparison of the two depth
    estimation approaches:

    1. **Q‑matrix reprojection** – full 4×4 homogeneous transform per pixel,
       accounts for principal‑point offsets and can recover (X, Y, Z).
    2. **Direct f·B/d** – scalar depth only, assumes ideal rectified geometry.

    Also shows the per‑pixel difference and a histogram.
    """
    valid = disparity > min_disparity
    q_valid = np.abs(q_depth[valid])
    d_valid = direct_depth[valid]

    if q_valid.size == 0:
        print("No valid pixels to compare.")
        return

    diff = q_valid - d_valid

    print("\n── Depth‑method comparison (valid pixels only) ─────────────")
    print(f"  {'':30s} {'Q‑matrix':>12s}  {'f·B/d':>12s}")
    print(f"  {'Min depth':30s} {q_valid.min():12.2f}  {d_valid.min():12.2f}")
    print(f"  {'Max depth':30s} {q_valid.max():12.2f}  {d_valid.max():12.2f}")
    print(f"  {'Mean depth':30s} {q_valid.mean():12.2f}  {d_valid.mean():12.2f}")
    print(f"  {'Std depth':30s} {q_valid.std():12.4f}  {d_valid.std():12.4f}")
    print(f"  {'Mean |difference|':30s} {np.abs(diff).mean():12.4f}")
    print(f"  {'Max  |difference|':30s} {np.abs(diff).max():12.4f}")
    print()

    # ── Build full‑resolution maps for visualisation ──────────────────────
    q_depth_abs = np.abs(q_depth.copy())
    # Shared colour limits from the union of both maps
    all_valid = np.concatenate([q_valid, d_valid])
    vmin, vmax = np.percentile(all_valid, [2, 98])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Q‑matrix depth
    q_vis = np.clip(q_depth_abs, vmin, vmax)
    q_norm = cv2.normalize(q_vis, None, 0, 255, cv2.NORM_MINMAX)
    axes[0, 0].imshow(
        cv2.cvtColor(
            cv2.applyColorMap(q_norm.astype(np.uint8), cv2.COLORMAP_MAGMA),
            cv2.COLOR_BGR2RGB,
        )
    )
    axes[0, 0].set_title("Depth via Q‑matrix reprojection")
    axes[0, 0].axis("off")

    # Direct depth
    d_vis = np.clip(direct_depth, vmin, vmax)
    d_norm = cv2.normalize(d_vis, None, 0, 255, cv2.NORM_MINMAX)
    axes[0, 1].imshow(
        cv2.cvtColor(
            cv2.applyColorMap(d_norm.astype(np.uint8), cv2.COLORMAP_MAGMA),
            cv2.COLOR_BGR2RGB,
        )
    )
    axes[0, 1].set_title("Depth via f·B / d")
    axes[0, 1].axis("off")

    # Absolute difference map
    diff_map = np.abs(q_depth_abs - direct_depth)
    diff_norm = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX)
    axes[1, 0].imshow(
        cv2.cvtColor(
            cv2.applyColorMap(diff_norm.astype(np.uint8), cv2.COLORMAP_HOT),
            cv2.COLOR_BGR2RGB,
        )
    )
    axes[1, 0].set_title("|Q‑matrix − f·B/d| difference")
    axes[1, 0].axis("off")

    # Histogram of differences
    axes[1, 1].hist(diff, bins=80, color="steelblue", edgecolor="black", alpha=0.8)
    axes[1, 1].set_title("Difference histogram (Q − f·B/d)")
    axes[1, 1].set_xlabel("Depth difference (world units)")
    axes[1, 1].set_ylabel("Pixel count")
    axes[1, 1].axvline(0, color="red", linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.savefig(os.path.join(REPO_ROOT, "output", "depth_method_comparison.png"), dpi=150)
    print("Saved depth_method_comparison.png to output/")
    plt.show()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_results(
    left_img: np.ndarray,
    right_img: np.ndarray,
    disparity: np.ndarray,
    points_3D: np.ndarray,
    has_real_camera_params: bool = False,
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

    depth_label = ("Metric depth  Z = f·B / d  (world units)"
                   if has_real_camera_params
                   else "Depth  Z = f·B / d  (⚠ default params — values NOT meaningful)")

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
    axes[1, 1].set_title(depth_label)
    axes[1, 1].axis("off")
    fig.colorbar(im_depth, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(REPO_ROOT, "output", "stereo_depth_maps.png"), dpi=150)
    print("Saved stereo_depth_maps.png to output/")
    plt.show()

    # ── 5. 3-D point cloud scatter plot ──────────────────────────────────
    step = max(1, points_3D.shape[0] * points_3D.shape[1] // 50_000)
    pts = points_3D[::step, ::step].reshape(-1, 3)
    disp_sub = disparity[::step, ::step].reshape(-1)
    colours = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)[::step, ::step].reshape(-1, 3) / 255.0

    # Only keep pixels with meaningful disparity (filters near-zero junk)
    mask = (
        (disp_sub > min_disparity)
        & (np.abs(pts[:, 2]) > 0)
        & np.isfinite(pts).all(axis=1)
    )
    pts = pts[mask]
    colours = colours[mask]

    if len(pts) == 0:
        print("No valid 3D points to plot -- skipping point cloud.")
        return

    # Clip depth outliers so the cloud isn't dominated by extreme points
    z_abs = np.abs(pts[:, 2])
    z_lo, z_hi = np.percentile(z_abs, [2, 98])
    inlier = (z_abs >= z_lo) & (z_abs <= z_hi)
    pts = pts[inlier]
    colours = colours[inlier]

    fig3d = plt.figure(figsize=(12, 9))
    ax3d = fig3d.add_subplot(111, projection="3d")
    ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                 c=colours, s=0.3, marker=".")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z (depth)")
    ax3d.set_title("3D Point Cloud")
    ax3d.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(REPO_ROOT, "output", "stereo_point_cloud.png"), dpi=150)
    print("Saved stereo_point_cloud.png to output/")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test StereoDepthEstimator: visualise disparity, depth, and 3D point cloud.",
    )
    parser.add_argument("--left", type=str, default=None,
                        help="Path to the left image (will be rectified if --calibration is given).")
    parser.add_argument("--right", type=str, default=None,
                        help="Path to the right image (will be rectified if --calibration is given).")
    parser.add_argument("--calibration", type=str, default=None,
                        help="Path to OpenCV YAML calibration file. When supplied with "
                             "--left/--right the images are rectified via the Rectifier "
                             "and Q is taken from the rectifier.")
    parser.add_argument("--num-disparities", type=int, default=16 * 6,
                        help="numDisparities for StereoBM (must be divisible by 16).")
    parser.add_argument("--block-size", type=int, default=11,
                        help="blockSize for StereoBM (odd, >= 5).")
    parser.add_argument("--focal-length", type=float, default=None,
                        help="Focal length in pixels.  Required for meaningful metric "
                             "depth when no --calibration is given.")
    parser.add_argument("--baseline", type=float, default=None,
                        help="Stereo baseline (world units, e.g. mm).  Required for "
                             "meaningful metric depth when no --calibration is given.")
    parser.add_argument("--target-depth", type=float, default=1000.0,
                        help="Target scene depth (same units as baseline) for the "
                             "synthetic stereo pair.  disparity = f*B/Z.")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Default camera parameters (used for synthetic pair or as fallback) ──
    DEFAULT_FOCAL_LENGTH = 500.0   # pixels
    DEFAULT_BASELINE     = 60.0    # mm
    FOCAL_LENGTH = args.focal_length if args.focal_length is not None else DEFAULT_FOCAL_LENGTH
    BASELINE     = args.baseline     if args.baseline     is not None else DEFAULT_BASELINE
    has_real_camera_params = (args.calibration is not None
                              or args.focal_length is not None
                              or args.baseline is not None)

    # ── Load or generate stereo pair ──────────────────────────────────────
    if args.left and args.right:
        left_img = cv2.imread(args.left)
        right_img = cv2.imread(args.right)
        if left_img is None or right_img is None:
            print(f"Error: could not read images ({args.left}, {args.right})")
            sys.exit(1)
        print(f"Loaded stereo pair: {args.left}  /  {args.right}")

        h, w = left_img.shape[:2]
        print(f"Image size: {w}×{h}")

        if args.calibration:
            # Rectify the supplied images and obtain Q from the Rectifier
            print(f"Rectifying images with calibration: {args.calibration}")
            calib_data = load_calibration_data(args.calibration)
            rectifier = Rectifier(calibration_data=calib_data, img_size=(w, h))
            left_img, right_img = rectifier.rectify((left_img, right_img))
            Q = rectifier.Q
            print("Images rectified via Rectifier.")
        else:
            # No calibration -- build Q from user-supplied or default params.
            cx, cy = w / 2.0, h / 2.0
            Q = build_Q_matrix(FOCAL_LENGTH, cx, cy, BASELINE)

            if not has_real_camera_params:
                print(
                    "\n"
                    "  ⚠⚠⚠  WARNING: No --calibration, --focal-length, or --baseline provided.\n"
                    f"       Using MADE-UP defaults (f={DEFAULT_FOCAL_LENGTH}, B={DEFAULT_BASELINE}).\n"
                    "       The metric depth map and point cloud will NOT be meaningful.\n"
                    "       Disparity and relative depth (1/d) are still valid.\n"
                    "       Supply real values via --focal-length / --baseline for correct depth.\n"
                )
            else:
                print(f"Using user-supplied camera parameters: "
                      f"f={FOCAL_LENGTH}, B={BASELINE}")
    else:
        print("No images supplied -- generating synthetic stereo pair …")
        W, H = 640, 480
        cx, cy = W / 2.0, H / 2.0
        has_real_camera_params = True  # synthetic params are self-consistent

        # Generate the pair using the *same* camera parameters that will
        # be baked into the Q matrix, so geometry is self-consistent.
        left_img, right_img, shift_px = generate_synthetic_stereo_pair(
            width=W,
            height=H,
            focal_length=FOCAL_LENGTH,
            baseline=BASELINE,
            target_depth=args.target_depth,
        )
        print(f"Synthetic disparity (shift): {shift_px} px  "
              f"(target depth {args.target_depth}, f={FOCAL_LENGTH}, B={BASELINE})")

        h, w = left_img.shape[:2]
        print(f"Image size: {w}×{h}")

        # Build Q from the *same* parameters used to create the pair
        Q = build_Q_matrix(FOCAL_LENGTH, cx, cy, BASELINE)

    print(f"Q matrix:\n{Q}\n")

    # ── Run StereoDepthEstimator ──────────────────────────────────────────
    estimator = StereoDepthEstimator(
        num_disparities=args.num_disparities,
        block_size=args.block_size
    )

    print("Computing stereo depth …")
    points_3D, disparity = estimator.run((left_img, right_img), Q)

    # Print quick statistics
    valid_mask = disparity > 0
    if valid_mask.any():
        print(f"Disparity  – min: {disparity[valid_mask].min():.2f}, "
              f"max: {disparity[valid_mask].max():.2f}, "
              f"mean: {disparity[valid_mask].mean():.2f}")
    else:
        print("Warning: no valid disparities found.")

    depth_z = np.abs(points_3D[..., 2])
    valid_depth = depth_z[depth_z > 0]
    if valid_depth.size:
        print(f"Depth (Z)  – min: {valid_depth.min():.2f}, "
              f"max: {valid_depth.max():.2f}, "
              f"mean: {valid_depth.mean():.2f}")
    else:
        print("Warning: no valid depth values found.")

    # ── Comparison: Q-matrix reprojection vs direct f·B/d ─────────────────
    # Extract focal length and baseline from Q so the comparison works for
    # both the synthetic Q and one produced by cv2.stereoRectify.
    #
    # Standard Q layout (OpenCV):
    #   Q[2,3] = focal_length
    #   Q[3,2] = -1 / Tx   (Tx = signed baseline component along X)
    # Our synthetic Q uses +1/B in Q[3,2], so we take the absolute value.
    # f_from_Q = Q[2, 3]
    # b_from_Q = np.abs(1.0 / Q[3, 2]) if Q[3, 2] != 0 else BASELINE

    # print(f"\nParameters extracted from Q for direct method: "
    #       f"f = {f_from_Q:.2f} px, B = {b_from_Q:.2f}")

    # direct_depth = compute_depth_direct(
    #     disparity,
    #     focal_length=f_from_Q,
    #     baseline=b_from_Q,
    #     min_disparity=args.min_disparity,
    # )

    # compare_depth_methods(
    #     disparity,
    #     q_depth=points_3D[..., 2],
    #     direct_depth=direct_depth,
    #     min_disparity=args.min_disparity,
    # )

    # ── Visualise ─────────────────────────────────────────────────────────
    os.makedirs(os.path.join(REPO_ROOT, "output"), exist_ok=True)
    visualize_results(left_img, right_img, disparity, points_3D,
                      has_real_camera_params=has_real_camera_params)

    print("\nDone.")


if __name__ == "__main__":
    main()
