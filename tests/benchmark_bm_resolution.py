"""
Benchmark BM and SGBM StereoDepthEstimators at varying image resolutions.

Measures disparity computation time and full run (disparity + 3D reprojection)
for both StereoBM and StereoSGBM across a range of resolutions, using either
the bundled Middlebury stereo datasets or a synthetic stereo pair.

Results are printed as a table and saved as a plot to output/.

Usage
-----
# Default: benchmark with synthetic pair at standard resolutions
    python tests/benchmark_sgbm_resolution.py

# Use a Middlebury dataset as the source image pair
    python tests/benchmark_sgbm_resolution.py \
        --dataset examples/stereo/Playtable-perfect

# Custom resolution list (width values; height is derived from aspect ratio)
    python tests/benchmark_sgbm_resolution.py --widths 320 640 1280 1920 2560

# Change number of timing iterations
    python tests/benchmark_sgbm_resolution.py --iterations 20

# Override stereo parameters
    python tests/benchmark_sgbm_resolution.py --num-disparities 128 --block-size 11
"""

import sys
import os
import argparse
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Make sure the jv package is importable from the repo root
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from jv.scene.depth_estimator import BMStereoDepthEstimator, SGBMStereoDepthEstimator  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_WIDTHS = [320, 480, 640, 800, 960, 1280, 1600, 1920, 2560]

DEFAULT_FOCAL_LENGTH = 500.0   # pixels  (synthetic pair)
DEFAULT_BASELINE = 60.0        # mm      (synthetic pair)


def build_Q_matrix(
    focal_length: float,
    cx: float,
    cy: float,
    baseline: float,
) -> np.ndarray:
    """Build the 4×4 disparity-to-depth (Q) matrix."""
    return np.array([
        [1, 0,  0,              -cx],
        [0, 1,  0,              -cy],
        [0, 0,  0,     focal_length],
        [0, 0,  1.0 / baseline,   0],
    ], dtype=np.float64)


def generate_synthetic_stereo_pair(
    width: int = 640,
    height: int = 480,
    focal_length: float = 500.0,
    baseline: float = 60.0,
    target_depth: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic stereo pair with rich texture for block matching."""
    shift_px = max(1, int(round(focal_length * baseline / target_depth)))

    left = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        left[y, :] = (
            int(50 + 150 * y / height),
            int(100 + 100 * (1 - y / height)),
            int(80 + 120 * y / height),
        )

    rng = np.random.RandomState(42)
    for _ in range(15):
        x1 = rng.randint(50, max(51, width - 150))
        y1 = rng.randint(50, max(51, height - 150))
        w = rng.randint(40, 140)
        h = rng.randint(40, 140)
        color = tuple(int(c) for c in rng.randint(0, 255, 3))
        cv2.rectangle(left, (x1, y1), (x1 + w, y1 + h), color, -1)

    for _ in range(10):
        cx_r = rng.randint(60, max(61, width - 60))
        cy_r = rng.randint(60, max(61, height - 60))
        r = rng.randint(15, 60)
        color = tuple(int(c) for c in rng.randint(0, 255, 3))
        cv2.circle(left, (cx_r, cy_r), r, color, -1)

    cv2.putText(left, "STEREO", (width // 6, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.5, width / 640), (255, 255, 255), 2)

    right = np.zeros_like(left)
    if shift_px > 0 and shift_px < width:
        right[:, :width - shift_px] = left[:, shift_px:]
    else:
        right = left.copy()

    return left, right


def load_middlebury_pair(dataset_dir: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load a Middlebury stereo pair and its calibration data.

    Returns
    -------
    left, right : np.ndarray
        BGR images.
    calib : dict
        Parsed calibration with keys: focal_length, baseline, cx0, cy, cx1, ndisp.
    """
    left_path = os.path.join(dataset_dir, "im0.png")
    right_path = os.path.join(dataset_dir, "im1.png")
    calib_path = os.path.join(dataset_dir, "calib.txt")

    left = cv2.imread(left_path)
    right = cv2.imread(right_path)
    if left is None or right is None:
        raise FileNotFoundError(
            f"Could not load images from {dataset_dir}. "
            f"Expected im0.png and im1.png."
        )

    calib = {}
    with open(calib_path, "r") as f:
        for line in f:
            if line.startswith("cam0="):
                # Parse [f 0 cx; 0 f cy; 0 0 1]
                mat_str = line.split("=", 1)[1].strip().strip("[]")
                rows = mat_str.split(";")
                vals = rows[0].split()
                calib["focal_length"] = float(vals[0])
                calib["cx0"] = float(vals[2])
                vals2 = rows[1].split()
                calib["cy"] = float(vals2[2])
            elif line.startswith("cam1="):
                mat_str = line.split("=", 1)[1].strip().strip("[]")
                rows = mat_str.split(";")
                vals = rows[0].split()
                calib["cx1"] = float(vals[2])
            elif line.startswith("baseline="):
                calib["baseline"] = float(line.split("=")[1].strip())
            elif line.startswith("ndisp="):
                calib["ndisp"] = int(line.split("=")[1].strip())

    return left, right, calib


def resize_pair(
    left: np.ndarray,
    right: np.ndarray,
    target_width: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Resize a stereo pair to a target width, preserving aspect ratio.

    Returns
    -------
    left_resized, right_resized : np.ndarray
    scale : float
        The scaling factor applied (target_width / original_width).
    """
    h, w = left.shape[:2]
    scale = target_width / w
    target_height = int(h * scale)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    left_r = cv2.resize(left, (target_width, target_height), interpolation=interp)
    right_r = cv2.resize(right, (target_width, target_height), interpolation=interp)
    return left_r, right_r, scale


def benchmark_at_resolution(
    estimator: SGBMStereoDepthEstimator,
    left: np.ndarray,
    right: np.ndarray,
    Q: np.ndarray,
    iterations: int = 10,
) -> dict:
    """
    Time each stage of stereo depth estimation at a given resolution.

    Stages timed independently:
      1. Grayscale conversion (BGR → GRAY for both images)
      2. Raw stereo inference  (stereo.compute())
      3. calc_disparity        (grayscale + inference + post-process)
      4. 3D reprojection       (reprojectImageTo3D)
      5. Full run              (calc_disparity + reprojection)

    Returns
    -------
    dict with timing statistics for every stage.
    """
    h, w = left.shape[:2]

    # Warm-up run (not timed)
    _ = estimator.calc_disparity((left, right))

    # ── Stage 1+2: grayscale conversion + raw stereo.compute() ────────
    gray_times = []
    inference_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        t1 = time.perf_counter()
        raw_disp = estimator.stereo.compute(left_gray, right_gray)
        t2 = time.perf_counter()
        gray_times.append((t1 - t0) * 1000.0)
        inference_times.append((t2 - t1) * 1000.0)

    # ── Stage 3: full calc_disparity (grayscale + inference + clip) ───
    disp_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        disparity = estimator.calc_disparity((left, right))
        t1 = time.perf_counter()
        disp_times.append((t1 - t0) * 1000.0)

    # ── Stage 4: 3D reprojection only ────────────────────────────────
    reproj_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        points_3D = estimator.reprojectImageTo3D(disparity, Q)
        points_3D[np.isinf(points_3D)] = 0
        t1 = time.perf_counter()
        reproj_times.append((t1 - t0) * 1000.0)

    # ── Stage 5: full run (disparity + 3D reprojection) ──────────────
    run_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        points_3D, disparity = estimator.run((left, right), Q)
        t1 = time.perf_counter()
        run_times.append((t1 - t0) * 1000.0)

    valid_ratio = float(np.count_nonzero(disparity > 0)) / (h * w)

    return {
        "width": w,
        "height": h,
        "pixels": w * h,
        "gray_ms": float(np.mean(gray_times)),
        "gray_std_ms": float(np.std(gray_times)),
        "inference_ms": float(np.mean(inference_times)),
        "inference_std_ms": float(np.std(inference_times)),
        "disparity_ms": float(np.mean(disp_times)),
        "disparity_std_ms": float(np.std(disp_times)),
        "reproj_ms": float(np.mean(reproj_times)),
        "reproj_std_ms": float(np.std(reproj_times)),
        "run_ms": float(np.mean(run_times)),
        "run_std_ms": float(np.std(run_times)),
        "valid_ratio": valid_ratio,
    }


def print_results_table(results: list[dict], label: str = ""):
    """Print benchmark results as a formatted table."""
    header = (
        f"{'Resolution':>14s}  {'Pixels':>10s}  "
        f"{'Grayscale (ms)':>16s}  {'Inference (ms)':>16s}  "
        f"{'Disparity (ms)':>16s}  {'Reproject (ms)':>16s}  "
        f"{'Full Run (ms)':>16s}  {'Valid %':>8s}"
    )
    if label:
        print(f"\n{'─' * 20}  {label}  {'─' * 20}")
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        res_str = f"{r['width']}×{r['height']}"
        px_str = f"{r['pixels']:,}"
        gray_str = f"{r['gray_ms']:7.2f} ± {r['gray_std_ms']:.2f}"
        infer_str = f"{r['inference_ms']:7.1f} ± {r['inference_std_ms']:.1f}"
        disp_str = f"{r['disparity_ms']:7.1f} ± {r['disparity_std_ms']:.1f}"
        reproj_str = f"{r['reproj_ms']:7.1f} ± {r['reproj_std_ms']:.1f}"
        run_str = f"{r['run_ms']:7.1f} ± {r['run_std_ms']:.1f}"
        valid_str = f"{r['valid_ratio'] * 100:5.1f}%"
        print(f"{res_str:>14s}  {px_str:>10s}  {gray_str:>16s}  {infer_str:>16s}  "
              f"{disp_str:>16s}  {reproj_str:>16s}  {run_str:>16s}  {valid_str:>8s}")

    print("=" * len(header))


def _extract_series(results: list[dict]):
    """Extract plotting series from a list of benchmark result dicts."""
    return {
        "widths": [r["width"] for r in results],
        "pixels": [r["pixels"] for r in results],
        "gray_ms": [r["gray_ms"] for r in results],
        "infer_ms": [r["inference_ms"] for r in results],
        "infer_std": [r["inference_std_ms"] for r in results],
        "disp_ms": [r["disparity_ms"] for r in results],
        "disp_std": [r["disparity_std_ms"] for r in results],
        "reproj_ms": [r["reproj_ms"] for r in results],
        "run_ms": [r["run_ms"] for r in results],
        "run_std": [r["run_std_ms"] for r in results],
        "valid": [r["valid_ratio"] * 100 for r in results],
    }


def plot_results(
    bm_results: list[dict],
    sgbm_results: list[dict],
    output_path: str,
    source_label: str,
):
    """Save a multi-panel benchmark plot comparing BM vs SGBM."""
    bm = _extract_series(bm_results)
    sg = _extract_series(sgbm_results)
    widths = sg["widths"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f"BM vs SGBM Resolution Benchmark — {source_label}", fontsize=14, fontweight="bold")

    x = np.arange(len(widths))
    tick_labels = [f"{w}" for w in widths]
    bar_w = 0.35

    # -- (0,0): Grouped bar — inference time BM vs SGBM --
    ax = axes[0, 0]
    ax.bar(x - bar_w / 2, bm["infer_ms"], bar_w, label="BM Inference", color="#4e79a7")
    ax.bar(x + bar_w / 2, sg["infer_ms"], bar_w, label="SGBM Inference", color="#e15759")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_xlabel("Image width (px)")
    ax.set_ylabel("Inference time (ms)")
    ax.set_title("Inference Time: BM vs SGBM")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # -- (0,1): Line plot — inference latency vs width --
    ax = axes[0, 1]
    ax.errorbar(widths, bm["infer_ms"], yerr=bm["infer_std"], fmt="o-", capsize=4,
                label="BM Inference", color="#4e79a7")
    ax.errorbar(widths, sg["infer_ms"], yerr=sg["infer_std"], fmt="s-", capsize=4,
                label="SGBM Inference", color="#e15759")
    ax.errorbar(widths, bm["run_ms"], yerr=bm["run_std"], fmt="o--", capsize=4,
                label="BM Full Run", color="#4e79a7", alpha=0.5)
    ax.errorbar(widths, sg["run_ms"], yerr=sg["run_std"], fmt="s--", capsize=4,
                label="SGBM Full Run", color="#e15759", alpha=0.5)
    ax.set_xlabel("Image width (px)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Latency vs. Resolution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # -- (0,2): Speedup ratio (SGBM / BM) --
    ax = axes[0, 2]
    infer_speedup = [s / b if b > 0 else 0 for s, b in zip(sg["infer_ms"], bm["infer_ms"])]
    run_speedup = [s / b if b > 0 else 0 for s, b in zip(sg["run_ms"], bm["run_ms"])]
    ax.plot(widths, infer_speedup, "o-", label="Inference (SGBM / BM)", color="#e15759")
    ax.plot(widths, run_speedup, "s--", label="Full Run (SGBM / BM)", color="#76b7b2")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Image width (px)")
    ax.set_ylabel("Slowdown factor (×)")
    ax.set_title("SGBM Slowdown vs BM")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # -- (1,0): FPS vs width --
    ax = axes[1, 0]
    bm_infer_fps = [1000.0 / t if t > 0 else 0 for t in bm["infer_ms"]]
    sg_infer_fps = [1000.0 / t if t > 0 else 0 for t in sg["infer_ms"]]
    bm_run_fps = [1000.0 / t if t > 0 else 0 for t in bm["run_ms"]]
    sg_run_fps = [1000.0 / t if t > 0 else 0 for t in sg["run_ms"]]
    ax.plot(widths, bm_infer_fps, "o-", label="BM Inference", color="#4e79a7")
    ax.plot(widths, sg_infer_fps, "s-", label="SGBM Inference", color="#e15759")
    ax.plot(widths, bm_run_fps, "o--", label="BM Full Run", color="#4e79a7", alpha=0.5)
    ax.plot(widths, sg_run_fps, "s--", label="SGBM Full Run", color="#e15759", alpha=0.5)
    ax.set_xlabel("Image width (px)")
    ax.set_ylabel("FPS")
    ax.set_title("Throughput vs. Resolution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # -- (1,1): Stacked time breakdown for both (side by side) --
    ax = axes[1, 1]
    # BM bars (left)
    ax.bar(x - bar_w / 2, bm["gray_ms"], bar_w, label="Grayscale", color="#b07aa1")
    ax.bar(x - bar_w / 2, bm["infer_ms"], bar_w, bottom=bm["gray_ms"],
           label="BM Inference", color="#4e79a7")
    bm_rp_bottom = [g + i for g, i in zip(bm["gray_ms"], bm["infer_ms"])]
    ax.bar(x - bar_w / 2, bm["reproj_ms"], bar_w, bottom=bm_rp_bottom,
           label="3D Reproj", color="#59a14f")
    # SGBM bars (right)
    ax.bar(x + bar_w / 2, sg["gray_ms"], bar_w, color="#b07aa1")
    ax.bar(x + bar_w / 2, sg["infer_ms"], bar_w, bottom=sg["gray_ms"],
           label="SGBM Inference", color="#e15759")
    sg_rp_bottom = [g + i for g, i in zip(sg["gray_ms"], sg["infer_ms"])]
    ax.bar(x + bar_w / 2, sg["reproj_ms"], bar_w, bottom=sg_rp_bottom, color="#59a14f")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_xlabel("Image width (px)  [left=BM, right=SGBM]")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Per-Stage Breakdown (BM vs SGBM)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    # -- (1,2): Valid disparity ratio --
    ax = axes[1, 2]
    ax.bar(x - bar_w / 2, bm["valid"], bar_w, label="BM", color="#4e79a7", alpha=0.8)
    ax.bar(x + bar_w / 2, sg["valid"], bar_w, label="SGBM", color="#e15759", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_xlabel("Image width (px)")
    ax.set_ylabel("Valid disparity (%)")
    ax.set_title("Valid Disparity Ratio")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark BM & SGBM StereoDepthEstimators at varying resolutions.",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to a Middlebury-format stereo directory (with im0.png, im1.png, calib.txt). "
             "If omitted, a synthetic pair is generated.",
    )
    parser.add_argument(
        "--widths", type=int, nargs="+", default=None,
        help="List of target widths to benchmark. Heights are derived from aspect ratio. "
             f"Default: {DEFAULT_WIDTHS}",
    )
    parser.add_argument(
        "--iterations", type=int, default=10,
        help="Number of timing iterations per resolution (default: 10).",
    )
    parser.add_argument(
        "--num-disparities", type=int, default=16 * 6,
        help="numDisparities for both BM and SGBM (must be divisible by 16, default: 96).",
    )
    parser.add_argument(
        "--block-size", type=int, default=11,
        help="blockSize for both BM and SGBM (odd, >= 5 for BM, default: 11).",
    )
    parser.add_argument(
        "--max-depth", type=float, default=10000.0,
        help="Maximum depth value in mm (default: 10000).",
    )
    return parser.parse_args()


def _run_benchmark_sweep(
    label: str,
    estimator,
    left_src: np.ndarray,
    right_src: np.ndarray,
    widths: list[int],
    focal_length: float,
    baseline: float,
    iterations: int,
) -> list[dict]:
    """Benchmark a single estimator across all target widths."""
    print(f"\n── {label} {'─' * (50 - len(label))}")
    results = []
    for target_w in widths:
        left_r, right_r, scale = resize_pair(left_src, right_src, target_w)
        h, w = left_r.shape[:2]

        f_scaled = focal_length * scale
        cx = w / 2.0
        cy = h / 2.0
        Q = build_Q_matrix(f_scaled, cx, cy, baseline)

        print(f"  {w}×{h} ({w * h:,} px, scale={scale:.3f}) ...", end="", flush=True)

        result = benchmark_at_resolution(
            estimator, left_r, right_r, Q, iterations=iterations,
        )
        results.append(result)
        print(f"  inference: {result['inference_ms']:.1f} ms, "
              f"disparity: {result['disparity_ms']:.1f} ms, "
              f"full: {result['run_ms']:.1f} ms")
    return results


def main():
    args = parse_args()
    widths = args.widths if args.widths else DEFAULT_WIDTHS

    # BM requires block_size >= 5 and odd
    bm_block_size = max(5, args.block_size)
    if bm_block_size % 2 == 0:
        bm_block_size += 1

    print("=" * 60)
    print("  BM vs SGBM Stereo Depth — Resolution Benchmark")
    print("=" * 60)
    print(f"  numDisparities={args.num_disparities}")
    print(f"  BM  blockSize={bm_block_size}")
    print(f"  SGBM blockSize={args.block_size}")
    print(f"  Iterations per resolution: {args.iterations}")
    print(f"  Target widths: {widths}")

    # ── Load source images ────────────────────────────────────────────────
    if args.dataset:
        dataset_path = os.path.join(REPO_ROOT, args.dataset) \
            if not os.path.isabs(args.dataset) else args.dataset
        left_src, right_src, calib = load_middlebury_pair(dataset_path)
        focal_length = calib["focal_length"]
        baseline = calib["baseline"]
        source_label = os.path.basename(dataset_path)
        print(f"  Source: {dataset_path}")
        print(f"  Original size: {left_src.shape[1]}×{left_src.shape[0]}")
        print(f"  Focal length: {focal_length:.1f} px, Baseline: {baseline:.1f} mm")
    else:
        max_w = max(widths)
        max_h = int(max_w * 3 / 4)
        focal_length = DEFAULT_FOCAL_LENGTH * (max_w / 640)
        baseline = DEFAULT_BASELINE
        left_src, right_src = generate_synthetic_stereo_pair(
            width=max_w, height=max_h,
            focal_length=focal_length, baseline=baseline,
        )
        source_label = f"Synthetic {max_w}×{max_h}"
        print(f"  Source: synthetic pair ({max_w}×{max_h})")
        print(f"  Focal length: {focal_length:.1f} px, Baseline: {baseline:.1f} mm")

    # ── Create estimators ─────────────────────────────────────────────────
    bm_estimator = BMStereoDepthEstimator(
        num_disparities=args.num_disparities,
        block_size=bm_block_size,
        max_depth=args.max_depth,
    )
    sgbm_estimator = SGBMStereoDepthEstimator(
        num_disparities=args.num_disparities,
        block_size=args.block_size,
        max_depth=args.max_depth,
    )

    # ── Benchmark BM ──────────────────────────────────────────────────────
    bm_results = _run_benchmark_sweep(
        "StereoBM", bm_estimator, left_src, right_src,
        widths, focal_length, baseline, args.iterations,
    )

    # ── Benchmark SGBM ────────────────────────────────────────────────────
    sgbm_results = _run_benchmark_sweep(
        "StereoSGBM", sgbm_estimator, left_src, right_src,
        widths, focal_length, baseline, args.iterations,
    )

    # ── Print tables ──────────────────────────────────────────────────────
    print_results_table(bm_results, label=f"StereoBM (blockSize={bm_block_size})")
    print_results_table(sgbm_results, label=f"StereoSGBM (blockSize={args.block_size})")

    # ── Save plot ─────────────────────────────────────────────────────────
    os.makedirs(os.path.join(REPO_ROOT, "output"), exist_ok=True)
    output_path = os.path.join(REPO_ROOT, "output", "bm_vs_sgbm_resolution_benchmark.png")
    plot_results(bm_results, sgbm_results, output_path, source_label)

    print("\nDone.")


if __name__ == "__main__":
    main()
