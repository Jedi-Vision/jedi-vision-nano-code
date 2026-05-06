#!/usr/bin/env python3
"""
Benchmark disparity accuracy against Middlebury pairs.

Usage examples:
  python tests/benchmark_accuracy_middlebury.py --dataset /path/to/middlebury/scene_dir
  python tests/benchmark_accuracy_middlebury.py --dataset /path/to/all_scenes --mode sweep

Outputs:
  - results.csv: per-trial per-scene metrics
  - results.json: aggregated summary
"""

import os
import argparse
import time
import json
from glob import glob
from typing import Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd

from jv.scene.depth_estimator import SGBMStereoDepthEstimator, BMStereoDepthEstimator


# -------------------------
# IO helpers
# -------------------------
def read_calib(calib_path: str) -> Dict[str, float]:
    calib = {}
    with open(calib_path, "r") as f:
        for line in f:
            if "=" not in line:
                continue
            k, v = line.strip().split("=", 1)
            calib[k] = v

    # Basic numeric extraction helpers for cam0/cam1
    def parse_cam(s):
        s = s.strip()
        s = s[s.find("[") + 1 : s.rfind("]")]
        vals = [float(x) for x in s.replace(";", " ").split()]
        # f assumed at (0,0)
        f = vals[0]
        cx = vals[2]
        cy = vals[5]
        return {"f": f, "cx": cx, "cy": cy}

    if "cam0" in calib:
        cam0 = parse_cam(calib["cam0"])
        calib["focal_length"] = cam0["f"]
        calib["cx0"] = cam0["cx"]
        calib["cy"] = cam0["cy"]
    if "baseline" in calib:
        calib["baseline"] = float(calib["baseline"])
    if "doffs" in calib:
        calib["doffs"] = float(calib["doffs"])
    if "ndisp" in calib:
        try:
            calib["ndisp"] = int(calib["ndisp"])
        except:
            pass
    if "width" in calib and "height" in calib:
        calib["width"] = int(calib["width"])
        calib["height"] = int(calib["height"])
    return calib


def read_pfm(filename: str) -> np.ndarray:
    """Load a single-channel PFM disparity (floating point)."""
    with open(filename, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        if header not in ("PF", "Pf"):
            raise ValueError("Not a PFM file.")
        dims = f.readline().decode("utf-8")
        while dims.startswith("#"):
            dims = f.readline().decode("utf-8")
        width, height = map(int, dims.split())
        scale = float(f.readline().decode("utf-8").strip())
        data = np.fromfile(f, "<f" if scale < 0 else ">f", count=width * height)
        data = data.reshape((height, width))
        return data


# -------------------------
# Metrics
# -------------------------
def compute_metrics(
    gt: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray
) -> Dict[str, Any]:
    diff = pred[valid_mask] - gt[valid_mask]
    abs_diff = np.abs(diff)
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff**2)))
    bad1 = float(np.mean(abs_diff > 1.0)) * 100.0
    bad3 = float(np.mean(abs_diff > 3.0)) * 100.0
    valid_ratio = float(np.sum(valid_mask) / valid_mask.size) * 100.0
    return {
        "MAE_px": mae,
        "RMSE_px": rmse,
        "Bad1_%": bad1,
        "Bad3_%": bad3,
        "Valid_%": valid_ratio,
    }


# -------------------------
# Run one scene
# -------------------------
def run_scene(
    left_path: str,
    right_path: str,
    gt_disp_path: str,
    calib_path: str,
    estimator_ctor,
    estimator_kwargs,
) -> Dict[str, Any]:
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)
    if left is None or right is None:
        raise FileNotFoundError("Unable to read images.")
    gt_disp = read_pfm(gt_disp_path)
    calib = read_calib(calib_path)
    # Ensure same sizes
    h, w = gt_disp.shape
    if left.shape[1] != w or left.shape[0] != h:
        # Resize inputs to GT resolution (important)
        left = cv2.resize(left, (w, h), interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, (w, h), interpolation=cv2.INTER_LINEAR)

    estimator = estimator_ctor(**estimator_kwargs)
    t0 = time.perf_counter()
    pred_disp = estimator.calc_disparity((left, right))
    t1 = time.perf_counter()

    # Enforce same shape
    if pred_disp.shape != gt_disp.shape:
        pred_disp = cv2.resize(pred_disp, (w, h), interpolation=cv2.INTER_NEAREST)

    # Valid mask: GT > 0 and finite
    valid_mask = np.isfinite(gt_disp) & (gt_disp > 0)
    metrics = compute_metrics(gt_disp, pred_disp, valid_mask)
    metrics.update(
        {
            "scene": os.path.basename(os.path.dirname(calib_path)),
            "left": left_path,
            "right": right_path,
            "gt_disp": gt_disp_path,
            "runtime_ms": (t1 - t0) * 1000.0,
            "estimator": estimator.__class__.__name__,
            "estimator_params": estimator_kwargs,
        }
    )
    return metrics


# -------------------------
# Helpers to find scenes
# -------------------------
def find_middlebury_scenes(dataset_root: str) -> list:
    # If dataset_root points to a single scene dir, return it
    if os.path.isdir(os.path.join(dataset_root, "im0.png")) or os.path.isdir(
        dataset_root
    ):
        # Check if root contains multiple SCENE* dirs
        scenes = []
        for entry in sorted(os.listdir(dataset_root)):
            full = os.path.join(dataset_root, entry)
            if os.path.isdir(full):
                if os.path.exists(os.path.join(full, "im0.png")):
                    scenes.append(full)
        if not scenes:
            # maybe dataset_root itself is a scene
            if os.path.exists(os.path.join(dataset_root, "im0.png")):
                return [dataset_root]
        return scenes
    return []


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to Middlebury scenes directory (or single scene).",
    )
    parser.add_argument("--estimator", choices=["sgbm", "bm"], default="sgbm")
    parser.add_argument("--mode", choices=["single", "sweep"], default="single")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scenes = find_middlebury_scenes(args.dataset)
    if not scenes:
        raise RuntimeError(
            "No scenes found. Each scene should contain im0.png, im1.png, calib.txt, disp0.pfm"
        )

    trials = []
    # Default single-run params
    if args.estimator == "sgbm":
        Est = SGBMStereoDepthEstimator
        default_params = {
            "num_disparities": 128,
            "block_size": 5,
            "max_depth": 10000.0,
            "wls_filter": False,
        }
    else:
        Est = BMStereoDepthEstimator
        default_params = {
            "num_disparities": 128,
            "block_size": 11,
            "max_depth": 10000.0,
            "wls_filter": False,
        }

    # Optionally do parameter sweep
    sweep_grid = (
        [
            {"num_disparities": 96, "block_size": 3},
            {"num_disparities": 128, "block_size": 5},
            {"num_disparities": 160, "block_size": 7},
        ]
        if args.mode == "sweep"
        else [default_params]
    )

    rows = []
    for params in sweep_grid:
        # validate params
        if params["num_disparities"] % 16 != 0:
            params["num_disparities"] = ((params["num_disparities"] // 16) + 1) * 16
        if params["block_size"] % 2 == 0:
            params["block_size"] += 1
        for scene in scenes:
            left = os.path.join(scene, "im0.png")
            right = os.path.join(scene, "im1.png")
            gt = os.path.join(scene, "disp0.pfm")
            calib = os.path.join(scene, "calib.txt")
            if not (
                os.path.exists(left)
                and os.path.exists(right)
                and os.path.exists(gt)
                and os.path.exists(calib)
            ):
                print(f"Skipping incomplete scene: {scene}")
                continue
            metrics = run_scene(left, right, gt, calib, Est, params)
            rows.append(metrics)
            print(
                f"[{params}] {scene} -> MAE {metrics['MAE_px']:.3f} px, Bad1% {metrics['Bad1_%']:.2f}, runtime {metrics['runtime_ms']:.1f} ms"
            )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out, "results.csv")
    json_path = os.path.join(args.out, "results.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(
            {"summary": df.describe().to_dict(), "raw": df.to_dict(orient="records")},
            f,
            indent=2,
        )
    print("Results written:", csv_path, json_path)


if __name__ == "__main__":
    main()
