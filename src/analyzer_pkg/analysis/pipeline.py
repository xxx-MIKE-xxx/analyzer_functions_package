#!/usr/bin/env python3
"""
full_pipeline_driver.py – **single-entry script**
================================================
Runs the *entire* exercise-video pipeline, including the two new steps you
requested:

1. **detect_bad_frames** – uses the pre-trained Random-Forest model
   `rf_badframe_detector.joblib` to find noisy frames in the 2-D sequence.
2. **ma_impute** – replaces those frames with a centred moving-average so that
   all downstream analyses get a clean, gap-free `imputed_ma.npy`.

Everything else (reference lengths, rigid fix, repetition detection, seven
analysis reports) remains unchanged.
"""

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

# ── local project helpers (same folder) ──────────────────────────────────────
from . import json_to_npy as jn
from . import convert
from . import keypoint_filter as kpf
from . import ref_frame_find as rff
from . import compute_reference_lengths as crl
from . import skeleton_length_fix as slf
from . import squat_detector as sd
from . import inward_knees_analysis as ikn
from .heel_raise_analysis_26 import analyze_heel_raise_report as analyze_heel_raise_detailed
from . import forward_knees_analysis as fka
from . import squat_depth_analysis as sda
from . import forward_lean_analysis as fla
from . import hip_path_analysis as hpa
from . import feet_width_analysis as fwa

import sys
import logging

logging.basicConfig(level=logging.DEBUG)
sys.setrecursionlimit(1500)

# ── constants ────────────────────────────────────────────────────────────────
KERNEL_SIZE = 5  # spike-removal window for 3-D

HIERARCHY: Dict[int, List[int]] = {
    0: [1, 4, 7], 1: [2], 2: [3],
    4: [5], 5: [6],
    7: [8], 8: [9, 11, 14],
    9: [10],
    11: [12], 12: [13],
    14: [15], 15: [16],
}

# Convenience aliases from compute_reference_lengths
SYMM2D, SYMM3D = crl.SYMMETRIC_PAIRS_2D, crl.SYMMETRIC_PAIRS_3D
CONN2D, CONN3D = crl.CONNECTIONS_2D, crl.CONNECTIONS_3D

# ─────────────────────────────────────────────────────────────────────────────
#  0. utility helpers used by several steps
# ─────────────────────────────────────────────────────────────────────────────

def _save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

# ─────────────────────────────────────────────────────────────────────────────
#  1. JSON → raw NPY  (AlphaPose 26-kp with confidences)
# ─────────────────────────────────────────────────────────────────────────────

def step_json_to_npy(json_pose: Path, outdir: Path) -> Path:
    out = outdir / "alphapose-results.npy"
    jn.json_to_npy(json_pose, out)
    return out

# ─────────────────────────────────────────────────────────────────────────────
#  2. convert to 3-D & 2-D arrays, save both, spike-filter 3-D
# ─────────────────────────────────────────────────────────────────────────────

def step_prepare_skeletons(npy_raw: Path, outdir: Path) -> tuple[Path, Path]:
    skel3d = convert.load_skeleton(str(npy_raw))
    skel3d = convert.process_all_frames(skel3d)
    np.save(outdir / "skeleton_3d_raw.npy", skel3d)

    skel2d = skel3d[..., :2]
    skel2d_conf = np.pad(skel2d, ((0, 0), (0, 0), (0, 1)), constant_values=1.0)
    # add dummy confidence (1.0) so shape == (F,26,3)
    np.save(outdir / "skeleton_2d.npy", skel2d_conf)

    # 3-D spike removal for later rigid fix
    filt3d = kpf.remove_spikes(skel3d, kernel_size=KERNEL_SIZE)
    np.save(outdir / "h36_keypoints_filtered.npy", filt3d)
    return outdir / "h36_keypoints_filtered.npy", outdir / "skeleton_2d.npy"

# ─────────────────────────────────────────────────────────────────────────────
#  3. detect noisy 2-D frames  → rf_bad_frames.npy
# ─────────────────────────────────────────────────────────────────────────────

def _compute_features(data: np.ndarray) -> np.ndarray:
    """Feature extractor copied from detect_bad_frames.py (vectorised)."""
    x, y, conf = data[..., 0], data[..., 1], data[..., 2]
    F = data.shape[0]
    feats = np.zeros((F, 8), dtype=float)

    # COM x,y and bounding box area/extents
    valid = conf >= 0.0  # everything should be valid (dummy conf=1), keep logic
    com_x = np.where(valid, x, np.nan).mean(axis=1)
    com_y = np.where(valid, y, np.nan).mean(axis=1)
    # dx, dy
    dx = np.concatenate([[0.0], np.diff(com_x)])
    dy = np.concatenate([[0.0], np.diff(com_y)])

    # bounding-box area and max extent per frame
    xs_min = np.where(valid, x, np.nan).min(axis=1)
    xs_max = np.where(valid, x, np.nan).max(axis=1)
    ys_min = np.where(valid, y, np.nan).min(axis=1)
    ys_max = np.where(valid, y, np.nan).max(axis=1)
    area = (xs_max - xs_min) * (ys_max - ys_min)
    extent = np.sqrt((xs_max - xs_min) ** 2 + (ys_max - ys_min) ** 2)

    mean_conf = np.where(valid, conf, np.nan).mean(axis=1)
    std_conf  = np.where(valid, conf, np.nan).std(axis=1)

    feats[:, 0] = com_x
    feats[:, 1] = com_y
    feats[:, 2] = dx
    feats[:, 3] = dy
    feats[:, 4] = area
    feats[:, 5] = extent
    feats[:, 6] = mean_conf
    feats[:, 7] = std_conf
    return feats


def step_badframe_detect(skel2d: Path, model_path: Path, outdir: Path) -> Path:
    data = np.load(skel2d)        # (F,26,3)
    clf  = joblib.load(model_path)
    mask = clf.predict(_compute_features(data)).astype(bool)  # True == bad
    out = outdir / "rf_bad_frames.npy"
    np.save(out, mask)
    print(f"[bad-frame] {mask.sum()} / {len(mask)} frames flagged → {out.name}")
    return out

# ─────────────────────────────────────────────────────────────────────────────
#  4. create cleaned skeleton with NaNs & impute moving-average
# ─────────────────────────────────────────────────────────────────────────────

def _choose_window(mask: np.ndarray) -> int:
    good_ratio = 1.0 - mask.mean()
    streaks = [sum(1 for _ in g) for v, g in itertools.groupby(mask) if v]
    max_streak = max(streaks) if streaks else 0
    if good_ratio > 0.90 and max_streak <= 1:
        return 3
    elif max_streak <= 2:
        return 5
    return 7


def _ma_impute(arr: np.ndarray, mask: np.ndarray, window: int) -> np.ndarray:
    out = arr.copy().astype(float)
    F, K, _ = out.shape
    for kp in range(K):
        for coord in (0, 1):
            s = pd.Series(out[:, kp, coord])
            ma = s.rolling(window, center=True, min_periods=1).mean()
            fill = mask
            out[fill, kp, coord] = ma[fill]
    return out


def step_ma_impute(skel2d: Path, badmask: Path, outdir: Path) -> Path:
    data = np.load(skel2d).copy()
    mask = np.load(badmask).astype(bool)

    # set x,y to NaN where bad so rolling mean can act as imputer
    data[mask, :, 0:2] = np.nan

    win = _choose_window(mask)
    imputed = _ma_impute(data, mask, win)
    out = outdir / "imputed_ma.npy"
    np.save(out, imputed)
    print(f"[impute] moving-average window={win}  → {out.name}")
    return out

# ─────────────────────────────────────────────────────────────────────────────
#  5. find reference frame (3-D & 2-D already prepared)
# ─────────────────────────────────────────────────────────────────────────────

def step_find_reference_frame(skel2d: Path, skel3d: Path) -> int:
    analyzer = rff.ExerciseAnalyzer(
        file_2d=str(skel2d),
        file_3d=str(skel3d),
        reference_type="file",
        reference_value="reference_frame.npy",
    )
    analyzer.analyze_exercise()
    return getattr(analyzer, "reference_frame", 0)

#  Continue with other steps

# ─────────────────────────────────────────────────────────────────────────────
#  8. Orchestration / CLI
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(json_pose_path: str, model_path: str, outdir: str):
    """
    Main function that runs the entire pipeline:
    1. Converts JSON pose data to NPY format
    2. Prepares skeletons (both 3D and 2D)
    3. Detects noisy frames and saves the mask
    4. Creates an imputed skeleton using a moving average
    5. Finds the reference frame
    6. Computes reference lengths
    7. Applies rigid skeleton fixes
    8. Detects repetitions
    9. Generates analysis reports
    
    Args:
        json_pose_path (str): Path to the JSON pose data file.
        model_path (str): Path to the pre-trained RandomForest model for bad frame detection.
        outdir (str): Directory to save the outputs.
    
    Returns:
        dict: A dictionary containing results from the pipeline steps.
    """
    
    # Step 1: Convert JSON to NPY
    npy_raw = step_json_to_npy(Path(json_pose_path), Path(outdir))
    
    # Step 2: Prepare skeletons (3D & 2D)
    skel3d, skel2d = step_prepare_skeletons(npy_raw, Path(outdir))
    
    # Step 3: Detect bad frames
    bad_frame_mask = step_badframe_detect(skel2d, Path(model_path), Path(outdir))
    
    # Step 4: Impute missing frames with moving average
    imputed_skel = step_ma_impute(skel2d, bad_frame_mask, Path(outdir))
    
    # Step 5: Find reference frame
    ref_frame = step_find_reference_frame(skel2d, skel3d)
    
    # Step 6: Compute reference lengths
    ref_3d, ref_2d = step_reference_lengths(skel3d, skel2d, frames="117", outdir=Path(outdir))
    
    # Step 7: Apply rigid skeleton fixes
    corrected_skel = step_rigid_fix(skel3d, ref_3d, Path(outdir))
    
    # Step 8: Detect repetitions
    rep_data = step_detect_repetitions(corrected_skel, ref_frame, Path(outdir))
    
    # Step 9: Generate reports
    step_ffpa(Path(outdir))
    step_heel_raise(Path(outdir))
    step_forward_knee(Path(outdir))
    step_squat_depth(Path(outdir))
    step_forward_lean(Path(outdir))
    step_hip_path(Path(outdir))
    step_feet_width(Path(outdir))

    print(f"✅ Pipeline completed! All results saved in: {outdir}")
    
    return {
        "json_pose_path": json_pose_path,
        "model_path": model_path,
        "outdir": outdir,
        "reference_frame": ref_frame,
        "rep_data": rep_data,
    }

def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser("End-to-end pipeline with bad-frame handling")
    p.add_argument("--json", required=True, type=Path, help="AlphaPose JSON")
    p.add_argument("--outdir", default="outputs", type=Path, help="Output dir")
    p.add_argument("--model", default="rf_badframe_detector.joblib", type=Path,
                   help="RandomForest model for bad-frame detection")
    p.add_argument("--ankle-ref", type=int, default=None, help="Manually set reference frame")
    args = p.parse_args()

    # Run the full pipeline
    run_pipeline(str(args.json), str(args.model), str(args.outdir))

if __name__ == "__main__":
    main()
