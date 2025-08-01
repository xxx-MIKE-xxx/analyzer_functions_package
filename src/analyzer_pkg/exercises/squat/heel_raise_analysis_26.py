#!/usr/bin/env python3
"""
heel_raise_analysis_26.py  🦶
============================
Detect per-foot heel raise during squat repetitions (HALPE-26 keypoints).

Supports both: (1) distance-based (Y-coordinate) and (2) angle-based detection.

Assumptions
-----------
* Keypoints are **unit-square normalised**: x, y ∈ [0, 1].
* y-axis may point up (Cartesian) or down (image). We auto-detect sign.

Public API
----------
analyze_heel_raise_report(...)   # (distance)
pipeline(...)                    # (distance, DataFrame)
angle(...)                       # (angle, DataFrame)
plot_heel_raise(...)             # (plot Y)
plot_heel_angle(...)             # (plot angle)

Change log (2025-08-01)
-----------------------
* Added angle-based API: angle(...) and plot_heel_angle(...).
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# HALPE-26 keypoint indices
LEFT_ANKLE, RIGHT_ANKLE = 15, 16
LEFT_HEEL,   RIGHT_HEEL = 24, 25
LEFT_TOE,    RIGHT_TOE  = 22, 23
ANKLES = (LEFT_ANKLE, RIGHT_ANKLE)
HEELS  = (LEFT_HEEL,   RIGHT_HEEL)
TOES   = (LEFT_TOE,    RIGHT_TOE)

# Angle-based threshold (deg)
ANGLE_THRESHOLD = 15.0

# -----------------------------------------------------------------------------
# Helpers (shared)
# -----------------------------------------------------------------------------
def _median_pos(
    data: np.ndarray,
    frames: Sequence[int],
    idx: int,
    conf_thresh: float = 0.0,
) -> np.ndarray:
    """Median (x, y) of keypoint *idx* across *frames* (skip low-conf pts)."""
    coords = [
        data[f, idx, :2]
        for f in frames
        if data[f, idx, 2] >= conf_thresh
    ]
    if not coords:
        return np.array([np.nan, np.nan])
    arr = np.array(coords)
    return np.median(arr, axis=0)

def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle in degrees between two 2D vectors."""
    if np.linalg.norm(u) < 1e-8 or np.linalg.norm(v) < 1e-8:
        return np.nan
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))

def _heel_angle_per_frame(
    kps: np.ndarray,
    resting_heel: np.ndarray,
    toe: np.ndarray,
    frames: Sequence[int],
    heel_idx: int,
    toe_idx: int,
    conf_thresh: float = 0.0,
):
    """
    For each frame, compute angle between resting_heel→toe and curr_heel→toe.
    Returns: (angles (deg), valid mask)
    """
    angles = []
    valids = []
    for f in frames:
        if kps[f, heel_idx, 2] < conf_thresh or kps[f, toe_idx, 2] < conf_thresh:
            angles.append(np.nan)
            valids.append(False)
            continue
        curr_heel = kps[f, heel_idx, :2]
        curr_toe  = kps[f, toe_idx, :2]
        v0 = toe - resting_heel
        v1 = curr_toe - curr_heel
        angle = _angle_between(v0, v1)
        angles.append(angle)
        valids.append(True)
    return np.array(angles), np.array(valids)

# -----------------------------------------------------------------------------
# Distance-based detection (original)
# -----------------------------------------------------------------------------
def _median_y(
    data: np.ndarray,
    frames: Sequence[int],
    idx: int,
    conf_thresh: float = 0.0,
) -> float:
    """Median y of keypoint *idx* across *frames* (skip low-conf pts)."""
    coords = [
        data[f, idx, 1]
        for f in frames
        if data[f, idx, 2] >= conf_thresh
    ]
    return float("nan") if not coords else float(np.median(coords))

def plot_heel_raise(
    keypoints: str | Path | np.ndarray,
    out_path: str | Path,
    *,
    left_heel_idx: int = 24,
    right_heel_idx: int = 25,
    y_threshold: float = 0.01,
) -> None:
    """
    Saves a plot of left/right heel Y-coordinates over all frames to out_path (PNG).
    """
    kps = np.load(keypoints) if isinstance(keypoints, (str, Path)) else keypoints
    F = kps.shape[0]

    left_heel_y  = kps[:, left_heel_idx, 1]
    right_heel_y = kps[:, right_heel_idx, 1]
    frames = np.arange(F)

    plt.figure(figsize=(12, 5))
    plt.plot(frames, left_heel_y,  label="Left Heel",  color="blue")
    plt.plot(frames, right_heel_y, label="Right Heel", color="red")
    plt.axhline(y_threshold, color="gray", ls="--", lw=1, label=f"Y threshold ({y_threshold})")
    plt.xlabel("Frame")
    plt.ylabel("Heel Y position (unit square)")
    plt.title("Heel Y-coordinates over time")
    plt.legend()
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"📉  Saved heel Y plot → {out_path}")

def _first_lift_frame(
    data: np.ndarray,
    baseline_y: float,
    frames: Sequence[int],
    kp_indices: Sequence[int],
    *,
    conf_thresh: float,
    threshold: float,
    sign: int,
) -> int:
    """
    Return first frame in *frames* where any kp in *kp_indices*
    rises above the baseline by > threshold (after sign correction).
    Returns −1 if no lift is detected.
    """
    for f in frames:
        for kp in kp_indices:
            if data[f, kp, 2] < conf_thresh:
                continue
            diff = sign * (baseline_y - data[f, kp, 1])
            if diff > threshold:
                return f
    return -1

def analyze_heel_raise_report(
    keypoints: np.ndarray,
    reps: str | Path | pd.DataFrame,
    *,
    threshold: float = 0.01,
    conf_thresh: float = 0.00,
) -> list[dict]:
    """
    Detect heel raise for each repetition. (Distance-based)
    """
    # --- load reps table --------------------------------------------------
    if isinstance(reps, pd.DataFrame):
        reps_df = reps.copy()
    else:
        reps_df = pd.read_csv(reps)

    # Normalise column names if needed
    if {"rep_start", "rep_mid", "rep_end"}.issubset(reps_df.columns):
        reps_df = reps_df.rename(
            columns={"rep_start": "start", "rep_mid": "mid", "rep_end": "end"}
        )
    required = {"rep_id", "start", "mid", "end"}
    if not required.issubset(reps_df.columns):
        raise ValueError(f"Reps table must contain {sorted(required)}")

    F = keypoints.shape[0]
    records: list[dict] = []

    # Detect y-axis orientation: hip descends at mid in image coords
    sample = reps_df.iloc[0]
    hip_idx = 11  # L-hip
    down = keypoints[sample.start, hip_idx, 1] < keypoints[sample.mid, hip_idx, 1]
    sign = -1 if not down else 1  # sign = +1 if y grows downwards

    # --- iterate reps -----------------------------------------------------
    for _, row in reps_df.iterrows():
        rep_id, start, mid, end = map(int, (row.rep_id, row.start, row.mid, row.end))

        # Baseline y: ankles in 5 frames before start + 5 after end
        before = list(range(max(0, start - 5), start))
        after  = list(range(end, min(F, end + 6)))
        baseline_frames = before + after

        base_left  = _median_y(keypoints, baseline_frames, LEFT_ANKLE,  conf_thresh)
        base_right = _median_y(keypoints, baseline_frames, RIGHT_ANKLE, conf_thresh)

        # Only inspect the exact mid frame (list for API compatibility)
        mid_frames = [mid]

        lf = _first_lift_frame(
            keypoints, base_left, mid_frames, (LEFT_ANKLE, LEFT_HEEL),
            conf_thresh=conf_thresh, threshold=threshold, sign=sign
        )
        rf = _first_lift_frame(
            keypoints, base_right, mid_frames, (RIGHT_ANKLE, RIGHT_HEEL),
            conf_thresh=conf_thresh, threshold=threshold, sign=sign
        )

        records.append(
            {
                "rep_id": rep_id,
                "lifted": json.dumps([lf != -1, rf != -1]),
                "frames": json.dumps([lf, rf]),
            }
        )

    return records

# -----------------------------------------------------------------------------
# Main distance-based pipeline
# -----------------------------------------------------------------------------
def pipeline(
    keypoints: str | Path | np.ndarray,
    reps: str | Path | pd.DataFrame,
    *,
    threshold: float = 0.01,
    conf_thresh: float = 0.0,
) -> pd.DataFrame:
    """
    Run the heel-raise analysis (distance-based) end-to-end and return a DataFrame.
    """
    kps = np.load(keypoints) if isinstance(keypoints, (str, Path)) else keypoints
    reps_df = reps.copy() if isinstance(reps, pd.DataFrame) else pd.read_csv(reps)
    records = analyze_heel_raise_report(
        kps, reps_df, threshold=threshold, conf_thresh=conf_thresh
    )
    return pd.DataFrame(records)

# -----------------------------------------------------------------------------
# Angle-based detection
# -----------------------------------------------------------------------------
def angle(
    keypoints: str | Path | np.ndarray,
    reps: str | Path | pd.DataFrame,
    *,
    angle_threshold: float = ANGLE_THRESHOLD,
    conf_thresh: float = 0.0,
) -> pd.DataFrame:
    """
    Detect per-foot heel raise based on heel angle for each repetition.

    For each rep:
        1. Compute resting heel and toe position (median over baseline frames).
        2. For all mid frames, compute angle between resting_heel→toe and curr_heel→toe.
        3. If angle > angle_threshold, flag as raised.

    Returns a DataFrame with rep_id, left_angle_deg, right_angle_deg, left_raised, right_raised.
    """
    kps = np.load(keypoints) if isinstance(keypoints, (str, Path)) else keypoints
    reps_df = reps.copy() if isinstance(reps, pd.DataFrame) else pd.read_csv(reps)
    if {"rep_start", "rep_mid", "rep_end"}.issubset(reps_df.columns):
        reps_df = reps_df.rename(
            columns={"rep_start": "start", "rep_mid": "mid", "rep_end": "end"}
        )
    required = {"rep_id", "start", "mid", "end"}
    if not required.issubset(reps_df.columns):
        raise ValueError(f"Reps table must contain {sorted(required)}")

    F = kps.shape[0]
    records: list[dict] = []

    for _, row in reps_df.iterrows():
        rep_id, start, mid, end = map(int, (row.rep_id, row.start, row.mid, row.end))
        before = list(range(max(0, start - 5), start))
        after  = list(range(end, min(F, end + 6)))
        baseline_frames = before + after

        # Resting heel and toe positions
        left_rest_heel  = _median_pos(kps, baseline_frames, LEFT_HEEL,  conf_thresh)
        left_rest_toe   = _median_pos(kps, baseline_frames, LEFT_TOE,   conf_thresh)
        right_rest_heel = _median_pos(kps, baseline_frames, RIGHT_HEEL, conf_thresh)
        right_rest_toe  = _median_pos(kps, baseline_frames, RIGHT_TOE,  conf_thresh)

        # Mid-frames: usually [mid], but you could expand to window if desired
        mid_frames = [mid]

        left_angles, left_valids   = _heel_angle_per_frame(
            kps, left_rest_heel, left_rest_toe, mid_frames, LEFT_HEEL, LEFT_TOE, conf_thresh)
        right_angles, right_valids = _heel_angle_per_frame(
            kps, right_rest_heel, right_rest_toe, mid_frames, RIGHT_HEEL, RIGHT_TOE, conf_thresh)

        left_raised  = bool(left_valids[0] and left_angles[0] > angle_threshold)
        right_raised = bool(right_valids[0] and right_angles[0] > angle_threshold)

        records.append({
            "rep_id": rep_id,
            "left_angle_deg": left_angles[0],
            "right_angle_deg": right_angles[0],
            "left_raised": left_raised,
            "right_raised": right_raised,
        })
    return pd.DataFrame(records)

def plot_heel_angle(
    keypoints: str | Path | np.ndarray,
    out_path: str | Path,
    *,
    conf_thresh: float = 0.0,
    angle_threshold: float = ANGLE_THRESHOLD,
    reps: str | Path | pd.DataFrame = None
):
    """
    Plot left/right heel raise angle over all frames.
    Uses the *resting* position from all frames before/after reps.
    If `reps` is provided, resting position will be median over before/after frames of *all* reps.
    """
    kps = np.load(keypoints) if isinstance(keypoints, (str, Path)) else keypoints
    F = kps.shape[0]
    frames = np.arange(F)

    if reps is not None:
        reps_df = reps.copy() if isinstance(reps, pd.DataFrame) else pd.read_csv(reps)
        if {"rep_start", "rep_mid", "rep_end"}.issubset(reps_df.columns):
            reps_df = reps_df.rename(
                columns={"rep_start": "start", "rep_mid": "mid", "rep_end": "end"}
            )
        required = {"rep_id", "start", "mid", "end"}
        if not required.issubset(reps_df.columns):
            raise ValueError(f"Reps table must contain {sorted(required)}")
        baseline_frames = []
        for _, row in reps_df.iterrows():
            start, end = int(row.start), int(row.end)
            before = list(range(max(0, start - 5), start))
            after  = list(range(end, min(F, end + 6)))
            baseline_frames += before + after
        baseline_frames = sorted(set(baseline_frames))
    else:
        baseline_frames = list(range(F))

    left_rest_heel  = _median_pos(kps, baseline_frames, LEFT_HEEL,  conf_thresh)
    left_rest_toe   = _median_pos(kps, baseline_frames, LEFT_TOE,   conf_thresh)
    right_rest_heel = _median_pos(kps, baseline_frames, RIGHT_HEEL, conf_thresh)
    right_rest_toe  = _median_pos(kps, baseline_frames, RIGHT_TOE,  conf_thresh)

    left_angles,  _ = _heel_angle_per_frame(kps, left_rest_heel,  left_rest_toe,  frames, LEFT_HEEL,  LEFT_TOE,  conf_thresh)
    right_angles, _ = _heel_angle_per_frame(kps, right_rest_heel, right_rest_toe, frames, RIGHT_HEEL, RIGHT_TOE, conf_thresh)

    plt.figure(figsize=(12, 5))
    plt.plot(frames, left_angles,  label="Left heel raise angle (deg)", color="blue")
    if np.all(np.isnan(right_angles)):
        print("⚠️ No valid right heel raise angle data for right foot!")
    else:
        plt.plot(frames, right_angles, label="Right heel raise angle (deg)", color="red")
    plt.axhline(angle_threshold, color="gray", ls="--", lw=1, label=f"Angle threshold ({angle_threshold}°)")
    plt.xlabel("Frame")
    plt.plot(frames, right_angles, label="Right heel raise angle (deg)", color="red", alpha=0.85)
    plt.plot(frames, left_angles, label="Left heel raise angle (deg)", color="blue", alpha=0.85)
    plt.ylabel("Heel raise angle (degrees)")
    plt.title("Heel Raise Angle per Frame (vs resting heel-toe)")
    plt.plot(frames, np.where(np.isnan(right_angles), -10, right_angles), 'r-', label="Right heel raise angle (deg)")
    plt.legend()
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"📉  Saved heel raise angle plot → {out_path}")

# -----------------------------------------------------------------------------
# CLI utility (distance-based only)
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect per-foot heel raise and write a detailed CSV report (distance-based)."
    )
    parser.add_argument("--keypoints", type=Path, default="imputed_ma.npy",
                        help="NumPy .npy file with (F,26,3) keypoints in unit coords")
    parser.add_argument("--reps",      type=Path, default="repetition_data.csv",
                        help="CSV with rep_id,start,mid,end (or rep_* variants)")
    parser.add_argument("--output",    type=Path, default="heel_raise_report.csv")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Lift distance threshold in unit coordinates (default 0.01)")
    parser.add_argument("--conf",      type=float, default=0.0,
                        help="Minimum keypoint confidence")
    args = parser.parse_args()

    kps  = np.load(args.keypoints)
    reps = pd.read_csv(args.reps)

    records = analyze_heel_raise_report(
        kps, reps, threshold=args.threshold, conf_thresh=args.conf
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rep_id", "lifted", "frames"])
        writer.writeheader()
        writer.writerows(records)

    print(f"Report written → {args.output.resolve()}")

if __name__ == "__main__":
    main()
