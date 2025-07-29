#!/usr/bin/env python3
"""
heel_raise_analysis_26.py  🦶
============================
Detect per-foot heel raise during squat repetitions (HALPE-26 keypoints).

Assumptions
-----------
* Keypoints are **unit-square normalised**: x, y ∈ [0, 1].
* y-axis may point up (Cartesian) or down (image). We auto-detect sign.

Public API
----------
analyze_heel_raise_report(...)
pipeline(...)                    # convenience – returns a DataFrame

Change log (2025-07-06)
-----------------------
* Threshold is now unit-scale (default 0.02) instead of 15 px.
* Repetition tables may use either ['start','mid','end'] or
  ['rep_start','rep_mid','rep_end'].
* `analyze_heel_raise_report` accepts a DataFrame or CSV path.
"""
from __future__ import annotations

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
ANKLES = (LEFT_ANKLE, RIGHT_ANKLE)
HEELS  = (LEFT_HEEL,   RIGHT_HEEL)

# -----------------------------------------------------------------------------
# Helpers
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


# -----------------------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------------------
def analyze_heel_raise_report(
    keypoints: np.ndarray,
    reps: str | Path | pd.DataFrame,
    *,
    threshold: float = 0.01,
    conf_thresh: float = 0.00,
) -> list[dict]:
    """
    Detect heel raise for each repetition.

    Parameters
    ----------
    keypoints : ndarray (F, 26, 3) – (x, y, conf) in unit square
    reps      : CSV path or DataFrame with rep boundaries
    threshold : float (unit coords)  – raise distance to flag a lift
    conf_thresh : float             – minimum kp confidence

    Returns
    -------
    list of dicts with keys: rep_id, lifted, frames
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
# CLI utility
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect per-foot heel raise and write a detailed CSV report."
    )
    parser.add_argument("--keypoints", type=Path, default="imputed_ma.npy",
                        help="NumPy .npy file with (F,26,3) keypoints in unit coords")
    parser.add_argument("--reps",      type=Path, default="repetition_data.csv",
                        help="CSV with rep_id,start,mid,end (or rep_* variants)")
    parser.add_argument("--output",    type=Path, default="heel_raise_report.csv")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Lift distance threshold in unit coordinates (default 0.02)")
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


# -----------------------------------------------------------------------------
# ❶ Convenience wrapper – in-memory pipeline
# -----------------------------------------------------------------------------
def pipeline(
    keypoints: str | Path | np.ndarray,
    reps: str | Path | pd.DataFrame,
    *,
    threshold: float = 0.01,
    conf_thresh: float = 0.0,
) -> pd.DataFrame:
    """
    Run the heel-raise analysis end-to-end and return a DataFrame.
    """
    # 1) keypoints
    kps = np.load(keypoints) if isinstance(keypoints, (str, Path)) else keypoints

    # 2) reps table, column normalisation handled in analyze_… function
    reps_df = (
        reps.copy() if isinstance(reps, pd.DataFrame) else pd.read_csv(reps)
    )

    # 3) analysis
    records = analyze_heel_raise_report(
        kps, reps_df, threshold=threshold, conf_thresh=conf_thresh
    )

    # 4) DataFrame
    return pd.DataFrame(records)


if __name__ == "__main__":
    main()
