#!/usr/bin/env python3
"""
feet_width_analysis.py  👣
=========================

Measure stance width for each squat repetition.

For every frame f in a rep:
    feet_dist = ‖ankle_L – ankle_R‖₂
    hip_dist  = ‖hip_L   – hip_R‖₂
    ratio_f   = feet_dist / hip_dist

For the rep we keep the **smallest** ratio (narrowest stance).

Severity (too-narrow only) – *scale-independent*:
    ratio < 0.90 → "severe"
    0.90 ≤ ratio < 1.00 → "mild"
    ≥ 1.00 → "none"

All coordinates are assumed to be unit-square (0–1) but
the metric is a **ratio**, so thresholds stay unchanged.

Output columns
--------------
rep_id, severity, frame, ratio
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ HALPE-26 indices
L_HIP, R_HIP = 11, 12
L_ANK, R_ANK = 15, 16

# ------------------------------------------------------------------ helpers
def dist2d(pt1: np.ndarray, pt2: np.ndarray) -> float:
    """Euclidean distance in 2-D; returns nan if any NaN present."""
    if np.isnan(pt1).any() or np.isnan(pt2).any():
        return float("nan")
    return float(np.linalg.norm(pt1 - pt2))

def severity(r: float) -> str:
    if math.isnan(r):
        return "unknown"
    if r < 0.90:
        return "severe"
    if r < 1.00:
        return "mild"
    return "none"

# ------------------------------------------------------------------ core
def generate_report(
    kps: np.ndarray,
    reps: pd.DataFrame,
    *,
    out_csv: str | None = "feet_width_report.csv",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    kps      : ndarray (F,26,3)
    reps     : DataFrame with ['rep_id','start','end']
    out_csv  : None → don’t write; otherwise path for CSV
    """
    records: List[Dict[str, Union[int, float, str]]] = []

    for _, row in reps.iterrows():
        start, end = int(row.start), int(row.end)

        ratios: List[float] = []
        for f in range(start, end + 1):
            feet_d = dist2d(kps[f, L_ANK, :2], kps[f, R_ANK, :2])
            hip_d  = dist2d(kps[f, L_HIP, :2], kps[f, R_HIP, :2])
            ratios.append(feet_d / hip_d if hip_d > 0 else float("nan"))

        if ratios:
            arr = np.array(ratios)
            idx_min = int(np.nanargmin(arr))
            ratio_min = float(arr[idx_min])
            frame_min = start + idx_min
        else:
            ratio_min, frame_min = float("nan"), -1

        records.append({
            "rep_id":   int(row.rep_id),
            "severity": severity(ratio_min),
            "frame":    frame_min,
            "ratio":    round(ratio_min, 3) if not math.isnan(ratio_min) else float("nan"),
        })

    df = pd.DataFrame(records)

    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"✅ feet-width report → {os.path.abspath(out_csv)}")

    return df

# ------------------------------------------------------------------ pipeline helper
def pipeline(
    keypoints: str | Path | np.ndarray,
    reps: str | Path | pd.DataFrame,
    *,
    out_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    In-memory pipeline; writes CSV only if `out_csv` is provided.
    """
    # 1) keypoints
    kps = np.load(keypoints) if isinstance(keypoints, (str, Path)) else keypoints

    # 2) reps
    reps_df = reps.copy() if isinstance(reps, pd.DataFrame) else pd.read_csv(reps)
    if {"rep_start", "rep_end"}.issubset(reps_df.columns):
        reps_df = reps_df.rename(columns={"rep_start": "start", "rep_end": "end"})

    # 3) compute & optional write
    return generate_report(kps, reps_df, out_csv=str(out_csv) if out_csv else None)

# ------------------------------------------------------------------ CLI
def main() -> None:                # pragma: no cover
    ap = argparse.ArgumentParser("Feet-width vs hip-width report")
    ap.add_argument("--keypoints", type=Path, required=True)
    ap.add_argument("--reps",      type=Path, required=True)
    ap.add_argument("--output",    type=Path, default="feet_width_report.csv")
    args = ap.parse_args()

    kps  = np.load(args.keypoints)
    reps = pd.read_csv(args.reps)
    generate_report(kps, reps, out_csv=str(args.output))

if __name__ == "__main__":
    main()
