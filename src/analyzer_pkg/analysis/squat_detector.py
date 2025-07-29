#!/usr/bin/env python3
"""
squat_detector.py – 2-D hip-height repetition detector
------------------------------------------------------

Assumes unit-square key-points with (0,0)=top-left  (y grows downwards).

Public API
----------
detect_repetitions_threshold(...)   # low-level core
pipeline(...)                       # convenience → pandas.DataFrame
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Sequence, List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def detect_repetitions_threshold(
    hip_y: Sequence[float],
    hip_y_ref: float,
    start_frame: int,
    end_frame: int,
    *,
    standing_ratio: float = 0.70,
    depth_ratio:    float = 0.30,
    min_distance:   int   = 3,
    prominence:     float = 1e-4,
) -> Tuple[List[Dict], float, float]:
    """
    Detect squat reps by looking for ‘valleys’ in hip-Y.

    Returns
    -------
    reps          : list of dicts {rep_id, rep_start, rep_mid, rep_end}
    depth_thresh  : float  (in depth units)
    stand_thresh  : float  (in depth units)
    """
    # ❶ Build a ‘depth’ signal: 0 at standing frame; positive = deeper
    depth = hip_y - hip_y_ref
    seg   = depth[start_frame : end_frame + 1]

    # ❷ Compute thresholds from the empirical max depth
    max_d = seg.max() if seg.size>0 else 0.0
    depth_thresh = max_d * depth_ratio
    stand_thresh = max_d * standing_ratio

    # ❸ Find candidate mids = peaks of ‘depth’
    cand_mid, _ = find_peaks(
        seg,
        distance=min_distance,
        prominence=prominence
    )
    deep_mid = [m for m in cand_mid if seg[m] >= depth_thresh]
    if not deep_mid:
        return [], depth_thresh, stand_thresh

    # ❹ Enforce stand-up between dips
    deep_mid.sort()
    valid_mid = [deep_mid[0]]
    for m in deep_mid[1:]:
        prev = valid_mid[-1]
        if seg[prev:m+1].min() <= stand_thresh:
            valid_mid.append(m)

    # ❺ Locate boundaries and filter short dips
    reps = []
    rep_id = 1
    for mid in valid_mid:
        # find start
        s = mid
        while s > 0 and seg[s] >= depth_thresh:
            s -= 1
        s += 1
        # find end (½ stand threshold)
        e = mid
        half_stand = stand_thresh * 0.5
        while e < len(seg)-1 and seg[e] > half_stand:
            e += 1
        # skip too-short
        if (e - s + 1) < 5:
            continue
        reps.append({
            "rep_id":    rep_id,
            "rep_start": s + start_frame,
            "rep_mid":   mid + start_frame,
            "rep_end":   e + start_frame,
        })
        rep_id += 1

    return reps, depth_thresh, stand_thresh


def pipeline(
    keypoints: str | Path | np.ndarray,
    *,
    ankle_ref_frame: int | None = None,
    right_hip_idx:   int = 11,
    left_hip_idx:    int = 12,
    start_frame:     int | None = None,
    end_frame:       int | None = None,
    standing_ratio:  float = 0.70,
    depth_ratio:     float = 0.30,
    min_distance:    int   = 3,
    prominence:      float = 1e-4,
) -> pd.DataFrame:
    """
    Convenience wrapper: returns DataFrame with columns
    ['rep_id','rep_start','rep_mid','rep_end'] even if no reps.

    Accepts optional ankle_ref_frame for backward compatibility:
    if provided and start_frame=None, uses that as start_frame.
    """
    # 1) load ndarray
    if isinstance(keypoints, (str, Path)):
        keypoints = np.load(keypoints)
    F = keypoints.shape[0]

    # 2) determine start/end
    if ankle_ref_frame is not None and start_frame is None:
        start_frame = ankle_ref_frame
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = F - 1
    if not (0 <= start_frame <= end_frame < F):
        raise ValueError("Invalid start_frame/end_frame")

    # 3) build hip_Y
    hip_y = (
        keypoints[:, right_hip_idx, 1] +
        keypoints[:, left_hip_idx, 1]
    ) / 2.0
    hip_y_ref = hip_y[start_frame]

    # 4) detect
    reps, depth_thr, stand_thr = detect_repetitions_threshold(
        hip_y, hip_y_ref,
        start_frame, end_frame,
        standing_ratio=standing_ratio,
        depth_ratio=depth_ratio,
        min_distance=min_distance,
        prominence=prominence,
    )

    # 5) return DataFrame with fixed columns
    cols = ["rep_id","rep_start","rep_mid","rep_end"]
    return pd.DataFrame(reps, columns=cols)


def main() -> None:
    p = argparse.ArgumentParser(description="Detect squat repetitions")
    p.add_argument("--input",  required=True, type=Path,
                   help="Input .npy keypoints (F,K,3)")
    p.add_argument("--output", default="reps.csv", type=Path)
    p.add_argument("--ankle_ref_frame", type=int, default=None,
                   help="Optional frame to use for baseline")
    p.add_argument("--standing_ratio", type=float, default=0.70)
    p.add_argument("--depth_ratio",    type=float, default=0.30)
    p.add_argument("--min_distance",   type=int,   default=3)
    p.add_argument("--prominence",     type=float, default=1e-4)
    args = p.parse_args()

    df = pipeline(
        keypoints=args.input,
        ankle_ref_frame=args.ankle_ref_frame,
        standing_ratio=args.standing_ratio,
        depth_ratio=args.depth_ratio,
        min_distance=args.min_distance,
        prominence=args.prominence,
    )
    df.to_csv(args.output, index=False)
    print("Saved →", args.output)


if __name__ == "__main__":
    main()
