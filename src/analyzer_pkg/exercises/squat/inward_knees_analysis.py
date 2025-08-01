#!/usr/bin/env python3
"""
inward_knees_analysis.py  🦵
===========================
FPPA and line-crossing analytics for squat reps (2-D unit-square coordinates).

Public API
----------
generate_ffpa_report(...)
generate_line_crossing_report(...)
pipeline(...)                     # convenience – returns the FPPA report as a DataFrame

Change log (2025-07-06)
-----------------------
* Added dual support for ['start','end'] **and** ['rep_start','rep_end'] rep tables.
* Normalised default `crossing_thresh` to 0.05 (was 10.0 in pixel space).
* `generate_line_crossing_report` now accepts a DataFrame as well as a CSV path.
* Guarded `_signed_dist` against zero-length hip-ankle vectors.
* Updated CLI defaults/help accordingly.
"""


from __future__ import annotations
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd



def _signed_outward_dist(pt: np.ndarray, hip: np.ndarray, ankle: np.ndarray, side: str) -> float:
    """
    Signed outward distance from knee to hip-ankle line.
    Positive = knee is outside, Negative = knee is inside (valgus error).
    Side must be "L" or "R".
    """
    v = ankle - hip
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return 0.0
    # Outward is always to the left for the left leg and to the right for the right leg,
    # so we flip the direction accordingly
    # For left leg: outward is +ve, for right leg: outward is +ve (from the body's perspective)
    perp = np.array([-v[1], v[0]]) / norm
    if side == "R":
        perp = -perp   # flip for right side
    return float(np.dot(pt - hip, perp))



def plot_fppa_over_time(
    keypoints_array: np.ndarray,
    reps_df: pd.DataFrame,
    out_path: str | Path,
    mild_thresh: float = 170.0,
    severe_thresh: float = 160.0,
):
    """Plot FPPA (deg) per frame with severity bands and thresholds."""
    kp2 = keypoints_array[..., :2]
    frames = range(len(kp2))
    fppa_left, fppa_right = [], []

    for f in frames:
        fppa_left.append(_fppa_outside(kp2[f], "L"))
        fppa_right.append(_fppa_outside(kp2[f], "R"))

    plt.figure(figsize=(12, 5))
    plt.plot(frames, fppa_left, label="Left FPPA", color="blue")
    plt.plot(frames, fppa_right, label="Right FPPA", color="red")

    plt.axhline(mild_thresh, color="orange", ls="--", label=f"Mild threshold ({mild_thresh}°)")
    plt.axhline(severe_thresh, color="red", ls="--", label=f"Severe threshold ({severe_thresh}°)")

    # Fill severity regions
    plt.fill_between(frames, 0, severe_thresh, color="red", alpha=0.10, label="Severe zone")
    plt.fill_between(frames, severe_thresh, mild_thresh, color="orange", alpha=0.08, label="Mild zone")

    plt.xlabel("Frame")
    plt.ylabel("FPPA (deg)")
    plt.title("FPPA (Frontal Plane Projection Angle) per Frame")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"📉  Saved FPPA plot → {out_path}")

_EPS = 1e-8 
def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Interior angle (deg) between two 2‑D vectors.
    Returns NaN if one of the vectors has ~zero length.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    den = n1 * n2
    if den < _EPS:
        return float("nan")          # unreliable – later filters will skip
    num = float(np.dot(v1, v2))
    return float(np.degrees(np.arccos(np.clip(num / den, -1.0, 1.0))))



def _signed_dist(pt: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Signed perpendicular distance from point *pt* to the line through *p1*→*p2*.
    Positive = medial side (hip-midpoint test later).
    """
    v = p2 - p1
    norm = np.linalg.norm(v)
    if norm < 1e-8:        # hip and ankle coincide – avoid divide-by-zero
        return 0.0
    perp = np.array([-v[1], v[0]]) / norm
    return float(np.dot(pt - p1, perp))




def plot_knee_distances_over_time(
    keypoints_array: np.ndarray,
    out_path: str | Path,
    left: bool = True,
    right: bool = True,
    frames: list[int] | None = None,
    show_thresh: float = 0.05,
):
    """
    Plots the signed (absolute) knee distance from the hip-ankle line vs time for each frame.

    Parameters
    ----------
    keypoints_array : ndarray (F, 17, 2|3)
    out_path        : PNG output path
    left, right     : Plot left/right leg (default both)
    frames          : If given, restrict to these frames only (otherwise all)
    show_thresh     : Show threshold as horizontal line
    """
    kp2 = keypoints_array[..., :2]  # always 2D

    if frames is None:
        frames = list(range(len(kp2)))
    frames = [f for f in frames if 0 <= f < len(kp2)]

    dists_left, dists_right = [], []

    for f in frames:
        k = kp2[f]
        # Left leg:  hip=11, knee=13, ankle=15
        ldist = _signed_outward_dist(k[13], k[11], k[15], "L")
        dists_left.append(ldist)
        # Right leg: hip=12, knee=14, ankle=16
        rdist = _signed_outward_dist(k[14], k[12], k[16], "R")
        dists_right.append(rdist)

    plt.figure(figsize=(12, 4))
    if left:
        plt.plot(frames, dists_left, label="Left knee distance", color="blue")
    if right:
        plt.plot(frames, dists_right, label="Right knee distance", color="red")
    plt.axhline(show_thresh, color="gray", ls="--", lw=1, label=f"thresh={show_thresh}")
    plt.xlabel("Frame")
    plt.ylabel("Signed knee distance from hip-ankle line (unit square)")
    plt.title("Knee-to-line distance over time (left/right legs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"📉  Saved knee distances plot → {out_path}")




def _fppa_outside(kp2: np.ndarray, side: str = "L") -> float:
    """
    Outside-knee frontal-plane projection angle (FPPA) for one frame.

    Parameters
    ----------
    kp2  : ndarray (17, 2)  – 2-D keypoints in unit square
    side : {"L","R"}        – left or right leg
    """
    h, k, a = (11, 13, 15) if side == "L" else (12, 14, 16)  # hip, knee, ankle
    hip, knee, ankle = kp2[h], kp2[k], kp2[a]

    # Interior (anatomical) knee angle
    interior = _angle(hip - knee, ankle - knee)

    # Determine if knee is medial to hip-ankle line
    midhip = (kp2[11] + kp2[12]) / 2.0
    medial = _signed_dist(midhip, hip, ankle) * _signed_dist(knee, hip, ankle) > 0

    return interior if medial else 360.0 - interior


# ---------------- FPPA report with start/peak/return frames ---------------- #
def generate_ffpa_report(
    keypoints_array: np.ndarray,
    reps: str | Path | pd.DataFrame,
    mild_thresh: float = 170.0,
    severe_thresh: float = 160.0,
    output_csv: str | Path | None = "ffpa_report.csv",
) -> pd.DataFrame:
    reps_df = reps if isinstance(reps, pd.DataFrame) else pd.read_csv(reps)
    if {"start", "end"}.issubset(reps_df.columns):
        start_col, end_col = "start", "end"
    elif {"rep_start", "rep_end"}.issubset(reps_df.columns):
        start_col, end_col = "rep_start", "rep_end"
    else:
        raise ValueError("Reps table must contain either ['start','end'] or ['rep_start','rep_end'] columns")

    def _find_events(vals: list[float], mild, severe, start_idx):
        arr = np.array(vals)
        if not len(arr) or np.all(np.isnan(arr)):
            return dict(severity='none', min_val=float('nan'), 
                        start=None, peak=None, end=None)
        # Determine threshold (severe takes priority)
        if np.nanmin(arr) < severe:
            th = severe
            sev = 'severe'
        elif np.nanmin(arr) < mild:
            th = mild
            sev = 'mild'
        else:
            th = mild
            sev = 'none'
        # Start: first frame below threshold
        below = np.where(arr < th)[0]
        if below.size == 0:
            return dict(severity=sev, min_val=float('nan'), 
                        start=None, peak=None, end=None)
        start = int(start_idx + below[0])
        # Peak: minimum value index
        peak_rel = int(np.nanargmin(arr))
        peak = int(start_idx + peak_rel)
        # End: first frame after peak where FPPA returns above threshold
        above = np.where((np.arange(len(arr)) > peak_rel) & (arr >= th))[0]
        end = int(start_idx + above[0]) if above.size > 0 else int(start_idx + len(arr) - 1)
        return dict(severity=sev, min_val=float(np.nanmin(arr)), 
                    start=start, peak=peak, end=end)

    rows = []
    for _, rep in reps_df.iterrows():
        rep_id = int(rep["rep_id"])
        start, end = int(rep[start_col]), int(rep[end_col])
        left_vals, right_vals = [], []
        for f in range(start, end + 1):
            kp2 = keypoints_array[f, :, :2]
            left_vals.append(_fppa_outside(kp2, "L"))
            right_vals.append(_fppa_outside(kp2, "R"))

        ldict = _find_events(left_vals, mild_thresh, severe_thresh, start)
        rdict = _find_events(right_vals, mild_thresh, severe_thresh, start)
        rows.append({
            "rep_id": rep_id,
            "left_min_FPPA": ldict["min_val"],
            "left_severity": ldict["severity"],
            "left_problem_start": ldict["start"],
            "left_problem_peak": ldict["peak"],
            "left_problem_end": ldict["end"],
            "right_min_FPPA": rdict["min_val"],
            "right_severity": rdict["severity"],
            "right_problem_start": rdict["start"],
            "right_problem_peak": rdict["peak"],
            "right_problem_end": rdict["end"],
        })

    df = pd.DataFrame(rows)
    if output_csv is not None:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"✅ FPPA report → {out}")
    return df

# ------------- Line-crossing report with start/peak/return frames ------------- #
def generate_line_crossing_report(
    keypoints_array: np.ndarray,
    reps: str | Path | pd.DataFrame,
    crossing_thresh: float = 0.05,
    output_csv: str | Path = "line_crossing_report.csv",
) -> pd.DataFrame:
    reps_df = reps if isinstance(reps, pd.DataFrame) else pd.read_csv(reps)
    if {"start", "end"}.issubset(reps_df.columns):
        start_col, end_col = "start", "end"
    elif {"rep_start", "rep_end"}.issubset(reps_df.columns):
        start_col, end_col = "rep_start", "rep_end"
    else:
        raise ValueError("Reps table must contain either ['start','end'] or ['rep_start','rep_end'] columns")

    def _find_events(arr, th, start_idx):
        arr = np.array(arr)
        if not len(arr) or np.all(np.isnan(arr)):
            return dict(severity='none', max_val=float('nan'),
                        start=None, peak=None, end=None)
        # Apply threshold: only consider values above threshold
        over = np.where(arr >= th)[0]
        if over.size == 0:
            return dict(severity='none', max_val=float('nan'),
                        start=None, peak=None, end=None)
        mx = np.nanmax(arr)
        sev = 'severe' if mx >= 2 * th else 'mild'
        start = int(start_idx + over[0])
        peak_rel = int(np.nanargmax(arr))
        peak = int(start_idx + peak_rel)
        # End: first after peak where distance drops below threshold
        below = np.where((np.arange(len(arr)) > peak_rel) & (arr < th))[0]
        end = int(start_idx + below[0]) if below.size > 0 else int(start_idx + len(arr) - 1)
        return dict(severity=sev, max_val=mx, start=start, peak=peak, end=end)

    rows = []
    for _, rep in reps_df.iterrows():
        rep_id = int(rep["rep_id"])
        start, end = int(rep[start_col]), int(rep[end_col])
        ldist, rdist = [], []
        for f in range(start, end + 1):
            kp2 = keypoints_array[f, :, :2]
            # Left side
            medial_l = _signed_dist((kp2[11] + kp2[12]) / 2, kp2[11], kp2[15]) * \
                       _signed_dist(kp2[13], kp2[11], kp2[15]) > 0
            if medial_l:
                dist_l = abs(_signed_dist(kp2[13], kp2[11], kp2[15]))
                ldist.append(dist_l)
            else:
                ldist.append(0.0)
            # Right side
            medial_r = _signed_dist((kp2[11] + kp2[12]) / 2, kp2[12], kp2[16]) * \
                       _signed_dist(kp2[14], kp2[12], kp2[16]) > 0
            if medial_r:
                dist_r = abs(_signed_dist(kp2[14], kp2[12], kp2[16]))
                rdist.append(dist_r)
            else:
                rdist.append(0.0)
        ldict = _find_events(ldist, crossing_thresh, start)
        rdict = _find_events(rdist, crossing_thresh, start)
        rows.append({
            "rep_id": rep_id,
            "left_max_dist": ldict["max_val"],
            "left_severity": ldict["severity"],
            "left_problem_start": ldict["start"],
            "left_problem_peak": ldict["peak"],
            "left_problem_end": ldict["end"],
            "right_max_dist": rdict["max_val"],
            "right_severity": rdict["severity"],
            "right_problem_start": rdict["start"],
            "right_problem_peak": rdict["peak"],
            "right_problem_end": rdict["end"],
        })

    df = pd.DataFrame(rows)
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"✅ Line-crossing report → {out}")
    return df

# ---------------------------------------------------------------------
# ❸  Convenience wrapper – FPPA only
# ---------------------------------------------------------------------
def pipeline(
    keypoints_array: np.ndarray,
    reps: str | Path | pd.DataFrame,
    *,
    mild_thresh: float = 175.0,
    severe_thresh: float = 170.0,
    output_csv: str | Path | None = "ffpa_report.csv",
) -> pd.DataFrame:
    """Run FPPA analysis end-to-end and return the report."""
    return generate_ffpa_report(
        keypoints_array,
        reps,
        mild_thresh=mild_thresh,
        severe_thresh=severe_thresh,
        output_csv=output_csv,
    )


# ---------------------------------------------------------------------
# ❹  CLI entry-point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser("FFPA & line-crossing reporter")
    ap.add_argument("--keypoints",       default="imputed_ma.npy",
                    help="NumPy npy file with (F,17,3) keypoints")
    ap.add_argument("--reps",            default="repetition_data.csv",
                    help="CSV or feather file with rep boundaries")
    ap.add_argument("--out-fppa",        default="ffpa_report.csv",
                    help="Output CSV for FPPA report")
    ap.add_argument("--out-line",        default="line_crossing_report.csv",
                    help="Output CSV for line-crossing report")
    ap.add_argument("--crossing-thresh", type=float, default=0.05,
                    help="Distance threshold for line-cross detection (unit coords)")

    args = ap.parse_args()

    kps = np.load(args.keypoints)
    generate_ffpa_report(kps, args.reps, output_csv=args.out_fppa)
    generate_line_crossing_report(
        kps, args.reps,
        crossing_thresh=args.crossing_thresh,
        output_csv=args.out_line
    )
