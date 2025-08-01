#!/usr/bin/env python3
"""
forward_knees_analysis.py  🦵
============================
Estimate forward-knee angle per squat repetition (HALPE-26 keypoints).

► **NEW (2025-07-06): `lengths_json` is now mandatory.**
  Calling `build_report` or `pipeline` without it raises `ValueError`.

Public API
----------
build_report(keypoints, reps_csv_or_df, out_csv, lengths_json=...) -> pd.DataFrame
pipeline(...)                                   # lightweight in-memory helper
"""
from __future__ import annotations
import math, json
from pathlib import Path
from typing import Mapping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

CONF_TH = 0.0                     # keypoint confidence threshold
SEVERE_TH, MILD_TH = 20.0, 10.0   # angle thresholds (deg)






def plot_forward_knees(
    keypoints: np.ndarray,
    reps: pd.DataFrame,
    out_path: str | Path,
    lengths_json: str | Path | dict = None,
):
    """
    Plot per-frame forward knee angle for each rep.
    One line for left knee, one for right.
    """
    # Load lengths as in build_report
    if lengths_json is None:
        raise ValueError("Must supply lengths_json for plotting")
    if isinstance(lengths_json, dict):
        lengths = lengths_json
    else:
        txt = str(lengths_json).strip()
        from pathlib import Path
        import json
        if not (txt.startswith("{") and txt.endswith("}")):
            with open(txt) as f:
                raw = json.load(f)
        else:
            raw = json.loads(txt)
        lengths = {
            (int(a), int(b)): float(v)
            for k, v in raw.items()
            for a, b in [k if isinstance(k, (list, tuple))
                         else map(int, str(k).strip("()").split(","))]
        }

    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16

    left_angles, right_angles, frame_nums = [], [], []

    if {"rep_start", "rep_end"}.issubset(reps.columns):
        reps = reps.rename(columns={"rep_start": "start", "rep_end": "end"})

    for _, rep in reps.iterrows():
        start, end = int(rep.start), int(rep.end)
        for f in range(start, end + 1):
            kp = keypoints[f]
            θL, okL = _knee_forward_angle(kp, L_KNEE, L_ANKLE, lengths.get((L_KNEE, L_ANKLE), 0.0))
            θR, okR = _knee_forward_angle(kp, R_KNEE, R_ANKLE, lengths.get((R_KNEE, R_ANKLE), 0.0))
            left_angles.append(θL if okL else np.nan)
            right_angles.append(θR if okR else np.nan)
            frame_nums.append(f)

    plt.figure(figsize=(12, 5))
    plt.plot(frame_nums, left_angles, label="Left knee angle (deg)", lw=1.2)
    plt.plot(frame_nums, right_angles, label="Right knee angle (deg)", lw=1.2)
    plt.xlabel("Frame")
    plt.ylabel("Knee forward angle (degrees)")
    plt.title("Forward Knee Angle Per Frame")
    plt.legend()
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"📈  Saved forward-knee plot → {out_path}")
# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _load_lengths(path: Path) -> dict[tuple[int, int], float]:
    """Load shin lengths JSON → dict[(idx,idx)] → float."""
    with open(path) as f:
        raw = json.load(f)
    return {tuple(map(int, k.strip("()").split(","))): float(v)
            for k, v in raw.items()}


def _shin_len(lengths: Mapping[tuple[int, int], float],
              knee: int, ankle: int) -> float:
    """Fetch shin length (0 if missing/invalid)."""
    val = lengths.get((knee, ankle)) or lengths.get((ankle, knee), 0.0)
    return float(val) if val and val > 0 else 0.0


def _knee_forward_angle(kp: np.ndarray,
                        knee_idx: int,
                        ankle_idx: int,
                        shin_len: float) -> tuple[float, bool]:
    """
    3-D forward-knee angle from 2-D projection.

    Returns (theta_deg, valid_flag).
    """
    x1, y1, c1 = kp[ankle_idx]
    x2, y2, c2 = kp[knee_idx]

    if min(c1, c2) < CONF_TH or np.isnan([x1, y1, x2, y2]).any():
        return 0.0, False

    planar = math.hypot(x2 - x1, y2 - y1)
    shin   = shin_len

    if planar <= 0 or planar >= shin:
        return 0.0, False

    z = math.sqrt(shin ** 2 - planar ** 2)
    theta = math.degrees(math.atan2(z, planar))
    return theta, True


# --------------------------------------------------------------------------
# Main disk-writing function
# --------------------------------------------------------------------------
def build_report(
    keypoints: np.ndarray,
    reps_csv_or_df: str | Path | pd.DataFrame,
    out_csv: str | Path,
    lengths_json: str | Path | Mapping,
) -> pd.DataFrame:
    """
    Write the forward-knee report and return it.

    `lengths_json` **must be provided**; otherwise a `ValueError` is raised.
    """
    if lengths_json is None:
        raise ValueError(
            "The `lengths_json` argument is required (shin-length reference)."
        )

    # ---------------- reps table ----------------
    reps_df = (
        reps_csv_or_df.copy()
        if isinstance(reps_csv_or_df, pd.DataFrame)
        else pd.read_csv(reps_csv_or_df)
    )
    if {"rep_start", "rep_end"}.issubset(reps_df.columns):
        reps_df = reps_df.rename(columns={"rep_start": "start",
                                          "rep_end":   "end"})
    required = {"rep_id", "start", "end"}
    if not required.issubset(reps_df.columns):
        raise ValueError(f"Reps table must contain {sorted(required)}")

    # ---------------- shin lengths -------------
    if isinstance(lengths_json, Mapping):
        raw = lengths_json
    else:  # path or raw JSON string
        txt = str(lengths_json).strip()
        raw = _load_lengths(Path(txt)) if not (txt.startswith("{") and txt.endswith("}")) \
              else json.loads(txt)

    lengths: dict[tuple[int, int], float] = {
        (int(a), int(b)): float(v)
        for k, v in raw.items()
        for a, b in [k if isinstance(k, (list, tuple))
                     else map(int, str(k).strip("()").split(","))]
    }

    # ---------------- analyse reps -------------
    rows = []
    for _, rep in reps_df.iterrows():
        rep_id, start, end = int(rep.rep_id), int(rep.start), int(rep.end)
        thetas: list[float] = []

        for f in range(start, end + 1):
            kp = keypoints[f]
            θL, okL = _knee_forward_angle(kp, L_KNEE, L_ANKLE,
                                          _shin_len(lengths, L_KNEE, L_ANKLE))
            θR, okR = _knee_forward_angle(kp, R_KNEE, R_ANKLE,
                                          _shin_len(lengths, R_KNEE, R_ANKLE))
            if okL or okR:
                thetas.append(max(θL, θR))

        maxθ = max(thetas) if thetas else 0.0
        avgθ = float(np.mean(thetas)) if thetas else 0.0
        severity = ("none", "mild", "severe")[2 if maxθ >= SEVERE_TH
                                              else 1 if maxθ >= MILD_TH else 0]

        rows.append({
            "rep_id": rep_id,
            "avg_forward_angle_deg": round(avgθ, 2),
            "max_forward_angle_deg": round(maxθ, 2),
            "severity": severity,
        })

    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ forward-knee report → {out_csv}")
    return df




def _knee_forward_angle_3d(
    kp: np.ndarray,
    knee_idx: int,
    ankle_idx: int
) -> tuple[float, bool]:
    """
    Returns (theta_deg, valid_flag) for 3D keypoints.
    Angle is the deviation of shin from vertical (z axis).
    """
    # 3D positions: (x, y, z)
    x1, y1, z1 = kp[ankle_idx, :3]
    x2, y2, z2 = kp[knee_idx, :3]
    v = np.array([x2 - x1, y2 - y1, z2 - z1])
    norm = np.linalg.norm(v)
    if norm < 1e-6 or np.isnan(v).any():
        return 0.0, False
    # Reference vertical direction: (0, 0, 1)
    dot = v[2] / norm  # projection onto z axis
    dot = np.clip(dot, -1.0, 1.0)
    theta = math.degrees(math.acos(dot))  # angle from vertical
    return theta, True

def _find_error_ranges(frames, values, th):
    """Given frames and values, return list of [start, end] where value > th."""
    ranges = []
    active = False
    for i, (f, v) in enumerate(zip(frames, values)):
        if v > th:
            if not active:
                start = f
                active = True
        else:
            if active:
                end = frames[i - 1]
                ranges.append([start, end])
                active = False
    if active:
        ranges.append([start, frames[-1]])
    return ranges

def build_report_3d(
    keypoints_3d: np.ndarray,
    reps_csv_or_df: str | Path | pd.DataFrame,
    out_csv: str | Path,
    lengths_json: str | Path | Mapping,
) -> pd.DataFrame:
    # ---------------- reps table ----------------
    reps_df = (
        reps_csv_or_df.copy()
        if isinstance(reps_csv_or_df, pd.DataFrame)
        else pd.read_csv(reps_csv_or_df)
    )
    if {"rep_start", "rep_end"}.issubset(reps_df.columns):
        reps_df = reps_df.rename(columns={"rep_start": "start",
                                          "rep_end":   "end"})
    required = {"rep_id", "start", "end"}
    if not required.issubset(reps_df.columns):
        raise ValueError(f"Reps table must contain {sorted(required)}")

    rows = []
    for _, rep in reps_df.iterrows():
        rep_id, start, end = int(rep.rep_id), int(rep.start), int(rep.end)
        left_angles, right_angles, frames = [], [], []
        for f in range(start, end + 1):
            kp = keypoints_3d[f]
            θL, okL = _knee_forward_angle_3d(kp, L_KNEE, L_ANKLE)
            θR, okR = _knee_forward_angle_3d(kp, R_KNEE, R_ANKLE)
            left_angles.append(θL if okL else np.nan)
            right_angles.append(θR if okR else np.nan)
            frames.append(f)

        # Identify frame ranges with errors
        left_severe = _find_error_ranges(frames, left_angles, SEVERE_TH)
        left_mild   = _find_error_ranges(frames, left_angles, MILD_TH)
        right_severe = _find_error_ranges(frames, right_angles, SEVERE_TH)
        right_mild   = _find_error_ranges(frames, right_angles, MILD_TH)

        maxθL = np.nanmax(left_angles) if left_angles else 0.0
        maxθR = np.nanmax(right_angles) if right_angles else 0.0
        avgθL = float(np.nanmean(left_angles)) if left_angles else 0.0
        avgθR = float(np.nanmean(right_angles)) if right_angles else 0.0

        severityL = ("none", "mild", "severe")[2 if maxθL >= SEVERE_TH else 1 if maxθL >= MILD_TH else 0]
        severityR = ("none", "mild", "severe")[2 if maxθR >= SEVERE_TH else 1 if maxθR >= MILD_TH else 0]

        rows.append({
            "rep_id": rep_id,
            "left_avg_forward_angle_deg": round(avgθL, 2),
            "right_avg_forward_angle_deg": round(avgθR, 2),
            "left_max_forward_angle_deg": round(maxθL, 2),
            "right_max_forward_angle_deg": round(maxθR, 2),
            "left_severity": severityL,
            "right_severity": severityR,
            "left_mild_error_ranges": left_mild,
            "left_severe_error_ranges": left_severe,
            "right_mild_error_ranges": right_mild,
            "right_severe_error_ranges": right_severe,
        })

    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ 3D forward-knee report → {out_csv}")
    return df


    

# --------------------------------------------------------------------------
# Lightweight in-memory helper
# --------------------------------------------------------------------------
def pipeline(
    type: str,
    keypoints: str | Path | np.ndarray,
    keypoints_3d: str | Path | np.ndarray,
    reps: str | Path | pd.DataFrame,
    *,
    lengths_json: str | Path | Mapping,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Forward-knee analysis, flexible for 2D or 3D.

    Parameters
    ----------
    type          : "2d" or "3d"
    keypoints     : .npy path or ndarray (F,26,3) keypoints in unit coords (2D)
    keypoints_3d  : .npy path or ndarray (F,26,3) keypoints (3D)
    reps          : CSV path or DataFrame with ['rep_id','start','end']
    lengths_json  : shin lengths in the same unit scale
    output_csv    : optional CSV path
    """
    if lengths_json is None:
        raise ValueError(
            "The `lengths_json` argument is required; shin lengths are essential."
        )

    reps_df = reps.copy() if isinstance(reps, pd.DataFrame) else pd.read_csv(reps)
    if {"rep_start", "rep_end"}.issubset(reps_df.columns):
        reps_df = reps_df.rename(columns={"rep_start": "start",
                                          "rep_end":   "end"})

    # Always load arrays only when needed (don't error on None)
    if type == "3d":
        kps_3d = np.load(keypoints_3d) if isinstance(keypoints_3d, (str, Path)) else keypoints_3d
        df = build_report_3d(
            kps_3d, reps_df, output_csv or Path("/dev/null"), lengths_json
        )
    else:
        kps = np.load(keypoints) if isinstance(keypoints, (str, Path)) else keypoints
        df = build_report(
            kps, reps_df, output_csv or Path("/dev/null"), lengths_json
        )

    if output_csv is None:                     # silence dummy path print
        df.attrs.pop("filepath_or_buffer", None)
    return df


# --------------------------------------------------------------------------
# CLI entry-point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        "Forward-knee angle reporter (unit-square coords)"
    )
    ap.add_argument("--type", required=True)
    ap.add_argument("--keypoints_3d", required=False, default=None,)
    ap.add_argument("--keypoints", required=True,
                    help="NumPy .npy with (F,26,3) keypoints in unit coords")
    ap.add_argument("--reps",      required=True,
                    help="CSV/feather with rep_id,start,end (or rep_* variants)")
    ap.add_argument("--lengths",   required=True,
                    help="Shin-lengths JSON file or raw JSON string")
    ap.add_argument("--out",       default="forward_knee_angle_report.csv",
                    help="Output CSV path")
    args = ap.parse_args()

    kps = np.load(args.keypoints)
    kps_3d = np.load(args.keypoints_3d) if args.keypoints_3d else None
    if args.type == "3d" and kps_3d is None:
        raise ValueError("Must supply --keypoints_3d for 3D forward-knee analysis")
    if args.type == "3d" and kps_3d is not None:
        build_report_3d(
            kps_3d, args.reps, args.out, args.lengths
        )
    else:
        # 2D forward-knee analysis
        build_report(
            kps, args.reps, args.out, args.lengths
        )
    
