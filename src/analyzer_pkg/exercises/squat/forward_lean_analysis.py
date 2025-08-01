#!/usr/bin/env python3
"""
forward_lean_analysis.py  📏
===========================

Compute *deviation from vertical* for each squat repetition.

Definitions
-----------
Hip              = (hx,  hy,    0)
Shoulder         = (sx,  sy,   ±z)      # depth reconstructed from torso length
Vertical unit v  = (0,  sign,  0)       # sign = +1 if y grows upward, –1 if y grows downward
Torso vector t   = Shoulder − Hip

Angle = arccos( ⟨t, v⟩ / ||t|| ).
• Small angle ⇒ upright torso
• Large angle ⇒ forward-lean

Output
------
forward_lean_report.csv with columns
    rep_id, severity, frame, angle_deg

Changes (2025-07-06)
--------------------
* Works with **unit-square coordinates** by scaling xyz ×1000 internally.
* Auto-detects y-axis orientation (image vs Cartesian).
* `lengths_json` is **mandatory**; ValueError if omitted.
"""
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
from typing import Dict, List, Mapping, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ constants & indices
# HALPE-26 indices
L_HIP, R_HIP   = 11, 12
L_KNEE, R_KNEE = 13, 14
L_SHO, R_SHO   = 5,  6

CONF_TH_DEF = 0.20       # default confidence
SCALE       = 1000.0     # up-scale to kill 0-1 rounding errors
MILD_TH     = 45.0       # >45° ⇒ mild lean
SEV_TH      = 55.0       # >55° ⇒ severe lean





def plot_forward_lean(
    keypoints: np.ndarray,
    reps: pd.DataFrame,
    out_path: str | Path,
    lengths_json: str | Path | dict = None,
    window: int = 5,
):
    """
    Plot per-frame forward-lean angle with mild/severe threshold lines.
    """
    # Thresholds from your constants
    MILD_TH = 45.0
    SEV_TH  = 55.0

    # Lengths
    if lengths_json is None:
        raise ValueError("`lengths_json` is required for plotting.")
    if isinstance(lengths_json, dict):
        lens = _normalize_lengths(lengths_json)
    else:
        txt = str(lengths_json).strip()
        lens = (
            _normalize_lengths(json.loads(txt)) if txt.startswith("{")
            else _normalize_lengths(json.load(open(txt)))
        )

    # Prepare
    if "rep_mid" in reps.columns and "mid" not in reps.columns:
        reps = reps.rename(columns={"rep_mid": "mid"})

    frame_nums = []
    angles = []

    for _, row in reps.iterrows():
        mid = int(row.mid)
        sign = _orientation_sign(keypoints, mid)
        frames = range(max(0, mid - window), min(keypoints.shape[0], mid + window + 1))
        for f in frames:
            a = _frame_deviation(keypoints[f], lens, sign)
            if not math.isnan(a):
                frame_nums.append(f)
                angles.append(a)

    plt.figure(figsize=(12, 5))
    plt.plot(frame_nums, angles, label="Torso deviation from vertical (deg)", lw=1.5)
    plt.axhline(MILD_TH, color="orange", ls="--", lw=1.5, label="Mild threshold (45°)")
    plt.axhline(SEV_TH, color="red", ls="--", lw=1.5, label="Severe threshold (55°)")
    plt.xlabel("Frame")
    plt.ylabel("Torso angle from vertical (degrees)")
    plt.title("Forward Lean Angle Over Frames")
    plt.legend()
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"📈  Saved forward-lean plot → {out_path}")




# ------------------------------------------------------------------ helpers
def _get_len(pairs: Mapping[str, float], i: int, j: int) -> float | None:
    """Fetch length under any key permutation "(i,j)" etc."""
    for k in (f"({i},{j})", f"({i}, {j})", f"({j},{i})", f"({j}, {i})"):
        if k in pairs:
            return float(pairs[k])
    return None


def _normalize_lengths(raw: Mapping[str, float]) -> Dict[str, float]:
    """Return dict with torsoL, torsoR … in unit-square scale."""
    if any(k.startswith("(") for k in raw):       # "(i,j)" style keys
        return {
            "torsoL": _get_len(raw, L_HIP, L_SHO) or 0.0,
            "torsoR": _get_len(raw, R_HIP, R_SHO) or 0.0,
            # thighs unused but kept for completeness
            "thighL": _get_len(raw, L_HIP, L_KNEE) or 0.0,
            "thighR": _get_len(raw, R_HIP, R_KNEE) or 0.0,
        }
    # already in flat form
    return {k: float(v) for k, v in raw.items()}


def _rel_z(x1: float, y1: float,
           x2: float, y2: float,
           L: float) -> float:
    """Depth from 2-D projection (positive)."""
    d2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if d2 < L * L:
        return math.sqrt(L * L - d2)
    return math.sqrt(d2 - L * L)


def _orientation_sign(kps: np.ndarray, mid: int) -> int:
    """
    Detect whether y grows upward (+1) or downward (−1).
    Hip descends at 'mid' in image coords.
    """
    hip_y0 = kps[max(0, mid - 3), L_HIP, 1]
    hip_y1 = kps[mid, L_HIP, 1]
    return -1 if hip_y1 > hip_y0 else +1


def _vert_torso_angle(
    kp: np.ndarray,
    hip_i: int,
    sho_i: int,
    torso_len: float,
    sign: int,
) -> float:
    """Angle (deg) between torso vector and vertical axis."""
    hx, hy, hc = kp[hip_i]
    sx, sy, sc = kp[sho_i]
    if min(hc, sc) < CONF_TH_DEF or math.isnan(hx+hy+sx+sy) or torso_len == 0:
        return float("nan")

    # scale coords and length to avoid precision issues
    hx, hy, sx, sy, L = (hx * SCALE, hy * SCALE, sx * SCALE, sy * SCALE, torso_len * SCALE)
    dz = _rel_z(hx, hy, sx, sy, L)

    v_t = np.array([sx - hx, sy - hy, dz])
    norm = np.linalg.norm(v_t)
    if norm == 0.0:
        return float("nan")

    v_v = np.array([0.0, sign, 0.0])     # vertical reference
    cosang = np.clip(np.dot(v_t, v_v) / norm, -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def _frame_deviation(kp: np.ndarray,
                     lens: Mapping[str, float],
                     sign: int) -> float:
    """Return the larger deviation angle of the two sides for one frame."""
    left  = _vert_torso_angle(kp, L_HIP, R_SHO, lens["torsoL"], sign)
    right = _vert_torso_angle(kp, R_HIP, L_SHO, lens["torsoR"], sign)
    vals = [a for a in (left, right) if not math.isnan(a)]
    return max(vals) if vals else float("nan")


def _severity(a: float) -> str:
    if math.isnan(a):
        return "unknown"
    if a > SEV_TH:
        return "severe"
    if a > MILD_TH:
        return "mild"
    return "none"


# ------------------------------------------------------------------ core report generator
def generate_report(
    kps: np.ndarray,
    reps: pd.DataFrame,
    lens: Mapping[str, float],
    *,
    window: int = 5,
    out_csv: str = "forward_lean_report.csv",
) -> pd.DataFrame:
    recs: List[Dict[str, Union[int, float, str]]] = []
    F = kps.shape[0]

    for _, row in reps.iterrows():
        rep_id = int(row.rep_id)
        mid    = int(row.mid)
        sign   = _orientation_sign(kps, mid)

        frames = range(max(0, mid - window), min(F, mid + window + 1))
        vals = [
            _frame_deviation(kps[f], lens, sign)
            for f in frames
            if not math.isnan(_frame_deviation(kps[f], lens, sign))
        ]
        val = max(vals) if vals else float("nan")

        recs.append({
            "rep_id":   rep_id,
            "severity": _severity(val),
            "frame":    mid,
            "angle_deg": round(val, 2) if not math.isnan(val) else float("nan"),
        })

    df = pd.DataFrame(recs)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ forward-lean report → {os.path.abspath(out_csv)}")
    return df



def generate_report_3d(
    kps_3d: np.ndarray,
    reps: pd.DataFrame,
    lens: Mapping[str, float],  # Not actually needed for 3D version, but kept for symmetry
    *,
    window: int = 5,
    out_csv: str = "forward_lean_report_3d.csv",
) -> pd.DataFrame:
    """
    Analyze 3D keypoints for forward lean during reps, outputting
    both the maximal angle and all frame ranges where lean exceeds thresholds.
    """
    MILD_TH = 45.0
    SEV_TH  = 55.0
    F = kps_3d.shape[0]
    L_HIP, R_HIP, L_SHO, R_SHO = 11, 12, 5, 6

    # Decide which hip/shoulder pairs to use (as in your 2D logic)
    recs: List[Dict[str, Union[int, float, str, list]]] = []

    for _, row in reps.iterrows():
        rep_id = int(row.rep_id)
        mid = int(row.mid)

        # (Optionally) use your y-orientation logic, or just set sign = -1
        # Here we just use image convention:
        sign = 1

        frames = range(max(0, mid - window), min(F, mid + window + 1))
        angles = []
        for f in frames:
            kp = kps_3d[f]
            # Left and right side (shoulder - hip)
            v_t_L = kp[R_SHO] - kp[L_HIP]   # Note: right shoulder, left hip
            v_t_R = kp[L_SHO] - kp[R_HIP]   # Note: left shoulder, right hip

            # Use the side with bigger angle (i.e., larger forward lean)
            vertical = np.array([0, sign, 0])

            # Defensive: skip if NaN or zero length
            angle_L = float("nan")
            angle_R = float("nan")
            if not np.any(np.isnan(v_t_L)) and np.linalg.norm(v_t_L) > 1e-6:
                v_L_norm = v_t_L / np.linalg.norm(v_t_L)
                cos_L = np.clip(np.dot(v_L_norm, vertical), -1, 1)
                angle_L = math.degrees(math.acos(cos_L))
            if not np.any(np.isnan(v_t_R)) and np.linalg.norm(v_t_R) > 1e-6:
                v_R_norm = v_t_R / np.linalg.norm(v_t_R)
                cos_R = np.clip(np.dot(v_R_norm, vertical), -1, 1)
                angle_R = math.degrees(math.acos(cos_R))
            # Use the bigger
            if not math.isnan(angle_L) and not math.isnan(angle_R):
                angle = max(angle_L, angle_R)
            elif not math.isnan(angle_L):
                angle = angle_L
            elif not math.isnan(angle_R):
                angle = angle_R
            else:
                angle = float("nan")
            angles.append((f, angle))

        # Find the maximal angle in this rep
        valid_angles = [(f, a) for f, a in angles if not math.isnan(a)]
        if valid_angles:
            max_frame, max_angle = max(valid_angles, key=lambda x: x[1])
        else:
            max_frame, max_angle = -1, float("nan")

        # Find frame ranges where angle exceeds thresholds
        def find_ranges(threshold):
            # Find contiguous ranges above threshold
            above = [f for f, a in angles if not math.isnan(a) and a > threshold]
            ranges = []
            if not above:
                return ranges
            start = above[0]
            prev = above[0]
            for f in above[1:]:
                if f == prev + 1:
                    prev = f
                else:
                    ranges.append([start, prev])
                    start = f
                    prev = f
            ranges.append([start, prev])
            return ranges

        mild_ranges = find_ranges(MILD_TH)
        severe_ranges = find_ranges(SEV_TH)

        # Compute severity for max angle
        if math.isnan(max_angle):
            severity = "unknown"
        elif max_angle > SEV_TH:
            severity = "severe"
        elif max_angle > MILD_TH:
            severity = "mild"
        else:
            severity = "none"

        recs.append({
            "rep_id": rep_id,
            "mid_frame": mid,
            "max_angle_deg": round(max_angle, 2) if not math.isnan(max_angle) else float("nan"),
            "severity": severity,
            "severe_ranges": severe_ranges,  # List of [start, end] frames
            "mild_ranges": mild_ranges,      # List of [start, end] frames
        })

    df = pd.DataFrame(recs)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ 3D forward-lean report → {os.path.abspath(out_csv)}")
    return df
 


# ------------------------------------------------------------------ CLI (lengths required)
def main() -> None:                # pragma: no cover
    ap = argparse.ArgumentParser("Deviation from vertical per squat rep")
    ap.add_argument("--type",    type=str, required=True)
    ap.add_argument("--keypoints", type=Path, required=True)
    ap.add_argument("--keypoints_3d",   type=Path, required=True)
    ap.add_argument("--reps",      type=Path, required=True)
    ap.add_argument("--lengths",   type=Path, required=True,
                    help="Reference lengths JSON in unit-square scale")
    ap.add_argument("--window",    type=int,  default=5)
    ap.add_argument("--output",    type=Path, default="forward_lean_report.csv")
    args = ap.parse_args()

    kps  = np.load(args.keypoints)
    kps_3d = np.load(args.keypoints_3d)
    reps = pd.read_csv(args.reps)
    lens = _normalize_lengths(json.load(open(args.lengths)))

    if args.type == "3d":
        generate_report_3d(kps_3d, reps, lens,
                           window=args.window,
                           out_csv=args.output)
    else:
        generate_report(kps, reps, lens,
                    window=args.window,
                    out_csv=str(args.output))
    


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------ pipeline helper
def pipeline(
    type: str,
    keypoints: str | Path | np.ndarray,
    keypoints_3d: str | Path | np.ndarray,
    reps: str | Path | pd.DataFrame,
    *,
    lengths_json: str | Path | Mapping,
    window: int = 5,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    In-memory forward-lean analysis, 2D or 3D depending on 'type'.

    Parameters
    ----------
    type         : '2d' or '3d'
    keypoints    : .npy path or ndarray (F,26,3) – unit-square coords (2D)
    keypoints_3d : .npy path or ndarray (F,26,3) – (3D)
    reps         : CSV path or DataFrame with ['rep_id','mid'] or ['rep_mid']
    lengths_json : required – torso lengths in the same unit scale
    window       : ±frames around each mid to search
    output_csv   : optional CSV path
    """
    # --- reps --------------------------------------------------------
    reps_df = reps.copy() if isinstance(reps, pd.DataFrame) else pd.read_csv(reps)
    if "rep_mid" in reps_df.columns and "mid" not in reps_df.columns:
        reps_df = reps_df.rename(columns={"rep_mid": "mid"})

    # --- lengths -----------------------------------------------------
    if lengths_json is None:
        raise ValueError("`lengths_json` is required (torso reference).")

    if isinstance(lengths_json, Mapping):
        raw = lengths_json
    else:
        txt = str(lengths_json).strip()
        raw = json.loads(txt) if txt.startswith("{") else json.load(open(txt))

    lens = _normalize_lengths(raw)

    tmp_out = output_csv or Path(os.devnull)

    # --- main logic: dispatch based on type
    if type == "3d":
        kps_3d = np.load(keypoints_3d) if isinstance(keypoints_3d, (str, Path)) else keypoints_3d
        df = generate_report_3d(kps_3d, reps_df, lens, window=window, out_csv=tmp_out)
    else:
        kps = np.load(keypoints) if isinstance(keypoints, (str, Path)) else keypoints
        df = generate_report(kps, reps_df, lens, window=window, out_csv=tmp_out)

    if output_csv is None:          # silence dummy file path in attrs
        df.attrs.pop("filepath_or_buffer", None)
    return df
