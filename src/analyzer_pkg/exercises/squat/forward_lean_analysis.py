#!/usr/bin/env python3
"""
forward_lean_analysis.py  📏
===========================

Compute *deviation from vertical* for each squat repetition, in 2D or 3D.

Definitions
-----------
In 2D: Hip and Shoulder are (x, y, ±z) in unit-square. 
In 3D: Use actual 3D positions, no length reference required.

Angle = arccos( ⟨torso, vertical⟩ / ||torso|| ).
• Small angle ⇒ upright torso
• Large angle ⇒ forward-lean

Output
------
forward_lean_report.csv (2D/3D) with columns
    rep_id, severity, frame, angle_deg

Changes (2025-08-01)
--------------------
* 3D support (no lengths required)
* Unified pipeline and plot interface for "2d"/"3d" data
"""

from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
from typing import Dict, List, Mapping, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Constants & indices
L_HIP, R_HIP   = 11, 12
L_KNEE, R_KNEE = 13, 14
L_SHO, R_SHO   = 5,  6

CONF_TH_DEF = 0.20       # default confidence
SCALE       = 1000.0     # up-scale to kill 0-1 rounding errors
MILD_TH     = 45.0       # >45° ⇒ mild lean
SEV_TH      = 55.0       # >55° ⇒ severe lean

# ------------------------------------------------------------------
def _normalize_lengths(raw: Mapping[str, float]) -> Dict[str, float]:
    if any(k.startswith("(") for k in raw):       # "(i,j)" style keys
        return {
            "torsoL": float(raw.get(f"({L_HIP},{L_SHO})", 0.0)),
            "torsoR": float(raw.get(f"({R_HIP},{R_SHO})", 0.0)),
        }
    return {k: float(v) for k, v in raw.items()}

def _rel_z(x1: float, y1: float, x2: float, y2: float, L: float) -> float:
    d2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
    return math.sqrt(L * L - d2) if d2 < L * L else 0.0

def _orientation_sign(kps: np.ndarray, mid: int) -> int:
    hip_y0 = kps[max(0, mid - 3), L_HIP, 1]
    hip_y1 = kps[mid, L_HIP, 1]
    return -1 if hip_y1 > hip_y0 else +1

def _vert_torso_angle_2d(
    kp: np.ndarray,
    hip_i: int,
    sho_i: int,
    torso_len: float,
    sign: int,
) -> float:
    hx, hy, hc = kp[hip_i]
    sx, sy, sc = kp[sho_i]
    if min(hc, sc) < CONF_TH_DEF or math.isnan(hx+hy+sx+sy) or torso_len == 0:
        return float("nan")
    hx, hy, sx, sy, L = (hx * SCALE, hy * SCALE, sx * SCALE, sy * SCALE, torso_len * SCALE)
    dz = _rel_z(hx, hy, sx, sy, L)
    v_t = np.array([sx - hx, sy - hy, dz])
    norm = np.linalg.norm(v_t)
    if norm == 0.0:
        return float("nan")
    v_v = np.array([0.0, sign, 0.0])     # vertical reference
    cosang = np.clip(np.dot(v_t, v_v) / norm, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def _frame_deviation_2d(kp: np.ndarray, lens: Mapping[str, float], sign: int) -> float:
    left  = _vert_torso_angle_2d(kp, L_HIP, R_SHO, lens["torsoL"], sign)
    right = _vert_torso_angle_2d(kp, R_HIP, L_SHO, lens["torsoR"], sign)
    vals = [a for a in (left, right) if not math.isnan(a)]
    return max(vals) if vals else float("nan")

def _frame_deviation_3d(kp: np.ndarray) -> float:
    # Use both (hip, opp shoulder) pairs; vertical is y+
    angles = []
    for hip_idx, sho_idx in [(L_HIP, R_SHO), (R_HIP, L_SHO)]:
        hip = kp[hip_idx]
        sho = kp[sho_idx]
        if np.any(np.isnan(hip)) or np.any(np.isnan(sho)):
            continue
        torso = sho - hip
        norm = np.linalg.norm(torso)
        if norm < 1e-6:
            continue
        vertical = np.array([0, 0, 1])   # MotionBERT: z is up
        cosang = np.clip(np.dot(torso, vertical) / norm, -1.0, 1.0)
        angle = math.degrees(math.acos(cosang))
        angles.append(angle)
    return max(angles) if angles else float("nan")

def _severity(a: float) -> str:
    if math.isnan(a):
        return "unknown"
    if a > SEV_TH:
        return "severe"
    if a > MILD_TH:
        return "mild"
    return "none"

# ------------------------------------------------------------------
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
            _frame_deviation_2d(kps[f], lens, sign)
            for f in frames
            if not math.isnan(_frame_deviation_2d(kps[f], lens, sign))
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
    *,
    window: int = 5,
    out_csv: str = "forward_lean_report_3d.csv",
) -> pd.DataFrame:
    recs: List[Dict[str, Union[int, float, str]]] = []
    F = kps_3d.shape[0]
    for _, row in reps.iterrows():
        rep_id = int(row.rep_id)
        mid = int(row.mid)
        frames = range(max(0, mid - window), min(F, mid + window + 1))
        vals = [
            _frame_deviation_3d(kps_3d[f])
            for f in frames
            if not math.isnan(_frame_deviation_3d(kps_3d[f]))
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
    print(f"✅ 3D forward-lean report → {os.path.abspath(out_csv)}")
    return df

# ------------------------------------------------------------------
def plot_forward_lean(
    type: str,
    keypoints: np.ndarray,
    keypoints_3d: np.ndarray,
    reps: pd.DataFrame,
    out_path: str | Path,
    lengths_json: str | Path | dict = None,
    window: int = 5,
):
    """
    Plot per-frame forward-lean angle for each rep (2D or 3D).
    """

    if "rep_mid" in reps.columns and "mid" not in reps.columns:
        reps = reps.rename(columns={"rep_mid": "mid"})
    frame_nums = []
    angles = []

    if type == "3d":
        for _, row in reps.iterrows():
            mid = int(row.mid)
            frames = range(max(0, mid - window), min(keypoints_3d.shape[0], mid + window + 1))
            for f in frames:
                a = _frame_deviation_3d(keypoints_3d[f])
                if not math.isnan(a):
                    frame_nums.append(f)
                    angles.append(a)
    else:
        if lengths_json is None:
            raise ValueError("`lengths_json` is required for 2D plotting.")
        if isinstance(lengths_json, dict):
            lens = _normalize_lengths(lengths_json)
        else:
            txt = str(lengths_json).strip()
            lens = _normalize_lengths(json.loads(txt)) if txt.startswith("{") else _normalize_lengths(json.load(open(txt)))
        for _, row in reps.iterrows():
            mid = int(row.mid)
            sign = _orientation_sign(keypoints, mid)
            frames = range(max(0, mid - window), min(keypoints.shape[0], mid + window + 1))
            for f in frames:
                a = _frame_deviation_2d(keypoints[f], lens, sign)
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

# ------------------------------------------------------------------
def pipeline(
    type: str,
    keypoints: str | Path | np.ndarray,
    keypoints_3d: str | Path | np.ndarray,
    reps: str | Path | pd.DataFrame,
    *,
    lengths_json: str | Path | Mapping = None,
    window: int = 5,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    reps_df = reps.copy() if isinstance(reps, pd.DataFrame) else pd.read_csv(reps)
    if "rep_mid" in reps_df.columns and "mid" not in reps_df.columns:
        reps_df = reps_df.rename(columns={"rep_mid": "mid"})

    tmp_out = output_csv or Path(os.devnull)
    if type == "3d":
        kps_3d = np.load(keypoints_3d) if isinstance(keypoints_3d, (str, Path)) else keypoints_3d
        df = generate_report_3d(kps_3d, reps_df, window=window, out_csv=tmp_out)
    else:
        if lengths_json is None:
            raise ValueError("`lengths_json` is required for 2D pipeline.")
        if isinstance(lengths_json, Mapping):
            raw = lengths_json
        else:
            txt = str(lengths_json).strip()
            raw = json.loads(txt) if txt.startswith("{") else json.load(open(txt))
        lens = _normalize_lengths(raw)
        kps = np.load(keypoints) if isinstance(keypoints, (str, Path)) else keypoints
        df = generate_report(kps, reps_df, lens, window=window, out_csv=tmp_out)
    if output_csv is None:
        df.attrs.pop("filepath_or_buffer", None)
    return df

# ------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Deviation from vertical per squat rep (2D/3D)")
    ap.add_argument("--type",    type=str, required=True)
    ap.add_argument("--keypoints", type=Path, required=False)
    ap.add_argument("--keypoints_3d", type=Path, required=False)
    ap.add_argument("--reps",    type=Path, required=True)
    ap.add_argument("--lengths", type=Path, required=False,
                    help="Reference lengths JSON (unit-square scale, for 2D)")
    ap.add_argument("--window",  type=int,  default=5)
    ap.add_argument("--output",  type=Path, default="forward_lean_report.csv")
    args = ap.parse_args()

    reps = pd.read_csv(args.reps)
    if args.type == "3d":
        kps_3d = np.load(args.keypoints_3d)
        generate_report_3d(kps_3d, reps, window=args.window, out_csv=args.output)
    else:
        kps = np.load(args.keypoints)
        lens = _normalize_lengths(json.load(open(args.lengths)))
        generate_report(kps, reps, lens, window=args.window, out_csv=str(args.output))
