#!/usr/bin/env python3
"""
squat_depth_analysis.py  🏋️‍♀️  (signed-depth + hip-height version)

What’s new (2025-08-01)
-----------------------
* Plot shows only hip y - knee y (signed) for both sides.
* Kept all original reporting logic, now modular for your pipeline.

Public API
----------
generate_report(...)
pipeline(...)                 # returns DataFrame
plot_squat_depth(...)         # NEW: shows hip-knee y-diff for both sides
"""

from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
from typing import Dict, List, Mapping, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants & indices
# --------------------------------------------------------------------------
# HALPE-26 indices
L_HIP, R_HIP   = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK   = 15, 16
JOINTS = {"left": (L_HIP, L_KNEE, L_ANK),
          "right": (R_HIP, R_KNEE, R_ANK)}

CONF_TH_DEF   = 0.20      # default confidence threshold
SCALE         = 1000.0    # coordinate up-scaling factor
HIP_MILD_TH   = 0.02      # hip depth ≤ 2 % of frame → mild
KNEE_MILD_TH  = 85.0      # angle > 85°  → mild
KNEE_SEV_TH   = 95.0      # angle > 95°  → severe

# --------------------------------------------------------------------------
# Plot: Hip y - Knee y for both sides
# --------------------------------------------------------------------------
def plot_squat_depth(
    keypoints: np.ndarray,
    reps: pd.DataFrame,
    out_path: str | Path,
    conf_thresh: float = 0.0,
):
    """
    Plot hip_y - knee_y for both left and right sides over all frames.
    Mark rep midpoints and rep start/end with vertical lines.
    """
    if "rep_mid" in reps.columns and "mid" not in reps.columns:
        reps = reps.rename(columns={"rep_mid": "mid"})
    if "rep_start" in reps.columns and "start" not in reps.columns:
        reps = reps.rename(columns={"rep_start": "start"})
    if "rep_end" in reps.columns and "end" not in reps.columns:
        reps = reps.rename(columns={"rep_end": "end"})
    N = keypoints.shape[0]
    left_depth = np.full(N, np.nan)
    right_depth = np.full(N, np.nan)

    # Compute depth for every frame
    for f in range(N):
        fkp = keypoints[f]
        # Left
        if all(fkp[idx, 2] >= conf_thresh for idx in (L_HIP, L_KNEE)):
            left_depth[f] = fkp[L_HIP, 1] - fkp[L_KNEE, 1]
        # Right
        if all(fkp[idx, 2] >= conf_thresh for idx in (R_HIP, R_KNEE)):
            right_depth[f] = fkp[R_HIP, 1] - fkp[R_KNEE, 1]

    plt.figure(figsize=(14, 5))
    plt.plot(left_depth, label="Left (hip y - knee y)", color="blue", alpha=0.8)
    plt.plot(right_depth, label="Right (hip y - knee y)", color="red", alpha=0.8)
    plt.axhline(0.0, color="black", ls="--", lw=1, label="Hip at knee level")
    plt.axhline(HIP_MILD_TH, color="orange", ls="--", lw=1, label="Mild threshold (0.02)")
    plt.xlabel("Frame")
    plt.ylabel("Hip depth minus knee (unit square)")
    plt.title("Hip-Knee Y Difference Over Frames (Squat Depth)")
    
    # Mark rep midpoints and start/end with vertical lines
    for _, row in reps.iterrows():
        start, end, mid = int(row.start), int(row.end), int(row.mid)
        plt.axvline(start, color="green", ls="--", lw=1, alpha=0.3)
        plt.axvline(end, color="green", ls="--", lw=1, alpha=0.3)
        plt.scatter(mid, left_depth[mid], color="blue", s=40, zorder=5)
        plt.scatter(mid, right_depth[mid], color="red", s=40, zorder=5)

    plt.legend()
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"📈  Saved squat-depth plot → {out_path}")


# --------------------------------------------------------------------------
# Length helpers (unchanged)
# --------------------------------------------------------------------------
def _get_len(pairs: Mapping[str, float], i: int, j: int) -> float | None:
    for key in (f"({i},{j})", f"({i}, {j})", f"({j},{i})", f"({j}, {i})"):
        if key in pairs:
            return float(pairs[key])
    return None

def _normalize_lengths(raw: Mapping[str, float]) -> dict[str, float]:
    if any(k.startswith("(") for k in raw):
        return {
            "left_shin":  _get_len(raw, L_KNEE, L_ANK),
            "right_shin": _get_len(raw, R_KNEE, R_ANK),
            "left_thigh":  _get_len(raw, L_HIP,  L_KNEE),
            "right_thigh": _get_len(raw, R_HIP,  R_KNEE),
        }
    return {k: float(v) for k, v in raw.items()}

# --------------------------------------------------------------------------
# Main per-rep report logic (unchanged)
# --------------------------------------------------------------------------
def _sign_orientation(kps: np.ndarray, mid: int) -> int:
    hip_idx = L_HIP
    pre = kps[max(0, mid - 3), hip_idx, 1]
    at  = kps[mid, hip_idx, 1]
    return +1 if at > pre else -1

def _frame_metrics(
    fkp: np.ndarray,
    lens: Mapping[str, float],
    conf_th: float,
    sign: int,
) -> tuple[list[float], list[float]]:
    angles, depths = [], []
    for side, (h, k, a) in JOINTS.items():
        if any(fkp[idx, 2] < conf_th for idx in (h, k, a)):
            continue
        ax, ay = fkp[a, :2] * SCALE
        kx, ky = fkp[k, :2] * SCALE
        hx, hy = fkp[h, :2] * SCALE

        shin_len  = lens[f"{side}_shin"]  * SCALE
        thigh_len = lens[f"{side}_thigh"] * SCALE

        knee_z = _rel_z(ax, ay, kx, ky, shin_len)
        hip_z  = knee_z - _rel_z(kx, ky, hx, hy, thigh_len)

        hip   = np.array([hx, hy, hip_z])
        knee  = np.array([kx, ky, knee_z])
        ankle = np.array([ax, ay, 0.0])

        theta = _angle(hip, knee, ankle)
        angles.append(theta)

        hip_y, knee_y = fkp[h, 1], fkp[k, 1]
        depth = hip_y - knee_y
        depths.append(depth)

    return angles, depths

def _rel_z(x1: float, y1: float, x2: float, y2: float, L: float) -> float:
    d2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
    return 0.0 if d2 >= L * L else math.sqrt(L * L - d2)

def _angle(hip: np.ndarray, knee: np.ndarray, ank: np.ndarray) -> float:
    v1, v2 = hip - knee, ank - knee
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    if den == 0:
        return float("nan")
    return math.degrees(math.acos(np.clip(float(np.dot(v1, v2) / den), -1, 1)))

def _angle_sev(a: float) -> str:
    if math.isnan(a): return "unknown"
    if a > KNEE_SEV_TH: return "severe"
    if a > KNEE_MILD_TH: return "mild"
    return "none"

def _hip_sev(d: float) -> str:
    if math.isnan(d): return "unknown"
    if d <= 0.0: return "severe"
    if d <= HIP_MILD_TH: return "mild"
    return "none"

def generate_report(
    kps: np.ndarray,
    reps: pd.DataFrame,
    lens: Mapping[str, float],
    conf_th: float,
    out: str | Path,
) -> pd.DataFrame:
    recs: List[Dict[str, Union[int, float, str]]] = []
    for _, row in reps.iterrows():
        mid = int(row.mid)
        sign = _sign_orientation(kps, mid)
        win = range(max(0, mid - 5), min(kps.shape[0], mid + 6))
        angles_all, depths_all = [], []
        for f in win:
            angs, deps = _frame_metrics(kps[f], lens, conf_th, sign)
            angles_all.extend(angs)
            depths_all.extend(deps)
        angle_min  = min(angles_all)  if angles_all  else float("nan")
        depth_max  = max(depths_all)  if depths_all  else float("nan")
        recs.append({
            "rep_id":          int(row.rep_id),
            "frame":           mid,
            "angle_deg":       round(angle_min, 2) if not math.isnan(angle_min) else float("nan"),
            "angle_severity":  _angle_sev(angle_min),
            "hip_depth":       round(depth_max, 4) if not math.isnan(depth_max) else float("nan"),
            "hip_severity":    _hip_sev(depth_max),
        })
    df = pd.DataFrame(recs)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("✅ squat-depth report →", out.resolve())
    return df

# --------------------------------------------------------------------------
# CLI entry-point
# --------------------------------------------------------------------------
if __name__ == "__main__":          # pragma: no cover
    p = argparse.ArgumentParser("Squat-depth reporter (angle + hip level)")
    p.add_argument("--keypoints", type=Path, default="2d.npy")
    p.add_argument("--reps",      type=Path, default="repetition_data.csv")
    p.add_argument("--lengths",   type=Path, required=True,
                   help="Reference lengths JSON (unit-scale)")
    p.add_argument("--conf",      type=float, default=CONF_TH_DEF)
    p.add_argument("--out",       type=Path,  default="squat_depth_report.csv")
    a = p.parse_args()

    kps  = np.load(a.keypoints)
    reps = pd.read_csv(a.reps)
    lens = _normalize_lengths(json.load(open(a.lengths)))

    generate_report(kps, reps, lens, a.conf, a.out)

# --------------------------------------------------------------------------
# Pipeline helper
# --------------------------------------------------------------------------
def pipeline(
    keypoints: str | Path | np.ndarray,
    reps: str | Path | pd.DataFrame,
    *,
    lengths_json: str | Path | Mapping,
    conf_thresh: float = 0.0,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    In-memory squat depth analysis. (Report unchanged)
    """
    kps = np.load(keypoints) if isinstance(keypoints, (str, Path)) else keypoints
    reps_df = reps.copy() if isinstance(reps, pd.DataFrame) else pd.read_csv(reps)
    if "rep_mid" in reps_df.columns and "mid" not in reps_df.columns:
        reps_df = reps_df.rename(columns={"rep_mid": "mid"})
    if "rep_start" in reps_df.columns and "start" not in reps_df.columns:
        reps_df = reps_df.rename(columns={"rep_start": "start"})
    if "rep_end" in reps_df.columns and "end" not in reps_df.columns:
        reps_df = reps_df.rename(columns={"rep_end": "end"})
    if lengths_json is None:
        raise ValueError("`lengths_json` is required (shin/thigh reference).")
    if isinstance(lengths_json, Mapping):
        raw_lengths = lengths_json
    else:
        txt = str(lengths_json).strip()
        raw_lengths = (
            json.loads(txt) if txt.startswith("{") else json.load(open(txt))
        )
    lens = _normalize_lengths(raw_lengths)
    tmp_out = output_csv or Path(os.devnull)
    df = generate_report(kps, reps_df, lens, conf_thresh, tmp_out)
    if output_csv is None:
        df.attrs.pop("filepath_or_buffer", None)
    return df
