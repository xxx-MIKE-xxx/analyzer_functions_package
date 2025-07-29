#!/usr/bin/env python3
"""
head_orientation_analysis.py
---------------------------
Analyse head pointing direction (pitch & yaw) for each squat repetition.

• **Pitch** (Y-axis)  = Up / Down tilt.
    Vector used: *shoulder-midpoint → nose*  (HALPE-26 idx 0).
    Positive  = looking up  (head back)
    Negative  = looking down (chin toward chest)

• **Yaw** (X-axis)    = Left / Right turn.
    Same vector projected in ground plane.
    We subtract a **heading** angle (athlete rotation) so 0° ≈ facing
    straight ahead relative to torso.

Severity encoding (signed, like foot-analysis):
    +2 severe up/right   | +1 mild up/right
     0 none              | −1 mild down/left | −2 severe down/left

Default thresholds
    pitch_up_mild  = 15°   pitch_up_severe  = 25°
    pitch_dn_mild  = −10°  pitch_dn_severe  = −20°
    yaw_rt_mild    = 15°   yaw_rt_severe    = 25°
    yaw_lt_mild    = −15°  yaw_lt_severe    = −25°

Report columns
    rep_id      int
    severity    list[2]  [pitch_sev, yaw_sev]
    frames      list[2]  [frame_of_max_abs_pitch, frame_of_max_abs_yaw]
    value       list[2]  [pitch_deg, yaw_deg]
"""
from __future__ import annotations
import argparse, math, os, json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd

# HALPE-26 indices
NOSE        = 0
L_SHO, R_SHO = 5, 6

# ------------------------------------------------------------------ helpers

def shoulder_mid(kp: np.ndarray) -> np.ndarray:
    return (kp[L_SHO, :2] + kp[R_SHO, :2]) * 0.5

def vector_pitch_yaw(kp: np.ndarray, heading_deg: float) -> tuple[float, float]:
    """Return (pitch, yaw) in degrees. If NaNs present → (nan, nan)."""
    if np.isnan(kp[[NOSE, L_SHO, R_SHO], :2]).any():
        return float('nan'), float('nan')
    nose = kp[NOSE, :2]
    sho  = shoulder_mid(kp)
    vec  = nose - sho  # image coords (+x right, +y down)

    # Pitch: deviation from vertical.
    # We want 0° = upright; positive = head back/up; negative = head down.
    pitch = 90 - math.degrees(math.atan2(abs(vec[0]), vec[1]))

    # Yaw: horizontal rotation, subtract heading.
    raw_yaw = math.degrees(math.atan2(vec[1], vec[0]))
    yaw = raw_yaw - heading_deg
    return pitch, yaw

# ------------------------------------------------------------------ severity

def classify(v: float,
             mild_pos: float, sev_pos: float,
             mild_neg: float, sev_neg: float) -> int:
    if math.isnan(v):
        return 0
    if v >= sev_pos:
        return 2
    if v >= mild_pos:
        return 1
    if v <= sev_neg:
        return -2
    if v <= mild_neg:
        return -1
    return 0

# ------------------------------------------------------------------ report

def generate_report(kps: np.ndarray,
                    reps: pd.DataFrame,
                    *,
                    heading: float = 0.0,
                    out_csv: str = "head_orientation_report.csv"
                   ) -> pd.DataFrame:

    # thresholds
    p_up_mild, p_up_sev = 15, 25
    p_dn_mild, p_dn_sev = -10, -20
    y_rt_mild, y_rt_sev = 15, 25
    y_lt_mild, y_lt_sev = -15, -25

    records: List[Dict[str, Union[int, List[int], List[float]]]] = []

    for _, row in reps.iterrows():
        start, end = int(row.start), int(row.end)
        pitch_vals = []
        yaw_vals   = []
        for f in range(start, end + 1):
            p, y = vector_pitch_yaw(kps[f], heading)
            pitch_vals.append(p)
            yaw_vals.append(y)

        # find frame of maximum absolute pitch
        if pitch_vals:
            p_arr = np.array(pitch_vals)
            pitch_idx = int(np.nanargmax(np.abs(p_arr)))
            pitch = float(p_arr[pitch_idx])
            pitch_frame = start + pitch_idx
        else:
            pitch, pitch_frame = float('nan'), -1

        # find frame of maximum absolute yaw
        if yaw_vals:
            y_arr = np.array(yaw_vals)
            yaw_idx = int(np.nanargmax(np.abs(y_arr)))
            yaw = float(y_arr[yaw_idx])
            yaw_frame = start + yaw_idx
        else:
            yaw, yaw_frame = float('nan'), -1

        sev_pitch = classify(pitch, p_up_mild, p_up_sev, p_dn_mild, p_dn_sev)
        sev_yaw   = classify(yaw,   y_rt_mild, y_rt_sev, y_lt_mild, y_lt_sev)

        records.append({
            "rep_id":   int(row.rep_id),
            "severity": [sev_pitch, sev_yaw],
            "frames":   [pitch_frame, yaw_frame],
            "value":    [pitch, yaw],
        })

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"saved → {os.path.abspath(out_csv)}")
    return df

# ------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description="Head pitch/yaw report")
    ap.add_argument("--keypoints", type=Path, default="imputed_ma.npy")
    ap.add_argument("--reps",      type=Path, default="repetition_data.csv")
    ap.add_argument("--heading",   type=float, default=0.0,
                    help="global heading (deg) to subtract from yaw")
    ap.add_argument("--output",    type=Path, default="head_orientation_report.csv")
    args = ap.parse_args()

    kps  = np.load(args.keypoints)
    reps = pd.read_csv(args.reps)
    generate_report(kps, reps, heading=args.heading, out_csv=str(args.output))

if __name__ == "__main__":
    main()
