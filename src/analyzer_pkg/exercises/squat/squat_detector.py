#!/usr/bin/env python3
"""
squat_detector.py – 2‑D hip‑height repetition detector (unit‑square, y‑up)
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Sequence, List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

RIGHT_HIP, LEFT_HIP = 11, 12

# ------------------------------------------------------------------
def inspect_frames(arr: np.ndarray,
                   *,
                   frames: list[int] | None = None,
                   out_video: Path | None = None,
                   fps: int = 30) -> None:
    # Stub – not used in headless/driver mode
    pass

# ───────────────────── detection core ─────────────────────────────
def detect_repetitions_threshold(
    hip_y: Sequence[float],
    hip_y_ref: float,
    start_frame: int,
    end_frame: int,
    *,
    standing_ratio: float = 0.75,
    depth_ratio:    float = 0.65,
    min_distance:   int   = 3,
    prominence:     float = 1e-4,
    refine_delta:   float = 0.001,
) -> Tuple[List[Dict], float, float, np.ndarray, np.ndarray]:
    stand_thr = hip_y_ref * standing_ratio
    depth_thr = hip_y_ref * depth_ratio
    hip_seg = np.asarray(hip_y[start_frame:end_frame + 1])

    mids, _ = find_peaks(-hip_seg, distance=min_distance, prominence=prominence)
    mids = [m for m in mids if hip_seg[m] <= depth_thr]
    if not mids:
        return [], depth_thr, stand_thr, np.asarray(hip_y), hip_seg

    reps: List[Dict] = []
    rep_id = 1
    for mid in mids:
        s = mid
        while s > 0 and hip_seg[s] < stand_thr: s -= 1
        while s + 1 < mid and hip_seg[s + 1] >= hip_seg[s]: s += 1
        e = mid
        while e < len(hip_seg) - 1 and hip_seg[e] < stand_thr: e += 1
        while e - 1 > mid and hip_seg[e - 1] >= hip_seg[e]: e -= 1

        # ----------- NEW LOGIC: refine start/end using derivative ----------
        # Back up from s to earliest point with slope < -delta (keep falling fast)
        s_ref = s
        while s_ref > 0 and (hip_seg[s_ref] - hip_seg[s_ref-1]) < -refine_delta:
            s_ref -= 1
        # Forward from e to latest point with slope > +delta (keep rising fast)
        e_ref = e
        while e_ref < len(hip_seg) - 1 and (hip_seg[e_ref+1] - hip_seg[e_ref]) > refine_delta:
            e_ref += 1

        if (e_ref - s_ref) < 5:
            continue
        reps.append(dict(rep_id=rep_id,
                         rep_start=s_ref + start_frame,
                         rep_mid=mid + start_frame,
                         rep_end=e_ref + start_frame))
        rep_id += 1
    return reps, depth_thr, stand_thr, np.asarray(hip_y), hip_seg

# ───────────────────────── pipeline ───────────────────────────────
def pipeline(
    keypoints: str | Path | np.ndarray,
    *,
    right_hip_idx: int = RIGHT_HIP,
    left_hip_idx : int = LEFT_HIP,
    start_frame:  int | None = None,
    end_frame:    int | None = None,
    **kwargs,
) -> Tuple[pd.DataFrame, np.ndarray, List[Dict], np.ndarray, np.ndarray]:
    if isinstance(keypoints, (str, Path)):
        keypoints = np.load(keypoints)
    F = keypoints.shape[0]
    if start_frame is None: start_frame = 0
    if end_frame is None:   end_frame   = F - 1

    hip_y = (keypoints[:, right_hip_idx, 1] +
             keypoints[:, left_hip_idx, 1]) / 2.0
    hip_ref = hip_y[start_frame]

    df_reps, depth_thr, stand_thr, hip_full, hip_seg = \
        detect_repetitions_threshold(
            hip_y, hip_ref, start_frame, end_frame, **kwargs)

    df = pd.DataFrame(df_reps, columns=["rep_id", "rep_start", "rep_mid", "rep_end"])
    return df, hip_full, df_reps, hip_seg, hip_full

# ───────────────────────── plotting helper ────────────────────────
def plot_hipy(hip_y: np.ndarray,
              reps: List[Dict],
              stand_thr: float,
              depth_thr: float,
              out: Path) -> None:
    """Save hip‑height‑vs‑frame PNG with rep highlights."""
    plt.figure(figsize=(12, 5))
    plt.plot(hip_y, lw=1.3, label="hip_y (y‑up, 0=bottom)")

    plt.axhline(stand_thr, color="green", ls="--", lw=1, label="standing 75 %")
    plt.axhline(depth_thr, color="red",   ls="--", lw=1, label="depth 65 %")
    for rep in reps:
        s, m, e = rep["rep_start"], rep["rep_mid"], rep["rep_end"]
        plt.axvspan(s, e, color="yellow", alpha=0.25)
        plt.plot([s], [hip_y[s]], "go"); plt.plot([m], [hip_y[m]], "ro")
        plt.plot([e], [hip_y[e]], "bo")

    plt.title("Hip height over time")
    plt.xlabel("Frame"); plt.ylabel("hip_y (unit square)")
    plt.legend(loc="best"); plt.tight_layout()
    plt.savefig(out, dpi=160); plt.close()
    print(f"📉  Saved hip‑height plot → {out}")

# ------------------------------------------------------------------
def save_hip_height_plot(hip_y: np.ndarray,
                         reps: List[Dict],
                         out_path: Path) -> None:
    """Wrapper used by the driver."""
    if reps:
        stand_thr = hip_y[reps[0]["rep_start"]]
    else:
        stand_thr = hip_y[0]
    depth_thr = hip_y.min()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_hipy(hip_y, reps, stand_thr, depth_thr, out_path)

# ───────────────────── CLI for standalone testing ────────────────
def main() -> None:
    ap = argparse.ArgumentParser("Squat repetition detector / debugger")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", default="reps.csv", type=Path)
    ap.add_argument("--plot",   default=None, help="Save hip_y plot PNG")
    ap.add_argument("--inspect",
                    help="Comma‑separated frames to visualise instead of detection")
    args = ap.parse_args()

    if args.inspect:
        # Disabled GUI/interactive
        return

    df, hip_y, reps, _, _ = pipeline(args.input)
    df.to_csv(args.output, index=False)
    print("✅  Repetitions →", args.output)

    if args.plot:
        save_hip_height_plot(hip_y, reps, Path(args.plot))

if __name__ == "__main__":
    main()
