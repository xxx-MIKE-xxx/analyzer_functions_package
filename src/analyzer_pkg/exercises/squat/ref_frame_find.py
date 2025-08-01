#!/usr/bin/env python3
# ref_frame_find.py  – 2025‑07 hot‑fix
# ------------------------------------------------------------------
# Robust “best‑standing‑frame” selector for 2D (HALPE-26 or COCO-17).
# • Finds the frame whose pose **shape** is closest to a 2D reference skeleton (or picks the middle).
# • Never compares to 3D data.
# • Now includes debug plotting for confidence and uprightness.
# ------------------------------------------------------------------

from __future__ import annotations
import logging, os
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ───────────────────── logging setup ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("exercise_analysis")

# ───────────────── joint subsets per exercise ─────────────────
EXERCISE_KEYPOINTS: dict[str, Sequence[int]] = {
    # Squat and default use all 26 for HALPE-26
    "squat":   list(range(26)),
    "default": list(range(26)),
}

# For "uprightness": hips/shoulders
STANDING_Y_IDX = [5, 6, 11, 12]  # LShoulder, RShoulder, LHip, RHip

# ───────────── helper: robust pose distance ────────────────
def _pose_distance(
    frame: np.ndarray,
    ref: np.ndarray,
    kp_idx: Sequence[int],
    *,
    mirror: bool = False
) -> float:
    """
    NaN‑aware, translation‑invariant distance between two 2D poses.
    Optionally mirrors the *frame* horizontally (needed for selfie cams).
    """
    f = frame[kp_idx, :2].copy()
    r = ref[kp_idx, :2].copy()

    if mirror:
        f[:, 0] = 1.0 - f[:, 0]

    # centre both poses at hip‑centre if present, else mean
    def _centre(pts: np.ndarray) -> np.ndarray:
        if 11 in kp_idx and 12 in kp_idx:
            return pts[[kp_idx.index(11), kp_idx.index(12)]].mean(axis=0)
        return np.nanmean(pts, axis=0)

    f -= _centre(f)
    r -= _centre(r)

    valid = ~np.isnan(f).any(axis=1) & ~np.isnan(r).any(axis=1)
    if not valid.any():
        return np.inf

    d = np.linalg.norm(f[valid] - r[valid], axis=1)
    keep = max(1, int(0.75 * len(d)))
    return float(np.sort(d)[:keep].mean())

# ───────────── main analyser class ────────────────
class ExerciseAnalyzer:
    """
    Finds the reference (standing) frame using only 2D keypoints.
    Compatible with legacy API.
    """
    def __init__(
        self,
        file_2d=None,
        file_3d=None,
        reference_type: str = "auto",
        reference_value=None,
        exercise_type: str = "default",
        output_dir: str | None = None,
    ):
        self.file_2d         = file_2d
        self.file_3d         = file_3d  # Ignored, for API compatibility only
        self.reference_type  = reference_type
        self.reference_value = reference_value
        self.exercise_type   = exercise_type
        self.output_dir      = output_dir

        if self.reference_value is not None and self.reference_type == "auto":
            self.reference_type = "file"
            logger.info("Reference provided → switch to reference_type='file'")

        self.skel2d: np.ndarray | None = None
        self.reference_frame_idx: int | None = None
        self.kp_subset = EXERCISE_KEYPOINTS.get(
            exercise_type, EXERCISE_KEYPOINTS["default"]
        )

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def _load(self) -> None:
        if self.file_2d is not None:
            self.skel2d = (
                np.load(self.file_2d) if isinstance(self.file_2d, str)
                else np.asarray(self.file_2d)
            )
            logger.info(f"Using 2‑D  → {self.skel2d.shape}")
        if self.skel2d is None:
            raise ValueError("Need file_2d (2D skeleton)")

    def detect_reference_frame(self) -> None:
        self._load()
        data = self.skel2d
        total_frames = data.shape[0]
        dims         = data.shape[2]

        # 1. explicit frame supplied ----------------------------------
        if self.reference_type == "frame":
            self.reference_frame_idx = int(self.reference_value)
            logger.info(f"Fixed reference frame → {self.reference_frame_idx}")
            return

        # 2. use external reference skeleton (2D only) ----------------
        if self.reference_type == "file":
            ref = (
                np.load(self.reference_value) if isinstance(self.reference_value, str)
                else np.asarray(self.reference_value)
            )
            if ref.shape != (data.shape[1], dims):
                raise ValueError(f"Reference skeleton must be {(data.shape[1], dims)}")

            search_limit = max(1, int(0.30 * total_frames))   # ← 30 %
            distances = np.empty(search_limit, np.float32)

            for f in range(search_limit):
                d  = _pose_distance(data[f], ref, self.kp_subset, mirror=False)
                dm = _pose_distance(data[f], ref, self.kp_subset, mirror=True)
                distances[f] = min(d, dm)

            # smooth a bit for noise
            if search_limit >= 3:
                distances = np.convolve(distances, np.ones(3)/3, mode="same")

            best = int(np.nanargmin(distances))
            self.reference_frame_idx = best
            logger.info(f"Matched reference at frame {best} (first 30 %)")
            return

        # 3. fallback → take the frame with **highest average y** of upright joints with good confidence
        conf_thresh = 0.2
        search_limit = max(1, int(0.30 * total_frames))
        best_avg_y = -np.inf
        best_idx = total_frames // 2  # fallback to middle

        for f in range(search_limit):
            yvals = data[f, STANDING_Y_IDX, 1]
            confs = data[f, STANDING_Y_IDX, 2]
            if np.any(confs < conf_thresh) or np.any(np.isnan(yvals)):
                continue
            avg_y = np.mean(yvals)
            if avg_y > best_avg_y:
                best_avg_y = avg_y
                best_idx = f

        self.reference_frame_idx = best_idx
        logger.info(f"Auto-chosen reference frame {self.reference_frame_idx} (highest upright y in first 30%)")

    def analyze_exercise(self) -> bool:
        try:
            self.detect_reference_frame()
            return True
        except Exception as e:
            logger.error(f"Reference‑frame detection failed: {e}")
            return False

# ───────────── functional wrapper (back‑compat) ─────────────
def run_exercise_analysis(
    file_2d,
    file_3d=None,
    reference_type: str = "auto",
    reference_value=None,
    exercise_type: str = "default",
    output_dir: str | None = None,
):
    """
    Returns the best reference‑frame index (int) or **None** on failure.
    """
    ana = ExerciseAnalyzer(
        file_2d=file_2d,
        file_3d=file_3d,
        reference_type=reference_type,
        reference_value=reference_value,
        exercise_type=exercise_type,
        output_dir=output_dir,
    )
    return ana.reference_frame_idx if ana.analyze_exercise() else None

# ───────────── Debug Plotting ─────────────
def plot_reference_frame(
    skel2d: np.ndarray,
    ref_frame_idx: int,
    out_path: Path,
    conf_thresh: float = 0.2,
    upright_idx: Sequence[int] = STANDING_Y_IDX,
    exercise_type: str = "default"
):
    """
    Plots (1) match score to reference frame for all frames,
          (2) average uprightness (mean y of upright joints),
          and marks the reference frame.
    """
    kp_subset = EXERCISE_KEYPOINTS.get(exercise_type, EXERCISE_KEYPOINTS["default"])
    ref = skel2d[ref_frame_idx]

    # 1. Percent match to reference frame for all frames
    F = skel2d.shape[0]
    scores = np.full(F, np.nan)
    for f in range(F):
        d  = _pose_distance(skel2d[f], ref, kp_subset, mirror=False)
        dm = _pose_distance(skel2d[f], ref, kp_subset, mirror=True)
        score = 1 - min(min(d, dm) / 0.5, 1.0)  # normalize for typical 0-1 scale
        scores[f] = score

    # 2. Average y for uprightness
    avg_y = np.full(F, np.nan)
    for f in range(F):
        yvals = skel2d[f, upright_idx, 1]
        confs = skel2d[f, upright_idx, 2]
        if np.any(confs < conf_thresh) or np.any(np.isnan(yvals)):
            continue
        avg_y[f] = np.mean(yvals)

    # Plotting
    plt.figure(figsize=(14, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(scores, color="tab:blue", label="Match to ref frame (1=identical)")
    ax2.plot(avg_y, color="tab:orange", alpha=0.6, label="Avg. y (uprightness)")
    ax1.axvline(ref_frame_idx, color="red", ls="--", label="Reference frame")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Match to reference", color="tab:blue")
    ax2.set_ylabel("Avg. y (upright joints)", color="tab:orange")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("Reference Frame: Match Score & Uprightness")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=160)
    plt.close()
    print(f"📉  Saved reference frame plot → {out_path}")

# ───────────── CLI for quick testing ─────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Reference frame detection and debug plot (2D only)")
    ap.add_argument("--keypoints", type=str, required=True, help="2D keypoints .npy")
    ap.add_argument("--plot", type=str, default=None, help="Path to save debug plot")
    args = ap.parse_args()
    kps = np.load(args.keypoints)
    ref_frame = run_exercise_analysis(kps)
    print(f"Auto reference frame idx: {ref_frame}")
    if args.plot:
        plot_reference_frame(kps, ref_frame, Path(args.plot))
