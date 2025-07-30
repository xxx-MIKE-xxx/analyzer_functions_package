#!/usr/bin/env python3
# ref_frame_find.py  – 2025‑07 hot‑fix
# ------------------------------------------------------------------
# Robust “best‑standing‑frame” selector
#
# • Finds the frame whose pose **shape** is closest to a reference
#   skeleton (or picks the middle of the video if no reference given).
# • Translation‑invariant, NaN‑aware, tolerant to mirror flips.
# • Now scans the **first 30 %** of frames and, for squats, uses
#   **all 17 COCO key‑points**.
# ------------------------------------------------------------------
from __future__ import annotations

import logging, os
from typing import Sequence

import numpy as np

# ───────────────────── logging setup ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("exercise_analysis")

# ───────────────── joint subsets per exercise ─────────────────
EXERCISE_KEYPOINTS: dict[str, Sequence[int]] = {
    # use the *full* skeleton for squats
    "squat":   list(range(17)),                 # 0 … 16
    # legacy defaults for other movements
    "pushup":  [5, 6, 7, 8,  9, 10],            # shoulders → wrists
    "plank":   [5, 6, 11, 12],                  # shoulders + hips
    "default": list(range(17)),
}

# ───────────── helper: robust pose distance ────────────────
def _pose_distance(frame: np.ndarray,
                   ref: np.ndarray,
                   kp_idx: Sequence[int],
                   *,
                   mirror: bool = False) -> float:
    """
    NaN‑aware, translation‑invariant distance between two poses.
    Optionally mirrors the *frame* horizontally (needed for selfie cams).
    """
    f = frame[kp_idx].copy()
    r = ref[kp_idx].copy()

    if mirror:
        # x ∈ [0,1]; flip horizontally
        f[:, 0] = 1.0 - f[:, 0]

    # centre both poses at hip‑centre (if available) or overall mean
    def _centre(pts: np.ndarray) -> np.ndarray:
        if 11 in kp_idx and 12 in kp_idx:
            return pts[[kp_idx.index(11), kp_idx.index(12)]].mean(axis=0)
        return np.nanmean(pts, axis=0)

    f -= _centre(f)
    r -= _centre(r)

    # per‑joint distance, ignoring NaNs
    valid = ~np.isnan(f).any(axis=1) & ~np.isnan(r).any(axis=1)
    if not valid.any():
        return np.inf

    d = np.linalg.norm(f[valid] - r[valid], axis=1)

    # trimmed mean (drop the noisiest 25 %)
    keep = max(1, int(0.75 * len(d)))
    return float(np.sort(d)[:keep].mean())


# ───────────────────── main analyser class ────────────────────
class ExerciseAnalyzer:
    """
    Finds the reference (standing) frame.

    Parameters
    ----------
    file_2d / file_3d : path or ndarray
        Normalised unit‑square keypoint sequences, shape (F,17,2/3).
    reference_type : "auto" | "frame" | "file"
    reference_value:
        • int  – exact frame index (if reference_type=="frame")
        • path/ndarray – reference skeleton (if reference_type=="file")
    exercise_type  : e.g. "squat"
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
        self.file_3d         = file_3d
        self.reference_type  = reference_type
        self.reference_value = reference_value
        self.exercise_type   = exercise_type
        self.output_dir      = output_dir

        if self.reference_value is not None and self.reference_type == "auto":
            self.reference_type = "file"
            logger.info("Reference provided → switch to reference_type='file'")

        self.skel2d: np.ndarray | None = None
        self.skel3d: np.ndarray | None = None
        self.reference_frame_idx: int | None = None

        self.kp_subset = EXERCISE_KEYPOINTS.get(
            exercise_type, EXERCISE_KEYPOINTS["default"]
        )

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # ───────────── data loading helpers ──────────────
    def _load(self) -> None:
        if self.file_2d is not None:
            self.skel2d = (
                np.load(self.file_2d) if isinstance(self.file_2d, str)
                else np.asarray(self.file_2d)
            )
            logger.info(f"Using 2‑D  → {self.skel2d.shape}")

        if self.file_3d is not None:
            self.skel3d = (
                np.load(self.file_3d) if isinstance(self.file_3d, str)
                else np.asarray(self.file_3d)
            )
            logger.info(f"Using 3‑D  → {self.skel3d.shape}")

        if self.skel2d is None and self.skel3d is None:
            raise ValueError("Need at least one of file_2d / file_3d")

    # ───────────── reference‑frame search ─────────────
    def detect_reference_frame(self) -> None:
        self._load()
        data = self.skel2d if self.skel2d is not None else self.skel3d
        total_frames = data.shape[0]
        dims         = data.shape[2]

        # 1. explicit frame supplied ----------------------------------
        if self.reference_type == "frame":
            self.reference_frame_idx = int(self.reference_value)
            logger.info(f"Fixed reference frame → {self.reference_frame_idx}")
            return

        # 2. use external reference skeleton --------------------------
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

            # light 3‑pt running mean to smooth noise
            if search_limit >= 3:
                distances = np.convolve(distances, np.ones(3)/3, mode="same")

            best = int(np.nanargmin(distances))
            self.reference_frame_idx = best
            logger.info(f"Matched reference at frame {best} (first 30 %)")
            return

        # 3. fallback → simply take the middle frame ------------------
        self.reference_frame_idx = total_frames // 2
        logger.info(f"Auto‑chosen reference frame {self.reference_frame_idx}")

    # ───────────── convenience façade ─────────────
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
