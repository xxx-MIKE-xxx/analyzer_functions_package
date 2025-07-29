#!/usr/bin/env python3
# ref_frame_find.py              (2025-07 hot-fix)
# ──────────────────────────────────────────────────────────────────────────────
# Robust “best-standing-frame” selector
#
#  • Finds the frame whose pose **shape** matches a reference skeleton
#     (or, if no reference provided, just returns the middle frame).
#  • Ignores global translation, handles NaNs, and is tolerant to
#    mirror-flipped recordings.
#  • API is *drop-in compatible* with the previous version.
# ──────────────────────────────────────────────────────────────────────────────
import os, logging, numpy as np
from typing import Sequence

# ── logging -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("exercise_analysis")

# ── joint subsets per exercise ---------------------------------------
EXERCISE_KEYPOINTS: dict[str, Sequence[int]] = {
    "squat":   [11, 12, 13, 14, 15, 16],        # hips → ankles
    "pushup":  [ 5,  6,  7,  8,  9, 10],        # shoulders → wrists
    "plank":   [ 5,  6, 11, 12],                # shoulders + hips
    "default": list(range(17)),                 # all 17 COCO joints
}

# ── helper: robust distance frame ↔ reference ------------------------
def _pose_distance(frame: np.ndarray,
                   ref: np.ndarray,
                   kp_idx: Sequence[int],
                   *,
                   mirror: bool = False) -> float:
    """
    Translation-invariant, NaN-aware, optional left/right-mirror distance.
    frame, ref : (J, D) arrays   D = 2 or 3
    """
    # 1) restrict to the joints we care about
    f = frame[kp_idx].copy()
    r = ref[kp_idx].copy()

    if mirror:
        # horizontal mirror: X → 1-X  (works for unit-square coords)
        f[:, 0] = 1.0 - f[:, 0]

    # 2) subtract hip-centre to remove global XY offset (if hips available)
    #    – fall back to mean over kp_idx otherwise
    def _centre(pts: np.ndarray) -> np.ndarray:
        if 11 in kp_idx and 12 in kp_idx:
            return pts[[kp_idx.index(11), kp_idx.index(12)]].mean(axis=0)
        return np.nanmean(pts, axis=0)

    f -= _centre(f)
    r -= _centre(r)

    # 3) per-joint Euclidean distance, ignoring NaNs
    valid = ~np.isnan(f).any(axis=1) & ~np.isnan(r).any(axis=1)
    if not valid.any():
        return np.inf                      # no usable joints in this frame

    d = np.linalg.norm(f[valid] - r[valid], axis=1)

    # 4) robust aggregate – trimmed mean (drop top 25 %)
    k = max(1, int(0.75 * len(d)))
    return np.sort(d)[:k].mean()


# ── main class --------------------------------------------------------
class ExerciseAnalyzer:
    def __init__(
        self,
        file_2d=None,
        file_3d=None,
        reference_type: str = "auto",     # "auto" | "frame" | "file"
        reference_value=None,             # int | str/ndarray
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
            logger.info("Reference value provided → switching reference_type to 'file'")

        self.skeleton_2d: np.ndarray | None = None
        self.skeleton_3d: np.ndarray | None = None
        self.reference_frame_idx: int | None = None

        self.keypoints = EXERCISE_KEYPOINTS.get(exercise_type,
                                                EXERCISE_KEYPOINTS["default"])

        logger.info(f"Initializing analyzer for '{exercise_type}'")
        logger.info(f"Reference mode: '{self.reference_type}'")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # ── data loading ----------------------------------------------------
    def _load(self) -> bool:
        try:
            if self.file_2d is not None:
                self.skeleton_2d = (
                    np.load(self.file_2d) if isinstance(self.file_2d, str)
                    else np.asarray(self.file_2d)
                )
                logger.info(f"Using 2D → {self.skeleton_2d.shape}")

            if self.file_3d is not None:
                self.skeleton_3d = (
                    np.load(self.file_3d) if isinstance(self.file_3d, str)
                    else np.asarray(self.file_3d)
                )
                logger.info(f"Using 3D → {self.skeleton_3d.shape}")

            if self.skeleton_2d is None and self.skeleton_3d is None:
                raise ValueError("Need at least one of file_2d / file_3d")

            return True
        except Exception as e:
            logger.error(f"Loading error: {e}")
            return False

    # ── reference-frame detection --------------------------------------
    def detect_reference_frame(self) -> None:
        # 1) fixed idx supplied
        if self.reference_type == "frame":
            self.reference_frame_idx = int(self.reference_value)
            logger.info(f"Fixed reference frame → {self.reference_frame_idx}")
            return

        # 2) pick data stream (prefer 2-D)
        data = self.skeleton_2d if self.skeleton_2d is not None else self.skeleton_3d
        total = data.shape[0]
        dims  = data.shape[2]

        # 3) external reference skeleton supplied
        if self.reference_type == "file":
            ref = (
                np.load(self.reference_value) if isinstance(self.reference_value, str)
                else np.asarray(self.reference_value)
            )
            if ref.shape != (data.shape[1], dims):
                raise ValueError(f"Reference skeleton must be {(data.shape[1], dims)}")

            # distance curve for original + mirrored pose
            cutoff = max(1, int(0.2 * total))
            dists  = np.zeros(cutoff)

            for f in range(cutoff):
                d  = _pose_distance(data[f], ref, self.keypoints, mirror=False)
                dm = _pose_distance(data[f], ref, self.keypoints, mirror=True)
                dists[f] = min(d, dm)

            # light smoothing
            if cutoff >= 3:
                dists = np.convolve(dists, np.ones(3)/3, mode="same")

            best = int(np.nanargmin(dists))
            self.reference_frame_idx = best
            logger.info(f"Matched reference at frame {best} (first 20 %)")
            return

        # 4) auto = pick middle frame
        self.reference_frame_idx = total // 2
        logger.info(f"Auto-chosen reference frame {self.reference_frame_idx}")

    # ── public driver ---------------------------------------------------
    def analyze_exercise(self) -> bool:
        if not self._load():
            return False
        self.detect_reference_frame()
        return True


# ── functional wrapper (kept for back-compat) ------------------------
def run_exercise_analysis(
    file_2d,
    file_3d=None,
    reference_type: str = "auto",
    reference_value=None,
    exercise_type: str = "default",
    output_dir: str | None = None,
):
    """Return best reference frame index (or None on failure)."""
    ana = ExerciseAnalyzer(
        file_2d=file_2d,
        file_3d=file_3d,
        reference_type=reference_type,
        reference_value=reference_value,
        exercise_type=exercise_type,
        output_dir=output_dir,
    )
    return ana.reference_frame_idx if ana.analyze_exercise() else None
