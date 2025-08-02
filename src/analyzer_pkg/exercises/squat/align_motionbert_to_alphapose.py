#!/usr/bin/env python3
"""
align_motionbert_to_alphapose.py
--------------------------------
Estimate a *similarity transform* (scale *s*, rotation *R*, translation *t*)
that aligns raw 3-D MotionBERT skeleton coordinates to the raw 2-D AlphaPose
detections (HALPE-26).  Subsequent biomechanics metrics can then work in an
upright, image-consistent frame.

Only **imports** and **global constants / mappings** are declared below.
The rest of the implementation will follow in the next steps.
"""

# ── standard library ─────────────────────────────────────────────────────
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Sequence

# ── third-party ───────────────────────────────────────────────────────────
import numpy as np   # numerical backbone

# ── joint mapping (HALPE-26 → H36M-17) ────────────────────────────────────
# These indices let us pick *corresponding* keypoints between the two
# modalities.  Left/Right hips both map to the H36M “root” (pelvis) because
# MotionBERT’s 17-joint set has a single pelvis joint.
JOINT_MAP = {
    5:  11,   # LShoulder → LShoulder
    6:  14,   # RShoulder → RShoulder
    7:  12,   # LElbow    → LElbow
    8:  15,   # RElbow    → RElbow
    9:  13,   # LWrist    → LWrist
    10: 16,   # RWrist    → RWrist
    11: 4,    # LHip      → LHip
    12: 1,    # RHip      → RHip
    13: 5,    # LKnee     → LKnee
    14: 2,    # RKnee     → RKnee
    15: 6,    # LAnkle    → LAnkle
    16: 3,    # RAnkle    → RAnkle
}

# Convenience lists with identical ordering for 2-D and 3-D selections.
IDX_2D: Sequence[int] = list(JOINT_MAP.keys())          # in AlphaPose arrays
IDX_3D: Sequence[int] = [JOINT_MAP[i] for i in IDX_2D]  # in MotionBERT arrays

from typing import Sequence, Tuple






def load_raw_alphapose(json_path: str | Path) -> np.ndarray:
    """
    Load *raw* AlphaPose detections from a JSON/JSONL file and return an array
    shaped (F, 26, 3) in the original coordinate units (typically pixels).

    The reader:
      • accepts a JSON array OR JSON-lines (one object per line)
      • picks the best person per frame (highest 'score' if available,
        otherwise highest mean keypoint confidence)
      • infers frame order using one of: 'frame_id' | 'image_id' | 'idx'
      • validates there are exactly 26 joints (HALPE-26)

    Raises
    ------
    FileNotFoundError, ValueError on malformed inputs.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"AlphaPose JSON not found: {json_path}")

    # --- read JSON or JSONL safely -------------------------------------
    try:
        text = json_path.read_text(encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Failed to read {json_path}: {e}") from e

    records = None
    try:
        # Try JSON array first
        parsed = json.loads(text)
        if isinstance(parsed, list):
            records = parsed
        elif isinstance(parsed, dict):
            # Some tools wrap results in a dict with 'result' key
            records = parsed.get("result", [])
        else:
            records = None
    except Exception:
        # Fallback: JSON lines
        records = []
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                records.append(json.loads(ln))
            except Exception as e:
                raise ValueError(
                    f"Invalid JSON line in {json_path}: {ln[:120]}..."
                ) from e

    if not records:
        raise ValueError(f"No detections found in {json_path}")

    # --- group by frame id, keep best person per frame ------------------
    def _frame_key(rec: dict) -> int:
        import re
        if "frame_id" in rec:
            return int(rec["frame_id"])
        if "idx" in rec:
            return int(rec["idx"])
        if "image_id" in rec:
            s = str(rec["image_id"])
            m = re.findall(r"\d+", s)
            if m:
                return int(m[-1])
        # Fallback – unknown
        return -1

    buckets: dict[int, list[dict]] = {}
    # NOTE: enumerate so each unknown frame gets its own id
    for i, rec in enumerate(records):
        if "keypoints" not in rec:
            raise ValueError("AlphaPose record missing 'keypoints' field")
        kps = rec["keypoints"]
        if not isinstance(kps, (list, tuple)) or len(kps) % 3 != 0:
            raise ValueError("AlphaPose 'keypoints' must be a flat list of x,y,conf triples")
        J = len(kps) // 3
        if J != 26:
            raise ValueError(
                f"Expected 26 joints (HALPE-26), got {J}. "
                "Ensure the detector exported HALPE-26."
            )
        fid = _frame_key(rec)
        if fid < 0:
            fid = i  # ← key change: fallback to a unique, per-record id
        buckets.setdefault(fid, []).append(rec)

    # Create sorted frame list
    frame_ids = sorted(buckets.keys())  # ← simpler & correct after the change

    F = len(frame_ids)
    out = np.full((F, 26, 3), np.nan, dtype=np.float32)

    for i, fid in enumerate(frame_ids):
        candidates = buckets[fid]
        def _cand_score(rec: dict) -> float:
            if "score" in rec and isinstance(rec["score"], (int, float)):
                return float(rec["score"])
            arr = np.array(rec["keypoints"], dtype=np.float64).reshape(-1, 3)
            return float(np.nanmean(arr[:, 2]))

        best = max(candidates, key=_cand_score)
        arr = np.array(best["keypoints"], dtype=np.float64).reshape(26, 3)
        if not np.isfinite(arr[:, :2]).all():
            raise ValueError(f"Non-finite AlphaPose coordinates at frame {fid}")
        out[i] = arr.astype(np.float32)

    return out


def load_raw_motionbert(x3d_path: str | Path) -> np.ndarray:
    """
    Load *raw* MotionBERT 3-D coordinates and return array (F, 17, 3)
    as float32. Accepts several common layouts:

      1) Directory containing 'X3D.npy'
      2) File '3d-pose-results.npz' (NumPy .npz) with key 'X3D'
      3) Plain '.npy' file shaped (F,17,3)

    Raises
    ------
    FileNotFoundError, ValueError on malformed inputs.
    """
    x3d_path = Path(x3d_path)
    if not x3d_path.exists():
        raise FileNotFoundError(f"MotionBERT path not found: {x3d_path}")

    poses = None

    if x3d_path.is_dir():
        candidate = x3d_path / "X3D.npy"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Directory {x3d_path} does not contain 'X3D.npy'"
            )
        poses = np.load(candidate)
    else:
        suffix = x3d_path.suffix.lower()
        if suffix == ".npy":
            poses = np.load(x3d_path)
        elif suffix == ".npz":
            with np.load(x3d_path) as z:
                # Common key name is 'X3D'
                if "X3D" not in z.files:
                    raise ValueError(
                        f"NPZ file {x3d_path} missing 'X3D' array (has {z.files})"
                    )
                poses = z["X3D"]
        else:
            raise ValueError(
                f"Unsupported MotionBERT file type: {x3d_path.name} "
                "(expected .npy, .npz, or a directory with X3D.npy)"
            )

    if poses.ndim != 3 or poses.shape[1:] != (17, 3):
        raise ValueError(
            f"Expected MotionBERT shape (F,17,3), got {poses.shape}"
        )
    if not np.isfinite(poses).all():
        raise ValueError("MotionBERT array contains NaNs or infs")

    return poses.astype(np.float32, copy=False)


# ---------------------------------------------------------------------
# ❷  Core maths – similarity-transform estimation
# ---------------------------------------------------------------------
def _umeyama_from_point_sets(
    P: np.ndarray,   # (k,3) source (3D)
    Q: np.ndarray,   # (k,3) target (3D; for 2D use z=0)
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Return (s, R, t) that minimises ‖ s·R·P + t − Q ‖² with det(R)=+1.
    Both P and Q must be (k,3) and correspond row-by-row.
    """
    if P.ndim != 2 or Q.ndim != 2 or P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError(f"P,Q must be same shape (k,3); got {P.shape} vs {Q.shape}")
    if P.shape[0] < 3:
        raise RuntimeError("Need at least 3 correspondences")

    if not (np.isfinite(P).all() and np.isfinite(Q).all()):
        raise ValueError("Non-finite values in correspondences")

    μP, μQ = P.mean(axis=0), Q.mean(axis=0)
    P0, Q0 = P - μP, Q - μQ

    Σ = P0.T @ Q0 / len(P0)  # (3,3)
    if not np.isfinite(Σ).all():
        raise ValueError("Covariance Σ contains non-finite values")
    # Debug (optional):
    # print(f"[DEBUG] Umeyama: Σ cond={np.linalg.cond(Σ):.3e}")

    U, S, Vt = np.linalg.svd(Σ, full_matrices=True)
    if S[0] <= 0 or (S[-1] / S[0]) < 1e-6:
        print(f"[WARN] Umeyama: singular values ratio very small "
            f"(Smin/Smax={S[-1]/max(S[0], 1e-12):.3e}); solution may be unstable.")
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    var_P = (P0 ** 2).sum() / len(P0)
    if var_P <= 1e-12 or not np.isfinite(var_P):
        raise RuntimeError(f"Degenerate point set; var_P={var_P}")
    s = S.sum() / var_P
    if not np.isfinite(s) or s <= 0:
        raise RuntimeError(f"Invalid scale s={s}")

    t = μQ - s * R @ μP
    if not np.isfinite(t).all():
        raise RuntimeError("Translation t contains non-finite values")

    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5):
        raise RuntimeError("Rotation matrix R not orthonormal within tolerance")

    return float(s), R.astype(np.float32), t.astype(np.float32)




def _umeyama_alignment(
    kps2: np.ndarray,                 # (26,3) AlphaPose (x,y,conf)
    kps3: np.ndarray,                 # (17,3) MotionBERT (x,y,z)
    *,
    idx2d: Sequence[int],
    idx3d: Sequence[int],
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Convenience wrapper: build matched clouds from full-frame arrays and
    call the point-cloud solver.
    """
    if len(idx2d) != len(idx3d):
        raise ValueError("idx2d and idx3d must have the same length")

    P = kps3[idx3d, :3].astype(np.float64)            # (m,3)
    Q2 = kps2[idx2d, :2].astype(np.float64)           # (m,2)
    Q = np.column_stack([Q2, np.zeros(len(Q2))])      # (m,3) z=0

    # Drop any non-finite rows (and rows with NaNs in either set)
    mask = np.isfinite(P).all(1) & np.isfinite(Q).all(1)
    P, Q = P[mask], Q[mask]
    if P.shape[0] < 3:
        raise RuntimeError("Not enough valid correspondences for alignment")

    return _umeyama_from_point_sets(P, Q)

# --------------------------------------------------------------------------
# ❷  Core maths – frame-wise similarity estimation
# --------------------------------------------------------------------------
def estimate_frame_transform(
    kps2: np.ndarray,                  # AlphaPose one frame  (26,3)  x,y,conf
    kps3: np.ndarray,                  # MotionBERT one frame (17,3)  x,y,z
    idx2d: Sequence[int] = IDX_2D,     # joints in 2-D cloud   (default global list)
    idx3d: Sequence[int] = IDX_3D,     # matching joints in 3-D (same length)
    conf_th: float = 0.05,             # drop very low-conf. 2-D points
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate a *single-frame* similarity transform (s, R, t) that best aligns
    the raw 3-D MotionBERT skeleton onto the AlphaPose detections.

    The solver:
        • selects only the joint pairs listed in idx2d / idx3d
        • ignores 2-D points whose confidence < conf_th
        • lifts every 2-D (x,y) to (x,y,0) and runs Umeyama closed-form

    Returns
    -------
    s : float            – uniform scale
    R : ndarray (3,3)    – proper rotation, det(R)=+1
    t : ndarray (3,)     – translation vector   (all float32)
    """
    if len(idx2d) != len(idx3d):
        raise ValueError("idx2d and idx3d must have identical length")

    # 1) gather matching point clouds ------------------------------------
    P3 = kps3[idx3d, :3].astype(np.float64)
    P2 = kps2[idx2d, :3].astype(np.float64)

    if P3.shape[0] != P2.shape[0]:
        raise ValueError("Mismatch in joint counts between 3D and 2D selections")

    vis = (
        np.isfinite(P3).all(1) &
        np.isfinite(P2[:, :2]).all(1) &
        (P2[:, 2] >= conf_th)
    )
    print(f"[DEBUG] estimate_frame_transform: vis={vis.sum()}/{len(vis)} joints")

    if vis.sum() < 3:
        raise RuntimeError("Not enough valid correspondences in this frame")

    P3_valid = P3[vis]                             # (k,3)
    Q2d      = P2[vis, :2]                         # (k,2)
    Q3_valid = np.column_stack([Q2d, np.zeros(len(Q2d))])  # (k,3) z=0

    s, R, t = _umeyama_from_point_sets(P3_valid, Q3_valid)
    print(f"[DEBUG] estimate_frame_transform: s={s:.6f}, det(R)={np.linalg.det(R):.6f}")

    return float(s), R.astype(np.float32), t.astype(np.float32)
# --------------------------------------------------------------------------
# ❸  Sequence-level alignment helper
# --------------------------------------------------------------------------
def _frame_score(k2_slice: np.ndarray, idx2d: Sequence[int], conf_th: float) -> tuple[int, float]:
    """
    Score a frame by (visible_joint_count, sum_confidence) over the selected joints.
    Higher is better. Requires k2_slice shaped (26,3).
    """
    k2 = k2_slice[idx2d]  # (m,3)
    vis = np.isfinite(k2[:, :2]).all(1) & (k2[:, 2] >= conf_th)
    return int(vis.sum()), float(k2[vis, 2].sum())


def align_motionbert_sequence(
    kps2_all: np.ndarray,             # (F,26,3)  AlphaPose raw  x,y,conf
    kps3_all: np.ndarray,             # (F,17,3)  MotionBERT raw x,y,z
    *,
    robust: bool = True,
    idx2d: Sequence[int] = IDX_2D,
    idx3d: Sequence[int] = IDX_3D,
    conf_th: float = 0.05,
    rmse_thresh: float | None = None,     # if None → only warn, don't enforce
    recalc_every: int = 5,                # NEW: re-fit every N frames
    lookahead: int = 3,                   # NEW: choose best of next K frames
    fix_rotation: bool = True,            # NEW: keep R from the first fit
    strict_rmse: bool = False,            # NEW: raise if ref-RMSE > rmse_thresh
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a similarity transform and apply it to every MotionBERT frame.
    Enhancements:
      • Use more joints (via idx2d/idx3d) to better condition the fit.
      • Re-estimate (s,t) (and optionally R) every `recalc_every` frames.
      • Within each re-estimation step, choose the best frame among the
        next `lookahead` frames by visibility/confidence.
      • Optionally keep R fixed after the first stable fit.

    Returns
    -------
    skel_3d_aligned : (F,17,3)
    M               : (4,4) similarity matrix built from the *last* (s,R,t)
    """

    # ── basic checks ────────────────────────────────────────────────────
    if kps2_all.ndim != 3 or kps2_all.shape[1:] != (26, 3):
        raise ValueError(f"kps2_all must be (F,26,3); got {kps2_all.shape}")
    if kps3_all.ndim != 3 or kps3_all.shape[1:] != (17, 3):
        raise ValueError(f"kps3_all must be (F,17,3); got {kps3_all.shape}")
    if kps2_all.shape[0] != kps3_all.shape[0]:
        raise ValueError("kps2_all and kps3_all must have identical #frames")
    if not (np.isfinite(kps2_all).all() and np.isfinite(kps3_all).all()):
        raise ValueError("Inputs contain NaNs/Infs")

    F = kps2_all.shape[0]

    # ── helper: RMSE on a single frame for diagnostics ──────────────────
    def _rmse_frame(k2_frame: np.ndarray, k3_frame: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> float:
        k2 = k2_frame[idx2d]                                # (m,3)
        k3p = ((s * (R @ k3_frame[idx3d].T)).T + t)[:, :2]  # (m,2)
        vis = np.isfinite(k2[:, :2]).all(1) & np.isfinite(k3p).all(1) & (k2[:, 2] >= conf_th)
        if vis.sum() < 2:
            return np.nan
        diffs = k3p[vis] - k2[vis, :2]
        return float(np.sqrt(np.mean(diffs * diffs)))

    # ── helper: (s,t) least-squares with fixed R ────────────────────────
    def _fit_scale_translation_fixed_R(
        k2_frame: np.ndarray,     # (26,3)
        k3_frame: np.ndarray,     # (17,3)
        R_fixed: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        # Build correspondences
        P3 = k3_frame[idx3d, :3].astype(np.float64)          # (k,3)
        Q2 = k2_frame[idx2d, :2].astype(np.float64)          # (k,2)
        Q3 = np.column_stack([Q2, np.zeros(len(Q2))])         # (k,3)

        # Visibility mask
        conf = k2_frame[idx2d, 2]
        vis = np.isfinite(P3).all(1) & np.isfinite(Q3).all(1) & (conf >= conf_th)
        if vis.sum() < 3:
            raise RuntimeError("Not enough correspondences for fixed-R fit")

        P3 = P3[vis]
        Q3 = Q3[vis]

        # Rotate P, then solve min_{s,t} || s*P' + t - Q ||
        Pp = (R_fixed @ P3.T).T                              # (k,3)
        muP = Pp.mean(axis=0); muQ = Q3.mean(axis=0)
        P0 = Pp - muP; Q0 = Q3 - muQ

        # Because Q3 has zero Z, including the Z dimension of P0 in the
        # least-squares scale estimate biases the result.  Use only XY
        # components so scale is determined from comparable dimensions.
        P0_xy = P0[:, :2]
        Q0_xy = Q0[:, :2]
        denom = float((P0_xy * P0_xy).sum())
        if denom <= 1e-12:
            raise RuntimeError("Degenerate fixed-R fit (denom≈0)")
        s = float((P0_xy * Q0_xy).sum() / denom)
        t = (muQ - s * muP).astype(np.float32)
        return s, t

    # ── 1) Seed on a good reference frame (same as before, with soft RMSE) ─
    if robust:
        n_cand = max(1, int(0.3 * F))
        cand_frames = range(n_cand)
    else:
        cand_frames = range(F)

    best = {"rmse": np.inf, "frame": None, "params": None}
    for f in cand_frames:
        try:
            s0, R0, t0 = estimate_frame_transform(
                kps2_all[f], kps3_all[f], idx2d=idx2d, idx3d=idx3d, conf_th=conf_th
            )
        except RuntimeError:
            continue
        err = _rmse_frame(kps2_all[f], kps3_all[f], s0, R0, t0)
        if np.isnan(err):
            continue
        if err < best["rmse"]:
            best.update(rmse=err, frame=f, params=(s0, R0, t0))
            if not robust:
                break

    if best["frame"] is None:
        raise RuntimeError("No frame with ≥3 reliable correspondences found")

    if rmse_thresh is not None and best["rmse"] > rmse_thresh:
        msg = (f"Reference frame RMSE {best['rmse']:.5f} exceeds threshold "
               f"({rmse_thresh}). Alignment may be unreliable.")
        if strict_rmse:
            raise ValueError(msg)
        else:
            print(f"[WARN] {msg}")

    ref_f = int(best["frame"])
    s, R, t = best["params"]
    print(f"[INFO] reference frame {ref_f}  |  RMSE = {best['rmse']:.5f}")

    # ── 2) Roll forward and periodically refresh ─────────────────────────
    out = np.empty_like(kps3_all)

    def _apply_block(f0: int, f1: int, s_: float, R_: np.ndarray, t_: np.ndarray) -> None:
        flat = kps3_all[f0:f1].reshape(-1, 3).T      # (3, (f1-f0)*17)
        aligned_flat = (s_ * (R_ @ flat)).T + t_     # ((f1-f0)*17, 3)
        out[f0:f1] = aligned_flat.reshape((f1 - f0, ) + kps3_all.shape[1:]).astype(np.float32)

    start = 0
    have_fixed_R = False

    while start < F:
        # pick best frame in [start, start+lookahead)
        end_win = min(F, start + max(1, lookahead))
        best_loc = (-1, -1.0, None)  # (vis_count, conf_sum, frame_idx)
        for f in range(start, end_win):
            vis_count, conf_sum = _frame_score(kps2_all[f], idx2d, conf_th)
            if vis_count >= 3 and (vis_count > best_loc[0] or
                                   (vis_count == best_loc[0] and conf_sum > best_loc[1])):
                best_loc = (vis_count, conf_sum, f)
        f_ref = best_loc[2] if best_loc[2] is not None else start

        # (re-)estimate on the chosen frame
        try:
            s_new, R_new, t_new = estimate_frame_transform(
                kps2_all[f_ref], kps3_all[f_ref],
                idx2d=idx2d, idx3d=idx3d, conf_th=conf_th
            )
        except RuntimeError:
            # fallback to previous params if estimation fails
            s_new, R_new, t_new = s, R, t

        if fix_rotation:
            if not have_fixed_R:
                # lock in rotation from the first successful step
                R = R_new.astype(np.float32, copy=False)
                have_fixed_R = True
            # re-fit (s,t) with fixed R to absorb mild depth/scale drift
            try:
                s, t = _fit_scale_translation_fixed_R(kps2_all[f_ref], kps3_all[f_ref], R)
            except RuntimeError:
                # if fixed-R fit fails, keep previous (s,t)
                pass
        else:
            s, R, t = s_new, R_new, t_new

        # quick diagnostic RMSE on the local reference
        err_local = _rmse_frame(kps2_all[f_ref], kps3_all[f_ref], s, R, t)
        if not np.isnan(err_local):
            if rmse_thresh is not None and err_local > rmse_thresh:
                print(f"[WARN] local ref frame {f_ref} RMSE={err_local:.5f} > {rmse_thresh:.5f}")
            else:
                print(f"[DEBUG] local ref {f_ref}: RMSE={err_local:.5f}")

        # apply to the next block
        next_cut = min(F, start + max(1, recalc_every))
        _apply_block(start, next_cut, s, R, t)
        start = next_cut

    # ── 3) Pack final 4×4 similarity matrix (last params) ───────────────
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = s * R
    M[:3,  3] = t

    return out, M


# --------------------------------------------------------------------------
# ❹  Utility – apply a similarity transform and drop Z
# --------------------------------------------------------------------------
def project_3d_to_2d(
    kps3: np.ndarray,                 # (K,3) raw or already-aligned 3-D points
    s: float,
    R: np.ndarray,                    # (3,3) rotation
    t: np.ndarray,                    # (3,)  translation
) -> np.ndarray:                      # → (K,2)   projected X,Y
    """
    Apply the similarity transform *(s,R,t)* to an arbitrary 3-D point cloud and
    return its 2-D projection (XY components only).

    Formula
    -------
        P2D  =  ( s · R · P3Dᵀ  + t )ᵀ   → discard Z-column

    Parameters
    ----------
    kps3 : ndarray (K,3)
        3-D point set to project (Pelvis-centred or arbitrary).
    s    : float
        Uniform scale factor obtained from the alignment step.
    R    : ndarray (3,3)
        Proper rotation matrix (det = +1).
    t    : ndarray (3,)
        Translation vector in the **image** coordinate system.

    Returns
    -------
    ndarray (K,2)
        Projected 2-D coordinates in the AlphaPose frame.

    Notes
    -----
    • Works on either a single frame (17,3) or any (K,3) array.  
    • The caller is responsible for giving the same *(s,R,t)* triplet that
      was estimated from the matching AlphaPose/MotionBERT frame.
    """
    if R.shape != (3,3) or t.shape != (3,):
        raise ValueError(f"Bad shapes for R {R.shape} or t {t.shape}")
    if not np.isfinite(s) or not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError("Non-finite s/R/t")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5):
        raise ValueError("R must be orthonormal")

    if kps3.ndim != 2 or kps3.shape[1] != 3:
        raise ValueError("kps3 must have shape (K,3)")

    # (3,K) ← (3,3)@(3,K)
    transformed = (s * (R @ kps3.T)).T + t          # (K,3)
    return transformed[:, :2].astype(np.float32)


# --------------------------------------------------------------------------
# ❺  Utility – per-frame reprojection RMSE
# --------------------------------------------------------------------------
def per_frame_reprojection_error(
    kps2: np.ndarray,                 # AlphaPose one-frame slice (K,3)  x,y,conf
    kps3_proj: np.ndarray,            # projected MotionBERT XY (K,2) – **same K/order**
    vis_mask: np.ndarray | None = None,
) -> float:
    """
    Root-mean-square reprojection error between *observed* AlphaPose points
    and *projected* MotionBERT points **for one frame**.

    Parameters
    ----------
    kps2 : ndarray (K,3)
        AlphaPose keypoints for the chosen K joints.  We use only columns 0–1;
        column 2 is confidence.
    kps3_proj : ndarray (K,2)
        2-D coordinates obtained from project_3d_to_2d(…) for the same joints.
    vis_mask : ndarray (K,) | None
        Boolean array marking which rows to include in the error metric.
        If *None*, points with conf < 0.05 or non-finite coords are ignored.

    Returns
    -------
    float
        RMSE in the **same unit** as the AlphaPose data (usually unit-square
        after normalisation, or pixels in raw space).

    Raises
    ------
    ValueError
        * if input shapes are inconsistent
        * if fewer than 2 valid points remain after masking
    """
    # ---------------- shape / dtype checks --------------------------------
    if kps2.ndim != 2 or kps2.shape[1] < 2:
        raise ValueError("kps2 must have shape (K,≥2)")
    if kps3_proj.shape != (kps2.shape[0], 2):
        raise ValueError(
            f"Shape mismatch: kps2 ({kps2.shape}) vs kps3_proj ({kps3_proj.shape})"
        )

    kps2_xy = kps2[:, :2].astype(np.float64)
    kps3_xy = kps3_proj.astype(np.float64)

    # ---------------- visibility mask -------------------------------------
    if vis_mask is None:
        if kps2.shape[1] >= 3:
            conf = kps2[:, 2]
        else:
            conf = np.ones(kps2.shape[0])
        vis_mask = (
            np.isfinite(kps2_xy).all(1)
            & np.isfinite(kps3_xy).all(1)
            & (conf >= 0.05)
        )
    else:
        vis_mask = np.asarray(vis_mask, dtype=bool)
        if vis_mask.shape != (kps2.shape[0],):
            raise ValueError("vis_mask must have shape (K,)")

    if vis_mask.sum() < 2:
        raise ValueError("Not enough valid correspondences for RMSE evaluation")

    # ---------------- RMSE ------------------------------------------------
    diffs = kps2_xy[vis_mask] - kps3_xy[vis_mask]
    rmse = float(np.sqrt((diffs ** 2).mean()))
    return rmse

# --------------------------------------------------------------------------
# ❻  Quality-control – RMSE for *every* frame in the clip
# --------------------------------------------------------------------------
def evaluate_alignment_over_sequence(
    kps2_all: np.ndarray,                # (F,26,3)  AlphaPose  (raw, NOT unit-sq.)
    kps3_aligned: np.ndarray,            # (F,17,3)  MotionBERT already aligned
    *,
    idx2d: Sequence[int] = IDX_2D,
    idx3d: Sequence[int] = IDX_3D,
    conf_th: float = 0.05,
) -> np.ndarray:
    """
    Compute the per-frame *re-projection RMSE* between **aligned** MotionBERT
    XY-coordinates and AlphaPose detections across the WHOLE sequence.

    Returns
    -------
    ndarray (F,)  – RMSE for every frame; `np.nan` if < 2 valid joints.

    Raises
    ------
    ValueError
        If `kps2_all` / `kps3_aligned` shapes disagree or joint-index lengths differ.
    """
    # -------- sanity checks ---------------------------------------------
    if kps2_all.shape[0] != kps3_aligned.shape[0]:
        raise ValueError(
            "kps2_all and kps3_aligned must have the same number of frames"
        )
    if len(idx2d) != len(idx3d):
        raise ValueError("idx2d and idx3d must be the same length")

    F = kps2_all.shape[0]
    rmse_per_frame = np.full(F, np.nan, dtype=np.float32)

    # --------------------------------------------------------------------
    if max(idx2d, default=-1) >= 26 or min(idx2d, default=0) < 0:
        raise ValueError("idx2d contains out-of-range indices for HALPE-26")
    if max(idx3d, default=-1) >= 17 or min(idx3d, default=0) < 0:
        raise ValueError("idx3d contains out-of-range indices for H36M-17")



    for f in range(F):
        k2 = kps2_all[f, idx2d]            # (K,3)
        k3_xy = kps3_aligned[f, idx3d, :2] # (K,2)

        # mask: finite & confidence
        vis = (
            np.isfinite(k2[:, :2]).all(1)
            & np.isfinite(k3_xy).all(1)
            & (k2[:, 2] >= conf_th)
        )

        if vis.sum() < 2:
            # Not enough data → leave np.nan so caller can decide
            continue

        diffs = k2[vis, :2].astype(np.float64) - k3_xy[vis].astype(np.float64)
        rmse_per_frame[f] = np.sqrt((diffs**2).mean()).astype(np.float32)

    # Debug safety: warn if *all* frames are NaN
    if np.isnan(rmse_per_frame).all():
        raise RuntimeError(
            "evaluate_alignment_over_sequence(): No frame had ≥2 valid joints "
            "— cannot assess alignment quality."
        )

    return rmse_per_frame



# --------------------------------------------------------------------------
# ❼  I/O – persist the aligned skeleton (+ metadata) in a single call
# --------------------------------------------------------------------------
import argparse   # (placed here to avoid a second top-level import block)


def save_aligned_motionbert(
    aligned_3d: np.ndarray,
    out_path: str | Path,
) -> None:
    """
    Write `aligned_3d` to disk as a NumPy ``.npy`` file.
    Extra safety guards ensure shape / dtype are sensible and the directory
    exists before saving.

    Parameters
    ----------
    aligned_3d : ndarray (F,17,3) – *float32* preferred
    out_path   : str | pathlib.Path
        Destination file name.  Parents are created if necessary.
    """
    if aligned_3d.ndim != 3 or aligned_3d.shape[2] != 3:
        raise ValueError("aligned_3d must have shape (F,17,3) or (F,N,3)")
    if not np.isfinite(aligned_3d).all():
        raise ValueError("aligned_3d contains NaNs or infs – aborting save()")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, aligned_3d.astype(np.float32, copy=False))
    print(f"💾  Saved aligned MotionBERT   →  {out_path}  (shape={aligned_3d.shape})")


# --------------------------------------------------------------------------
# ❽  CLI helpers
# --------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit a similarity transform that aligns raw MotionBERT 3-D "
                    "skeletons to AlphaPose 2-D detections, save the upright "
                    "3-D sequence and a small JSON with the transform."
    )
    p.add_argument("--alphapose", required=True,
                   help=".npy with raw AlphaPose detections  (F,26,3)")
    p.add_argument("--motionbert", required=True,
                   help=".npy with raw MotionBERT X3D coordinates (F,17,3)")
    p.add_argument("--out-skel", default="motionbert_aligned.npy",
                   help="Output .npy for the upright 3-D skeleton sequence")
    p.add_argument("--out-meta", default="motionbert_alignment_meta.json",
                   help="Output JSON with similarity transform & diagnostics")
    p.add_argument("--no-robust", action="store_true",
                   help="Disable robust frame search – use first valid frame")
    p.add_argument("--rmse-thresh", type=float, default=5e-3,
                   help="Max RMS reprojection error allowed on the reference "
                        "frame (default 0.005 unit-square ≈ few pixels).")
    return p


def parse_args() -> argparse.Namespace:        # Public so callers can re-use.
    return _build_arg_parser().parse_args()


# --------------------------------------------------------------------------
# ❾  CLI entry-point
# --------------------------------------------------------------------------
def main() -> None:
    """
    End-to-end utility:

      1. Load *.npy* inputs.
      2. Fit similarity transform (+ robust search unless --no-robust).
      3. Save the aligned 3-D skeleton and a meta-JSON (scale, R, t, RMSEs).
      4. Print a short summary.
    """
    args = parse_args()

    # ---------- load -----------------------------------------------------
    kps2_all = np.load(args.alphapose)
    kps3_all = np.load(args.motionbert)

    # ---------- align ----------------------------------------------------
    aligned_3d, M = align_motionbert_sequence(
        kps2_all, kps3_all,
        robust=not args.no_robust,
        rmse_thresh=args.rmse_thresh,
    )

    # Compute per-frame RMSE for curiosity / logging
    rmse_seq = evaluate_alignment_over_sequence(kps2_all, aligned_3d)
    mean_rmse = float(np.nanmean(rmse_seq))
    max_rmse  = float(np.nanmax(rmse_seq))

    # ---------- save -----------------------------------------------------
    save_aligned_motionbert(aligned_3d, args.out_skel)

    meta = {
        "similarity_matrix": M.tolist(),      # 4×4  (float32)
        "mean_RMSE": mean_rmse,
        "max_RMSE":  max_rmse,
        "reference_frame": int(np.nanargmin(rmse_seq)),
        "rmse_per_frame": rmse_seq.tolist(),
    }
    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with out_meta.open("w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"📝  Metadata JSON written     →  {out_meta}")

    # ---------- final summary -------------------------------------------
    print(
        f"\n✅  Alignment finished.\n"
        f"    mean RMSE  = {mean_rmse:.6f}\n"
        f"    max  RMSE  = {max_rmse:.6f}\n"
        f"    output     = {args.out_skel}"
    )


# --------------------------------------------------------------------------
if __name__ == "__main__":     # pragma: no cover
    main()
