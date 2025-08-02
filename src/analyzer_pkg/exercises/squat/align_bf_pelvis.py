#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

# --- HALPE-26 key indices we’ll use (2D) ---
HALPE = dict(
    LShoulder=5, RShoulder=6, LElbow=7, RElbow=8, LWrist=9, RWrist=10,
    LHip=11, RHip=12, LKnee=13, RKnee=14, LAnkle=15, RAnkle=16,
    Hip=19  # HALPE has a central Hip; if missing we’ll avg L/R hips
)
# --- H36M-17 indices we’ll use (3D) ---
H36 = dict(
    Pelvis=0, RHip=1, RKnee=2, RAnkle=3, LHip=4, LKnee=5, LAnkle=6,
    Spine=7, Thorax=8, Head=10, LShoulder=11, LElbow=12, LWrist=13,
    RShoulder=14, RElbow=15, RWrist=16
)

# Joints used for scoring/alignment (pairs must correspond semantically)
IDX2D = np.array([
    HALPE["LShoulder"], HALPE["RShoulder"],
    HALPE["LHip"], HALPE["RHip"],
    HALPE["LKnee"], HALPE["RKnee"],
    HALPE["LAnkle"], HALPE["RAnkle"],
    HALPE["LElbow"], HALPE["RElbow"],
    HALPE["LWrist"], HALPE["RWrist"],
], dtype=int)

IDX3D = np.array([
    H36["LShoulder"], H36["RShoulder"],
    H36["LHip"],      H36["RHip"],
    H36["LKnee"],     H36["RKnee"],
    H36["LAnkle"],    H36["RAnkle"],
    H36["LElbow"],    H36["RElbow"],
    H36["LWrist"],    H36["RWrist"],
], dtype=int)

def rot_yaw_pitch_roll(yaw, pitch, roll):
    """Rz(yaw) * Rx(pitch) * Ry(roll) in radians."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cx, sx = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(roll), np.sin(roll)

    Rz = np.array([[ cy,-sy, 0],
                   [ sy, cy, 0],
                   [  0,  0, 1]])
    Rx = np.array([[1,  0,  0],
                   [0, cx,-sx],
                   [0, sx, cx]])
    Ry = np.array([[ cz, 0, sz],
                   [  0, 1,  0],
                   [-sz, 0, cz]])
    return Rz @ Rx @ Ry

def pick_ap_pelvis_xy(ap_frame, conf_th=0.05):
    """Return pelvis XY from AlphaPose; prefer HALPE central Hip, else avg L/R hips."""
    hip = HALPE["Hip"]
    if ap_frame.shape[1] >= 3 and ap_frame[hip,2] >= conf_th and np.isfinite(ap_frame[hip,:2]).all():
        return ap_frame[hip,:2]
    # fallback: average L/R hip if both good
    L, R = HALPE["LHip"], HALPE["RHip"]
    okL = (ap_frame[L,2] >= conf_th) and np.isfinite(ap_frame[L,:2]).all()
    okR = (ap_frame[R,2] >= conf_th) and np.isfinite(ap_frame[R,:2]).all()
    if okL and okR:
        return 0.5*(ap_frame[L,:2] + ap_frame[R,:2])
    # last resort: use whatever is finite (may shift a bit)
    for j in (hip, L, R):
        if np.isfinite(ap_frame[j,:2]).all():
            return ap_frame[j,:2]
    return np.array([0.0,0.0], dtype=np.float32)

def best_scale_2d(Axy, Bxy, w=None, eps=1e-12):
    """
    Closed-form s that minimises sum w || s*A - B ||^2  in 2D.
    Axy, Bxy: (K,2)
    """
    if w is None:
        w = np.ones(len(Axy), dtype=np.float64)
    w = w.astype(np.float64)
    num = np.sum(w * (Axy * Bxy).sum(axis=1))
    den = np.sum(w * (Axy * Axy).sum(axis=1))
    if den < eps:
        return np.nan
    return float(num / den)

def frame_align_bruteforce(ap2d, mb3d, yaw_range, pitch_range, roll_range=(0,),
                           conf_th=0.05):
    """
    Align one frame, pelvis-anchored, brute-force rotation; return (s,R,t) and rmse.
    t is chosen so pelvis XY matches AlphaPose pelvis XY (Z=0).
    """
    # Anchors
    pelvis3d = mb3d[H36["Pelvis"], :].astype(np.float64)       # (3,)
    pelvis2d = pick_ap_pelvis_xy(ap2d, conf_th).astype(np.float64)  # (2,)

    P = (mb3d - pelvis3d).astype(np.float64)          # centre 3D at pelvis
    Q = ap2d.copy().astype(np.float64)
    Q[:, :2] -= pelvis2d                              # centre 2D at pelvis

    # Select corresponding joints and build weights
    A3 = P[IDX3D]                                     # (K,3)
    B2 = Q[IDX2D, :2]                                 # (K,2)
    conf = ap2d[IDX2D, 2] if ap2d.shape[1] >= 3 else np.ones(len(IDX2D))
    vis = np.isfinite(A3).all(1) & np.isfinite(B2).all(1) & (conf >= conf_th)
    if vis.sum() < 3:
        return np.nan, np.eye(3), np.zeros(3), np.inf  # not enough info

    A3 = A3[vis]
    B2 = B2[vis]
    w = conf[vis].astype(np.float64)

    best = dict(rmse=np.inf, s=np.nan, R=np.eye(3))
    # search grid
    for yaw in yaw_range:
        for pitch in pitch_range:
            for roll in roll_range:
                R = rot_yaw_pitch_roll(yaw, pitch, roll)
                Axy = (R @ A3.T).T[:, :2]                    # rotate then drop Z
                s = best_scale_2d(Axy, B2, w=w)
                if not np.isfinite(s):
                    continue
                repro = s * Axy                              # (K,2)
                diffs = repro - B2
                rmse = float(np.sqrt(np.mean((w * (diffs**2).sum(1)) / (w.mean()+1e-12))))
                if rmse < best["rmse"]:
                    best.update(rmse=rmse, s=s, R=R)

    s, R = best["s"], best["R"]
    # translation to put pelvis back to image XY (Z stays 0)
    t = np.array([pelvis2d[0], pelvis2d[1], 0.0], dtype=np.float64)
    return s, R.astype(np.float32), t.astype(np.float32), best["rmse"]

def align_sequence(ap_all, mb_all, deg_step_coarse=5, deg_step_fine=1,
                   roll_max_deg=5, conf_th=0.05):
    """
    Align every frame independently (simple & robust). Returns aligned (F,17,3).
    """
    F = ap_all.shape[0]
    out = np.zeros_like(mb_all, dtype=np.float32)

    # Build angle grids (coarse → fine around 0)
    def grid(deg_max, step):
        r = np.deg2rad(np.arange(-deg_max, deg_max+1e-6, step, dtype=float))
        return r

    yaw_coarse   = grid(60, deg_step_coarse)
    pitch_coarse = grid(45, deg_step_coarse)
    roll_coarse  = grid(roll_max_deg, max(deg_step_coarse, roll_max_deg))

    yaw_fine   = grid(15, deg_step_fine)
    pitch_fine = grid(15, deg_step_fine)
    roll_fine  = grid(min(roll_max_deg, 5), max(1, deg_step_fine))

    for f in range(F):
        s, R, t, _ = frame_align_bruteforce(ap_all[f], mb_all[f],
                                            yaw_coarse, pitch_coarse, roll_coarse, conf_th)
        # optional local refinement around best (centered at 0 because we don’t track R; simple second pass)
        s, R, t, _ = frame_align_bruteforce(ap_all[f], mb_all[f],
                                            yaw_fine, pitch_fine, roll_fine, conf_th)

        pelvis3d = mb_all[f, H36["Pelvis"]]
        centered = mb_all[f] - pelvis3d
        aligned  = (s * (R @ centered.T)).T + t  # (17,3); XY in image coords
        out[f] = aligned.astype(np.float32)

    return out

def main():
    p = argparse.ArgumentParser(description="Pelvis-anchored brute-force alignment (MotionBERT → AlphaPose XY frame).")
    p.add_argument("--alphapose", required=True, help=".npy AlphaPose (F,26,3)")
    p.add_argument("--motionbert", required=True, help=".npy MotionBERT (F,17,3)")
    p.add_argument("--out", default="motionbert_aligned.npy", help="output .npy (F,17,3)")
    p.add_argument("--conf-th", type=float, default=0.05, help="AlphaPose confidence threshold")
    args = p.parse_args()

    ap = np.load(args.alphapose)   # (F,26,3)
    x3 = np.load(args.motionbert)  # (F,17,3)
    assert ap.ndim==3 and ap.shape[1]>=26 and ap.shape[2]>=2, "Bad AlphaPose shape"
    assert x3.shape[1:]==(17,3), "Bad MotionBERT shape"

    aligned = align_sequence(ap[:, :26, :3], x3.astype(np.float32), conf_th=args.conf_th)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, aligned)
    print(f"✅ Saved aligned MotionBERT → {args.out}  shape={aligned.shape}")

if __name__ == "__main__":
    main()
