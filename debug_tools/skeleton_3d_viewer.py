#!/usr/bin/env python3
"""
Interactive 3-D skeleton viewer for MotionBERT outputs (H36M-17 joints).

▸ Mouse   –  left-drag = rotate,  middle-drag = pan,  scroll = zoom
▸ Keys    –  space      pause / resume
             ← / →      prev / next frame
             + / -      faster / slower
             ↑ / ↓      tilt camera (small steps)
             , / .      pan camera
             q / ESC    quit

Auto-orbits around the skeleton when playing.

Usage
-----
$ python debug_tools/skeleton_3d_viewer.py \
        --jobdir  /path/to/job_X          \
        [--fps 30]
"""

from __future__ import annotations
import argparse
import os
import warnings
import numpy as np
import matplotlib

# ── backend fallback ───────────────────────────────────────────────
try:
    matplotlib.use("Qt5Agg")
except Exception:
    warnings.warn("⚠️  Qt backend unavailable – falling back to TkAgg")
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3-D)

# ── H36M-17 skeleton definition ────────────────────────────────────
_JOINT_NAMES = [
    "root", "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle",
    "torso", "neck", "nose", "head",
    "LShoulder", "LElbow", "LWrist",
    "RShoulder", "RElbow", "RWrist",
]
_EDGES = [  # (parent, child) pairs
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

_PERM_YUP_TO_ZUP = np.array([[1, 0, 0],
                             [0, 0, 1],
                             [0, 1, 0]], dtype=np.float32)

_PERM_XZY = np.array([[1, 0, 0],
                      [0, 0, 1],
                      [0, 1, 0]], dtype=np.float32)  # (x,y,z)->(x,z,y)

def _yup_to_zup(P: np.ndarray) -> np.ndarray:
    """(x right, y up, z forward)  →  (x right, y forward, z up)."""
    shp = P.shape
    return (_PERM_XZY @ P.reshape(-1, 3).T).T.reshape(shp)

def _level_by_feet_Z(poses: np.ndarray) -> np.ndarray:
    """
    Rotate around Z so the feet line (LAnkle-RAnkle) is horizontal in XY.
    poses: (T,17,3) in z-up.
    """
    LAN, RAN = 6, 3  # H36M-17 indices: LAnkle=6, RAnkle=3
    # robust XY vector from R->L ankles over time
    foot_xy = np.nanmedian(poses[:, LAN, :2] - poses[:, RAN, :2], axis=0)
    if not np.all(np.isfinite(foot_xy)) or np.linalg.norm(foot_xy) < 1e-6:
        # (rare) fall back to hips if feet are together or missing
        LHIP, RHIP = 4, 1
        foot_xy = np.nanmedian(poses[:, LHIP, :2] - poses[:, RHIP, :2], axis=0)

    theta = np.arctan2(foot_xy[1], foot_xy[0])   # angle vs +X
    c, s = np.cos(-theta), np.sin(-theta)        # rotate by -theta to make it flat
    Rz = np.array([[c, -s, 0],
                   [s,  c, 0],
                   [0,  0, 1]], dtype=np.float32)
    return (Rz @ poses.reshape(-1, 3).T).T.reshape(poses.shape)



def _load_motionbert_X3D(jobdir: str | os.PathLike) -> np.ndarray:
    """
    Locate and load the aligned MotionBERT skeleton.  Tries in order:
      * {jobdir}/analysis/motionbert_aligned.npy
      * {jobdir}/analysis/motionbert/motionbert_aligned.npy
      * {jobdir}/motionbert/motionbert_aligned.npy
      * fallback to {jobdir}/analysis/*.npz or .npy containing aligned output
    Returns float32 array (T, 17, 3)
    """
    jobdir = os.fspath(jobdir)
    candidates = [
        os.path.join(jobdir, "analysis", "motionbert_aligned.npy"),
        os.path.join(jobdir, "analysis", "motionbert", "motionbert_aligned.npy"),
        os.path.join(jobdir, "motionbert", "motionbert_aligned.npy"),
        os.path.join(jobdir, "analysis", "motionbert_aligned.npz"),
    ]
    poses = None

    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            if p.lower().endswith(".npz"):
                with np.load(p) as z:
                    # try common keys
                    for key in ("motionbert_aligned", "aligned_3d", "X3D"):
                        if key in z.files:
                            poses = z[key]
                            break
                    if poses is None:
                        # fallback to first array
                        if z.files:
                            poses = z[z.files[0]]
            else:
                poses = np.load(p)
        except Exception:
            continue
        if poses is not None:
            break

    if poses is None:
        raise FileNotFoundError(
            f"Could not find aligned MotionBERT skeleton in expected locations under {jobdir}."
        )

    poses = np.asarray(poses)
    if poses.ndim != 3 or poses.shape[1:] != (17, 3):
        raise ValueError(f"expected (T,17,3) but got {poses.shape}")
    return poses.astype(np.float32)


def _set_equal_aspect(ax: Axes3D, pts: np.ndarray) -> None:
    """Makes X,Y,Z scales equal so the skeleton isn’t distorted."""
    mins, maxs = pts.min(axis=(0, 1)), pts.max(axis=(0, 1))
    ranges = maxs - mins
    max_range = ranges.max()
    mid = (maxs + mins) / 2
    for setter, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
        setter(m - max_range / 2, m + max_range / 2)


def launch_viewer(jobdir: str | os.PathLike, fps: int = 30) -> None:
    poses_yup = _load_motionbert_X3D(jobdir)   # (T,17,3)
    poses = _yup_to_zup(poses_yup)
    poses = _level_by_feet_Z(poses)
    ROOT = 0
    pelvis0 = np.nanmedian(poses[:, ROOT, :], axis=0)
    poses -= pelvis0
    T = poses.shape[0]

    fig = plt.figure("Skeleton 3-D viewer")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (up)")


    # initial draw
    pts = poses[0]
    scat = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                      s=40, c="red", depthshade=True)
    lines = [
        ax.plot([pts[p, 0], pts[c, 0]],
                [pts[p, 1], pts[c, 1]],
                [pts[p, 2], pts[c, 2]],
                lw=2, c="black")[0]
        for p, c in _EDGES
    ]
    _set_equal_aspect(ax, poses)

    # interaction state
    speed = [1.0]
    paused = [False]
    frame = [0]

    # auto orbit params
    orbit_speed_deg_per_frame = 0.5  # degrees of azimuth per update

    def on_key(ev):
        if ev.key in (" ", "pause"):
            paused[0] = not paused[0]
        elif ev.key == "left":
            frame[0] = (frame[0] - 1) % T
        elif ev.key == "right":
            frame[0] = (frame[0] + 1) % T
        elif ev.key == "+":
            speed[0] = min(speed[0] * 1.5, 15)
        elif ev.key == "-":
            speed[0] = max(speed[0] / 1.5, 0.1)
        elif ev.key == "up":
            ax.view_init(elev=ax.elev + 5, azim=ax.azim)
        elif ev.key == "down":
            ax.view_init(elev=ax.elev - 5, azim=ax.azim)
        elif ev.key == ",":
            ax.view_init(elev=ax.elev, azim=ax.azim - 5)
        elif ev.key == ".":
            ax.view_init(elev=ax.elev, azim=ax.azim + 5)
        elif ev.key in ("q", "escape"):
            plt.close(ev.canvas.figure)

    fig.canvas.mpl_connect("key_press_event", on_key)

    # mouse drag track-ball
    drag_state = {"press": None}

    def on_press(ev):
        if ev.inaxes != ax:
            return
        drag_state["press"] = (ev.x, ev.y, ax.elev, ax.azim)

    def on_release(_ev):
        drag_state["press"] = None

    def on_move(ev):
        info = drag_state["press"]
        if info is None or ev.inaxes != ax:
            return
        x0, y0, elev0, azim0 = info
        dx, dy = ev.x - x0, ev.y - y0
        ax.view_init(elev=elev0 - dy * 0.3, azim=azim0 - dx * 0.3)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)

    # animation callback
    def update(_):
        if not paused[0]:
            frame[0] = (frame[0] + max(int(speed[0]), 1)) % T
            # auto orbit: increment azimuth slightly
            ax.view_init(elev=ax.elev, azim=ax.azim + orbit_speed_deg_per_frame)
        i = frame[0]
        # update skeleton
        scat._offsets3d = (poses[i, :, 0],
                           poses[i, :, 1],
                           poses[i, :, 2])
        for ln, (p, c) in zip(lines, _EDGES):
            ln.set_data_3d([poses[i, p, 0], poses[i, c, 0]],
                           [poses[i, p, 1], poses[i, c, 1]],
                           [poses[i, p, 2], poses[i, c, 2]])
        return [scat, *lines]

    # keep animation object alive by assigning to a variable
    ani = FuncAnimation(fig, update, interval=1000 / fps)
    plt.show()

    # prevent garbage collection (if running in some embed contexts)
    return ani


def _main() -> None:
    ap = argparse.ArgumentParser(description="3-D skeleton viewer (H36M-17)")
    ap.add_argument("--jobdir", required=True, help="job directory root")
    ap.add_argument("--fps", type=int, default=30,
                    help="playback fps (default 30)")
    args = ap.parse_args()
    launch_viewer(**vars(args))


if __name__ == "__main__":
    _main()
