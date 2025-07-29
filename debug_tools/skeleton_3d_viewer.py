#!/usr/bin/env python3
"""
Interactive 3‑D skeleton viewer for MotionBERT outputs (H36M‑17 joints).

▸ Mouse   –  left‑drag = rotate,  middle‑drag = pan,  scroll = zoom
▸ Keys    –  space      pause / resume
             ← / →      prev / next frame
             + / ‑      faster / slower
             ↑ / ↓      tilt camera (small steps)
             , / .      pan camera
             q / ESC    quit

Usage
-----
$ python debug_tools/skeleton_3d_viewer.py \
        --jobdir  /path/to/job_X          \
        [--fps 30]
"""

from __future__ import annotations
import argparse, os, sys, itertools, warnings, math
import numpy as np
import matplotlib

# ── backend fallback ───────────────────────────────────────────────
try:                                   # try Qt first (best UX if present)
    matplotlib.use("Qt5Agg")
except Exception:                      # Wayland / headless? → Tk
    warnings.warn("⚠️  Qt backend unavailable – falling back to TkAgg")
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3‑D)

# ── H36M‑17 skeleton definition ────────────────────────────────────
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

# ── helpers ────────────────────────────────────────────────────────
def _load_motionbert_X3D(jobdir: str | os.PathLike) -> np.ndarray:
    """
    Accepts either
      •  .../motionbert/3d-pose-results.npz   (file)
      •  .../motionbert/3d-pose-results.npz/  (directory with X3D.npy)
    Returns float32 array (T, 17, 3)
    """
    mb_root = os.path.join(jobdir, "motionbert")
    path = os.path.join(mb_root, "3d-pose-results.npz")
    if os.path.isdir(path):                        # new layout
        x3d_path = os.path.join(path, "X3D.npy")
        if not os.path.exists(x3d_path):
            raise FileNotFoundError(f"missing {x3d_path}")
        poses = np.load(x3d_path)
    elif os.path.isfile(path):                     # legacy .npz file
        poses = np.load(path)["X3D"]
    else:
        raise FileNotFoundError(f"{path} (not a file nor dir)")
    if poses.shape[1:] != (17, 3):
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


# ── main viewer ────────────────────────────────────────────────────
def launch_viewer(jobdir: str | os.PathLike, fps: int = 30) -> None:
    poses = _load_motionbert_X3D(jobdir)            # (T,17,3)
    T = poses.shape[0]

    fig = plt.figure("Skeleton 3‑D viewer")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # draw once, then update data in‑place
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

    # ── interactive controls ───────────────────────────────────────
    speed   = [1.0]   # mutable containers → editable inside closures
    paused  = [False]
    frame   = [0]
    last_elev, last_azim = [ax.elev], [ax.azim]

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

    # → mouse drag track‑ball
    drag_state = {"press": None}  # (x,y, elev, azim)

    def on_press(ev):
        if ev.inaxes != ax:      # ignore clicks outside the 3‑D axes
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

    # ── animation callback ─────────────────────────────────────────
    def update(_):
        if not paused[0]:
            frame[0] = (frame[0] + max(int(speed[0]), 1)) % T
        i = frame[0]
        scat._offsets3d = (poses[i, :, 0],
                           poses[i, :, 1],
                           poses[i, :, 2])
        for ln, (p, c) in zip(lines, _EDGES):
            ln.set_data_3d([poses[i, p, 0], poses[i, c, 0]],
                           [poses[i, p, 1], poses[i, c, 1]],
                           [poses[i, p, 2], poses[i, c, 2]])
        return [scat, *lines]

    FuncAnimation(fig, update, interval=1000 / fps)
    plt.show()


# ── CLI ────────────────────────────────────────────────────────────
def _main() -> None:
    ap = argparse.ArgumentParser(description="3‑D skeleton viewer (H36M‑17)")
    ap.add_argument("--jobdir", required=True, help="job directory root")
    ap.add_argument("--fps", type=int, default=30,
                    help="playback fps (default 30)")
    launch_viewer(**vars(ap.parse_args()))


if __name__ == "__main__":
    _main()
