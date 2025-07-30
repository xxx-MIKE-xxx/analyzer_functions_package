#!/usr/bin/env python3
"""
alphapose_bbox_overlay.py – stable‑bbox overlay + diagnostics
──────────────────────────────────────────────────────────────
Draws the **stable yellow square** detected from the video, not the
per‑frame adaptive one.  Also fixes stick‑figure orientation and colours
the diagnostic plot.

Usage
-----
python -m debug_tools.alphapose_bbox_overlay                               \
       --alphapose alphapose-results.json --video src.mp4                  \
       --out bbox_overlay.mp4                                              \
       --stickman stickman.mp4                                             \
       --plot side_length.png
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import List, Tuple

import cv2, numpy as np, matplotlib.pyplot as plt


# ───────────────────────── loading ──────────────────────────
def load_kps(json_path: str | Path) -> dict[int, np.ndarray]:
    with open(json_path) as f:
        raw = json.load(f)
    out = {}
    for e in raw:
        idx = int(Path(e["image_id"]).stem)
        out[idx] = np.asarray(e["keypoints"], np.float32).reshape(-1, 3)
    return out


# ───────────────────── square utilities ─────────────────────
def adaptive_bbox(k: np.ndarray) -> Tuple[float, float, float]:
    vis = k[:, 2] > 0.05
    if not np.any(vis):
        return 0.0, 0.0, 1.0
    xs, ys = k[vis, 0], k[vis, 1]
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) or 1.0
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    return cx - side / 2, cy - side / 2, side


def compute_stable_side(sides: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """Return keep‑mask, S* and smoothed series."""
    smooth = cv2.blur(sides.reshape(-1, 1), (9, 1)).ravel()
    p5, p95 = np.nanpercentile(smooth, [5, 95])
    keep = (smooth >= p5) & (smooth <= p95)
    S_star = np.nanmedian(smooth[keep]) or 1.0
    return keep, float(S_star), smooth


# ─────────────────────── overlay video ──────────────────────
def overlay_video(src: str | Path, kps: dict[int, np.ndarray],
                  dst: str | Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {src}")
    fps, W, H = (cap.get(cv2.CAP_PROP_FPS) or 30,
                 int(cap.get(3)), int(cap.get(4)))
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    adaptive_sides: List[float] = []
    centres: List[Tuple[float, float]] = []

    # pass 1 – gather adaptive bboxes
    F = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f in range(F):
        k = kps.get(f)
        if k is None:
            adaptive_sides.append(np.nan)
            centres.append((np.nan, np.nan))
        else:
            x0, y0, s = adaptive_bbox(k)
            adaptive_sides.append(s)
            centres.append((x0, y0))

    keep, S_star, _ = compute_stable_side(np.asarray(adaptive_sides))

    # rewind & draw with *stable* side
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for f in range(F):
        ok, frame = cap.read()
        if not ok:
            break
        x0, y0 = centres[f]
        if not np.isnan(x0):
            x1, y1 = int(max(0,           x0)),          int(max(0,           y0))
            x2, y2 = int(min(W - 1, x0 + S_star)), int(min(H - 1, y0 + S_star))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        vw.write(frame)

    cap.release(); vw.release()
    print(f"🎞️  overlay → {dst}")
    return np.asarray(adaptive_sides, np.float32)


# ───────────────────── stick‑figure video ────────────────────
def stickman_video(kps: dict[int, np.ndarray], dst: str | Path, L: int = 512):
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), 30, (L, L))
    F = max(kps) + 1
    for f in range(F):
        canvas = np.full((L, L, 3), 255, np.uint8)

        # grid
        for v in np.linspace(0, 1, 11):
            y = int((1 - v) * (L - 1)); x = int(v * (L - 1))
            cv2.line(canvas, (0, y), (L - 1, y), (230, 230, 230), 1)
            cv2.line(canvas, (x, 0), (x, L - 1), (230, 230, 230), 1)
        cv2.rectangle(canvas, (0, 0), (L - 1, L - 1), (150, 150, 150), 2)

        if f in kps:
            k = kps[f]
            x0, y0, s = adaptive_bbox(k)
            vis = k[:, 2] > 0.05
            xs = (k[vis, 0] - x0) / s
            ys = 1.0 - (k[vis, 1] - y0) / s  # flip once → correct orientation
            for xx, yy in zip(xs, ys):
                cx, cy = int(xx * (L - 1)), int(yy * (L - 1))
                cv2.circle(canvas, (cx, cy), 5, (0, 0, 255), -1)

        vw.write(canvas)
    vw.release()
    print(f"🦴  stickman → {dst}")


# ───────────────────── diagnostic plot ───────────────────────
def plot_series(sides: np.ndarray, png: str | Path):
    keep, S_star, smooth = compute_stable_side(sides)

    plt.figure(figsize=(10, 4))
    plt.plot(sides, lw=1.0, label="raw")
    plt.scatter(np.where(~keep), sides[~keep], s=8, c="red",   label="discarded")
    plt.scatter(np.where(keep),  sides[keep],  s=8, c="blue",  label="kept")
    plt.axhline(S_star, color="green", lw=1.5, label=f"S* = {S_star:.0f}px")
    plt.title("Per‑frame square side length")
    plt.xlabel("frame"); plt.ylabel("pixels")
    plt.grid(ls="--", alpha=.4); plt.legend(); plt.tight_layout()
    plt.savefig(png, dpi=150); plt.close()
    print(f"📊  plot → {png}")


# ───────────────────────────── CLI ────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphapose", required=True)
    ap.add_argument("--video",     required=True)
    ap.add_argument("--out",       required=True, help="overlay .mp4")
    ap.add_argument("--stickman",  help="unit‑square stickman .mp4")
    ap.add_argument("--plot",      help="side‑length diagnostic .png")
    args = ap.parse_args()

    kps   = load_kps(args.alphapose)
    sides = overlay_video(args.video, kps, args.out)

    if args.stickman:
        stickman_video(kps, args.stickman)
    if args.plot:
        plot_series(sides, args.plot)


if __name__ == "__main__":
    main()
