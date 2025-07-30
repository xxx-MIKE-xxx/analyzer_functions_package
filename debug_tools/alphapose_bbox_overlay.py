#!/usr/bin/env python3
"""
alphapose_bbox_overlay.py – overlay + diagnostics
──────────────────────────────────────────────────
Adds a yellow per‑frame square to the source video **and** produces:

• a stick‑figure unit‑square video  (optional)
• a *side‑length vs frame* plot with coloured dots:
      ▸  blue  – frames kept for stable‑side estimation
      ▸  red   – frames discarded as out‑of‑range / high slope
      ▸  green – frames in the final “best run” used for S*
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import List, Tuple

import cv2, numpy as np, matplotlib.pyplot as plt


# ────────────────── loading & basic geometry ──────────────────
def load_kps(json_path: str | Path) -> dict[int, np.ndarray]:
    with open(json_path) as f:
        raw = json.load(f)
    out = {}
    for e in raw:
        i = int(Path(e["image_id"]).stem)
        out[i] = np.asarray(e["keypoints"], np.float32).reshape(-1, 3)
    return out


def square_bbox(kps: np.ndarray) -> Tuple[float, float, float]:
    vis = kps[:, 2] > 0.05
    if not np.any(vis):
        return 0.0, 0.0, 1.0
    xs, ys = kps[vis, 0], kps[vis, 1]
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) or 1.0
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    return cx - side/2, cy - side/2, side


# ────────────────── overlay video  (+ collect sides) ───────────
def overlay_video(src: str | Path, kps: dict[int, np.ndarray],
                  dst: str | Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src}")
    fps, W, H = cap.get(cv2.CAP_PROP_FPS) or 30, int(cap.get(3)), int(cap.get(4))
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    sides = []
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        k = kps.get(f)
        if k is not None:
            x0,y0,s = square_bbox(k)
            sides.append(s)
            x1,y1 = int(max(0,x0)), int(max(0,y0))
            x2,y2 = int(min(W-1,x0+s)), int(min(H-1,y0+s))
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
        else:
            sides.append(np.nan)
        vw.write(frame); f += 1

    cap.release(); vw.release()
    print("🎞️  overlay →", dst)
    return np.asarray(sides, np.float32)


# ────────────────── stick‑figure video  (unit square) ───────────
def stickman_video(kps: dict[int, np.ndarray], dst: str | Path, L=512):
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), 30, (L,L))
    F = max(kps)+1
    for f in range(F):
        img = np.full((L,L,3),255,np.uint8)
        for g in np.linspace(0,1,11):
            y = int((1-g)*(L-1)); x = int(g*(L-1))
            cv2.line(img,(0,y),(L-1,y),(230,230,230),1)
            cv2.line(img,(x,0),(x,L-1),(230,230,230),1)
        cv2.rectangle(img,(0,0),(L-1,L-1),(150,150,150),2)

        if f in kps:
            xn,yn,_ = kps[f].T
            vis = kps[f][:,2]>0.05
            xs,ys = xn[vis],yn[vis]
            if xs.size:
                x0,x1,y0,y1 = xs.min(),xs.max(),ys.min(),ys.max()
                s = max(x1-x0,y1-y0) or 1
                cx,cy = (x0+x1)/2,(y0+y1)/2
                xs = (xs-(cx-s/2))/s; ys = (ys-(cy-s/2))/s
                ys = 1-ys
                for xx,yy in zip(xs,ys):
                    cv2.circle(img,(int(xx*(L-1)),int(yy*(L-1))),5,(0,0,255),-1)
        vw.write(img)
    vw.release()
    print("🦴  stickman →", dst)


# ────────────────── diagnostic plot  (with masks) ───────────────
def plot_series(sides: np.ndarray, png: str | Path):
    smooth = cv2.blur(sides.reshape(-1,1), (9,1)).ravel()
    p5,p95 = np.nanpercentile(smooth, [5,95])
    keep = (smooth>=p5)&(smooth<=p95)
    S_star = np.nanmedian(smooth[keep])

    plt.figure(figsize=(10,4))
    plt.plot(sides, lw=1.0, label="raw")
    plt.scatter(np.where(~keep), sides[~keep], s=8, c="red", label="discarded")
    plt.scatter(np.where(keep),  sides[keep],  s=8, c="blue", label="kept")
    plt.axhline(S_star, color="green", lw=1.5, label=f"S*={S_star:.0f}px")
    plt.title("Per‑frame square side")
    plt.xlabel("frame"); plt.ylabel("pixels"); plt.grid(ls="--",alpha=.4)
    plt.legend(); plt.tight_layout(); plt.savefig(png,dpi=150); plt.close()
    print("📊  plot →", png)


# ───────────────────────────── CLI ───────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alphapose", required=True)
    p.add_argument("--video",     required=True)
    p.add_argument("--out",       required=True)
    p.add_argument("--stickman")
    p.add_argument("--plot")
    a = p.parse_args()

    kps = load_kps(a.alphapose)
    sides = overlay_video(a.video, kps, a.out)

    if a.stickman:
        stickman_video(kps, a.stickman)
    if a.plot:
        plot_series(sides, a.plot)


if __name__ == "__main__":
    main()
