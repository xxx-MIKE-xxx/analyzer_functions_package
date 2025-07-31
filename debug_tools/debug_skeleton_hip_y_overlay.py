
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

RIGHT_HIP = 11
LEFT_HIP = 12

def plot_skeleton_with_hip_y(ax, kpts, frame_idx=None):
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(f"Skeleton hip heights (frame {frame_idx})")
    xs, ys = kpts[:, 0], kpts[:, 1]
    ax.scatter(xs, ys, c='blue', s=40, zorder=2)
    # Draw hips and show y values as overlay
    ax.scatter([xs[RIGHT_HIP], xs[LEFT_HIP]], [ys[RIGHT_HIP], ys[LEFT_HIP]], c='red', s=70, zorder=3)
    ax.text(xs[RIGHT_HIP]+0.03, ys[RIGHT_HIP], f"{ys[RIGHT_HIP]:.3f}", color='red', fontsize=12, weight='bold')
    ax.text(xs[LEFT_HIP]+0.03, ys[LEFT_HIP], f"{ys[LEFT_HIP]:.3f}", color='red', fontsize=12, weight='bold')
    # Optionally: draw some main bones (Halpe-26: you can adjust as needed)
    for pair in [(5,6), (6,8), (5,7), (11,12)]:
        if max(pair) < kpts.shape[0] and not np.any(np.isnan(kpts[list(pair), :2])):
            ax.plot(xs[list(pair)], ys[list(pair)], c='gray', lw=2, zorder=1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.grid(True, ls="--", alpha=0.3)

def main():
    parser = argparse.ArgumentParser(description="Debug: skeleton hip heights overlay")
    parser.add_argument("--input", required=True, help=".npy file, shape (F, K, 2/3)")
    parser.add_argument("--out", default=None, help="Output video .avi")
    parser.add_argument("--show", action="store_true", help="Show animation live")
    args = parser.parse_args()

    arr = np.load(args.input)
    print("Input skeleton shape:", arr.shape)
    F, K, D = arr.shape
    if D > 2:
        arr = arr[:, :, :2]  # Only use (x, y), drop confidence

    if args.out:
        fig, ax = plt.subplots(figsize=(4, 4))
        writer = FFMpegWriter(fps=30, metadata=dict(artist='debug_hip_y'), codec='mpeg4')
        with writer.saving(fig, args.out, dpi=150):
            for f in tqdm(range(F)):
                plot_skeleton_with_hip_y(ax, arr[f], frame_idx=f)
                writer.grab_frame()
        print(f"✅ Saved debug video to {args.out}")

    if args.show:
        plt.ion()
        fig, ax = plt.subplots(figsize=(4, 4))
        for f in range(F):
            plot_skeleton_with_hip_y(ax, arr[f], frame_idx=f)
            plt.pause(0.03)
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()
