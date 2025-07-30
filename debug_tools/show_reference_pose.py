import numpy as np
import matplotlib.pyplot as plt

# --- SET THIS TO YOUR FILE ---
FILE = "reference_frame.npy"

# --- COCO/17 stickman edges for clarity (optional) ---
EDGES = [
    (0,1),(1,2),(2,3),(3,4),    # nose to right arm
    (0,5),(5,6),(6,7),(7,8),    # nose to left arm
    (0,9),(9,10),(10,11),(11,12), # nose to left leg
    (0,13),(13,14),(14,15),(15,16) # nose to right leg
]

def show_pose(path):
    kp = np.load(path)  # shape = (17,2) or (17,3)
    if kp.shape[1] == 3:
        kp = kp[:, :2]
    xs, ys = kp[:, 0], kp[:, 1]
    plt.figure(figsize=(5, 5))
    plt.scatter(xs, ys, c='purple', s=50, label="joints")
    # Draw stickman
    for i, j in EDGES:
        if not (np.any(np.isnan(kp[i])) or np.any(np.isnan(kp[j]))):
            plt.plot([xs[i], xs[j]], [ys[i], ys[j]], 'k-', lw=2)
    plt.title("Reference Pose")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    # Optionally invert if image coordinates (top-left origin)
    if np.ptp(ys) > 0 and ys.mean() > 100:  # heuristic
        plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_pose(FILE)
