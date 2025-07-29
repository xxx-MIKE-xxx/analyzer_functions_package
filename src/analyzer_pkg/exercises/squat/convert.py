#!/usr/bin/env python3
"""
AlphaPose → 17-keypoint converter
=================================
[docstring unchanged …]
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------- Utilities ------------------------------------ #

def load_skeleton(path):
    data = np.load(path)
    print(f"Loaded {path} → shape {data.shape}")
    return data

def impute_spine(frame):
    """Place a spine key-point ⅔ of the way from shoulders to hips."""
    l_sh, r_sh   = frame[5],  frame[6]
    l_hip, r_hip = frame[11], frame[12]
    neck   = (l_sh + r_sh) / 2
    pelvis = (l_hip + r_hip) / 2
    return neck + (2/3) * (pelvis - neck)

# Halpe-26 → 17 mapping (spine inserted at index 16)
MAPPING = {
     0: 0,  5: 1,  6: 2,  7: 3,  8: 4,  9: 5, 10: 6,
    11: 7, 12: 8, 13: 9, 14:10, 15:11, 16:12,
    17:13, 18:14, 19:15               # 16 reserved for spine
}

def reorder_frame(frame, spine):
    out = np.zeros((17, 3), dtype=frame.dtype)
    for src, dst in MAPPING.items():
        out[dst] = frame[src]
    out[16] = spine
    return out

# -------------------------- Processing Loop ------------------------------- #

def process_all_frames(data):
    F = data.shape[0]
    out = np.zeros((F, 17, 3), dtype=data.dtype)
    for i in range(F):
        spine   = impute_spine(data[i])
        out[i]  = reorder_frame(data[i], spine)
        if i % 50 == 0:
            print(f"Processed frame {i}/{F}")
    print(f"All {F} frames processed.")
    return out

# ------------------------  NEW: simple pipeline  -------------------------- #

def pipeline(input_npy: str | Path | np.ndarray) -> np.ndarray:
    """
    Convenience helper for other modules.

    Parameters
    ----------
    input_npy : str | Path | np.ndarray
        • Path to a (F, 26, 3) AlphaPose HALPE-26 `.npy`, **or**
        • An already-loaded NumPy array of that shape.

    Returns
    -------
    np.ndarray
        A (F, 17, 2) array with z-axis dropped and spine imputed.
    """
    # 1) load if a path was given
    if not isinstance(input_npy, np.ndarray):
        input_npy = load_skeleton(Path(input_npy))

    # 2) convert
    converted_3d = process_all_frames(input_npy)

    # 3) drop z; hand back
    return converted_3d[..., :2]

# ----------------------------- Main --------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Convert AlphaPose Halpe-26 → 17×(x,y)")
    p.add_argument('--input',            required=True,
                   help='AlphaPose .npy (F,26,3)')
    p.add_argument('--output-2d',        required=True,
                   help='Output 2D sequence .npy (F,17,2)')
    p.add_argument('--output-reference', default=None,
                   help='(Optional) save one reference pose here')
    p.add_argument('--reference-frame',  type=int, default=45,
                   help='Index to use for reference pose')
    p.add_argument('--visualize',        action='store_true',
                   help='Plot the reference pose for sanity check')
    args = p.parse_args()

    data3d = load_skeleton(args.input)
    if data3d.ndim != 3 or data3d.shape[1] < 20 or data3d.shape[2] < 3:
        raise ValueError("Expect (F,26,3) input array")

    # 1) convert all frames
    processed3d = process_all_frames(data3d)
    processed2d = processed3d[..., :2]         # drop z-axis
    np.save(args.output_2d, processed2d)
    print(f"Saved 2D sequence → {args.output_2d}  shape={processed2d.shape}")

    # 2) optionally write reference pose
    if args.output_reference:
        F = processed2d.shape[0]
        if not (0 <= args.reference_frame < F):
            raise IndexError(f"reference-frame must be 0…{F-1}")
        ref = processed2d[args.reference_frame]
        np.save(args.output_reference, ref)
        print(f"Saved reference pose frame {args.reference_frame} → {args.output_reference}")

        if args.visualize:
            fig, ax = plt.subplots()
            ax.scatter(ref[:, 0], ref[:, 1], c='purple')
            ax.scatter(ref[16, 0], ref[16, 1], c='green', s=100, label='spine')
            ax.invert_yaxis()
            ax.set_title(f"Reference pose (frame {args.reference_frame})")
            ax.legend()
            plt.show()

if __name__ == '__main__':
    main()
