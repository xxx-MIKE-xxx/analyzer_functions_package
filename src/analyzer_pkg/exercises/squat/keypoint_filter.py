#!/usr/bin/env python3
import numpy as np
from scipy.signal import medfilt
import argparse
from pathlib import Path

def remove_spikes(keypoints, kernel_size):
    """
    Applies a median filter to each coordinate of every keypoint over time.

    Parameters
    ----------
    keypoints : np.ndarray
        Array with shape (num_frames, num_keypoints, dims)
    kernel_size : int
        Size of the median filter window (must be odd).

    Returns
    -------
    np.ndarray
        Keypoints array with reduced random spikes.
    """
    num_frames, num_keypoints, dims = keypoints.shape
    filtered = np.copy(keypoints)

    for kp in range(num_keypoints):
        for d in range(dims):
            filtered[:, kp, d] = medfilt(keypoints[:, kp, d],
                                         kernel_size=kernel_size)
    return filtered


# --------------------------------------------------------------------------- #
# New convenience wrapper – keeps the CLI untouched
# --------------------------------------------------------------------------- #
def pipeline(keypoints_input, kernel_size: int = 3) -> np.ndarray:
    """
    Simple programmatic wrapper around `remove_spikes`.

    Parameters
    ----------
    keypoints_input : str | Path | np.ndarray
        • Path to an `.npy` file holding (F, K, D) keypoints **or**
        • An already-loaded NumPy array of that shape.
    kernel_size : int, default 3
        Median-filter window. Even values are auto-bumped to the next odd.

    Returns
    -------
    np.ndarray
        The filtered keypoints array (same shape as input).
    """
    if isinstance(keypoints_input, (str, Path)):
        keypoints = np.load(keypoints_input)
    else:
        keypoints = np.asarray(keypoints_input)

    if kernel_size % 2 == 0:
        kernel_size += 1  # ensure odd

    return remove_spikes(keypoints, kernel_size)
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description='Filter keypoints to remove random spikes'
    )
    parser.add_argument('--input',  required=True,
                        help='Input keypoints file (.npy)')
    parser.add_argument('--output', required=True,
                        help='Output filtered keypoints file (.npy)')
    parser.add_argument('--kernel', type=int, default=3,
                        help='Median filter kernel size (odd number, default: 3)')
    args = parser.parse_args()

    if args.kernel % 2 == 0:
        print(f"Warning: Kernel size {args.kernel} is even. Using {args.kernel + 1}.")
        args.kernel += 1

    print(f"Loading keypoints from {args.input}…")
    keypoints = np.load(args.input)
    print(f"Original keypoints shape: {keypoints.shape}")

    print(f"Applying median filter with kernel size {args.kernel}…")
    filtered_keypoints = remove_spikes(keypoints, args.kernel)
    print(f"Filtered keypoints shape: {filtered_keypoints.shape}")

    print(f"Saving filtered keypoints to {args.output}…")
    # np.save(args.output, filtered_keypoints)
    print("Completed!")

if __name__ == "__main__":
    main()
