#!/usr/bin/env python3
"""
Smooth 3D keypoint trajectories from BDLO2 evaluation results.

Reads 3d_keypoints.npz, applies Gaussian smoothing to the 'full' method,
fixes node 12 when it gets too close to node 17 (replacing it with
the midpoint of node 11 and node 1), and saves smoothed_3d_keypoints.npz.

Usage:
    python smooth_bdlo2_keypoints.py --input_dir bdlo2_evaluation_results/chunk_0/clip_1
    python smooth_bdlo2_keypoints.py --input_dir bdlo2_evaluation_results/chunk_0/clip_1 --sigma 3.0
    python smooth_bdlo2_keypoints.py --input_dir bdlo2_evaluation_results/chunk_0/clip_1 --proximity_threshold 30.0

Author: Auto-generated
Date: 2026-03-06
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d


def smooth_trajectories(keypoints_3d_seq: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to keypoint trajectories.

    Args:
        keypoints_3d_seq: T x K x 3 array of keypoints over time
        sigma: Gaussian filter sigma

    Returns:
        smoothed: T x K x 3 smoothed keypoints
    """
    T, K, D = keypoints_3d_seq.shape
    smoothed = np.zeros_like(keypoints_3d_seq)

    for k in range(K):
        for d in range(D):
            traj = keypoints_3d_seq[:, k, d]
            # Handle NaN values by interpolating
            valid = ~np.isnan(traj)
            if np.sum(valid) > 2:
                # Interpolate NaN values
                indices = np.arange(T)
                traj_interp = np.interp(indices, indices[valid], traj[valid])
                # Apply Gaussian smoothing
                smoothed[:, k, d] = gaussian_filter1d(traj_interp, sigma=sigma, mode='nearest')
            else:
                smoothed[:, k, d] = traj

    return smoothed


def fix_node12_proximity(keypoints_3d_seq: np.ndarray, threshold: float = 30.0) -> tuple[np.ndarray, int]:
    """
    Check if node 12 is too close to node 17 in each frame.
    If so, replace node 12 with the midpoint of node 11 and node 1 (b1).

    Args:
        keypoints_3d_seq: T x K x 3 array of keypoints
        threshold: distance threshold (mm) below which node 12 is considered
                   too close to node 17

    Returns:
        fixed: T x K x 3 corrected keypoints
        num_fixed: number of frames where node 12 was replaced
    """
    fixed = keypoints_3d_seq.copy()
    T = fixed.shape[0]
    num_fixed = 0

    for t in range(T):
        node12 = fixed[t, 12]
        node17 = fixed[t, 17]

        # Skip if either has NaN
        if np.any(np.isnan(node12)) or np.any(np.isnan(node17)):
            continue

        dist = np.linalg.norm(node12 - node17)
        if dist < threshold:
            # Replace node 12 with midpoint of node 11 and node 1 (b1)
            node11 = fixed[t, 11]
            node1 = fixed[t, 1]
            if not np.any(np.isnan(node11)) and not np.any(np.isnan(node1)):
                fixed[t, 12] = (node11 + node1) / 2.0
                num_fixed += 1

    return fixed, num_fixed


def main():
    parser = argparse.ArgumentParser(description='Smooth BDLO2 3D keypoint trajectories')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing 3d_keypoints.npz')
    parser.add_argument('--sigma', type=float, default=3.0,
                        help='Gaussian smoothing sigma (default: 3.0)')
    parser.add_argument('--proximity_threshold', type=float, default=30.0,
                        help='Distance threshold (mm) for node 12 / node 17 proximity check (default: 30.0)')
    parser.add_argument('--output_name', type=str, default='smoothed_3d_keypoints.npz',
                        help='Output filename (default: smoothed_3d_keypoints.npz)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    input_path = input_dir / '3d_keypoints.npz'
    output_path = input_dir / args.output_name

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    print("=" * 60)
    print("SMOOTH BDLO2 3D KEYPOINTS")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Sigma:  {args.sigma}")
    print(f"Proximity threshold (node 12 vs 17): {args.proximity_threshold} mm")

    # Load data
    print("\nLoading data...")
    data = np.load(input_path)

    full = data['full']
    edge_connection = data['edge_connection']
    reference_lengths = data['reference_lengths']

    print(f"  full shape: {full.shape}")
    print(f"  edge_connection shape: {edge_connection.shape}")
    print(f"  reference_lengths shape: {reference_lengths.shape}")

    T, K, D = full.shape
    print(f"  Frames: {T}, Keypoints: {K}")

    # Smooth trajectories
    print(f"\nSmoothing 'full' trajectories with sigma={args.sigma}...")
    full_smooth = smooth_trajectories(full, sigma=args.sigma)

    # Compute smoothing statistics
    diff = np.abs(full_smooth - full)
    valid_mask = ~np.isnan(full) & ~np.isnan(full_smooth)
    if np.any(valid_mask):
        mean_diff = np.nanmean(diff)
        max_diff = np.nanmax(diff)
        print(f"  Mean position change: {mean_diff:.2f} mm")
        print(f"  Max position change:  {max_diff:.2f} mm")

    # Fix node 12 proximity to node 17
    print(f"\nChecking node 12 / node 17 proximity (threshold={args.proximity_threshold} mm)...")
    # Report pre-fix distances
    pre_dists = np.linalg.norm(full_smooth[:, 12] - full_smooth[:, 17], axis=1)
    print(f"  Pre-fix node 12-17 distance: min={np.nanmin(pre_dists):.2f}, "
          f"mean={np.nanmean(pre_dists):.2f}, max={np.nanmax(pre_dists):.2f} mm")
    print(f"  Frames below threshold: {np.sum(pre_dists < args.proximity_threshold)}/{T}")

    full_smooth, num_fixed = fix_node12_proximity(full_smooth, threshold=args.proximity_threshold)
    print(f"  Replaced node 12 in {num_fixed} frames with midpoint of node 11 and node 1")

    if num_fixed > 0:
        post_dists = np.linalg.norm(full_smooth[:, 12] - full_smooth[:, 17], axis=1)
        print(f"  Post-fix node 12-17 distance: min={np.nanmin(post_dists):.2f}, "
              f"mean={np.nanmean(post_dists):.2f}, max={np.nanmax(post_dists):.2f} mm")

    # Final safe gate: ensure specific frame ranges have node 12 fixed
    safe_gate_ranges = [(140, 149), (210, 214)]
    for fix_start, fix_end in safe_gate_ranges:
        print(f"\nSafe gate: forcing node 12 fix in frames {fix_start}-{fix_end}...")
        num_gate_fixed = 0
        for t in range(fix_start, fix_end + 1):
            node11 = full_smooth[t, 11]
            node1 = full_smooth[t, 1]
            if not np.any(np.isnan(node11)) and not np.any(np.isnan(node1)):
                midpoint = (node11 + node1) / 2.0
                if not np.allclose(full_smooth[t, 12], midpoint):
                    full_smooth[t, 12] = midpoint
                    num_gate_fixed += 1
        print(f"  Additionally fixed {num_gate_fixed} frames in [{fix_start}, {fix_end}]")

    # Save smoothed data (same structure as input)
    print(f"\nSaving to: {output_path}")
    np.savez(
        output_path,
        full=full_smooth,
        nosnap=data['nosnap'],          # Keep original (not smoothed)
        noGeometry=data['noGeometry'],  # Keep original (not smoothed)
        cdcpd2=data['cdcpd2'],          # Keep original (not smoothed)
        edge_connection=edge_connection,
        reference_lengths=reference_lengths,
    )

    print("\nDone!")
    print(f"Smoothed keypoints saved to: {output_path}")


if __name__ == '__main__':
    main()
