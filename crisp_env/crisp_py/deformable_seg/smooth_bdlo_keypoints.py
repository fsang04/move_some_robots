#!/usr/bin/env python3
"""
Smooth 3D keypoint trajectories from BDLO evaluation results.

Reads 3d_keypoints.npz, applies Gaussian smoothing to the 'full' method,
and saves smoothed_3d_keypoints.npz with the same structure.

Usage:
    python smooth_bdlo_keypoints.py --input_dir bdlo1_faster_free_ee_evaluation_results/chunk_0/clip_0
    python smooth_bdlo_keypoints.py --input_dir bdlo1_faster_free_ee_evaluation_results/chunk_0/clip_0 --sigma 3.0

Author: Auto-generated
Date: 2026-03-03
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


def main():
    parser = argparse.ArgumentParser(description='Smooth BDLO 3D keypoint trajectories')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing 3d_keypoints.npz')
    parser.add_argument('--sigma', type=float, default=3.0,
                        help='Gaussian smoothing sigma (default: 2.0)')
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
    print("SMOOTH BDLO 3D KEYPOINTS")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Sigma:  {args.sigma}")
    
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
    
    # Save smoothed data (same structure as input)
    print(f"\nSaving to: {output_path}")
    np.savez(
        output_path,
        full=full_smooth,
        nosnap=data['nosnap'],      # Keep original (not smoothed)
        noGeometry=data['noGeometry'],  # Keep original (not smoothed)
        cdcpd2=data['cdcpd2'],      # Keep original (not smoothed)
        edge_connection=edge_connection,
        reference_lengths=reference_lengths,
    )
    
    print("\nDone!")
    print(f"Smoothed keypoints saved to: {output_path}")


if __name__ == '__main__':
    main()
