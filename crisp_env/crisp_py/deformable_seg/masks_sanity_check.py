#!/usr/bin/env python3
"""
Sanity check: Visualize point cloud of each mask and save as PNG.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def extract_point_cloud(mask, depth, max_depth, fx, fy, cx, cy):
    """Extract 3D point cloud from mask and depth."""
    valid = mask & (depth > 0) & (depth < max_depth)
    rows, cols = np.where(valid)
    
    if len(rows) == 0:
        return np.array([]).reshape(0, 3)
    
    z = depth[rows, cols]
    x = (cols - cx) * z / fx
    y = (rows - cy) * z / fy
    
    return np.stack([x, y, z], axis=1)


def visualize_point_cloud(point_cloud, title, save_path, max_points=5000):
    """Visualize point cloud and save as PNG."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample if too many points
    if len(point_cloud) > max_points:
        indices = np.random.choice(len(point_cloud), max_points, replace=False)
        pc = point_cloud[indices]
    else:
        pc = point_cloud
    
    if len(pc) > 0:
        # Color by depth (z)
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], 
                   c=pc[:, 2], cmap='viridis', s=1, alpha=0.5)
        
        # Set labels
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        # Set view angle (top-down like camera view)
        ax.view_init(elev=-90, azim=-90)
        
        # Add stats
        stats_text = f"Points: {len(point_cloud)}\n"
        stats_text += f"X: [{pc[:, 0].min():.0f}, {pc[:, 0].max():.0f}]\n"
        stats_text += f"Y: [{pc[:, 1].min():.0f}, {pc[:, 1].max():.0f}]\n"
        stats_text += f"Z: [{pc[:, 2].min():.0f}, {pc[:, 2].max():.0f}]"
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 0.5, 'No points!', fontsize=20, ha='center')
    
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return len(point_cloud)


def main():
    # Paths
    data_dir = Path("/home/yehengz/deformable_seg/data")
    tracking_data_path = data_dir / "full" / "tracking_fabric2_data.npy"
    masks_dir = data_dir / "arm_traj4_fabric" / "masks"
    output_dir = data_dir / "arm_traj4_fabric" / "masks_sanity_check"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Camera intrinsics
    fx, fy, cx, cy = 606.6875, 606.24609375, 641.7900390625, 366.8428955078125
    max_depth = 1100.0  # Increased from 1100 to avoid clipping fabric
    
    # Load tracking data
    print("Loading tracking data...")
    tracking_data = np.load(str(tracking_data_path), allow_pickle=True).item()
    frame_keys = sorted([k for k in tracking_data.keys() if isinstance(k, int)])
    print(f"Found {len(frame_keys)} frames")
    
    # Get list of mask files
    mask_files = sorted(masks_dir.glob("mask_frame_*.npy"))
    print(f"Found {len(mask_files)} mask files")
    
    # Process each mask
    summary = []
    for i, mask_file in enumerate(mask_files):
        # Extract frame index from filename
        frame_idx = int(mask_file.stem.split('_')[-1])
        
        # Load mask
        mask_raw = np.load(str(mask_file))
        
        # Get depth for this frame
        if frame_idx in tracking_data:
            depth = tracking_data[frame_idx]['transformed_depth']
        elif frame_idx < len(frame_keys):
            depth = tracking_data[frame_keys[frame_idx]]['transformed_depth']
        else:
            print(f"  Frame {frame_idx}: No depth data, skipping")
            continue
        
        # Apply depth thresholding
        valid_depth = (depth > 0) & (depth < max_depth)
        mask = mask_raw & valid_depth
        
        # Extract point cloud
        point_cloud = extract_point_cloud(mask, depth, max_depth, fx, fy, cx, cy)
        
        # Visualize and save
        title = f"Frame {frame_idx:04d} - Mask: {np.sum(mask_raw)} -> {np.sum(mask)} (depth filtered)"
        save_path = output_dir / f"pc_frame_{frame_idx:04d}.png"
        
        n_points = visualize_point_cloud(point_cloud, title, save_path)
        
        summary.append({
            'frame': frame_idx,
            'mask_raw': np.sum(mask_raw),
            'mask_filtered': np.sum(mask),
            'n_points': n_points,
        })
        
        if i % 10 == 0:
            print(f"  Frame {frame_idx:04d}: {np.sum(mask_raw)} -> {np.sum(mask)} pixels, {n_points} points")
    
    # Save summary
    print(f"\nSaved {len(summary)} visualizations to: {output_dir}")
    
    # Print summary stats
    if summary:
        mask_raw_counts = [s['mask_raw'] for s in summary]
        mask_filtered_counts = [s['mask_filtered'] for s in summary]
        
        print(f"\nSummary:")
        print(f"  Raw mask pixels: min={min(mask_raw_counts)}, max={max(mask_raw_counts)}, mean={np.mean(mask_raw_counts):.0f}")
        print(f"  Filtered mask pixels: min={min(mask_filtered_counts)}, max={max(mask_filtered_counts)}, mean={np.mean(mask_filtered_counts):.0f}")
        
        # Find problematic frames
        problematic = [s for s in summary if s['mask_filtered'] < 1000]
        if problematic:
            print(f"\nProblematic frames (< 1000 pixels after filtering):")
            for s in problematic:
                print(f"  Frame {s['frame']}: {s['mask_raw']} -> {s['mask_filtered']}")


if __name__ == "__main__":
    main()
