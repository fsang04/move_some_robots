#!/usr/bin/env python3
"""
Visualize foreground point cloud for frames 265 and 266 using Plotly.
"""

import numpy as np
import plotly.graph_objects as go
from pathlib import Path


def extract_point_cloud(mask, depth, rgb, max_depth, fx, fy, cx, cy):
    """Extract 3D point cloud from mask and depth with colors."""
    valid = mask & (depth > 0) & (depth < max_depth)
    rows, cols = np.where(valid)
    
    if len(rows) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    z = depth[rows, cols].astype(np.float64)
    x = (cols - cx) * z / fx
    y = (rows - cy) * z / fy
    
    colors = rgb[rows, cols]
    
    return np.stack([x, y, z], axis=1), colors


def create_plotly_viz(point_cloud, colors, title, save_path):
    """Create interactive Plotly 3D scatter plot."""
    
    # Subsample if too many points
    max_points = 50000
    if len(point_cloud) > max_points:
        indices = np.random.choice(len(point_cloud), max_points, replace=False)
        pc = point_cloud[indices]
        c = colors[indices]
    else:
        pc = point_cloud
        c = colors
    
    # Convert colors to plotly format
    color_strs = [f'rgb({r},{g},{b})' for r, g, b in c]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=pc[:, 0],
        y=pc[:, 1],
        z=pc[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=color_strs,
            opacity=0.8
        ),
        hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
        ),
        width=1200,
        height=900,
    )
    
    fig.write_html(str(save_path))
    print(f"Saved: {save_path}")


def main():
    # Paths
    data_path = Path("/home/yehengz/deformable_seg/data/full/tracking_fabric2_data.npy")
    masks_dir = Path("/home/yehengz/deformable_seg/data/arm_traj4_fabric/masks")
    output_dir = Path("/home/yehengz/deformable_seg/data/arm_traj4_fabric")
    
    # Camera intrinsics
    fx, fy = 606.1124267578125, 605.8821411132812
    cx, cy = 641.7578125, 365.6518859863281
    max_depth = 1100.0
    
    # Load tracking data
    print("Loading tracking data...")
    tracking_data = np.load(str(data_path), allow_pickle=True).item()
    frame_keys = sorted(tracking_data.keys())
    
    # Process frames 265 and 266
    for frame_idx in [265, 266]:
        print(f"\nProcessing frame {frame_idx}...")
        
        # Get data
        frame_key = frame_keys[frame_idx]
        data = tracking_data[frame_key]
        
        rgb = data['color'][:, :, ::-1]  # BGR to RGB
        depth = data['transformed_depth']
        
        # Load mask
        mask_path = masks_dir / f"mask_frame_{frame_idx:04d}.npy"
        mask_raw = np.load(str(mask_path))
        
        # Apply depth thresholding
        valid_depth = (depth > 0) & (depth < max_depth)
        mask = mask_raw & valid_depth
        
        print(f"  Mask pixels: {np.sum(mask_raw)} -> {np.sum(mask)} (after depth filter)")
        
        # Extract point cloud
        point_cloud, colors = extract_point_cloud(
            mask, depth, rgb, max_depth, fx, fy, cx, cy
        )
        
        print(f"  Point cloud: {len(point_cloud)} points")
        if len(point_cloud) > 0:
            print(f"  X range: [{point_cloud[:, 0].min():.1f}, {point_cloud[:, 0].max():.1f}]")
            print(f"  Y range: [{point_cloud[:, 1].min():.1f}, {point_cloud[:, 1].max():.1f}]")
            print(f"  Z range: [{point_cloud[:, 2].min():.1f}, {point_cloud[:, 2].max():.1f}]")
        
        # Create visualization
        save_path = output_dir / f"foreground_pc_frame_{frame_idx}.html"
        title = f"Frame {frame_idx} - Foreground Point Cloud ({len(point_cloud)} points)"
        create_plotly_viz(point_cloud, colors, title, save_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
