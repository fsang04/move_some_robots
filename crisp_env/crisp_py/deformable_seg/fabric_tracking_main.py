"""
Fabric Tracking Main Script

Uses FabricTracker class for 5x5 grid-based fabric tracking.

Pipeline:
    1. Load precomputed masks and tracking data
    2. Load EE poses from get_ee_pose_fabric.py output
    3. Run FabricTracker on each frame
    4. Visualize and save results (purple trajectory tails)

Usage:
    python fabric_tracking_main.py

Author: Auto-generated
Date: 2026-02-23
"""

import numpy as np
import cv2
import time
import gc
import io
import argparse
from pathlib import Path

import PIL.Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fabric_tracker import FabricTracker


# ============================================================================
# CONFIGURATION
# ============================================================================

# Start frame (skip initial frames where fabric may not be fully visible)
START_FRAME = 18


# ============================================================================
# CAMERA INTRINSICS
# ============================================================================

INTRINSICS = np.array([
    [606.1124267578125, 0, 641.7578125],
    [0, 605.8821411132812, 365.6518859863281],
    [0, 0, 1]
], dtype=np.float64)


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualization(
    rgb: np.ndarray,
    foreground_mask: np.ndarray,
    keypoints_2d: np.ndarray,
    edges: list,
    mode: str = 'track',
    frame_idx: int = 0,
    traj_history_2d: np.ndarray = None,
    tail_length: int = 60,
    ee_to_corner_mapping: dict = None,
    corner_indices: list = None,
    border_indices: list = None,
) -> np.ndarray:
    """
    Create 2x2 visualization grid with purple trajectory tails.
    
    Layout:
        [Mask]             [Mask Overlay]
        [Keypoints Only]   [Keypoints Overlay]
    
    Args:
        rgb: H × W × 3 RGB image
        foreground_mask: H × W binary mask
        keypoints_2d: K × 2 keypoint pixel coords (row, col)
        edges: List of (i, j) edge tuples
        mode: 'init', 'track', 'restart', 'skip'
        frame_idx: Current frame index
        traj_history_2d: T × K × 2 trajectory history for all keypoints
        tail_length: Number of frames for trajectory tail
        ee_to_corner_mapping: Dict mapping EE index to corner keypoint index
        corner_indices: List of corner node indices (from tracker)
        border_indices: List of border node indices (from tracker)
    
    Returns:
        grid: Visualization grid
    """
    H, W = rgb.shape[:2]
    
    # Colors (purple theme for fabric)
    MASK_COLOR = [0, 255, 0]       # Green for mask (fabric is green)
    EDGE_COLOR = [255, 165, 0]     # Orange for edges
    CORNER_COLOR = [255, 0, 0]     # Red for corner nodes (EE-held)
    BORDER_COLOR = [255, 255, 0]   # Yellow for border nodes
    INTERIOR_COLOR = [0, 255, 255] # Cyan for interior nodes
    TAIL_COLOR = [255, 105, 180]   # Hot pink for trajectory tails
    
    # Use provided indices or empty lists
    CORNER_INDICES = corner_indices if corner_indices is not None else []
    BORDER_INDICES = border_indices if border_indices is not None else []
    
    # Visualization parameters
    KEYPOINT_RADIUS = 4
    EDGE_THICKNESS = 2
    TAIL_THICKNESS = 2
    
    def draw_trajectory_tail(canvas, traj_history, tail_length):
        """Draw purple trajectory tails for all keypoints."""
        if traj_history is None or len(traj_history) == 0:
            return
        
        T_hist, K, _ = traj_history.shape
        actual_tail = min(tail_length, T_hist)
        
        for idx in range(K):
            traj = traj_history[-actual_tail:, idx, :]
            
            for t in range(len(traj) - 1):
                pt1 = traj[t]
                pt2 = traj[t + 1]
                
                if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
                    continue
                
                row1, col1 = int(pt1[0]), int(pt1[1])
                row2, col2 = int(pt2[0]), int(pt2[1])
                
                if not (0 <= row1 < H and 0 <= col1 < W):
                    continue
                if not (0 <= row2 < H and 0 <= col2 < W):
                    continue
                
                alpha = (t + 1) / len(traj)
                color = [int(c * alpha) for c in TAIL_COLOR]
                cv2.line(canvas, (col1, row1), (col2, row2), color, TAIL_THICKNESS)
    
    # Row 1: Mask + Mask overlay
    mask_vis = np.zeros((H, W, 3), dtype=np.uint8)
    mask_vis[foreground_mask > 0] = MASK_COLOR
    
    mask_overlay = rgb.copy()
    # Draw mask contour
    contours, _ = cv2.findContours(
        foreground_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(mask_overlay, contours, -1, MASK_COLOR, 2)
    
    # Add mode and frame label to Row 1
    cv2.putText(mask_overlay, f"Mode: {mode}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(mask_overlay, f"Frame: {frame_idx}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Row 2: Keypoints + Keypoints overlay
    keypoint_vis = np.zeros((H, W, 3), dtype=np.uint8)
    keypoint_vis[foreground_mask > 0] = [30, 30, 30]  # Dim mask background
    
    keypoint_overlay = rgb.copy()
    
    # Draw trajectory tails first (purple)
    draw_trajectory_tail(keypoint_vis, traj_history_2d, tail_length)
    draw_trajectory_tail(keypoint_overlay, traj_history_2d, tail_length)
    
    # Draw edges
    if keypoints_2d is not None and len(keypoints_2d) > 0 and edges is not None:
        kp_int = keypoints_2d.astype(int)
        for (i, j) in edges:
            if i < len(kp_int) and j < len(kp_int):
                row1, col1 = kp_int[i]
                row2, col2 = kp_int[j]
                if (0 <= row1 < H and 0 <= col1 < W and 
                    0 <= row2 < H and 0 <= col2 < W):
                    cv2.line(keypoint_vis, (col1, row1), (col2, row2), EDGE_COLOR, EDGE_THICKNESS)
                    cv2.line(keypoint_overlay, (col1, row1), (col2, row2), EDGE_COLOR, EDGE_THICKNESS)
        
        # Draw keypoints with different colors based on type
        for idx, (row, col) in enumerate(kp_int):
            if 0 <= row < H and 0 <= col < W:
                if idx in CORNER_INDICES:
                    color = CORNER_COLOR
                elif idx in BORDER_INDICES:
                    color = BORDER_COLOR
                else:
                    color = INTERIOR_COLOR
                
                cv2.circle(keypoint_vis, (col, row), KEYPOINT_RADIUS, color, -1)
                cv2.circle(keypoint_overlay, (col, row), KEYPOINT_RADIUS, color, -1)
                
                # Draw white outline
                cv2.circle(keypoint_vis, (col, row), KEYPOINT_RADIUS + 1, (255, 255, 255), 1)
                cv2.circle(keypoint_overlay, (col, row), KEYPOINT_RADIUS + 1, (255, 255, 255), 1)
    
    # Add mode and frame label to Row 2
    cv2.putText(keypoint_overlay, f"Mode: {mode}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(keypoint_overlay, f"Frame: {frame_idx}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add legend
    legend_y = 90
    cv2.circle(keypoint_overlay, (20, legend_y), 6, CORNER_COLOR, -1)
    cv2.putText(keypoint_overlay, "Corner (EE)", (35, legend_y + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(keypoint_overlay, (20, legend_y + 25), 6, BORDER_COLOR, -1)
    cv2.putText(keypoint_overlay, "Border", (35, legend_y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(keypoint_overlay, (20, legend_y + 50), 6, INTERIOR_COLOR, -1)
    cv2.putText(keypoint_overlay, "Interior", (35, legend_y + 55), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.line(keypoint_overlay, (15, legend_y + 75), (40, legend_y + 75), TAIL_COLOR, 2)
    cv2.putText(keypoint_overlay, "Trajectory", (50, legend_y + 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Create grid (2x2)
    row1 = np.concatenate([mask_vis, mask_overlay], axis=1)
    row2 = np.concatenate([keypoint_vis, keypoint_overlay], axis=1)
    
    grid = np.concatenate([row1, row2], axis=0)
    
    return grid


def create_3d_visualization(
    keypoints_3d: np.ndarray,
    edges: list,
    full_pc: np.ndarray,
    pc_colors: np.ndarray,
    frame_idx: int = 0,
    mode: str = 'track',
    pc_downsample: int = 500,
    figsize: tuple = (16, 16),
    fixed_xlim: tuple = None,
    fixed_ylim: tuple = None,
    fixed_zlim: tuple = None,
    traj_history_3d: np.ndarray = None,
    tail_length: int = 60,
    corner_indices: list = None,
    border_indices: list = None,
) -> np.ndarray:
    """
    Create 2x2 3D visualization grid with purple trajectory tails.
    """
    # Colors
    CORNER_COLOR = np.array([255, 0, 0]) / 255.0      # Red
    BORDER_COLOR = np.array([255, 255, 0]) / 255.0    # Yellow
    INTERIOR_COLOR = np.array([0, 255, 255]) / 255.0  # Cyan
    EDGE_COLOR = np.array([255, 165, 0]) / 255.0      # Orange
    TAIL_COLOR = np.array([255, 105, 180]) / 255.0    # Hot pink
    
    # Use provided indices or empty lists
    CORNER_INDICES = corner_indices if corner_indices is not None else []
    BORDER_INDICES = border_indices if border_indices is not None else []
    
    fig = plt.figure(figsize=figsize)
    
    # Downsample point cloud
    if len(full_pc) > pc_downsample:
        indices = np.random.choice(len(full_pc), pc_downsample, replace=False)
        pc_downsampled = full_pc[indices]
        colors_downsampled = pc_colors[indices] / 255.0
    else:
        pc_downsampled = full_pc
        colors_downsampled = pc_colors / 255.0 if len(pc_colors) > 0 else np.empty((0, 3))
    
    def setup_ax(ax, title, show_pc=False, elev=100, azim=-90):
        ax.set_title(f"{title}\nFrame: {frame_idx} | Mode: {mode}", fontsize=10)
        
        if show_pc and len(pc_downsampled) > 0:
            ax.scatter(pc_downsampled[:, 0], pc_downsampled[:, 1], pc_downsampled[:, 2],
                      c=colors_downsampled, s=2, alpha=0.7, zorder=1, depthshade=False)
        
        # Draw trajectory tails (purple)
        if traj_history_3d is not None and len(traj_history_3d) > 0:
            T_hist = traj_history_3d.shape[0]
            actual_tail = min(tail_length, T_hist)
            K = traj_history_3d.shape[1]
            
            for idx in range(K):
                traj = traj_history_3d[-actual_tail:, idx, :]
                valid = ~np.any(np.isnan(traj), axis=1)
                if np.sum(valid) > 1:
                    traj_valid = traj[valid]
                    ax.plot(traj_valid[:, 0], traj_valid[:, 1], traj_valid[:, 2],
                           c=TAIL_COLOR, linewidth=1, alpha=0.7, zorder=2)
        
        # Draw edges
        if len(keypoints_3d) > 0 and edges is not None:
            for (i, j) in edges:
                if i < len(keypoints_3d) and j < len(keypoints_3d):
                    pts = keypoints_3d[[i, j], :]
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                           c=EDGE_COLOR, linewidth=2, zorder=3)
        
        # Draw keypoints
        if len(keypoints_3d) > 0:
            for idx in range(len(keypoints_3d)):
                if idx in CORNER_INDICES:
                    color = CORNER_COLOR
                    size = 80/2
                elif idx in BORDER_INDICES:
                    color = BORDER_COLOR
                    size = 60/2
                else:
                    color = INTERIOR_COLOR
                    size = 50/2

                ax.scatter(keypoints_3d[idx, 0], keypoints_3d[idx, 1], keypoints_3d[idx, 2],
                          c=[color], s=size, zorder=4, depthshade=False)
        
        if fixed_xlim is not None:
            ax.set_xlim(fixed_xlim)
        if fixed_ylim is not None:
            ax.set_ylim(fixed_ylim)
        if fixed_zlim is not None:
            ax.set_zlim(fixed_zlim)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.view_init(elev=elev, azim=azim)
        ax.invert_yaxis()
    
    ax1 = fig.add_subplot(221, projection='3d')
    setup_ax(ax1, "View 1: Keypoints + Edges", show_pc=False, elev=100, azim=-90)
    
    ax2 = fig.add_subplot(222, projection='3d')
    setup_ax(ax2, "View 1: + Point Cloud", show_pc=True, elev=100, azim=-90)
    
    ax3 = fig.add_subplot(223, projection='3d')
    setup_ax(ax3, "View 2: Keypoints + Edges", show_pc=False, elev=24, azim=-90)
    
    ax4 = fig.add_subplot(224, projection='3d')
    setup_ax(ax4, "View 2: + Point Cloud", show_pc=True, elev=24, azim=-90)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    
    img = np.array(PIL.Image.open(buf))[:, :, :3]
    plt.close(fig)
    
    return img


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main tracking loop."""
    
    print("=" * 70)
    print("FABRIC TRACKING WITH 5x5 GRID")
    print("=" * 70)
    
    # Paths
    data_path = Path("/home/yehengz/deformable_seg/data/full/tracking_fabric2_data.npy")
    masks_dir = Path("/home/yehengz/deformable_seg/data/arm_traj4_fabric/masks")
    ee_pose_path = Path("/home/yehengz/deformable_seg/data/arm_traj4_fabric/ee_pose_output/ee_poses_3d.npy")
    output_dir = Path("/home/yehengz/deformable_seg/data/arm_traj4_fabric/fabric_tracking_output")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    frames_3d_dir = output_dir / "frames_3d"
    frames_3d_dir.mkdir(exist_ok=True)
    
    # Load data
    print(f"\nLoading tracking data from: {data_path}")
    tracking_data = np.load(str(data_path), allow_pickle=True).item()
    
    frame_keys = sorted(tracking_data.keys())
    total_frames = len(frame_keys)
    print(f"Found {total_frames} frames total")
    
    # Apply start frame offset
    frame_keys = frame_keys[START_FRAME:]
    n_frames = len(frame_keys)
    print(f"Starting from frame {START_FRAME}, {n_frames} frames remaining")
    
    # Load masks
    print(f"\nLoading masks from: {masks_dir}")
    mask_files = sorted(masks_dir.glob("mask_frame_*.npy"))
    print(f"Found {len(mask_files)} mask files")
    
    n_frames = min(n_frames, len(mask_files) - START_FRAME)
    
    # Load EE poses
    ee_poses_3d = None
    if ee_pose_path.exists():
        ee_data = np.load(str(ee_pose_path), allow_pickle=True).item()
        ee_poses_3d_full = ee_data['ee_3d']
        # Slice EE poses to start from START_FRAME
        ee_poses_3d = ee_poses_3d_full[START_FRAME:]
        print(f"Loaded EE poses: full shape = {ee_poses_3d_full.shape}, sliced shape = {ee_poses_3d.shape}")
    else:
        print(f"No EE poses found at: {ee_pose_path}")
    
    # Initialize tracker
    print("\nInitializing FabricTracker...")
    tracker = FabricTracker(
        intrinsics=INTRINSICS,
        max_depth=1080.0,
        # CPD parameters - higher beta = more rigid deformation
        cpd_beta=50.0,
        cpd_lambda=0.1,
        cpd_w=0.1,
        cpd_max_iter=300,
        cpd_downsample=3000,
        cpd_tol=1e-5,
        # Geometry constraints
        n_outer_iterations=5,
        n_edge_iterations=20,
        edge_weight=0.5,
        edge_tolerance=0.05,
        # Repulsion - low lr to prevent divergence
        repulsion_iterations=100,
        repulsion_lr=0.05,
        # EE poses
        ee_poses_3d=ee_poses_3d,
    )
    
    print(f"  Grid size: {tracker.GRID_ROWS}x{tracker.GRID_COLS} = {tracker.N_KEYPOINTS} keypoints")
    print(f"  Edges: {len(tracker.grid_edges)}")
    
    # Compute fixed axis limits
    print("\nComputing fixed axis limits...")
    first_data = tracking_data[frame_keys[0]]
    first_depth = first_data['transformed_depth']
    
    crop_z_min = 600
    crop_z_max = 1400
    crop_x_min = -700
    crop_x_max = 700
    crop_y_min = -500
    crop_y_max = 500
    
    fixed_xlim = (crop_x_min, crop_x_max)
    fixed_ylim = (crop_y_min, crop_y_max)
    fixed_zlim = (crop_z_min, crop_z_max)
    
    print(f"  X: [{fixed_xlim[0]}, {fixed_xlim[1]}]")
    print(f"  Y: [{fixed_ylim[0]}, {fixed_ylim[1]}]")
    print(f"  Z: [{fixed_zlim[0]}, {fixed_zlim[1]}]")
    
    # Process frames
    print(f"\n{'='*70}")
    print("PROCESSING FRAMES")
    print("=" * 70)
    
    all_results = []
    video_writer = None
    video_writer_3d = None
    fps = 30
    
    traj_history_2d = []
    traj_history_3d = []
    tail_length = 60
    
    total_time = 0.0
    
    for i in range(n_frames):
        frame_start = time.time()
        
        frame_key = frame_keys[i]
        data = tracking_data[frame_key]
        
        rgb = data['color'][:, :, ::-1]  # BGR to RGB
        depth = data['transformed_depth']
        
        # Load mask and apply depth thresholding
        # Use original frame index (i + START_FRAME) for mask file
        original_frame_idx = i + START_FRAME
        mask_path = masks_dir / f"mask_frame_{original_frame_idx:04d}.npy"
        mask_raw = np.load(str(mask_path))
        
        # IMPORTANT: Only keep pixels with valid depth (0 < depth < max_depth)
        valid_depth = (depth > 0) & (depth < tracker.max_depth)
        mask = mask_raw & valid_depth
        
        # Process frame (use i as frame_idx for EE pose indexing relative to start)
        result = tracker.process_frame(depth, mask, frame_idx=i)
        
        frame_time = time.time() - frame_start
        total_time += frame_time
        
        success = result.get('success', False)
        mode = result.get('mode', 'unknown')
        foreground_mask = result.get('foreground_mask', np.zeros_like(depth, dtype=np.uint8))
        
        if success:
            keypoints = result['keypoints']
            keypoints_2d = result['keypoints_2d']
            edges = result['edges']
            
            edge_errors = result.get('edge_errors', np.array([]))
            if len(edge_errors) > 0:
                err_mean = np.mean(edge_errors) * 100
                err_max = np.max(edge_errors) * 100
            else:
                err_mean, err_max = 0, 0
            
            print(f"Frame {i:4d}: {mode:8s} | {len(keypoints):2d} keypoints | "
                  f"edge_err: mean={err_mean:5.1f}%, max={err_max:5.1f}% | "
                  f"time: {frame_time*1000:.1f}ms")
        else:
            keypoints = np.empty((0, 3))
            keypoints_2d = np.empty((0, 2))
            edges = []
            
            reason = result.get('reason', 'unknown')
            print(f"Frame {i:4d}: {mode:8s} | FAILED ({reason}) | time: {frame_time*1000:.1f}ms")
        
        # Store result
        all_results.append({
            'frame_idx': i,
            'success': success,
            'mode': mode,
            'keypoints_3d': result.get('keypoints', np.empty((0, 3))),
            'keypoints_2d': keypoints_2d,
            'edges': edges,
            'edge_errors': result.get('edge_errors', np.array([])),
        })
        
        # Update trajectory history
        keypoints_3d = result.get('keypoints', np.empty((0, 3)))
        if success and len(keypoints_3d) > 0:
            traj_history_3d.append(keypoints_3d.copy())
            traj_history_2d.append(keypoints_2d.copy())
        else:
            if len(traj_history_3d) > 0:
                nan_3d = np.full_like(traj_history_3d[-1], np.nan)
                nan_2d = np.full_like(traj_history_2d[-1], np.nan)
            else:
                nan_3d = np.full((tracker.N_KEYPOINTS, 3), np.nan)
                nan_2d = np.full((tracker.N_KEYPOINTS, 2), np.nan)
            traj_history_3d.append(nan_3d)
            traj_history_2d.append(nan_2d)
        
        traj_hist_2d_arr = np.array(traj_history_2d) if traj_history_2d else None
        traj_hist_3d_arr = np.array(traj_history_3d) if traj_history_3d else None
        
        # Create visualization
        viz = create_visualization(
            rgb, foreground_mask, keypoints_2d, edges,
            mode=mode, frame_idx=i,
            traj_history_2d=traj_hist_2d_arr,
            tail_length=tail_length,
            ee_to_corner_mapping=tracker.ee_to_corner_mapping,
            corner_indices=tracker.CORNER_INDICES,
            border_indices=tracker.BORDER_INDICES,
        )
        
        if video_writer is None:
            H_viz, W_viz = viz.shape[:2]
            video_path = str(output_dir / "fabric_tracking_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W_viz, H_viz))
        
        video_writer.write(cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
        
        frame_path = frames_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
        
        # Extract point cloud for 3D viz
        valid = (depth > crop_z_min) & (depth < crop_z_max)
        rows, cols = np.where(valid)
        if len(rows) > 0:
            z = depth[rows, cols].astype(np.float64)
            x = (cols - INTRINSICS[0, 2]) * z / INTRINSICS[0, 0]
            y = (rows - INTRINSICS[1, 2]) * z / INTRINSICS[1, 1]
            colors = rgb[rows, cols]
            
            crop_mask = (x >= crop_x_min) & (x <= crop_x_max) & (y >= crop_y_min) & (y <= crop_y_max)
            x, y, z = x[crop_mask], y[crop_mask], z[crop_mask]
            colors = colors[crop_mask]
            
            full_pc = np.stack([x, y, z], axis=1)
            pc_colors = colors
        else:
            full_pc = np.empty((0, 3))
            pc_colors = np.empty((0, 3))
        
        viz_3d = create_3d_visualization(
            keypoints_3d=keypoints_3d,
            edges=edges if success else [],
            full_pc=full_pc,
            pc_colors=pc_colors,
            frame_idx=i,
            mode=mode,
            pc_downsample=5000,
            fixed_xlim=fixed_xlim,
            fixed_ylim=fixed_ylim,
            fixed_zlim=fixed_zlim,
            traj_history_3d=traj_hist_3d_arr,
            tail_length=tail_length,
            corner_indices=tracker.CORNER_INDICES,
            border_indices=tracker.BORDER_INDICES,
        )
        
        if video_writer_3d is None:
            H_3d, W_3d = viz_3d.shape[:2]
            video_path_3d = str(output_dir / "fabric_tracking_3d.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer_3d = cv2.VideoWriter(video_path_3d, fourcc, fps, (W_3d, H_3d))
        
        video_writer_3d.write(cv2.cvtColor(viz_3d, cv2.COLOR_RGB2BGR))
        
        frame_path_3d = frames_3d_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(frame_path_3d), cv2.cvtColor(viz_3d, cv2.COLOR_RGB2BGR))
        
        del viz, viz_3d, full_pc, pc_colors
        if i % 10 == 0:
            gc.collect()
    
    # Finalize
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved to: {output_dir / 'fabric_tracking_video.mp4'}")
    
    if video_writer_3d is not None:
        video_writer_3d.release()
        print(f"3D video saved to: {output_dir / 'fabric_tracking_3d.mp4'}")
    
    results_path = output_dir / "tracking_results.npy"
    np.save(str(results_path), all_results, allow_pickle=True)
    print(f"Results saved to: {results_path}")
    
    # Statistics
    print(f"\n{'='*70}")
    print("TRACKING STATISTICS")
    print("=" * 70)
    
    n_success = sum(1 for r in all_results if r['success'])
    n_init = sum(1 for r in all_results if r['mode'] == 'init')
    n_track = sum(1 for r in all_results if r['mode'] == 'track')
    n_restart = sum(1 for r in all_results if r['mode'] == 'restart')
    n_skip = sum(1 for r in all_results if r['mode'] == 'skip')
    
    print(f"Total frames: {n_frames}")
    print(f"Successful:   {n_success} ({n_success/n_frames*100:.1f}%)")
    print(f"  Init:       {n_init}")
    print(f"  Track:      {n_track}")
    print(f"  Restart:    {n_restart}")
    print(f"  Skip:       {n_skip}")
    print(f"Average time: {total_time/n_frames:.4f}s per frame")
    
    all_edge_errors = [r['edge_errors'] for r in all_results if len(r['edge_errors']) > 0]
    if all_edge_errors:
        all_errors = np.concatenate(all_edge_errors)
        print(f"\nEdge Length Errors:")
        print(f"  Mean:   {np.mean(all_errors)*100:.2f}%")
        print(f"  Std:    {np.std(all_errors)*100:.2f}%")
        print(f"  Max:    {np.max(all_errors)*100:.2f}%")
    
    print(f"\nOutput directory: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
