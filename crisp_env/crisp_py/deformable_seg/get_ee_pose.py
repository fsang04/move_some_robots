"""
Get End-Effector 3D Pose from 2D CoTracker Results

Pipeline:
    1. Load CoTracker 2D tracking results (2 end-effectors tracked)
    2. For each frame:
        - Segment to get skeleton mask
        - Identify leaf nodes (end-effector candidates)
        - Match CoTracker 2D points to closest leaf nodes
        - Back-project matched leaf nodes to 3D using depth
    3. Visualize with trajectory tails
    4. Save 3D EE positions

Usage:
    python get_ee_pose.py --traj 1
    python get_ee_pose.py --traj 2
    python get_ee_pose.py --traj 3

Author: Auto-generated
Date: 2026-02-18
"""

import numpy as np
import cv2
from pathlib import Path
import argparse

from wire_tracker import WireTracker


# ============================================================================
# CAMERA INTRINSICS
# ============================================================================

INTRINSICS = np.array([
    [606.1124267578125, 0, 641.7578125],
    [0, 605.8821411132812, 365.6518859863281],
    [0, 0, 1]
], dtype=np.float64)


# ============================================================================
# TRAJECTORY CONFIGURATIONS
# ============================================================================

TRAJECTORY_CONFIGS = {
    1: {
        'arm_data_path': Path('./data/arm_traj1/arm_traj1.npy'),
        'full_data_path': Path('./data/arm_traj1/arm_with_wires_traj1.npy'),
        'cotracker_path': Path('./data/arm_traj1/cotracker-online_results.npz'),
        'output_dir': Path('./data/arm_traj1/ee_pose_output'),
        'precomputed_mask_dir': None,
        'arm_green_frame': 66,
        'full_green_frame': 66,
    },
    2: {
        'arm_data_path': Path('./data/arm_traj2/arm_traj2.npy'),
        'full_data_path': Path('./data/arm_traj2/arm_with_wires_traj2.npy'),
        'cotracker_path': Path('./data/arm_traj2/cotracker-online_results.npz'),
        'output_dir': Path('./data/arm_traj2/ee_pose_output'),
        'precomputed_mask_dir': Path('./data/arm_traj2/masks'),
        'arm_green_frame': 0,
        'full_green_frame': 0,
    },
    3: {
        'arm_data_path': Path('./data/arm_traj3/arm_traj3_contact.npy'),
        'full_data_path': Path('./data/arm_traj3/arm_with_wires_traj3_contact.npy'),
        'cotracker_path': Path('./data/arm_traj3/cotracker-online_results.npz'),
        'output_dir': Path('./data/arm_traj3/ee_pose_output'),
        'precomputed_mask_dir': None,
        'arm_green_frame': 84,
        'full_green_frame': 100,
    },
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def snap_to_foreground(pixel_coords: np.ndarray, foreground_mask: np.ndarray, 
                       max_search_radius: int = 100) -> np.ndarray:
    """
    Snap 2D pixel coordinates to the closest foreground pixel.
    
    Args:
        pixel_coords: N × 2 array of (row, col)
        foreground_mask: H × W binary mask (1 = foreground)
        max_search_radius: Maximum search radius in pixels
    
    Returns:
        snapped_coords: N × 2 array of (row, col) snapped to foreground
    """
    if len(pixel_coords) == 0:
        return np.empty((0, 2), dtype=np.float64)
    
    H, W = foreground_mask.shape
    
    # Get all foreground pixel coordinates
    fg_rows, fg_cols = np.where(foreground_mask > 0)
    if len(fg_rows) == 0:
        return np.full_like(pixel_coords, np.nan, dtype=np.float64)
    
    fg_coords = np.stack([fg_rows, fg_cols], axis=1)  # (N_fg, 2)
    
    snapped_coords = []
    for row, col in pixel_coords:
        # Compute distances to all foreground pixels
        distances = np.sqrt((fg_coords[:, 0] - row)**2 + (fg_coords[:, 1] - col)**2)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        
        if min_dist <= max_search_radius:
            snapped_coords.append(fg_coords[min_idx])
        else:
            snapped_coords.append([np.nan, np.nan])
    
    return np.array(snapped_coords, dtype=np.float64)


def pixel_to_3d(pixel_coords: np.ndarray, depth: np.ndarray, 
                intrinsics: np.ndarray = INTRINSICS, max_depth: float = 1000.0) -> np.ndarray:
    """
    Back-project 2D pixel coordinates to 3D.
    
    Args:
        pixel_coords: N × 2 array of (row, col)
        depth: H × W depth image
        intrinsics: 3x3 camera intrinsic matrix
        max_depth: Maximum valid depth
    
    Returns:
        coords_3d: N × 3 array of (x, y, z)
    """
    if len(pixel_coords) == 0:
        return np.empty((0, 3), dtype=np.float64)
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    coords_3d = []
    H, W = depth.shape
    
    for row, col in pixel_coords:
        row, col = int(row), int(col)
        if 0 <= row < H and 0 <= col < W:
            z = depth[row, col]
            if z > 0 and z < max_depth:
                x = (col - cx) * z / fx
                y = (row - cy) * z / fy
                coords_3d.append([x, y, z])
            else:
                coords_3d.append([np.nan, np.nan, np.nan])
        else:
            coords_3d.append([np.nan, np.nan, np.nan])
    
    return np.array(coords_3d, dtype=np.float64)


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualization(
    rgb: np.ndarray,
    skeleton_mask: np.ndarray,
    cotracker_2d: np.ndarray,
    matched_leaf_2d: np.ndarray,
    ee_3d: np.ndarray,
    frame_idx: int,
    traj_history_cotracker: np.ndarray = None,
    traj_history_matched: np.ndarray = None,
    tail_length: int = 60,
) -> np.ndarray:
    """
    Create visualization with CoTracker points, matched leaf nodes, and trajectory tails.
    
    Args:
        rgb: H × W × 3 RGB image
        skeleton_mask: H × W skeleton mask
        cotracker_2d: N_ee × 2 CoTracker points (x, y) = (col, row)
        matched_leaf_2d: N_ee × 2 matched leaf node positions (row, col)
        ee_3d: N_ee × 3 back-projected 3D positions
        frame_idx: Current frame index
        traj_history_cotracker: T × N_ee × 2 trajectory history for CoTracker (col, row)
        traj_history_matched: T × N_ee × 2 trajectory history for matched leaves (row, col)
        tail_length: Number of frames for trajectory tail
    
    Returns:
        vis: Visualization image
    """
    H, W = rgb.shape[:2]
    
    # Colors
    SKELETON_COLOR = [0, 191, 255]      # Deep sky blue
    COTRACKER_COLOR = [255, 0, 0]       # Red for CoTracker points
    MATCHED_COLOR = [0, 255, 0]         # Green for matched leaf nodes
    COTRACKER_TAIL_COLOR = [255, 100, 100]  # Light red
    MATCHED_TAIL_COLOR = [100, 255, 100]    # Light green
    
    vis = rgb.copy()
    
    # Draw skeleton
    skeleton_thick = cv2.dilate(skeleton_mask, np.ones((3, 3), np.uint8), iterations=1)
    vis[skeleton_thick > 0] = SKELETON_COLOR
    
    # Draw CoTracker trajectory tails
    if traj_history_cotracker is not None and len(traj_history_cotracker) > 1:
        n_history = len(traj_history_cotracker)
        n_ee = traj_history_cotracker.shape[1]
        
        for ee_idx in range(n_ee):
            start_idx = max(0, n_history - tail_length)
            for t in range(start_idx, n_history - 1):
                pt1 = traj_history_cotracker[t, ee_idx]
                pt2 = traj_history_cotracker[t + 1, ee_idx]
                
                if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
                    continue
                
                col1, row1 = int(pt1[0]), int(pt1[1])
                col2, row2 = int(pt2[0]), int(pt2[1])
                
                if not (0 <= row1 < H and 0 <= col1 < W):
                    continue
                if not (0 <= row2 < H and 0 <= col2 < W):
                    continue
                
                age = n_history - 1 - t
                alpha = max(0.2, 1.0 - age / tail_length)
                color = [int(c * alpha) for c in COTRACKER_TAIL_COLOR]
                cv2.line(vis, (col1, row1), (col2, row2), color, 2)
    
    # Draw matched leaf trajectory tails
    if traj_history_matched is not None and len(traj_history_matched) > 1:
        n_history = len(traj_history_matched)
        n_ee = traj_history_matched.shape[1]
        
        for ee_idx in range(n_ee):
            start_idx = max(0, n_history - tail_length)
            for t in range(start_idx, n_history - 1):
                pt1 = traj_history_matched[t, ee_idx]
                pt2 = traj_history_matched[t + 1, ee_idx]
                
                if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
                    continue
                
                row1, col1 = int(pt1[0]), int(pt1[1])
                row2, col2 = int(pt2[0]), int(pt2[1])
                
                if not (0 <= row1 < H and 0 <= col1 < W):
                    continue
                if not (0 <= row2 < H and 0 <= col2 < W):
                    continue
                
                age = n_history - 1 - t
                alpha = max(0.2, 1.0 - age / tail_length)
                color = [int(c * alpha) for c in MATCHED_TAIL_COLOR]
                cv2.line(vis, (col1, row1), (col2, row2), color, 2)
    
    # Draw CoTracker points (red circles)
    for i, (x, y) in enumerate(cotracker_2d):
        col, row = int(x), int(y)
        if 0 <= row < H and 0 <= col < W:
            cv2.circle(vis, (col, row), 10, COTRACKER_COLOR, 2)
            cv2.putText(vis, f"CT{i}", (col + 12, row - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COTRACKER_COLOR, 1)
    
    # Draw matched leaf nodes (green circles) and connection lines
    for i, (row, col) in enumerate(matched_leaf_2d):
        if np.isnan(row) or np.isnan(col):
            continue
        row, col = int(row), int(col)
        if 0 <= row < H and 0 <= col < W:
            cv2.circle(vis, (col, row), 8, MATCHED_COLOR, -1)
            
            # Draw line connecting CoTracker to matched leaf
            ct_col, ct_row = int(cotracker_2d[i, 0]), int(cotracker_2d[i, 1])
            cv2.line(vis, (ct_col, ct_row), (col, row), [255, 255, 0], 1)
    
    # Add text info
    cv2.putText(vis, f"Frame: {frame_idx}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add 3D positions
    y_offset = 60
    for i, pos_3d in enumerate(ee_3d):
        if np.any(np.isnan(pos_3d)):
            text = f"EE{i}: No match"
        else:
            text = f"EE{i}: ({pos_3d[0]:.1f}, {pos_3d[1]:.1f}, {pos_3d[2]:.1f}) mm"
        cv2.putText(vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
    
    # Add legend
    y_offset += 10
    cv2.circle(vis, (20, y_offset), 6, COTRACKER_COLOR, 2)
    cv2.putText(vis, "CoTracker 2D", (35, y_offset + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += 25
    cv2.circle(vis, (20, y_offset), 6, MATCHED_COLOR, -1)
    cv2.putText(vis, "Matched Leaf (3D)", (35, y_offset + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function to extract EE 3D poses."""
    
    # ========================================================================
    # Parse Arguments
    # ========================================================================
    
    parser = argparse.ArgumentParser(description='Get End-Effector 3D Pose from CoTracker 2D Results')
    parser.add_argument('--traj', type=int, required=True, choices=[1, 2, 3],
                        help='Trajectory number (1, 2, or 3)')
    args = parser.parse_args()
    
    config = TRAJECTORY_CONFIGS[args.traj]
    
    # ========================================================================
    # Load Data
    # ========================================================================
    
    print("=" * 70)
    print("GET END-EFFECTOR 3D POSE FROM COTRACKER 2D TRACKING")
    print("=" * 70)
    print(f"\nTrajectory: {args.traj}")
    
    # Load CoTracker results
    print(f"\nLoading CoTracker results from: {config['cotracker_path']}")
    cotracker_data = np.load(str(config['cotracker_path']))
    pred_tracks = cotracker_data['pred_tracks']  # Shape: (1, n_frames, n_points, 2)
    print(f"  pred_tracks shape (raw): {pred_tracks.shape}")
    
    # Remove batch dimension: (n_frames, n_points, 2)
    pred_tracks = pred_tracks[0]
    
    # Crop CoTracker data to start from full_green_frame (synchronize with arm data)
    full_green_frame = config['full_green_frame']
    pred_tracks = pred_tracks[full_green_frame:]
    print(f"  Cropped from frame {full_green_frame}: shape = {pred_tracks.shape}")
    
    n_cotracker_frames, n_ee, _ = pred_tracks.shape
    print(f"  Number of frames (after crop): {n_cotracker_frames}")
    print(f"  Number of end-effectors: {n_ee}")
    
    # Load arm-only data
    print(f"\nLoading arm-only data from: {config['arm_data_path']}")
    arm_only_data = np.load(str(config['arm_data_path']), allow_pickle=True).item()
    
    # Load full scene data
    print(f"Loading full scene data from: {config['full_data_path']}")
    full_scene_data = np.load(str(config['full_data_path']), allow_pickle=True).item()
    
    # Synchronize sequences
    arm_frame_keys = sorted(arm_only_data.keys())[config['arm_green_frame']:]
    full_frame_keys = sorted(full_scene_data.keys())[config['full_green_frame']:]
    
    n_frames = min(len(arm_frame_keys), len(full_frame_keys), n_cotracker_frames)
    arm_frame_keys = arm_frame_keys[:n_frames]
    full_frame_keys = full_frame_keys[:n_frames]
    
    print(f"\nSynchronized sequences:")
    print(f"  Total frames: {n_frames}")
    
    # Create output directory
    output_dir = config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Initialize Tracker (for segmentation and node detection only)
    # ========================================================================
    
    print("\nInitializing WireTracker for segmentation...")
    tracker = WireTracker(
        intrinsics=INTRINSICS,
        n_keypoints=21,
        target_branch_nodes=2,
        target_leaf_nodes=4,
        bg_threshold=80.0,
        max_depth=1000.0,
        top_k_components=5,
        arm_dilation_pixels=5,
    )
    
    # ========================================================================
    # Process Frames
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("PROCESSING FRAMES")
    print("=" * 70)
    
    # Storage
    all_ee_3d = []  # List of (n_ee, 3) arrays
    all_matched_2d = []  # List of (n_ee, 2) arrays (row, col)
    
    # Trajectory history for visualization
    traj_history_cotracker = []  # List of (n_ee, 2) arrays (col, row)
    traj_history_matched = []    # List of (n_ee, 2) arrays (row, col)
    tail_length = 60
    
    # Video writer
    video_writer = None
    fps = 30
    
    for i in range(n_frames):
        arm_frame_key = arm_frame_keys[i]
        full_frame_key = full_frame_keys[i]
        
        # Load data
        arm_data = arm_only_data[arm_frame_key]
        arm_depth = arm_data['transformed_depth'].copy()
        
        full_data = full_scene_data[full_frame_key]
        full_rgb = full_data['color'][:, :, ::-1]  # BGR to RGB
        full_depth = full_data['transformed_depth'].copy()
        
        # Load precomputed arm mask if available
        precomputed_arm_mask = None
        if config['precomputed_mask_dir'] is not None:
            mask_path = config['precomputed_mask_dir'] / f"mask_frame_{i:04d}.npy"
            if mask_path.exists():
                precomputed_arm_mask = np.load(str(mask_path))
        
        # Get CoTracker 2D points for this frame (x, y) = (col, row)
        cotracker_2d = pred_tracks[i]  # (n_ee, 2)
        
        # Segment to get skeleton and foreground mask
        seg_result = tracker.segment(
            full_depth, arm_depth, 
            n_components=tracker.top_k_components,
            precomputed_arm_mask=precomputed_arm_mask
        )
        skeleton_mask = seg_result['skeleton_mask']
        foreground_mask = seg_result['foreground_mask']
        
        # Convert CoTracker (x, y) = (col, row) to (row, col) for snapping
        cotracker_rc = cotracker_2d[:, ::-1]  # (col, row) -> (row, col)
        
        # Snap CoTracker 2D points to closest foreground pixel
        snapped_2d = snap_to_foreground(cotracker_rc, foreground_mask, max_search_radius=100)
        
        # Back-project snapped 2D points to 3D
        ee_3d = pixel_to_3d(snapped_2d, full_depth, INTRINSICS)
        
        # Store results (snapped_2d is in row, col format)
        all_ee_3d.append(ee_3d)
        all_matched_2d.append(snapped_2d)
        
        # Print progress
        ee_str = " | ".join([
            f"EE{j}: ({ee_3d[j, 0]:.0f}, {ee_3d[j, 1]:.0f}, {ee_3d[j, 2]:.0f})" 
            if not np.any(np.isnan(ee_3d[j])) else f"EE{j}: N/A"
            for j in range(n_ee)
        ])
        print(f"Frame {i:4d}: {ee_str}")
    
    # ========================================================================
    # Convert to arrays
    # ========================================================================
    
    all_ee_3d = np.array(all_ee_3d)  # (n_frames, n_ee, 3)
    all_matched_2d = np.array(all_matched_2d)  # (n_frames, n_ee, 2)
    
    # ========================================================================
    # Fix known outliers by interpolation
    # ========================================================================
    
    # Traj1: EE1 at frame 261 is an outlier - interpolate from neighbors
    if args.traj == 1 and n_frames > 262:
        outlier_frame = 261
        ee_idx = 1
        print(f"\nFixing outlier: EE{ee_idx} at frame {outlier_frame}")
        print(f"  3D Before: {all_ee_3d[outlier_frame, ee_idx]}")
        print(f"  2D Before: {all_matched_2d[outlier_frame, ee_idx]}")
        
        # Interpolate 3D from frame 260 and 262
        prev_pos_3d = all_ee_3d[outlier_frame - 1, ee_idx]
        next_pos_3d = all_ee_3d[outlier_frame + 1, ee_idx]
        
        # Interpolate 2D from frame 260 and 262
        prev_pos_2d = all_matched_2d[outlier_frame - 1, ee_idx]
        next_pos_2d = all_matched_2d[outlier_frame + 1, ee_idx]
        
        if not np.any(np.isnan(prev_pos_3d)) and not np.any(np.isnan(next_pos_3d)):
            all_ee_3d[outlier_frame, ee_idx] = (prev_pos_3d + next_pos_3d) / 2.0
            print(f"  3D After:  {all_ee_3d[outlier_frame, ee_idx]}")
        else:
            print(f"  Warning: Cannot interpolate 3D, neighbors have NaN")
        
        if not np.any(np.isnan(prev_pos_2d)) and not np.any(np.isnan(next_pos_2d)):
            all_matched_2d[outlier_frame, ee_idx] = (prev_pos_2d + next_pos_2d) / 2.0
            print(f"  2D After:  {all_matched_2d[outlier_frame, ee_idx]}")
        else:
            print(f"  Warning: Cannot interpolate 2D, neighbors have NaN")
    
    # Traj3: EE1 at frames 115, 123, 216, and 233 are outliers - interpolate from neighbors
    if args.traj == 3:
        outlier_frames = [115, 123, 216, 233]
        ee_idx = 1
        for outlier_frame in outlier_frames:
            if n_frames > outlier_frame + 1:
                print(f"\nFixing outlier: EE{ee_idx} at frame {outlier_frame}")
                print(f"  3D Before: {all_ee_3d[outlier_frame, ee_idx]}")
                print(f"  2D Before: {all_matched_2d[outlier_frame, ee_idx]}")
                
                # Interpolate 3D from neighboring frames
                prev_pos_3d = all_ee_3d[outlier_frame - 1, ee_idx]
                next_pos_3d = all_ee_3d[outlier_frame + 1, ee_idx]
                
                # Interpolate 2D from neighboring frames
                prev_pos_2d = all_matched_2d[outlier_frame - 1, ee_idx]
                next_pos_2d = all_matched_2d[outlier_frame + 1, ee_idx]
                
                if not np.any(np.isnan(prev_pos_3d)) and not np.any(np.isnan(next_pos_3d)):
                    all_ee_3d[outlier_frame, ee_idx] = (prev_pos_3d + next_pos_3d) / 2.0
                    print(f"  3D After:  {all_ee_3d[outlier_frame, ee_idx]}")
                else:
                    print(f"  Warning: Cannot interpolate 3D, neighbors have NaN")
                
                if not np.any(np.isnan(prev_pos_2d)) and not np.any(np.isnan(next_pos_2d)):
                    all_matched_2d[outlier_frame, ee_idx] = (prev_pos_2d + next_pos_2d) / 2.0
                    print(f"  2D After:  {all_matched_2d[outlier_frame, ee_idx]}")
                else:
                    print(f"  Warning: Cannot interpolate 2D, neighbors have NaN")
    
    # ========================================================================
    # Create visualization video (after correction)
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("CREATING VISUALIZATION VIDEO (with corrections)")
    print("=" * 70)
    
    video_writer = None
    fps = 30
    tail_length = 60
    
    # Create frames directory
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving individual frames to: {frames_dir}")
    
    for i in range(n_frames):
        arm_frame_key = arm_frame_keys[i]
        full_frame_key = full_frame_keys[i]
        
        # Load data for visualization
        arm_data = arm_only_data[arm_frame_key]
        arm_depth = arm_data['transformed_depth'].copy()
        
        full_data = full_scene_data[full_frame_key]
        full_rgb = full_data['color'][:, :, ::-1]  # BGR to RGB
        full_depth = full_data['transformed_depth'].copy()
        
        # Load precomputed arm mask if available
        precomputed_arm_mask = None
        if config['precomputed_mask_dir'] is not None:
            mask_path = config['precomputed_mask_dir'] / f"mask_frame_{i:04d}.npy"
            if mask_path.exists():
                precomputed_arm_mask = np.load(str(mask_path))
        
        # Segment to get skeleton mask for visualization
        seg_result = tracker.segment(
            full_depth, arm_depth, 
            n_components=tracker.top_k_components,
            precomputed_arm_mask=precomputed_arm_mask
        )
        skeleton_mask = seg_result['skeleton_mask']
        
        # Get data for this frame
        cotracker_2d = pred_tracks[i]
        snapped_2d = all_matched_2d[i]
        ee_3d = all_ee_3d[i]  # This now includes the corrected values
        
        # Build trajectory history up to this frame
        traj_hist_ct_arr = pred_tracks[:i+1] if i > 0 else None
        traj_hist_matched_arr = all_matched_2d[:i+1] if i > 0 else None
        
        # Create visualization
        vis = create_visualization(
            full_rgb, skeleton_mask, cotracker_2d, snapped_2d, ee_3d,
            frame_idx=i,
            traj_history_cotracker=traj_hist_ct_arr,
            traj_history_matched=traj_hist_matched_arr,
            tail_length=tail_length
        )
        
        # # Save individual frame as image
        # frame_path = frames_dir / f"frame_{i:04d}.png"
        # cv2.imwrite(str(frame_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        # Initialize video writer
        if video_writer is None:
            H, W = vis.shape[:2]
            video_path = str(output_dir / "ee_pose_visualization.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
        
        # Write frame
        video_writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        if i % 50 == 0:
            print(f"  Writing frame {i}/{n_frames}")
    
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved to: {output_dir / 'ee_pose_visualization.mp4'}")
    
    # Save results
    results = {
        'ee_3d': all_ee_3d,
        'matched_2d': all_matched_2d,
        'cotracker_2d': pred_tracks[:n_frames],
        'n_frames': n_frames,
        'n_ee': n_ee,
        'trajectory': args.traj,
    }
    
    results_path = output_dir / "ee_poses_3d.npy"
    np.save(str(results_path), results, allow_pickle=True)
    print(f"Results saved to: {results_path}")
    
    # Print statistics
    print(f"\n{'='*70}")
    print("STATISTICS")
    print("=" * 70)
    
    for ee_idx in range(n_ee):
        ee_positions = all_ee_3d[:, ee_idx, :]
        valid_mask = ~np.any(np.isnan(ee_positions), axis=1)
        n_valid = np.sum(valid_mask)
        
        print(f"\nEE{ee_idx}:")
        print(f"  Valid frames: {n_valid}/{n_frames} ({n_valid/n_frames*100:.1f}%)")
        
        if n_valid > 0:
            valid_positions = ee_positions[valid_mask]
            print(f"  X range: [{valid_positions[:, 0].min():.1f}, {valid_positions[:, 0].max():.1f}] mm")
            print(f"  Y range: [{valid_positions[:, 1].min():.1f}, {valid_positions[:, 1].max():.1f}] mm")
            print(f"  Z range: [{valid_positions[:, 2].min():.1f}, {valid_positions[:, 2].max():.1f}] mm")
    
    print(f"\nOutput directory: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
