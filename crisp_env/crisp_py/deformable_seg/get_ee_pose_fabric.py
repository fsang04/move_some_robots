"""
Get End-Effector 3D Pose from Fabric Masks

Pipeline:
    1. Load precomputed fabric masks
    2. For each frame:
        - Apply depth thresholding (< 1250mm) to validate mask
        - Find top-left corner as EE0, top-right corner as EE1
        - Back-project to 3D using depth
    3. Visualize with trajectory tails
    4. Save 3D EE positions

Usage:
    python get_ee_pose_fabric.py

Author: Auto-generated
Date: 2026-02-23
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


# ============================================================================
# CAMERA INTRINSICS
# ============================================================================

INTRINSICS = np.array([
    [606.1124267578125, 0, 641.7578125],
    [0, 605.8821411132812, 365.6518859863281],
    [0, 0, 1]
], dtype=np.float64)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def find_corner_points(mask: np.ndarray) -> tuple:
    """
    Find top-left and top-right corner points of the mask.
    
    Top-left: minimize (row + col) among foreground pixels
    Top-right: minimize (row - col) among foreground pixels (equivalently, minimize row, maximize col)
    
    Args:
        mask: H × W binary mask
    
    Returns:
        ee0_rc: (row, col) for top-left corner (EE0)
        ee1_rc: (row, col) for top-right corner (EE1)
    """
    # Get all foreground pixel coordinates
    rows, cols = np.where(mask > 0)
    
    if len(rows) == 0:
        return (np.nan, np.nan), (np.nan, np.nan)
    
    # Top-left: minimize (row + col)
    top_left_score = rows + cols
    top_left_idx = np.argmin(top_left_score)
    ee0_rc = (rows[top_left_idx], cols[top_left_idx])
    
    # Top-right: minimize (row - col) which is equivalent to min row, max col
    top_right_score = rows - cols
    top_right_idx = np.argmin(top_right_score)
    ee1_rc = (rows[top_right_idx], cols[top_right_idx])
    
    return ee0_rc, ee1_rc


def pixel_to_3d(pixel_coords: np.ndarray, depth: np.ndarray, 
                intrinsics: np.ndarray = INTRINSICS, max_depth: float = 1250.0) -> np.ndarray:
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
        if np.isnan(row) or np.isnan(col):
            coords_3d.append([np.nan, np.nan, np.nan])
            continue
            
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


def apply_depth_threshold_to_mask(mask: np.ndarray, depth: np.ndarray, 
                                   max_depth: float = 1250.0) -> np.ndarray:
    """
    Apply depth thresholding to mask - only keep pixels with valid depth < max_depth.
    
    Args:
        mask: H × W binary mask
        depth: H × W depth image
        max_depth: Maximum valid depth
    
    Returns:
        filtered_mask: H × W binary mask with depth filtering applied
    """
    depth_valid = (depth > 0) & (depth < max_depth)
    return mask & depth_valid


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualization(
    rgb: np.ndarray,
    mask: np.ndarray,
    ee0_2d: tuple,
    ee1_2d: tuple,
    ee_3d: np.ndarray,
    frame_idx: int,
    traj_history: np.ndarray = None,
    tail_length: int = 60,
) -> np.ndarray:
    """
    Create visualization with EE points and trajectory tails.
    
    Args:
        rgb: H × W × 3 RGB image
        mask: H × W binary mask
        ee0_2d: (row, col) for EE0
        ee1_2d: (row, col) for EE1
        ee_3d: 2 × 3 back-projected 3D positions
        frame_idx: Current frame index
        traj_history: T × 2 × 2 trajectory history (row, col)
        tail_length: Number of frames for trajectory tail
    
    Returns:
        vis: Visualization image
    """
    H, W = rgb.shape[:2]
    
    # Colors
    EE0_COLOR = [255, 0, 0]       # Red for EE0 (top-left)
    EE1_COLOR = [0, 255, 0]       # Green for EE1 (top-right)
    EE0_TAIL_COLOR = [255, 100, 100]  # Light red
    EE1_TAIL_COLOR = [100, 255, 100]  # Light green
    MASK_COLOR = [0, 191, 255]    # Deep sky blue for mask contour
    
    vis = rgb.copy()
    
    # Draw mask contour (simplified - no dimming)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, MASK_COLOR, 2)
    
    # Draw trajectory tails (only last few frames for speed)
    if traj_history is not None and len(traj_history) > 1:
        n_history = len(traj_history)
        
        for ee_idx in range(2):
            tail_color = EE0_TAIL_COLOR if ee_idx == 0 else EE1_TAIL_COLOR
            start_idx = max(0, n_history - tail_length)
            
            for t in range(start_idx, n_history - 1):
                pt1 = traj_history[t, ee_idx]
                pt2 = traj_history[t + 1, ee_idx]
                
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
                color = [int(c * alpha) for c in tail_color]
                cv2.line(vis, (col1, row1), (col2, row2), color, 2)
    
    # Draw EE0 (top-left)
    if not np.isnan(ee0_2d[0]):
        row, col = int(ee0_2d[0]), int(ee0_2d[1])
        if 0 <= row < H and 0 <= col < W:
            cv2.circle(vis, (col, row), 12, EE0_COLOR, -1)
            cv2.circle(vis, (col, row), 14, (255, 255, 255), 2)
            cv2.putText(vis, "EE0", (col + 15, row - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, EE0_COLOR, 2)
    
    # Draw EE1 (top-right)
    if not np.isnan(ee1_2d[0]):
        row, col = int(ee1_2d[0]), int(ee1_2d[1])
        if 0 <= row < H and 0 <= col < W:
            cv2.circle(vis, (col, row), 12, EE1_COLOR, -1)
            cv2.circle(vis, (col, row), 14, (255, 255, 255), 2)
            cv2.putText(vis, "EE1", (col + 15, row - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, EE1_COLOR, 2)
    
    # Add text info
    cv2.putText(vis, f"Frame: {frame_idx}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add 3D positions
    y_offset = 60
    for i, (name, pos_3d) in enumerate(zip(["EE0 (top-left)", "EE1 (top-right)"], ee_3d)):
        if np.any(np.isnan(pos_3d)):
            text = f"{name}: No valid depth"
        else:
            text = f"{name}: ({pos_3d[0]:.1f}, {pos_3d[1]:.1f}, {pos_3d[2]:.1f}) mm"
        color = EE0_COLOR if i == 0 else EE1_COLOR
        cv2.putText(vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 25
    
    return vis


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function to extract EE 3D poses from fabric masks."""
    
    print("=" * 70)
    print("GET END-EFFECTOR 3D POSE FROM FABRIC MASKS")
    print("=" * 70)
    
    # Paths
    data_path = Path("/home/yehengz/deformable_seg/data/full/tracking_fabric2_data.npy")
    masks_dir = Path("/home/yehengz/deformable_seg/data/arm_traj4_fabric/masks")
    output_dir = Path("/home/yehengz/deformable_seg/data/arm_traj4_fabric/ee_pose_output")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tracking data
    print(f"\nLoading tracking data from: {data_path}")
    tracking_data = np.load(str(data_path), allow_pickle=True).item()
    
    # Get sorted frame keys
    frame_keys = sorted(tracking_data.keys())
    n_frames = len(frame_keys)
    print(f"Found {n_frames} frames")
    
    # Get mask files
    mask_files = sorted(masks_dir.glob("mask_frame_*.npy"))
    print(f"Found {len(mask_files)} mask files")
    
    if len(mask_files) != n_frames:
        print(f"Warning: Number of masks ({len(mask_files)}) != number of frames ({n_frames})")
        n_frames = min(len(mask_files), n_frames)
    
    # Storage
    all_ee_3d = []      # List of (2, 3) arrays
    all_ee_2d = []      # List of (2, 2) arrays (row, col)
    n_ee = 2
    
    # Process frames
    print(f"\n{'='*70}")
    print("PROCESSING FRAMES")
    print("=" * 70)
    
    for i in tqdm(range(n_frames)):
        frame_key = frame_keys[i]
        data = tracking_data[frame_key]
        
        color_bgr = data['color']
        depth = data['transformed_depth']
        
        # Load mask
        mask_path = masks_dir / f"mask_frame_{i:04d}.npy"
        mask = np.load(str(mask_path))
        
        # Apply depth thresholding to mask
        mask_filtered = apply_depth_threshold_to_mask(mask, depth, max_depth=1250.0)
        
        # Find corner points
        ee0_rc, ee1_rc = find_corner_points(mask_filtered)
        
        # Store 2D positions
        ee_2d = np.array([ee0_rc, ee1_rc], dtype=np.float64)
        all_ee_2d.append(ee_2d)
        
        # Back-project to 3D
        ee_3d = pixel_to_3d(ee_2d, depth, INTRINSICS, max_depth=1250.0)
        all_ee_3d.append(ee_3d)
        
        # Print progress every 50 frames
        if i % 50 == 0:
            ee_str = " | ".join([
                f"EE{j}: ({ee_3d[j, 0]:.0f}, {ee_3d[j, 1]:.0f}, {ee_3d[j, 2]:.0f})" 
                if not np.any(np.isnan(ee_3d[j])) else f"EE{j}: N/A"
                for j in range(n_ee)
            ])
            print(f"Frame {i:4d}: {ee_str}")
    
    # Convert to arrays
    all_ee_3d = np.array(all_ee_3d)  # (n_frames, 2, 3)
    all_ee_2d = np.array(all_ee_2d)  # (n_frames, 2, 2)
    
    # ========================================================================
    # Create visualization video
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("CREATING VISUALIZATION VIDEO")
    print("=" * 70)
    
    # Pre-load all masks for faster processing
    print("Pre-loading masks...")
    all_masks = []
    for i in range(n_frames):
        mask_path = masks_dir / f"mask_frame_{i:04d}.npy"
        mask = np.load(str(mask_path))
        all_masks.append(mask)
    print(f"Loaded {len(all_masks)} masks")
    
    video_writer = None
    fps = 30
    tail_length = 60
    
    for i in tqdm(range(n_frames)):
        frame_key = frame_keys[i]
        data = tracking_data[frame_key]
        
        color_bgr = data['color']
        color_rgb = color_bgr[:, :, ::-1]  # BGR to RGB
        depth = data['transformed_depth']
        
        # Get pre-loaded mask
        mask = all_masks[i]
        mask_filtered = apply_depth_threshold_to_mask(mask, depth, max_depth=1250.0)
        
        # Get data for this frame
        ee0_2d = tuple(all_ee_2d[i, 0])
        ee1_2d = tuple(all_ee_2d[i, 1])
        ee_3d = all_ee_3d[i]
        
        # Build trajectory history
        traj_history = all_ee_2d[:i+1] if i > 0 else None
        
        # Create visualization
        vis = create_visualization(
            color_rgb, mask_filtered, ee0_2d, ee1_2d, ee_3d,
            frame_idx=i,
            traj_history=traj_history,
            tail_length=tail_length
        )
        
        # Initialize video writer
        if video_writer is None:
            H, W = vis.shape[:2]
            video_path = str(output_dir / "ee_pose_visualization.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
        
        # Write frame
        video_writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved to: {output_dir / 'ee_pose_visualization.mp4'}")
    
    # ========================================================================
    # Save results
    # ========================================================================
    
    results = {
        'ee_3d': all_ee_3d,
        'ee_2d': all_ee_2d,
        'n_frames': n_frames,
        'n_ee': n_ee,
        'ee_names': ['top-left', 'top-right'],
    }
    
    results_path = output_dir / "ee_poses_3d.npy"
    np.save(str(results_path), results, allow_pickle=True)
    print(f"Results saved to: {results_path}")
    
    # ========================================================================
    # Print statistics
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("STATISTICS")
    print("=" * 70)
    
    ee_names = ["EE0 (top-left)", "EE1 (top-right)"]
    for ee_idx in range(n_ee):
        ee_positions = all_ee_3d[:, ee_idx, :]
        valid_mask = ~np.any(np.isnan(ee_positions), axis=1)
        n_valid = np.sum(valid_mask)
        
        print(f"\n{ee_names[ee_idx]}:")
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
