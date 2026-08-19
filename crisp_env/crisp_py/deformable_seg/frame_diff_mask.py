"""
Frame Difference Mask Generator

Loads arm_with_wires_traj1.npy and computes binary masks based on
3D point cloud differences between consecutive frames.

Outputs:
- Binary mask visualization
- Mask overlay on RGB (side-by-side)

Usage:
    python frame_diff_mask.py
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


def depth_to_point_cloud(depth: np.ndarray, intrinsics: np.ndarray, max_depth: float = 1000.0) -> np.ndarray:
    """
    Convert depth image to 3D point cloud.
    
    Args:
        depth: H x W depth image (mm)
        intrinsics: 3x3 camera intrinsic matrix
        max_depth: Maximum valid depth (mm)
    
    Returns:
        pc: H x W x 3 point cloud (x, y, z) in mm
    """
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Create pixel coordinate grids
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)
    
    # Back-project to 3D
    z = depth.astype(np.float64)
    z_safe = np.where(z > 0, z, 1.0)  # Avoid division by zero
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack into point cloud
    pc = np.stack([x, y, z], axis=-1)
    
    # Invalidate points with zero or excessive depth
    invalid = (depth <= 0) | (depth > max_depth)
    pc[invalid] = 0
    
    return pc


def compute_frame_diff_mask(
    depth_curr: np.ndarray,
    depth_prev: np.ndarray,
    intrinsics: np.ndarray,
    threshold_mm: float = 10.0,
    max_depth: float = 1000.0,
) -> dict:
    """
    Compute masks based on 3D point cloud difference between frames.
    
    Args:
        depth_curr: H x W current depth image (mm)
        depth_prev: H x W previous depth image (mm)
        intrinsics: 3x3 camera intrinsic matrix
        threshold_mm: Distance threshold in mm for motion detection
        max_depth: Maximum valid depth (mm)
    
    Returns:
        dict with:
            'mask': H x W binary mask (1 = changed, 0 = unchanged)
            'positive_mask': H x W mask for positive depth change (moved closer to camera)
            'negative_mask': H x W mask for negative depth change (moved away from camera)
    """
    # Convert to point clouds
    pc_curr = depth_to_point_cloud(depth_curr, intrinsics, max_depth)
    pc_prev = depth_to_point_cloud(depth_prev, intrinsics, max_depth)
    
    # Compute per-pixel 3D distance
    diff = pc_curr - pc_prev
    dist = np.linalg.norm(diff, axis=-1)
    
    # Compute depth difference (z-axis): positive = moved closer, negative = moved away
    # Note: smaller depth = closer to camera, so negative diff means moved closer
    depth_diff = depth_curr.astype(np.float64) - depth_prev.astype(np.float64)
    
    # Create mask for valid pixels (both frames have valid depth)
    valid_curr = (depth_curr > 0) & (depth_curr < max_depth)
    valid_prev = (depth_prev > 0) & (depth_prev < max_depth)
    valid_both = valid_curr & valid_prev
    
    # Threshold to get motion mask
    changed = (dist > threshold_mm) & valid_both
    
    # Positive: depth decreased (moved closer to camera)
    positive_mask = changed & (depth_diff < -threshold_mm)
    
    # Negative: depth increased (moved away from camera)
    negative_mask = changed & (depth_diff > threshold_mm)
    
    # Also include pixels that appeared/disappeared
    appeared = valid_curr & ~valid_prev
    disappeared = ~valid_curr & valid_prev
    
    # Appeared = positive (something came into view), Disappeared = negative
    positive_mask = positive_mask | appeared
    negative_mask = negative_mask | disappeared
    
    # Combined mask
    mask = positive_mask | negative_mask
    
    return {
        'mask': mask.astype(np.uint8),
        'positive_mask': positive_mask.astype(np.uint8),
        'negative_mask': negative_mask.astype(np.uint8),
    }


def create_visualization(rgb: np.ndarray, positive_mask: np.ndarray, negative_mask: np.ndarray, frame_idx: int) -> np.ndarray:
    """
    Create side-by-side visualization: colored mask | overlay
    
    Colors:
        - Red: positive (moved closer to camera)
        - White: negative (moved away from camera)  
        - Black: background (no change)
    
    Args:
        rgb: H x W x 3 RGB image
        positive_mask: H x W binary mask for positive depth change
        negative_mask: H x W binary mask for negative depth change
        frame_idx: Frame index for title
    
    Returns:
        vis: H x (2*W) x 3 visualization (colored mask | overlay)
    """
    H, W = positive_mask.shape
    
    # Left: Colored mask (red=positive, white=negative, black=background)
    mask_vis = np.zeros((H, W, 3), dtype=np.uint8)
    mask_vis[positive_mask > 0] = [255, 0, 0]    # Red for positive (closer)
    mask_vis[negative_mask > 0] = [255, 255, 255]  # White for negative (away)
    
    # Add text and legend
    cv2.putText(mask_vis, f"Frame {frame_idx} Diff Mask", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(mask_vis, "Red: closer", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(mask_vis, "White: away", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Count pixels
    n_pos = np.sum(positive_mask)
    n_neg = np.sum(negative_mask)
    cv2.putText(mask_vis, f"Pos: {n_pos}, Neg: {n_neg}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Right: Overlay (colored mask blended with RGB)
    overlay = rgb.copy()
    
    # Red overlay for positive
    red_color = np.array([255, 0, 0], dtype=np.uint8)
    overlay[positive_mask > 0] = (0.4 * rgb[positive_mask > 0] + 0.6 * red_color).astype(np.uint8)
    
    # Cyan overlay for negative (white would be hard to see on overlay)
    cyan_color = np.array([0, 255, 255], dtype=np.uint8)
    overlay[negative_mask > 0] = (0.4 * rgb[negative_mask > 0] + 0.6 * cyan_color).astype(np.uint8)
    
    cv2.putText(overlay, f"Frame {frame_idx} Overlay", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(overlay, "Red: closer, Cyan: away", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Concatenate side by side
    vis = np.concatenate([mask_vis, overlay], axis=1)
    
    return vis


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Frame Difference Mask Generator")
    parser.add_argument('--frame_gap', type=int, default=1,
                        help='Frame gap for comparison (1=consecutive, 5=every 5 frames, etc.)')
    parser.add_argument('--threshold', type=float, default=10.0,
                        help='3D distance threshold in mm (default: 10.0)')
    parser.add_argument('--data_path', type=str, 
                        default="/home/yehengz/deformable_seg/data/arm_traj1/arm_with_wires_traj1.npy",
                        help='Path to input .npy file')
    parser.add_argument('--output_dir', type=str,
                        default="/home/yehengz/deformable_seg/data/arm_traj1/frame_diff_mask",
                        help='Output directory')
    args = parser.parse_args()
    
    # Camera intrinsics (same as wire_tracker)
    intrinsics = np.array([
        [606.1124267578125, 0, 641.7578125],
        [0, 605.8821411132812, 365.6518859863281],
        [0, 0, 1]
    ])
    
    # Paths
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parameters
    threshold_mm = args.threshold
    max_depth = 1000.0   # Max valid depth in mm
    frame_gap = args.frame_gap
    
    print(f"Loading data from: {data_path}")
    data = np.load(str(data_path), allow_pickle=True).item()
    
    frame_keys = sorted(data.keys())
    n_frames = len(frame_keys)
    print(f"Total frames: {n_frames}")
    print(f"Frame gap: {frame_gap}")
    print(f"Threshold: {threshold_mm} mm")
    print(f"Output dir: {output_dir}")
    
    # Video writer
    video_path = output_dir / f"frame_diff_gap{frame_gap}_video.mp4"
    video_writer = None
    fps = 30
    
    # Process frames
    prev_depth = None
    prev_frame_idx = None
    
    for i, frame_key in enumerate(frame_keys):
        frame_data = data[frame_key]
        
        # Get depth and RGB
        depth = frame_data['transformed_depth'].copy()
        rgb = frame_data['color'][:, :, ::-1].copy()  # BGR to RGB
        
        if prev_depth is None:
            # First frame - no previous to compare
            prev_depth = depth.copy()
            prev_frame_idx = i
            print(f"Frame {i:4d}: First frame, skipping")
            continue
        
        # Skip frames based on frame_gap
        if (i - prev_frame_idx) < frame_gap:
            continue
        
        # Compute difference mask
        mask_result = compute_frame_diff_mask(
            depth, prev_depth, intrinsics,
            threshold_mm=threshold_mm,
            max_depth=max_depth
        )
        
        mask = mask_result['mask']
        positive_mask = mask_result['positive_mask']
        negative_mask = mask_result['negative_mask']
        
        # Count changed pixels
        n_changed = np.sum(mask)
        n_pos = np.sum(positive_mask)
        n_neg = np.sum(negative_mask)
        pct_changed = 100.0 * n_changed / mask.size
        
        print(f"Frame {i:4d} (vs {prev_frame_idx:4d}): {n_changed:6d} pixels ({pct_changed:.2f}%) | Pos: {n_pos:6d}, Neg: {n_neg:6d}")
        
        # Create visualization
        vis = create_visualization(rgb, positive_mask, negative_mask, i)
        
        # Initialize video writer
        if video_writer is None:
            H_vid, W_vid = vis.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (W_vid, H_vid))
        
        # Write frame
        video_writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        # Save individual frames (every 10th frame to save space)
        if i % 10 == 0:
            # Save colored mask (red=positive, white=negative, black=background)
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            colored_mask[positive_mask > 0] = [255, 0, 0]  # Red (BGR: Blue channel)
            colored_mask[negative_mask > 0] = [255, 255, 255]  # White
            mask_path = output_dir / f"mask_{i:04d}.png"
            cv2.imwrite(str(mask_path), colored_mask[:, :, ::-1])  # Convert RGB to BGR for saving
            
            # Save visualization
            vis_path = output_dir / f"vis_{i:04d}.png"
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        # Update previous depth
        prev_depth = depth.copy()
        prev_frame_idx = i
    
    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved to: {video_path}")
    
    print(f"\nDone! Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
