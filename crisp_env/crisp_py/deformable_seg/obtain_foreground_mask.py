#!/usr/bin/env python3
"""
Script to obtain foreground masks by:
1. Removing robot arm (using robot arm masks)
2. Depth thresholding (depth > 0 and depth < 2000mm)

Outputs:
- fg_mask.npz: foreground masks
- fg_seg_overlay.mp4: side-by-side binary mask and mask overlay on RGB
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def load_data(chunk_path: Path):
    """Load RGBD data and robot arm masks."""
    rgbd_path = chunk_path / "rgbd.npz"
    masks_path = chunk_path / "masks" / "masks.npz"
    
    print(f"Loading RGBD from {rgbd_path}")
    rgbd = np.load(rgbd_path)
    color = rgbd["color"]  # (N, H, W, 3) BGR
    depth = rgbd["depth"]  # (N, H, W) uint16, mm
    
    print(f"Loading robot arm masks from {masks_path}")
    masks_data = np.load(masks_path)
    robot_arm_mask = masks_data["masks"]  # (N, H, W) binary, 1 = robot arm
    
    print(f"  Color shape: {color.shape}")
    print(f"  Depth shape: {depth.shape}, range: [{depth.min()}, {depth.max()}]")
    print(f"  Robot arm mask shape: {robot_arm_mask.shape}")
    
    return color, depth, robot_arm_mask


def compute_foreground_mask(depth: np.ndarray, robot_arm_mask: np.ndarray, 
                            min_depth: float = 0, max_depth: float = 2000):
    """
    Compute foreground mask by:
    1. Removing robot arm (robot_arm_mask == 1 means background)
    2. Depth thresholding: depth > min_depth AND depth < max_depth
    
    Args:
        depth: (N, H, W) depth in mm
        robot_arm_mask: (N, H, W) binary, 1 = robot arm (background)
        min_depth: minimum depth threshold (exclusive)
        max_depth: maximum depth threshold (exclusive)
    
    Returns:
        foreground_mask: (N, H, W) binary, 1 = foreground
    """
    n_frames, h, w = depth.shape
    foreground_mask = np.zeros_like(depth, dtype=np.uint8)
    
    # Border masks: top 20% rows and left 25% columns are background
    top_border = int(h * 0.2)
    left_border = int(w * 0.25)
    
    print(f"Computing foreground masks with depth in ({min_depth}, {max_depth})mm...")
    print(f"  Border exclusion: top {top_border} rows, left {left_border} columns")
    
    # Dilation kernel for robot arm mask
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    
    for i in tqdm(range(n_frames)):
        # Dilate robot arm mask by 2 pixels to remove edge artifacts
        dilated_robot_mask = cv2.dilate(robot_arm_mask[i], dilate_kernel, iterations=1)
        
        # Not robot arm
        not_robot = dilated_robot_mask == 0
        
        # Depth thresholding
        valid_depth = (depth[i] > min_depth) & (depth[i] < max_depth)
        
        # Foreground = not robot arm AND valid depth
        foreground_mask[i] = (not_robot & valid_depth).astype(np.uint8)
        
        # Exclude top 20% rows and left 20% columns
        foreground_mask[i, :top_border, :] = 0  # top 20% rows
        foreground_mask[i, :, :left_border] = 0  # left 20% columns
        
        # Keep only the largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground_mask[i], connectivity=8
        )
        if num_labels > 1:  # label 0 is background
            # Find largest component (excluding background at index 0)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            foreground_mask[i] = (labels == largest_label).astype(np.uint8)
        
        # Smooth the boundary with morphological opening (removes small protrusions/peaks)
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        foreground_mask[i] = cv2.morphologyEx(foreground_mask[i], cv2.MORPH_OPEN, smooth_kernel)
        # Then closing to fill small holes
        foreground_mask[i] = cv2.morphologyEx(foreground_mask[i], cv2.MORPH_CLOSE, smooth_kernel)
        
        # Shrink foreground by 1 pixel to remove edge noise/outliers
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        foreground_mask[i] = cv2.erode(foreground_mask[i], erode_kernel, iterations=1)
    
    return foreground_mask


def create_visualization_video(color: np.ndarray, foreground_mask: np.ndarray, 
                               output_path: Path, fps: int = 30):
    """
    Create side-by-side video: binary mask | mask overlay on RGB
    
    Args:
        color: (N, H, W, 3) BGR images
        foreground_mask: (N, H, W) binary mask
        output_path: path to save video
        fps: frames per second
    """
    n_frames, h, w = foreground_mask.shape[:3]
    
    # Border lines at 15% top and 27% left
    top_border = int(h * 0.15)
    left_border = int(w * 0.27)
    BORDER_COLOR = (0, 0, 255)  # Red in BGR
    
    # Output video: side by side, so width = 2 * original width
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w * 2, h))
    
    # Colors for overlay
    MASK_COLOR = (0, 255, 0)  # Green in BGR
    ALPHA = 0.4
    
    print(f"Creating visualization video: {output_path}")
    for i in tqdm(range(n_frames)):
        # Left: binary mask as grayscale (converted to 3-channel for stacking)
        mask_vis = (foreground_mask[i] * 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        
        # Right: mask overlay on RGB
        rgb_frame = color[i].copy()
        
        # Create colored overlay where mask is 1
        overlay = rgb_frame.copy()
        overlay[foreground_mask[i] == 1] = MASK_COLOR
        
        # Blend
        rgb_overlay = cv2.addWeighted(rgb_frame, 1 - ALPHA, overlay, ALPHA, 0)
        
        # Also draw contour for better visibility
        contours, _ = cv2.findContours(foreground_mask[i], cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb_overlay, contours, -1, (0, 255, 255), 2)  # Yellow contour
        
        # Draw border lines showing top 20% and left 20% exclusion zones
        cv2.line(rgb_overlay, (0, top_border), (w, top_border), BORDER_COLOR, 2)  # Horizontal line
        cv2.line(rgb_overlay, (left_border, 0), (left_border, h), BORDER_COLOR, 2)  # Vertical line
        
        # Stack side by side
        frame = np.hstack([mask_bgr, rgb_overlay])
        out.write(frame)
    
    out.release()
    print(f"Video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Obtain foreground mask from RGBD + robot arm masks")
    parser.add_argument("--chunk_path", type=str, required=True,
                        help="Path to chunk folder containing rgbd.npz and masks/masks.npz")
    parser.add_argument("--min_depth", type=float, default=0,
                        help="Minimum depth threshold in mm (exclusive, default: 0)")
    parser.add_argument("--max_depth", type=float, default=1500,
                        help="Maximum depth threshold in mm (exclusive, default: 1500)")
    parser.add_argument("--fps", type=int, default=30,
                        help="FPS for output video (default: 30)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as chunk_path)")
    args = parser.parse_args()
    
    chunk_path = Path(args.chunk_path)
    output_dir = Path(args.output_dir) if args.output_dir else chunk_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    color, depth, robot_arm_mask = load_data(chunk_path)
    
    # Compute foreground mask
    foreground_mask = compute_foreground_mask(
        depth, robot_arm_mask, 
        min_depth=args.min_depth, 
        max_depth=args.max_depth
    )
    
    # Final pass: keep only the largest connected component per frame
    print("\nFinal pass: keeping only largest connected component per frame...")
    for i in tqdm(range(len(foreground_mask))):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground_mask[i], connectivity=8
        )
        if num_labels > 1:  # label 0 is background
            # Find largest component (excluding background at index 0)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            foreground_mask[i] = (labels == largest_label).astype(np.uint8)
    
    # Statistics
    total_pixels = foreground_mask.size
    fg_pixels = foreground_mask.sum()
    print(f"\nForeground statistics:")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Foreground pixels: {fg_pixels:,} ({100*fg_pixels/total_pixels:.2f}%)")
    
    # Save foreground mask
    fg_mask_path = output_dir / "fg_mask.npz"
    np.savez_compressed(fg_mask_path, fg_mask=foreground_mask)
    print(f"\nForeground mask saved: {fg_mask_path}")
    
    # Create visualization video
    video_path = output_dir / "fg_seg_overlay.mp4"
    create_visualization_video(color, foreground_mask, video_path, fps=args.fps)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
