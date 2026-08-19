#!/usr/bin/env python3
"""
Script to obtain foreground masks for yellow BDLO by:
1. Removing robot arm (using robot arm masks)
2. Recovering yellow-like pixels that were incorrectly masked as robot
3. Depth thresholding (depth > 0 and depth < max_depth)

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


def is_robot_color(bgr_frame: np.ndarray) -> np.ndarray:
    """
    Detect robot arm colors: gray/white/black (low saturation) and red.
    Pixels NOT matching these are likely the yellow BDLO.

    Returns:
        (H, W) uint8 mask, 1 = robot-like color
    """
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    # Gray/white/black: low saturation
    is_gray = s < 40
    # Red: hue near 0 or 180 with some saturation
    is_red = ((h < 10) | (h > 165)) & (s > 30)
    return (is_gray | is_red).astype(np.uint8)


def compute_foreground_mask(color: np.ndarray, depth: np.ndarray,
                            robot_arm_mask: np.ndarray,
                            min_depth: float = 0, max_depth: float = 2000):
    """
    Compute foreground mask by:
    1. Removing robot arm (robot_arm_mask == 1 means background)
    2. Recovering yellow-like pixels incorrectly masked as robot
    3. Depth thresholding

    Args:
        color: (N, H, W, 3) BGR images
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

        # Detect robot-like colors (gray/white/black + red)
        robot_color = is_robot_color(color[i])

        # In robot mask: only keep pixels with robot-like color as background
        # Non-robot-colored pixels in robot mask are recovered as BDLO
        not_robot = (dilated_robot_mask == 0) | (robot_color == 0)

        # Depth thresholding
        valid_depth = (depth[i] > min_depth) & (depth[i] < max_depth)

        # Foreground = (not robot OR not robot-colored) AND valid depth
        foreground_mask[i] = (not_robot & valid_depth).astype(np.uint8)

        # Exclude border regions
        foreground_mask[i, :top_border, :] = 0
        foreground_mask[i, :, :left_border] = 0

        # Keep components above minimum area (BDLO branches may be disconnected)
        min_component_area = 500
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground_mask[i], connectivity=8
        )
        if num_labels > 1:
            keep_mask = np.zeros_like(foreground_mask[i])
            for lbl in range(1, num_labels):
                if stats[lbl, cv2.CC_STAT_AREA] >= min_component_area:
                    keep_mask[labels == lbl] = 1
            foreground_mask[i] = keep_mask

        # Smooth boundary
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        foreground_mask[i] = cv2.morphologyEx(foreground_mask[i], cv2.MORPH_OPEN, smooth_kernel)
        foreground_mask[i] = cv2.morphologyEx(foreground_mask[i], cv2.MORPH_CLOSE, smooth_kernel)

        # Shrink foreground by 1 pixel to remove edge noise
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

    top_border = int(h * 0.15)
    left_border = int(w * 0.27)
    BORDER_COLOR = (0, 0, 255)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w * 2, h))

    MASK_COLOR = (0, 255, 0)
    ALPHA = 0.4

    print(f"Creating visualization video: {output_path}")
    for i in tqdm(range(n_frames)):
        mask_vis = (foreground_mask[i] * 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

        rgb_frame = color[i].copy()
        overlay = rgb_frame.copy()
        overlay[foreground_mask[i] == 1] = MASK_COLOR
        rgb_overlay = cv2.addWeighted(rgb_frame, 1 - ALPHA, overlay, ALPHA, 0)

        contours, _ = cv2.findContours(foreground_mask[i], cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb_overlay, contours, -1, (0, 255, 255), 2)

        cv2.line(rgb_overlay, (0, top_border), (w, top_border), BORDER_COLOR, 2)
        cv2.line(rgb_overlay, (left_border, 0), (left_border, h), BORDER_COLOR, 2)

        frame = np.hstack([mask_bgr, rgb_overlay])
        out.write(frame)

    out.release()
    print(f"Video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Obtain foreground mask for yellow BDLO")
    parser.add_argument("--chunk_path", type=str,
                        default="/mnt/mydisk/captured_data_double_arm/bdlo_yellow_2sec/chunk_0",
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
        color, depth, robot_arm_mask,
        min_depth=args.min_depth,
        max_depth=args.max_depth
    )

    # # Final pass: remove small noise components
    # min_component_area = 500
    # print(f"\nFinal pass: removing components < {min_component_area} pixels...")
    # for i in tqdm(range(len(foreground_mask))):
    #     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    #         foreground_mask[i], connectivity=8
    #     )
    #     if num_labels > 1:
    #         keep_mask = np.zeros_like(foreground_mask[i])
    #         for lbl in range(1, num_labels):
    #             if stats[lbl, cv2.CC_STAT_AREA] >= min_component_area:
    #                 keep_mask[labels == lbl] = 1
    #         foreground_mask[i] = keep_mask

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
