"""
Naive Cloth Segmentation Script

Segments cloth by:
1. Depth filtering: keep pixels with depth < 1000mm
2. Color filtering: select cloth-like pixels as foreground

Input: /home/yehengz/deformable_seg/data/arm_traj5_cloth/rgbd.npz
Output: 
    - masks/mask_frame_{xxxx}.npy
    - masks_viz_overlay/ (binary mask + overlay visualization)

Author: Auto-generated
Date: 2026-02-25
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def segment_cloth(color_bgr: np.ndarray, depth: np.ndarray, 
                  max_depth: float = 1000.0) -> np.ndarray:
    """
    Segment cloth using depth and color filtering.
    
    Args:
        color_bgr: BGR image (H, W, 3)
        depth: Depth map (H, W) in mm
        max_depth: Maximum depth threshold in mm
    
    Returns:
        mask: Binary mask (H, W) where True = foreground (cloth)
    """
    H, W = depth.shape
    
    # 1. Depth mask: keep pixels with valid depth < max_depth
    depth_mask = (depth > 980) & (depth < max_depth)
    
    # 2. Color mask: select cloth-like pixels
    # Convert BGR to HSV for better color segmentation
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    
    # Cloth color range in HSV - adjust these thresholds based on your cloth color
    # Default: targeting a general cloth color range
    # You may need to adjust these values based on your specific cloth color
    
    # Option 1: Green cloth (similar to fabric)
    # lower_color = np.array([35, 40, 40])
    # upper_color = np.array([85, 255, 255])
    
    # Option 2: Blue cloth
    # lower_color = np.array([90, 40, 40])
    # upper_color = np.array([130, 255, 255])
    
    # Option 3: White/gray cloth (low saturation)
    # lower_color = np.array([0, 0, 100])
    # upper_color = np.array([180, 50, 255])
    
    # Option 4: Red cloth (note: red wraps around in HSV)
    # lower_color1 = np.array([0, 40, 40])
    # upper_color1 = np.array([10, 255, 255])
    # lower_color2 = np.array([170, 40, 40])
    # upper_color2 = np.array([180, 255, 255])
    
    # Brown cloth detection
    # Reference colors: #8B5740 (HSV ~18, 54%, 55%) and #CB856B (HSV ~16, 47%, 80%)
    # OpenCV uses H: 0-179, S: 0-255, V: 0-255
    # Hue 16-18 → OpenCV ~8-9, with margin: 5-15
    # Saturation 47-54% → OpenCV ~120-138, with margin: 80-180
    # Value 55-80% → OpenCV ~140-204, with margin: 100-255
    lower_color = np.array([5, 80, 100])
    upper_color = np.array([20, 200, 255])
    
    color_mask = cv2.inRange(color_hsv, lower_color, upper_color) > 0
    
    # 3. Combine depth and color masks
    mask = depth_mask & color_mask
    
    # 4. Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = mask.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small noise
    
    # 5. Keep only the largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:  # More than just background
        # Find the largest component (excluding background label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8)
    
    return mask.astype(bool)


def create_visualization(color_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Create side-by-side visualization: binary mask | overlay.
    
    Args:
        color_bgr: BGR image (H, W, 3)
        mask: Binary mask (H, W)
    
    Returns:
        viz: Visualization image (H, W*2, 3)
    """
    H, W = mask.shape
    
    # Binary mask visualization (white on black)
    mask_viz = np.zeros((H, W, 3), dtype=np.uint8)
    mask_viz[mask] = [255, 255, 255]
    
    # Overlay visualization (highlight mask region)
    overlay = color_bgr.copy()
    # Dim non-mask regions
    overlay[~mask] = (overlay[~mask] * 0.3).astype(np.uint8)
    # Add green tint to mask region
    overlay[mask, 1] = np.minimum(overlay[mask, 1].astype(np.int32) + 50, 255).astype(np.uint8)
    
    # Add contour
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    
    # Side-by-side
    viz = np.concatenate([mask_viz, overlay], axis=1)
    
    return viz


def main():
    # Paths
    data_path = Path("/home/yehengz/deformable_seg/data/arm_traj5_cloth/rgbd.npz")
    output_dir = Path("/home/yehengz/deformable_seg/data/arm_traj5_cloth")
    masks_dir = output_dir / "masks"
    viz_dir = output_dir / "masks_viz_overlay"
    
    # Create output directories
    masks_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data from NPZ format
    print(f"Loading data from: {data_path}")
    data = np.load(str(data_path))
    
    # NPZ format: 'color' (N, H, W, 3) in BGR, 'depth' (N, H, W)
    color_data_bgr = data['color']  # Shape: (N, H, W, 3), BGR format
    depth_data = data['depth']      # Shape: (N, H, W)
    
    n_frames = color_data_bgr.shape[0]
    print(f"Found {n_frames} frames")
    print(f"Color shape: {color_data_bgr.shape}")
    print(f"Depth shape: {depth_data.shape}")
    
    # Process each frame
    print("\nSegmenting frames...")
    for idx in tqdm(range(n_frames)):
        color_bgr = color_data_bgr[idx]  # (H, W, 3) BGR
        depth = depth_data[idx]          # (H, W)
        
        # Segment cloth
        mask = segment_cloth(color_bgr, depth, max_depth=1200.0)
        
        # Save mask
        mask_path = masks_dir / f"mask_frame_{idx:04d}.npy"
        np.save(str(mask_path), mask)
        
        # Create and save visualization
        viz = create_visualization(color_bgr, mask)
        viz_path = viz_dir / f"mask_viz_{idx:04d}.png"
        cv2.imwrite(str(viz_path), viz)
    
    print(f"\nDone!")
    print(f"Masks saved to: {masks_dir}")
    print(f"Visualizations saved to: {viz_dir}")
    
    # Print summary statistics
    sample_mask = np.load(str(masks_dir / "mask_frame_0000.npy"))
    print(f"\nSample mask shape: {sample_mask.shape}")
    print(f"Sample mask foreground pixels: {np.sum(sample_mask)} ({100*np.mean(sample_mask):.1f}%)")


if __name__ == "__main__":
    main()
