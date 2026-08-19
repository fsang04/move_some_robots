"""
Naive Fabric Segmentation Script

Segments fabric by:
1. Depth filtering: keep pixels with depth < 1000mm
2. Color filtering: select green-like pixels as foreground

Input: tracking_fabric2_data.npy
Output: 
    - masks/mask_frame_{xxxx}.npy
    - masks_viz_overlay/ (binary mask + overlay visualization)

Author: Auto-generated
Date: 2026-02-23
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def segment_green_fabric(color_bgr: np.ndarray, depth: np.ndarray, 
                         max_depth: float = 1000.0) -> np.ndarray:
    """
    Segment green fabric using depth and color filtering.
    
    Args:
        color_bgr: BGR image (H, W, 3)
        depth: Depth map (H, W) in mm
        max_depth: Maximum depth threshold in mm
    
    Returns:
        mask: Binary mask (H, W) where True = foreground (fabric)
    """
    H, W = depth.shape
    
    # 1. Depth mask: keep pixels with valid depth < max_depth
    depth_mask = (depth > 0) & (depth < max_depth)
    
    # 2. Color mask: select green-like pixels
    # Convert BGR to HSV for better color segmentation
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    
    # Green hue range in HSV (H: 35-85, S: 40-255, V: 40-255)
    # Adjust these thresholds based on your fabric color
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    green_mask = cv2.inRange(color_hsv, lower_green, upper_green) > 0
    
    # 3. Combine depth and color masks
    mask = depth_mask & green_mask
    
    # 4. Optional: morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = mask.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small noise
    
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
    data_path = Path("/home/yehengz/deformable_seg/data/full/tracking_fabric2_data.npy")
    output_dir = Path("/home/yehengz/deformable_seg/data/arm_traj4_fabric")
    masks_dir = output_dir / "masks"
    viz_dir = output_dir / "masks_viz_overlay"
    
    # Create output directories
    masks_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {data_path}")
    tracking_data = np.load(str(data_path), allow_pickle=True).item()
    
    # Get sorted frame keys
    frame_keys = sorted(tracking_data.keys())
    print(f"Found {len(frame_keys)} frames")
    
    # Print sample frame info
    sample_data = tracking_data[frame_keys[0]]
    print("Frame data keys and shapes:")
    for k, v in sample_data.items():
        print(f"  {k}: {v.shape}")
    
    # Process each frame
    print("\nSegmenting frames...")
    for idx, frame_key in enumerate(tqdm(frame_keys)):
        data = tracking_data[frame_key]
        color_bgr = data['color']
        depth = data['transformed_depth']
        
        # Segment fabric
        mask = segment_green_fabric(color_bgr, depth, max_depth=1200.0)
        
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
