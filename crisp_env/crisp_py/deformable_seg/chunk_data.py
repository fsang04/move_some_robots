#!/usr/bin/env python3
"""
Script to chunk large trajectory data into smaller segments for sequential processing.

Converts:
- arm_with_wires_traj2.npy (color, transformed_depth) -> rgbd.npz per chunk
- ee_poses_3d.npy -> ee_pose_3d.npy per chunk  
- masks/ folder -> masks/ per chunk

Output structure:
    bdlo_traj2/
    ├── chunk0/
    │   ├── rgbd.npz
    │   ├── ee_pose_3d.npy
    │   └── masks/
    ├── chunk1/
    │   └── ...
    └── ...
"""

import numpy as np
import os
import shutil
from pathlib import Path
from tqdm import tqdm


def chunk_trajectory_data(
    input_dir: str,
    output_dir: str,
    traj_name: str = "traj2",
    total_frames: int = 225,
    chunk_size: int = 45,
):
    """
    Chunk trajectory data into smaller segments.
    
    Args:
        input_dir: Path to input data folder (e.g., /home/yehengz/deformable_seg/data)
        output_dir: Path to output folder (e.g., /home/yehengz/deformable_seg/data)
        traj_name: Trajectory name (e.g., "traj2")
        total_frames: Total number of frames to process
        chunk_size: Number of frames per chunk
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Input paths
    arm_traj_dir = input_dir / f"arm_{traj_name}"
    arm_with_wires_path = arm_traj_dir / f"arm_with_wires_{traj_name}.npy"
    ee_pose_path = arm_traj_dir / "ee_pose_output" / "ee_poses_3d.npy"
    masks_dir = arm_traj_dir / "masks"
    
    # Output directory
    bdlo_dir = output_dir / f"bdlo_{traj_name}"
    
    print(f"Input directory: {arm_traj_dir}")
    print(f"Output directory: {bdlo_dir}")
    print(f"Total frames: {total_frames}, Chunk size: {chunk_size}")
    
    # Verify input files exist
    assert arm_with_wires_path.exists(), f"File not found: {arm_with_wires_path}"
    assert ee_pose_path.exists(), f"File not found: {ee_pose_path}"
    assert masks_dir.exists(), f"Directory not found: {masks_dir}"
    
    # Load data
    print("\nLoading arm_with_wires data...")
    arm_with_wires_data = np.load(arm_with_wires_path, allow_pickle=True).item()
    
    # Data is organized as {frame_idx: {'color': ..., 'transformed_depth': ...}}
    # Extract color and depth arrays
    color_list = []
    depth_list = []
    for i in range(total_frames):
        color_list.append(arm_with_wires_data[i]['color'])
        depth_list.append(arm_with_wires_data[i]['transformed_depth'])
    
    color_data = np.array(color_list)  # N x H x W x 3
    depth_data = np.array(depth_list)  # N x H x W
    
    print(f"  Color shape: {color_data.shape}")
    print(f"  Depth shape: {depth_data.shape}")
    
    print("\nLoading ee_poses_3d...")
    ee_poses_data = np.load(ee_pose_path, allow_pickle=True).item()
    ee_poses = np.array(ee_poses_data['ee_3d'])[:total_frames]  # N x 2 x 3
    print(f"  EE poses shape: {ee_poses.shape}")
    
    # Get mask files (sorted)
    mask_files = sorted(masks_dir.glob("*.npy"))[:total_frames]
    if len(mask_files) == 0:
        mask_files = sorted(masks_dir.glob("*.png"))[:total_frames]
    if len(mask_files) == 0:
        mask_files = sorted(masks_dir.glob("*.jpg"))[:total_frames]
    if len(mask_files) == 0:
        mask_files = sorted(masks_dir.glob("*"))[:total_frames]
    print(f"\nFound {len(mask_files)} mask files")
    
    # Calculate number of chunks
    n_chunks = (total_frames + chunk_size - 1) // chunk_size
    print(f"\nCreating {n_chunks} chunks...")
    
    # Create output directory
    bdlo_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each chunk
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_frames)
        actual_chunk_size = end_idx - start_idx
        
        print(f"\n--- Chunk {chunk_idx}: frames {start_idx}-{end_idx-1} ({actual_chunk_size} frames) ---")
        
        # Create chunk directory
        chunk_dir = bdlo_dir / f"chunk{chunk_idx}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save rgbd.npz
        chunk_color = color_data[start_idx:end_idx]
        chunk_depth = depth_data[start_idx:end_idx]
        
        rgbd_path = chunk_dir / "rgbd.npz"
        np.savez_compressed(
            rgbd_path,
            color=chunk_color,
            depth=chunk_depth
        )
        print(f"  Saved rgbd.npz: color {chunk_color.shape}, depth {chunk_depth.shape}")
        
        # 2. Save ee_pose_3d.npy
        chunk_ee_poses = ee_poses[start_idx:end_idx]
        ee_pose_out_path = chunk_dir / "ee_pose_3d.npy"
        np.save(ee_pose_out_path, chunk_ee_poses)
        print(f"  Saved ee_pose_3d.npy: {chunk_ee_poses.shape}")
        
        # 3. Copy masks
        chunk_masks_dir = chunk_dir / "masks"
        chunk_masks_dir.mkdir(parents=True, exist_ok=True)
        
        for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
            if global_idx < len(mask_files):
                src_mask = mask_files[global_idx]
                # Keep original filename or rename to sequential
                dst_mask = chunk_masks_dir / f"mask_{local_idx:04d}{src_mask.suffix}"
                shutil.copy2(src_mask, dst_mask)
        
        print(f"  Copied {actual_chunk_size} masks to {chunk_masks_dir}")
    
    print(f"\n✓ Done! Output saved to: {bdlo_dir}")
    print(f"  Total chunks: {n_chunks}")
    print(f"  Frames per chunk: {chunk_size} (last chunk may have fewer)")
    
    # Summary
    print("\nOutput structure:")
    for chunk_idx in range(n_chunks):
        chunk_dir = bdlo_dir / f"chunk{chunk_idx}"
        print(f"  {chunk_dir.name}/")
        for item in sorted(chunk_dir.iterdir()):
            if item.is_dir():
                n_files = len(list(item.iterdir()))
                print(f"    {item.name}/ ({n_files} files)")
            else:
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"    {item.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    # Configuration
    DATA_DIR = "/home/yehengz/deformable_seg/data"
    
    chunk_trajectory_data(
        input_dir=DATA_DIR,
        output_dir=DATA_DIR,
        traj_name="traj2",
        total_frames=225,
        chunk_size=45,
    )
