import numpy as np
from pathlib import Path
import cv2
from scipy import ndimage
import time 

from seg_with_arms_utils import (
    depth_to_point_cloud_full,
    background_subtraction,
    apply_depth_threshold,
    get_largest_connected_component,
    skeletonize_mask,
    create_overlay,
    create_skeleton_overlay,
    node_identification,
    prune_to_target_nodes,
    draw_mst_with_nodes,
    draw_mst_overlay,
    track_nodes_cpd_3d,
    pixel_to_3d,
    draw_tracked_nodes,
    draw_tracked_nodes_overlay,
    draw_skeleton_with_detected_nodes,
    draw_skeleton_with_tracked_nodes,
    draw_skeleton_with_tracked_nodes_overlay
)


# all helper functions defined in seg_with_arms_utils.py
# This file only contains the main execution logic


if __name__ == "__main__":
    # Camera intrinsics
    intrinsics = np.array([
        [606.1124267578125, 0, 641.7578125],
        [0, 605.8821411132812, 365.6518859863281],
        [0, 0, 1]
    ])
    
    # Load robot arm only trajectory data
    arm_only_data = np.load("./data/arm_traj1/arm_traj1.npy", allow_pickle=True).item()
    
    # Load full scene (arm with wires) trajectory data
    full_scene_data = np.load("./data/arm_traj1/arm_with_wires_traj1.npy", allow_pickle=True).item()

    # Output directory for masks and overlays
    output_dir = Path("./data/arm_traj1/wire_segmentation")
    output_dir.mkdir(exist_ok=True)
    
    # Create frames subdirectory
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Get sorted frame keys
    arm_frame_keys = sorted(arm_only_data.keys())
    full_frame_keys = sorted(full_scene_data.keys())
    
    # ============================================================
    # Synchronization: align sequences by green light frame
    # arm_traj3 turns green at frame 52
    # arm_with_wire_traj3 turns green at frame 68
    # for arm_traj1, both turn green at frame 57
    # We need to offset so that both sequences start at the same event
    # ============================================================
    arm_green_frame = 66
    full_green_frame = 66
    
    # Slice frame keys to synchronize: start from green frame for each
    arm_frame_keys = arm_frame_keys[arm_green_frame:]
    full_frame_keys = full_frame_keys[full_green_frame:]
    
    # Crop to same length
    n_frames = min(len(arm_frame_keys), len(full_frame_keys))
    arm_frame_keys = arm_frame_keys[:n_frames]
    full_frame_keys = full_frame_keys[:n_frames]
    
    print(f"Synchronized sequences:")
    print(f"  Arm-only: starting from frame {arm_green_frame}, {n_frames} frames")
    print(f"  Full scene: starting from frame {full_green_frame}, {n_frames} frames")
    
    # Parameters
    arm_subtraction_threshold = 80.0  # mm - points within this distance are considered arm
    depth_threshold = 1000.0  # mm - points beyond this depth are background
    arm_dilation_pixels = 0  # pixels to expand the arm mask
    
    # CPD tracking parameters
    cpd_beta = 10.0  # Smoothness (larger = more rigid)
    cpd_lambda = 2.0  # Regularization
    cpd_w = 0.1  # Outlier weight
    cpd_snap_threshold = 15.0  # Max distance to snap to skeleton
    
    # Target topology
    target_branch_nodes = 2
    target_leaf_nodes = 4
    
    # Video writer setup
    video_writer = None
    fps = 30
    
    # Collect all masks for saving as single array
    all_masks = []
    
    # Previous frame nodes for CPD tracking
    prev_nodes = None

    for i in range(n_frames):
        start = time.time()
        arm_frame_key = arm_frame_keys[i]
        full_frame_key = full_frame_keys[i]
        
        # Load arm-only data
        arm_data = arm_only_data[arm_frame_key]
        arm_depth = arm_data['transformed_depth'].copy()
        
        # Create arm valid mask and dilate it by 5 pixels
        arm_valid_mask = (arm_depth > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # 11x11 for 5 pixel dilation
        arm_valid_mask_dilated = cv2.dilate(arm_valid_mask, kernel, iterations=1)
        
        # Expand arm depth to dilated region using nearest neighbor
        # For dilated pixels, use the nearest valid arm depth value
        arm_depth_expanded = arm_depth.copy()
        new_pixels = (arm_valid_mask_dilated > 0) & (arm_valid_mask == 0)
        if np.any(new_pixels):
            # Use distance transform to find nearest valid pixel
            dist, indices = ndimage.distance_transform_edt(arm_valid_mask == 0, return_indices=True)
            arm_depth_expanded[new_pixels] = arm_depth[indices[0][new_pixels], indices[1][new_pixels]]
        
        # Load full scene data
        full_data = full_scene_data[full_frame_key]
        full_rgb = full_data['color'][:, :, ::-1]  # BGR to RGB
        full_depth = full_data['transformed_depth'].copy()
        
        # Convert depth to point clouds [H, W, 3]
        arm_pc = depth_to_point_cloud_full(arm_depth_expanded, intrinsics)
        full_pc = depth_to_point_cloud_full(full_depth, intrinsics)

        start = time.time()
        # Background subtraction: remove robot arm
        foreground_mask = background_subtraction(full_pc, arm_pc, threshold=arm_subtraction_threshold, arm_dilation=arm_dilation_pixels)
        
        # Apply depth thresholding
        foreground_mask = apply_depth_threshold(foreground_mask, full_depth, max_depth=depth_threshold)
        end1 = time.time()
        print(f"Frame {i} processing time: {end1 - start:.3f} seconds")
        
        # Get largest connected component
        largest_cc_mask = get_largest_connected_component(foreground_mask, 5)
        # largest_cc_mask = foreground_mask.copy()  # For now, skip largest CC and use full foreground mask for skeletonization
        
        # Skeletonize the largest connected component
        skeleton = skeletonize_mask(largest_cc_mask)
        
        # ============================================================
        # Node Identification with CPD Tracking (3D)
        # ============================================================
        if i == 0:
            # First frame: identify and prune nodes to target topology
            branch_nodes, leaf_nodes, mst_adj, node_coords = node_identification(skeleton)
            
            # Store detected nodes (before pruning) for visualization
            detected_branch = branch_nodes.copy() if len(branch_nodes) > 0 else np.empty((0, 2))
            detected_leaf = leaf_nodes.copy() if len(leaf_nodes) > 0 else np.empty((0, 2))
            
            if mst_adj is not None and node_coords is not None:
                pruned_result = prune_to_target_nodes(mst_adj, node_coords, 
                                                       target_branch_nodes=target_branch_nodes, 
                                                       target_leaf_nodes=target_leaf_nodes)
                tracked_branch = pruned_result["branch_coords"]
                tracked_leaf = pruned_result["leaf_coords"]
            else:
                tracked_branch = np.empty((0, 2))
                tracked_leaf = np.empty((0, 2))
            
            # Convert to 3D for next frame tracking
            tracked_branch_3d = pixel_to_3d(tracked_branch, full_depth, intrinsics)
            tracked_leaf_3d = pixel_to_3d(tracked_leaf, full_depth, intrinsics)
            
            confidence = 1.0  # First frame is always confident
            print(f"Frame {i}: Initialized with {len(tracked_branch)} branch, {len(tracked_leaf)} leaf nodes, detected {len(detected_branch)} branch, {len(detected_leaf)} leaf")
            
            # ============================================================
            # Print and visualize first frame node coordinates
            # ============================================================
            print("\n" + "=" * 60)
            print("FIRST FRAME NODE COORDINATES")
            print("=" * 60)
            print("\nBranch Nodes (2D pixel coords):")
            for idx, coord in enumerate(tracked_branch):
                print(f"  Branch {idx}: row={coord[0]:.1f}, col={coord[1]:.1f}")
            print("\nBranch Nodes (3D coords in mm):")
            for idx, coord in enumerate(tracked_branch_3d):
                print(f"  Branch {idx}: x={coord[0]:.2f}, y={coord[1]:.2f}, z={coord[2]:.2f}")
            print("\nLeaf Nodes (2D pixel coords):")
            for idx, coord in enumerate(tracked_leaf):
                print(f"  Leaf {idx}: row={coord[0]:.1f}, col={coord[1]:.1f}")
            print("\nLeaf Nodes (3D coords in mm):")
            for idx, coord in enumerate(tracked_leaf_3d):
                print(f"  Leaf {idx}: x={coord[0]:.2f}, y={coord[1]:.2f}, z={coord[2]:.2f}")
            print("=" * 60 + "\n")
            
            # Create first frame node visualization with coordinates
            first_frame_vis = full_rgb.copy()
            
            # Draw skeleton
            skeleton_dilated = cv2.dilate(skeleton, np.ones((3, 3), np.uint8), iterations=1)
            first_frame_vis[skeleton_dilated > 0] = [0, 191, 255]  # Deep sky blue
            
            # Draw branch nodes (purple) with coordinates
            for idx, coord in enumerate(tracked_branch):
                row, col = int(coord[0]), int(coord[1])
                cv2.circle(first_frame_vis, (col, row), 8, (128, 0, 128), -1)  # Purple filled
                cv2.circle(first_frame_vis, (col, row), 8, (255, 255, 255), 2)  # White border
                # Add coordinate label
                label = f"B{idx}({col},{row})"
                cv2.putText(first_frame_vis, label, (col + 10, row - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(first_frame_vis, label, (col + 10, row - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 1)
            
            # Draw leaf nodes (yellow) with coordinates
            for idx, coord in enumerate(tracked_leaf):
                row, col = int(coord[0]), int(coord[1])
                cv2.circle(first_frame_vis, (col, row), 8, (255, 255, 0), -1)  # Yellow filled
                cv2.circle(first_frame_vis, (col, row), 8, (255, 255, 255), 2)  # White border
                # Add coordinate label
                label = f"L{idx}({col},{row})"
                cv2.putText(first_frame_vis, label, (col + 10, row + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(first_frame_vis, label, (col + 10, row + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Add title
            cv2.putText(first_frame_vis, "First Frame Nodes", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(first_frame_vis, "First Frame Nodes", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            
            # Save the visualization
            first_frame_path = output_dir / "first_frame_nodes.png"
            cv2.imwrite(str(first_frame_path), cv2.cvtColor(first_frame_vis, cv2.COLOR_RGB2BGR))
            print(f"First frame node visualization saved to: {first_frame_path}")
            
        else:
            # Subsequent frames: use 3D CPD to track previous nodes to current skeleton
            tracked_nodes, detected_nodes, confidence = track_nodes_cpd_3d(
                prev_nodes, 
                skeleton,
                full_depth,
                intrinsics,
                beta=cpd_beta,
                lmbda=cpd_lambda,
                w=cpd_w
            )
            tracked_branch = tracked_nodes["branch_coords"]
            tracked_leaf = tracked_nodes["leaf_coords"]
            tracked_branch_3d = tracked_nodes["branch_3d"]
            tracked_leaf_3d = tracked_nodes["leaf_3d"]
            detected_branch = detected_nodes["branch_coords"]
            detected_leaf = detected_nodes["leaf_coords"]
            
            print(f"Frame {i}: 3D CPD tracking confidence = {confidence:.3f}, detected {len(detected_branch)} branch, {len(detected_leaf)} leaf")
        
        # Update prev_nodes for next frame (include 3D coordinates)
        prev_nodes = {
            "branch_coords": tracked_branch.copy(),
            "leaf_coords": tracked_leaf.copy(),
            "branch_3d": tracked_branch_3d.copy(),
            "leaf_3d": tracked_leaf_3d.copy()
        }
        
        end2 = time.time()
        print(f"Frame {i} total processing time: {end2 - start:.3f} seconds")

        # Create visualizations for 3x2 grid
        # Top-left: Original mask (black and white)
        mask_vis = np.stack([foreground_mask * 255] * 3, axis=-1).astype(np.uint8)
        
        # Top-right: Mask overlay on RGB (RED)
        mask_overlay = create_overlay(full_rgb, foreground_mask, color=[255, 0, 0], alpha=0.5)
        
        # Middle-left: Skeleton with ALL detected nodes (before pruning/matching) - DEEP SKY BLUE
        detected_nodes_dict = {"branch_coords": detected_branch, "leaf_coords": detected_leaf}
        skeleton_vis = draw_skeleton_with_detected_nodes(skeleton, detected_nodes_dict)
        
        # Middle-right: Skeleton overlay on RGB (DEEP SKY BLUE: 0, 191, 255) - thicker
        skeleton_overlay = create_skeleton_overlay(full_rgb, skeleton, color=[0, 191, 255], thickness=4)
        
        # Bottom-left: Skeleton with tracked nodes visualization - FOREST GREEN
        nodes_vis = draw_skeleton_with_tracked_nodes(skeleton, tracked_branch, tracked_leaf)
        
        # Bottom-right: Skeleton with tracked nodes overlay on RGB (FOREST GREEN: 34, 139, 34)
        nodes_overlay = draw_skeleton_with_tracked_nodes_overlay(full_rgb, skeleton, tracked_branch, tracked_leaf, 
                                                                   skeleton_color=(34, 139, 34))
        
        # Create 3x2 grid
        top_row = np.concatenate([mask_vis, mask_overlay], axis=1)
        middle_row = np.concatenate([skeleton_vis, skeleton_overlay], axis=1)
        bottom_row = np.concatenate([nodes_vis, nodes_overlay], axis=1)
        grid_3x2 = np.concatenate([top_row, middle_row, bottom_row], axis=0)
        
        # Initialize video writer on first frame
        if video_writer is None:
            H, W = grid_3x2.shape[:2]
            video_path = str(output_dir / "wire_segmentation_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
        
        # Write frame to video (convert RGB to BGR for OpenCV)
        video_writer.write(cv2.cvtColor(grid_3x2, cv2.COLOR_RGB2BGR))
        
        # Collect mask for batch saving
        all_masks.append(foreground_mask)
        
        # Save visualization frame
        viz_path = frames_dir / f"wire_viz_{i:04d}.png"
        cv2.imwrite(str(viz_path), cv2.cvtColor(grid_3x2, cv2.COLOR_RGB2BGR))
        
        if i % 50 == 0:
            print(f"Frame {i}: {np.sum(foreground_mask)} foreground pixels, {np.sum(skeleton)} skeleton pixels")
    
    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {output_dir / 'wire_segmentation_video.mp4'}")
    
    # Save all masks as single n x H x W array
    mask_path = output_dir / "wire_masks.npy"
    all_masks = np.stack(all_masks, axis=0)
    np.save(mask_path, all_masks)
    print(f"Masks saved to {mask_path} with shape {all_masks.shape}")
    
    print(f"\nProcessed {n_frames} frames")
    print(f"Results saved to {output_dir}")
    print(f"Frames saved to {frames_dir}")