"""
CPD-based Node Identification Ablation Study

This script compares three node tracking methods:
1. Per-frame detection + pruning (no temporal consistency)
2. Pure CPD tracking in 2D (uses interpolated positions, may drift)
3. CPD-based tracking in 3D with Hungarian assignment (our method)

Output: Side-by-side comparison video showing the effectiveness of each method.
"""

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
    cpd_register,
    pixel_to_3d,
    track_nodes_cpd_3d,
    draw_skeleton_with_tracked_nodes,
    draw_skeleton_with_tracked_nodes_overlay
)


def add_title_to_image(image, title, font_scale=0.8, thickness=2, color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Add a title bar to the top of an image."""
    H, W = image.shape[:2]
    title_height = 40
    
    # Create title bar
    title_bar = np.full((title_height, W, 3), bg_color, dtype=np.uint8)
    
    # Calculate text position (centered)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
    text_x = (W - text_size[0]) // 2
    text_y = (title_height + text_size[1]) // 2
    
    cv2.putText(title_bar, title, (text_x, text_y), font, font_scale, color, thickness)
    
    # Concatenate title bar with image
    return np.concatenate([title_bar, image], axis=0)


def track_nodes_cpd_no_hungarian(prev_nodes, curr_skeleton, beta=10.0, lmbda=2.0, w=0.1):
    """
    CPD + Node Identification but NO Hungarian matching.
    
    This uses:
    - Node identification (detect branch/leaf candidates) ✓
    - CPD registration ✓
    - Greedy nearest-neighbor snapping (NO optimal Hungarian assignment) ✓
    
    This will show correspondence errors when CPD's transformed positions
    compete for the same candidate node (greedy can make suboptimal choices).
    
    Args:
        prev_nodes: dict with "branch_coords" and "leaf_coords"
        curr_skeleton: H x W binary skeleton mask
        beta: CPD smoothness parameter
        lmbda: CPD regularization parameter
        w: Outlier weight
    """
    prev_branch = prev_nodes.get("branch_coords", np.empty((0, 2)))
    prev_leaf = prev_nodes.get("leaf_coords", np.empty((0, 2)))
    n_branch = len(prev_branch)
    n_leaf = len(prev_leaf)
    
    if n_branch == 0 and n_leaf == 0:
        return prev_nodes.copy(), {"branch_coords": np.empty((0, 2)), "leaf_coords": np.empty((0, 2))}
    
    # Step 1: Detect all branch/leaf nodes from current skeleton
    all_branch, all_leaf, mst_adj, node_coords = node_identification(curr_skeleton)
    
    detected_nodes = {
        "branch_coords": all_branch.copy() if len(all_branch) > 0 else np.empty((0, 2)),
        "leaf_coords": all_leaf.copy() if len(all_leaf) > 0 else np.empty((0, 2))
    }
    
    # Step 2a: Track BRANCH nodes using CPD + greedy snapping (no Hungarian)
    if n_branch > 0 and len(all_branch) > 0:
        Y_branch = prev_branch.astype(np.float64)
        X_branch = all_branch.astype(np.float64)
        
        # Run CPD to get transformed positions
        T_Y_branch, _ = cpd_register(Y_branch, X_branch, beta=beta, lmbda=lmbda, w=w)
        
        # Greedy nearest-neighbor snapping (NOT optimal Hungarian)
        tracked_branch = np.zeros((n_branch, 2), dtype=np.float64)
        used_candidates = set()
        
        for i, node in enumerate(T_Y_branch):
            dists = np.linalg.norm(X_branch - node, axis=1)
            sorted_indices = np.argsort(dists)
            
            # Greedy: take nearest unused candidate
            for idx in sorted_indices:
                if idx not in used_candidates:
                    tracked_branch[i] = X_branch[idx]
                    used_candidates.add(idx)
                    break
    elif n_branch > 0:
        tracked_branch = prev_branch.copy()
    else:
        tracked_branch = np.empty((0, 2))
    
    # Step 2b: Track LEAF nodes using CPD + greedy snapping (no Hungarian)
    if n_leaf > 0 and len(all_leaf) > 0:
        Y_leaf = prev_leaf.astype(np.float64)
        X_leaf = all_leaf.astype(np.float64)
        
        # Run CPD to get transformed positions
        T_Y_leaf, _ = cpd_register(Y_leaf, X_leaf, beta=beta, lmbda=lmbda, w=w)
        
        # Greedy nearest-neighbor snapping (NOT optimal Hungarian)
        tracked_leaf = np.zeros((n_leaf, 2), dtype=np.float64)
        used_candidates = set()
        
        for i, node in enumerate(T_Y_leaf):
            dists = np.linalg.norm(X_leaf - node, axis=1)
            sorted_indices = np.argsort(dists)
            
            for idx in sorted_indices:
                if idx not in used_candidates:
                    tracked_leaf[i] = X_leaf[idx]
                    used_candidates.add(idx)
                    break
    elif n_leaf > 0:
        tracked_leaf = prev_leaf.copy()
    else:
        tracked_leaf = np.empty((0, 2))
    
    return {
        "branch_coords": tracked_branch,
        "leaf_coords": tracked_leaf,
    }, detected_nodes


if __name__ == "__main__":
    # Camera intrinsics
    intrinsics = np.array([
        [606.1124267578125, 0, 641.7578125],
        [0, 605.8821411132812, 365.6518859863281],
        [0, 0, 1]
    ])
    
    # Load data
    arm_only_data = np.load("./data/arm_traj1/arm_traj1.npy", allow_pickle=True).item()
    full_scene_data = np.load("./data/arm_traj1/arm_with_wires_traj1.npy", allow_pickle=True).item()

    # Output
    output_dir = Path("./data/arm_traj1/wire_segmentation")
    output_dir.mkdir(exist_ok=True)
    
    # Frame synchronization
    arm_frame_keys = sorted(arm_only_data.keys())
    full_frame_keys = sorted(full_scene_data.keys())
    
    arm_green_frame = 66
    full_green_frame = 66
    
    arm_frame_keys = arm_frame_keys[arm_green_frame:]
    full_frame_keys = full_frame_keys[full_green_frame:]
    
    n_frames = min(len(arm_frame_keys), len(full_frame_keys))
    arm_frame_keys = arm_frame_keys[:n_frames]
    full_frame_keys = full_frame_keys[:n_frames]
    
    print(f"Processing {n_frames} frames for ablation comparison")
    
    # Parameters
    arm_subtraction_threshold = 80.0
    depth_threshold = 1000.0
    
    # CPD parameters
    cpd_beta = 10.0
    cpd_lambda = 2.0
    cpd_w = 0.1
    
    # Target topology
    target_branch_nodes = 2
    target_leaf_nodes = 4
    
    # Track state for each method
    prev_nodes_method1 = None  # Per-frame (resets each frame)
    prev_nodes_method2 = None  # Pure CPD (2D)
    prev_nodes_method3 = None  # CPD-based 3D (our method)
    
    # Video writer
    video_writer = None
    fps = 30
    
    for i in range(n_frames):
        start = time.time()
        arm_frame_key = arm_frame_keys[i]
        full_frame_key = full_frame_keys[i]
        
        # Load and process data
        arm_data = arm_only_data[arm_frame_key]
        arm_depth = arm_data['transformed_depth'].copy()
        
        arm_valid_mask = (arm_depth > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        arm_valid_mask_dilated = cv2.dilate(arm_valid_mask, kernel, iterations=1)
        
        arm_depth_expanded = arm_depth.copy()
        new_pixels = (arm_valid_mask_dilated > 0) & (arm_valid_mask == 0)
        if np.any(new_pixels):
            dist, indices = ndimage.distance_transform_edt(arm_valid_mask == 0, return_indices=True)
            arm_depth_expanded[new_pixels] = arm_depth[indices[0][new_pixels], indices[1][new_pixels]]
        
        full_data = full_scene_data[full_frame_key]
        full_rgb = full_data['color'][:, :, ::-1]  # BGR to RGB
        full_depth = full_data['transformed_depth'].copy()
        
        arm_pc = depth_to_point_cloud_full(arm_depth_expanded, intrinsics)
        full_pc = depth_to_point_cloud_full(full_depth, intrinsics)
        
        # Background subtraction
        foreground_mask = background_subtraction(full_pc, arm_pc, threshold=arm_subtraction_threshold, arm_dilation=0)
        foreground_mask = apply_depth_threshold(foreground_mask, full_depth, max_depth=depth_threshold)
        
        # Get largest connected component and skeletonize
        largest_cc_mask = get_largest_connected_component(foreground_mask, 4)
        skeleton = skeletonize_mask(largest_cc_mask)
        
        # ============================================================
        # Method 1: Per-frame detection + pruning (no temporal tracking)
        # ============================================================
        branch_nodes, leaf_nodes, mst_adj, node_coords = node_identification(skeleton)
        
        if mst_adj is not None and node_coords is not None:
            pruned_result = prune_to_target_nodes(mst_adj, node_coords,
                                                   target_branch_nodes=target_branch_nodes,
                                                   target_leaf_nodes=target_leaf_nodes)
            method1_branch = pruned_result["branch_coords"]
            method1_leaf = pruned_result["leaf_coords"]
        else:
            method1_branch = np.empty((0, 2))
            method1_leaf = np.empty((0, 2))
        
        # ============================================================
        # Method 2: CPD + Node ID + Greedy (no Hungarian matching)
        # Uses node identification but greedy nearest neighbor assignment
        # ============================================================
        if i == 0:
            # Initialize with pruned result
            method2_branch = method1_branch.copy()
            method2_leaf = method1_leaf.copy()
            prev_nodes_method2 = {"branch_coords": method2_branch.copy(), "leaf_coords": method2_leaf.copy()}
        else:
            # CPD + Node ID + Greedy nearest neighbor (no Hungarian)
            result_m2, _ = track_nodes_cpd_no_hungarian(prev_nodes_method2, skeleton,
                                                         beta=10.0, lmbda=2.0, w=0.1)
            method2_branch = result_m2["branch_coords"]
            method2_leaf = result_m2["leaf_coords"]
            prev_nodes_method2 = {"branch_coords": method2_branch.copy(), "leaf_coords": method2_leaf.copy()}
        
        # ============================================================
        # Method 3: 3D CPD with Hungarian assignment (our method)
        # ============================================================
        if i == 0:
            # Initialize with pruned result
            method3_branch = method1_branch.copy()
            method3_leaf = method1_leaf.copy()
            # Get 3D coordinates
            method3_branch_3d = pixel_to_3d(method3_branch, full_depth, intrinsics)
            method3_leaf_3d = pixel_to_3d(method3_leaf, full_depth, intrinsics)
            prev_nodes_method3 = {
                "branch_coords": method3_branch.copy(), 
                "leaf_coords": method3_leaf.copy(),
                "branch_3d": method3_branch_3d.copy(),
                "leaf_3d": method3_leaf_3d.copy()
            }
        else:
            result_m3, _, _ = track_nodes_cpd_3d(prev_nodes_method3, skeleton, full_depth, intrinsics,
                                                  beta=cpd_beta, lmbda=cpd_lambda, w=cpd_w)
            method3_branch = result_m3["branch_coords"]
            method3_leaf = result_m3["leaf_coords"]
            method3_branch_3d = result_m3["branch_3d"]
            method3_leaf_3d = result_m3["leaf_3d"]
            prev_nodes_method3 = {
                "branch_coords": method3_branch.copy(), 
                "leaf_coords": method3_leaf.copy(),
                "branch_3d": method3_branch_3d.copy(),
                "leaf_3d": method3_leaf_3d.copy()
            }
        
        # ============================================================
        # Create visualizations
        # ============================================================
        
        # Method 1: Per-frame detection + pruning
        m1_skeleton_vis = draw_skeleton_with_tracked_nodes(skeleton, method1_branch, method1_leaf)
        m1_overlay = draw_skeleton_with_tracked_nodes_overlay(full_rgb, skeleton, method1_branch, method1_leaf,
                                                               skeleton_color=(34, 139, 34))  # Green
        m1_combined = np.concatenate([m1_skeleton_vis, m1_overlay], axis=1)
        m1_combined = add_title_to_image(m1_combined, "Method 1: Node ID + Pruning")
        
        # Method 2: CPD + Node ID + Greedy (no Hungarian)
        m2_skeleton_vis = draw_skeleton_with_tracked_nodes(skeleton, method2_branch, method2_leaf)
        m2_overlay = draw_skeleton_with_tracked_nodes_overlay(full_rgb, skeleton, method2_branch, method2_leaf,
                                                               skeleton_color=(34, 139, 34))  # Green
        m2_combined = np.concatenate([m2_skeleton_vis, m2_overlay], axis=1)
        m2_combined = add_title_to_image(m2_combined, "Method 2: CPD")
        
        # Method 3: 3D CPD with Hungarian (our method)
        m3_skeleton_vis = draw_skeleton_with_tracked_nodes(skeleton, method3_branch, method3_leaf)
        m3_overlay = draw_skeleton_with_tracked_nodes_overlay(full_rgb, skeleton, method3_branch, method3_leaf,
                                                               skeleton_color=(34, 139, 34))  # Green
        m3_combined = np.concatenate([m3_skeleton_vis, m3_overlay], axis=1)
        m3_combined = add_title_to_image(m3_combined, "Method 3: Ours")
        
        # Stack all three rows
        grid = np.concatenate([m1_combined, m2_combined, m3_combined], axis=0)
        
        # Initialize video writer
        if video_writer is None:
            H, W = grid.shape[:2]
            video_path = str(output_dir / "CPD_comparison.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
            print(f"Video size: {W}x{H}")
        
        # Write frame
        video_writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        
        if i % 20 == 0:
            elapsed = time.time() - start
            print(f"Frame {i}/{n_frames}: {elapsed:.3f}s")
    
    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved to {output_dir / 'CPD_comparison.mp4'}")
    
    print(f"\nProcessed {n_frames} frames")
    print("Ablation comparison complete!")
