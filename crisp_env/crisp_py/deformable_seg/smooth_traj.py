#!/usr/bin/env python3
"""
Smooth Trajectory Tracking for Wire Nodes.

Pipeline:
1. Frame 0: Identify all nodes, prune to 2 branch + 4 leaf, index them
2. Frame N>0: Identify candidates, CPD + Hungarian match to previous frame
3. Outlier removal: 3x3 window per node, fix jumps
4. Trajectory smoothing
5. Visualization: 2x2 grid (all candidates, matched, outlier-removed, smoothed)

Works for arm_traj1, arm_traj2, arm_traj3
"""

import numpy as np
import cv2
import argparse
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter1d, label as ndimage_label
from skimage.morphology import skeletonize


# ============================================================
# CAMERA INTRINSICS
# ============================================================
INTRINSICS = np.array([
    [606.11, 0.0, 641.76],
    [0.0, 605.88, 365.65],
    [0.0, 0.0, 1.0]
], dtype=np.float64)


# ============================================================
# TRAJECTORY CONFIGURATIONS
# ============================================================
TRAJECTORY_CONFIGS = {
    1: {
        'arm_data_path': Path('data/arm_traj1/arm_traj1.npy'),
        'full_data_path': Path('data/arm_traj1/arm_with_wires_traj1.npy'),
        'output_dir': Path('tracking_output/smooth_traj1'),
        'arm_green_frame': 66,
        'full_green_frame': 66,
        'precomputed_mask_dir': None,  # No precomputed masks for traj1
    },
    2: {
        'arm_data_path': Path('data/arm_traj2/arm_traj2.npy'),
        'full_data_path': Path('data/arm_traj2/arm_with_wires_traj2.npy'),
        'output_dir': Path('tracking_output/smooth_traj2'),
        'arm_green_frame': 0,
        'full_green_frame': 0,
        'precomputed_mask_dir': Path('data/arm_traj2/masks'),  # Precomputed arm masks for traj2
    },
    3: {
        'arm_data_path': Path('data/arm_traj3/arm_traj3_contact.npy'),
        'full_data_path': Path('data/arm_traj3/arm_with_wires_traj3_contact.npy'),
        'output_dir': Path('tracking_output/smooth_traj3'),
        'arm_green_frame': 84,
        'full_green_frame': 100,
        'precomputed_mask_dir': None,  # No precomputed masks for traj3
    },
}


# ============================================================
# DEPTH AND PROJECTION UTILITIES
# ============================================================

def depth_to_point_cloud(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Convert depth image to 3D point cloud."""
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth.astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    return np.stack([x, y, z], axis=-1)


def pixel_to_3d(pixel_coords: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Back-project 2D pixel coordinates to 3D."""
    if len(pixel_coords) == 0:
        return np.empty((0, 3), dtype=np.float64)
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    H, W = depth.shape
    
    coords_3d = []
    for row, col in pixel_coords:
        row_int, col_int = int(round(row)), int(round(col))
        if 0 <= row_int < H and 0 <= col_int < W:
            z = depth[row_int, col_int]
            if z > 0 and not np.isnan(z):
                x = (col_int - cx) * z / fx
                y = (row_int - cy) * z / fy
                coords_3d.append([x, y, z])
            else:
                coords_3d.append([np.nan, np.nan, np.nan])
        else:
            coords_3d.append([np.nan, np.nan, np.nan])
    
    return np.array(coords_3d, dtype=np.float64)


def project_3d_to_2d(points_3d: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Project 3D points to 2D pixel coordinates."""
    if len(points_3d) == 0:
        return np.empty((0, 2), dtype=np.float64)
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    z_safe = np.maximum(z, 1e-6)
    
    u = (x * fx) / z_safe + cx  # col
    v = (y * fy) / z_safe + cy  # row
    
    return np.stack([v, u], axis=1)  # row, col


# ============================================================
# SEGMENTATION
# ============================================================

def segment_foreground(
    full_depth: np.ndarray,
    arm_depth: np.ndarray,
    intrinsics: np.ndarray,
    bg_threshold: float = 80.0,
    max_depth: float = 1000.0,
    n_components: int = 5,
    precomputed_arm_mask: np.ndarray = None,
) -> tuple:
    """Segment foreground (wire) from depth images."""
    if precomputed_arm_mask is not None:
        foreground_mask = (1 - precomputed_arm_mask).astype(np.uint8)
    else:
        full_pc = depth_to_point_cloud(full_depth, intrinsics)
        arm_pc = depth_to_point_cloud(arm_depth, intrinsics)
        diff = np.linalg.norm(full_pc - arm_pc, axis=-1)
        foreground_mask = (diff > bg_threshold).astype(np.uint8)
    
    # Depth thresholding
    foreground_mask[full_depth > max_depth] = 0
    foreground_mask[full_depth <= 0] = 0
    foreground_mask[np.isnan(full_depth)] = 0
    
    # Keep top-K connected components
    labeled, num_features = ndimage_label(foreground_mask)
    if num_features > 0:
        component_sizes = np.array([np.sum(labeled == i) for i in range(1, num_features + 1)])
        k = min(n_components, num_features)
        largest_labels = np.argsort(component_sizes)[::-1][:k] + 1
        foreground_mask = np.isin(labeled, largest_labels).astype(np.uint8)
    
    # Skeletonize
    skeleton_mask = skeletonize(foreground_mask > 0).astype(np.uint8)
    
    return foreground_mask, skeleton_mask


# ============================================================
# NODE IDENTIFICATION
# ============================================================

def identify_nodes(skeleton_mask: np.ndarray) -> tuple:
    """
    Identify branch and leaf nodes via MST degree analysis on 8-connected graph.
    
    Returns:
        branch_2d: B x 2 branch node coords (row, col)
        leaf_2d: L x 2 leaf node coords (row, col)
        mst_adjacency: N x N MST adjacency matrix
        all_coords: N x 2 all skeleton coords
    """
    from scipy.sparse import csr_matrix
    
    binary = skeleton_mask > 0
    coords = np.column_stack(np.nonzero(binary)).astype(np.int64)
    
    if coords.shape[0] == 0:
        return np.empty((0, 2)), np.empty((0, 2)), None, None
    
    if coords.shape[0] == 1:
        return np.empty((0, 2)), coords.copy(), None, coords
    
    # Build distance matrix
    dists = cdist(coords, coords, metric='euclidean')
    
    # 8-connected adjacency: only connect pixels within sqrt(2) distance
    adjacency = np.where(dists <= np.sqrt(2) + 1e-6, dists, 0)
    np.fill_diagonal(adjacency, 0)
    
    # Build MST from 8-connected graph
    sparse_adj = csr_matrix(adjacency)
    mst = minimum_spanning_tree(sparse_adj)
    mst_dense = mst.toarray()
    mst_symmetric = mst_dense + mst_dense.T
    
    # Compute degrees
    degrees = (mst_symmetric > 0).sum(axis=1)
    
    # Branch nodes: degree >= 3, Leaf nodes: degree == 1
    branch_mask = degrees >= 3
    leaf_mask = degrees == 1
    
    branch_2d = coords[branch_mask]
    leaf_2d = coords[leaf_mask]
    
    return branch_2d, leaf_2d, mst_symmetric, coords


def prune_to_target_topology(
    mst_adjacency: np.ndarray,
    node_coords: np.ndarray,
    target_branch: int = 2,
    target_leaf: int = 4,
) -> dict:
    """Prune MST to target topology (2 branch + 4 leaf) by removing shortest leaf segments."""
    if mst_adjacency is None or node_coords is None or len(node_coords) == 0:
        return {"branch_coords": np.empty((0, 2)), "leaf_coords": np.empty((0, 2))}
    
    adjacency = np.array(mst_adjacency, dtype=np.float64)
    coords = np.array(node_coords, dtype=np.int64)
    n = adjacency.shape[0]
    
    # Make symmetric
    adjacency = np.maximum(adjacency, adjacency.T)
    
    # Track active nodes
    active = np.ones(n, dtype=bool)
    max_iterations = n
    
    # Prune leaf segments
    for _ in range(max_iterations):
        degrees = np.zeros(n, dtype=np.int64)
        for i in range(n):
            if active[i]:
                degrees[i] = np.sum((adjacency[i, :] > 0) & active)
        
        leaf_mask = (degrees == 1) & active
        num_leaves = np.sum(leaf_mask)
        
        if num_leaves <= target_leaf:
            break
        
        # Find shortest leaf segment to prune
        leaf_indices = np.where(leaf_mask)[0]
        min_length = np.inf
        prune_idx = -1
        
        for leaf_idx in leaf_indices:
            current = leaf_idx
            path = [current]
            visited = {current}
            
            while True:
                neighbors = np.where((adjacency[current, :] > 0) & active)[0]
                neighbors = [nb for nb in neighbors if nb not in visited]
                
                if len(neighbors) == 0:
                    break
                
                next_node = neighbors[0]
                path.append(next_node)
                visited.add(next_node)
                
                deg = np.sum((adjacency[next_node, :] > 0) & active)
                if deg >= 3 or deg == 1:
                    break
                
                current = next_node
            
            path_length = sum(
                adjacency[path[j], path[j + 1]]
                for j in range(len(path) - 1)
            )
            
            if path_length < min_length:
                min_length = path_length
                prune_idx = leaf_idx
        
        if prune_idx >= 0:
            active[prune_idx] = False
            adjacency[prune_idx, :] = 0
            adjacency[:, prune_idx] = 0
    
    # Extract results
    active_indices = np.where(active)[0]
    new_coords = coords[active_indices]
    new_adjacency = adjacency[np.ix_(active_indices, active_indices)]
    
    new_degrees = (new_adjacency > 0).sum(axis=1)
    branch_mask = new_degrees >= 3
    leaf_mask = new_degrees == 1
    
    return {
        "branch_coords": new_coords[branch_mask],
        "leaf_coords": new_coords[leaf_mask],
    }


# ============================================================
# STEP 2: HUNGARIAN MATCHING (same as wire_tracker)
# ============================================================

def hungarian_match_nodes(
    prev_nodes_3d: np.ndarray,
    candidate_nodes_3d: np.ndarray,
) -> np.ndarray:
    """
    Match previous frame nodes to detected candidates using Hungarian algorithm.
    Same logic as wire_tracker._hungarian_replace_anchors.
    
    Uses simple Euclidean distance as cost matrix.
    
    Args:
        prev_nodes_3d: M x 3 previous frame nodes
        candidate_nodes_3d: N x 3 detected candidate nodes
    
    Returns:
        matched_3d: M x 3 matched nodes in previous frame ordering
    """
    M = len(prev_nodes_3d)
    N = len(candidate_nodes_3d)
    
    if M == 0:
        return np.empty((0, 3), dtype=np.float64)
    
    # Start with previous positions (fallback if no match)
    matched_3d = prev_nodes_3d.copy()
    
    if N == 0:
        return matched_3d
    
    # Filter valid candidates (no NaN)
    valid_mask = ~np.any(np.isnan(candidate_nodes_3d), axis=1)
    valid_cand_3d = candidate_nodes_3d[valid_mask]
    
    if len(valid_cand_3d) == 0:
        return matched_3d
    
    # Cost matrix: Euclidean distance (same as wire_tracker)
    cost = cdist(prev_nodes_3d, valid_cand_3d)
    
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Replace matched positions
    for r, c in zip(row_ind, col_ind):
        matched_3d[r] = valid_cand_3d[c]
    
    return matched_3d


# ============================================================
# STEP 3: TRAJECTORY SMOOTHING
# ============================================================

def smooth_trajectories(
    trajectories_3d: np.ndarray,
    sigma: float = 3.0,
) -> np.ndarray:
    """
    Apply Gaussian smoothing to 3D trajectories.
    
    Args:
        trajectories_3d: T x N x 3 array
        sigma: Gaussian filter sigma
    
    Returns:
        smoothed: T x N x 3 smoothed trajectories
    """
    T, N, _ = trajectories_3d.shape
    smoothed = trajectories_3d.copy()
    
    for node_idx in range(N):
        for dim in range(3):
            traj = trajectories_3d[:, node_idx, dim]
            
            # Handle NaN by interpolation
            valid_mask = ~np.isnan(traj)
            if np.sum(valid_mask) < 2:
                continue
            
            valid_indices = np.where(valid_mask)[0]
            valid_values = traj[valid_mask]
            
            all_indices = np.arange(T)
            interpolated = np.interp(all_indices, valid_indices, valid_values)
            
            # Gaussian smoothing
            smoothed[:, node_idx, dim] = gaussian_filter1d(interpolated, sigma=sigma, mode='nearest')
    
    return smoothed


# ============================================================
# STEP 5: VISUALIZATION (2x2 grid, 2D only)
# ============================================================

def create_2x2_visualization(
    rgb: np.ndarray,
    foreground_mask: np.ndarray,
    all_candidates_2d: np.ndarray,  # All detected nodes
    matched_2d: np.ndarray,         # After Hungarian matching
    outlier_removed_2d: np.ndarray, # After outlier removal
    smooth_2d: np.ndarray,          # After smoothing
    frame_idx: int,
) -> np.ndarray:
    """
    Create 2x2 grid visualization:
    - Top-left: All candidates + point cloud on RGB
    - Top-right: Matched nodes
    - Bottom-left: After outlier removal
    - Bottom-right: Final smooth output
    """
    H, W = rgb.shape[:2]
    
    # Create point cloud overlay
    pc_overlay = rgb.copy()
    pc_overlay[foreground_mask > 0] = [255, 255, 0]  # Yellow for foreground
    
    def draw_nodes(img, branch_2d, leaf_2d, title):
        """Draw nodes on image with title."""
        out = img.copy()
        
        # Draw branch nodes (red circles)
        for pt in branch_2d:
            if not np.any(np.isnan(pt)):
                cv2.circle(out, (int(pt[1]), int(pt[0])), 8, (255, 0, 0), -1)
                cv2.circle(out, (int(pt[1]), int(pt[0])), 8, (0, 0, 0), 2)
        
        # Draw leaf nodes (green circles)
        for pt in leaf_2d:
            if not np.any(np.isnan(pt)):
                cv2.circle(out, (int(pt[1]), int(pt[0])), 8, (0, 255, 0), -1)
                cv2.circle(out, (int(pt[1]), int(pt[0])), 8, (0, 0, 0), 2)
        
        # Add title
        cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(out, f"Frame {frame_idx}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return out
    
    # Split candidates into branch (first 2) and leaf (last 4) for consistent coloring
    # For "all candidates", we just show all detected points
    n_branch = 2
    
    # Panel 1: All candidates + point cloud
    panel1 = draw_nodes(
        pc_overlay,
        all_candidates_2d[:n_branch] if len(all_candidates_2d) >= n_branch else all_candidates_2d,
        all_candidates_2d[n_branch:] if len(all_candidates_2d) > n_branch else np.empty((0, 2)),
        "1. All Candidates"
    )
    
    # Panel 2: Matched nodes
    panel2 = draw_nodes(
        rgb.copy(),
        matched_2d[:n_branch],
        matched_2d[n_branch:],
        "2. CPD Matched"
    )
    
    # Panel 3: Same as matched (no outlier removal anymore)
    panel3 = draw_nodes(
        rgb.copy(),
        outlier_removed_2d[:n_branch],
        outlier_removed_2d[n_branch:],
        "3. Raw Matched"
    )
    
    # Panel 4: Final smooth
    panel4 = draw_nodes(
        rgb.copy(),
        smooth_2d[:n_branch],
        smooth_2d[n_branch:],
        "4. Smoothed"
    )
    
    # Combine into 2x2 grid
    top_row = np.hstack([panel1, panel2])
    bottom_row = np.hstack([panel3, panel4])
    grid = np.vstack([top_row, bottom_row])
    
    return grid


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_trajectory(traj_id: int):
    """Process a single trajectory."""
    config = TRAJECTORY_CONFIGS[traj_id]
    
    print("=" * 60)
    print(f"PROCESSING TRAJECTORY {traj_id}")
    print("=" * 60)
    
    # Load BOTH arm-only and full scene data
    print(f"\nLoading arm-only data from: {config['arm_data_path']}")
    arm_data = np.load(str(config['arm_data_path']), allow_pickle=True).item()
    
    print(f"Loading full scene data from: {config['full_data_path']}")
    full_data = np.load(str(config['full_data_path']), allow_pickle=True).item()
    
    # Get frame keys and apply synchronization offset
    arm_frame_keys = sorted(arm_data.keys())
    full_frame_keys = sorted(full_data.keys())
    
    arm_green_frame = config['arm_green_frame']
    full_green_frame = config['full_green_frame']
    
    arm_frame_keys = arm_frame_keys[arm_green_frame:]
    full_frame_keys = full_frame_keys[full_green_frame:]
    
    n_frames = min(len(arm_frame_keys), len(full_frame_keys))
    arm_frame_keys = arm_frame_keys[:n_frames]
    full_frame_keys = full_frame_keys[:n_frames]
    
    print(f"Synchronized sequences:")
    print(f"  Arm-only: starting from frame {arm_green_frame}")
    print(f"  Full scene: starting from frame {full_green_frame}")
    print(f"  Total frames: {n_frames}")
    
    # Create output directory
    output_dir = config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parameters
    bg_threshold = 80.0
    max_depth = 1000.0
    n_components = 5
    target_branch = 2
    target_leaf = 4
    smooth_sigma = 3.0
    
    # Storage
    all_rgb = []
    all_depth = []
    all_foreground_masks = []
    all_candidates_2d = []      # All detected candidates
    all_matched_3d = []         # After Hungarian matching (indexed)
    
    # Reference from frame 0
    ref_branch_3d = None
    ref_leaf_3d = None
    prev_branch_3d = None
    prev_leaf_3d = None
    prev_nodes_3d = None  # Combined [branch, leaf] for matching
    
    # ================================================================
    # PASS 1: Detection and Matching
    # ================================================================
    print("\n--- Pass 1: Detection and Matching ---")
    
    for i in range(n_frames):
        arm_frame_key = arm_frame_keys[i]
        full_frame_key = full_frame_keys[i]
        
        # Load arm data
        arm_frame_data = arm_data[arm_frame_key]
        arm_depth = arm_frame_data['transformed_depth'].copy()
        
        # Load full scene data
        full_frame_data = full_data[full_frame_key]
        rgb = full_frame_data['color'][:, :, ::-1]  # BGR to RGB
        full_depth = full_frame_data['transformed_depth'].copy()
        
        # Load precomputed mask if available
        precomputed_mask = None
        if config['precomputed_mask_dir'] is not None:
            mask_path = config['precomputed_mask_dir'] / f"mask_frame_{i:04d}.npy"
            if mask_path.exists():
                precomputed_mask = np.load(str(mask_path))
        
        # Segment foreground (using arm_depth for background subtraction)
        foreground_mask, skeleton_mask = segment_foreground(
            full_depth, arm_depth, INTRINSICS,
            bg_threshold, max_depth, n_components,
            precomputed_mask
        )
        
        # Identify all nodes
        branch_2d, leaf_2d, mst_adj, all_coords = identify_nodes(skeleton_mask)
        
        # Store all candidates (for visualization)
        candidates_2d = np.vstack([branch_2d, leaf_2d]) if len(branch_2d) > 0 or len(leaf_2d) > 0 else np.empty((0, 2))
        
        if i == 0:
            # ============================================================
            # STEP 1: Frame 0 - Prune and index nodes
            # ============================================================
            print(f"  Frame 0: Detected {len(branch_2d)} branch, {len(leaf_2d)} leaf (before pruning)")
            
            # Prune to target topology
            if mst_adj is not None and all_coords is not None:
                pruned = prune_to_target_topology(mst_adj, all_coords, target_branch, target_leaf)
                branch_2d = pruned["branch_coords"]
                leaf_2d = pruned["leaf_coords"]
            
            print(f"  Frame 0: After pruning: {len(branch_2d)} branch, {len(leaf_2d)} leaf")
            
            # Convert to 3D
            branch_3d = pixel_to_3d(branch_2d, full_depth, INTRINSICS)
            leaf_3d = pixel_to_3d(leaf_2d, full_depth, INTRINSICS)
            
            # Index: [branch_0, branch_1, leaf_0, leaf_1, leaf_2, leaf_3]
            matched_3d = np.vstack([branch_3d, leaf_3d]) if len(branch_3d) > 0 or len(leaf_3d) > 0 else np.empty((0, 3))
            
            # Set reference
            ref_branch_3d = branch_3d.copy()
            ref_leaf_3d = leaf_3d.copy()
            prev_branch_3d = branch_3d.copy()
            prev_leaf_3d = leaf_3d.copy()
            prev_nodes_3d = matched_3d.copy()
            
        else:
            # ============================================================
            # STEP 2: Frame N>0 - Hungarian match (same as wire_tracker)
            # ============================================================
            # Convert all candidates to 3D
            branch_3d_cand = pixel_to_3d(branch_2d, full_depth, INTRINSICS)
            leaf_3d_cand = pixel_to_3d(leaf_2d, full_depth, INTRINSICS)
            
            n_ref_branch = len(ref_branch_3d)
            n_ref_leaf = len(ref_leaf_3d)
            
            # Match branch nodes using Hungarian (distance-based)
            if n_ref_branch > 0 and len(branch_3d_cand) > 0:
                matched_branch_3d = hungarian_match_nodes(prev_branch_3d, branch_3d_cand)
            elif n_ref_branch > 0:
                matched_branch_3d = prev_branch_3d.copy()
            else:
                matched_branch_3d = np.empty((0, 3))
            
            # Match leaf nodes using Hungarian (distance-based)
            if n_ref_leaf > 0 and len(leaf_3d_cand) > 0:
                matched_leaf_3d = hungarian_match_nodes(prev_leaf_3d, leaf_3d_cand)
            elif n_ref_leaf > 0:
                matched_leaf_3d = prev_leaf_3d.copy()
            else:
                matched_leaf_3d = np.empty((0, 3))
            
            matched_3d = np.vstack([matched_branch_3d, matched_leaf_3d]) if n_ref_branch > 0 or n_ref_leaf > 0 else np.empty((0, 3))
            
            # Update previous reference for next frame
            prev_branch_3d = matched_branch_3d.copy()
            prev_leaf_3d = matched_leaf_3d.copy()
            prev_nodes_3d = matched_3d.copy()
        
        # Store
        all_rgb.append(rgb)
        all_depth.append(full_depth)
        all_foreground_masks.append(foreground_mask)
        all_candidates_2d.append(candidates_2d)
        all_matched_3d.append(matched_3d)
        
        if i % 50 == 0:
            print(f"  Frame {i}: matched {len(matched_3d)} nodes")
    
    # ================================================================
    # STEP 3: Trajectory Smoothing (directly on matched results)
    # ================================================================
    print("\n--- Step 3: Trajectory Smoothing ---")
    
    # Stack to T x N x 3
    matched_stacked = np.stack(all_matched_3d, axis=0)
    print(f"  Trajectories shape: {matched_stacked.shape}")
    
    smooth_3d = smooth_trajectories(matched_stacked, sigma=smooth_sigma)
    print(f"  Smoothing applied (sigma={smooth_sigma})")
    
    # ================================================================
    # STEP 4: Visualization (2x2 grid)
    # ================================================================
    print("\n--- Step 4: Creating Visualization ---")
    
    video_writer = None
    fps = 30
    
    for i in range(n_frames):
        rgb = all_rgb[i]
        foreground_mask = all_foreground_masks[i]
        
        # Get 2D coordinates for each stage
        candidates_2d = all_candidates_2d[i]
        
        matched_3d_frame = all_matched_3d[i]
        smooth_3d_frame = smooth_3d[i]
        
        # Project to 2D
        valid_matched = ~np.any(np.isnan(matched_3d_frame), axis=1)
        valid_smooth = ~np.any(np.isnan(smooth_3d_frame), axis=1)
        
        matched_2d = project_3d_to_2d(matched_3d_frame[valid_matched], INTRINSICS) if np.any(valid_matched) else np.empty((0, 2))
        smooth_2d = project_3d_to_2d(smooth_3d_frame[valid_smooth], INTRINSICS) if np.any(valid_smooth) else np.empty((0, 2))
        
        # Create 2x2 visualization (candidates, matched, matched again, smooth)
        viz = create_2x2_visualization(
            rgb, foreground_mask,
            candidates_2d,
            matched_2d,
            matched_2d,  # No outlier removal, just show matched twice
            smooth_2d,
            i
        )
        
        # Initialize video writer
        if video_writer is None:
            H, W = viz.shape[:2]
            video_path = str(output_dir / "smooth_traj.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
        
        video_writer.write(cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
        
        if i % 50 == 0:
            print(f"  Frame {i}/{n_frames}")
    
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved: {output_dir / 'smooth_traj.mp4'}")
    
    # Save trajectories
    np.save(str(output_dir / "matched_3d.npy"), np.stack(all_matched_3d, axis=0))
    np.save(str(output_dir / "smooth_3d.npy"), smooth_3d)
    print(f"Trajectories saved to {output_dir}")
    
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Smooth Trajectory Tracking")
    parser.add_argument("--traj", type=int, default=1, choices=[1, 2, 3],
                        help="Trajectory ID (1, 2, or 3)")
    args = parser.parse_args()
    
    process_trajectory(args.traj)


if __name__ == "__main__":
    main()
