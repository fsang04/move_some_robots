from typing import Dict, List, Optional, Tuple

import math
import time
import numpy as np  # type: ignore[import]
import open3d as o3d  # type: ignore[import]

from seg_utils import (
    compute_point_cloud_mask,
    filter_pcd_mask_dbscan,
    skelentonize,
    node_identification,
    prune_leaf_segments,
    mask_from_mst,
    filter_point_cloud_radius,
    remove_small_components
)
from gmm_utils import (
    extract_gmm_keypoints,
    classify_deformable_topology,
    build_wire_connections,
    refine_keypoints_uniform_edges,
)
from types import SimpleNamespace

from viz_utils import (
    create_skel_mask_viz,
    create_color_point_cloud,
    visualize_foreground_background_gmm,
)
from constrained_gmm_utils import ConstrainedGMM
from repulsion_utils import (
    repulsion_relaxation,
)
from repulsion_wires_utils import (
    repulsion_relaxation_wire,
    compute_spacing_stats,
)


def farthest_point_sampling(
	points: np.ndarray,
	colors: Optional[np.ndarray],
	num_samples: int,
	*,
	seed: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
	num_points = len(points)
	if num_points == 0 or num_samples is None or num_samples <= 0 or num_samples >= num_points:
		return points.copy(), None if colors is None else colors.copy()

	rng = np.random.default_rng(seed)
	chosen = np.empty(num_samples, dtype=np.int64)
	chosen[0] = int(rng.integers(num_points))
	distances = np.linalg.norm(points - points[chosen[0]], axis=1)
	for i in range(1, num_samples):
		next_idx = int(np.argmax(distances))
		chosen[i] = next_idx
		new_distances = np.linalg.norm(points - points[next_idx], axis=1)
		distances = np.minimum(distances, new_distances)

	sampled_points = points[chosen]
	sampled_colors = None if colors is None else colors[chosen]
	return sampled_points, sampled_colors



if __name__ == "__main__":
    print("Loading data for segmentation...")
    
    # Load sample data
    data = np.load("/home/yehengz/deformable_seg/data/full/test_wire.npy", allow_pickle=True).item()
    sampled_points = data[0]

    bg_data = np.load("/home/yehengz/deformable_seg/data/bg/test_wire_bg.npy", allow_pickle=True).item()
    sampled_points_bg = bg_data[0]

    # convert BGR to RGB for visualization and refinement colouring
    rgb_image = sampled_points['color'][:, :, ::-1]
    
    print("Creating segmentation masks...")
    
    # Camera intrinsics
    intrinsics = np.array([
        [606.1124267578125, 0, 641.7578125],
        [0, 605.8821411132812, 365.6518859863281],
        [0, 0, 1]
    ])
    
    # compute point cloud mask
    pc_mask_initial = compute_point_cloud_mask(
        sampled_points_bg['transformed_depth'],
        sampled_points['transformed_depth'],
        intrinsics,
        distance_threshold=18
    )
    print("Filtering point cloud mask using DBSCAN...")
    pc_mask = filter_pcd_mask_dbscan(
        pc_mask_initial,
        sampled_points['transformed_depth'],
        intrinsics,
        eps=30.0,
        min_samples=18,
    )
    
    print("point cloud masks created successfully")

    pc_mask = remove_small_components(pc_mask, min_size=800)


    skeleton_pc_mask = skelentonize(pc_mask)
    branch_nodes, end_nodes, adjacency, coords = node_identification(
        skeleton_pc_mask,
        return_graph=True,
    )

    expected_leaf_nodes = 4
    if adjacency is not None and coords is not None:
        pruning_result = prune_leaf_segments(
            adjacency,
            coords,
            expected_num_leaf_nodes=expected_leaf_nodes,
        )
        branch_nodes = pruning_result["branch_coords"]
        end_nodes = pruning_result["leaf_coords"]
        skeleton_pc_mask = mask_from_mst(
            pruning_result["adjacency"],
            pruning_result["coords"],
            skeleton_pc_mask.shape,
        )
        pc_mask = np.where(skeleton_pc_mask > 0, pc_mask, 0).astype(np.uint8)

    print(branch_nodes.shape, end_nodes.shape)

    depth_data = sampled_points['transformed_depth']
    points, colors, valid_mask = create_color_point_cloud(
        rgb_image,
        depth_data,
        intrinsics,
        return_valid_mask=True,
    )

    # Convert branch and end nodes (2D pixel coords) to 3D points
    def pixel_to_3d(pixel_coords, depth, intrinsics):
        """Convert 2D pixel coordinates to 3D points using depth."""
        if pixel_coords.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)
        
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        points_3d = []
        for coord in pixel_coords:
            # coord is (row, col) = (y, x)
            row, col = int(coord[0]), int(coord[1])
            if 0 <= row < depth.shape[0] and 0 <= col < depth.shape[1]:
                z = depth[row, col]
                if z > 0:  # valid depth
                    x = (col - cx) * z / fx
                    y = (row - cy) * z / fy
                    points_3d.append([x, y, z])
        
        return np.array(points_3d, dtype=np.float64) if points_3d else np.empty((0, 3), dtype=np.float64)
    
    branch_nodes_3d = pixel_to_3d(branch_nodes, depth_data, intrinsics)
    end_nodes_3d = pixel_to_3d(end_nodes, depth_data, intrinsics)
    
    # Combine branch and end nodes as anchor points
    anchor_points_3d = np.vstack([branch_nodes_3d, end_nodes_3d]) if branch_nodes_3d.size > 0 or end_nodes_3d.size > 0 else np.empty((0, 3))
    print(f"  Anchor points (branch + leaf): {anchor_points_3d.shape[0]} (branch: {branch_nodes_3d.shape[0]}, leaf: {end_nodes_3d.shape[0]})")

    valid_flat = valid_mask.reshape(-1)
    valid_indices = np.flatnonzero(valid_flat)
    pc_mask_flat = (pc_mask > 0).reshape(-1)
    if valid_indices.size:
        foreground_mask_points = pc_mask_flat[valid_indices].astype(bool)
    else:
        foreground_mask_points = np.zeros(points.shape[0], dtype=bool)

    foreground_points = points[foreground_mask_points]
    foreground_colors = colors[foreground_mask_points]

    # foreground_points, foreground_colors = filter_point_cloud_radius(
    #     foreground_points, 
    #     foreground_colors, 
    #     radius=20.0, 
    #     min_neighbors=100
    # )


    ds_fg_points, ds_fg_colors = foreground_points, foreground_colors

    if foreground_points.shape[0] < 10:
        raise ValueError("Not enough foreground points to fit constrained GMM")

    print(
        f"Foreground points available for keypoint extraction: {foreground_points.shape[0]:,} (from {points.shape[0]:,} valid points)"
    )

    # ============================================================
    # Stage 1: Farthest Point Sampling (FPS) initialization
    # with anchor points (branch + leaf nodes) enforced
    # ============================================================
    n_keypoints = 21
    n_anchors = anchor_points_3d.shape[0]
    n_fps_additional = max(0, n_keypoints - n_anchors)
    
    print(f"Stage 1 (FPS): Initializing {n_keypoints} keypoints ({n_anchors} anchors + {n_fps_additional} FPS)...")
    start_time = time.time()

    if n_anchors > 0:
        # Start with anchor points, then FPS the rest
        # First, find the closest points in ds_fg_points to each anchor
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(ds_fg_points)
        _, anchor_indices = nn.kneighbors(anchor_points_3d)
        anchor_indices = anchor_indices.flatten()
        
        # Initialize with anchors
        chosen = list(anchor_indices)
        chosen_set = set(chosen)
        
        # Initialize distance array based on all anchor points
        distances = np.full(ds_fg_points.shape[0], np.inf)
        for idx in chosen:
            new_distances = np.linalg.norm(ds_fg_points - ds_fg_points[idx], axis=1)
            distances = np.minimum(distances, new_distances)
        
        # FPS for remaining keypoints
        for k in range(n_fps_additional):
            # Mask out already chosen points
            masked_distances = distances.copy()
            for idx in chosen_set:
                masked_distances[idx] = -np.inf
            
            next_idx = int(np.argmax(masked_distances))
            chosen.append(next_idx)
            chosen_set.add(next_idx)
            new_distances = np.linalg.norm(ds_fg_points - ds_fg_points[next_idx], axis=1)
            distances = np.minimum(distances, new_distances)
        
        chosen = np.array(chosen, dtype=np.int64)
    else:
        # No anchors, fall back to standard FPS
        centroid = np.mean(ds_fg_points, axis=0)
        dists_to_centroid = np.linalg.norm(ds_fg_points - centroid, axis=1)
        start_idx = int(np.argmax(dists_to_centroid))

        chosen = np.empty(n_keypoints, dtype=np.int64)
        chosen[0] = start_idx
        distances = np.linalg.norm(ds_fg_points - ds_fg_points[chosen[0]], axis=1)

        for k in range(1, n_keypoints):
            next_idx = int(np.argmax(distances))
            chosen[k] = next_idx
            new_distances = np.linalg.norm(ds_fg_points - ds_fg_points[next_idx], axis=1)
            distances = np.minimum(distances, new_distances)

    fps_keypoints = ds_fg_points[chosen]
    fps_time = time.time() - start_time
    print(f"  FPS completed in {fps_time:.3f}s: {fps_keypoints.shape[0]} keypoints")
    
    # Create fixed mask: branch nodes should not move
    # Branch nodes are the first branch_nodes_3d.shape[0] in the chosen array (if anchors were used)
    n_branch = branch_nodes_3d.shape[0] if n_anchors > 0 else 0
    fixed_mask = np.zeros(fps_keypoints.shape[0], dtype=bool)
    fixed_mask[:n_branch] = True  # First n_branch keypoints are branch nodes
    print(f"  Branch nodes (fixed): {n_branch}")

    # Compute spacing stats after FPS
    fps_stats = compute_spacing_stats(fps_keypoints, n_neighbors=1)
    print(
        f"  FPS spacing → min: {fps_stats['min_dist']:.2f}, "
        f"max: {fps_stats['max_dist']:.2f}, "
        f"uniformity: {fps_stats['uniformity']:.3f}"
    )

    # ============================================================
    # Stage 2: Repulsion-based relaxation with KNN + skeleton path validation
    # For each node: KNN to find neighbors, remove those without valid skeleton path
    # Valid path = exists in skeleton AND does not pass through any other keypoint
    # ============================================================
    print("Stage 2 (Repulsion with KNN + Skeleton Path Validation): Relaxing keypoint positions...")
    relax_start = time.time()
    relaxation_result = repulsion_relaxation_wire(
        fps_keypoints,
        foreground_points,  # Project to FULL cloud
        skeleton_pc_mask,   # Use skeleton mask for path validation
        intrinsics,
        fixed_mask=fixed_mask,  # Branch nodes stay fixed
        n_iterations=40,
        learning_rate=5.0,
        k_neighbors=3,          # KNN neighbors to consider
        target_edge_length=None,  # Auto-compute from initial mean
        epsilon=1e-8,
        project_each_step=True,
        rebuild_neighbors_every=20,  # Rebuild neighbor graph periodically
        return_debug=True,
    )
    keypoints = relaxation_result["keypoints"]
    degrees = relaxation_result["degrees"]
    edges = relaxation_result["edges"]
    relax_debug = relaxation_result.get("debug")
    relax_time = time.time() - relax_start
    print(f"  Repulsion relaxation completed in {relax_time:.3f}s")

    # Print degree distribution
    print(f"  Keypoint degrees: {degrees.tolist()}")
    degree_counts = {}
    for d in degrees:
        degree_counts[int(d)] = degree_counts.get(int(d), 0) + 1
    print(f"  Degree distribution: {degree_counts}")
    print(f"  Valid edges found: {len(edges)}")

    # Compute spacing stats after relaxation (based on valid edges)
    if relax_debug:
        init_uniformity = relax_debug.get("initial_uniformity", 0)
        final_uniformity = relax_debug.get("final_uniformity", 0)
        init_min = relax_debug.get("initial_min_edge", 0)
        init_max = relax_debug.get("initial_max_edge", 0)
        final_min = relax_debug.get("final_min_edge", 0)
        final_max = relax_debug.get("final_max_edge", 0)
        
        print(f"  Edge length stats (valid neighbors only):")
        print(f"    Initial: min={init_min:.2f}, max={init_max:.2f}, uniformity={init_uniformity:.3f}")
        print(f"    Final:   min={final_min:.2f}, max={final_max:.2f}, uniformity={final_uniformity:.3f}")
        
        if init_uniformity > 0:
            uniformity_improvement = (final_uniformity - init_uniformity) / init_uniformity * 100
            print(f"  Uniformity improvement: {init_uniformity:.3f} → {final_uniformity:.3f} ({uniformity_improvement:+.1f}%)")
        
        # Graph stats
        init_graph = relax_debug.get("initial_graph", {})
        final_graph = relax_debug.get("final_graph", {})
        print(f"  Initial valid edges: {relax_debug.get('initial_n_edges', 0)} (candidate pairs: {init_graph.get('n_candidate_pairs', 0)}, checked: {init_graph.get('n_checked', 0)}, valid: {init_graph.get('n_valid_edges', 0)})")
        print(f"  Final valid edges: {relax_debug.get('final_n_edges', 0)}")

    total_time = time.time() - start_time
    print(f"Total keypoint extraction time: {total_time:.3f}s")
    print(f"Uniform keypoints extracted: {keypoints.shape}")

    # Convert edges to format for visualization
    wire_edges = [(int(e[0]), int(e[1])) for e in edges]

    gmm_visual_path = "/home/yehengz/deformable_seg/point_cloud_gmm_constrained_wires.html"
    gmm_stub = SimpleNamespace(
        keypoints=keypoints,
        covariances=np.full((keypoints.shape[0], 3), 10.0),  # diagonal covariances (K, 3)
        metadata={
            "covariance_type": "diag",
            "fps_stats": fps_stats,
            "degrees": degrees.tolist(),
        },
        weights=np.ones(keypoints.shape[0]) / keypoints.shape[0],
        filtered_points=foreground_points,
    )
    visualize_foreground_background_gmm(
        points,
        colors,
        foreground_mask_points,
        keypoints,
        gmm_visual_path,
        point_size=2,
        keypoint_size=6,
        max_background_points=60_000,
        max_foreground_points=30_000,
        percentile_clip=99.0,
        gmm_result=gmm_stub,
        downsampled_points=fps_keypoints,
        graph_edges=wire_edges,
        graph_edge_color="rgba(255, 215, 0, 0.9)",
        graph_edge_width=6,
        foreground_color="rgba(148, 0, 211, 0.9)",
        downsampled_color="rgba(0, 120, 255, 0.9)",
    )
    print(f"Saved uniform keypoints visualization: {gmm_visual_path}")

    
    # create 2D mask visualization (requires skeleton code to be enabled)
    print("Creating 2D mask visualization...")
    mask_save_path = "/home/yehengz/deformable_seg/pc_mask_skeleton.png"
    create_skel_mask_viz(
        rgb_image,
        pc_mask,
        skeleton_pc_mask,
        branch_nodes,
        end_nodes,
        save_path=mask_save_path,
    )
    print(f"Saved 2D mask visualization: {mask_save_path}")

    print("Demo complete!")