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
    compute_spacing_stats,
)


def build_bpa_mesh(keypoints, radii=None):
    """
    Build Ball-Pivoting Algorithm mesh on keypoints and extract edges.
    
    Parameters
    ----------
    keypoints : np.ndarray, shape (K, 3)
        Keypoint positions.
    radii : list of float, optional
        Ball radii for BPA. If None, auto-computed from point spacing.
    
    Returns
    -------
    edges : list of (int, int)
        List of edges (i, j) where i < j.
    triangles : list of (int, int, int)
        List of triangle vertex indices.
    """
    from sklearn.neighbors import NearestNeighbors
    
    K = keypoints.shape[0]
    if K < 3:
        return [], []
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(keypoints)
    
    # Estimate normals (required for BPA)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=10)
    
    # Auto-compute radii if not provided
    if radii is None:
        nn = NearestNeighbors(n_neighbors=min(4, K), algorithm="auto")
        nn.fit(keypoints)
        dists, _ = nn.kneighbors(keypoints)
        mean_dist = np.mean(dists[:, 1:])
        radii = [mean_dist * 0.8, mean_dist * 1.0, mean_dist * 1.5, mean_dist * 2.0]
        radii = [mean_dist * 1.0, mean_dist * 1.0, mean_dist * 1.0, mean_dist * 1.0]
    
    # Run BPA
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    
    # Extract triangles and edges
    triangles = np.asarray(mesh.triangles).tolist()
    edges_set = set()
    
    for tri in triangles:
        i, j, k = tri
        edges_set.add((min(i, j), max(i, j)))
        edges_set.add((min(j, k), max(j, k)))
        edges_set.add((min(i, k), max(i, k)))
    
    edges = list(edges_set)
    return edges, triangles


def build_delaunay_mesh(keypoints: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]]]:
    """
    Build triangle mesh using Delaunay triangulation with PCA projection.
    
    For 2D surfaces (like towels/fabrics), this is more robust than BPA:
    - No parameter tuning needed
    - Always produces valid triangulation
    - Fast O(n log n)
    
    Algorithm:
    1. PCA to find tangent plane (v1, v2 are in-plane axes)
    2. Project 3D points to 2D
    3. Run Delaunay triangulation in 2D
    4. Map triangles back to original 3D point indices
    
    Parameters
    ----------
    keypoints : np.ndarray, shape (K, 3)
        Keypoint positions in 3D.
    
    Returns
    -------
    edges : list of (int, int)
        List of edges (i, j) where i < j.
    triangles : list of (int, int, int)
        List of triangle vertex indices.
    """
    from scipy.spatial import Delaunay
    
    K = keypoints.shape[0]
    if K < 3:
        return [], []
    
    # Step 1: PCA to find tangent plane
    centroid = np.mean(keypoints, axis=0)
    centered = keypoints - centroid
    
    # Covariance matrix
    cov = (centered.T @ centered) / K
    
    # Eigendecomposition (sorted by eigenvalue descending)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # v1, v2 are in-plane axes (largest eigenvalues)
    v1 = eigenvectors[:, 0]
    v2 = eigenvectors[:, 1]
    
    # Step 2: Project to 2D
    coords_2d = np.column_stack([
        centered @ v1,
        centered @ v2,
    ])
    
    # Step 3: Delaunay triangulation in 2D
    try:
        tri = Delaunay(coords_2d)
    except Exception as e:
        print(f"  Warning: Delaunay failed ({e}), returning empty mesh")
        return [], []
    
    # Step 4: Extract triangles and edges
    triangles = [tuple(simplex) for simplex in tri.simplices]
    
    edges_set = set()
    for simplex in tri.simplices:
        i, j, k = simplex
        edges_set.add((min(i, j), max(i, j)))
        edges_set.add((min(j, k), max(j, k)))
        edges_set.add((min(i, k), max(i, k)))
    
    edges = list(edges_set)
    return edges, triangles


def build_grid_mesh(keypoints: np.ndarray, angle_tolerance_deg: float = 45.0) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]]]:
    """
    Build grid-like mesh by connecting neighbors along principal axes.
    
    Creates rectangle + X pattern by finding nearest neighbors in 4 directions
    (along v1, -v1, v2, -v2) plus diagonals.
    
    Algorithm:
    1. PCA to find principal axes (v1, v2)
    2. For each keypoint, find nearest neighbor in each of 8 directions
    3. Connect to create grid-like pattern with diagonals
    
    Parameters
    ----------
    keypoints : np.ndarray, shape (K, 3)
        Keypoint positions in 3D.
    angle_tolerance_deg : float
        Maximum angle deviation from axis direction to consider a neighbor.
    
    Returns
    -------
    edges : list of (int, int)
        List of edges (i, j) where i < j.
    triangles : list of (int, int, int)
        List of triangle vertex indices.
    """
    from sklearn.neighbors import NearestNeighbors
    
    K = keypoints.shape[0]
    if K < 3:
        return [], []
    
    # Step 1: PCA to find principal axes
    centroid = np.mean(keypoints, axis=0)
    centered = keypoints - centroid
    
    cov = (centered.T @ centered) / K
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    v1 = eigenvectors[:, 0]  # Primary axis
    v2 = eigenvectors[:, 1]  # Secondary axis
    
    # 8 directions: ±v1, ±v2, and 4 diagonals
    directions = [
        v1,                          # right
        -v1,                         # left
        v2,                          # up
        -v2,                         # down
        (v1 + v2) / np.linalg.norm(v1 + v2),   # diagonal up-right
        (v1 - v2) / np.linalg.norm(v1 - v2),   # diagonal down-right
        (-v1 + v2) / np.linalg.norm(-v1 + v2), # diagonal up-left
        (-v1 - v2) / np.linalg.norm(-v1 - v2), # diagonal down-left
    ]
    
    angle_tolerance_rad = np.deg2rad(angle_tolerance_deg)
    
    # Step 2: Build KNN for fast neighbor lookup
    nn = NearestNeighbors(n_neighbors=min(20, K), algorithm="auto")
    nn.fit(keypoints)
    distances, indices = nn.kneighbors(keypoints)
    
    # Step 3: For each keypoint, find best neighbor in each direction
    edges_set = set()
    
    for i in range(K):
        for direction in directions:
            best_neighbor = -1
            best_dist = float('inf')
            
            # Check all KNN neighbors
            for j_idx in range(1, len(indices[i])):  # skip self (index 0)
                j = indices[i][j_idx]
                
                # Vector from i to j
                vec_ij = keypoints[j] - keypoints[i]
                dist_ij = np.linalg.norm(vec_ij)
                
                if dist_ij < 1e-8:
                    continue
                
                # Check angle with target direction
                cos_angle = (vec_ij @ direction) / dist_ij
                
                # Must be in positive direction (cos > 0) and within tolerance
                if cos_angle > np.cos(angle_tolerance_rad) and dist_ij < best_dist:
                    best_dist = dist_ij
                    best_neighbor = j
            
            # Add edge to best neighbor in this direction
            if best_neighbor >= 0:
                edges_set.add((min(i, best_neighbor), max(i, best_neighbor)))
    
    edges = list(edges_set)
    
    # Step 4: Build triangles from edges (find triangular faces)
    # Create adjacency for efficient triangle detection
    adjacency = {i: set() for i in range(K)}
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    
    triangles = []
    seen_triangles = set()
    
    for i, j in edges:
        # Find common neighbors to form triangles
        common = adjacency[i] & adjacency[j]
        for k in common:
            tri = tuple(sorted([i, j, k]))
            if tri not in seen_triangles:
                seen_triangles.add(tri)
                triangles.append(tri)
    
    print(f"  Principal axes found via PCA")
    print(f"  Edges: {len(edges)}, Triangles: {len(triangles)}")
    
    return edges, triangles


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
    data = np.load("/home/yehengz/deformable_seg/data/full/test_colored_fabric.npy", allow_pickle=True).item()
    # data = np.load("/home/yehengz/deformable_seg/data/full/test_wire.npy", allow_pickle=True).item()
    sampled_points = data[0]

    bg_data = np.load("/home/yehengz/deformable_seg/data/bg/test_fabric_bg.npy", allow_pickle=True).item()
    # bg_data = np.load("/home/yehengz/deformable_seg/data/bg/test_wire_bg.npy", allow_pickle=True).item()
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

    # pc_mask = remove_small_components(pc_mask, min_size=800)


    # skeleton_pc_mask = skelentonize(pc_mask)
    # branch_nodes, end_nodes, adjacency, coords = node_identification(
    #     skeleton_pc_mask,
    #     return_graph=True,
    # )

    # expected_leaf_nodes = 4
    # if adjacency is not None and coords is not None:
    #     pruning_result = prune_leaf_segments(
    #         adjacency,
    #         coords,
    #         expected_num_leaf_nodes=expected_leaf_nodes,
    #     )
    #     branch_nodes = pruning_result["branch_coords"]
    #     end_nodes = pruning_result["leaf_coords"]
    #     skeleton_pc_mask = mask_from_mst(
    #         pruning_result["adjacency"],
    #         pruning_result["coords"],
    #         skeleton_pc_mask.shape,
    #     )
    #     pc_mask = np.where(skeleton_pc_mask > 0, pc_mask, 0).astype(np.uint8)

    # print(branch_nodes.shape, end_nodes.shape)

    depth_data = sampled_points['transformed_depth']
    points, colors, valid_mask = create_color_point_cloud(
        rgb_image,
        depth_data,
        intrinsics,
        return_valid_mask=True,
    )

    valid_flat = valid_mask.reshape(-1)
    valid_indices = np.flatnonzero(valid_flat)
    pc_mask_flat = (pc_mask > 0).reshape(-1)
    if valid_indices.size:
        foreground_mask_points = pc_mask_flat[valid_indices].astype(bool)
    else:
        foreground_mask_points = np.zeros(points.shape[0], dtype=bool)

    foreground_points = points[foreground_mask_points]
    foreground_colors = colors[foreground_mask_points]

    foreground_points, foreground_colors = filter_point_cloud_radius(
        foreground_points, 
        foreground_colors, 
        radius=20.0, 
        min_neighbors=100
    )


    ds_fg_points, ds_fg_colors = foreground_points, foreground_colors

    if foreground_points.shape[0] < 20:
        raise ValueError("Not enough foreground points to fit constrained GMM")

    print(
        f"Foreground points available for keypoint extraction: {foreground_points.shape[0]:,} (from {points.shape[0]:,} valid points)"
    )

    # ============================================================
    # Stage 1: Farthest Point Sampling (FPS) initialization
    # ============================================================
    n_keypoints = 25

    print(f"Stage 1 (FPS): Initializing {n_keypoints} keypoints...")
    start_time = time.time()

    # FPS: pick starting point farthest from centroid for stability
    centroid = np.mean(ds_fg_points, axis=0)
    dists_to_centroid = np.linalg.norm(ds_fg_points - centroid, axis=1)
    start_idx = int(np.argmax(dists_to_centroid))

    # Initialize chosen array and distance array
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

    # Compute spacing stats after FPS
    fps_stats = compute_spacing_stats(fps_keypoints, n_neighbors=1)
    print(
        f"  FPS spacing → min: {fps_stats['min_dist']:.2f}, "
        f"max: {fps_stats['max_dist']:.2f}, "
        f"uniformity: {fps_stats['uniformity']:.3f}"
    )

    # ============================================================
    # Stage 2: Repulsion-based relaxation (KNN-based, no BPA)
    # ============================================================
    print("Stage 2 (Repulsion): Relaxing keypoint positions...")
    relax_start = time.time()
    relaxation_result = repulsion_relaxation(
        fps_keypoints,
        foreground_points,  # Project to FULL cloud
        n_neighbors=2,
        n_iterations=200,
        learning_rate=0.5,
        epsilon=1e-8,
        project_each_step=True,
        return_debug=True,
    )
    keypoints = relaxation_result["keypoints"]
    relax_debug = relaxation_result.get("debug")
    relax_time = time.time() - relax_start
    print(f"  Repulsion relaxation completed in {relax_time:.3f}s")

    # Compute spacing stats after relaxation
    final_stats = compute_spacing_stats(keypoints, n_neighbors=1)
    print(
        f"  Relaxed spacing → min: {final_stats['min_dist']:.2f}, "
        f"max: {final_stats['max_dist']:.2f}, "
        f"uniformity: {final_stats['uniformity']:.3f}"
    )

    if relax_debug:
        init_min = relax_debug.get("initial_min_dist", 0)
        final_min = relax_debug.get("final_min_dist", 0)
        if init_min > 0:
            improvement = (final_min - init_min) / init_min * 100
            print(f"  Min-dist improvement: {init_min:.2f} → {final_min:.2f} ({improvement:+.1f}%)")

    # ============================================================
    # Stage 3: Delaunay Triangle Mesh
    # ============================================================
    print("Stage 3 (Delaunay): Building triangle mesh on relaxed keypoints...")
    mesh_start = time.time()
    edges, triangles = build_delaunay_mesh(keypoints)
    mesh_time = time.time() - mesh_start
    print(f"  Delaunay mesh completed in {mesh_time:.3f}s")
    print(f"  Triangles: {len(triangles)}, Edges: {len(edges)}")
    
    # Compute degrees from edges
    degrees = np.zeros(keypoints.shape[0], dtype=int)
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1
    
    print(f"  Keypoint degrees: {degrees.tolist()}")
    degree_counts = {}
    for d in degrees:
        degree_counts[int(d)] = degree_counts.get(int(d), 0) + 1
    print(f"  Degree distribution: {degree_counts}")

    total_time = time.time() - start_time
    print(f"Total keypoint extraction time: {total_time:.3f}s")
    print(f"Uniform keypoints extracted: {keypoints.shape}")

    # Convert edges to format for visualization
    fabric_edges = [(int(e[0]), int(e[1])) for e in edges]

    gmm_visual_path = "/home/yehengz/deformable_seg/point_cloud_gmm_constrained_fabrics.html"
    gmm_stub = SimpleNamespace(
        keypoints=keypoints,
        covariances=np.full((keypoints.shape[0], 3), 10.0),  # diagonal covariances (K, 3)
        metadata={
            "covariance_type": "diag",
            "fps_stats": fps_stats,
            "relaxed_stats": final_stats,
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
        graph_edges=fabric_edges,
        graph_edge_color="rgba(255, 215, 0, 0.9)",
        graph_edge_width=6,
        foreground_color="rgba(148, 0, 211, 0.9)",
        downsampled_color="rgba(0, 120, 255, 0.9)",
    )
    print(f"Saved uniform keypoints visualization: {gmm_visual_path}")

    
    # create 2D mask visualization (requires skeleton code to be enabled)
    # print("Creating 2D mask visualization...")
    # mask_save_path = "/home/yehengz/deformable_seg/pc_mask_skeleton.png"
    # create_skel_mask_viz(
    #     rgb_image,
    #     pc_mask,
    #     skeleton_pc_mask,
    #     branch_nodes,
    #     end_nodes,
    #     save_path=mask_save_path,
    # )
    # print(f"Saved 2D mask visualization: {mask_save_path}")

    print("Demo complete!")