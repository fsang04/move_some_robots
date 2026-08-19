"""
Repulsion-based relaxation utilities for wire/skeleton keypoint spacing.

For each node:
1. Use KNN to find neighbors
2. Remove neighbors that don't have a valid path in the 2D skeleton
3. Valid path = exists in skeleton AND does not pass through any other keypoint node
"""

from __future__ import annotations

import math
import heapq
import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
from typing import Optional, Dict, Any, Tuple, List, Set


def build_skeleton_graph(
    skeleton_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], int], List[List[Tuple[int, float]]]]:
    """
    Build a graph representation of the skeleton mask.
    
    Returns
    -------
    coords : np.ndarray, shape (N, 2)
        Pixel coordinates (row, col) for each node.
    coord_to_idx : dict
        Mapping from (row, col) tuple to node index.
    adjacency : list of list of (neighbor_idx, distance)
        Adjacency list representation.
    """
    skeleton_coords = np.argwhere(skeleton_mask > 0)
    n_nodes = skeleton_coords.shape[0]
    
    if n_nodes == 0:
        return np.empty((0, 2), dtype=np.int64), {}, []
    
    coord_to_idx = {tuple(coord.tolist()): idx for idx, coord in enumerate(skeleton_coords)}
    
    adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(n_nodes)]
    neighbor_offsets = [
        (-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
        (0, -1, 1.0),                          (0, 1, 1.0),
        (1, -1, math.sqrt(2)),  (1, 0, 1.0),  (1, 1, math.sqrt(2)),
    ]
    
    for idx, (row, col) in enumerate(skeleton_coords):
        for dr, dc, dist in neighbor_offsets:
            neighbor = (row + dr, col + dc)
            neighbor_idx = coord_to_idx.get(neighbor)
            if neighbor_idx is not None:
                adjacency[idx].append((neighbor_idx, dist))
    
    return skeleton_coords, coord_to_idx, adjacency


def project_3d_to_pixel(
    point_3d: np.ndarray,
    intrinsics: np.ndarray,
) -> Tuple[int, int]:
    """Project a 3D point to pixel coordinates."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x, y, z = point_3d
    if z <= 1e-6:
        return -1, -1
    
    col = int(round(x * fx / z + cx))
    row = int(round(y * fy / z + cy))
    return row, col


def snap_to_skeleton(
    pixel: Tuple[int, int],
    skeleton_coords: np.ndarray,
    skeleton_tree: Optional[KDTree] = None,
) -> int:
    """Find the nearest skeleton pixel to a given pixel."""
    if skeleton_coords.shape[0] == 0:
        return -1
    
    row, col = pixel
    if skeleton_tree is not None:
        _, idx = skeleton_tree.query([[row, col]], k=1)
        return int(np.asarray(idx).reshape(-1)[0])
    else:
        diffs = skeleton_coords - np.array([row, col])
        dists_sq = np.sum(diffs * diffs, axis=1)
        return int(np.argmin(dists_sq))


def skeleton_path_exists_no_blocked(
    adjacency: List[List[Tuple[int, float]]],
    start: int,
    goal: int,
    blocked_skeleton_indices: Set[int],
) -> bool:
    """
    Check if a path exists between two skeleton nodes without passing through blocked nodes.
    
    Parameters
    ----------
    adjacency : list of list of (neighbor_idx, distance)
        Skeleton graph adjacency.
    start : int
        Start skeleton node index.
    goal : int
        Goal skeleton node index.
    blocked_skeleton_indices : set of int
        Skeleton node indices that are blocked (other keypoint locations).
        The start and goal nodes themselves should NOT be in this set.
    
    Returns
    -------
    bool
        True if path exists without passing through blocked nodes.
    """
    if start == goal:
        return True
    if start < 0 or goal < 0:
        return False
    
    # BFS to find if path exists
    visited: Set[int] = set()
    queue = [start]
    visited.add(start)
    
    while queue:
        node = queue.pop(0)
        
        for nbr, _ in adjacency[node]:
            if nbr == goal:
                return True
            if nbr in visited:
                continue
            if nbr in blocked_skeleton_indices:
                continue  # Cannot pass through blocked nodes
            visited.add(nbr)
            queue.append(nbr)
    
    return False


def find_valid_neighbors_knn(
    keypoints: np.ndarray,
    skeleton_mask: np.ndarray,
    intrinsics: np.ndarray,
    k_neighbors: int = 6,
    block_radius: int = 2,
) -> Tuple[List[List[int]], np.ndarray, Dict[str, Any]]:
    """
    For each keypoint, find KNN neighbors, then filter to only keep those
    with a valid skeleton path that doesn't pass through other keypoints.
    
    Parameters
    ----------
    keypoints : np.ndarray, shape (K, 3)
        Keypoint positions in 3D.
    skeleton_mask : np.ndarray, shape (H, W)
        Binary skeleton mask.
    intrinsics : np.ndarray, shape (3, 3)
        Camera intrinsics.
    k_neighbors : int
        Number of KNN neighbors to consider initially.
    block_radius : int
        Pixel radius around each keypoint to block in skeleton path search.
    
    Returns
    -------
    valid_neighbors : list of list of int
        For each keypoint, list of valid neighbor indices.
    degrees : np.ndarray, shape (K,)
        Number of valid neighbors for each keypoint.
    debug_info : dict
        Debug information.
    """
    K = keypoints.shape[0]
    
    if K < 2:
        return [[] for _ in range(K)], np.zeros(K, dtype=int), {}
    
    # Build skeleton graph
    skeleton_coords, coord_to_idx, adjacency = build_skeleton_graph(skeleton_mask)
    
    if skeleton_coords.shape[0] == 0:
        return [[] for _ in range(K)], np.zeros(K, dtype=int), {"error": "empty_skeleton"}
    
    # Build KDTree for skeleton
    skeleton_tree = KDTree(skeleton_coords) if skeleton_coords.shape[0] > 1 else None
    
    # Project each keypoint to its nearest skeleton pixel
    keypoint_skeleton_indices = []
    H, W = skeleton_mask.shape
    for i in range(K):
        pixel = project_3d_to_pixel(keypoints[i], intrinsics)
        if pixel[0] < 0 or pixel[1] < 0:
            keypoint_skeleton_indices.append(-1)
        else:
            row = max(0, min(pixel[0], H - 1))
            col = max(0, min(pixel[1], W - 1))
            skel_idx = snap_to_skeleton((row, col), skeleton_coords, skeleton_tree)
            keypoint_skeleton_indices.append(skel_idx)
    
    keypoint_skeleton_indices = np.array(keypoint_skeleton_indices, dtype=int)
    
    # Pre-compute blocked regions for each keypoint (skeleton indices within block_radius)
    skeleton_coords_float = skeleton_coords.astype(np.float64)
    keypoint_blocked_regions: List[Set[int]] = []
    for i in range(K):
        skel_idx = keypoint_skeleton_indices[i]
        if skel_idx < 0:
            keypoint_blocked_regions.append(set())
            continue
        
        center = skeleton_coords_float[skel_idx]
        # Find all skeleton pixels within block_radius
        diffs = skeleton_coords_float - center
        dists_sq = np.sum(diffs * diffs, axis=1)
        within_radius = np.where(dists_sq <= block_radius * block_radius)[0]
        keypoint_blocked_regions.append(set(within_radius.tolist()))
    
    # KNN in 3D space
    k = min(k_neighbors + 1, K)  # +1 because query includes self
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(keypoints)
    _, knn_indices = nn.kneighbors(keypoints)
    
    # Pre-compute the union of ALL blocked regions
    all_blocked = set()
    for region in keypoint_blocked_regions:
        all_blocked.update(region)
    
    # Collect candidate pairs (avoid checking both i→j and j→i)
    candidate_pairs = set()
    for i in range(K):
        knn_nbrs = knn_indices[i, 1:]  # Exclude self
        for j in knn_nbrs:
            if j != i:
                candidate_pairs.add((min(i, j), max(i, j)))
    
    # For each candidate pair, check if valid path exists
    valid_edges: List[Tuple[int, int]] = []
    n_checked = 0
    
    for i, j in candidate_pairs:
        skel_i = keypoint_skeleton_indices[i]
        skel_j = keypoint_skeleton_indices[j]
        
        if skel_i < 0 or skel_j < 0:
            continue
        
        n_checked += 1
        
        # Blocked = all blocked regions EXCEPT those of i and j
        blocked = all_blocked - keypoint_blocked_regions[i] - keypoint_blocked_regions[j]
        
        # Check if path exists from i to j without passing through blocked nodes
        if skeleton_path_exists_no_blocked(adjacency, skel_i, skel_j, blocked):
            valid_edges.append((i, j))
    
    # Build neighbor lists from valid edges
    valid_neighbors: List[List[int]] = [[] for _ in range(K)]
    degrees = np.zeros(K, dtype=int)
    
    for i, j in valid_edges:
        valid_neighbors[i].append(j)
        valid_neighbors[j].append(i)
        degrees[i] += 1
        degrees[j] += 1
    
    debug_info = {
        "n_keypoints": K,
        "n_skeleton_pixels": skeleton_coords.shape[0],
        "n_candidate_pairs": len(candidate_pairs),
        "n_checked": n_checked,
        "n_valid_edges": len(valid_edges),
    }
    
    return valid_neighbors, degrees, debug_info


def compute_repulsion_forces(
    keypoints: np.ndarray,
    valid_neighbors: List[List[int]],
    target_edge_length: Optional[float] = None,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Compute repulsion/attraction forces between valid neighbors to equalize spacing.
    
    Parameters
    ----------
    keypoints : np.ndarray, shape (K, 3)
        Keypoint positions.
    valid_neighbors : list of list of int
        For each keypoint, list of valid neighbor indices.
    target_edge_length : float, optional
        Target distance between neighbors. If None, use mean current distance.
    epsilon : float
        Numerical stability.
    
    Returns
    -------
    forces : np.ndarray, shape (K, 3)
        Force vector for each keypoint.
    """
    K = keypoints.shape[0]
    forces = np.zeros((K, 3), dtype=np.float64)
    
    # Collect all unique edges and their lengths
    edges_set = set()
    for i in range(K):
        for j in valid_neighbors[i]:
            edge = (min(i, j), max(i, j))
            edges_set.add(edge)
    
    if not edges_set:
        return forces
    
    # Compute current edge lengths
    edge_lengths = []
    for i, j in edges_set:
        d = np.linalg.norm(keypoints[i] - keypoints[j])
        edge_lengths.append(d)
    
    if target_edge_length is None:
        target_edge_length = np.mean(edge_lengths)
    
    # Compute forces for each keypoint from its valid neighbors
    for i in range(K):
        for j in valid_neighbors[i]:
            v = keypoints[i] - keypoints[j]  # Vector from j to i
            d = np.linalg.norm(v)
            
            if d < epsilon:
                v = np.random.randn(3)
                d = np.linalg.norm(v) + epsilon
            
            unit_v = v / d
            
            # Spring-like force: push if too close, pull if too far
            force_mag = (target_edge_length - d) / target_edge_length
            forces[i] += force_mag * unit_v
    
    return forces


def project_to_cloud(
    keypoints: np.ndarray,
    point_cloud: np.ndarray,
    nn_index: Optional[NearestNeighbors] = None,
) -> np.ndarray:
    """Project each keypoint to nearest point in cloud."""
    if nn_index is None:
        nn_index = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn_index.fit(point_cloud)
    
    _, indices = nn_index.kneighbors(keypoints)
    return point_cloud[indices.flatten()]


def repulsion_relaxation_wire(
    keypoints: np.ndarray,
    point_cloud: np.ndarray,
    skeleton_mask: np.ndarray,
    intrinsics: np.ndarray,
    *,
    fixed_mask: Optional[np.ndarray] = None,
    n_iterations: int = 50,
    learning_rate: float = 3.0,
    k_neighbors: int = 6,
    block_radius: int = 2,
    target_edge_length: Optional[float] = None,
    epsilon: float = 1e-8,
    project_each_step: bool = True,
    rebuild_neighbors_every: int = 10,
    return_debug: bool = False,
    validate_edges: bool = True,
) -> Dict[str, Any]:
    """
    Apply repulsion-based relaxation using KNN + skeleton path validation.
    
    For each node:
    1. Use KNN to find neighbors
    2. Remove neighbors without valid skeleton path (path must exist and not pass through other nodes)
    3. Apply spring forces between valid neighbors
    
    Parameters
    ----------
    keypoints : np.ndarray, shape (K, 3)
        Initial keypoint positions.
    point_cloud : np.ndarray, shape (N, 3)
        Point cloud for projection.
    skeleton_mask : np.ndarray, shape (H, W)
        Binary skeleton mask.
    intrinsics : np.ndarray, shape (3, 3)
        Camera intrinsics.
    fixed_mask : np.ndarray, shape (K,), optional
        Boolean mask indicating which keypoints should stay fixed (not move).
        True = fixed, False = can move.
    n_iterations : int
        Number of relaxation iterations.
    learning_rate : float
        Step size multiplier.
    k_neighbors : int
        Number of KNN neighbors to consider.
    target_edge_length : float, optional
        Target distance between neighbors. If None, computed from initial mean.
    epsilon : float
        Numerical stability.
    project_each_step : bool
        If True, project to cloud after each step.
    rebuild_neighbors_every : int
        Rebuild neighbor graph every N iterations.
    return_debug : bool
        If True, return debug information.
    validate_edges : bool
        If True, validate edges using skeleton path. If False, use simple KNN
        for repulsion forces (for tracking mode where topology is fixed).
    
    Returns
    -------
    result : dict
        - "keypoints": relaxed positions
        - "degrees": final degree for each keypoint
        - "edges": list of (i, j) valid edges
        - "debug": debug info (if requested)
    """
    K = keypoints.shape[0]
    if K < 2:
        return {
            "keypoints": keypoints.copy(),
            "degrees": np.zeros(K, dtype=int),
            "edges": [],
            "debug": None,
        }
    
    # Handle fixed mask
    if fixed_mask is None:
        fixed_mask = np.zeros(K, dtype=bool)
    else:
        fixed_mask = np.asarray(fixed_mask, dtype=bool)
    
    n_fixed = np.sum(fixed_mask)
    print(f"  Fixed keypoints (branch nodes): {n_fixed} / {K}")
    
    # NN index for projection
    cloud_nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    cloud_nn.fit(point_cloud)
    
    mu = keypoints.copy().astype(np.float64)
    
    # Initial neighbor graph
    if validate_edges:
        valid_neighbors, degrees, graph_debug = find_valid_neighbors_knn(
            mu, skeleton_mask, intrinsics, k_neighbors=k_neighbors, block_radius=block_radius
        )
    else:
        # Simple KNN without skeleton path validation (for tracking mode)
        knn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, K), algorithm="auto")
        knn.fit(mu)
        _, knn_indices = knn.kneighbors(mu)
        valid_neighbors = [list(knn_indices[i, 1:]) for i in range(K)]  # Exclude self
        degrees = np.array([len(v) for v in valid_neighbors], dtype=int)
        graph_debug = {"n_candidate_pairs": 0, "n_checked": 0, "n_valid_edges": 0}
    
    # Compute target edge length from initial valid edges
    if target_edge_length is None:
        edges_set = set()
        for i in range(K):
            for j in valid_neighbors[i]:
                edges_set.add((min(i, j), max(i, j)))
        edge_lengths = [np.linalg.norm(mu[i] - mu[j]) for i, j in edges_set]
        target_edge_length = np.mean(edge_lengths) if edge_lengths else 50.0
    
    # Debug info
    debug_info = None
    if return_debug:
        edges_set = set()
        for i in range(K):
            for j in valid_neighbors[i]:
                edges_set.add((min(i, j), max(i, j)))
        edge_lengths = [np.linalg.norm(mu[i] - mu[j]) for i, j in edges_set]
        
        debug_info = {
            "initial_graph": graph_debug,
            "target_edge_length": target_edge_length,
            "initial_n_edges": len(edges_set),
        }
        if edge_lengths:
            debug_info["initial_min_edge"] = float(np.min(edge_lengths))
            debug_info["initial_max_edge"] = float(np.max(edge_lengths))
            debug_info["initial_mean_edge"] = float(np.mean(edge_lengths))
            debug_info["initial_uniformity"] = float(np.min(edge_lengths) / np.max(edge_lengths)) if np.max(edge_lengths) > 0 else 1.0
    
    # Relaxation iterations
    for iteration in range(n_iterations):
        # Optionally rebuild neighbor graph
        if iteration > 0 and rebuild_neighbors_every > 0 and iteration % rebuild_neighbors_every == 0:
            if validate_edges:
                valid_neighbors, degrees, _ = find_valid_neighbors_knn(
                    mu, skeleton_mask, intrinsics, k_neighbors=k_neighbors, block_radius=block_radius
                )
            else:
                # Simple KNN without skeleton path validation
                knn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, K), algorithm="auto")
                knn.fit(mu)
                _, knn_indices = knn.kneighbors(mu)
                valid_neighbors = [list(knn_indices[i, 1:]) for i in range(K)]
                degrees = np.array([len(v) for v in valid_neighbors], dtype=int)
        
        # Compute forces
        forces = compute_repulsion_forces(
            mu, valid_neighbors,
            target_edge_length=target_edge_length,
            epsilon=epsilon,
        )
        
        # Zero out forces for fixed keypoints
        forces[fixed_mask] = 0.0
        
        # Normalize and apply
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        max_norm = np.max(force_norms)
        if max_norm > epsilon:
            forces = forces / max_norm * learning_rate
        
        mu = mu + forces
        
        # Project to cloud (only for non-fixed keypoints)
        if project_each_step:
            movable_mask = ~fixed_mask
            if np.any(movable_mask):
                mu_movable = mu[movable_mask]
                mu_movable_proj = project_to_cloud(mu_movable, point_cloud, nn_index=cloud_nn)
                mu[movable_mask] = mu_movable_proj
    
    # Final projection (only for non-fixed keypoints)
    movable_mask = ~fixed_mask
    if np.any(movable_mask):
        mu_movable = mu[movable_mask]
        mu_movable_proj = project_to_cloud(mu_movable, point_cloud, nn_index=cloud_nn)
        mu[movable_mask] = mu_movable_proj
    
    # Final neighbor graph (only compute if validate_edges is True)
    if validate_edges:
        valid_neighbors, degrees, final_graph_debug = find_valid_neighbors_knn(
            mu, skeleton_mask, intrinsics, k_neighbors=k_neighbors, block_radius=block_radius
        )
        
        # Extract unique edges
        edges_set = set()
        for i in range(K):
            for j in valid_neighbors[i]:
                edges_set.add((min(i, j), max(i, j)))
        edges = list(edges_set)
    else:
        # Don't compute edges - they will be provided from first frame
        edges = []
        final_graph_debug = {}
    
    # Final debug info
    if return_debug:
        edges_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
        edge_lengths = [np.linalg.norm(mu[i] - mu[j]) for i, j in edges_set]
        debug_info["final_graph"] = final_graph_debug
        debug_info["final_n_edges"] = len(edges_set)
        if edge_lengths:
            debug_info["final_min_edge"] = float(np.min(edge_lengths))
            debug_info["final_max_edge"] = float(np.max(edge_lengths))
            debug_info["final_mean_edge"] = float(np.mean(edge_lengths))
            debug_info["final_uniformity"] = float(np.min(edge_lengths) / np.max(edge_lengths)) if np.max(edge_lengths) > 0 else 1.0
    
    return {
        "keypoints": mu,
        "degrees": degrees,
        "edges": edges,
        "debug": debug_info,
    }


def compute_spacing_stats(keypoints: np.ndarray, n_neighbors: int = 1) -> Dict[str, float]:
    """Compute statistics about keypoint spacing (Euclidean nearest neighbor)."""
    if keypoints.shape[0] < 2:
        return {
            "min_dist": 0.0,
            "max_dist": 0.0,
            "mean_dist": 0.0,
            "std_dist": 0.0,
            "uniformity": 1.0,
        }
    
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="auto")
    nn.fit(keypoints)
    dists, _ = nn.kneighbors(keypoints)
    
    min_dists = dists[:, 1]  # Exclude self
    
    min_dist = float(np.min(min_dists))
    max_dist = float(np.max(min_dists))
    
    return {
        "min_dist": min_dist,
        "max_dist": max_dist,
        "mean_dist": float(np.mean(min_dists)),
        "std_dist": float(np.std(min_dists)),
        "uniformity": min_dist / max_dist if max_dist > 0 else 1.0,
    }
