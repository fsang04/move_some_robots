import numpy as np
from collections import deque
from skimage.morphology import skeletonize as skimage_skeletonize
from sklearn.cluster import DBSCAN 
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from scipy.ndimage import label



def compute_rgb_mask(bg, fg, threshold_percentile=85):
    diff = fg.astype(np.int32) - bg.astype(np.int32)
    magnitude = np.sqrt(np.sum(diff**2, axis=2))
    thres = np.percentile(magnitude, threshold_percentile)
    mask = (magnitude > thres).astype(np.uint8) * 255
    return mask

def compute_point_cloud_mask(bg_depth, fg_depth, intrinsics, distance_threshold=0.05):
    """Compute a change mask by comparing two depth frames in 3D."""

    H, W = bg_depth.shape

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    def depth_to_points(depth):
        valid = (depth > 0) & np.isfinite(depth)
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy
        x = x_norm * depth
        y = y_norm * depth
        z = depth
        return x, y, z, valid

    bg_x, bg_y, bg_z, bg_valid = depth_to_points(bg_depth)
    fg_x, fg_y, fg_z, fg_valid = depth_to_points(fg_depth)

    bg_points = np.stack([bg_x, bg_y, bg_z], axis=-1)
    fg_points = np.stack([fg_x, fg_y, fg_z], axis=-1)

    distance_3d = np.linalg.norm(fg_points - bg_points, axis=-1)

    valid_mask = bg_valid & fg_valid
    change_mask = distance_3d > distance_threshold

    mask = (valid_mask & change_mask).astype(np.uint8) * 255
    mask = change_mask.astype(np.uint8) * 255

    return mask


def refine_rgb_mask_dbscan(
    rgb_mask, # HxW
    depth_image, # HxW
    intrinsics, # camera intrinsics 3x3
    eps=40.0, # DBSCAN neighbourhood radius
    min_samples=20,
    propagation_radius=100.0,
    propagation_min_neighbors=1,
    depth_tolerance=None,
    max_iterations=5,
    min_growth=0,
    max_pixel_distance=None,
):
    """Refine an RGB mask using DBSCAN clustering and 3D distance propagation.
    Returns
    -------
    ndarray (H, W) uint8
        Refined mask (0 background, 255 foreground).
    """
    rgb_mask = np.asarray(rgb_mask)
    depth_image = np.asarray(depth_image, dtype=np.float32)
    if rgb_mask.shape != depth_image.shape:
        raise ValueError("Mask and depth image must have the same shape")

    valid_depth = (depth_image > 0) & np.isfinite(depth_image)
    mask_bool = (rgb_mask > 0) & valid_depth
    if not np.any(mask_bool):
        return np.zeros_like(rgb_mask, dtype=np.uint8)

    h, w = depth_image.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    ys, xs = np.nonzero(mask_bool)
    depth_vals = depth_image[ys, xs]
    x = (xs - cx) / fx * depth_vals
    y = (ys - cy) / fy * depth_vals
    z = depth_vals
    points = np.stack([x, y, z], axis=1)

    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(points)
    valid_clusters = labels >= 0
    if not np.any(valid_clusters):
        return np.zeros_like(rgb_mask, dtype=np.uint8)

    label_counts = np.bincount(labels[labels >= 0])
    dominant_label = np.argmax(label_counts)
    keep_mask = labels == dominant_label

    refined_mask = np.zeros((h, w), dtype=bool)
    refined_mask[ys[keep_mask], xs[keep_mask]] = True

    def lift_pixels(px, py, pz):
        return np.stack([
            (px - cx) / fx * pz,
            (py - cy) / fy * pz,
            pz,
        ], axis=1)

    if propagation_radius is None or propagation_radius <= 0:
        return refined_mask.astype(np.uint8) * 255

    valid_ys, valid_xs = np.nonzero(valid_depth)
    valid_depth_vals = depth_image[valid_ys, valid_xs]
    all_candidate_points = lift_pixels(valid_xs.astype(np.float32), valid_ys.astype(np.float32), valid_depth_vals)

    for iteration in range(max_iterations):
        seed_mask = refined_mask & valid_depth
        if not np.any(seed_mask):
            break

        seed_y, seed_x = np.nonzero(seed_mask)
        seed_depth = depth_image[seed_y, seed_x]
        seed_points = lift_pixels(seed_x.astype(np.float32), seed_y.astype(np.float32), seed_depth)

        nbrs = NearestNeighbors(radius=propagation_radius)
        nbrs.fit(seed_points)

        remaining_mask = valid_depth & (~refined_mask)
        if not np.any(remaining_mask):
            break

        remaining_indices = np.nonzero(remaining_mask)
        rem_y, rem_x = remaining_indices
        rem_depth = depth_image[rem_y, rem_x]
        rem_points = lift_pixels(rem_x.astype(np.float32), rem_y.astype(np.float32), rem_depth)

        neighbor_lists = nbrs.radius_neighbors(rem_points, return_distance=False)
        new_indices = []
        for idx, neighbors in enumerate(neighbor_lists):
            if len(neighbors) < propagation_min_neighbors:
                continue
            if depth_tolerance is not None:
                depth_diff = np.abs(seed_points[neighbors, 2] - rem_points[idx, 2])
                if depth_diff.size == 0 or np.min(depth_diff) > depth_tolerance:
                    continue
            new_indices.append(idx)

        if not new_indices:
            break

        new_indices = np.asarray(new_indices, dtype=np.int64)
        refined_mask[rem_y[new_indices], rem_x[new_indices]] = True

        if min_growth > 0 and new_indices.size < min_growth:
            break

    if max_pixel_distance is not None:
        seed_coords = np.column_stack(np.nonzero(mask_bool))
        refined_coords = np.column_stack(np.nonzero(refined_mask))
        if seed_coords.size > 0 and refined_coords.size > 0:
            coord_nbrs = NearestNeighbors(n_neighbors=1)
            coord_nbrs.fit(seed_coords.astype(np.float32))
            distances, _ = coord_nbrs.kneighbors(refined_coords.astype(np.float32), n_neighbors=1)
            far_indices = np.squeeze(distances, axis=1) > max_pixel_distance
            if np.any(far_indices):
                to_remove = refined_coords[far_indices]
                refined_mask[to_remove[:, 0], to_remove[:, 1]] = False

    return refined_mask.astype(np.uint8) * 255



def filter_pcd_mask_dbscan(
    rgb_mask,
    depth_image,
    intrinsics,
    eps=40.0,
    min_samples=20,
):
    if DBSCAN is None or NearestNeighbors is None:
        raise ImportError(
            "scikit-learn is required for refine_rgb_mask_dbscan. Install it via `pip install scikit-learn`."
        )

    rgb_mask = np.asarray(rgb_mask)
    depth_image = np.asarray(depth_image, dtype=np.float32)
    if rgb_mask.shape != depth_image.shape:
        raise ValueError("Mask and depth image must have the same shape")

    valid_depth = (depth_image > 0) & np.isfinite(depth_image)
    mask_bool = (rgb_mask > 0) & valid_depth
    if not np.any(mask_bool):
        return np.zeros_like(rgb_mask, dtype=np.uint8)

    h, w = depth_image.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    ys, xs = np.nonzero(mask_bool)
    depth_vals = depth_image[ys, xs]
    x = (xs - cx) / fx * depth_vals
    y = (ys - cy) / fy * depth_vals
    z = depth_vals
    points = np.stack([x, y, z], axis=1)

    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(points)
    valid_clusters = labels >= 0
    if not np.any(valid_clusters):
        return np.zeros_like(rgb_mask, dtype=np.uint8)

    label_counts = np.bincount(labels[labels >= 0])
    dominant_label = np.argmax(label_counts)
    keep_mask = labels == dominant_label

    refined_mask = np.zeros((h, w), dtype=bool)
    refined_mask[ys[keep_mask], xs[keep_mask]] = True

    return refined_mask.astype(np.uint8) * 255


def remove_small_components(mask, min_size=100):
    """Remove connected components smaller than min_size pixels."""
    mask = np.asarray(mask)
    binary = mask > 0
    if not np.any(binary):
        return np.zeros_like(mask, dtype=np.uint8)
    
    scale = mask.max() if mask.max() > 0 else 1
    labeled, num_components = label(binary)
    
    if num_components <= 1 or min_size <= 1:
        return (binary.astype(mask.dtype)) * scale
    
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # Ignore background
    keep_labels = np.flatnonzero(component_sizes >= min_size)
    
    if keep_labels.size == 0:
        keep_labels = np.array([component_sizes.argmax()], dtype=np.int64)
    
    filtered = np.isin(labeled, keep_labels)
    return (filtered.astype(mask.dtype)) * scale


def skelentonize(mask):
    """Skeletonize a binary mask using morphological thinning."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("Input mask must be a 2D array")
    
    binary_mask = mask > 0
    if not np.any(binary_mask):
        return np.zeros_like(mask, dtype=np.uint8)
    
    # Keep largest connected component(s)
    labeled_mask, num_components = label(binary_mask)
    if num_components > 1:
        component_sizes = np.bincount(labeled_mask.ravel())
        component_sizes[0] = 0
        largest_size = component_sizes.max()
        size_threshold = max(int(largest_size * 0.05), 64)
        keep_labels = np.flatnonzero(component_sizes >= size_threshold)
        if keep_labels.size == 0:
            keep_labels = np.array([component_sizes.argmax()], dtype=np.int64)
        binary_mask = np.isin(labeled_mask, keep_labels)
    
    skeleton = skimage_skeletonize(binary_mask)
    return skeleton.astype(np.uint8) * 255


def node_identification(skeleton_pc_mask, *, return_graph=False):
    """Identify branch and leaf nodes in a skeletonized mask via a SciPy MST."""
    mask = np.asarray(skeleton_pc_mask)
    if mask.ndim != 2:
        raise ValueError("Input mask must be a 2D array")
    
    binary_mask = mask > 0
    coords = np.column_stack(np.nonzero(binary_mask)).astype(np.int64)
    
    if coords.size == 0:
        empty = np.empty((0, 2), dtype=np.int64)
        result = (empty, empty)
        return (*result, None, None) if return_graph else result
    
    if coords.shape[0] == 1:
        empty = np.empty((0, 2), dtype=np.int64)
        result = (empty, coords.copy())
        return (*result, None, None) if return_graph else result
    
    # Build distance matrix
    dists = cdist(coords, coords, metric='euclidean')
    
    # Only keep edges for 8-connected neighbors (distance <= sqrt(2))
    adjacency = np.where(dists <= np.sqrt(2) + 1e-6, dists, 0)
    np.fill_diagonal(adjacency, 0)
    
    # Build MST
    sparse_adj = csr_matrix(adjacency)
    mst = minimum_spanning_tree(sparse_adj)
    mst_dense = mst.toarray()
    mst_symmetric = mst_dense + mst_dense.T
    
    # Compute degrees
    degrees = (mst_symmetric > 0).sum(axis=1)
    
    # Branch nodes: degree >= 3
    branch_indices = np.where(degrees >= 3)[0]
    branch_nodes = coords[branch_indices]
    
    # Leaf nodes: degree == 1
    leaf_indices = np.where(degrees == 1)[0]
    leaf_nodes = coords[leaf_indices]
    
    if return_graph:
        return branch_nodes, leaf_nodes, mst_symmetric, coords
    return branch_nodes, leaf_nodes


def prune_leaf_segments(adjacency, coords, expected_num_leaf_nodes=2):
    """
    Prune leaf segments from an MST until we have the expected number of leaf nodes.
    """
    adjacency = np.array(adjacency, dtype=np.float64)
    coords = np.array(coords, dtype=np.int64)
    n = adjacency.shape[0]
    
    if n == 0:
        return {
            "adjacency": adjacency,
            "coords": coords,
            "branch_coords": np.empty((0, 2), dtype=np.int64),
            "leaf_coords": np.empty((0, 2), dtype=np.int64),
        }
    
    # Make symmetric
    adjacency = np.maximum(adjacency, adjacency.T)
    
    # Iteratively prune
    active = np.ones(n, dtype=bool)
    max_iterations = n
    
    for _ in range(max_iterations):
        degrees = np.zeros(n, dtype=np.int64)
        for i in range(n):
            if active[i]:
                degrees[i] = np.sum((adjacency[i, :] > 0) & active)
        
        leaf_mask = (degrees == 1) & active
        num_leaves = np.sum(leaf_mask)
        
        if num_leaves <= expected_num_leaf_nodes:
            break
        
        # Find shortest leaf segment to prune
        leaf_indices = np.where(leaf_mask)[0]
        min_length = np.inf
        prune_idx = -1
        
        for leaf_idx in leaf_indices:
            # Trace back to branch
            current = leaf_idx
            path = [current]
            visited = {current}
            
            while True:
                neighbors = np.where((adjacency[current, :] > 0) & active)[0]
                neighbors = [n for n in neighbors if n not in visited]
                
                if len(neighbors) == 0:
                    break
                
                next_node = neighbors[0]
                path.append(next_node)
                visited.add(next_node)
                
                # Check if reached a branch (degree >= 3) or another leaf
                deg = np.sum((adjacency[next_node, :] > 0) & active)
                if deg >= 3 or deg == 1:
                    break
                
                current = next_node
            
            # Compute path length
            path_length = 0
            for j in range(len(path) - 1):
                path_length += adjacency[path[j], path[j + 1]]
            
            if path_length < min_length:
                min_length = path_length
                prune_idx = leaf_idx
        
        if prune_idx >= 0:
            active[prune_idx] = False
            adjacency[prune_idx, :] = 0
            adjacency[:, prune_idx] = 0
    
    # Extract final results
    active_indices = np.where(active)[0]
    new_coords = coords[active_indices]
    
    # Rebuild adjacency for active nodes
    new_adjacency = adjacency[np.ix_(active_indices, active_indices)]
    
    # Recompute degrees
    new_degrees = (new_adjacency > 0).sum(axis=1)
    
    branch_mask = new_degrees >= 3
    leaf_mask = new_degrees == 1
    
    return {
        "adjacency": new_adjacency,
        "coords": new_coords,
        "branch_coords": new_coords[branch_mask],
        "leaf_coords": new_coords[leaf_mask],
    }


def mask_from_mst(adjacency, coords, shape):
    """Reconstruct a skeleton mask from MST adjacency and coordinates."""
    mask = np.zeros(shape, dtype=np.uint8)
    
    if coords.shape[0] == 0:
        return mask
    
    # Draw all skeleton points
    for coord in coords:
        row, col = int(coord[0]), int(coord[1])
        if 0 <= row < shape[0] and 0 <= col < shape[1]:
            mask[row, col] = 255
    
    # Draw edges between connected points
    n = adjacency.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] > 0:
                r0, c0 = int(coords[i, 0]), int(coords[i, 1])
                r1, c1 = int(coords[j, 0]), int(coords[j, 1])
                
                # Bresenham-like line drawing
                dr = abs(r1 - r0)
                dc = abs(c1 - c0)
                sr = 1 if r0 < r1 else -1
                sc = 1 if c0 < c1 else -1
                err = dr - dc
                
                r, c = r0, c0
                while True:
                    if 0 <= r < shape[0] and 0 <= c < shape[1]:
                        mask[r, c] = 255
                    
                    if r == r1 and c == c1:
                        break
                    
                    e2 = 2 * err
                    if e2 > -dc:
                        err -= dc
                        r += sr
                    if e2 < dr:
                        err += dr
                        c += sc
    
    return mask


def filter_point_cloud_radius(points, colors, radius=10.0, min_neighbors=5):
    """
    Filter point cloud to remove isolated points using radius search.
    """
    if points.shape[0] == 0:
        return points, colors
    
    nn = NearestNeighbors(radius=radius, algorithm='auto')
    nn.fit(points)
    
    neighbor_counts = np.array([
        len(neighbors) - 1  # Subtract 1 to exclude self
        for neighbors in nn.radius_neighbors(points, return_distance=False)
    ])
    
    keep_mask = neighbor_counts >= min_neighbors
    
    filtered_points = points[keep_mask]
    filtered_colors = colors[keep_mask] if colors is not None else None
    
    return filtered_points, filtered_colors



