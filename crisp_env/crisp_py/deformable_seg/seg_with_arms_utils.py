import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import linear_sum_assignment


def depth_to_point_cloud_full(depth, intrinsics):
    """
    Convert full depth image to 3D point cloud using camera intrinsics.
    Returns point cloud in [H, W, 3] format.
    
    Args:
        depth: HxW depth image
        intrinsics: 3x3 camera intrinsic matrix
    
    Returns:
        points_3d: HxWx3 array of 3D points
    """
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    H, W = depth.shape
    
    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Back-project to 3D
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points_3d = np.stack([x, y, z], axis=-1)
    return points_3d


def background_subtraction(full_pc, arm_pc, threshold=5.0, arm_dilation=1):
    """
    Remove robot arm from full point cloud using background subtraction.
    Points with trivial difference from arm point cloud are set to background.
    
    Args:
        full_pc: HxWx3 full point cloud
        arm_pc: HxWx3 robot arm point cloud
        threshold: distance threshold for considering points as arm (mm)
        arm_dilation: number of pixels to dilate the arm mask (to expand arm region)
    
    Returns:
        foreground_mask: HxW binary mask (1 = foreground, 0 = background/arm)
    """
    # Compute Euclidean distance between corresponding points
    diff = np.linalg.norm(full_pc - arm_pc, axis=-1)
    
    # Points with small difference are robot arm (background)
    # Points with large difference are foreground (wires, objects)
    arm_mask = (diff <= threshold).astype(np.uint8)
    
    # Dilate arm mask to expand the arm region by specified pixels
    if arm_dilation > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                            (2 * arm_dilation + 1, 2 * arm_dilation + 1))
        arm_mask = cv2.dilate(arm_mask, kernel, iterations=1)
    
    # Foreground is the inverse of arm mask
    foreground_mask = (1 - arm_mask).astype(np.uint8)
    
    return foreground_mask


def apply_depth_threshold(mask, depth, max_depth=100.0):
    """
    Set points with invalid depth or depth > max_depth to background.
    
    Args:
        mask: HxW binary mask
        depth: HxW depth image
        max_depth: maximum depth threshold
    
    Returns:
        filtered_mask: HxW binary mask with depth filtering applied
    """
    filtered_mask = mask.copy()
    # Remove points with depth beyond threshold
    filtered_mask[depth > max_depth] = 0
    # Remove points with invalid depth (zero, negative, or NaN)
    filtered_mask[depth <= 0] = 0
    filtered_mask[np.isnan(depth)] = 0
    filtered_mask[np.isinf(depth)] = 0
    return filtered_mask


def get_largest_connected_component(mask, n=1):
    """
    Get the n largest connected components from a binary mask.
    
    Args:
        mask: HxW binary mask
        n: number of largest connected components to keep (default: 1)
    
    Returns:
        largest_cc: HxW binary mask with only the n largest connected components
    """
    # Label connected components
    labeled, num_features = ndimage.label(mask)
    
    if num_features == 0:
        return mask
    
    # Find component sizes
    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    
    # Get indices of n largest components (sorted by size, descending)
    n = min(n, num_features)  # Can't keep more than available
    largest_labels = np.argsort(component_sizes)[::-1][:n] + 1  # +1 because labels start at 1
    
    # Create mask with n largest components
    largest_cc = np.isin(labeled, largest_labels).astype(np.uint8)
    return largest_cc


def skeletonize_mask(mask):
    """
    Skeletonize a binary mask.
    
    Args:
        mask: HxW binary mask
    
    Returns:
        skeleton: HxW skeletonized mask
    """
    skeleton = skeletonize(mask > 0).astype(np.uint8)
    return skeleton


def create_overlay(rgb_image, mask, color=[0, 255, 0], alpha=0.5):
    """
    Create overlay of mask on RGB image.
    
    Args:
        rgb_image: HxWx3 RGB image
        mask: HxW binary mask
        color: RGB color for overlay
        alpha: transparency for overlay
    
    Returns:
        overlay: HxWx3 overlay image
    """
    overlay = rgb_image.copy()
    overlay[mask > 0] = (overlay[mask > 0] * (1 - alpha) + 
                          np.array(color) * alpha).astype(np.uint8)
    return overlay


def create_skeleton_overlay(rgb_image, skeleton, color=[255, 0, 0], thickness=2):
    """
    Create overlay of skeleton on RGB image with thicker lines for visibility.
    
    Args:
        rgb_image: HxWx3 RGB image
        skeleton: HxW binary skeleton mask
        color: RGB color for skeleton
        thickness: thickness of skeleton lines
    
    Returns:
        overlay: HxWx3 overlay image
    """
    overlay = rgb_image.copy()
    # Dilate skeleton for better visibility
    if thickness > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        skeleton_thick = cv2.dilate(skeleton, kernel, iterations=1)
    else:
        skeleton_thick = skeleton
    
    overlay[skeleton_thick > 0] = color
    return overlay


def node_identification(skeleton_mask):
    """
    Identify branch and leaf nodes in a skeletonized mask via MST.
    
    Args:
        skeleton_mask: HxW binary skeleton mask
    
    Returns:
        branch_nodes: Nx2 array of branch node coordinates (row, col)
        leaf_nodes: Mx2 array of leaf node coordinates (row, col)
        mst_adjacency: NxN adjacency matrix of MST
        coords: All skeleton pixel coordinates
    """
    mask = np.asarray(skeleton_mask)
    binary_mask = mask > 0
    coords = np.column_stack(np.nonzero(binary_mask)).astype(np.int64)
    
    if coords.size == 0:
        empty = np.empty((0, 2), dtype=np.int64)
        return empty, empty, None, None
    
    if coords.shape[0] == 1:
        empty = np.empty((0, 2), dtype=np.int64)
        return empty, coords.copy(), None, coords
    
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
    
    return branch_nodes, leaf_nodes, mst_symmetric, coords


def prune_to_target_nodes(adjacency, coords, target_branch_nodes=2, target_leaf_nodes=4):
    """
    Prune the MST to reach target number of branch and leaf nodes.
    
    Args:
        adjacency: NxN adjacency matrix
        coords: Nx2 array of coordinates
        target_branch_nodes: desired number of branch nodes (degree >= 3)
        target_leaf_nodes: desired number of leaf nodes (degree == 1)
    
    Returns:
        dict with pruned adjacency, coords, branch_coords, leaf_coords
    """
    if adjacency is None or coords is None or len(coords) == 0:
        return {
            "adjacency": None,
            "coords": np.empty((0, 2), dtype=np.int64),
            "branch_coords": np.empty((0, 2), dtype=np.int64),
            "leaf_coords": np.empty((0, 2), dtype=np.int64),
        }
    
    adjacency = np.array(adjacency, dtype=np.float64)
    coords = np.array(coords, dtype=np.int64)
    n = adjacency.shape[0]
    
    # Make symmetric
    adjacency = np.maximum(adjacency, adjacency.T)
    
    # Iteratively prune leaf segments until we reach target
    active = np.ones(n, dtype=bool)
    max_iterations = n
    
    for iteration in range(max_iterations):
        # Compute current degrees
        degrees = np.zeros(n, dtype=np.int64)
        for idx in range(n):
            if active[idx]:
                degrees[idx] = np.sum((adjacency[idx, :] > 0) & active)
        
        # Count current nodes
        leaf_mask = (degrees == 1) & active
        branch_mask = (degrees >= 3) & active
        num_leaves = np.sum(leaf_mask)
        num_branches = np.sum(branch_mask)
        
        # Check if we've reached target
        if num_leaves <= target_leaf_nodes:
            break
        
        # Find shortest leaf segment to prune
        leaf_indices = np.where(leaf_mask)[0]
        min_length = np.inf
        prune_path = []
        
        for leaf_idx in leaf_indices:
            # Trace path from leaf to branch or another endpoint
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
                prune_path = path[:-1]  # Don't remove the branch node
        
        # Prune the shortest leaf segment
        if len(prune_path) > 0:
            for idx in prune_path:
                active[idx] = False
                adjacency[idx, :] = 0
                adjacency[:, idx] = 0
        else:
            break
    
    # Now prune extra branch nodes if needed
    for iteration in range(max_iterations):
        degrees = np.zeros(n, dtype=np.int64)
        for idx in range(n):
            if active[idx]:
                degrees[idx] = np.sum((adjacency[idx, :] > 0) & active)
        
        branch_mask = (degrees >= 3) & active
        num_branches = np.sum(branch_mask)
        
        if num_branches <= target_branch_nodes:
            break
        
        # Find branch with smallest total edge weight and prune one of its edges
        branch_indices = np.where(branch_mask)[0]
        
        # For simplicity, merge the two closest branches
        if len(branch_indices) >= 2:
            branch_coords_current = coords[branch_indices]
            branch_dists = cdist(branch_coords_current, branch_coords_current)
            np.fill_diagonal(branch_dists, np.inf)
            min_idx = np.unravel_index(np.argmin(branch_dists), branch_dists.shape)
            
            # Find path between these two branches and remove one intermediate node
            b1, b2 = branch_indices[min_idx[0]], branch_indices[min_idx[1]]
            
            # Simple approach: just deactivate the branch with lower degree
            deg1 = degrees[b1]
            deg2 = degrees[b2]
            to_remove = b1 if deg1 <= deg2 else b2
            
            # Don't actually remove, just break one edge
            neighbors = np.where((adjacency[to_remove, :] > 0) & active)[0]
            if len(neighbors) > 0:
                # Remove edge to neighbor with smallest edge weight
                edge_weights = adjacency[to_remove, neighbors]
                min_neighbor = neighbors[np.argmin(edge_weights)]
                adjacency[to_remove, min_neighbor] = 0
                adjacency[min_neighbor, to_remove] = 0
        else:
            break
    
    # Extract final results
    active_indices = np.where(active)[0]
    new_coords = coords[active_indices]
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


def draw_mst_with_nodes(shape, adjacency, coords, branch_coords, leaf_coords):
    """
    Draw the MST with branch and leaf nodes highlighted.
    
    Args:
        shape: (H, W) output image shape
        adjacency: NxN adjacency matrix
        coords: Nx2 array of all node coordinates
        branch_coords: Bx2 array of branch node coordinates
        leaf_coords: Lx2 array of leaf node coordinates
    
    Returns:
        vis: HxWx3 visualization image
    """
    vis = np.zeros((*shape, 3), dtype=np.uint8)
    
    if adjacency is None or coords is None or len(coords) == 0:
        return vis
    
    # Draw edges (white)
    n = adjacency.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] > 0:
                r0, c0 = int(coords[i, 0]), int(coords[i, 1])
                r1, c1 = int(coords[j, 0]), int(coords[j, 1])
                cv2.line(vis, (c0, r0), (c1, r1), (255, 255, 255), 1)
    
    # Draw branch nodes (purple: RGB = [128, 0, 128])
    for coord in branch_coords:
        r, c = int(coord[0]), int(coord[1])
        cv2.circle(vis, (c, r), 5, (128, 0, 128), -1)
        cv2.circle(vis, (c, r), 5, (255, 255, 255), 1)
    
    # Draw leaf nodes (gold: RGB = [255, 215, 0])
    for coord in leaf_coords:
        r, c = int(coord[0]), int(coord[1])
        cv2.circle(vis, (c, r), 5, (255, 215, 0), -1)
        cv2.circle(vis, (c, r), 5, (255, 255, 255), 1)
    
    return vis


def draw_mst_overlay(rgb_image, adjacency, coords, branch_coords, leaf_coords, edge_color=[255, 0, 0], thickness=2):
    """
    Draw the MST overlaid on RGB image with nodes highlighted.
    
    Args:
        rgb_image: HxWx3 RGB image
        adjacency: NxN adjacency matrix
        coords: Nx2 array of all node coordinates
        branch_coords: Bx2 array of branch node coordinates
        leaf_coords: Lx2 array of leaf node coordinates
        edge_color: RGB color for edges
        thickness: line thickness for edges
    
    Returns:
        overlay: HxWx3 overlay image
    """
    overlay = rgb_image.copy()
    
    if adjacency is None or coords is None or len(coords) == 0:
        return overlay
    
    # Draw edges
    n = adjacency.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] > 0:
                r0, c0 = int(coords[i, 0]), int(coords[i, 1])
                r1, c1 = int(coords[j, 0]), int(coords[j, 1])
                cv2.line(overlay, (c0, r0), (c1, r1), edge_color, thickness)
    
    # Draw branch nodes (purple)
    for coord in branch_coords:
        r, c = int(coord[0]), int(coord[1])
        cv2.circle(overlay, (c, r), 6, (128, 0, 128), -1)
        cv2.circle(overlay, (c, r), 6, (255, 255, 255), 2)
    
    # Draw leaf nodes (gold)
    for coord in leaf_coords:
        r, c = int(coord[0]), int(coord[1])
        cv2.circle(overlay, (c, r), 6, (255, 215, 0), -1)
        cv2.circle(overlay, (c, r), 6, (255, 255, 255), 2)
    
    return overlay


def expand_arm_depth(arm_depth, dilation_pixels=5):
    """
    Expand arm depth mask by dilating and filling with nearest neighbor values.
    
    Args:
        arm_depth: HxW depth image
        dilation_pixels: number of pixels to dilate
    
    Returns:
        arm_depth_expanded: HxW expanded depth image
    """
    arm_valid_mask = (arm_depth > 0).astype(np.uint8)
    kernel_size = 2 * dilation_pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    arm_valid_mask_dilated = cv2.dilate(arm_valid_mask, kernel, iterations=1)
    
    # Expand arm depth to dilated region using nearest neighbor
    arm_depth_expanded = arm_depth.copy()
    new_pixels = (arm_valid_mask_dilated > 0) & (arm_valid_mask == 0)
    if np.any(new_pixels):
        # Use distance transform to find nearest valid pixel
        dist, indices = ndimage.distance_transform_edt(arm_valid_mask == 0, return_indices=True)
        arm_depth_expanded[new_pixels] = arm_depth[indices[0][new_pixels], indices[1][new_pixels]]
    
    return arm_depth_expanded


# ============================================================
# Coherent Point Drift (CPD) for Node Tracking
# ============================================================

def cpd_register(Y, X, beta=2.0, lmbda=2.0, w=0.1, max_iter=100, tol=1e-5):
    """
    Non-rigid Coherent Point Drift registration.
    
    Aligns template point set Y to target point set X using a Gaussian mixture
    model with motion coherence regularization.
    
    Args:
        Y: M x D template point set (previous frame nodes)
        X: N x D target point set (current frame skeleton pixels or nodes)
        beta: Gaussian kernel width for motion coherence (larger = more rigid)
        lmbda: Regularization weight (larger = smoother deformation)
        w: Outlier weight in [0, 1] (expected fraction of outliers)
        max_iter: Maximum EM iterations
        tol: Convergence tolerance
    
    Returns:
        T_Y: M x D transformed template points
        P: M x N correspondence matrix (posterior probabilities)
    """
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    
    M, D = Y.shape
    N = X.shape[0]
    
    if M == 0 or N == 0:
        return Y.copy(), np.zeros((M, N))
    
    # Initialize
    T_Y = Y.copy()  # Transformed template
    W = np.zeros((M, D))  # Deformation weights
    
    # Compute Gaussian kernel matrix G for motion coherence
    # G_ij = exp(-||y_i - y_j||^2 / (2 * beta^2))
    diff_Y = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]  # M x M x D
    G = np.exp(-np.sum(diff_Y ** 2, axis=2) / (2 * beta ** 2))  # M x M
    
    # Initialize sigma^2 (variance)
    diff_init = X[np.newaxis, :, :] - Y[:, np.newaxis, :]  # M x N x D
    sigma2 = np.sum(diff_init ** 2) / (M * N * D)
    
    for iteration in range(max_iter):
        # ============================================================
        # E-step: Compute posterior probabilities P(m | x_n)
        # ============================================================
        diff = X[np.newaxis, :, :] - T_Y[:, np.newaxis, :]  # M x N x D
        dist2 = np.sum(diff ** 2, axis=2)  # M x N
        
        # Numerator: exp(-||x_n - T(y_m)||^2 / (2 * sigma^2))
        P_num = np.exp(-dist2 / (2 * sigma2))  # M x N
        
        # Denominator with outlier term
        c = (w / (1 - w)) * (M / N) * ((2 * np.pi * sigma2) ** (D / 2))
        P_den = np.sum(P_num, axis=0, keepdims=True) + c  # 1 x N
        
        P = P_num / (P_den + 1e-10)  # M x N
        
        # ============================================================
        # M-step: Update W and sigma^2
        # ============================================================
        P1 = np.sum(P, axis=1)  # M - sum over n
        Pt1 = np.sum(P, axis=0)  # N - sum over m
        Np = np.sum(P1)  # Total probability mass
        
        # Solve for W: (G + lambda * sigma^2 * diag(1/P1)) W = diag(1/P1) P X - Y
        # Simplified: (G + lambda * sigma^2 * D_inv) W = D_inv @ P @ X - Y
        # where D = diag(P1)
        
        P1_safe = np.maximum(P1, 1e-10)  # Avoid division by zero
        D_inv = np.diag(1.0 / P1_safe)
        
        A = G + lmbda * sigma2 * D_inv
        B = D_inv @ P @ X - Y
        
        # Solve linear system A @ W = B
        try:
            W = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            W = np.linalg.lstsq(A, B, rcond=None)[0]
        
        # Update transformed points
        T_Y_new = Y + G @ W
        
        # Update sigma^2
        diff_new = X[np.newaxis, :, :] - T_Y_new[:, np.newaxis, :]  # M x N x D
        dist2_new = np.sum(diff_new ** 2, axis=2)  # M x N
        sigma2_new = np.sum(P * dist2_new) / (Np * D)
        sigma2_new = max(sigma2_new, 1e-10)  # Prevent zero variance
        
        # Check convergence
        change = np.max(np.abs(T_Y_new - T_Y))
        T_Y = T_Y_new
        sigma2 = sigma2_new
        
        if change < tol:
            break
    
    return T_Y, P


def pixel_to_3d(coords_2d, depth, intrinsics):
    """
    Back-project 2D pixel coordinates to 3D using depth and camera intrinsics.
    
    Args:
        coords_2d: N x 2 array of (row, col) pixel coordinates
        depth: H x W depth image
        intrinsics: 3x3 camera intrinsic matrix
    
    Returns:
        coords_3d: N x 3 array of (x, y, z) 3D coordinates
    """
    if len(coords_2d) == 0:
        return np.empty((0, 3))
    
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    coords_3d = np.zeros((len(coords_2d), 3), dtype=np.float64)
    
    for i, (r, c) in enumerate(coords_2d):
        r_int, c_int = int(r), int(c)
        # Clamp to image bounds
        r_int = max(0, min(r_int, depth.shape[0] - 1))
        c_int = max(0, min(c_int, depth.shape[1] - 1))
        
        z = depth[r_int, c_int]
        if z <= 0:
            # Invalid depth - try to find nearby valid depth
            for dr in range(-5, 6):
                for dc in range(-5, 6):
                    rr, cc = r_int + dr, c_int + dc
                    if 0 <= rr < depth.shape[0] and 0 <= cc < depth.shape[1]:
                        if depth[rr, cc] > 0:
                            z = depth[rr, cc]
                            break
                if z > 0:
                    break
        
        if z > 0:
            x = (c_int - cx) * z / fx
            y = (r_int - cy) * z / fy
            coords_3d[i] = [x, y, z]
        else:
            # Fallback: use a default depth
            z = 500.0  # 500mm default
            x = (c_int - cx) * z / fx
            y = (r_int - cy) * z / fy
            coords_3d[i] = [x, y, z]
    
    return coords_3d


def track_nodes_cpd_3d(prev_nodes, curr_skeleton, depth, intrinsics, 
                        beta=10.0, lmbda=2.0, w=0.1):
    """
    Track nodes from previous frame to current frame using CPD in 3D space.
    
    This version:
    1. Detects all branch/leaf nodes from skeleton (2D pixel coords)
    2. Back-projects both prev_nodes and detected nodes to 3D using depth
    3. Runs CPD in 3D space to get correspondence matrix P
    4. Uses Hungarian algorithm on P to find optimal assignment
    5. Returns 2D pixel coordinates of matched candidates
    
    Args:
        prev_nodes: dict with "branch_coords" (B x 2), "leaf_coords" (L x 2), 
                    and optionally "branch_3d" (B x 3), "leaf_3d" (L x 3)
        curr_skeleton: H x W binary skeleton mask
        depth: H x W depth image for current frame
        intrinsics: 3x3 camera intrinsic matrix
        beta: CPD smoothness parameter (larger = more rigid tracking)
        lmbda: CPD regularization parameter
        w: Outlier weight
    
    Returns:
        tracked_nodes: dict with "branch_coords", "leaf_coords" (2D), 
                       "branch_3d", "leaf_3d" (3D)
        detected_nodes: dict with all detected nodes (2D and 3D)
        confidence: float indicating tracking quality
    """
    prev_branch_2d = prev_nodes.get("branch_coords", np.empty((0, 2)))
    prev_leaf_2d = prev_nodes.get("leaf_coords", np.empty((0, 2)))
    prev_branch_3d = prev_nodes.get("branch_3d", None)
    prev_leaf_3d = prev_nodes.get("leaf_3d", None)
    
    n_branch = len(prev_branch_2d)
    n_leaf = len(prev_leaf_2d)
    
    if n_branch == 0 and n_leaf == 0:
        return prev_nodes.copy(), {
            "branch_coords": np.empty((0, 2)), "leaf_coords": np.empty((0, 2)),
            "branch_3d": np.empty((0, 3)), "leaf_3d": np.empty((0, 3))
        }, 0.0
    
    # ============================================================
    # Step 1: Detect all branch/leaf nodes from current skeleton
    # ============================================================
    all_branch_2d, all_leaf_2d, mst_adj, node_coords = node_identification(curr_skeleton)
    
    # Convert detected nodes to 3D
    all_branch_3d = pixel_to_3d(all_branch_2d, depth, intrinsics) if len(all_branch_2d) > 0 else np.empty((0, 3))
    all_leaf_3d = pixel_to_3d(all_leaf_2d, depth, intrinsics) if len(all_leaf_2d) > 0 else np.empty((0, 3))
    
    detected_nodes = {
        "branch_coords": all_branch_2d.copy() if len(all_branch_2d) > 0 else np.empty((0, 2)),
        "leaf_coords": all_leaf_2d.copy() if len(all_leaf_2d) > 0 else np.empty((0, 2)),
        "branch_3d": all_branch_3d.copy(),
        "leaf_3d": all_leaf_3d.copy()
    }
    
    # Convert previous nodes to 3D if not already
    if prev_branch_3d is None and n_branch > 0:
        prev_branch_3d = pixel_to_3d(prev_branch_2d, depth, intrinsics)
    if prev_leaf_3d is None and n_leaf > 0:
        prev_leaf_3d = pixel_to_3d(prev_leaf_2d, depth, intrinsics)
    
    all_confidences = []
    
    # ============================================================
    # Step 2a: Track BRANCH nodes using CPD in 3D
    # ============================================================
    if n_branch > 0 and len(all_branch_3d) > 0:
        Y_branch = prev_branch_3d.astype(np.float64)  # Template: previous branch nodes (M x 3)
        X_branch = all_branch_3d.astype(np.float64)   # Target: detected branch candidates (N x 3)
        
        # Run CPD registration in 3D to get correspondence matrix P
        T_Y_branch, P_branch = cpd_register(Y_branch, X_branch, beta=beta, lmbda=lmbda, w=w)
        # P_branch is M x N: P[m, n] = probability that prev node m corresponds to candidate n
        
        # Use Hungarian algorithm to find optimal assignment
        cost_matrix = -P_branch  # M x N
        
        # Handle case where M > N
        M, N = P_branch.shape
        if M > N:
            padding = np.zeros((M, M - N))
            cost_matrix = np.hstack([cost_matrix, padding])
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Assign each previous node to its matched candidate
        tracked_branch_2d = np.zeros((n_branch, 2), dtype=np.float64)
        tracked_branch_3d = np.zeros((n_branch, 3), dtype=np.float64)
        
        for i, j in zip(row_ind, col_ind):
            if i < n_branch:
                if j < len(all_branch_2d):
                    tracked_branch_2d[i] = all_branch_2d[j]
                    tracked_branch_3d[i] = all_branch_3d[j]
                    all_confidences.append(float(P_branch[i, j]))
                else:
                    # Assigned to dummy - use nearest candidate in 3D
                    dists = np.linalg.norm(X_branch - Y_branch[i], axis=1)
                    nearest = np.argmin(dists)
                    tracked_branch_2d[i] = all_branch_2d[nearest]
                    tracked_branch_3d[i] = all_branch_3d[nearest]
                    all_confidences.append(0.1)
    elif n_branch > 0:
        tracked_branch_2d = prev_branch_2d.copy()
        tracked_branch_3d = prev_branch_3d.copy() if prev_branch_3d is not None else np.empty((0, 3))
        all_confidences.append(0.0)
    else:
        tracked_branch_2d = np.empty((0, 2))
        tracked_branch_3d = np.empty((0, 3))
    
    # ============================================================
    # Step 2b: Track LEAF nodes using CPD in 3D
    # ============================================================
    if n_leaf > 0 and len(all_leaf_3d) > 0:
        Y_leaf = prev_leaf_3d.astype(np.float64)  # Template: previous leaf nodes (M x 3)
        X_leaf = all_leaf_3d.astype(np.float64)   # Target: detected leaf candidates (N x 3)
        
        # Run CPD registration in 3D to get correspondence matrix P
        T_Y_leaf, P_leaf = cpd_register(Y_leaf, X_leaf, beta=beta, lmbda=lmbda, w=w)
        
        # Use Hungarian algorithm
        cost_matrix = -P_leaf
        M, N = P_leaf.shape
        if M > N:
            padding = np.zeros((M, M - N))
            cost_matrix = np.hstack([cost_matrix, padding])
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Assign each previous node to its matched candidate
        tracked_leaf_2d = np.zeros((n_leaf, 2), dtype=np.float64)
        tracked_leaf_3d = np.zeros((n_leaf, 3), dtype=np.float64)
        
        for i, j in zip(row_ind, col_ind):
            if i < n_leaf:
                if j < len(all_leaf_2d):
                    tracked_leaf_2d[i] = all_leaf_2d[j]
                    tracked_leaf_3d[i] = all_leaf_3d[j]
                    all_confidences.append(float(P_leaf[i, j]))
                else:
                    # Assigned to dummy - use nearest candidate in 3D
                    dists = np.linalg.norm(X_leaf - Y_leaf[i], axis=1)
                    nearest = np.argmin(dists)
                    tracked_leaf_2d[i] = all_leaf_2d[nearest]
                    tracked_leaf_3d[i] = all_leaf_3d[nearest]
                    all_confidences.append(0.1)
    elif n_leaf > 0:
        tracked_leaf_2d = prev_leaf_2d.copy()
        tracked_leaf_3d = prev_leaf_3d.copy() if prev_leaf_3d is not None else np.empty((0, 3))
        all_confidences.append(0.0)
    else:
        tracked_leaf_2d = np.empty((0, 2))
        tracked_leaf_3d = np.empty((0, 3))
    
    avg_confidence = np.mean(all_confidences) if len(all_confidences) > 0 else 0.0
    
    return {
        "branch_coords": tracked_branch_2d,
        "leaf_coords": tracked_leaf_2d,
        "branch_3d": tracked_branch_3d,
        "leaf_3d": tracked_leaf_3d,
    }, detected_nodes, avg_confidence


def track_nodes_cpd(prev_nodes, curr_skeleton, beta=10.0, lmbda=2.0, w=0.1,
                    snap_threshold=15.0):
    """
    Track nodes from previous frame to current frame using CPD.
    
    CORRECT APPROACH using CPD's correspondence matrix P:
    1. Detect all branch nodes and all leaf nodes from skeleton
    2. Run CPD to get correspondence matrix P (probability that prev node i matches candidate j)
    3. Use Hungarian algorithm on P to find optimal assignment
    4. Each tracked node is ALWAYS selected from detected candidates
    
    Args:
        prev_nodes: dict with "branch_coords" (B x 2) and "leaf_coords" (L x 2)
        curr_skeleton: H x W binary skeleton mask
        beta: CPD smoothness parameter (larger = more rigid tracking)
        lmbda: CPD regularization parameter
        w: Outlier weight
        snap_threshold: (unused - always select from candidates)
    
    Returns:
        tracked_nodes: dict with "branch_coords" and "leaf_coords" (always from candidates)
        detected_nodes: dict with all detected "branch_coords" and "leaf_coords"
        confidence: float indicating tracking quality
    """
    prev_branch = prev_nodes.get("branch_coords", np.empty((0, 2)))
    prev_leaf = prev_nodes.get("leaf_coords", np.empty((0, 2)))
    n_branch = len(prev_branch)
    n_leaf = len(prev_leaf)
    
    if n_branch == 0 and n_leaf == 0:
        return prev_nodes.copy(), {"branch_coords": np.empty((0, 2)), "leaf_coords": np.empty((0, 2))}, 0.0
    
    # ============================================================
    # Step 1: Detect all branch/leaf nodes from current skeleton
    # ============================================================
    all_branch, all_leaf, mst_adj, node_coords = node_identification(curr_skeleton)
    
    detected_nodes = {
        "branch_coords": all_branch.copy() if len(all_branch) > 0 else np.empty((0, 2)),
        "leaf_coords": all_leaf.copy() if len(all_leaf) > 0 else np.empty((0, 2))
    }
    
    all_confidences = []
    
    # ============================================================
    # Step 2a: Track BRANCH nodes using CPD correspondence matrix
    # ============================================================
    if n_branch > 0 and len(all_branch) > 0:
        Y_branch = prev_branch.astype(np.float64)  # Template: previous branch nodes (M x 2)
        X_branch = all_branch.astype(np.float64)   # Target: detected branch candidates (N x 2)
        
        # Run CPD registration to get correspondence matrix P
        T_Y_branch, P_branch = cpd_register(Y_branch, X_branch, beta=beta, lmbda=lmbda, w=w)
        # P_branch is M x N: P[m, n] = probability that prev node m corresponds to candidate n
        
        # Use Hungarian algorithm to find optimal assignment
        # Convert probability to cost (we want to maximize probability, so negate)
        cost_matrix = -P_branch  # M x N
        
        # Handle case where M > N (more prev nodes than candidates)
        M, N = P_branch.shape
        if M > N:
            # Pad cost matrix with high cost columns (dummy candidates)
            padding = np.zeros((M, M - N))
            cost_matrix = np.hstack([cost_matrix, padding])
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Assign each previous node to its matched candidate
        tracked_branch = np.zeros((n_branch, 2), dtype=np.float64)
        for i, j in zip(row_ind, col_ind):
            if i < n_branch:
                if j < len(X_branch):
                    # Valid assignment to a real candidate
                    tracked_branch[i] = X_branch[j]
                    all_confidences.append(float(P_branch[i, j]))
                else:
                    # Assigned to dummy - no good match, use nearest candidate
                    dists = np.linalg.norm(X_branch - Y_branch[i], axis=1)
                    nearest = np.argmin(dists)
                    tracked_branch[i] = X_branch[nearest]
                    all_confidences.append(0.1)
    elif n_branch > 0:
        # No branch candidates detected - keep previous (will show as error)
        tracked_branch = prev_branch.copy()
        all_confidences.append(0.0)
    else:
        tracked_branch = np.empty((0, 2))
    
    # ============================================================
    # Step 2b: Track LEAF nodes using CPD correspondence matrix
    # ============================================================
    if n_leaf > 0 and len(all_leaf) > 0:
        Y_leaf = prev_leaf.astype(np.float64)  # Template: previous leaf nodes (M x 2)
        X_leaf = all_leaf.astype(np.float64)   # Target: detected leaf candidates (N x 2)
        
        # Run CPD registration to get correspondence matrix P
        T_Y_leaf, P_leaf = cpd_register(Y_leaf, X_leaf, beta=beta, lmbda=lmbda, w=w)
        # P_leaf is M x N: P[m, n] = probability that prev node m corresponds to candidate n
        
        # Use Hungarian algorithm to find optimal assignment
        cost_matrix = -P_leaf  # M x N
        
        # Handle case where M > N (more prev nodes than candidates)
        M, N = P_leaf.shape
        if M > N:
            # Pad cost matrix with high cost columns (dummy candidates)
            padding = np.zeros((M, M - N))
            cost_matrix = np.hstack([cost_matrix, padding])
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Assign each previous node to its matched candidate
        tracked_leaf = np.zeros((n_leaf, 2), dtype=np.float64)
        for i, j in zip(row_ind, col_ind):
            if i < n_leaf:
                if j < len(X_leaf):
                    # Valid assignment to a real candidate
                    tracked_leaf[i] = X_leaf[j]
                    all_confidences.append(float(P_leaf[i, j]))
                else:
                    # Assigned to dummy - no good match, use nearest candidate
                    dists = np.linalg.norm(X_leaf - Y_leaf[i], axis=1)
                    nearest = np.argmin(dists)
                    tracked_leaf[i] = X_leaf[nearest]
                    all_confidences.append(0.1)
    elif n_leaf > 0:
        # No leaf candidates detected - keep previous (will show as error)
        tracked_leaf = prev_leaf.copy()
        all_confidences.append(0.0)
    else:
        tracked_leaf = np.empty((0, 2))
    
    avg_confidence = np.mean(all_confidences) if len(all_confidences) > 0 else 0.0
    
    return {
        "branch_coords": tracked_branch,
        "leaf_coords": tracked_leaf,
    }, detected_nodes, avg_confidence


def match_nodes_hungarian(prev_coords, curr_coords):
    """
    Match nodes from curr_coords to prev_coords using Hungarian algorithm.
    Returns curr_coords reordered to match prev_coords ordering.
    
    Args:
        prev_coords: N x 2 array of previous frame coordinates
        curr_coords: N x 2 array of current frame coordinates
    
    Returns:
        matched_coords: N x 2 array of curr_coords reordered to match prev_coords
    """
    from scipy.optimize import linear_sum_assignment
    
    if len(prev_coords) == 0 or len(curr_coords) == 0:
        return curr_coords.copy()
    
    if len(prev_coords) != len(curr_coords):
        # Different number of nodes - use greedy matching
        return match_nodes_greedy(prev_coords, curr_coords)
    
    # Compute cost matrix (distances)
    cost_matrix = cdist(prev_coords, curr_coords)
    
    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Reorder curr_coords to match prev_coords ordering
    matched_coords = curr_coords[col_ind]
    
    return matched_coords


def match_nodes_greedy(prev_coords, curr_coords):
    """
    Greedy matching of nodes from curr_coords to prev_coords based on distance.
    
    Args:
        prev_coords: N x 2 array of reference coordinates
        curr_coords: M x 2 array of coordinates to match
    
    Returns:
        matched_coords: N x 2 array of matched coordinates
    """
    if len(prev_coords) == 0:
        return curr_coords.copy() if len(curr_coords) > 0 else np.empty((0, 2))
    
    if len(curr_coords) == 0:
        return prev_coords.copy()  # Keep previous if no current
    
    dists = cdist(prev_coords, curr_coords)
    matched_coords = np.zeros_like(prev_coords, dtype=np.float64)
    used = set()
    
    for i in range(len(prev_coords)):
        min_dist = np.inf
        min_j = -1
        for j in range(len(curr_coords)):
            if j not in used and dists[i, j] < min_dist:
                min_dist = dists[i, j]
                min_j = j
        if min_j >= 0:
            matched_coords[i] = curr_coords[min_j]
            used.add(min_j)
        else:
            matched_coords[i] = prev_coords[i]  # Keep previous if no match
    
    return matched_coords


def draw_tracked_nodes(shape, branch_coords, leaf_coords, 
                       branch_color=(128, 0, 128), leaf_color=(255, 215, 0)):
    """
    Draw tracked nodes on a blank image.
    
    Args:
        shape: (H, W) output image shape
        branch_coords: B x 2 array of branch node coordinates
        leaf_coords: L x 2 array of leaf node coordinates
        branch_color: RGB color for branch nodes
        leaf_color: RGB color for leaf nodes
    
    Returns:
        vis: H x W x 3 visualization image
    """
    vis = np.zeros((*shape, 3), dtype=np.uint8)
    
    # Draw branch nodes (purple)
    for coord in branch_coords:
        r, c = int(coord[0]), int(coord[1])
        cv2.circle(vis, (c, r), 5, branch_color, -1)
        cv2.circle(vis, (c, r), 5, (255, 255, 255), 1)
    
    # Draw leaf nodes (gold)
    for coord in leaf_coords:
        r, c = int(coord[0]), int(coord[1])
        cv2.circle(vis, (c, r), 5, leaf_color, -1)
        cv2.circle(vis, (c, r), 5, (255, 255, 255), 1)
    
    return vis


def draw_tracked_nodes_overlay(rgb_image, branch_coords, leaf_coords,
                                branch_color=(128, 0, 128), leaf_color=(255, 215, 0)):
    """
    Draw tracked nodes overlaid on RGB image.
    
    Args:
        rgb_image: H x W x 3 RGB image
        branch_coords: B x 2 array of branch node coordinates
        leaf_coords: L x 2 array of leaf node coordinates
    
    Returns:
        overlay: H x W x 3 overlay image
    """
    overlay = rgb_image.copy()
    
    # Draw branch nodes (purple)
    for coord in branch_coords:
        r, c = int(coord[0]), int(coord[1])
        cv2.circle(overlay, (c, r), 6, branch_color, -1)
        cv2.circle(overlay, (c, r), 6, (255, 255, 255), 2)
    
    # Draw leaf nodes (gold)
    for coord in leaf_coords:
        r, c = int(coord[0]), int(coord[1])
        cv2.circle(overlay, (c, r), 6, leaf_color, -1)
        cv2.circle(overlay, (c, r), 6, (255, 255, 255), 2)
    
    return overlay


def draw_skeleton_with_detected_nodes(skeleton, detected_nodes,
                                       branch_color=(128, 0, 128), leaf_color=(255, 255, 0)):
    """
    Draw skeleton with all detected nodes (before matching/pruning).
    
    Args:
        skeleton: H x W binary skeleton mask
        detected_nodes: dict with "branch_coords" and "leaf_coords"
        branch_color: RGB color for detected branch nodes (purple)
        leaf_color: RGB color for detected leaf nodes (yellow)
    
    Returns:
        vis: H x W x 3 visualization image
    """
    vis = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    
    # Draw skeleton in white (binary)
    vis[skeleton > 0] = [255, 255, 255]
    
    branch_coords = detected_nodes.get("branch_coords", np.empty((0, 2)))
    leaf_coords = detected_nodes.get("leaf_coords", np.empty((0, 2)))
    
    # Draw detected branch nodes (purple - solid)
    for coord in branch_coords:
        r, c = int(coord[0]), int(coord[1])
        cv2.circle(vis, (c, r), 7, branch_color, -1)
    
    # Draw detected leaf nodes (yellow - solid)
    for coord in leaf_coords:
        r, c = int(coord[0]), int(coord[1])
        cv2.circle(vis, (c, r), 7, leaf_color, -1)
    
    return vis


def draw_skeleton_with_tracked_nodes(skeleton, tracked_branch, tracked_leaf,
                                      branch_color=(128, 0, 128), leaf_color=(255, 255, 0)):
    """
    Draw skeleton with tracked/confirmed nodes.
    
    Args:
        skeleton: H x W binary skeleton mask
        tracked_branch: B x 2 array of tracked branch node coordinates
        tracked_leaf: L x 2 array of tracked leaf node coordinates
        branch_color: RGB color for branch nodes (purple)
        leaf_color: RGB color for leaf nodes (yellow)
    
    Returns:
        vis: H x W x 3 visualization image
    """
    vis = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    
    # Draw skeleton in white (binary)
    vis[skeleton > 0] = [255, 255, 255]
    
    # Draw tracked branch nodes (purple - solid)
    for coord in tracked_branch:
        r, c = int(coord[0]), int(coord[1])
        cv2.circle(vis, (c, r), 7, branch_color, -1)
    
    # Draw tracked leaf nodes (yellow - solid)
    for coord in tracked_leaf:
        r, c = int(coord[0]), int(coord[1])
        cv2.circle(vis, (c, r), 7, leaf_color, -1)
    
    return vis


def draw_skeleton_with_tracked_nodes_overlay(rgb_image, skeleton, tracked_branch, tracked_leaf,
                                              skeleton_color=(34, 139, 34),
                                              branch_color=(128, 0, 128), leaf_color=(255, 255, 0)):
    """
    Draw skeleton with tracked nodes overlaid on RGB image.
    
    Args:
        rgb_image: H x W x 3 RGB image
        skeleton: H x W binary skeleton mask
        tracked_branch: B x 2 array of tracked branch coordinates
        tracked_leaf: L x 2 array of tracked leaf coordinates
        skeleton_color: RGB color for skeleton (forest green)
        branch_color: RGB color for branch nodes (purple)
        leaf_color: RGB color for leaf nodes (yellow)
    
    Returns:
        overlay: H x W x 3 overlay image
    """
    overlay = rgb_image.copy()
    
    # Draw skeleton (dilated for visibility - thicker)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    skeleton_thick = cv2.dilate(skeleton.astype(np.uint8), kernel, iterations=1)
    overlay[skeleton_thick > 0] = skeleton_color
    
    # Draw tracked branch nodes (purple - solid)
    for coord in tracked_branch:
        r, c = int(coord[0]), int(coord[1])
        cv2.circle(overlay, (c, r), 9, branch_color, -1)
    
    # Draw tracked leaf nodes (yellow - solid)
    for coord in tracked_leaf:
        r, c = int(coord[0]), int(coord[1])
        cv2.circle(overlay, (c, r), 9, leaf_color, -1)
    
    return overlay



