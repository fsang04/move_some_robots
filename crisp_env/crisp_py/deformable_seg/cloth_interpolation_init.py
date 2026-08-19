#!/usr/bin/env python3
"""
Cloth Initialization with Interpolation-based Interior Filling

Flow:
1. Find contour (ordered pixel sequence)
2. Find N corners on contour
3. FPS on each segment between corners → contour keypoints
4. Interpolation to fill interior (3 methods)
5. Generate edges from interpolation structure

Three interpolation methods:
- Radial: centroid → boundary rays
- Layered: concentric shrinking of contour
- Mean-value: weighted interpolation from boundary

Author: Auto-generated
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import cv2
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d


def pixel_to_3d(pixels_2d: np.ndarray, depth: np.ndarray, fx, fy, cx, cy) -> np.ndarray:
    """Convert 2D pixels (row, col) to 3D points."""
    points_3d = []
    for row, col in pixels_2d:
        row, col = int(row), int(col)
        z = depth[row, col]
        if z > 0:
            x = (col - cx) * z / fx
            y = (row - cy) * z / fy
            points_3d.append([x, y, z])
        else:
            points_3d.append([np.nan, np.nan, np.nan])
    return np.array(points_3d)


def extract_point_cloud(mask: np.ndarray, depth: np.ndarray, max_depth: float,
                        fx, fy, cx, cy) -> np.ndarray:
    """Extract 3D point cloud from mask and depth."""
    valid = mask & (depth > 0) & (depth < max_depth)
    rows, cols = np.where(valid)
    
    if len(rows) == 0:
        return np.array([]).reshape(0, 3)
    
    z = depth[rows, cols]
    x = (cols - cx) * z / fx
    y = (rows - cy) * z / fy
    
    return np.stack([x, y, z], axis=1)


def farthest_point_sampling(points: np.ndarray, n_samples: int, 
                            seed_points: np.ndarray = None) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) algorithm.
    Optionally start with seed points that are already "selected".
    """
    if len(points) <= n_samples:
        return points.copy()
    
    n_points = len(points)
    
    # Initialize distances
    if seed_points is not None and len(seed_points) > 0:
        # Start with seed points - compute min distance to any seed
        min_distances = np.full(n_points, np.inf)
        for seed in seed_points:
            dists = np.linalg.norm(points - seed, axis=1)
            min_distances = np.minimum(min_distances, dists)
    else:
        min_distances = np.full(n_points, np.inf)
        # Pick first point randomly
        first_idx = np.random.randint(n_points)
        min_distances = np.linalg.norm(points - points[first_idx], axis=1)
    
    selected_indices = []
    
    for _ in range(n_samples):
        # Select point with maximum minimum distance
        idx = np.argmax(min_distances)
        selected_indices.append(idx)
        
        # Update distances
        new_dists = np.linalg.norm(points - points[idx], axis=1)
        min_distances = np.minimum(min_distances, new_dists)
    
    return points[selected_indices]


def find_n_corners_on_contour(mask: np.ndarray, depth: np.ndarray, max_depth: float,
                               n_corners: int = 10, epsilon_factor: float = 0.02) -> tuple:
    """
    Find exactly N corners on contour using polygon approximation.
    Uses binary search on epsilon to get exactly N corners.
    
    Returns:
        corners_rc: (N, 2) array of corner positions in (row, col) format
        corner_contour_indices: (N,) array of indices into contour
    """
    valid_mask = mask & (depth > 0) & (depth < max_depth)
    valid_mask_uint8 = (valid_mask > 0).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(valid_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None
    
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = largest_contour.squeeze()  # (M, 2) in (col, row)
    
    if len(contour_points.shape) == 1:
        return None, None
    
    # Binary search for epsilon that gives exactly n_corners
    arc_length = cv2.arcLength(largest_contour, True)
    
    eps_low, eps_high = 0.001 * arc_length, 0.1 * arc_length
    best_epsilon = epsilon_factor * arc_length
    best_corners = None
    
    for _ in range(30):  # Binary search iterations
        eps_mid = (eps_low + eps_high) / 2
        approx = cv2.approxPolyDP(largest_contour, eps_mid, True)
        n_found = len(approx)
        
        if n_found == n_corners:
            best_epsilon = eps_mid
            best_corners = approx
            break
        elif n_found > n_corners:
            eps_low = eps_mid
        else:
            eps_high = eps_mid
        
        if best_corners is None or abs(n_found - n_corners) < abs(len(best_corners) - n_corners):
            best_corners = approx
            best_epsilon = eps_mid
    
    if best_corners is None:
        return None, None
    
    corners_cr = best_corners.squeeze()  # (N, 2) in (col, row)
    corners_rc = corners_cr[:, ::-1]  # Convert to (row, col)
    
    # Find indices on original contour
    corner_contour_indices = []
    for corner in corners_cr:
        dists = np.linalg.norm(contour_points - corner, axis=1)
        idx = np.argmin(dists)
        corner_contour_indices.append(idx)
    
    corner_contour_indices = np.array(corner_contour_indices)
    
    print(f"  Target corners: {n_corners}, Found: {len(corners_rc)} (epsilon={best_epsilon/arc_length:.4f})")
    
    return corners_rc, corner_contour_indices


def get_contour_keypoints(mask: np.ndarray, depth: np.ndarray, max_depth: float,
                           fx, fy, cx, cy,
                           n_corners: int = 10,
                           n_fps_per_segment: int = 2,
                           adaptive_fps: bool = True,
                           min_segment_length_for_fps: float = 30.0) -> tuple:
    """
    Get contour keypoints: corners + FPS on each segment.
    
    Args:
        n_fps_per_segment: Max FPS points per segment (used if adaptive_fps=False)
        adaptive_fps: If True, shorter segments get fewer FPS (can be 0)
        min_segment_length_for_fps: Minimum segment length (mm) to add 1 FPS point
    
    Returns:
        contour_keypoints_3d: (N, 3) array of 3D contour keypoints in order
        keypoint_types: list of 'corner' or 'contour' for each
        segment_info: dict with segment details
        contour_2d: original 2D contour in (col, row)
    """
    valid_mask = mask & (depth > 0) & (depth < max_depth)
    valid_mask_uint8 = (valid_mask > 0).astype(np.uint8) * 255
    
    # Get ordered contour
    contours, _ = cv2.findContours(valid_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None, None, None
    
    largest_contour = max(contours, key=cv2.contourArea)
    contour_2d = largest_contour.squeeze()  # (M, 2) in (col, row)
    n_contour = len(contour_2d)
    
    # Find corners
    corners_rc, corner_indices = find_n_corners_on_contour(mask, depth, max_depth, n_corners)
    if corners_rc is None:
        return None, None, None, None
    
    # Sort corners in contour order
    order = np.argsort(corner_indices)
    corner_indices = corner_indices[order]
    corners_rc = corners_rc[order]
    
    # Convert corners to 3D
    H, W = mask.shape
    corners_3d = pixel_to_3d(corners_rc, depth, fx, fy, cx, cy)
    
    # Filter invalid corners
    valid = ~np.any(np.isnan(corners_3d), axis=1)
    corners_3d = corners_3d[valid]
    corner_indices = corner_indices[valid]
    corners_rc = corners_rc[valid]
    n_corners_valid = len(corners_3d)
    
    print(f"  Valid corners: {n_corners_valid}")
    
    # Convert full contour to 3D
    contour_3d_full = []
    contour_3d_idx_map = []  # Maps 3D index to 2D index
    for i, (col, row) in enumerate(contour_2d):
        if 0 <= row < H and 0 <= col < W:
            z = depth[row, col]
            if 0 < z < max_depth:
                x = (col - cx) * z / fx
                y = (row - cy) * z / fy
                contour_3d_full.append([x, y, z])
                contour_3d_idx_map.append(i)
    
    contour_3d_full = np.array(contour_3d_full)
    contour_3d_idx_map = np.array(contour_3d_idx_map)
    
    # First pass: compute segment lengths to determine adaptive FPS counts
    segment_lengths = []
    for seg_idx in range(n_corners_valid):
        corner_start = corners_3d[seg_idx]
        corner_end = corners_3d[(seg_idx + 1) % n_corners_valid]
        seg_length = np.linalg.norm(corner_end - corner_start)
        segment_lengths.append(seg_length)
    
    segment_lengths = np.array(segment_lengths)
    max_seg_length = np.max(segment_lengths)
    
    print(f"  Segment lengths: min={np.min(segment_lengths):.1f}mm, max={max_seg_length:.1f}mm, mean={np.mean(segment_lengths):.1f}mm")
    
    # Build contour keypoints: corner + FPS on each segment
    all_keypoints = []
    keypoint_types = []
    segment_info = {'segments': [], 'n_corners': n_corners_valid}
    
    for seg_idx in range(n_corners_valid):
        idx_start = corner_indices[seg_idx]
        idx_end = corner_indices[(seg_idx + 1) % n_corners_valid]
        
        # Get segment 2D indices (handle wrap-around)
        if idx_start <= idx_end:
            seg_2d_indices = np.arange(idx_start, idx_end + 1)
        else:
            seg_2d_indices = np.concatenate([
                np.arange(idx_start, n_contour),
                np.arange(0, idx_end + 1)
            ])
        
        # Get segment 3D points
        segment_3d = []
        segment_contour_idx = []
        for idx_2d in seg_2d_indices:
            match = np.where(contour_3d_idx_map == idx_2d)[0]
            if len(match) > 0:
                segment_3d.append(contour_3d_full[match[0]])
                segment_contour_idx.append(idx_2d)
        
        segment_3d = np.array(segment_3d) if segment_3d else np.array([]).reshape(0, 3)
        segment_contour_idx = np.array(segment_contour_idx)
        
        corner_start = corners_3d[seg_idx]
        corner_end = corners_3d[(seg_idx + 1) % n_corners_valid]
        seg_length = segment_lengths[seg_idx]
        
        # Determine number of FPS points for this segment
        if adaptive_fps:
            # Adaptive: scale FPS count by segment length
            # n_fps = floor(segment_length / min_segment_length_for_fps)
            # But cap at n_fps_per_segment
            n_fps_this_seg = min(
                n_fps_per_segment,
                int(seg_length / min_segment_length_for_fps)
            )
        else:
            n_fps_this_seg = n_fps_per_segment
        
        seg_start_kp_idx = len(all_keypoints)
        
        # Add corner
        all_keypoints.append(corner_start)
        keypoint_types.append('corner')
        
        # FPS on segment (only if we have enough points and n_fps > 0)
        if len(segment_3d) > n_fps_this_seg + 2 and n_fps_this_seg > 0:
            fps_pts = farthest_point_sampling(
                segment_3d, n_fps_this_seg,
                seed_points=np.array([corner_start, corner_end])
            )
            
            # Order FPS points along contour
            fps_contour_idx = []
            for pt in fps_pts:
                dists = np.linalg.norm(segment_3d - pt, axis=1)
                closest = np.argmin(dists)
                fps_contour_idx.append(segment_contour_idx[closest])
            fps_contour_idx = np.array(fps_contour_idx)
            
            # Sort by contour position
            if idx_start <= idx_end:
                order = np.argsort(fps_contour_idx)
            else:
                adjusted = np.where(fps_contour_idx >= idx_start,
                                   fps_contour_idx - idx_start,
                                   fps_contour_idx + (n_contour - idx_start))
                order = np.argsort(adjusted)
            
            fps_pts = fps_pts[order]
            
            for pt in fps_pts:
                all_keypoints.append(pt)
                keypoint_types.append('contour')
        
        segment_info['segments'].append({
            'idx': seg_idx,
            'start_kp': seg_start_kp_idx,
            'end_kp': len(all_keypoints) - 1,
            'n_kps': len(all_keypoints) - seg_start_kp_idx,
            'length_mm': seg_length,
            'n_fps': n_fps_this_seg
        })
        
        print(f"    Segment {seg_idx}: length={seg_length:.1f}mm, fps={n_fps_this_seg}")
    
    all_keypoints = np.array(all_keypoints)
    
    return all_keypoints, keypoint_types, segment_info, contour_2d


# ================================================================
# INTERPOLATION METHOD 1: RADIAL
# ================================================================
def interpolate_radial(contour_keypoints: np.ndarray, 
                       keypoint_types: list,
                       n_radial_layers: int = 2) -> tuple:
    """
    Radial interpolation: connect centroid to boundary, create concentric layers.
    
    Creates:
    - Centroid point
    - n_radial_layers of points along rays from centroid to each contour keypoint
    
    Returns:
        all_keypoints: contour + interior keypoints
        edges: list of (i, j) connections
        all_types: keypoint types including 'interior'
    """
    n_contour = len(contour_keypoints)
    
    # Compute centroid
    centroid = np.mean(contour_keypoints, axis=0)
    
    # Build keypoints: contour + interior layers + centroid
    all_keypoints = list(contour_keypoints)
    all_types = list(keypoint_types)
    
    # Interior layer keypoints (along rays from centroid to each contour point)
    layer_indices = []  # layer_indices[layer][contour_idx] = keypoint_idx
    
    for layer in range(n_radial_layers):
        t = (layer + 1) / (n_radial_layers + 1)  # e.g., 0.33, 0.67 for 2 layers
        layer_kps = []
        for i in range(n_contour):
            # Interpolate between centroid and contour point
            pt = (1 - t) * centroid + t * contour_keypoints[i]
            layer_kps.append(len(all_keypoints))
            all_keypoints.append(pt)
            all_types.append('interior')
        layer_indices.append(layer_kps)
    
    # Add centroid
    centroid_idx = len(all_keypoints)
    all_keypoints.append(centroid)
    all_types.append('interior')
    
    all_keypoints = np.array(all_keypoints)
    
    # Build edges
    edges = []
    
    # 1. Contour edges (sequential along boundary)
    for i in range(n_contour):
        edges.append((i, (i + 1) % n_contour))
    
    # 2. Radial edges (centroid to innermost layer, then layer to layer, then to contour)
    for i in range(n_contour):
        # Centroid to innermost layer
        edges.append((centroid_idx, layer_indices[0][i]))
        
        # Between layers
        for layer in range(n_radial_layers - 1):
            edges.append((layer_indices[layer][i], layer_indices[layer + 1][i]))
        
        # Outermost layer to contour
        edges.append((layer_indices[-1][i], i))
    
    # 3. Circumferential edges within each layer
    for layer in range(n_radial_layers):
        for i in range(n_contour):
            edges.append((layer_indices[layer][i], layer_indices[layer][(i + 1) % n_contour]))
    
    return all_keypoints, edges, all_types


# ================================================================
# INTERPOLATION METHOD 2: LAYERED (Concentric contour shrinking)
# ================================================================
def interpolate_layered(contour_keypoints: np.ndarray,
                        keypoint_types: list,
                        n_layers: int = 2,
                        shrink_factor: float = 0.3) -> tuple:
    """
    Layered interpolation: shrink contour inward to create concentric layers.
    
    Each layer is the contour shrunk toward centroid by a factor.
    
    Returns:
        all_keypoints: contour + interior keypoints
        edges: list of (i, j) connections
        all_types: keypoint types
    """
    n_contour = len(contour_keypoints)
    centroid = np.mean(contour_keypoints, axis=0)
    
    all_keypoints = list(contour_keypoints)
    all_types = list(keypoint_types)
    
    layer_indices = []
    
    for layer in range(n_layers):
        # Shrink factor increases with each layer (closer to centroid)
        t = shrink_factor * (layer + 1) / n_layers  # e.g., 0.15, 0.30 for 2 layers
        
        layer_kps = []
        for i in range(n_contour):
            # Shrink toward centroid
            pt = (1 - t) * contour_keypoints[i] + t * centroid
            layer_kps.append(len(all_keypoints))
            all_keypoints.append(pt)
            all_types.append('interior')
        layer_indices.append(layer_kps)
    
    # Add centroid
    centroid_idx = len(all_keypoints)
    all_keypoints.append(centroid)
    all_types.append('interior')
    
    all_keypoints = np.array(all_keypoints)
    
    # Build edges
    edges = []
    
    # 1. Contour edges
    for i in range(n_contour):
        edges.append((i, (i + 1) % n_contour))
    
    # 2. Edges between layers (same position)
    for i in range(n_contour):
        # Contour to first layer
        edges.append((i, layer_indices[0][i]))
        
        # Between layers
        for layer in range(n_layers - 1):
            edges.append((layer_indices[layer][i], layer_indices[layer + 1][i]))
        
        # Innermost layer to centroid
        edges.append((layer_indices[-1][i], centroid_idx))
    
    # 3. Circumferential edges within each layer
    for layer in range(n_layers):
        for i in range(n_contour):
            edges.append((layer_indices[layer][i], layer_indices[layer][(i + 1) % n_contour]))
    
    return all_keypoints, edges, all_types


# ================================================================
# INTERPOLATION METHOD 3: MEAN-VALUE COORDINATES
# ================================================================
def interpolate_mean_value(contour_keypoints: np.ndarray,
                           keypoint_types: list,
                           n_interior_rows: int = 3,
                           n_interior_cols: int = 3) -> tuple:
    """
    Mean-value coordinate interpolation.
    
    Creates a grid of interior points, each positioned based on 
    mean-value weights from boundary points.
    
    Returns:
        all_keypoints: contour + interior keypoints
        edges: list of (i, j) connections
        all_types: keypoint types
    """
    n_contour = len(contour_keypoints)
    
    # Compute bounding box
    min_xyz = np.min(contour_keypoints, axis=0)
    max_xyz = np.max(contour_keypoints, axis=0)
    centroid = np.mean(contour_keypoints, axis=0)
    
    all_keypoints = list(contour_keypoints)
    all_types = list(keypoint_types)
    
    # Create interior grid points using mean-value interpolation
    interior_indices = []  # [row][col] = keypoint_idx
    
    for row in range(n_interior_rows):
        row_indices = []
        for col in range(n_interior_cols):
            # Normalized position in [0, 1]
            u = (col + 1) / (n_interior_cols + 1)
            v = (row + 1) / (n_interior_rows + 1)
            
            # Initial guess: lerp in bounding box
            initial_pt = min_xyz + np.array([u, v, 0.5]) * (max_xyz - min_xyz)
            
            # Mean-value coordinate weighting
            # Weight each boundary point by angle subtended
            weights = np.zeros(n_contour)
            for i in range(n_contour):
                # Simple distance-based weight (inverse distance)
                dist = np.linalg.norm(contour_keypoints[i] - initial_pt)
                weights[i] = 1.0 / (dist + 1e-6)
            
            weights /= np.sum(weights)
            
            # Interpolated point
            pt = np.sum(weights[:, None] * contour_keypoints, axis=0)
            
            # Blend with grid position for more regular spacing
            grid_pt = centroid + (np.array([u - 0.5, v - 0.5, 0]) * (max_xyz - min_xyz) * 0.6)
            pt = 0.5 * pt + 0.5 * grid_pt  # Blend for regularity
            
            row_indices.append(len(all_keypoints))
            all_keypoints.append(pt)
            all_types.append('interior')
        
        interior_indices.append(row_indices)
    
    all_keypoints = np.array(all_keypoints)
    
    # Build edges
    edges = []
    
    # 1. Contour edges
    for i in range(n_contour):
        edges.append((i, (i + 1) % n_contour))
    
    # 2. Interior grid edges (horizontal and vertical)
    for row in range(n_interior_rows):
        for col in range(n_interior_cols):
            idx = interior_indices[row][col]
            # Right neighbor
            if col < n_interior_cols - 1:
                edges.append((idx, interior_indices[row][col + 1]))
            # Down neighbor
            if row < n_interior_rows - 1:
                edges.append((idx, interior_indices[row + 1][col]))
    
    # 3. Connect interior to nearest contour points
    interior_kps = all_keypoints[n_contour:]
    for i, int_pt in enumerate(interior_kps):
        # Find 2 nearest contour points
        dists = np.linalg.norm(contour_keypoints - int_pt, axis=1)
        nearest = np.argsort(dists)[:2]
        for j in nearest:
            edges.append((n_contour + i, j))
    
    return all_keypoints, edges, all_types


# ================================================================
# INTERPOLATION METHOD 4: RECTANGLE INPAINTING + BILINEAR + CUT
# ================================================================
def initialize_rect_inpaint_bilinear(mask: np.ndarray, depth: np.ndarray, max_depth: float,
                                      fx, fy, cx, cy,
                                      grid_rows: int = 8, grid_cols: int = 8,
                                      contour_3d: np.ndarray = None) -> tuple:
    """
    Rectangle inpainting approach:
    1. Fit minimum area rotated rectangle to contour (in 2D)
    2. Create 8x8 grid via bilinear interpolation in 2D pixel space
    3. For each grid point: if inside mask, get depth; if outside, project to contour
    4. Build connected topology
    
    Args:
        mask: Binary mask of T-shirt
        depth: Depth image
        max_depth: Max valid depth
        fx, fy, cx, cy: Camera intrinsics
        grid_rows, grid_cols: Grid size (default 8x8)
        contour_3d: Pre-computed 3D contour points (optional)
    
    Returns:
        keypoints: (N, 3) array of keypoints
        edges: List of (i, j) edge tuples
        keypoint_types: List of 'corner', 'boundary', 'interior'
        grid_info: Dict with grid indices for reference
    """
    H, W = mask.shape
    valid_mask = mask & (depth > 0) & (depth < max_depth)
    valid_mask_uint8 = (valid_mask > 0).astype(np.uint8) * 255
    
    # Step 1: Get contour and fit minimum area rotated rectangle in 2D
    contours, _ = cv2.findContours(valid_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None, None, None
    
    largest_contour = max(contours, key=cv2.contourArea)
    contour_2d = largest_contour.squeeze()  # (M, 2) in (col, row)
    
    # Minimum area rotated rectangle fitted to CONTOUR
    rect = cv2.minAreaRect(largest_contour)
    box_2d = cv2.boxPoints(rect)  # 4 corners of rotated rectangle in (col, row)
    
    # Order corners consistently: find top-left-ish corner first
    # Top-left = smallest sum of (col + row)
    min_sum_idx = np.argmin(box_2d[:, 0] + box_2d[:, 1])
    box_2d_ordered = np.roll(box_2d, -min_sum_idx, axis=0)
    
    # Now box_2d_ordered[0] is top-left-ish
    # Determine if we go clockwise or counter-clockwise
    # Check if next corner is to the right (TR) or below (BL)
    v1 = box_2d_ordered[1] - box_2d_ordered[0]
    v2 = box_2d_ordered[3] - box_2d_ordered[0]
    
    # If v1 goes more right than down, order is: TL, TR, BR, BL
    # Otherwise swap to make it consistent
    if v1[0] < v2[0]:  # v1 goes more left, so swap direction
        box_2d_ordered = box_2d_ordered[[0, 3, 2, 1]]
    
    TL_2d, TR_2d, BR_2d, BL_2d = box_2d_ordered[0], box_2d_ordered[1], box_2d_ordered[2], box_2d_ordered[3]
    
    print(f"  Rectangle corners (2D): TL={TL_2d}, TR={TR_2d}, BR={BR_2d}, BL={BL_2d}")
    
    # Step 2: Create 8x8 grid in 2D pixel space via bilinear interpolation
    grid_2d = np.zeros((grid_rows, grid_cols, 2))  # (row, col, [col_px, row_px])
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            u = c / (grid_cols - 1)  # 0 to 1 along horizontal
            v = r / (grid_rows - 1)  # 0 to 1 along vertical
            
            # Bilinear interpolation in 2D
            top = (1 - u) * TL_2d + u * TR_2d
            bottom = (1 - u) * BL_2d + u * BR_2d
            pt_2d = (1 - v) * top + v * bottom
            grid_2d[r, c] = pt_2d  # (col, row) in pixel coords
    
    # Step 3: Build contour 3D lookup for projection
    # Create dense contour 3D points
    contour_3d_dense = []
    contour_2d_dense = []
    for col, row in contour_2d:
        if 0 <= row < H and 0 <= col < W:
            z = depth[int(row), int(col)]
            if 0 < z < max_depth:
                x = (col - cx) * z / fx
                y = (row - cy) * z / fy
                contour_3d_dense.append([x, y, z])
                contour_2d_dense.append([col, row])
    contour_3d_dense = np.array(contour_3d_dense)
    contour_2d_dense = np.array(contour_2d_dense)
    
    print(f"  Contour points for projection: {len(contour_3d_dense)}")
    
    # Step 4: For each grid point, determine if inside/outside and get 3D position
    grid_keypoints_3d = np.zeros((grid_rows, grid_cols, 3))
    keypoint_status = np.zeros((grid_rows, grid_cols), dtype=int)  # 0=outside, 1=inside, 2=boundary
    
    # Debug: check a few grid points
    debug_count = 0
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            col_px, row_px = grid_2d[r, c]
            col_int, row_int = int(round(col_px)), int(round(row_px))
            
            # Check if inside mask
            inside_mask = False
            if 0 <= row_int < H and 0 <= col_int < W:
                if valid_mask[row_int, col_int]:
                    inside_mask = True
            
            # Debug first few
            if debug_count < 5:
                print(f"    Grid({r},{c}): px=({col_int},{row_int}), inside={inside_mask}, mask_val={valid_mask[row_int, col_int] if 0 <= row_int < H and 0 <= col_int < W else 'OOB'}")
                debug_count += 1
            
            if inside_mask:
                # Get depth at this pixel
                z = depth[row_int, col_int]
                if 0 < z < max_depth:
                    x = (col_px - cx) * z / fx
                    y = (row_px - cy) * z / fy
                    grid_keypoints_3d[r, c] = [x, y, z]
                    keypoint_status[r, c] = 1  # Inside
                else:
                    # Invalid depth inside mask - find nearest valid
                    dists_2d = np.linalg.norm(contour_2d_dense - np.array([col_px, row_px]), axis=1)
                    nearest_idx = np.argmin(dists_2d)
                    grid_keypoints_3d[r, c] = contour_3d_dense[nearest_idx]
                    keypoint_status[r, c] = 2  # Boundary (projected)
            else:
                # Outside mask - project to nearest contour point
                dists_2d = np.linalg.norm(contour_2d_dense - np.array([col_px, row_px]), axis=1)
                nearest_idx = np.argmin(dists_2d)
                grid_keypoints_3d[r, c] = contour_3d_dense[nearest_idx]
                keypoint_status[r, c] = 2  # Boundary (projected)
    
    # Step 5: Refine boundary classification
    # Interior points adjacent to boundary/outside become boundary
    for r in range(grid_rows):
        for c in range(grid_cols):
            if keypoint_status[r, c] == 1:  # Inside
                is_boundary = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_rows and 0 <= nc < grid_cols:
                        if keypoint_status[nr, nc] == 2:  # Adjacent to boundary
                            is_boundary = True
                            break
                    else:
                        is_boundary = True  # Edge of grid
                        break
                
                if is_boundary:
                    keypoint_status[r, c] = 2
    
    print(f"  Grid status: {np.sum(keypoint_status == 0)} outside, {np.sum(keypoint_status == 1)} inside, {np.sum(keypoint_status == 2)} boundary")
    
    # Step 6: Build keypoints and edges
    grid_to_idx = {}
    all_keypoints = []
    all_types = []
    
    idx = 0
    for r in range(grid_rows):
        for c in range(grid_cols):
            grid_to_idx[(r, c)] = idx
            all_keypoints.append(grid_keypoints_3d[r, c])
            
            # Determine type
            if keypoint_status[r, c] == 1:
                all_types.append('interior')
            else:
                # Check if grid corner
                if (r == 0 or r == grid_rows - 1) and (c == 0 or c == grid_cols - 1):
                    all_types.append('corner')
                else:
                    all_types.append('boundary')
            
            idx += 1
    
    all_keypoints = np.array(all_keypoints)
    
    # Build grid edges (horizontal and vertical)
    edges = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            idx_curr = grid_to_idx[(r, c)]
            
            # Right neighbor
            if c < grid_cols - 1:
                idx_right = grid_to_idx[(r, c + 1)]
                edges.append((idx_curr, idx_right))
            
            # Down neighbor
            if r < grid_rows - 1:
                idx_down = grid_to_idx[(r + 1, c)]
                edges.append((idx_curr, idx_down))
    
    # Grid info for reference
    grid_info = {
        'grid_rows': grid_rows,
        'grid_cols': grid_cols,
        'grid_to_idx': grid_to_idx,
        'keypoint_status': keypoint_status,
        'rect_corners_2d': box_2d_ordered,
        'grid_2d': grid_2d
    }
    
    n_corners = sum(1 for t in all_types if t == 'corner')
    n_boundary = sum(1 for t in all_types if t == 'boundary')
    n_interior = sum(1 for t in all_types if t == 'interior')
    
    print(f"  Total keypoints: {len(all_keypoints)} ({n_corners} corners, {n_boundary} boundary, {n_interior} interior)")
    print(f"  Total edges: {len(edges)}")
    
    return all_keypoints, edges, all_types, grid_info


# ================================================================
# VISUALIZATION
# ================================================================
def visualize_comparison_2d(point_cloud: np.ndarray,
                            contour_3d: np.ndarray,
                            results: dict,
                            save_path: str,
                            rect_corners_3d: np.ndarray = None):
    """
    Create side-by-side comparison of interpolation methods.
    
    Shows:
    - Sparse point cloud (gray)
    - Contour line (cyan)
    - Keypoints colored by type
    - Edges
    - Rectangle outline (magenta, dashed) for rect_bilinear method
    
    Args:
        point_cloud: Full 3D point cloud
        contour_3d: 3D contour points
        results: dict with method keys
        save_path: Path to save PNG
        rect_corners_3d: (4, 3) rectangle corners in 3D for visualization
    """
    n_methods = len(results)
    n_cols = min(n_methods, 4)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    method_display_names = {
        'radial': 'Radial',
        'layered': 'Layered (Concentric)',
        'mean_value': 'Mean-Value Coords',
        'rect_bilinear': 'Rect Inpaint + Bilinear (8×8)'
    }
    
    # Downsample point cloud for visualization
    pc_sparse = point_cloud[::10]  # Every 10th point
    
    for ax_idx, (key, (keypoints, edges, types)) in enumerate(results.items()):
        ax = axes[ax_idx]
        name = method_display_names.get(key, key)
        
        # 1. Sparse point cloud (gray) - use X, Y coordinates
        ax.scatter(pc_sparse[:, 0], pc_sparse[:, 1], 
                   c='lightgray', s=1, alpha=0.5, label='Point Cloud')
        
        # 2. Contour line (cyan)
        ax.plot(contour_3d[:, 0], contour_3d[:, 1], 
                'c-', linewidth=1.5, alpha=0.7, label='Contour')
        
        # 3. Edges (blue lines)
        for i, j in edges:
            ax.plot([keypoints[i, 0], keypoints[j, 0]], 
                    [keypoints[i, 1], keypoints[j, 1]], 
                    'b-', linewidth=1.5, alpha=0.7)
        
        # 4. Keypoints by type
        corner_idx = [i for i, t in enumerate(types) if t == 'corner']
        contour_idx = [i for i, t in enumerate(types) if t == 'contour']
        boundary_idx = [i for i, t in enumerate(types) if t == 'boundary']
        interior_idx = [i for i, t in enumerate(types) if t == 'interior']
        
        if corner_idx:
            pts = keypoints[corner_idx]
            ax.scatter(pts[:, 0], pts[:, 1], c='red', s=120, marker='s', 
                      edgecolors='black', linewidths=1, zorder=10, label=f'Corners ({len(corner_idx)})')
        
        if contour_idx:
            pts = keypoints[contour_idx]
            ax.scatter(pts[:, 0], pts[:, 1], c='green', s=80, marker='o',
                      edgecolors='black', linewidths=0.5, zorder=10, label=f'Contour ({len(contour_idx)})')
        
        if boundary_idx:
            pts = keypoints[boundary_idx]
            ax.scatter(pts[:, 0], pts[:, 1], c='lime', s=80, marker='o',
                      edgecolors='black', linewidths=0.5, zorder=10, label=f'Boundary ({len(boundary_idx)})')
        
        if interior_idx:
            pts = keypoints[interior_idx]
            ax.scatter(pts[:, 0], pts[:, 1], c='orange', s=80, marker='o',
                      edgecolors='black', linewidths=0.5, zorder=10, label=f'Interior ({len(interior_idx)})')
        
        # 5. Draw rectangle outline for rect_bilinear method
        if key == 'rect_bilinear' and rect_corners_3d is not None:
            # Close the rectangle by appending first corner at end
            rect_closed = np.vstack([rect_corners_3d, rect_corners_3d[0:1]])
            ax.plot(rect_closed[:, 0], rect_closed[:, 1], 
                    'm--', linewidth=2.5, alpha=0.9, label='Inpaint Rectangle')
        
        # Stats in title
        n_kps = len(keypoints)
        n_edges = len(edges)
        ax.set_title(f'{name}\n{n_kps} keypoints, {n_edges} edges', fontsize=12)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.axis('equal')
        ax.legend(loc='upper right', fontsize=8)
    
    # Hide unused axes
    for ax_idx in range(len(results), len(axes)):
        axes[ax_idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison to: {save_path}")


def visualize_comparison_3d(results: dict, point_cloud: np.ndarray, save_path: str):
    """
    Create 3D comparison visualization using Plotly.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    n_methods = len(results)
    method_display_names = {
        'radial': 'Radial',
        'layered': 'Layered',
        'mean_value': 'Mean-Value',
        'rect_bilinear': 'Rect Bilinear (8×8)'
    }
    
    type_colors = {
        'corner': 'red',
        'contour': 'green',
        'boundary': 'lime',
        'interior': 'orange'
    }
    
    # Build subplot grid
    n_cols = min(n_methods, 4)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    specs = [[{'type': 'scatter3d'} for _ in range(n_cols)] for _ in range(n_rows)]
    subplot_titles = [method_display_names.get(k, k) for k in results.keys()]
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=specs,
        subplot_titles=subplot_titles
    )
    
    for idx, (key, (keypoints, edges, types)) in enumerate(results.items()):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        # Point cloud (downsampled)
        pc_down = point_cloud[::30]
        fig.add_trace(
            go.Scatter3d(
                x=pc_down[:, 0], y=pc_down[:, 1], z=pc_down[:, 2],
                mode='markers',
                marker=dict(size=1, color='lightgray', opacity=0.3),
                name='Point Cloud',
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )
        
        # Edges
        for i, j in edges:
            fig.add_trace(
                go.Scatter3d(
                    x=[keypoints[i, 0], keypoints[j, 0]],
                    y=[keypoints[i, 1], keypoints[j, 1]],
                    z=[keypoints[i, 2], keypoints[j, 2]],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    showlegend=False
                ),
                row=row, col=col
            )
        
        # Keypoints by type
        for kp_type in ['corner', 'contour', 'boundary', 'interior']:
            indices = [i for i, t in enumerate(types) if t == kp_type]
            if indices:
                pts = keypoints[indices]
                fig.add_trace(
                    go.Scatter3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        mode='markers',
                        marker=dict(
                            size=10 if kp_type == 'corner' else 6,
                            color=type_colors[kp_type],
                            symbol='square' if kp_type == 'corner' else 'circle'
                        ),
                        name=kp_type.capitalize(),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )
    
    fig.update_layout(
        title='Cloth Initialization: Interpolation Methods Comparison',
        width=500 * n_cols,
        height=500 * n_rows
    )
    
    fig.write_html(save_path)
    print(f"Saved 3D comparison to: {save_path}")


def main():
    # Paths
    data_dir = Path("/home/yehengz/deformable_seg/data/arm_traj5_cloth")
    rgbd_path = data_dir / "rgbd.npz"
    masks_dir = data_dir / "masks"
    output_dir = data_dir / "init_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Camera intrinsics
    fx, fy, cx, cy = 606.6875, 606.24609375, 641.7900390625, 366.8428955078125
    max_depth = 1200.0
    
    # Load data
    print("Loading data...")
    rgbd_data = np.load(str(rgbd_path))
    color_bgr = rgbd_data['color'][140]
    depth = rgbd_data['depth'][140]
    
    mask_file = masks_dir / "mask_frame_0000.npy"
    if not mask_file.exists():
        print(f"Error: {mask_file} not found. Run naive_cloth_seg.py first.")
        return
    
    mask = np.load(str(mask_file))
    valid_mask = mask & (depth > 0) & (depth < max_depth)
    
    print(f"Mask pixels: {np.sum(valid_mask)}")
    
    # Extract point cloud
    point_cloud = extract_point_cloud(mask, depth, max_depth, fx, fy, cx, cy)
    print(f"Point cloud: {len(point_cloud)} points")
    
    # Extract contour in 3D for visualization
    H, W = mask.shape
    valid_mask_uint8 = (valid_mask > 0).astype(np.uint8) * 255
    contours_cv, _ = cv2.findContours(valid_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour_cv = max(contours_cv, key=cv2.contourArea).squeeze()
    
    contour_3d_viz = []
    for col, row in largest_contour_cv[::3]:  # Sample every 3rd point
        if 0 <= row < H and 0 <= col < W:
            z = depth[row, col]
            if 0 < z < max_depth:
                x = (col - cx) * z / fx
                y = (row - cy) * z / fy
                contour_3d_viz.append([x, y, z])
    contour_3d_viz = np.array(contour_3d_viz)
    print(f"Contour 3D (for viz): {len(contour_3d_viz)} points")
    
    # ================================================================
    # STEP 1: Get contour keypoints (shared by all methods)
    # ================================================================
    print("\n" + "="*60)
    print("Step 1: Contour Keypoints (corners + adaptive FPS on segments)")
    print("="*60)
    
    N_CORNERS = 10
    N_FPS_PER_SEGMENT_MAX = 3  # Maximum FPS per segment
    MIN_SEGMENT_LENGTH_FOR_FPS = 40.0  # mm - segments shorter than this get 0 FPS
    
    contour_kps, contour_types, segment_info, contour_2d = get_contour_keypoints(
        mask, depth, max_depth, fx, fy, cx, cy,
        n_corners=N_CORNERS,
        n_fps_per_segment=N_FPS_PER_SEGMENT_MAX,
        adaptive_fps=True,
        min_segment_length_for_fps=MIN_SEGMENT_LENGTH_FOR_FPS
    )
    
    if contour_kps is None:
        print("Failed to get contour keypoints!")
        return
    
    print(f"  Contour keypoints: {len(contour_kps)}")
    print(f"  Types: {sum(1 for t in contour_types if t == 'corner')} corners, "
          f"{sum(1 for t in contour_types if t == 'contour')} contour FPS")
    
    # ================================================================
    # STEP 2: Apply three interpolation methods
    # ================================================================
    print("\n" + "="*60)
    print("Step 2: Interpolation Methods")
    print("="*60)
    
    results = {}
    
    # Method 1: Radial
    print("\nMethod 1: Radial Interpolation")
    kps_radial, edges_radial, types_radial = interpolate_radial(
        contour_kps, contour_types, n_radial_layers=2
    )
    results['radial'] = (kps_radial, edges_radial, types_radial)
    print(f"  Keypoints: {len(kps_radial)}, Edges: {len(edges_radial)}")
    
    # Method 2: Layered
    print("\nMethod 2: Layered (Concentric) Interpolation")
    kps_layered, edges_layered, types_layered = interpolate_layered(
        contour_kps, contour_types, n_layers=2, shrink_factor=0.5
    )
    results['layered'] = (kps_layered, edges_layered, types_layered)
    print(f"  Keypoints: {len(kps_layered)}, Edges: {len(edges_layered)}")
    
    # Method 3: Mean-Value
    print("\nMethod 3: Mean-Value Coordinate Interpolation")
    kps_mv, edges_mv, types_mv = interpolate_mean_value(
        contour_kps, contour_types, n_interior_rows=3, n_interior_cols=3
    )
    results['mean_value'] = (kps_mv, edges_mv, types_mv)
    print(f"  Keypoints: {len(kps_mv)}, Edges: {len(edges_mv)}")
    
    # Method 4: Rectangle Inpainting + Bilinear (8x8 grid)
    print("\nMethod 4: Rectangle Inpainting + Bilinear (8x8 grid)")
    kps_rect, edges_rect, types_rect, grid_info = initialize_rect_inpaint_bilinear(
        mask, depth, max_depth, fx, fy, cx, cy,
        grid_rows=8, grid_cols=8,
        contour_3d=contour_3d_viz
    )
    rect_corners_3d = None
    if kps_rect is not None:
        results['rect_bilinear'] = (kps_rect, edges_rect, types_rect)
        print(f"  Keypoints: {len(kps_rect)}, Edges: {len(edges_rect)}")
        
        # Get rectangle corners in 3D for visualization
        # Convert 2D rectangle corners to 3D
        if grid_info is not None and 'rect_corners_2d' in grid_info:
            rect_corners_2d = grid_info['rect_corners_2d']
            rect_corners_3d = []
            for col_px, row_px in rect_corners_2d:
                col_int, row_int = int(round(col_px)), int(round(row_px))
                # Find nearest valid depth on contour
                dists = np.sqrt((largest_contour_cv[:, 0] - col_px)**2 + 
                               (largest_contour_cv[:, 1] - row_px)**2)
                nearest_idx = np.argmin(dists)
                nearest_col, nearest_row = largest_contour_cv[nearest_idx]
                z = depth[int(nearest_row), int(nearest_col)]
                if z > 0 and z < max_depth:
                    x = (col_px - cx) * z / fx
                    y = (row_px - cy) * z / fy
                    rect_corners_3d.append([x, y, z])
            if len(rect_corners_3d) == 4:
                rect_corners_3d = np.array(rect_corners_3d)
                print(f"  Rectangle corners (3D): {rect_corners_3d.shape}")
            else:
                rect_corners_3d = None
    else:
        print("  Failed to initialize with rectangle method!")
    
    # ================================================================
    # STEP 3: Visualize comparison
    # ================================================================
    print("\n" + "="*60)
    print("Step 3: Visualization")
    print("="*60)
    
    # 2D comparison (PNG) - with sparse point cloud and contour line
    visualize_comparison_2d(
        point_cloud, contour_3d_viz, results,
        save_path=str(output_dir / "interpolation_comparison.png"),
        rect_corners_3d=rect_corners_3d
    )
    
    # 3D comparison (HTML)
    visualize_comparison_3d(
        results, point_cloud,
        save_path=str(output_dir / "interpolation_comparison_3d.html")
    )
    
    # ================================================================
    # STEP 4: Save results
    # ================================================================
    print("\n" + "="*60)
    print("Step 4: Save Results")
    print("="*60)
    
    for key in results.keys():
        kps, edges, types = results[key]
        np.save(str(output_dir / f"keypoints_{key}.npy"), kps)
        np.save(str(output_dir / f"edges_{key}.npy"), np.array(edges))
        np.save(str(output_dir / f"types_{key}.npy"), np.array(types))
        print(f"  Saved {key}: {len(kps)} keypoints, {len(edges)} edges")
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
