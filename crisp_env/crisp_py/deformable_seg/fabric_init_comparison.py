#!/usr/bin/env python3
"""
Compare two fabric initialization approaches:
1. Current: Bilinear interpolation from corners + repulsion relaxation
2. New (FPS): FPS on contour for borders, FPS on interior for interior points

This script visualizes both approaches on the first frame.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import cv2
import plotly.graph_objects as go


# ================================================================
# Grid topology constants (5x5 = 25 keypoints)
# ================================================================
GRID_ROWS = 5
GRID_COLS = 5
N_KEYPOINTS = GRID_ROWS * GRID_COLS

# Corner indices (4 corners)
CORNER_INDICES = [0, 4, 20, 24]
# Border indices (12 nodes on edges, excluding corners)
BORDER_INDICES = [1, 2, 3, 5, 9, 10, 14, 15, 19, 21, 22, 23]
# Interior indices (9 nodes inside)
INTERIOR_INDICES = [6, 7, 8, 11, 12, 13, 16, 17, 18]


def grid_pos_to_idx(row: int, col: int) -> int:
    return row * GRID_COLS + col


def idx_to_grid_pos(idx: int) -> tuple:
    return idx // GRID_COLS, idx % GRID_COLS


def pixel_to_3d(pixels_2d: np.ndarray, depth: np.ndarray, fx, fy, cx, cy) -> np.ndarray:
    """Convert 2D pixels to 3D points."""
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


def extract_contour_3d(mask: np.ndarray, depth: np.ndarray, max_depth: float,
                       fx, fy, cx, cy, sample_step: int = 5) -> np.ndarray:
    """Extract 3D contour points from mask."""
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return np.array([]).reshape(0, 3)
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = largest_contour.squeeze()
    
    if len(contour_points.shape) == 1:
        return np.array([]).reshape(0, 3)
    
    # Sample contour points
    sampled = contour_points[::sample_step]
    
    # Convert to 3D
    H, W = mask.shape
    points_3d = []
    for col, row in sampled:
        if 0 <= row < H and 0 <= col < W:
            z = depth[row, col]
            if 0 < z < max_depth:
                x = (col - cx) * z / fx
                y = (row - cy) * z / fy
                points_3d.append([x, y, z])
    
    return np.array(points_3d) if points_3d else np.array([]).reshape(0, 3)


def visualize_pointcloud_contour_plotly(point_cloud: np.ndarray, contour_3d: np.ndarray,
                                         corners_3d: np.ndarray = None,
                                         downsample_factor: int = 10,
                                         save_path: str = None):
    """
    Visualize point cloud and contour in 3D using Plotly.
    
    Args:
        point_cloud: N x 3 array of 3D points (valid mask region)
        contour_3d: M x 3 array of contour points
        corners_3d: 4 x 3 array of corner points (optional)
        downsample_factor: Downsample point cloud by this factor
        save_path: Path to save HTML file
    """
    # Downsample point cloud
    pc_downsampled = point_cloud[::downsample_factor]
    
    print(f"  Point cloud: {len(point_cloud)} -> {len(pc_downsampled)} (downsampled {downsample_factor}x)")
    print(f"  Contour: {len(contour_3d)} points")
    
    traces = []
    
    # Point cloud (green)
    traces.append(go.Scatter3d(
        x=pc_downsampled[:, 0],
        y=pc_downsampled[:, 1],
        z=pc_downsampled[:, 2],
        mode='markers',
        marker=dict(size=2, color='green', opacity=0.5),
        name=f'Point Cloud ({len(pc_downsampled)} pts)'
    ))
    
    # Contour (purple)
    traces.append(go.Scatter3d(
        x=contour_3d[:, 0],
        y=contour_3d[:, 1],
        z=contour_3d[:, 2],
        mode='markers',
        marker=dict(size=5, color='purple', opacity=0.8),
        name=f'Contour ({len(contour_3d)} pts)'
    ))
    
    # Corners (red) if provided
    if corners_3d is not None and len(corners_3d) > 0:
        traces.append(go.Scatter3d(
            x=corners_3d[:, 0],
            y=corners_3d[:, 1],
            z=corners_3d[:, 2],
            mode='markers+text',
            marker=dict(size=12, color='red', symbol='square'),
            text=['TL', 'TR', 'BR', 'BL'],
            textposition='top center',
            name='Corners'
        ))
    
    fig = go.Figure(data=traces)
    
    fig.update_layout(
        title='Point Cloud (green) and Contour (purple)',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        width=1200,
        height=900
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"  Saved to: {save_path}")
    
    return fig


def find_mask_corners(mask: np.ndarray, depth: np.ndarray, max_depth: float) -> np.ndarray:
    """Find 4 corners of mask using convex hull."""
    valid_mask = mask & (depth > 0) & (depth < max_depth)
    valid_mask_uint8 = (valid_mask > 0).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(valid_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    hull_points = hull.squeeze()
    
    if len(hull_points) < 4:
        return None
    
    # Use approxPolyDP to get 4 corners
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    # If we don't get 4 corners, try adjusting epsilon
    attempts = 0
    while len(approx) != 4 and attempts < 20:
        if len(approx) > 4:
            epsilon *= 1.2
        else:
            epsilon *= 0.8
        approx = cv2.approxPolyDP(hull, epsilon, True)
        attempts += 1
    
    if len(approx) != 4:
        # Fall back: get 4 extreme points
        pts = hull_points
        top_left = pts[np.argmin(pts[:, 0] + pts[:, 1])]
        top_right = pts[np.argmax(pts[:, 0] - pts[:, 1])]
        bottom_right = pts[np.argmax(pts[:, 0] + pts[:, 1])]
        bottom_left = pts[np.argmin(pts[:, 0] - pts[:, 1])]
        corners = np.array([top_left, top_right, bottom_right, bottom_left])
    else:
        corners = approx.squeeze()
        # Sort corners: top-left, top-right, bottom-right, bottom-left
        center = corners.mean(axis=0)
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        corners = corners[sorted_indices]
        
        # Reorder to start from top-left
        top_indices = np.argsort(corners[:, 1])[:2]
        if corners[top_indices[0], 0] < corners[top_indices[1], 0]:
            tl_idx = top_indices[0]
        else:
            tl_idx = top_indices[1]
        corners = np.roll(corners, -tl_idx, axis=0)
    
    # Convert to row, col format
    corners_rc = np.array([[c[1], c[0]] for c in corners])
    return corners_rc


def farthest_point_sampling(points: np.ndarray, n_samples: int, 
                            seed_points: np.ndarray = None) -> np.ndarray:
    """
    Farthest Point Sampling.
    
    Args:
        points: N × 3 points to sample from
        n_samples: Number of samples to return
        seed_points: Optional K × 3 points to start with (included in distance computation but not returned)
    
    Returns:
        sampled: n_samples × 3 sampled points
    """
    N = len(points)
    if N == 0:
        return np.array([]).reshape(0, 3)
    
    if n_samples >= N:
        return points.copy()
    
    # Initialize distances
    if seed_points is not None and len(seed_points) > 0:
        # Start with distances to seed points
        distances = np.full(N, np.inf)
        for seed in seed_points:
            dist_to_seed = np.linalg.norm(points - seed, axis=1)
            distances = np.minimum(distances, dist_to_seed)
    else:
        distances = np.full(N, np.inf)
        # Start from random point
        first_idx = np.random.randint(N)
        distances = np.linalg.norm(points - points[first_idx], axis=1)
    
    sampled_indices = []
    
    for _ in range(n_samples):
        # Select farthest point
        farthest_idx = np.argmax(distances)
        sampled_indices.append(farthest_idx)
        
        # Update distances
        dist_to_new = np.linalg.norm(points - points[farthest_idx], axis=1)
        distances = np.minimum(distances, dist_to_new)
    
    return points[sampled_indices]


def farthest_point_sampling_with_indices(points: np.ndarray, n_samples: int,
                                         seed_points: np.ndarray = None) -> tuple:
    """
    FPS returning both points and indices.
    """
    N = len(points)
    if N == 0:
        return np.array([]).reshape(0, 3), []
    
    if n_samples >= N:
        return points.copy(), list(range(N))
    
    if seed_points is not None and len(seed_points) > 0:
        distances = np.full(N, np.inf)
        for seed in seed_points:
            dist_to_seed = np.linalg.norm(points - seed, axis=1)
            distances = np.minimum(distances, dist_to_seed)
    else:
        distances = np.full(N, np.inf)
        first_idx = np.random.randint(N)
        distances = np.linalg.norm(points - points[first_idx], axis=1)
    
    sampled_indices = []
    
    for _ in range(n_samples):
        farthest_idx = np.argmax(distances)
        sampled_indices.append(farthest_idx)
        dist_to_new = np.linalg.norm(points - points[farthest_idx], axis=1)
        distances = np.minimum(distances, dist_to_new)
    
    return points[sampled_indices], sampled_indices


# ================================================================
# APPROACH 1: Current (Bilinear interpolation + repulsion)
# ================================================================
def initialize_bilinear(corners_3d: np.ndarray, point_cloud: np.ndarray) -> np.ndarray:
    """
    Current approach: Bilinear interpolation from corners.
    """
    top_left, top_right, bottom_right, bottom_left = corners_3d
    
    keypoints = np.zeros((N_KEYPOINTS, 3), dtype=np.float64)
    
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            u = col / (GRID_COLS - 1)
            v = row / (GRID_ROWS - 1)
            
            top = (1 - u) * top_left + u * top_right
            bottom = (1 - u) * bottom_left + u * bottom_right
            point = (1 - v) * top + v * bottom
            
            idx = grid_pos_to_idx(row, col)
            keypoints[idx] = point
    
    # Snap to point cloud
    if len(point_cloud) > 0:
        nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
        nn.fit(point_cloud)
        distances, indices = nn.kneighbors(keypoints)
        for i in range(len(keypoints)):
            if distances[i, 0] < 50.0:  # max_distance
                keypoints[i] = point_cloud[indices[i, 0]]
    
    return keypoints


# ================================================================
# APPROACH 2: FPS-based initialization
# ================================================================
def initialize_fps(corners_3d: np.ndarray, contour_3d: np.ndarray, 
                   point_cloud: np.ndarray) -> np.ndarray:
    """
    New approach: FPS on contour for borders, FPS on interior for interior.
    
    Steps:
    1. Place corners at detected corners (fixed)
    2. Use FPS on contour (with corners as seeds) to get 12 border points
    3. Use FPS on interior point cloud (with corners+borders as seeds) to get 9 interior points
    """
    keypoints = np.zeros((N_KEYPOINTS, 3), dtype=np.float64)
    
    # Step 1: Place corners
    # corners_3d order: top-left, top-right, bottom-right, bottom-left
    # Grid corners: 0 (TL), 4 (TR), 24 (BR), 20 (BL)
    keypoints[0] = corners_3d[0]   # top-left
    keypoints[4] = corners_3d[1]   # top-right
    keypoints[24] = corners_3d[2]  # bottom-right
    keypoints[20] = corners_3d[3]  # bottom-left
    
    # Step 2: FPS on contour to get border points
    if len(contour_3d) < 12:
        print(f"Warning: Not enough contour points ({len(contour_3d)}), falling back to bilinear")
        return initialize_bilinear(corners_3d, point_cloud)
    
    # FPS with corners as seeds
    border_points, _ = farthest_point_sampling_with_indices(
        contour_3d, 
        n_samples=12,  # 12 border nodes
        seed_points=corners_3d
    )
    
    # Assign border points to border indices
    # We need to order them properly along the border
    # Border order: top (1,2,3), right (9,14,19), bottom (23,22,21), left (15,10,5)
    border_order = [1, 2, 3, 9, 14, 19, 23, 22, 21, 15, 10, 5]
    
    # Order border points by angle from center
    center = corners_3d.mean(axis=0)
    
    # Project to XY for angle computation (use X-Z plane since Y is usually up)
    angles = np.arctan2(border_points[:, 2] - center[2], border_points[:, 0] - center[0])
    sorted_border_indices = np.argsort(angles)
    sorted_border = border_points[sorted_border_indices]
    
    # Find the starting point (closest to corner 0 direction)
    corner0_angle = np.arctan2(corners_3d[0, 2] - center[2], corners_3d[0, 0] - center[0])
    angle_diffs = np.abs(angles[sorted_border_indices] - corner0_angle)
    angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
    start_idx = np.argmin(angle_diffs)
    
    # Roll to start from corner 0 direction
    sorted_border = np.roll(sorted_border, -start_idx, axis=0)
    
    # Assign to border indices
    for i, idx in enumerate(border_order):
        if i < len(sorted_border):
            keypoints[idx] = sorted_border[i]
    
    # Step 3: FPS on interior for interior points
    # First, identify interior points from point cloud
    # Interior is roughly the area between corners
    
    # Build convex hull of corners to define interior
    fixed_points = np.vstack([corners_3d, border_points])
    
    # Use FPS with corners and borders as seeds
    interior_points, _ = farthest_point_sampling_with_indices(
        point_cloud,
        n_samples=9,  # 9 interior nodes
        seed_points=fixed_points
    )
    
    # Assign to interior indices (row-major order)
    for i, idx in enumerate(INTERIOR_INDICES):
        if i < len(interior_points):
            keypoints[idx] = interior_points[i]
    
    return keypoints


def initialize_fps_ordered(corners_3d: np.ndarray, contour_3d: np.ndarray,
                           point_cloud: np.ndarray) -> np.ndarray:
    """
    FPS-based initialization with proper ordering along contour edges.
    
    For border nodes:
    - Find where corners intersect the contour (nearest contour points)
    - Split contour into 4 segments between corners
    - Use FPS on each segment to get 3 evenly spaced points
    
    For interior nodes:
    - FPS on point cloud with all boundary points as seeds
    """
    keypoints = np.zeros((N_KEYPOINTS, 3), dtype=np.float64)
    
    # Step 1: Place corners
    # corners_3d order: top-left (0), top-right (1), bottom-right (2), bottom-left (3)
    # Grid corners: 0 (TL), 4 (TR), 24 (BR), 20 (BL)
    keypoints[0] = corners_3d[0]   # top-left
    keypoints[4] = corners_3d[1]   # top-right
    keypoints[24] = corners_3d[2]  # bottom-right
    keypoints[20] = corners_3d[3]  # bottom-left
    
    if len(contour_3d) < 12:
        print(f"Warning: Not enough contour points ({len(contour_3d)}), falling back to bilinear")
        return initialize_bilinear(corners_3d, point_cloud)
    
    # Step 2: Find corner positions on contour
    # For each corner, find nearest contour point index
    nn_contour = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn_contour.fit(contour_3d)
    _, corner_contour_indices = nn_contour.kneighbors(corners_3d)
    corner_contour_indices = corner_contour_indices.flatten()
    
    print(f"  Corner indices on contour: {corner_contour_indices}")
    
    # Step 3: Segment contour into 4 edges by corner positions
    # The contour is a closed loop, so we need to handle wrap-around
    n_contour = len(contour_3d)
    
    # Sort corner indices by their position on contour (they should be in order)
    sorted_order = np.argsort(corner_contour_indices)
    
    # Edge definitions mapping: corner pair -> grid indices for that edge
    # We need to figure out which corners are adjacent on the contour
    # corners_3d: TL(0), TR(1), BR(2), BL(3)
    # Expected contour order (clockwise from TL): TL -> TR -> BR -> BL -> TL
    
    edge_definitions = [
        (0, 1, [1, 2, 3]),      # top edge: TL -> TR
        (1, 2, [9, 14, 19]),    # right edge: TR -> BR
        (2, 3, [23, 22, 21]),   # bottom edge: BR -> BL
        (3, 0, [15, 10, 5]),    # left edge: BL -> TL
    ]
    
    for edge_id, (c_start, c_end, grid_indices) in enumerate(edge_definitions):
        idx_start = corner_contour_indices[c_start]
        idx_end = corner_contour_indices[c_end]
        
        # Extract contour segment between these corners
        if idx_start < idx_end:
            segment = contour_3d[idx_start:idx_end+1]
        else:
            # Wrap around
            segment = np.vstack([contour_3d[idx_start:], contour_3d[:idx_end+1]])
        
        print(f"  Edge {edge_id} ({c_start}->{c_end}): {len(segment)} contour points")
        
        if len(segment) >= 5:  # Need at least 5 points: 2 corners + 3 border
            # Sample at 1/4, 2/4, 3/4 positions along the segment
            # The segment is already ordered from corner_start to corner_end
            n_seg = len(segment)
            
            # Get points at approximately 25%, 50%, 75% along the segment
            sample_indices = [
                int(n_seg * 0.25),
                int(n_seg * 0.50),
                int(n_seg * 0.75),
            ]
            
            sampled = segment[sample_indices]
            
            for i, idx in enumerate(grid_indices):
                keypoints[idx] = sampled[i]
        else:
            # Fall back to linear interpolation
            print(f"    Not enough points, using linear interpolation")
            corner_start = corners_3d[c_start]
            corner_end = corners_3d[c_end]
            for i, idx in enumerate(grid_indices):
                t = (i + 1) / 4.0
                keypoints[idx] = (1 - t) * corner_start + t * corner_end
    
    # Step 4: Initialize interior points using FPS on point cloud with corners+borders as anchors
    if len(point_cloud) > 0:
        # Collect all boundary points (corners + borders) as anchors
        boundary_indices = CORNER_INDICES + BORDER_INDICES
        anchor_points = keypoints[boundary_indices]
        
        print(f"  Interior FPS: using {len(anchor_points)} boundary anchors, sampling {len(INTERIOR_INDICES)} from {len(point_cloud)} points")
        
        # FPS to get 9 interior points
        interior_fps, _ = farthest_point_sampling_with_indices(
            point_cloud,
            n_samples=len(INTERIOR_INDICES),  # 9 interior points
            seed_points=anchor_points
        )
        
        # Assign FPS points to interior grid positions using greedy assignment
        # Compute expected positions for each interior index (bilinear from corners)
        expected_positions = []
        for idx in INTERIOR_INDICES:
            row, col = idx_to_grid_pos(idx)
            u = col / (GRID_COLS - 1)
            v = row / (GRID_ROWS - 1)
            
            top = (1 - u) * keypoints[0] + u * keypoints[4]
            bottom = (1 - u) * keypoints[20] + u * keypoints[24]
            expected = (1 - v) * top + v * bottom
            expected_positions.append(expected)
        
        expected_positions = np.array(expected_positions)
        
        # Greedy assignment: for each grid position, assign nearest unassigned FPS point
        assigned = set()
        for i, idx in enumerate(INTERIOR_INDICES):
            expected = expected_positions[i]
            best_dist = np.inf
            best_j = -1
            for j in range(len(interior_fps)):
                if j not in assigned:
                    dist = np.linalg.norm(interior_fps[j] - expected)
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j
            if best_j >= 0:
                keypoints[idx] = interior_fps[best_j]
                assigned.add(best_j)
            else:
                # Fallback to bilinear
                keypoints[idx] = expected
    
    return keypoints


# ================================================================
# APPROACH 3: Pure FPS - sample 25 points directly from point cloud
# ================================================================
def initialize_pure_fps(point_cloud: np.ndarray, corners_3d: np.ndarray) -> np.ndarray:
    """
    Pure FPS approach: Sample 25 points directly from point cloud using FPS,
    then assign them to grid positions based on their spatial arrangement.
    
    Steps:
    1. FPS on point cloud to get 25 evenly distributed points
    2. Assign points to grid indices by sorting them into a 5x5 grid layout
    """
    if len(point_cloud) < N_KEYPOINTS:
        print(f"Warning: Not enough points ({len(point_cloud)}), returning zeros")
        return np.zeros((N_KEYPOINTS, 3), dtype=np.float64)
    
    # Step 1: FPS to get 25 points
    fps_points = farthest_point_sampling(point_cloud, n_samples=N_KEYPOINTS)
    
    print(f"  FPS sampled {len(fps_points)} points from {len(point_cloud)} point cloud")
    
    # Step 2: Assign to grid by sorting spatially
    # We need to arrange the 25 points into a 5x5 grid
    # Strategy: use the detected corners to define a coordinate frame
    
    # Compute the fabric's local coordinate system from corners
    # corners_3d: TL(0), TR(1), BR(2), BL(3)
    top_left, top_right, bottom_right, bottom_left = corners_3d
    
    # X-axis: left to right (TL -> TR direction)
    x_axis = top_right - top_left
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    
    # Y-axis: top to bottom (TL -> BL direction)
    y_axis = bottom_left - top_left
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
    
    # Project each FPS point onto this coordinate frame
    # Use top_left as origin
    origin = top_left
    
    local_coords = []
    for pt in fps_points:
        rel = pt - origin
        local_x = np.dot(rel, x_axis)
        local_y = np.dot(rel, y_axis)
        local_coords.append([local_x, local_y])
    
    local_coords = np.array(local_coords)
    
    # Normalize to [0, 1] range
    x_min, x_max = local_coords[:, 0].min(), local_coords[:, 0].max()
    y_min, y_max = local_coords[:, 1].min(), local_coords[:, 1].max()
    
    if x_max > x_min:
        local_coords[:, 0] = (local_coords[:, 0] - x_min) / (x_max - x_min)
    if y_max > y_min:
        local_coords[:, 1] = (local_coords[:, 1] - y_min) / (y_max - y_min)
    
    # Assign each FPS point to expected grid position
    # Expected positions in normalized coordinates
    keypoints = np.zeros((N_KEYPOINTS, 3), dtype=np.float64)
    assigned_fps = [False] * len(fps_points)
    
    for grid_idx in range(N_KEYPOINTS):
        row, col = idx_to_grid_pos(grid_idx)
        expected_x = col / (GRID_COLS - 1)
        expected_y = row / (GRID_ROWS - 1)
        
        # Find nearest unassigned FPS point
        best_dist = np.inf
        best_fps_idx = -1
        for fps_idx in range(len(fps_points)):
            if not assigned_fps[fps_idx]:
                dist = np.sqrt((local_coords[fps_idx, 0] - expected_x)**2 + 
                              (local_coords[fps_idx, 1] - expected_y)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_fps_idx = fps_idx
        
        if best_fps_idx >= 0:
            keypoints[grid_idx] = fps_points[best_fps_idx]
            assigned_fps[best_fps_idx] = True
    
    return keypoints


# ================================================================
# APPROACH 4: FPS on contour segments for borders + bilinear snap for interior
# ================================================================
def initialize_contour_fps_bilinear_interior(corners_3d: np.ndarray, contour_3d: np.ndarray,
                                              point_cloud: np.ndarray) -> np.ndarray:
    """
    Best approach:
    - Corners: from detected corners (on contour)
    - Border nodes: FPS on contour segments (ensures they're ON the contour)
    - Interior nodes: bilinear interpolation from corners, snapped to point cloud
    
    This ensures:
    1. All border nodes lie exactly on the fabric contour
    2. Interior nodes are on the actual point cloud
    """
    keypoints = np.zeros((N_KEYPOINTS, 3), dtype=np.float64)
    
    # Step 1: Place corners
    keypoints[0] = corners_3d[0]   # top-left
    keypoints[4] = corners_3d[1]   # top-right
    keypoints[24] = corners_3d[2]  # bottom-right
    keypoints[20] = corners_3d[3]  # bottom-left
    
    print(f"  Corners placed at grid indices 0, 4, 20, 24")
    
    if len(contour_3d) < 12:
        print(f"Warning: Not enough contour points ({len(contour_3d)}), falling back to bilinear")
        return initialize_bilinear(corners_3d, point_cloud)
    
    # Step 2: Find corner positions on contour
    nn_contour = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn_contour.fit(contour_3d)
    _, corner_contour_indices = nn_contour.kneighbors(corners_3d)
    corner_contour_indices = corner_contour_indices.flatten()
    
    print(f"  Corner indices on contour: {corner_contour_indices}")
    
    # Step 3: Segment contour and use FPS on each segment
    n_contour = len(contour_3d)
    
    edge_definitions = [
        (0, 1, [1, 2, 3]),      # top edge: TL -> TR
        (1, 2, [9, 14, 19]),    # right edge: TR -> BR
        (2, 3, [23, 22, 21]),   # bottom edge: BR -> BL
        (3, 0, [15, 10, 5]),    # left edge: BL -> TL
    ]
    
    for edge_id, (c_start, c_end, grid_indices) in enumerate(edge_definitions):
        idx_start = corner_contour_indices[c_start]
        idx_end = corner_contour_indices[c_end]
        
        # Extract contour segment - choose SHORTER path
        if idx_start <= idx_end:
            forward_len = idx_end - idx_start + 1
            backward_len = n_contour - idx_end + idx_start + 1
        else:
            forward_len = n_contour - idx_start + idx_end + 1
            backward_len = idx_start - idx_end + 1
        
        if forward_len <= backward_len:
            # Forward direction
            if idx_start <= idx_end:
                segment = contour_3d[idx_start:idx_end+1]
            else:
                segment = np.vstack([contour_3d[idx_start:], contour_3d[:idx_end+1]])
        else:
            # Backward direction
            if idx_start >= idx_end:
                segment = contour_3d[idx_end:idx_start+1][::-1]
            else:
                segment = np.vstack([contour_3d[idx_end:], contour_3d[:idx_start+1]])[::-1]
        
        print(f"  Edge {edge_id} ({c_start}->{c_end}): {len(segment)} pts (fwd={forward_len}, bwd={backward_len})")
        
        if len(segment) >= 5:
            # Use FPS with corners as anchors to get 3 evenly spaced points
            anchor_start = corners_3d[c_start]
            anchor_end = corners_3d[c_end]
            
            fps_points = farthest_point_sampling(
                segment,
                n_samples=3,
                seed_points=np.array([anchor_start, anchor_end])
            )
            
            # Order FPS points by distance from start corner
            dists_from_start = np.linalg.norm(fps_points - anchor_start, axis=1)
            order = np.argsort(dists_from_start)
            fps_points = fps_points[order]
            
            for i, idx in enumerate(grid_indices):
                keypoints[idx] = fps_points[i]
        else:
            # Fall back to linear interpolation
            print(f"    Not enough points, using linear interpolation")
            corner_start = corners_3d[c_start]
            corner_end = corners_3d[c_end]
            for i, idx in enumerate(grid_indices):
                t = (i + 1) / 4.0
                keypoints[idx] = (1 - t) * corner_start + t * corner_end
    
    # Step 4: Interior points - bilinear interpolation, then snap to point cloud
    print(f"  Interior: bilinear interpolation + snap to point cloud")
    
    if len(point_cloud) > 0:
        nn_cloud = NearestNeighbors(n_neighbors=1, algorithm='auto')
        nn_cloud.fit(point_cloud)
        
        for idx in INTERIOR_INDICES:
            row, col = idx_to_grid_pos(idx)
            u = col / (GRID_COLS - 1)
            v = row / (GRID_ROWS - 1)
            
            # Bilinear interpolation from corners
            top = (1 - u) * keypoints[0] + u * keypoints[4]
            bottom = (1 - u) * keypoints[20] + u * keypoints[24]
            expected = (1 - v) * top + v * bottom
            
            # Snap to nearest point in point cloud (no distance threshold)
            _, nearest_idx = nn_cloud.kneighbors(expected.reshape(1, -1))
            keypoints[idx] = point_cloud[nearest_idx[0, 0]]
    
    return keypoints


def visualize_comparison(mask, depth, corners_3d, contour_3d, point_cloud,
                         keypoints_bilinear, keypoints_fps, keypoints_pure_fps, 
                         keypoints_contour_bilinear, intrinsics, save_path=None):
    """Visualize all four initialization approaches in 3D."""
    from mpl_toolkits.mplot3d import Axes3D
    
    # Build edges
    edges = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            idx = grid_pos_to_idx(row, col)
            if col < GRID_COLS - 1:
                edges.append((idx, grid_pos_to_idx(row, col + 1)))
            if row < GRID_ROWS - 1:
                edges.append((idx, grid_pos_to_idx(row + 1, col)))
    
    fig = plt.figure(figsize=(20, 16))
    
    # Subsample point cloud for visualization
    pc_subsample = point_cloud[::50] if len(point_cloud) > 2000 else point_cloud
    
    def draw_keypoints_3d(ax, keypoints, title):
        # Draw point cloud (light gray, small)
        ax.scatter(pc_subsample[:, 0], pc_subsample[:, 1], pc_subsample[:, 2],
                   c='lightgray', s=1, alpha=0.3, label='Point Cloud')
        
        # Draw contour (cyan)
        ax.scatter(contour_3d[:, 0], contour_3d[:, 1], contour_3d[:, 2],
                   c='cyan', s=5, alpha=0.5, label='Contour')
        
        # Draw edges
        for i, j in edges:
            if not np.any(np.isnan(keypoints[i])) and not np.any(np.isnan(keypoints[j])):
                ax.plot([keypoints[i, 0], keypoints[j, 0]], 
                       [keypoints[i, 1], keypoints[j, 1]], 
                       [keypoints[i, 2], keypoints[j, 2]], 'b-', linewidth=1.5)
        
        # Draw keypoints
        for idx in CORNER_INDICES:
            if not np.any(np.isnan(keypoints[idx])):
                ax.scatter(keypoints[idx, 0], keypoints[idx, 1], keypoints[idx, 2],
                          c='red', s=150, marker='s', zorder=10, edgecolors='black')
        for idx in BORDER_INDICES:
            if not np.any(np.isnan(keypoints[idx])):
                ax.scatter(keypoints[idx, 0], keypoints[idx, 1], keypoints[idx, 2],
                          c='green', s=100, marker='o', zorder=10, edgecolors='black')
        for idx in INTERIOR_INDICES:
            if not np.any(np.isnan(keypoints[idx])):
                ax.scatter(keypoints[idx, 0], keypoints[idx, 1], keypoints[idx, 2],
                          c='blue', s=80, marker='o', zorder=10, edgecolors='black')
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        # Set consistent view angle
        ax.view_init(elev=-70, azim=-90)
    
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    draw_keypoints_3d(ax1, keypoints_bilinear, 'Approach 1: Bilinear (may be off contour)')
    
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    draw_keypoints_3d(ax2, keypoints_fps, 'Approach 2: FPS Contour + FPS Interior')
    
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    draw_keypoints_3d(ax3, keypoints_pure_fps, 'Approach 3: Pure FPS (25 pts)')
    
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    draw_keypoints_3d(ax4, keypoints_contour_bilinear, 'Approach 4: FPS Contour + Bilinear Interior')
    
    # Add legend
    ax1.legend(['Point Cloud', 'Contour', 'Edges', 'Corners', 'Border', 'Interior'], 
               loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {save_path}")
    
    plt.show()


def compute_edge_stats(keypoints: np.ndarray, edges: list) -> dict:
    """Compute edge length statistics."""
    lengths = []
    for i, j in edges:
        length = np.linalg.norm(keypoints[i] - keypoints[j])
        lengths.append(length)
    
    lengths = np.array(lengths)
    return {
        'mean': np.mean(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths),
        'cv': np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0,
    }


def main():
    # Paths
    data_dir = Path("/home/yehengz/deformable_seg/data")
    tracking_data_path = data_dir / "full" / "tracking_fabric2_data.npy"
    masks_dir = data_dir / "arm_traj4_fabric" / "masks"
    output_dir = data_dir / "arm_traj4_fabric" / "init_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Camera intrinsics
    INTRINSICS = (606.6875, 606.24609375, 641.7900390625, 366.8428955078125)
    fx, fy, cx, cy = INTRINSICS
    max_depth = 1100.0
    
    # Load tracking data
    print("Loading tracking data...")
    tracking_data = np.load(str(tracking_data_path), allow_pickle=True).item()
    frame_keys = sorted([k for k in tracking_data.keys() if isinstance(k, int)])
    print(f"Found {len(frame_keys)} frames")
    
    # Load first frame's data
    frame_key = frame_keys[0]
    frame_data = tracking_data[frame_key]
    depth = frame_data['transformed_depth']
    
    # Load mask and apply depth thresholding
    mask_file = masks_dir / "mask_frame_0000.npy"
    mask_raw = np.load(str(mask_file))
    
    # IMPORTANT: Only keep pixels with valid depth (0 < depth < max_depth)
    valid_depth = (depth > 0) & (depth < max_depth)
    mask = mask_raw & valid_depth
    
    print(f"Raw mask pixels: {np.sum(mask_raw)}, Depth-thresholded: {np.sum(mask)}")
    
    # Find corners
    corners_2d = find_mask_corners(mask, depth, max_depth)
    if corners_2d is None:
        print("Failed to find corners!")
        return
    
    corners_3d = pixel_to_3d(corners_2d, depth, fx, fy, cx, cy)
    print(f"Corners 3D:\n{corners_3d}")
    
    # Extract contour
    contour_3d = extract_contour_3d(mask, depth, max_depth, fx, fy, cx, cy, sample_step=3)
    print(f"Contour 3D: {len(contour_3d)} points")
    
    # Extract point cloud
    point_cloud = extract_point_cloud(mask, depth, max_depth, fx, fy, cx, cy)
    print(f"Point cloud: {len(point_cloud)} points")
    
    # Visualize point cloud and contour in Plotly
    print("\n" + "="*60)
    print("Visualizing Point Cloud and Contour (Plotly)")
    print("="*60)
    visualize_pointcloud_contour_plotly(
        point_cloud, contour_3d, corners_3d,
        downsample_factor=10,
        save_path=str(output_dir / "pointcloud_contour.html")
    )
    
    # Build edges
    edges = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            idx = grid_pos_to_idx(row, col)
            if col < GRID_COLS - 1:
                edges.append((idx, grid_pos_to_idx(row, col + 1)))
            if row < GRID_ROWS - 1:
                edges.append((idx, grid_pos_to_idx(row + 1, col)))
    
    # Approach 1: Bilinear interpolation
    print("\n" + "="*60)
    print("Approach 1: Bilinear Interpolation")
    print("="*60)
    keypoints_bilinear = initialize_bilinear(corners_3d, point_cloud)
    stats_bilinear = compute_edge_stats(keypoints_bilinear, edges)
    print(f"Edge lengths: mean={stats_bilinear['mean']:.2f}mm, std={stats_bilinear['std']:.2f}mm")
    print(f"             min={stats_bilinear['min']:.2f}mm, max={stats_bilinear['max']:.2f}mm")
    print(f"             CV={stats_bilinear['cv']:.3f}")
    
    # Approach 2: FPS-based
    print("\n" + "="*60)
    print("Approach 2: FPS on Contour + Interior")
    print("="*60)
    keypoints_fps = initialize_fps_ordered(corners_3d, contour_3d, point_cloud)
    stats_fps = compute_edge_stats(keypoints_fps, edges)
    print(f"Edge lengths: mean={stats_fps['mean']:.2f}mm, std={stats_fps['std']:.2f}mm")
    print(f"             min={stats_fps['min']:.2f}mm, max={stats_fps['max']:.2f}mm")
    print(f"             CV={stats_fps['cv']:.3f}")
    
    # Approach 3: Pure FPS
    print("\n" + "="*60)
    print("Approach 3: Pure FPS (25 points directly)")
    print("="*60)
    keypoints_pure_fps = initialize_pure_fps(point_cloud, corners_3d)
    stats_pure_fps = compute_edge_stats(keypoints_pure_fps, edges)
    print(f"Edge lengths: mean={stats_pure_fps['mean']:.2f}mm, std={stats_pure_fps['std']:.2f}mm")
    print(f"             min={stats_pure_fps['min']:.2f}mm, max={stats_pure_fps['max']:.2f}mm")
    print(f"             CV={stats_pure_fps['cv']:.3f}")
    
    # Approach 4: FPS on contour + bilinear interior
    print("\n" + "="*60)
    print("Approach 4: FPS Contour + Bilinear Interior (snap to point cloud)")
    print("="*60)
    keypoints_contour_bilinear = initialize_contour_fps_bilinear_interior(corners_3d, contour_3d, point_cloud)
    stats_contour_bilinear = compute_edge_stats(keypoints_contour_bilinear, edges)
    print(f"Edge lengths: mean={stats_contour_bilinear['mean']:.2f}mm, std={stats_contour_bilinear['std']:.2f}mm")
    print(f"             min={stats_contour_bilinear['min']:.2f}mm, max={stats_contour_bilinear['max']:.2f}mm")
    print(f"             CV={stats_contour_bilinear['cv']:.3f}")
    
    # Compare
    print("\n" + "="*60)
    print("Comparison")
    print("="*60)
    print(f"Edge length CV (lower is more uniform):")
    print(f"  Bilinear:         {stats_bilinear['cv']:.4f}")
    print(f"  FPS+FPS Int:      {stats_fps['cv']:.4f}")
    print(f"  Pure FPS:         {stats_pure_fps['cv']:.4f}")
    print(f"  FPS+Bilinear:     {stats_contour_bilinear['cv']:.4f}")
    
    # Visualize
    visualize_comparison(
        mask, depth, corners_3d, contour_3d, point_cloud,
        keypoints_bilinear, keypoints_fps, keypoints_pure_fps, keypoints_contour_bilinear, 
        INTRINSICS, save_path=str(output_dir / "init_comparison.png")
    )
    
    # Save results
    np.save(str(output_dir / "keypoints_bilinear.npy"), keypoints_bilinear)
    np.save(str(output_dir / "keypoints_fps.npy"), keypoints_fps)
    np.save(str(output_dir / "keypoints_pure_fps.npy"), keypoints_pure_fps)
    np.save(str(output_dir / "keypoints_contour_bilinear.npy"), keypoints_contour_bilinear)
    print(f"\nSaved keypoints to: {output_dir}")


if __name__ == "__main__":
    main()
