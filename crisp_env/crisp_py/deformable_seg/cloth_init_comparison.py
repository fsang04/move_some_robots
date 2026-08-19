#!/usr/bin/env python3
"""
Compare cloth initialization approaches:
1. Current: Bilinear interpolation from corners + repulsion relaxation
2. New (FPS): FPS on contour for borders, FPS on interior for interior points

This script visualizes both approaches on the first frame.
Adapted for cloth data stored in NPZ format.
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
    """Find 4 corners of mask using convex hull (for rectangular shapes)."""
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


def find_adaptive_corners(mask: np.ndarray, depth: np.ndarray, max_depth: float,
                          min_angle_threshold: float = 60.0,
                          smoothing_window: int = 15) -> np.ndarray:
    """
    Find corners adaptively on contour based on curvature.
    Works for irregular shapes like T-shirts.
    
    Args:
        mask: Binary mask
        depth: Depth image
        max_depth: Maximum depth threshold
        min_angle_threshold: Minimum angle (degrees) to consider as corner (smaller = sharper corner)
        smoothing_window: Window size for contour smoothing
    
    Returns:
        corners_rc: N x 2 array of corner positions (row, col)
    """
    valid_mask = mask & (depth > 0) & (depth < max_depth)
    valid_mask_uint8 = (valid_mask > 0).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(valid_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = largest_contour.squeeze()
    
    if len(contour_points) < smoothing_window * 2:
        return None
    
    n_points = len(contour_points)
    
    # Compute curvature at each point using angle between neighboring vectors
    angles = []
    for i in range(n_points):
        # Get neighboring points with some offset for stability
        offset = smoothing_window
        p_prev = contour_points[(i - offset) % n_points]
        p_curr = contour_points[i]
        p_next = contour_points[(i + offset) % n_points]
        
        # Vectors
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        
        # Compute angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        angles.append(angle)
    
    angles = np.array(angles)
    
    # Find local minima in angles (sharp corners have small angles)
    corner_indices = []
    min_distance = n_points // 20  # Minimum distance between corners
    
    for i in range(n_points):
        angle = angles[i]
        
        # Check if this is a local minimum and below threshold
        if angle < min_angle_threshold:
            # Check if it's a local minimum in a neighborhood
            start = max(0, i - min_distance // 2)
            end = min(n_points, i + min_distance // 2)
            
            # Handle wrap-around
            if start <= i <= end:
                neighborhood = angles[start:end]
            else:
                neighborhood = np.concatenate([angles[start:], angles[:end]])
            
            if angle <= np.min(neighborhood):
                # Check distance from existing corners
                is_far_enough = True
                for existing_idx in corner_indices:
                    dist = min(abs(i - existing_idx), n_points - abs(i - existing_idx))
                    if dist < min_distance:
                        is_far_enough = False
                        break
                
                if is_far_enough:
                    corner_indices.append(i)
    
    if len(corner_indices) == 0:
        print(f"  No corners found with threshold {min_angle_threshold}, using convex hull fallback")
        return find_mask_corners(mask, depth, max_depth)
    
    # Sort corners by their position on contour (already in order)
    corner_indices = sorted(corner_indices)
    
    corners = contour_points[corner_indices]
    
    # Convert to row, col format
    corners_rc = np.array([[c[1], c[0]] for c in corners])
    
    print(f"  Found {len(corners_rc)} adaptive corners")
    print(f"  Corner angles: {angles[corner_indices]}")
    
    return corners_rc


def find_corners_by_polygon_approximation(mask: np.ndarray, depth: np.ndarray, max_depth: float,
                                          epsilon_factor: float = 0.01) -> np.ndarray:
    """
    Find corners using polygon approximation (Douglas-Peucker algorithm).
    Automatically finds corners without assuming a specific shape.
    
    Args:
        mask: Binary mask
        depth: Depth image
        max_depth: Maximum depth threshold
        epsilon_factor: Factor of arc length for approximation (smaller = more corners)
    
    Returns:
        corners_rc: N x 2 array of corner positions (row, col)
    """
    valid_mask = mask & (depth > 0) & (depth < max_depth)
    valid_mask_uint8 = (valid_mask > 0).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(valid_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Use approxPolyDP with adaptive epsilon
    arc_length = cv2.arcLength(largest_contour, True)
    epsilon = epsilon_factor * arc_length
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    corners = approx.squeeze()
    
    if len(corners.shape) == 1:
        return None
    
    # Convert to row, col format
    corners_rc = np.array([[c[1], c[0]] for c in corners])
    
    print(f"  Found {len(corners_rc)} corners via polygon approximation (epsilon_factor={epsilon_factor})")
    
    return corners_rc


def farthest_point_sampling(points: np.ndarray, n_samples: int, 
                            seed_points: np.ndarray = None) -> np.ndarray:
    """
    Farthest Point Sampling.
    """
    N = len(points)
    if N == 0:
        return np.array([]).reshape(0, 3)
    
    if n_samples >= N:
        return points.copy()
    
    # Initialize distances
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
def initialize_fps_ordered(corners_3d: np.ndarray, contour_3d: np.ndarray,
                           point_cloud: np.ndarray) -> np.ndarray:
    """
    FPS-based initialization with proper ordering along contour edges.
    """
    keypoints = np.zeros((N_KEYPOINTS, 3), dtype=np.float64)
    
    # Step 1: Place corners
    keypoints[0] = corners_3d[0]   # top-left
    keypoints[4] = corners_3d[1]   # top-right
    keypoints[24] = corners_3d[2]  # bottom-right
    keypoints[20] = corners_3d[3]  # bottom-left
    
    if len(contour_3d) < 12:
        print(f"Warning: Not enough contour points ({len(contour_3d)}), falling back to bilinear")
        return initialize_bilinear(corners_3d, point_cloud)
    
    # Step 2: Find corner positions on contour
    nn_contour = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn_contour.fit(contour_3d)
    _, corner_contour_indices = nn_contour.kneighbors(corners_3d)
    corner_contour_indices = corner_contour_indices.flatten()
    
    print(f"  Corner indices on contour: {corner_contour_indices}")
    
    # Step 3: Segment contour into 4 edges
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
        
        if idx_start < idx_end:
            segment = contour_3d[idx_start:idx_end+1]
        else:
            segment = np.vstack([contour_3d[idx_start:], contour_3d[:idx_end+1]])
        
        print(f"  Edge {edge_id} ({c_start}->{c_end}): {len(segment)} contour points")
        
        if len(segment) >= 5:
            n_seg = len(segment)
            sample_indices = [
                int(n_seg * 0.25),
                int(n_seg * 0.50),
                int(n_seg * 0.75),
            ]
            sampled = segment[sample_indices]
            
            for i, idx in enumerate(grid_indices):
                keypoints[idx] = sampled[i]
        else:
            print(f"    Not enough points, using linear interpolation")
            corner_start = corners_3d[c_start]
            corner_end = corners_3d[c_end]
            for i, idx in enumerate(grid_indices):
                t = (i + 1) / 4.0
                keypoints[idx] = (1 - t) * corner_start + t * corner_end
    
    # Step 4: Initialize interior points using FPS
    if len(point_cloud) > 0:
        boundary_indices = CORNER_INDICES + BORDER_INDICES
        anchor_points = keypoints[boundary_indices]
        
        print(f"  Interior FPS: using {len(anchor_points)} boundary anchors, sampling {len(INTERIOR_INDICES)} from {len(point_cloud)} points")
        
        interior_fps, _ = farthest_point_sampling_with_indices(
            point_cloud,
            n_samples=len(INTERIOR_INDICES),
            seed_points=anchor_points
        )
        
        # Compute expected positions
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
        
        # Greedy assignment
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
                keypoints[idx] = expected
    
    return keypoints


# ================================================================
# APPROACH 3: Pure FPS - sample 25 points directly from point cloud
# ================================================================
def initialize_pure_fps(point_cloud: np.ndarray, corners_3d: np.ndarray) -> np.ndarray:
    """
    Pure FPS approach: Sample 25 points directly from point cloud using FPS.
    """
    if len(point_cloud) < N_KEYPOINTS:
        print(f"Warning: Not enough points ({len(point_cloud)}), returning zeros")
        return np.zeros((N_KEYPOINTS, 3), dtype=np.float64)
    
    fps_points = farthest_point_sampling(point_cloud, n_samples=N_KEYPOINTS)
    
    print(f"  FPS sampled {len(fps_points)} points from {len(point_cloud)} point cloud")
    
    # Compute local coordinate system from corners
    top_left, top_right, bottom_right, bottom_left = corners_3d
    
    x_axis = top_right - top_left
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    
    y_axis = bottom_left - top_left
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
    
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
    
    # Assign to grid positions
    keypoints = np.zeros((N_KEYPOINTS, 3), dtype=np.float64)
    assigned_fps = [False] * len(fps_points)
    
    for grid_idx in range(N_KEYPOINTS):
        row, col = idx_to_grid_pos(grid_idx)
        expected_x = col / (GRID_COLS - 1)
        expected_y = row / (GRID_ROWS - 1)
        
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
    """
    keypoints = np.zeros((N_KEYPOINTS, 3), dtype=np.float64)
    
    # Step 1: Place corners
    keypoints[0] = corners_3d[0]
    keypoints[4] = corners_3d[1]
    keypoints[24] = corners_3d[2]
    keypoints[20] = corners_3d[3]
    
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
        (0, 1, [1, 2, 3]),
        (1, 2, [9, 14, 19]),
        (2, 3, [23, 22, 21]),
        (3, 0, [15, 10, 5]),
    ]
    
    for edge_id, (c_start, c_end, grid_indices) in enumerate(edge_definitions):
        idx_start = corner_contour_indices[c_start]
        idx_end = corner_contour_indices[c_end]
        
        # Choose shorter path
        if idx_start <= idx_end:
            forward_len = idx_end - idx_start + 1
            backward_len = n_contour - idx_end + idx_start + 1
        else:
            forward_len = n_contour - idx_start + idx_end + 1
            backward_len = idx_start - idx_end + 1
        
        if forward_len <= backward_len:
            if idx_start <= idx_end:
                segment = contour_3d[idx_start:idx_end+1]
            else:
                segment = np.vstack([contour_3d[idx_start:], contour_3d[:idx_end+1]])
        else:
            if idx_start >= idx_end:
                segment = contour_3d[idx_end:idx_start+1][::-1]
            else:
                segment = np.vstack([contour_3d[idx_end:], contour_3d[:idx_start+1]])[::-1]
        
        print(f"  Edge {edge_id} ({c_start}->{c_end}): {len(segment)} pts (fwd={forward_len}, bwd={backward_len})")
        
        if len(segment) >= 5:
            anchor_start = corners_3d[c_start]
            anchor_end = corners_3d[c_end]
            
            fps_points = farthest_point_sampling(
                segment,
                n_samples=3,
                seed_points=np.array([anchor_start, anchor_end])
            )
            
            dists_from_start = np.linalg.norm(fps_points - anchor_start, axis=1)
            order = np.argsort(dists_from_start)
            fps_points = fps_points[order]
            
            for i, idx in enumerate(grid_indices):
                keypoints[idx] = fps_points[i]
        else:
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
            
            top = (1 - u) * keypoints[0] + u * keypoints[4]
            bottom = (1 - u) * keypoints[20] + u * keypoints[24]
            expected = (1 - v) * top + v * bottom
            
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
        ax.scatter(pc_subsample[:, 0], pc_subsample[:, 1], pc_subsample[:, 2],
                   c='lightgray', s=1, alpha=0.3, label='Point Cloud')
        
        ax.scatter(contour_3d[:, 0], contour_3d[:, 1], contour_3d[:, 2],
                   c='cyan', s=5, alpha=0.5, label='Contour')
        
        for i, j in edges:
            if not np.any(np.isnan(keypoints[i])) and not np.any(np.isnan(keypoints[j])):
                ax.plot([keypoints[i, 0], keypoints[j, 0]], 
                       [keypoints[i, 1], keypoints[j, 1]], 
                       [keypoints[i, 2], keypoints[j, 2]], 'b-', linewidth=1.5)
        
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
        ax.view_init(elev=-70, azim=-90)
    
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    draw_keypoints_3d(ax1, keypoints_bilinear, 'Approach 1: Bilinear (may be off contour)')
    
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    draw_keypoints_3d(ax2, keypoints_fps, 'Approach 2: FPS Contour + FPS Interior')
    
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    draw_keypoints_3d(ax3, keypoints_pure_fps, 'Approach 3: Pure FPS (25 pts)')
    
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    draw_keypoints_3d(ax4, keypoints_contour_bilinear, 'Approach 4: FPS Contour + Bilinear Interior')
    
    ax1.legend(['Point Cloud', 'Contour', 'Edges', 'Corners', 'Border', 'Interior'], 
               loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {save_path}")
    
    plt.show()


def visualize_adaptive_corners(mask: np.ndarray, color_bgr: np.ndarray, 
                                corners_rc: np.ndarray, save_path: str = None):
    """Visualize detected corners on the image."""
    viz = color_bgr.copy()
    
    # Draw contour
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(viz, contours, -1, (0, 255, 0), 2)
    
    # Draw corners
    for i, (row, col) in enumerate(corners_rc):
        col, row = int(col), int(row)
        cv2.circle(viz, (col, row), 8, (0, 0, 255), -1)
        cv2.putText(viz, str(i), (col + 10, row + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    if save_path:
        cv2.imwrite(save_path, viz)
        print(f"  Saved corner visualization to: {save_path}")
    
    return viz


# ================================================================
# APPROACH 5: Adaptive corner-based initialization for irregular shapes
# ================================================================
def initialize_adaptive_corners(corners_3d: np.ndarray, contour_3d: np.ndarray,
                                 point_cloud: np.ndarray, n_keypoints: int = 25) -> np.ndarray:
    """
    Adaptive initialization for irregular shapes like T-shirts.
    
    Strategy:
    1. Use detected corners as anchor points
    2. FPS on contour to fill remaining border points
    3. FPS on interior to fill interior points
    
    Args:
        corners_3d: N x 3 detected corners (variable number)
        contour_3d: M x 3 contour points
        point_cloud: P x 3 point cloud
        n_keypoints: Total number of keypoints to place
    
    Returns:
        keypoints: n_keypoints x 3 array
    """
    n_corners = len(corners_3d)
    
    # Determine how many border vs interior points we need
    # Heuristic: ~40% on border, ~60% interior (for irregular shapes)
    n_border_total = max(n_corners, int(n_keypoints * 0.4))
    n_interior = n_keypoints - n_border_total
    
    # Additional border points needed (excluding corners)
    n_border_extra = n_border_total - n_corners
    
    print(f"  Adaptive init: {n_corners} corners, {n_border_extra} extra border, {n_interior} interior")
    
    keypoints = np.zeros((n_keypoints, 3), dtype=np.float64)
    
    # Step 1: Place detected corners
    for i in range(min(n_corners, n_keypoints)):
        keypoints[i] = corners_3d[i]
    
    # Step 2: FPS on contour to get additional border points (with corners as seeds)
    if n_border_extra > 0 and len(contour_3d) > n_border_extra:
        border_fps = farthest_point_sampling(
            contour_3d,
            n_samples=n_border_extra,
            seed_points=corners_3d
        )
        
        for i, pt in enumerate(border_fps):
            idx = n_corners + i
            if idx < n_keypoints:
                keypoints[idx] = pt
    
    # Step 3: FPS on point cloud for interior (with all border points as seeds)
    border_points = keypoints[:n_border_total]
    valid_border = border_points[~np.any(border_points == 0, axis=1)]
    
    if n_interior > 0 and len(point_cloud) > n_interior:
        interior_fps = farthest_point_sampling(
            point_cloud,
            n_samples=n_interior,
            seed_points=valid_border
        )
        
        for i, pt in enumerate(interior_fps):
            idx = n_border_total + i
            if idx < n_keypoints:
                keypoints[idx] = pt
    
    return keypoints


# ================================================================
# APPROACH 6: T-shirt specific - N corners + FPS on contour segments
# ================================================================
def find_n_corners_on_contour(mask: np.ndarray, depth: np.ndarray, max_depth: float,
                               n_corners: int = 10, epsilon_start: float = 0.02) -> tuple:
    """
    Find exactly N corners on the contour using polygon approximation.
    
    Args:
        mask: Binary mask
        depth: Depth image  
        max_depth: Maximum depth threshold
        n_corners: Target number of corners
        epsilon_start: Starting epsilon factor for approxPolyDP
    
    Returns:
        corners_rc: N x 2 corners in (row, col) format
        corner_contour_indices: indices of corners on the contour
    """
    valid_mask = mask & (depth > 0) & (depth < max_depth)
    valid_mask_uint8 = (valid_mask > 0).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(valid_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None
    
    largest_contour = max(contours, key=cv2.contourArea)
    arc_length = cv2.arcLength(largest_contour, True)
    
    # Binary search for epsilon that gives us n_corners
    epsilon_low = 0.001
    epsilon_high = 0.1
    epsilon = epsilon_start
    best_approx = None
    best_diff = float('inf')
    
    for _ in range(30):  # Max iterations
        epsilon = (epsilon_low + epsilon_high) / 2
        approx = cv2.approxPolyDP(largest_contour, epsilon * arc_length, True)
        n_found = len(approx)
        
        diff = abs(n_found - n_corners)
        if diff < best_diff:
            best_diff = diff
            best_approx = approx
        
        if n_found == n_corners:
            break
        elif n_found > n_corners:
            epsilon_low = epsilon  # Need larger epsilon to reduce corners
        else:
            epsilon_high = epsilon  # Need smaller epsilon to get more corners
    
    if best_approx is None:
        return None, None
    
    corners = best_approx.squeeze()
    if len(corners.shape) == 1:
        return None, None
    
    print(f"  Target corners: {n_corners}, Found: {len(corners)} (epsilon={epsilon:.4f})")
    
    # Find corner indices on the original contour
    contour_points = largest_contour.squeeze()
    corner_indices = []
    for corner in corners:
        # Find nearest point on contour
        dists = np.linalg.norm(contour_points - corner, axis=1)
        idx = np.argmin(dists)
        corner_indices.append(idx)
    
    # Convert to row, col format
    corners_rc = np.array([[c[1], c[0]] for c in corners])
    
    return corners_rc, np.array(corner_indices)


from scipy.spatial import Delaunay


def initialize_cloth_with_interior(mask: np.ndarray, depth: np.ndarray, max_depth: float,
                                    fx, fy, cx, cy,
                                    n_corners: int = 10,
                                    n_keypoints_per_segment: int = 2,
                                    n_interior: int = 15) -> tuple:
    """
    Clean cloth initialization with contour + interior points:
    1. Find contour (ordered pixel sequence)
    2. Find N corners on contour
    3. Use FPS on each segment between corners to get contour keypoints
    4. Use FPS on interior to get interior keypoints
    5. Build edges using Delaunay triangulation for clean mesh
    
    Args:
        mask: Binary mask
        depth: Depth image
        max_depth: Max depth threshold
        fx, fy, cx, cy: Camera intrinsics
        n_corners: Number of corners to detect
        n_keypoints_per_segment: FPS keypoints per segment (between corners)
        n_interior: Number of interior keypoints
    
    Returns:
        keypoints_3d: All keypoints (contour + interior)
        edges: List of (i, j) edges from Delaunay triangulation
        keypoint_types: List of 'corner', 'contour', or 'interior'
        corners_3d: Just the corner points
        contour_keypoints_3d: Just the contour keypoints (corners + segment FPS)
        segment_info: Dict with segment info
    """
    valid_mask = mask & (depth > 0) & (depth < max_depth)
    valid_mask_uint8 = (valid_mask > 0).astype(np.uint8) * 255
    
    # Step 1: Get ordered contour
    contours, _ = cv2.findContours(valid_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None, None, None, None, None
    
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points_2d = largest_contour.squeeze()  # (M, 2) in (col, row) format
    n_contour = len(contour_points_2d)
    
    print(f"  Contour has {n_contour} pixels")
    
    # Step 2: Find N corners on contour
    corners_rc, corner_contour_indices = find_n_corners_on_contour(
        mask, depth, max_depth, n_corners=n_corners
    )
    
    if corners_rc is None:
        print("Failed to find corners!")
        return None, None, None, None, None, None
    
    # Sort corner indices to be in contour order
    sorted_order = np.argsort(corner_contour_indices)
    corner_contour_indices = corner_contour_indices[sorted_order]
    corners_rc = corners_rc[sorted_order]
    
    # Convert corners to 3D
    H, W = mask.shape
    corners_3d = pixel_to_3d(corners_rc, depth, fx, fy, cx, cy)
    
    # Remove invalid corners (NaN depth)
    valid_mask_corners = ~np.any(np.isnan(corners_3d), axis=1)
    corners_3d = corners_3d[valid_mask_corners]
    corner_contour_indices = corner_contour_indices[valid_mask_corners]
    corners_rc = corners_rc[valid_mask_corners]
    
    n_corners_valid = len(corners_3d)
    print(f"  Valid corners: {n_corners_valid}")
    
    # Convert full contour to 3D (for FPS sampling)
    contour_3d_full = []
    contour_3d_indices = []
    for i, (col, row) in enumerate(contour_points_2d):
        if 0 <= row < H and 0 <= col < W:
            z = depth[row, col]
            if 0 < z < max_depth:
                x = (col - cx) * z / fx
                y = (row - cy) * z / fy
                contour_3d_full.append([x, y, z])
                contour_3d_indices.append(i)
    
    contour_3d_full = np.array(contour_3d_full)
    contour_3d_indices = np.array(contour_3d_indices)
    
    # Step 3: Build contour keypoints (corners + FPS on each segment)
    contour_keypoints = []
    keypoint_types = []
    segment_info = {'segments': []}
    
    for seg_idx in range(n_corners_valid):
        idx_start = corner_contour_indices[seg_idx]
        idx_end = corner_contour_indices[(seg_idx + 1) % n_corners_valid]
        
        # Get segment 2D indices (handle wrap-around)
        if idx_start <= idx_end:
            segment_2d_indices = np.arange(idx_start, idx_end + 1)
        else:
            segment_2d_indices = np.concatenate([
                np.arange(idx_start, n_contour),
                np.arange(0, idx_end + 1)
            ])
        
        # Get segment 3D points
        segment_3d = []
        segment_3d_to_contour = []
        for idx_2d in segment_2d_indices:
            match = np.where(contour_3d_indices == idx_2d)[0]
            if len(match) > 0:
                segment_3d.append(contour_3d_full[match[0]])
                segment_3d_to_contour.append(idx_2d)
        
        segment_3d = np.array(segment_3d) if segment_3d else np.array([]).reshape(0, 3)
        segment_3d_to_contour = np.array(segment_3d_to_contour)
        
        corner_start = corners_3d[seg_idx]
        corner_end = corners_3d[(seg_idx + 1) % n_corners_valid]
        
        # Segment start
        seg_start_idx = len(contour_keypoints)
        
        # Add corner
        contour_keypoints.append(corner_start)
        keypoint_types.append('corner')
        
        # FPS on segment
        if len(segment_3d) > n_keypoints_per_segment + 2 and n_keypoints_per_segment > 0:
            fps_points = farthest_point_sampling(
                segment_3d,
                n_samples=n_keypoints_per_segment,
                seed_points=np.array([corner_start, corner_end])
            )
            
            # Find contour index for each FPS point for ordering
            fps_contour_indices = []
            for fps_pt in fps_points:
                dists = np.linalg.norm(segment_3d - fps_pt, axis=1)
                closest_idx = np.argmin(dists)
                fps_contour_indices.append(segment_3d_to_contour[closest_idx])
            
            fps_contour_indices = np.array(fps_contour_indices)
            
            # Order by position along contour
            if idx_start <= idx_end:
                order = np.argsort(fps_contour_indices)
            else:
                adjusted = np.where(
                    fps_contour_indices >= idx_start,
                    fps_contour_indices - idx_start,
                    fps_contour_indices + (n_contour - idx_start)
                )
                order = np.argsort(adjusted)
            
            fps_points_ordered = fps_points[order]
            
            for pt in fps_points_ordered:
                contour_keypoints.append(pt)
                keypoint_types.append('contour')
        
        segment_info['segments'].append({
            'segment_idx': seg_idx,
            'keypoint_start': seg_start_idx,
            'keypoint_end': len(contour_keypoints) - 1,
            'n_keypoints': len(contour_keypoints) - seg_start_idx,
        })
    
    contour_keypoints = np.array(contour_keypoints)
    n_contour_kps = len(contour_keypoints)
    print(f"  Contour keypoints: {n_contour_kps} ({n_corners_valid} corners + {n_contour_kps - n_corners_valid} FPS)")
    
    # Step 4: FPS on interior
    point_cloud = extract_point_cloud(mask, depth, max_depth, fx, fy, cx, cy)
    
    interior_fps = farthest_point_sampling(
        point_cloud,
        n_samples=n_interior,
        seed_points=contour_keypoints
    )
    
    print(f"  Interior keypoints: {len(interior_fps)}")
    
    # Add interior keypoints and types
    for pt in interior_fps:
        keypoint_types.append('interior')
    
    # Combine all keypoints
    all_keypoints = np.vstack([contour_keypoints, interior_fps])
    n_total = len(all_keypoints)
    print(f"  Total keypoints: {n_total}")
    
    # Step 5: Build edges using Delaunay triangulation (on XY projection)
    # This creates a clean triangular mesh
    xy_points = all_keypoints[:, :2]  # Project to XY plane
    
    try:
        tri = Delaunay(xy_points)
        
        # Extract unique edges from triangulation
        edges_set = set()
        for simplex in tri.simplices:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
                edges_set.add(edge)
        
        edges = list(edges_set)
        print(f"  Edges from Delaunay: {len(edges)}")
        
    except Exception as e:
        print(f"  Delaunay failed: {e}, falling back to contour edges only")
        # Fallback: just connect contour sequentially
        edges = []
        for i in range(n_contour_kps):
            edges.append((i, (i + 1) % n_contour_kps))
    
    # Compute edge length stats
    edge_lengths = [np.linalg.norm(all_keypoints[i] - all_keypoints[j]) for i, j in edges]
    edge_lengths = np.array(edge_lengths)
    print(f"  Edge lengths: mean={np.mean(edge_lengths):.1f}mm, std={np.std(edge_lengths):.1f}mm")
    
    return all_keypoints, edges, keypoint_types, corners_3d, contour_keypoints, segment_info


def initialize_cloth_contour_clean(mask: np.ndarray, depth: np.ndarray, max_depth: float,
                                    fx, fy, cx, cy,
                                    n_corners: int = 10,
                                    n_keypoints_per_segment: int = 2) -> tuple:
    """
    Clean cloth initialization following the wire tracking flow:
    1. Find contour (ordered pixel sequence)
    2. Find N corners on contour
    3. Use FPS on each segment between corners to get keypoints
    4. All keypoints are on the contour (no interior)
    5. Connect sequentially along contour
    
    This is the CONTOUR-ONLY approach - clean and simple.
    
    Args:
        mask: Binary mask
        depth: Depth image
        max_depth: Max depth threshold
        fx, fy, cx, cy: Camera intrinsics
        n_corners: Number of corners to detect
        n_keypoints_per_segment: FPS keypoints per segment (between corners)
    
    Returns:
        keypoints_3d: All keypoints in contour order (N, 3)
        edges: List of (i, j) edges connecting sequential keypoints
        keypoint_types: List of 'corner' or 'contour' for each keypoint
        corners_3d: Just the corner points
        segment_info: Dict with segment lengths and keypoint ranges
    """
    valid_mask = mask & (depth > 0) & (depth < max_depth)
    valid_mask_uint8 = (valid_mask > 0).astype(np.uint8) * 255
    
    # Step 1: Get ordered contour
    contours, _ = cv2.findContours(valid_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None, None, None, None
    
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points_2d = largest_contour.squeeze()  # (M, 2) in (col, row) format
    n_contour = len(contour_points_2d)
    
    print(f"  Contour has {n_contour} pixels")
    
    # Step 2: Find N corners on contour
    corners_rc, corner_contour_indices = find_n_corners_on_contour(
        mask, depth, max_depth, n_corners=n_corners
    )
    
    if corners_rc is None:
        print("Failed to find corners!")
        return None, None, None, None, None
    
    # Sort corner indices to be in contour order
    sorted_order = np.argsort(corner_contour_indices)
    corner_contour_indices = corner_contour_indices[sorted_order]
    corners_rc = corners_rc[sorted_order]
    
    # Convert corners to 3D
    H, W = mask.shape
    corners_3d = pixel_to_3d(corners_rc, depth, fx, fy, cx, cy)
    
    # Remove invalid corners (NaN depth)
    valid_mask_corners = ~np.any(np.isnan(corners_3d), axis=1)
    corners_3d = corners_3d[valid_mask_corners]
    corner_contour_indices = corner_contour_indices[valid_mask_corners]
    corners_rc = corners_rc[valid_mask_corners]
    
    n_corners_valid = len(corners_3d)
    print(f"  Valid corners: {n_corners_valid}")
    
    # Convert full contour to 3D (for FPS sampling)
    contour_3d_full = []
    contour_3d_indices = []  # maps contour_3d index to contour_2d index
    for i, (col, row) in enumerate(contour_points_2d):
        if 0 <= row < H and 0 <= col < W:
            z = depth[row, col]
            if 0 < z < max_depth:
                x = (col - cx) * z / fx
                y = (row - cy) * z / fy
                contour_3d_full.append([x, y, z])
                contour_3d_indices.append(i)
    
    contour_3d_full = np.array(contour_3d_full)
    contour_3d_indices = np.array(contour_3d_indices)
    
    print(f"  Valid 3D contour points: {len(contour_3d_full)}")
    
    # Step 3: For each segment between corners, use FPS to get keypoints
    # Then ORDER them along the contour
    all_keypoints = []
    keypoint_types = []
    segment_info = {'segments': []}
    
    for seg_idx in range(n_corners_valid):
        idx_start = corner_contour_indices[seg_idx]
        idx_end = corner_contour_indices[(seg_idx + 1) % n_corners_valid]
        
        # Get segment 2D indices (handle wrap-around)
        if idx_start <= idx_end:
            segment_2d_indices = np.arange(idx_start, idx_end + 1)
        else:
            segment_2d_indices = np.concatenate([
                np.arange(idx_start, n_contour),
                np.arange(0, idx_end + 1)
            ])
        
        # Get segment 3D points (in order along contour)
        segment_3d = []
        segment_3d_to_contour = []  # Track which contour index each 3D point maps to
        for idx_2d in segment_2d_indices:
            match = np.where(contour_3d_indices == idx_2d)[0]
            if len(match) > 0:
                segment_3d.append(contour_3d_full[match[0]])
                segment_3d_to_contour.append(idx_2d)
        
        segment_3d = np.array(segment_3d) if segment_3d else np.array([]).reshape(0, 3)
        segment_3d_to_contour = np.array(segment_3d_to_contour)
        
        # Segment start and end corners
        corner_start = corners_3d[seg_idx]
        corner_end = corners_3d[(seg_idx + 1) % n_corners_valid]
        
        # Keypoints for this segment (in order)
        segment_keypoints = [corner_start]
        segment_types = ['corner']
        
        # FPS on this segment (excluding corners as seeds)
        if len(segment_3d) > n_keypoints_per_segment + 2 and n_keypoints_per_segment > 0:
            fps_points = farthest_point_sampling(
                segment_3d,
                n_samples=n_keypoints_per_segment,
                seed_points=np.array([corner_start, corner_end])
            )
            
            # Find the contour index for each FPS point (for ordering)
            fps_contour_indices = []
            for fps_pt in fps_points:
                dists = np.linalg.norm(segment_3d - fps_pt, axis=1)
                closest_idx = np.argmin(dists)
                fps_contour_indices.append(segment_3d_to_contour[closest_idx])
            
            fps_contour_indices = np.array(fps_contour_indices)
            
            # Order FPS points by their position along the contour
            # Handle wrap-around: normalize to distance from segment start
            if idx_start <= idx_end:
                # Simple case: just use index order
                order = np.argsort(fps_contour_indices)
            else:
                # Wrap-around: compute distance from start
                adjusted_indices = np.where(
                    fps_contour_indices >= idx_start,
                    fps_contour_indices - idx_start,
                    fps_contour_indices + (n_contour - idx_start)
                )
                order = np.argsort(adjusted_indices)
            
            fps_points_ordered = fps_points[order]
            
            for pt in fps_points_ordered:
                segment_keypoints.append(pt)
                segment_types.append('contour')
        
        # Store segment info
        seg_start_idx = len(all_keypoints)
        segment_info['segments'].append({
            'segment_idx': seg_idx,
            'keypoint_start': seg_start_idx,
            'keypoint_end': seg_start_idx + len(segment_keypoints) - 1,
            'n_keypoints': len(segment_keypoints),
        })
        
        # Add to global list (but NOT the end corner - it's the start of next segment)
        all_keypoints.extend(segment_keypoints)
        keypoint_types.extend(segment_types)
    
    all_keypoints = np.array(all_keypoints)
    n_total = len(all_keypoints)
    
    print(f"  Total keypoints (contour only): {n_total}")
    
    # Step 4: Build edges - simple sequential connectivity along contour
    edges = []
    for i in range(n_total):
        next_i = (i + 1) % n_total
        edges.append((i, next_i))
    
    print(f"  Total edges: {len(edges)}")
    
    # Summary
    n_corner_kps = len([t for t in keypoint_types if t == 'corner'])
    n_contour_kps = len([t for t in keypoint_types if t == 'contour'])
    print(f"  Keypoint types: {n_corner_kps} corners, {n_contour_kps} contour FPS")
    
    return all_keypoints, edges, keypoint_types, corners_3d, segment_info


def initialize_tshirt_contour_fps(mask: np.ndarray, depth: np.ndarray, max_depth: float,
                                   fx, fy, cx, cy,
                                   n_corners: int = 10,
                                   n_keypoints_per_segment: int = 2,
                                   n_interior: int = 15) -> tuple:
    """
    Initialize keypoints for T-shirt shape:
    1. Find N corners on contour
    2. FPS on each contour segment between corners
    3. FPS on interior
    
    Args:
        mask: Binary mask
        depth: Depth image
        max_depth: Max depth threshold
        fx, fy, cx, cy: Camera intrinsics
        n_corners: Number of corners to detect (default 10 for T-shirt)
        n_keypoints_per_segment: Additional keypoints per contour segment via FPS
        n_interior: Number of interior keypoints
    
    Returns:
        keypoints: All keypoints (corners + border + interior)
        corners_3d: Just the corner points
        contour_keypoints_3d: All contour keypoints (corners + segment points)
        edges: List of (i, j) tuples indicating connected keypoints
        keypoint_types: List of strings ('corner', 'contour', 'interior') for each keypoint
    """
    valid_mask = mask & (depth > 0) & (depth < max_depth)
    valid_mask_uint8 = (valid_mask > 0).astype(np.uint8) * 255
    
    # Get contour
    contours, _ = cv2.findContours(valid_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None, None, None, None
    
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points_2d = largest_contour.squeeze()  # (M, 2) in (col, row) format
    
    # Find N corners
    corners_rc, corner_contour_indices = find_n_corners_on_contour(
        mask, depth, max_depth, n_corners=n_corners
    )
    
    if corners_rc is None:
        print("Failed to find corners!")
        return None, None, None, None, None
    
    # Sort corner indices to be in contour order
    sorted_order = np.argsort(corner_contour_indices)
    corner_contour_indices = corner_contour_indices[sorted_order]
    corners_rc = corners_rc[sorted_order]
    
    # Convert corners to 3D
    corners_3d = pixel_to_3d(corners_rc, depth, fx, fy, cx, cy)
    
    # Remove invalid corners (NaN depth)
    valid_mask_corners = ~np.any(np.isnan(corners_3d), axis=1)
    corners_3d = corners_3d[valid_mask_corners]
    corner_contour_indices = corner_contour_indices[valid_mask_corners]
    corners_rc = corners_rc[valid_mask_corners]
    
    print(f"  Valid corners after depth check: {len(corners_3d)}")
    
    n_corners_valid = len(corners_3d)
    n_contour = len(contour_points_2d)
    
    # Convert full contour to 3D
    H, W = mask.shape
    contour_3d_full = []
    contour_3d_indices = []
    for i, (col, row) in enumerate(contour_points_2d):
        if 0 <= row < H and 0 <= col < W:
            z = depth[row, col]
            if 0 < z < max_depth:
                x = (col - cx) * z / fx
                y = (row - cy) * z / fy
                contour_3d_full.append([x, y, z])
                contour_3d_indices.append(i)
    
    contour_3d_full = np.array(contour_3d_full)
    contour_3d_indices = np.array(contour_3d_indices)
    
    print(f"  Valid contour points: {len(contour_3d_full)} / {n_contour}")
    
    # Build keypoints and edges
    all_contour_keypoints = []
    keypoint_types = []
    edges = []
    
    # Track keypoint indices for edge building
    # Structure: for each segment, we have [corner_start, fps1, fps2, ..., corner_end]
    # Edges connect: corner_start -> fps1 -> fps2 -> ... -> corner_end
    
    current_idx = 0
    segment_keypoint_indices = []  # List of lists, one per segment
    
    for i in range(n_corners_valid):
        idx_start = corner_contour_indices[i]
        idx_end = corner_contour_indices[(i + 1) % n_corners_valid]
        
        # Get segment indices (handle wrap-around)
        if idx_start < idx_end:
            segment_indices = np.arange(idx_start, idx_end + 1)
        else:
            segment_indices = np.concatenate([
                np.arange(idx_start, n_contour),
                np.arange(0, idx_end + 1)
            ])
        
        # Find valid 3D points in this segment
        segment_3d = []
        for idx in segment_indices:
            match = np.where(contour_3d_indices == idx)[0]
            if len(match) > 0:
                segment_3d.append(contour_3d_full[match[0]])
        
        segment_3d = np.array(segment_3d) if segment_3d else np.array([]).reshape(0, 3)
        
        # This segment's keypoint indices
        segment_kp_indices = []
        
        # Add corner (start of segment)
        all_contour_keypoints.append(corners_3d[i])
        keypoint_types.append('corner')
        segment_kp_indices.append(current_idx)
        current_idx += 1
        
        # Add FPS points for this segment
        if len(segment_3d) > n_keypoints_per_segment + 2:
            corner_start = corners_3d[i]
            corner_end = corners_3d[(i + 1) % n_corners_valid]
            
            segment_fps = farthest_point_sampling(
                segment_3d,
                n_samples=n_keypoints_per_segment,
                seed_points=np.array([corner_start, corner_end])
            )
            
            # Sort FPS points by distance from start corner (to maintain order along contour)
            dists_from_start = np.linalg.norm(segment_fps - corner_start, axis=1)
            order = np.argsort(dists_from_start)
            segment_fps = segment_fps[order]
            
            for pt in segment_fps:
                all_contour_keypoints.append(pt)
                keypoint_types.append('contour')
                segment_kp_indices.append(current_idx)
                current_idx += 1
        
        segment_keypoint_indices.append(segment_kp_indices)
    
    # Build contour edges (connect keypoints along each segment)
    for seg_idx, seg_kp_indices in enumerate(segment_keypoint_indices):
        # Connect within segment
        for j in range(len(seg_kp_indices) - 1):
            edges.append((seg_kp_indices[j], seg_kp_indices[j + 1]))
        
        # Connect last point of this segment to first point of next segment (the next corner)
        next_seg_idx = (seg_idx + 1) % len(segment_keypoint_indices)
        edges.append((seg_kp_indices[-1], segment_keypoint_indices[next_seg_idx][0]))
    
    all_contour_keypoints = np.array(all_contour_keypoints)
    n_contour_kps = len(all_contour_keypoints)
    print(f"  Total contour keypoints: {n_contour_kps} (corners + segment FPS)")
    print(f"  Contour edges: {len(edges)}")
    
    # FPS on interior
    point_cloud = extract_point_cloud(mask, depth, max_depth, fx, fy, cx, cy)
    
    interior_fps = farthest_point_sampling(
        point_cloud,
        n_samples=n_interior,
        seed_points=all_contour_keypoints
    )
    
    print(f"  Interior keypoints: {len(interior_fps)}")
    
    # Add interior keypoints
    interior_indices = []
    for pt in interior_fps:
        keypoint_types.append('interior')
        interior_indices.append(current_idx)
        current_idx += 1
    
    # Build interior edges using Delaunay triangulation or k-NN
    # Option 1: Connect each interior point to its k nearest neighbors
    all_keypoints = np.vstack([all_contour_keypoints, interior_fps])
    
    # Build edges for interior points using k-NN
    k_neighbors = 4  # Each interior point connects to 4 nearest points
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto')
    nn.fit(all_keypoints)
    
    for i, int_idx in enumerate(interior_indices):
        # Find k nearest neighbors for this interior point
        distances, indices = nn.kneighbors(all_keypoints[int_idx].reshape(1, -1))
        
        for neighbor_idx in indices[0, 1:]:  # Skip self (index 0)
            # Add edge if not already exists
            edge = tuple(sorted([int_idx, neighbor_idx]))
            if edge not in edges:
                edges.append(edge)
    
    # Also connect contour points to nearby interior points
    for contour_idx in range(n_contour_kps):
        distances, indices = nn.kneighbors(all_keypoints[contour_idx].reshape(1, -1))
        for neighbor_idx in indices[0, 1:3]:  # Connect to 2 nearest
            if neighbor_idx >= n_contour_kps:  # Only connect to interior points
                edge = tuple(sorted([contour_idx, neighbor_idx]))
                if edge not in edges:
                    edges.append(edge)
    
    print(f"  Total keypoints: {len(all_keypoints)}")
    print(f"  Total edges: {len(edges)}")
    
    return all_keypoints, corners_3d, all_contour_keypoints, edges, keypoint_types


def visualize_tshirt_keypoints(point_cloud: np.ndarray, contour_3d: np.ndarray,
                                all_keypoints: np.ndarray, corners_3d: np.ndarray,
                                contour_keypoints_3d: np.ndarray,
                                edges: list = None,
                                keypoint_types: list = None,
                                title: str = "T-shirt Keypoints",
                                save_path: str = None):
    """Visualize T-shirt keypoints in 3D using Plotly."""
    traces = []
    
    # Point cloud (gray)
    pc_down = point_cloud[::20]
    traces.append(go.Scatter3d(
        x=pc_down[:, 0], y=pc_down[:, 1], z=pc_down[:, 2],
        mode='markers',
        marker=dict(size=2, color='lightgray', opacity=0.3),
        name='Point Cloud'
    ))
    
    # Draw edges first (so they appear behind keypoints)
    if edges is not None:
        for i, j in edges:
            traces.append(go.Scatter3d(
                x=[all_keypoints[i, 0], all_keypoints[j, 0]],
                y=[all_keypoints[i, 1], all_keypoints[j, 1]],
                z=[all_keypoints[i, 2], all_keypoints[j, 2]],
                mode='lines',
                line=dict(color='blue', width=3),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Separate keypoints by type
    if keypoint_types is not None:
        corner_pts = [all_keypoints[i] for i, t in enumerate(keypoint_types) if t == 'corner']
        contour_pts = [all_keypoints[i] for i, t in enumerate(keypoint_types) if t == 'contour']
        interior_pts = [all_keypoints[i] for i, t in enumerate(keypoint_types) if t == 'interior']
        
        if corner_pts:
            corner_pts = np.array(corner_pts)
            traces.append(go.Scatter3d(
                x=corner_pts[:, 0], y=corner_pts[:, 1], z=corner_pts[:, 2],
                mode='markers+text',
                marker=dict(size=12, color='red', symbol='square'),
                text=[f'C{i}' for i in range(len(corner_pts))],
                textposition='top center',
                name=f'Corners ({len(corner_pts)})'
            ))
        
        if contour_pts:
            contour_pts = np.array(contour_pts)
            traces.append(go.Scatter3d(
                x=contour_pts[:, 0], y=contour_pts[:, 1], z=contour_pts[:, 2],
                mode='markers',
                marker=dict(size=8, color='green', opacity=0.8),
                name=f'Contour FPS ({len(contour_pts)})'
            ))
        
        if interior_pts:
            interior_pts = np.array(interior_pts)
            traces.append(go.Scatter3d(
                x=interior_pts[:, 0], y=interior_pts[:, 1], z=interior_pts[:, 2],
                mode='markers',
                marker=dict(size=8, color='orange', opacity=0.8),
                name=f'Interior FPS ({len(interior_pts)})'
            ))
    else:
        # Fallback: show all keypoints
        traces.append(go.Scatter3d(
            x=all_keypoints[:, 0], y=all_keypoints[:, 1], z=all_keypoints[:, 2],
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.8),
            name=f'Keypoints ({len(all_keypoints)})'
        ))
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
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


def visualize_cloth_contour_clean(point_cloud: np.ndarray, 
                                   keypoints_3d: np.ndarray,
                                   edges: list,
                                   keypoint_types: list,
                                   segment_info: dict = None,
                                   title: str = "Cloth Contour Keypoints",
                                   save_path: str = None):
    """
    Visualize clean contour keypoints with sequential edges.
    Color segments differently to show the flow.
    """
    traces = []
    
    # Point cloud (gray)
    pc_down = point_cloud[::20]
    traces.append(go.Scatter3d(
        x=pc_down[:, 0], y=pc_down[:, 1], z=pc_down[:, 2],
        mode='markers',
        marker=dict(size=2, color='lightgray', opacity=0.3),
        name='Point Cloud'
    ))
    
    # Color palette for segments
    segment_colors = [
        'red', 'blue', 'green', 'orange', 'purple', 
        'cyan', 'magenta', 'yellow', 'brown', 'pink',
        'olive', 'teal', 'navy', 'maroon', 'lime'
    ]
    
    # Draw edges with segment coloring
    if segment_info is not None and 'segments' in segment_info:
        # Color edges by segment
        for seg_idx, seg in enumerate(segment_info['segments']):
            color = segment_colors[seg_idx % len(segment_colors)]
            start_kp = seg['keypoint_start']
            n_kp = seg['n_keypoints']
            
            # Draw edges for this segment
            for i in range(n_kp - 1):
                kp_i = start_kp + i
                kp_j = start_kp + i + 1
                traces.append(go.Scatter3d(
                    x=[keypoints_3d[kp_i, 0], keypoints_3d[kp_j, 0]],
                    y=[keypoints_3d[kp_i, 1], keypoints_3d[kp_j, 1]],
                    z=[keypoints_3d[kp_i, 2], keypoints_3d[kp_j, 2]],
                    mode='lines',
                    line=dict(color=color, width=4),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Connect to next segment
            last_kp = start_kp + n_kp - 1
            next_seg_idx = (seg_idx + 1) % len(segment_info['segments'])
            next_start_kp = segment_info['segments'][next_seg_idx]['keypoint_start']
            traces.append(go.Scatter3d(
                x=[keypoints_3d[last_kp, 0], keypoints_3d[next_start_kp, 0]],
                y=[keypoints_3d[last_kp, 1], keypoints_3d[next_start_kp, 1]],
                z=[keypoints_3d[last_kp, 2], keypoints_3d[next_start_kp, 2]],
                mode='lines',
                line=dict(color=color, width=4),
                showlegend=False,
                hoverinfo='skip'
            ))
    else:
        # Fallback: draw all edges in blue
        for i, j in edges:
            traces.append(go.Scatter3d(
                x=[keypoints_3d[i, 0], keypoints_3d[j, 0]],
                y=[keypoints_3d[i, 1], keypoints_3d[j, 1]],
                z=[keypoints_3d[i, 2], keypoints_3d[j, 2]],
                mode='lines',
                line=dict(color='blue', width=3),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Draw keypoints by type
    corner_indices = [i for i, t in enumerate(keypoint_types) if t == 'corner']
    contour_indices = [i for i, t in enumerate(keypoint_types) if t == 'contour']
    
    if corner_indices:
        corner_pts = keypoints_3d[corner_indices]
        traces.append(go.Scatter3d(
            x=corner_pts[:, 0], y=corner_pts[:, 1], z=corner_pts[:, 2],
            mode='markers+text',
            marker=dict(size=14, color='red', symbol='square'),
            text=[f'C{i}' for i in range(len(corner_pts))],
            textposition='top center',
            name=f'Corners ({len(corner_pts)})'
        ))
    
    if contour_indices:
        contour_pts = keypoints_3d[contour_indices]
        traces.append(go.Scatter3d(
            x=contour_pts[:, 0], y=contour_pts[:, 1], z=contour_pts[:, 2],
            mode='markers',
            marker=dict(size=10, color='green', opacity=0.9),
            name=f'Contour FPS ({len(contour_pts)})'
        ))
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
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


def visualize_cloth_mesh(point_cloud: np.ndarray,
                          keypoints_3d: np.ndarray,
                          edges: list,
                          keypoint_types: list,
                          title: str = "Cloth Mesh",
                          save_path: str = None):
    """
    Visualize cloth keypoints with Delaunay mesh edges.
    Shows corners (red), contour (green), interior (orange).
    """
    traces = []
    
    # Point cloud (gray)
    pc_down = point_cloud[::20]
    traces.append(go.Scatter3d(
        x=pc_down[:, 0], y=pc_down[:, 1], z=pc_down[:, 2],
        mode='markers',
        marker=dict(size=2, color='lightgray', opacity=0.3),
        name='Point Cloud'
    ))
    
    # Draw all edges (blue mesh lines)
    for i, j in edges:
        traces.append(go.Scatter3d(
            x=[keypoints_3d[i, 0], keypoints_3d[j, 0]],
            y=[keypoints_3d[i, 1], keypoints_3d[j, 1]],
            z=[keypoints_3d[i, 2], keypoints_3d[j, 2]],
            mode='lines',
            line=dict(color='blue', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Draw keypoints by type
    corner_indices = [i for i, t in enumerate(keypoint_types) if t == 'corner']
    contour_indices = [i for i, t in enumerate(keypoint_types) if t == 'contour']
    interior_indices = [i for i, t in enumerate(keypoint_types) if t == 'interior']
    
    if corner_indices:
        corner_pts = keypoints_3d[corner_indices]
        traces.append(go.Scatter3d(
            x=corner_pts[:, 0], y=corner_pts[:, 1], z=corner_pts[:, 2],
            mode='markers+text',
            marker=dict(size=14, color='red', symbol='square'),
            text=[f'C{i}' for i in range(len(corner_pts))],
            textposition='top center',
            name=f'Corners ({len(corner_pts)})'
        ))
    
    if contour_indices:
        contour_pts = keypoints_3d[contour_indices]
        traces.append(go.Scatter3d(
            x=contour_pts[:, 0], y=contour_pts[:, 1], z=contour_pts[:, 2],
            mode='markers',
            marker=dict(size=10, color='green', opacity=0.9),
            name=f'Contour FPS ({len(contour_pts)})'
        ))
    
    if interior_indices:
        interior_pts = keypoints_3d[interior_indices]
        traces.append(go.Scatter3d(
            x=interior_pts[:, 0], y=interior_pts[:, 1], z=interior_pts[:, 2],
            mode='markers',
            marker=dict(size=10, color='orange', opacity=0.9),
            name=f'Interior FPS ({len(interior_pts)})'
        ))
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
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


def visualize_adaptive_keypoints(point_cloud: np.ndarray, contour_3d: np.ndarray,
                                  keypoints: np.ndarray, corners_3d: np.ndarray,
                                  title: str = "Adaptive Keypoints",
                                  save_path: str = None):
    """Visualize adaptive keypoints in 3D using Plotly."""
    traces = []
    
    # Point cloud (gray)
    pc_down = point_cloud[::20]
    traces.append(go.Scatter3d(
        x=pc_down[:, 0], y=pc_down[:, 1], z=pc_down[:, 2],
        mode='markers',
        marker=dict(size=2, color='lightgray', opacity=0.3),
        name='Point Cloud'
    ))
    
    # Contour (cyan)
    traces.append(go.Scatter3d(
        x=contour_3d[:, 0], y=contour_3d[:, 1], z=contour_3d[:, 2],
        mode='markers',
        marker=dict(size=3, color='cyan', opacity=0.5),
        name='Contour'
    ))
    
    # Detected corners (red squares)
    traces.append(go.Scatter3d(
        x=corners_3d[:, 0], y=corners_3d[:, 1], z=corners_3d[:, 2],
        mode='markers+text',
        marker=dict(size=10, color='red', symbol='square'),
        text=[str(i) for i in range(len(corners_3d))],
        textposition='top center',
        name=f'Detected Corners ({len(corners_3d)})'
    ))
    
    # All keypoints (blue)
    valid_kps = keypoints[~np.any(keypoints == 0, axis=1)]
    traces.append(go.Scatter3d(
        x=valid_kps[:, 0], y=valid_kps[:, 1], z=valid_kps[:, 2],
        mode='markers',
        marker=dict(size=8, color='blue', opacity=0.8),
        name=f'Keypoints ({len(valid_kps)})'
    ))
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
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
    # Paths - adapted for cloth data in NPZ format
    data_dir = Path("/home/yehengz/deformable_seg/data/arm_traj5_cloth")
    rgbd_path = data_dir / "rgbd.npz"
    masks_dir = data_dir / "masks"
    output_dir = data_dir / "init_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Camera intrinsics (same as fabric)
    INTRINSICS = (606.6875, 606.24609375, 641.7900390625, 366.8428955078125)
    fx, fy, cx, cy = INTRINSICS
    max_depth = 1200.0
    
    # Load RGBD data from NPZ format
    print("Loading RGBD data...")
    rgbd_data = np.load(str(rgbd_path))
    color_data_bgr = rgbd_data['color']  # (N, H, W, 3) BGR
    depth_data = rgbd_data['depth']      # (N, H, W)
    
    n_frames = color_data_bgr.shape[0]
    print(f"Found {n_frames} frames")
    print(f"Color shape: {color_data_bgr.shape}")
    print(f"Depth shape: {depth_data.shape}")
    
    # Load first frame's data
    depth = depth_data[0]
    color_bgr = color_data_bgr[0]
    
    # Load mask and apply depth thresholding
    mask_file = masks_dir / "mask_frame_0000.npy"
    if not mask_file.exists():
        print(f"Error: Mask file not found: {mask_file}")
        print("Please run naive_cloth_seg.py first to generate masks.")
        return
    
    mask_raw = np.load(str(mask_file))
    
    # IMPORTANT: Only keep pixels with valid depth (0 < depth < max_depth)
    valid_depth = (depth > 0) & (depth < max_depth)
    mask = mask_raw & valid_depth
    
    print(f"Raw mask pixels: {np.sum(mask_raw)}, Depth-thresholded: {np.sum(mask)}")
    
    # Find corners using different methods
    print("\n" + "="*60)
    print("Corner Detection Methods")
    print("="*60)
    
    # Method 1: Original 4-corner detection (for reference)
    print("\nMethod 1: 4-corner detection (rectangular assumption)")
    corners_2d_4 = find_mask_corners(mask, depth, max_depth)
    if corners_2d_4 is not None:
        print(f"  Found {len(corners_2d_4)} corners")
    
    # Method 2: Polygon approximation (adaptive)
    print("\nMethod 2: Polygon approximation (adaptive)")
    corners_2d_poly = find_corners_by_polygon_approximation(mask, depth, max_depth, epsilon_factor=0.015)
    
    # Method 3: Curvature-based (adaptive)
    print("\nMethod 3: Curvature-based detection")
    corners_2d_curv = find_adaptive_corners(mask, depth, max_depth, 
                                            min_angle_threshold=70.0, 
                                            smoothing_window=20)
    
    # Visualize all corner detection methods
    if corners_2d_poly is not None:
        visualize_adaptive_corners(mask, color_bgr, corners_2d_poly, 
                                   str(output_dir / "corners_polygon.png"))
    if corners_2d_curv is not None:
        visualize_adaptive_corners(mask, color_bgr, corners_2d_curv,
                                   str(output_dir / "corners_curvature.png"))
    
    # Use polygon approximation corners (usually works best for T-shirts)
    corners_2d = corners_2d_poly if corners_2d_poly is not None else corners_2d_4
    
    if corners_2d is None:
        print("Failed to find corners!")
        return
    
    print(f"\nUsing {len(corners_2d)} corners for initialization")
    
    corners_3d = pixel_to_3d(corners_2d, depth, fx, fy, cx, cy)
    # Remove NaN corners (invalid depth)
    valid_corners_mask = ~np.any(np.isnan(corners_3d), axis=1)
    corners_3d = corners_3d[valid_corners_mask]
    corners_2d = corners_2d[valid_corners_mask]
    print(f"Valid corners after depth check: {len(corners_3d)}")
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
    
    # Build edges (only used for 4-corner approaches)
    edges = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            idx = grid_pos_to_idx(row, col)
            if col < GRID_COLS - 1:
                edges.append((idx, grid_pos_to_idx(row, col + 1)))
            if row < GRID_ROWS - 1:
                edges.append((idx, grid_pos_to_idx(row + 1, col)))
    
    # Check if we have exactly 4 corners for grid-based approaches
    has_4_corners = len(corners_3d) == 4
    
    if has_4_corners:
        # Approach 1: Bilinear interpolation (requires 4 corners)
        print("\n" + "="*60)
        print("Approach 1: Bilinear Interpolation")
        print("="*60)
        keypoints_bilinear = initialize_bilinear(corners_3d, point_cloud)
        stats_bilinear = compute_edge_stats(keypoints_bilinear, edges)
        print(f"Edge lengths: mean={stats_bilinear['mean']:.2f}mm, std={stats_bilinear['std']:.2f}mm")
        print(f"             min={stats_bilinear['min']:.2f}mm, max={stats_bilinear['max']:.2f}mm")
        print(f"             CV={stats_bilinear['cv']:.3f}")
        
        # Approach 2: FPS-based (requires 4 corners)
        print("\n" + "="*60)
        print("Approach 2: FPS on Contour + Interior")
        print("="*60)
        keypoints_fps = initialize_fps_ordered(corners_3d, contour_3d, point_cloud)
        stats_fps = compute_edge_stats(keypoints_fps, edges)
        print(f"Edge lengths: mean={stats_fps['mean']:.2f}mm, std={stats_fps['std']:.2f}mm")
        print(f"             min={stats_fps['min']:.2f}mm, max={stats_fps['max']:.2f}mm")
        print(f"             CV={stats_fps['cv']:.3f}")
        
        # Approach 4: FPS on contour + bilinear interior (requires 4 corners)
        print("\n" + "="*60)
        print("Approach 4: FPS Contour + Bilinear Interior (snap to point cloud)")
        print("="*60)
        keypoints_contour_bilinear = initialize_contour_fps_bilinear_interior(corners_3d, contour_3d, point_cloud)
        stats_contour_bilinear = compute_edge_stats(keypoints_contour_bilinear, edges)
        print(f"Edge lengths: mean={stats_contour_bilinear['mean']:.2f}mm, std={stats_contour_bilinear['std']:.2f}mm")
        print(f"             min={stats_contour_bilinear['min']:.2f}mm, max={stats_contour_bilinear['max']:.2f}mm")
        print(f"             CV={stats_contour_bilinear['cv']:.3f}")
    else:
        print("\n" + "="*60)
        print(f"Skipping Approaches 1, 2, 4 (require exactly 4 corners, found {len(corners_3d)})")
        print("="*60)
        keypoints_bilinear = None
        keypoints_fps = None
        keypoints_contour_bilinear = None
    
    # Approach 3: Pure FPS (works with any number of corners)
    print("\n" + "="*60)
    print("Approach 3: Pure FPS (25 points directly)")
    print("="*60)
    # Use 4-corner fallback if we have more corners
    corners_for_fps = corners_3d[:4] if len(corners_3d) >= 4 else corners_3d
    keypoints_pure_fps = initialize_pure_fps(point_cloud, corners_for_fps)
    
    # Compute NN distances for pure FPS
    valid_kps_fps = keypoints_pure_fps[~np.any(keypoints_pure_fps == 0, axis=1)]
    if len(valid_kps_fps) > 1:
        nn = NearestNeighbors(n_neighbors=2, algorithm='auto')
        nn.fit(valid_kps_fps)
        distances, _ = nn.kneighbors(valid_kps_fps)
        nn_distances = distances[:, 1]
        print(f"  Total keypoints: {len(valid_kps_fps)}")
        print(f"  NN distances: mean={np.mean(nn_distances):.2f}mm, std={np.std(nn_distances):.2f}mm")
        print(f"               CV={np.std(nn_distances)/np.mean(nn_distances):.3f}")
    
    # ================================================================
    # APPROACH 5: CLEAN CONTOUR-ONLY (recommended for cloth tracking)
    # ================================================================
    print("\n" + "="*60)
    print("Approach 5: CLEAN CONTOUR-ONLY (corners + FPS on segments)")
    print("="*60)
    print("This follows the wire tracking flow:")
    print("  1. Find contour -> 2. Find corners -> 3. FPS on segments -> 4. Sequential edges")
    
    # Clean contour parameters
    N_CORNERS_CLEAN = 10  # Number of corners
    N_FPS_PER_SEGMENT = 3  # FPS points between each pair of corners
    
    keypoints_clean, edges_clean, types_clean, corners_clean_3d, segment_info = initialize_cloth_contour_clean(
        mask, depth, max_depth, fx, fy, cx, cy,
        n_corners=N_CORNERS_CLEAN,
        n_keypoints_per_segment=N_FPS_PER_SEGMENT
    )
    
    if keypoints_clean is not None:
        # Compute edge length statistics
        edge_lengths = []
        for i, j in edges_clean:
            length = np.linalg.norm(keypoints_clean[i] - keypoints_clean[j])
            edge_lengths.append(length)
        edge_lengths = np.array(edge_lengths)
        
        print(f"\n  Edge length stats:")
        print(f"    Mean: {np.mean(edge_lengths):.2f}mm")
        print(f"    Std:  {np.std(edge_lengths):.2f}mm")
        print(f"    Min:  {np.min(edge_lengths):.2f}mm")
        print(f"    Max:  {np.max(edge_lengths):.2f}mm")
        print(f"    CV:   {np.std(edge_lengths)/np.mean(edge_lengths):.3f}")
        
        # Visualize clean contour
        visualize_cloth_contour_clean(
            point_cloud, keypoints_clean, edges_clean, types_clean,
            segment_info=segment_info,
            title=f"Clean Contour ({N_CORNERS_CLEAN} corners, {N_FPS_PER_SEGMENT} FPS/segment)",
            save_path=str(output_dir / "keypoints_contour_clean.html")
        )
        
        # Save clean contour results
        np.save(str(output_dir / "keypoints_contour_clean.npy"), keypoints_clean)
        np.save(str(output_dir / "edges_contour_clean.npy"), np.array(edges_clean))
        np.save(str(output_dir / "keypoint_types_contour_clean.npy"), np.array(types_clean))
        np.save(str(output_dir / "corners_contour_clean.npy"), corners_clean_3d)
        
        print(f"\n  Summary:")
        print(f"    Total keypoints: {len(keypoints_clean)}")
        print(f"    Total edges: {len(edges_clean)}")
        print(f"    Corners: {len([t for t in types_clean if t == 'corner'])}")
        print(f"    Contour FPS: {len([t for t in types_clean if t == 'contour'])}")
        
        # Print segment info
        if segment_info is not None:
            print(f"\n  Segment breakdown:")
            for seg in segment_info['segments']:
                print(f"    Segment {seg['segment_idx']}: {seg['n_keypoints']} keypoints (indices {seg['keypoint_start']}-{seg['keypoint_end']})")
    
    # ================================================================
    # APPROACH 7: CLEAN MESH (contour + interior with Delaunay triangulation)
    # ================================================================
    print("\n" + "="*60)
    print("Approach 7: CLEAN MESH (contour + interior + Delaunay edges)")
    print("="*60)
    print("This is the RECOMMENDED approach for cloth tracking:")
    print("  1. Find contour -> 2. Find corners -> 3. FPS on segments")
    print("  4. FPS on interior -> 5. Delaunay triangulation for clean mesh")
    
    # Mesh parameters
    N_CORNERS_MESH = 10  # Number of corners
    N_FPS_PER_SEGMENT_MESH = 2  # FPS points between each pair of corners
    N_INTERIOR_MESH = 15  # Interior keypoints
    
    result = initialize_cloth_with_interior(
        mask, depth, max_depth, fx, fy, cx, cy,
        n_corners=N_CORNERS_MESH,
        n_keypoints_per_segment=N_FPS_PER_SEGMENT_MESH,
        n_interior=N_INTERIOR_MESH
    )
    
    if result[0] is not None:
        keypoints_mesh, edges_mesh, types_mesh, corners_mesh_3d, contour_mesh_3d, segment_info_mesh = result
        
        # Visualize clean mesh
        visualize_cloth_mesh(
            point_cloud, keypoints_mesh, edges_mesh, types_mesh,
            title=f"Clean Mesh ({N_CORNERS_MESH} corners, {N_FPS_PER_SEGMENT_MESH} FPS/segment, {N_INTERIOR_MESH} interior)",
            save_path=str(output_dir / "keypoints_mesh_clean.html")
        )
        
        # Save mesh results
        np.save(str(output_dir / "keypoints_mesh_clean.npy"), keypoints_mesh)
        np.save(str(output_dir / "edges_mesh_clean.npy"), np.array(edges_mesh))
        np.save(str(output_dir / "keypoint_types_mesh_clean.npy"), np.array(types_mesh))
        np.save(str(output_dir / "corners_mesh_clean.npy"), corners_mesh_3d)
        np.save(str(output_dir / "contour_keypoints_mesh_clean.npy"), contour_mesh_3d)
        
        print(f"\n  Summary:")
        print(f"    Total keypoints: {len(keypoints_mesh)}")
        print(f"    Total edges: {len(edges_mesh)}")
        print(f"    Corners: {len([t for t in types_mesh if t == 'corner'])}")
        print(f"    Contour FPS: {len([t for t in types_mesh if t == 'contour'])}")
        print(f"    Interior FPS: {len([t for t in types_mesh if t == 'interior'])}")
    
    # ================================================================
    # APPROACH 6: T-shirt with interior (messy k-NN, for comparison)
    # ================================================================
    print("\n" + "="*60)
    print("Approach 6: T-shirt with k-NN interior (messy, for comparison)")
    print("="*60)
    
    # T-shirt parameters - adjust these!
    N_TSHIRT_CORNERS = 10  # Number of corners for T-shirt
    N_KEYPOINTS_PER_SEGMENT = 2  # FPS points between each pair of corners
    N_INTERIOR_KEYPOINTS = 20  # Interior keypoints
    
    keypoints_tshirt, corners_tshirt_3d, contour_keypoints_3d, edges_tshirt, keypoint_types = initialize_tshirt_contour_fps(
        mask, depth, max_depth, fx, fy, cx, cy,
        n_corners=N_TSHIRT_CORNERS,
        n_keypoints_per_segment=N_KEYPOINTS_PER_SEGMENT,
        n_interior=N_INTERIOR_KEYPOINTS
    )
    
    if keypoints_tshirt is not None:
        # Visualize T-shirt keypoints with edges
        visualize_tshirt_keypoints(
            point_cloud, contour_3d, keypoints_tshirt, corners_tshirt_3d, contour_keypoints_3d,
            edges=edges_tshirt,
            keypoint_types=keypoint_types,
            title=f"T-shirt with Interior ({N_TSHIRT_CORNERS} corners, {N_KEYPOINTS_PER_SEGMENT}/segment, {N_INTERIOR_KEYPOINTS} interior)",
            save_path=str(output_dir / "keypoints_tshirt_interior.html")
        )
        
        # Also visualize corners on 2D image
        corners_tshirt_2d, _ = find_n_corners_on_contour(mask, depth, max_depth, n_corners=N_TSHIRT_CORNERS)
        if corners_tshirt_2d is not None:
            visualize_adaptive_corners(mask, color_bgr, corners_tshirt_2d,
                                       str(output_dir / "corners_tshirt_10.png"))
        
        # Compute NN distances
        if len(keypoints_tshirt) > 1:
            nn_tshirt = NearestNeighbors(n_neighbors=2, algorithm='auto')
            nn_tshirt.fit(keypoints_tshirt)
            distances, _ = nn_tshirt.kneighbors(keypoints_tshirt)
            nn_distances = distances[:, 1]
            print(f"  Total keypoints: {len(keypoints_tshirt)}")
            print(f"  NN distances: mean={np.mean(nn_distances):.2f}mm, std={np.std(nn_distances):.2f}mm")
            print(f"               min={np.min(nn_distances):.2f}mm, max={np.max(nn_distances):.2f}mm")
            print(f"               CV={np.std(nn_distances)/np.mean(nn_distances):.3f}")
        
        # Save T-shirt results
        np.save(str(output_dir / "keypoints_tshirt.npy"), keypoints_tshirt)
        np.save(str(output_dir / "corners_tshirt.npy"), corners_tshirt_3d)
        np.save(str(output_dir / "contour_keypoints_tshirt.npy"), contour_keypoints_3d)
        np.save(str(output_dir / "edges_tshirt.npy"), np.array(edges_tshirt))
        np.save(str(output_dir / "keypoint_types_tshirt.npy"), np.array(keypoint_types))
        
        print(f"\n  Topology summary:")
        print(f"    Keypoints: {len(keypoints_tshirt)}")
        print(f"    Edges: {len(edges_tshirt)}")
        print(f"    Types: {len([t for t in keypoint_types if t == 'corner'])} corners, "
              f"{len([t for t in keypoint_types if t == 'contour'])} contour, "
              f"{len([t for t in keypoint_types if t == 'interior'])} interior")
    
    # Visualize comparison (only if we have 4-corner results)
    if has_4_corners:
        visualize_comparison(
            mask, depth, corners_3d, contour_3d, point_cloud,
            keypoints_bilinear, keypoints_fps, keypoints_pure_fps, keypoints_contour_bilinear, 
            INTRINSICS, save_path=str(output_dir / "init_comparison.png")
        )
    
    # Save results
    np.save(str(output_dir / "keypoints_pure_fps.npy"), keypoints_pure_fps)
    np.save(str(output_dir / "corners_detected.npy"), corners_3d)
    
    if has_4_corners:
        np.save(str(output_dir / "keypoints_bilinear.npy"), keypoints_bilinear)
        np.save(str(output_dir / "keypoints_fps.npy"), keypoints_fps)
        np.save(str(output_dir / "keypoints_contour_bilinear.npy"), keypoints_contour_bilinear)
    
    print(f"\nSaved keypoints to: {output_dir}")


if __name__ == "__main__":
    main()
