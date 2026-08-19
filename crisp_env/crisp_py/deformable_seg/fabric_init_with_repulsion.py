#!/usr/bin/env python3
"""
Fabric initialization using Approach 4:
- Corners: Detected from mask
- Border nodes: FPS on contour segments (guaranteed ON contour)
- Interior nodes: Bilinear interpolation snapped to point cloud

Then apply repulsion relaxation to improve edge uniformity.
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

# Corner indices: 0 (TL), 4 (TR), 24 (BR), 20 (BL)
CORNER_INDICES = [0, 4, 20, 24]

# Border indices (12 nodes on edges, excluding corners)
BORDER_INDICES = [1, 2, 3, 5, 9, 10, 14, 15, 19, 21, 22, 23]

# Interior indices (9 nodes inside)
INTERIOR_INDICES = [6, 7, 8, 11, 12, 13, 16, 17, 18]

# Edge definitions for contour segmentation
# (corner_start_idx, corner_end_idx, [grid_indices for border nodes])
EDGE_DEFINITIONS = [
    (0, 1, [1, 2, 3]),      # Top edge: TL -> TR
    (1, 2, [9, 14, 19]),    # Right edge: TR -> BR
    (2, 3, [23, 22, 21]),   # Bottom edge: BR -> BL
    (3, 0, [15, 10, 5]),    # Left edge: BL -> TL
]

# Grid edges for repulsion/visualization
def build_grid_edges():
    """Build list of (i, j) pairs for adjacent nodes in the 5x5 grid."""
    edges = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            idx = row * GRID_COLS + col
            if col < GRID_COLS - 1:
                edges.append((idx, idx + 1))  # horizontal
            if row < GRID_ROWS - 1:
                edges.append((idx, idx + GRID_COLS))  # vertical
    return edges

GRID_EDGES = build_grid_edges()


# ================================================================
# Utility functions
# ================================================================
def grid_pos_to_idx(row: int, col: int) -> int:
    return row * GRID_COLS + col


def idx_to_grid_pos(idx: int) -> tuple:
    return idx // GRID_COLS, idx % GRID_COLS


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


def extract_contour_3d(mask: np.ndarray, depth: np.ndarray, max_depth: float,
                       fx, fy, cx, cy, sample_step: int = 3) -> np.ndarray:
    """Extract 3D contour points from mask."""
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return np.array([]).reshape(0, 3)
    
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


def find_mask_corners(mask: np.ndarray, depth: np.ndarray, max_depth: float) -> np.ndarray:
    """Find 4 corners of mask using convex hull + approxPolyDP."""
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
        center = corners.mean(axis=0)
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        corners = corners[sorted_indices]
        
        top_indices = np.argsort(corners[:, 1])[:2]
        if corners[top_indices[0], 0] < corners[top_indices[1], 0]:
            tl_idx = top_indices[0]
        else:
            tl_idx = top_indices[1]
        corners = np.roll(corners, -tl_idx, axis=0)
    
    # Return as (row, col)
    corners_rc = np.array([[c[1], c[0]] for c in corners])
    return corners_rc


def farthest_point_sampling(points: np.ndarray, n_samples: int, 
                            seed_points: np.ndarray = None) -> np.ndarray:
    """Farthest Point Sampling with optional seed points as anchors."""
    N = len(points)
    if N == 0:
        return np.array([]).reshape(0, 3)
    
    if n_samples >= N:
        return points.copy()
    
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


# ================================================================
# APPROACH 4: FPS on contour + bilinear interior snap
# ================================================================
def initialize_approach4(corners_3d: np.ndarray, contour_3d: np.ndarray,
                         point_cloud: np.ndarray) -> np.ndarray:
    """
    Approach 4: FPS on Contour Segments + Bilinear Interior Snap
    
    1. Corners: Detected 4 corners (on contour)
    2. Border nodes: FPS on each contour segment (guaranteed ON contour)
    3. Interior nodes: Bilinear interpolation from corners, snapped to point cloud
    """
    keypoints = np.zeros((N_KEYPOINTS, 3), dtype=np.float64)
    
    # Step 1: Place corners
    keypoints[0] = corners_3d[0]   # TL
    keypoints[4] = corners_3d[1]   # TR
    keypoints[24] = corners_3d[2]  # BR
    keypoints[20] = corners_3d[3]  # BL
    
    print("  Step 1: Corners placed")
    
    # Step 2: Find corner positions on contour
    nn_contour = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn_contour.fit(contour_3d)
    _, corner_contour_indices = nn_contour.kneighbors(corners_3d)
    corner_contour_indices = corner_contour_indices.flatten()
    
    print(f"  Step 2: Corner indices on contour: {corner_contour_indices}")
    
    # Step 3: FPS on each contour segment for border nodes
    n_contour = len(contour_3d)
    
    print("  Step 3: FPS on contour segments for border nodes")
    for edge_id, (c_start, c_end, grid_indices) in enumerate(EDGE_DEFINITIONS):
        idx_start = corner_contour_indices[c_start]
        idx_end = corner_contour_indices[c_end]
        
        # Choose shorter path around contour
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
        
        print(f"    Edge {edge_id}: {len(segment)} pts")
        
        if len(segment) >= 5:
            # FPS with corners as anchors
            anchor_start = corners_3d[c_start]
            anchor_end = corners_3d[c_end]
            
            fps_points = farthest_point_sampling(
                segment, n_samples=3, seed_points=np.array([anchor_start, anchor_end])
            )
            
            # Order by distance from start
            dists = np.linalg.norm(fps_points - anchor_start, axis=1)
            fps_points = fps_points[np.argsort(dists)]
            
            for i, idx in enumerate(grid_indices):
                keypoints[idx] = fps_points[i]
        else:
            # Linear interpolation fallback
            corner_start = corners_3d[c_start]
            corner_end = corners_3d[c_end]
            for i, idx in enumerate(grid_indices):
                t = (i + 1) / 4.0
                keypoints[idx] = (1 - t) * corner_start + t * corner_end
    
    # Step 4: Interior nodes - bilinear + snap to point cloud
    print("  Step 4: Bilinear interior + snap to point cloud")
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


# ================================================================
# Repulsion relaxation
# ================================================================
def apply_repulsion(keypoints: np.ndarray, point_cloud: np.ndarray, contour_3d: np.ndarray,
                    iterations: int = 100, lr: float = 0.05, 
                    target_length: float = None) -> np.ndarray:
    """
    Apply repulsion-based relaxation to make edge lengths more uniform.
    
    - Corners: FIXED (never move)
    - Border nodes: Can only move along contour
    - Interior nodes: Can move anywhere, snapped to point cloud
    
    Args:
        keypoints: 25 x 3 initial keypoints
        point_cloud: N x 3 point cloud
        contour_3d: M x 3 contour points
        iterations: Number of relaxation iterations
        lr: Learning rate for movement
        target_length: Target edge length (if None, use mean)
    
    Returns:
        relaxed_keypoints: 25 x 3 relaxed keypoints
    """
    keypoints = keypoints.copy()
    
    # Compute target length if not provided
    if target_length is None:
        lengths = [np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in GRID_EDGES]
        target_length = np.mean(lengths)
    
    print(f"  Target edge length: {target_length:.2f}mm")
    
    # Build nearest neighbor models
    nn_cloud = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn_cloud.fit(point_cloud)
    
    nn_contour = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn_contour.fit(contour_3d)
    
    for iteration in range(iterations):
        # Compute forces on each node
        forces = np.zeros_like(keypoints)
        
        for i, j in GRID_EDGES:
            diff = keypoints[j] - keypoints[i]
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                continue
            
            direction = diff / dist
            force_magnitude = (dist - target_length) * lr
            
            # Apply force to both nodes (opposite directions)
            forces[i] += force_magnitude * direction
            forces[j] -= force_magnitude * direction
        
        # Update positions with constraints
        for idx in range(N_KEYPOINTS):
            if idx in CORNER_INDICES:
                # Corners are fixed
                continue
            elif idx in BORDER_INDICES:
                # Border nodes: move along contour
                new_pos = keypoints[idx] + forces[idx]
                _, nearest_idx = nn_contour.kneighbors(new_pos.reshape(1, -1))
                keypoints[idx] = contour_3d[nearest_idx[0, 0]]
            else:
                # Interior nodes: move freely, snap to point cloud
                new_pos = keypoints[idx] + forces[idx]
                _, nearest_idx = nn_cloud.kneighbors(new_pos.reshape(1, -1))
                keypoints[idx] = point_cloud[nearest_idx[0, 0]]
    
    return keypoints


# ================================================================
# Statistics and visualization
# ================================================================
def compute_edge_stats(keypoints: np.ndarray) -> dict:
    """Compute edge length statistics."""
    lengths = [np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in GRID_EDGES]
    lengths = np.array(lengths)
    return {
        'mean': np.mean(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths),
        'cv': np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0,
        'lengths': lengths
    }


def print_stats(name: str, stats: dict):
    """Print edge statistics."""
    print(f"\n{name}:")
    print(f"  Mean: {stats['mean']:.2f}mm, Std: {stats['std']:.2f}mm")
    print(f"  Min: {stats['min']:.2f}mm, Max: {stats['max']:.2f}mm")
    print(f"  CV: {stats['cv']:.4f} (lower = more uniform)")


def visualize_3d_plotly(point_cloud: np.ndarray, contour_3d: np.ndarray,
                        keypoints_before: np.ndarray, keypoints_after: np.ndarray,
                        title: str = "Fabric Initialization",
                        save_path: str = None):
    """Visualize before and after repulsion in 3D using Plotly."""
    
    # Subsample point cloud
    pc_sub = point_cloud[::20] if len(point_cloud) > 5000 else point_cloud
    
    fig = go.Figure()
    
    # Point cloud (light gray)
    fig.add_trace(go.Scatter3d(
        x=pc_sub[:, 0], y=pc_sub[:, 1], z=pc_sub[:, 2],
        mode='markers',
        marker=dict(size=1.5, color='lightgray', opacity=0.3),
        name='Point Cloud'
    ))
    
    # Contour (cyan)
    fig.add_trace(go.Scatter3d(
        x=contour_3d[:, 0], y=contour_3d[:, 1], z=contour_3d[:, 2],
        mode='markers',
        marker=dict(size=3, color='cyan', opacity=0.6),
        name='Contour'
    ))
    
    # Before repulsion - edges (blue dashed)
    for i, j in GRID_EDGES:
        fig.add_trace(go.Scatter3d(
            x=[keypoints_before[i, 0], keypoints_before[j, 0]],
            y=[keypoints_before[i, 1], keypoints_before[j, 1]],
            z=[keypoints_before[i, 2], keypoints_before[j, 2]],
            mode='lines',
            line=dict(color='blue', width=2, dash='dash'),
            showlegend=False
        ))
    
    # Before repulsion - nodes
    fig.add_trace(go.Scatter3d(
        x=keypoints_before[CORNER_INDICES, 0],
        y=keypoints_before[CORNER_INDICES, 1],
        z=keypoints_before[CORNER_INDICES, 2],
        mode='markers',
        marker=dict(size=10, color='blue', symbol='square'),
        name='Before: Corners'
    ))
    fig.add_trace(go.Scatter3d(
        x=keypoints_before[BORDER_INDICES, 0],
        y=keypoints_before[BORDER_INDICES, 1],
        z=keypoints_before[BORDER_INDICES, 2],
        mode='markers',
        marker=dict(size=7, color='blue', symbol='circle'),
        name='Before: Border'
    ))
    fig.add_trace(go.Scatter3d(
        x=keypoints_before[INTERIOR_INDICES, 0],
        y=keypoints_before[INTERIOR_INDICES, 1],
        z=keypoints_before[INTERIOR_INDICES, 2],
        mode='markers',
        marker=dict(size=6, color='blue', symbol='diamond'),
        name='Before: Interior'
    ))
    
    # After repulsion - edges (red solid)
    for i, j in GRID_EDGES:
        fig.add_trace(go.Scatter3d(
            x=[keypoints_after[i, 0], keypoints_after[j, 0]],
            y=[keypoints_after[i, 1], keypoints_after[j, 1]],
            z=[keypoints_after[i, 2], keypoints_after[j, 2]],
            mode='lines',
            line=dict(color='red', width=3),
            showlegend=False
        ))
    
    # After repulsion - nodes
    fig.add_trace(go.Scatter3d(
        x=keypoints_after[CORNER_INDICES, 0],
        y=keypoints_after[CORNER_INDICES, 1],
        z=keypoints_after[CORNER_INDICES, 2],
        mode='markers',
        marker=dict(size=12, color='red', symbol='square'),
        name='After: Corners'
    ))
    fig.add_trace(go.Scatter3d(
        x=keypoints_after[BORDER_INDICES, 0],
        y=keypoints_after[BORDER_INDICES, 1],
        z=keypoints_after[BORDER_INDICES, 2],
        mode='markers',
        marker=dict(size=9, color='red', symbol='circle'),
        name='After: Border'
    ))
    fig.add_trace(go.Scatter3d(
        x=keypoints_after[INTERIOR_INDICES, 0],
        y=keypoints_after[INTERIOR_INDICES, 1],
        z=keypoints_after[INTERIOR_INDICES, 2],
        mode='markers',
        marker=dict(size=8, color='red', symbol='diamond'),
        name='After: Interior'
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        width=1400,
        height=1000,
        legend=dict(x=0.02, y=0.98)
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Saved to: {save_path}")
    
    return fig


# ================================================================
# Main
# ================================================================
def main():
    # Paths
    data_dir = Path("/home/yehengz/deformable_seg/data")
    tracking_data_path = data_dir / "full" / "tracking_fabric2_data.npy"
    masks_dir = data_dir / "arm_traj4_fabric" / "masks"
    output_dir = data_dir / "arm_traj4_fabric" / "init_with_repulsion"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Camera intrinsics
    fx, fy, cx, cy = 606.6875, 606.24609375, 641.7900390625, 366.8428955078125
    max_depth = 1100.0
    
    # Load data
    print("="*60)
    print("Loading data")
    print("="*60)
    tracking_data = np.load(str(tracking_data_path), allow_pickle=True).item()
    frame_keys = sorted([k for k in tracking_data.keys() if isinstance(k, int)])
    print(f"Found {len(frame_keys)} frames")
    
    frame_data = tracking_data[frame_keys[0]]
    depth = frame_data['transformed_depth']
    
    mask_file = masks_dir / "mask_frame_0000.npy"
    mask_raw = np.load(str(mask_file))
    
    # Apply depth thresholding
    valid_depth = (depth > 0) & (depth < max_depth)
    mask = mask_raw & valid_depth
    
    print(f"Mask: {np.sum(mask_raw)} -> {np.sum(mask)} (depth filtered)")
    
    # Extract data
    corners_2d = find_mask_corners(mask, depth, max_depth)
    if corners_2d is None:
        print("Failed to find corners!")
        return
    
    corners_3d = pixel_to_3d(corners_2d, depth, fx, fy, cx, cy)
    contour_3d = extract_contour_3d(mask, depth, max_depth, fx, fy, cx, cy, sample_step=3)
    point_cloud = extract_point_cloud(mask, depth, max_depth, fx, fy, cx, cy)
    
    print(f"Corners 3D: {corners_3d.shape}")
    print(f"Contour 3D: {contour_3d.shape}")
    print(f"Point cloud: {point_cloud.shape}")
    
    # Initialize with Approach 4
    print("\n" + "="*60)
    print("Approach 4: FPS Contour + Bilinear Interior")
    print("="*60)
    keypoints_before = initialize_approach4(corners_3d, contour_3d, point_cloud)
    
    stats_before = compute_edge_stats(keypoints_before)
    print_stats("Before Repulsion", stats_before)
    
    # Apply repulsion
    print("\n" + "="*60)
    print("Applying Repulsion Relaxation")
    print("="*60)
    keypoints_after = apply_repulsion(
        keypoints_before, point_cloud, contour_3d,
        iterations=100, lr=0.05
    )
    
    stats_after = compute_edge_stats(keypoints_after)
    print_stats("After Repulsion", stats_after)
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"CV improvement: {stats_before['cv']:.4f} -> {stats_after['cv']:.4f}")
    print(f"Std improvement: {stats_before['std']:.2f}mm -> {stats_after['std']:.2f}mm")
    
    # Visualize
    print("\n" + "="*60)
    print("Saving Visualization")
    print("="*60)
    visualize_3d_plotly(
        point_cloud, contour_3d, keypoints_before, keypoints_after,
        title="Fabric Init: Before (blue) vs After (red) Repulsion",
        save_path=str(output_dir / "init_with_repulsion.html")
    )
    
    # Save keypoints
    np.save(str(output_dir / "keypoints_before_repulsion.npy"), keypoints_before)
    np.save(str(output_dir / "keypoints_after_repulsion.npy"), keypoints_after)
    print(f"Saved keypoints to: {output_dir}")


if __name__ == "__main__":
    main()
