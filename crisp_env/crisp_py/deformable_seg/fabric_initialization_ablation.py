#!/usr/bin/env python3
"""Fabric initialization-only ablation.

Protocol (per clip):
- Clip length: 300 frames
- Use first 31 frames only
- Frame 0 initialization is the reference
- For frames 1..30, run initialization from scratch for each ablation method
- Compute metrics against frame-0 reference topology/lengths
- Save per-frame ablation images (no tracking videos)

Ablation Methods:
- Full: FPS on contour for borders, bilinear+snap interior, border→contour projection, interior→surface projection, edge repulsion
- NoProj: Same init but NO projection during repulsion
- NoEdge: Same init but NO edge repulsion (just project to surfaces)
- NoAnchor: Bilinear for ALL nodes (skip contour FPS), then normal projection + repulsion
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors

from fabric_tracker import FabricTracker
from fabric_batch_experiment import (
    load_chunk_data,
    load_transforms,
    get_ee_positions_cam,
    compute_edge_metrics,
    compute_position_metrics,
    extract_surface_point_cloud,
    sample_points_on_faces,
    compute_chamfer_metrics,
)


class FabricAblationTracker(FabricTracker):
    """FabricTracker with ablation flags for initialization analysis."""

    def __init__(
        self,
        *args,
        grid_rows: int = 6,
        grid_cols: int = 6,
        use_contour_fps: bool = True,  # If False, skip contour FPS for borders (NoAnchor)
        use_projection: bool = True,
        use_edge_repulsion: bool = True,
        **kwargs
    ):
        # Set grid size before super().__init__() since it computes indices
        self.GRID_ROWS = grid_rows
        self.GRID_COLS = grid_cols
        super().__init__(*args, **kwargs)
        self.use_contour_fps = use_contour_fps
        self.use_projection = use_projection
        self.use_edge_repulsion = use_edge_repulsion

    def _repulsion_relaxation_grid_ablation(
        self,
        keypoints: np.ndarray,
        point_cloud: np.ndarray,
        contour_3d: np.ndarray = None,
    ) -> np.ndarray:
        """
        Spring-based relaxation with ablation flags.
        
        Constraints (always):
        - Corner nodes: Fixed (no movement)
        
        Ablation flags:
        - use_projection: If True, border→contour, interior→surface. If False, no projection.
        - use_edge_repulsion: If True, apply edge springs. If False, just project (no iteration).
        """
        keypoints = keypoints.copy().astype(np.float64)
        K = keypoints.shape[0]
        epsilon = 1e-8

        if K <= 1 or len(point_cloud) == 0:
            return keypoints

        # Build NN index for projection
        cloud_nn = None
        contour_nn = None
        if self.use_projection:
            cloud_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
            cloud_nn.fit(point_cloud)
            
            if contour_3d is not None and len(contour_3d) > 0:
                contour_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
                contour_nn.fit(contour_3d)

        # Compute target edge length from contour perimeter
        contour_length = 0.0
        if contour_3d is not None and len(contour_3d) > 0:
            contour_length = np.sum(np.linalg.norm(np.diff(contour_3d, axis=0), axis=1))
            closing_dist = np.linalg.norm(contour_3d[-1] - contour_3d[0])
            if closing_dist < 100:
                contour_length += closing_dist

        n_border_edges = 4 * (self.GRID_COLS - 1)
        if contour_length > epsilon:
            target_length = contour_length / n_border_edges
            print(f"  Target edge length from contour: {target_length:.1f}mm")
        else:
            edge_lengths = [np.linalg.norm(keypoints[i] - keypoints[j]) 
                          for i, j in self.grid_edges if np.linalg.norm(keypoints[i] - keypoints[j]) > epsilon]
            target_length = np.mean(edge_lengths) if len(edge_lengths) > 0 else 50.0
            print(f"  Target edge length (mean): {target_length:.1f}mm")

        lr = self.repulsion_lr / 25.0
        print(f"  Repulsion: {self.repulsion_iterations} iters, lr={lr:.3f}")
        print(f"    use_projection={self.use_projection}, use_edge_repulsion={self.use_edge_repulsion}")

        # NoEdge: just project to surfaces and done
        if not self.use_edge_repulsion:
            print(f"  NoEdge mode: skipping edge optimization, just projecting")
            if self.use_projection:
                for i in range(K):
                    if i in self.CORNER_INDICES:
                        continue
                    if i in self.BORDER_INDICES and contour_nn is not None:
                        _, idx = contour_nn.kneighbors(keypoints[i:i+1])
                        keypoints[i] = contour_3d[idx[0, 0]]
                    elif cloud_nn is not None:
                        _, idx = cloud_nn.kneighbors(keypoints[i:i+1])
                        keypoints[i] = point_cloud[idx[0, 0]]
            return keypoints

        # Edge repulsion iterations
        for iteration in range(self.repulsion_iterations):
            forces = np.zeros_like(keypoints)

            for i, j in self.grid_edges:
                vec = keypoints[j] - keypoints[i]
                current_length = np.linalg.norm(vec)
                if current_length < epsilon:
                    continue

                direction = vec / current_length
                force_magnitude = (current_length - target_length)
                force = force_magnitude * direction

                forces[i] += force
                forces[j] -= force

            # Apply forces with constraints
            for i in range(K):
                if i in self.CORNER_INDICES:
                    continue  # Corners fixed

                # Apply force
                keypoints[i] += lr * forces[i]

                # Border nodes: snap to contour (if enabled)
                if self.use_projection and i in self.BORDER_INDICES and contour_nn is not None:
                    _, idx = contour_nn.kneighbors(keypoints[i:i+1])
                    keypoints[i] = contour_3d[idx[0, 0]]
                # Interior nodes: free movement during loop (no projection)

            if (iteration + 1) % 50 == 0:
                edge_lengths = [np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in self.grid_edges]
                print(f"    Iter {iteration+1}: mean={np.mean(edge_lengths):.1f}mm, std={np.std(edge_lengths):.1f}mm")

        # Final soft projection for interior nodes (alpha=0.3, same as fabric_tracker.py)
        if self.use_projection and cloud_nn is not None:
            for i in self.INTERIOR_INDICES:
                _, idx = cloud_nn.kneighbors(keypoints[i:i+1])
                nearest = point_cloud[idx[0, 0]]
                keypoints[i] = 0.7 * keypoints[i] + 0.3 * nearest

        return keypoints

    def initialize_ablation(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        frame_idx: int = 0,
    ) -> dict:
        """
        Initialize with ablation flags.
        
        Returns keypoints before and after relaxation for ablation comparison.
        """
        # Extract point cloud
        point_cloud = self._extract_point_cloud(mask, depth)
        if len(point_cloud) < self.min_foreground_pixels:
            return {'success': False, 'reason': 'insufficient_points'}

        # Detect corners from mask FIRST (needed for contour denoising)
        corners_2d = self._find_mask_corners(mask, depth)
        if corners_2d is None:
            return {'success': False, 'reason': 'no_corners'}

        corners_3d = self._pixel_to_3d(corners_2d, depth)
        if corners_3d is None or np.any(np.isnan(corners_3d)):
            return {'success': False, 'reason': 'invalid_corners'}

        # Extract contour WITH corners for denoising
        contour_3d = self._extract_contour_3d(mask, depth, corners_3d=corners_3d)

        # Initialize grid
        if self.use_contour_fps and contour_3d is not None and len(contour_3d) > 12:
            # Full/NoProj/NoEdge: FPS on contour for borders
            keypoints = self._initialize_grid_from_corners(corners_3d, point_cloud, contour_3d)
            print(f"  Init: FPS on contour for borders")
        else:
            # NoAnchor: bilinear for all (skip contour FPS)
            keypoints = self._initialize_grid_bilinear(corners_3d, point_cloud)
            print(f"  Init: Bilinear only (no contour FPS)")

        # Save initial keypoints BEFORE relaxation
        initial_keypoints = keypoints.copy()

        # Build reference lengths
        self.reference_lengths = {}
        for i, j in self.grid_edges:
            length = np.linalg.norm(keypoints[i] - keypoints[j])
            self.reference_lengths[(i, j)] = length
            self.reference_lengths[(j, i)] = length

        # Repulsion relaxation with ablation
        keypoints = self._repulsion_relaxation_grid_ablation(keypoints, point_cloud, contour_3d)

        # Update reference lengths after relaxation
        for i, j in self.grid_edges:
            length = np.linalg.norm(keypoints[i] - keypoints[j])
            self.reference_lengths[(i, j)] = length
            self.reference_lengths[(j, i)] = length

        self.reference_keypoints = keypoints.copy()
        self.prev_keypoints = keypoints.copy()
        self.is_initialized = True

        keypoints_2d = self._project_3d_to_2d(keypoints)

        return {
            'success': True,
            'keypoints': keypoints,
            'initial_keypoints': initial_keypoints,
            'keypoints_2d': keypoints_2d,
            'edges': self.grid_edges,
            'point_cloud': point_cloud,
            'contour_3d': contour_3d,
            'corners_3d': corners_3d,
        }


def run_relaxation_only(
    initial_keypoints: np.ndarray,
    point_cloud: np.ndarray,
    contour_3d: np.ndarray,
    grid_edges: list,
    corner_indices: list,
    border_indices: list,
    interior_indices: list,
    method_name: str,
    repulsion_iterations: int = 500,
    repulsion_lr: float = 5.0,
) -> np.ndarray:
    """
    Run only relaxation step with specified ablation flags.
    All methods share the same initial_keypoints from topology building.
    """
    flags = {
        'Full': dict(use_projection=True, use_edge_repulsion=True),
        'NoProj': dict(use_projection=False, use_edge_repulsion=True),
        'NoEdge': dict(use_projection=True, use_edge_repulsion=False),
        'NoAnchor': dict(use_projection=True, use_edge_repulsion=True),
    }

    use_projection = flags[method_name]['use_projection']
    use_edge_repulsion = flags[method_name]['use_edge_repulsion']

    keypoints = initial_keypoints.copy().astype(np.float64)
    K = keypoints.shape[0]
    epsilon = 1e-8

    if K <= 1 or len(point_cloud) == 0:
        return keypoints

    # Build NN indices
    cloud_nn = None
    contour_nn = None
    if use_projection:
        cloud_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
        cloud_nn.fit(point_cloud)
        if contour_3d is not None and len(contour_3d) > 0:
            contour_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
            contour_nn.fit(contour_3d)

    # Compute target edge length
    contour_length = 0.0
    if contour_3d is not None and len(contour_3d) > 0:
        contour_length = np.sum(np.linalg.norm(np.diff(contour_3d, axis=0), axis=1))
        closing_dist = np.linalg.norm(contour_3d[-1] - contour_3d[0])
        if closing_dist < 100:
            contour_length += closing_dist

    grid_cols = int(np.sqrt(K))  # Assuming square grid
    n_border_edges = 4 * (grid_cols - 1)
    if contour_length > epsilon:
        target_length = contour_length / n_border_edges
    else:
        edge_lengths = [np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in grid_edges]
        target_length = np.mean(edge_lengths) if len(edge_lengths) > 0 else 50.0

    lr = repulsion_lr / 25.0
    print(f"  {method_name}: target_edge={target_length:.1f}mm, projection={use_projection}, edge_repulsion={use_edge_repulsion}")

    # NoEdge: just project
    if not use_edge_repulsion:
        if use_projection:
            for i in range(K):
                if i in corner_indices:
                    continue
                if i in border_indices and contour_nn is not None:
                    _, idx = contour_nn.kneighbors(keypoints[i:i+1])
                    keypoints[i] = contour_3d[idx[0, 0]]
                elif cloud_nn is not None:
                    _, idx = cloud_nn.kneighbors(keypoints[i:i+1])
                    keypoints[i] = point_cloud[idx[0, 0]]
        return keypoints

    # Edge repulsion
    for iteration in range(repulsion_iterations):
        forces = np.zeros_like(keypoints)

        for i, j in grid_edges:
            vec = keypoints[j] - keypoints[i]
            current_length = np.linalg.norm(vec)
            if current_length < epsilon:
                continue
            direction = vec / current_length
            force_magnitude = (current_length - target_length)
            forces[i] += force_magnitude * direction
            forces[j] -= force_magnitude * direction

        for i in range(K):
            if i in corner_indices:
                continue
            keypoints[i] += lr * forces[i]

            # Border nodes: snap to contour
            if use_projection and i in border_indices and contour_nn is not None:
                _, idx = contour_nn.kneighbors(keypoints[i:i+1])
                keypoints[i] = contour_3d[idx[0, 0]]
            # Interior nodes: free movement (no projection during loop)

    # Final soft projection for interior (alpha=0.3, same as fabric_tracker.py)
    if use_projection and cloud_nn is not None:
        for i in interior_indices:
            _, idx = cloud_nn.kneighbors(keypoints[i:i+1])
            nearest = point_cloud[idx[0, 0]]
            keypoints[i] = 0.7 * keypoints[i] + 0.3 * nearest

    return keypoints


def build_init_tracker(intrinsics, grid_rows, grid_cols, ee_pose_single, method_name):
    """Build tracker with ablation flags."""
    ee_poses = np.array([ee_pose_single], dtype=np.float32)

    base_params = {
        'intrinsics': intrinsics,
        'max_depth': 2000.0,
        'repulsion_iterations': 500,
        'repulsion_lr': 5.0,
        'ee_poses_3d': ee_poses,
        'grid_rows': grid_rows,
        'grid_cols': grid_cols,
    }

    flags = {
        'Full': dict(use_contour_fps=True, use_projection=True, use_edge_repulsion=True),
        'NoProj': dict(use_contour_fps=True, use_projection=False, use_edge_repulsion=True),
        'NoEdge': dict(use_contour_fps=True, use_projection=True, use_edge_repulsion=False),
        'NoAnchor': dict(use_contour_fps=False, use_projection=True, use_edge_repulsion=True),
    }

    return FabricAblationTracker(**base_params, **flags[method_name])


def run_initialize_once(tracker, data, frame_idx):
    """Run initialization on a single frame."""
    depth = data['depth'][frame_idx].astype(np.float32)
    mask = data['fg_mask'][frame_idx]
    return tracker.initialize_ablation(mask, depth, frame_idx)


def create_method_panel(rgb, contour_2d, keypoints_2d, edges, method_name, metrics, frame_idx,
                        corner_indices, border_indices):
    """Create visualization panel for one method."""
    H, W = rgb.shape[:2]
    vis = rgb.copy()

    CONTOUR_COLOR = [0, 191, 255]  # Cyan
    EDGE_COLOR = [50, 205, 50]     # Green
    CORNER_COLOR = [128, 0, 128]   # Purple
    BORDER_COLOR = [255, 255, 0]   # Yellow
    INTERIOR_COLOR = [255, 165, 0] # Orange

    # Draw contour (dilated)
    if contour_2d is not None and len(contour_2d) > 0:
        contour_mask = np.zeros((H, W), dtype=np.uint8)
        for i in range(len(contour_2d) - 1):
            pt1 = (int(contour_2d[i, 1]), int(contour_2d[i, 0]))
            pt2 = (int(contour_2d[i+1, 1]), int(contour_2d[i+1, 0]))
            cv2.line(contour_mask, pt1, pt2, 255, 1)
        # Close the contour
        if len(contour_2d) > 2:
            pt1 = (int(contour_2d[-1, 1]), int(contour_2d[-1, 0]))
            pt2 = (int(contour_2d[0, 1]), int(contour_2d[0, 0]))
            cv2.line(contour_mask, pt1, pt2, 255, 1)
        contour_thick = cv2.dilate(contour_mask, np.ones((5, 5), np.uint8), iterations=1)
        vis[contour_thick > 0] = CONTOUR_COLOR

    # Draw edges
    if keypoints_2d is not None and len(keypoints_2d) > 0 and edges is not None:
        kp_int = keypoints_2d.astype(int)
        for (i, j) in edges:
            if i < len(kp_int) and j < len(kp_int):
                p1 = (kp_int[i, 1], kp_int[i, 0])
                p2 = (kp_int[j, 1], kp_int[j, 0])
                cv2.line(vis, p1, p2, EDGE_COLOR, 2)

        # Draw keypoints with colors by type
        for idx, (row, col) in enumerate(kp_int):
            if 0 <= row < H and 0 <= col < W:
                if idx in corner_indices:
                    color = CORNER_COLOR
                elif idx in border_indices:
                    color = BORDER_COLOR
                else:
                    color = INTERIOR_COLOR
                cv2.circle(vis, (col, row), 5, color, -1)
                cv2.putText(vis, str(idx), (col + 6, row - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Add text overlay
    cv2.putText(vis, method_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, f"Frame: {frame_idx}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"Edge: {metrics['edge_pct_mean']:.2f}% (max {metrics['edge_pct_max']:.2f}%)", (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(vis, f"Edge RMSE: {metrics['edge_rmse_mm']:.2f} mm", (10, 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(vis, f"Pos RMSE: {metrics['pos_rmse_mm']:.2f} mm", (10, 126),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


def create_ablation_grid(panels, frame_idx, shape_hw, method_names=None):
    """Create 2x2 grid from 4 panels."""
    H, W = shape_hw
    if len(panels) >= 4:
        row1 = np.concatenate([panels[0], panels[1]], axis=1)
        row2 = np.concatenate([panels[2], panels[3]], axis=1)
        return np.concatenate([row1, row2], axis=0)
    return panels[0] if len(panels) > 0 else np.zeros((H, W, 3), dtype=np.uint8)


def process_clip(data, transforms, ee_poses_3d, clip_idx, start_frame, end_frame, output_dir,
                 grid_rows=6, grid_cols=6, eval_frames=31):
    """Process one clip with all ablation methods."""
    clip_output_dir = output_dir / f'clip_{clip_idx}'
    clip_output_dir.mkdir(parents=True, exist_ok=True)

    frame_end = min(start_frame + eval_frames, end_frame)
    if frame_end - start_frame < 2:
        return None

    method_names = ['Full', 'NoProj', 'NoEdge', 'NoAnchor']
    K = transforms['K']
    intrinsics = np.array([[K[0, 0], 0, K[0, 2]], [0, K[1, 1], K[1, 2]], [0, 0, 1]])

    # Frame 0 reference initialization (Full)
    ref_frame = start_frame
    ref_tracker = build_init_tracker(intrinsics, grid_rows, grid_cols, ee_poses_3d[ref_frame], 'Full')
    ref_result = run_initialize_once(ref_tracker, data, ref_frame)
    if not ref_result.get('success', False):
        print(f"  Clip {clip_idx}: reference init failed at frame {ref_frame}")
        return None

    reference_edges = ref_tracker.grid_edges
    reference_lengths = ref_tracker.reference_lengths.copy()
    corner_indices = ref_tracker.CORNER_INDICES
    border_indices = ref_tracker.BORDER_INDICES
    interior_indices = ref_tracker.INTERIOR_INDICES

    # Compute reference contour length for scaling
    ref_contour_3d = ref_result.get('contour_3d')
    ref_contour_length = 0.0
    if ref_contour_3d is not None and len(ref_contour_3d) > 0:
        ref_contour_length = np.sum(np.linalg.norm(np.diff(ref_contour_3d, axis=0), axis=1))
        closing = np.linalg.norm(ref_contour_3d[-1] - ref_contour_3d[0])
        if closing < 100:
            ref_contour_length += closing
    print(f"  Reference contour length: {ref_contour_length:.1f}mm")

    all_metrics = {m: [] for m in method_names}
    frames_dir = clip_output_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    for global_idx in range(start_frame + 1, frame_end):
        rgb = cv2.cvtColor(data['color'][global_idx], cv2.COLOR_BGR2RGB)
        panel_images = []

        # Run Full init to get shared topology
        full_tracker = build_init_tracker(intrinsics, grid_rows, grid_cols, ee_poses_3d[global_idx], 'Full')
        full_result = run_initialize_once(full_tracker, data, global_idx)

        if not full_result.get('success', False):
            print(f"  Frame {global_idx}: topology building failed")
            for method in method_names:
                all_metrics[method].append({
                    'frame': global_idx - (start_frame + 1),
                    'global_frame': global_idx,
                    'success': False,
                })
            continue

        # Extract shared data
        initial_keypoints = full_result.get('initial_keypoints', full_result['keypoints'])
        shared_edges = full_result.get('edges', reference_edges)
        point_cloud = full_result.get('point_cloud')
        contour_3d = full_result.get('contour_3d')

        # Compute current frame's contour length for scaling
        cur_contour_length = 0.0
        if contour_3d is not None and len(contour_3d) > 0:
            cur_contour_length = np.sum(np.linalg.norm(np.diff(contour_3d, axis=0), axis=1))
            closing = np.linalg.norm(contour_3d[-1] - contour_3d[0])
            if closing < 100:
                cur_contour_length += closing

        # Scale reference lengths
        scale_factor = cur_contour_length / ref_contour_length if ref_contour_length > 1e-6 else 1.0
        scaled_reference_lengths = {k: v * scale_factor for k, v in reference_lengths.items()}

        print(f"  Frame {global_idx}: scale={scale_factor:.3f}")

        # For NoAnchor, we need bilinear initial keypoints
        noanchor_tracker = build_init_tracker(intrinsics, grid_rows, grid_cols, ee_poses_3d[global_idx], 'NoAnchor')
        noanchor_result = run_initialize_once(noanchor_tracker, data, global_idx)
        noanchor_initial = noanchor_result.get('initial_keypoints') if noanchor_result.get('success') else initial_keypoints

        # Run each method's relaxation
        for method in method_names:
            print(f"    Running {method} relaxation...")

            # NoAnchor uses bilinear initial keypoints
            init_kp = noanchor_initial if method == 'NoAnchor' else initial_keypoints

            keypoints = run_relaxation_only(
                initial_keypoints=init_kp,
                point_cloud=point_cloud,
                contour_3d=contour_3d,
                grid_edges=shared_edges,
                corner_indices=corner_indices,
                border_indices=border_indices,
                interior_indices=interior_indices,
                method_name=method,
                repulsion_iterations=500,
                repulsion_lr=5.0,
            )

            keypoints_2d = full_tracker._project_3d_to_2d(keypoints)

            # Compute metrics with scaled reference lengths
            edge_m = compute_edge_metrics(keypoints, shared_edges, scaled_reference_lengths)
            pos_m = compute_position_metrics(keypoints, point_cloud)

            # Chamfer distance - sample dense points on mesh faces to match point cloud size
            n_ref = len(point_cloud)
            n_faces = (grid_rows - 1) * (grid_cols - 1)  # 25 faces for 6x6 grid
            n_samples_per_face = max(10, n_ref // n_faces)  # Match ref point count
            pred_cloud = sample_points_on_faces(keypoints, grid_rows, grid_cols, n_samples_per_face=n_samples_per_face)
            cd_m = compute_chamfer_metrics(pred_cloud, point_cloud)

            metrics = {
                'frame': global_idx - (start_frame + 1),
                'global_frame': global_idx,
                'success': True,
                'edge_pct_mean': edge_m['pct_mean'],
                'edge_pct_std': edge_m['pct_std'],
                'edge_pct_max': edge_m['pct_max'],
                'edge_rmse_mm': edge_m['rmse_mm'],
                'edge_under_2pct': edge_m['under_2pct'],
                'edge_under_5pct': edge_m['under_5pct'],
                'edge_under_10pct': edge_m['under_10pct'],
                'pos_rmse_mm': pos_m['rmse_mm'],
                'pos_under_2mm': pos_m['under_2mm'],
                'pos_under_5mm': pos_m['under_5mm'],
                'pos_under_10mm': pos_m['under_10mm'],
                'cd': cd_m['cd'],
                'cd_pred2ref': cd_m['pred2ref_avg'],
                'cd_ref2pred': cd_m['ref2pred_avg'],
                'precision_2mm': cd_m['precision_2mm'],
                'precision_5mm': cd_m['precision_5mm'],
                'precision_10mm': cd_m['precision_10mm'],
                'recall_2mm': cd_m['recall_2mm'],
                'recall_5mm': cd_m['recall_5mm'],
                'recall_10mm': cd_m['recall_10mm'],
                'f_2mm': cd_m['f_2mm'],
                'f_5mm': cd_m['f_5mm'],
                'f_10mm': cd_m['f_10mm'],
            }
            all_metrics[method].append(metrics)

            # Project contour for visualization
            contour_2d = None
            if contour_3d is not None and len(contour_3d) > 0:
                contour_2d = full_tracker._project_3d_to_2d(contour_3d)

            panel = create_method_panel(
                rgb=rgb,
                contour_2d=contour_2d,
                keypoints_2d=keypoints_2d,
                edges=shared_edges,
                method_name=method,
                metrics=metrics,
                frame_idx=global_idx,
                corner_indices=corner_indices,
                border_indices=border_indices,
            )
            panel_images.append(panel)

        grid = create_ablation_grid(panel_images, global_idx, rgb.shape[:2], method_names=method_names)
        cv2.imwrite(str(frames_dir / f'frame_{global_idx:04d}_ablation.png'), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    # Write summary
    write_summary(clip_output_dir, all_metrics, method_names, start_frame, frame_end - 1)

    return {
        'clip_idx': clip_idx,
        'start_frame': start_frame,
        'end_frame': frame_end,
        'metrics': all_metrics,
    }


def write_summary(output_dir, all_metrics, method_names, start_frame, end_frame):
    """Write summary statistics to file."""
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Clip Init Summary (reference frame {start_frame}, eval frames {start_frame+1}-{end_frame})\n")
        f.write("=" * 100 + "\n")

        # Edge metrics
        f.write("\nEdge Length Metrics\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Method':<12} | {'Edge % Mean':<18} | {'Edge RMSE (mm)':<15} | {'<2%':<8} | {'<5%':<8} | {'<10%':<8}\n")
        f.write("-" * 100 + "\n")

        for method in method_names:
            metrics_list = [m for m in all_metrics[method] if m.get('success', False)]
            if len(metrics_list) == 0:
                continue
            edge_mean = np.mean([m['edge_pct_mean'] for m in metrics_list])
            edge_std = np.std([m['edge_pct_mean'] for m in metrics_list])
            edge_rmse = np.mean([m['edge_rmse_mm'] for m in metrics_list])
            edge_rmse_std = np.std([m['edge_rmse_mm'] for m in metrics_list])
            u2 = np.mean([m['edge_under_2pct'] for m in metrics_list])
            u5 = np.mean([m['edge_under_5pct'] for m in metrics_list])
            u10 = np.mean([m['edge_under_10pct'] for m in metrics_list])
            f.write(f"{method:<12} | {edge_mean:5.2f}% ± {edge_std:.2f}% | {edge_rmse:5.2f} ±{edge_rmse_std:.2f} mm | {u2:5.1f}% | {u5:5.1f}% | {u10:5.1f}%\n")

        # Position metrics
        f.write("\nPosition RMSE Metrics\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<12} | {'Pos RMSE (mm)':<18} | {'<2mm':<8} | {'<5mm':<8} | {'<10mm':<8}\n")
        f.write("-" * 80 + "\n")

        for method in method_names:
            metrics_list = [m for m in all_metrics[method] if m.get('success', False)]
            if len(metrics_list) == 0:
                continue
            pos_rmse = np.mean([m['pos_rmse_mm'] for m in metrics_list])
            pos_std = np.std([m['pos_rmse_mm'] for m in metrics_list])
            u2 = np.mean([m['pos_under_2mm'] for m in metrics_list])
            u5 = np.mean([m['pos_under_5mm'] for m in metrics_list])
            u10 = np.mean([m['pos_under_10mm'] for m in metrics_list])
            f.write(f"{method:<12} | {pos_rmse:5.2f} ± {pos_std:.2f} mm   | {u2:5.1f}% | {u5:5.1f}% | {u10:5.1f}%\n")

        # CD metrics
        f.write("\nChamfer Distance Metrics\n")
        f.write("-" * 130 + "\n")
        f.write(f"{'Method':<12} | {'CD (mm)':<15} | {'Pred→Ref':<10} | {'Ref→Pred':<10} | {'Prec@2mm':<8} | {'Prec@5mm':<8} | {'Prec@10mm':<8} | {'Rec@2mm':<8} | {'Rec@5mm':<8} | {'Rec@10mm':<8}\n")
        f.write("-" * 130 + "\n")

        for method in method_names:
            metrics_list = [m for m in all_metrics[method] if m.get('success', False)]
            if len(metrics_list) == 0:
                continue
            cd = np.mean([m['cd'] for m in metrics_list])
            cd_std = np.std([m['cd'] for m in metrics_list])
            p2r = np.mean([m['cd_pred2ref'] for m in metrics_list])
            r2p = np.mean([m['cd_ref2pred'] for m in metrics_list])
            prec2 = np.mean([m['precision_2mm'] for m in metrics_list])
            prec5 = np.mean([m['precision_5mm'] for m in metrics_list])
            prec10 = np.mean([m['precision_10mm'] for m in metrics_list])
            rec2 = np.mean([m['recall_2mm'] for m in metrics_list])
            rec5 = np.mean([m['recall_5mm'] for m in metrics_list])
            rec10 = np.mean([m['recall_10mm'] for m in metrics_list])
            f.write(f"{method:<12} | {cd:5.2f} ±{cd_std:.2f} mm | {p2r:8.2f} mm | {r2p:8.2f} mm | {prec2:5.1f}% | {prec5:5.1f}% | {prec10:5.1f}% | {rec2:5.1f}% | {rec5:5.1f}% | {rec10:5.1f}%\n")

        # F-scores
        f.write("\nF-Scores\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Method':<12} | {'F@2mm':<12} | {'F@5mm':<12} | {'F@10mm':<12}\n")
        f.write("-" * 60 + "\n")

        for method in method_names:
            metrics_list = [m for m in all_metrics[method] if m.get('success', False)]
            if len(metrics_list) == 0:
                continue
            f2 = np.mean([m['f_2mm'] for m in metrics_list])
            f5 = np.mean([m['f_5mm'] for m in metrics_list])
            f10 = np.mean([m['f_10mm'] for m in metrics_list])
            f.write(f"{method:<12} | {f2:10.2f}% | {f5:10.2f}% | {f10:10.2f}%\n")

    print(f"  Summary written to {summary_path}")


# Dataset configurations
DATASET_CONFIGS = {
    'cloth_no_occlusion_back_3sec': {'clip_frames': 300, 'max_chunks': 19},
    'cloth_no_occlusion_back_4sec': {'clip_frames': 300, 'max_chunks': 30},
    'cloth_no_occlusion_front_3sec': {'clip_frames': 300, 'max_chunks': 19},
    'cloth_no_occlusion_front_4sec': {'clip_frames': 300, 'max_chunks': 30},
}


def main():
    parser = argparse.ArgumentParser(description='Fabric Initialization Ablation')
    parser.add_argument('--chunk', type=int, required=True, help='Chunk index')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset name')
    parser.add_argument('--clip_frames', type=int, default=None, help='Frames per clip (auto from dataset)')
    parser.add_argument('--eval_frames', type=int, default=31, help='Frames to evaluate per clip')
    parser.add_argument('--grid_rows', type=int, default=6, help='Grid rows')
    parser.add_argument('--grid_cols', type=int, default=6, help='Grid columns')
    parser.add_argument('--use_last_n_frames', type=int, default=600, help='Use last N frames from chunk')
    args = parser.parse_args()

    # Dataset paths
    dataset_config = DATASET_CONFIGS[args.dataset]
    clip_frames = args.clip_frames if args.clip_frames is not None else dataset_config['clip_frames']
    max_chunks = dataset_config['max_chunks']
    
    data_dir = Path(f'/mnt/mydisk/captured_data_double_arm/{args.dataset}')
    if args.chunk >= max_chunks:
        print(f"Chunk {args.chunk} not found (max: {max_chunks - 1})")
        return

    chunk_dir = data_dir / f'chunk_{args.chunk}'
    calib_dir = Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/'
                     'roahm-deformable-objects/captured_calibration_data/dlo1_cloth1_calibration')

    output_dir = Path(f'./fabric_init_ablation_results/{args.dataset}/chunk_{args.chunk}')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Fabric Initialization Ablation")
    print(f"{'='*60}")
    print(f"Chunk: {args.chunk}")
    print(f"Dataset: {args.dataset}")
    print(f"Grid: {args.grid_rows}x{args.grid_cols}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Load data
    data = load_chunk_data(chunk_dir, max_frames=args.use_last_n_frames)
    transforms = load_transforms(calib_dir)

    # Compute EE positions for all frames
    n_frames = data['n_frames']
    ee_poses_3d = np.zeros((n_frames, 2, 3), dtype=np.float32)
    for i in range(n_frames):
        ee_poses_3d[i] = get_ee_positions_cam(
            data['left_poses'][i], data['right_poses'][i],
            transforms['T_left_base2cam'], transforms['T_right_base2cam'],
        )

    # Process clips
    n_clips = n_frames // clip_frames
    print(f"Processing {n_clips} clips...")

    for clip_idx in range(n_clips):
        start_frame = clip_idx * clip_frames
        end_frame = min((clip_idx + 1) * clip_frames, n_frames)

        print(f"\nClip {clip_idx}: frames {start_frame}-{end_frame}")
        result = process_clip(
            data, transforms, ee_poses_3d,
            clip_idx, start_frame, end_frame, output_dir,
            grid_rows=args.grid_rows, grid_cols=args.grid_cols,
            eval_frames=args.eval_frames,
        )

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
