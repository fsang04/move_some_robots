"""
Fabric Tracking Projection Ablation Study

Evaluates the impact of interior node point cloud projection during geometry constraint:
    - Full: CPD + geometry (corners fixed, borders→contour, interior→surface soft projection)
    - NoProj: CPD + geometry (corners fixed, borders→contour, interior NO projection)

Both methods use EE anchor constraints and corner snapping. This isolates the effect
of interior node projection toward the point cloud.

Datasets:
    - cloth_no_occlusion_back_3sec (chunks: 0, 3, 7, 12, 20)
    - cloth_no_occlusion_back_4sec (chunks: 8, 13)
    - cloth_no_occlusion_front_3sec (chunks: 2, 5, 6, 7, 11, 14, 17)
    - cloth_no_occlusion_front_4sec (chunks: 15, 21, 22, 23, 27, 28)

Output (exactly matches existing evaluation format):
    - 3d_keypoints.npz: {full, noproj, edge_connection, reference_lengths}
    - per_frame.csv
    - summary.txt
    - video_ablation.mp4
    - chunk_summary/

Usage:
    python fabric_tracking_ablation.py --dataset cloth_no_occlusion_back_3sec --chunk 0 --clip_seconds 10

Author: Auto-generated
Date: 2025-03-02
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fabric_tracker import FabricTracker


# ============================================================================
# CONSTANTS
# ============================================================================

DATA_BASE = Path('/mnt/mydisk/captured_data_double_arm')
CALIB_DIR = Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/'
                 'roahm-deformable-objects/captured_calibration_data/dlo1_cloth1_calibration')
OUTPUT_BASE = Path('./fabric_tracking_ablation_results')

FPS = 30
GRID_ROWS = 6
GRID_COLS = 6
N_KEYPOINTS = GRID_ROWS * GRID_COLS  # 36

# Datasets with their chunks (matching fabric_evaluation_results structure)
DATASETS = {
    'cloth_no_occlusion_back_3sec': [0, 3, 7, 12, 20],
    'cloth_no_occlusion_back_4sec': [8, 13],
    'cloth_no_occlusion_front_3sec': [2, 5, 6, 7, 11, 14, 17],
    'cloth_no_occlusion_front_4sec': [15, 21, 22, 23, 27, 28],
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_chunk_data(chunk_dir: Path, max_frames: int = 600) -> dict:
    """Load all data from a chunk directory."""
    rgbd = np.load(chunk_dir / 'rgbd.npz')
    n_total = rgbd['color'].shape[0]
    start_idx = max(0, n_total - max_frames)
    
    color = rgbd['color'][start_idx:]
    depth = rgbd['depth'][start_idx:]
    
    # Load foreground mask
    masks_path = chunk_dir / 'masks' / 'masks.npz'
    if masks_path.exists():
        fg_masks = np.load(masks_path)['masks'][start_idx:]
    else:
        fg_masks = None
    
    # Load poses
    left_poses_npz = np.load(chunk_dir / 'left_arm_poses.npz')
    right_poses_npz = np.load(chunk_dir / 'right_arm_poses.npz')
    
    n_poses_total = len(left_poses_npz.files)
    pose_start_idx = max(0, n_poses_total - max_frames)
    left_poses = np.array([left_poses_npz[f'arr_{i}'] for i in range(pose_start_idx, n_poses_total)])
    right_poses = np.array([right_poses_npz[f'arr_{i}'] for i in range(pose_start_idx, n_poses_total)])
    
    n_frames = color.shape[0]
    left_poses = left_poses[:n_frames]
    right_poses = right_poses[:n_frames]
    
    return {
        'color': color,
        'depth': depth,
        'fg_masks': fg_masks,
        'left_poses': left_poses,
        'right_poses': right_poses,
        'n_frames': n_frames,
    }


def load_transforms(calib_dir: Path) -> dict:
    """Load camera-robot transforms."""
    tf = np.load(calib_dir / 'transform_ee_cam_world.npz')
    return {
        'T_left_base2cam': tf['T_left_base2cam'],
        'T_right_base2cam': tf['T_right_base2cam'],
        'K': tf['K'],
    }


def pose7_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert [x,y,z,qw,qx,qy,qz] to 4x4 matrix."""
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    quat = pose[3:]
    T[:3, :3] = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    return T


def get_ee_positions_cam(left_pose, right_pose, T_left_base2cam, T_right_base2cam):
    """Convert EE poses to camera frame (mm)."""
    T_left_ee = pose7_to_matrix(left_pose)
    left_pos_cam = (T_left_base2cam @ T_left_ee)[:3, 3]
    T_right_ee = pose7_to_matrix(right_pose)
    right_pos_cam = (T_right_base2cam @ T_right_ee)[:3, 3]
    return np.array([left_pos_cam * 1000, right_pos_cam * 1000])


# ============================================================================
# METRICS
# ============================================================================

def compute_edge_metrics(keypoints, edges, reference_lengths):
    """Compute edge length metrics."""
    if keypoints is None or len(keypoints) == 0 or edges is None or len(edges) == 0:
        return {
            'pct_errors': np.array([]), 'abs_errors': np.array([]),
            'pct_mean': 0.0, 'pct_std': 0.0, 'pct_max': 0.0, 'rmse_mm': 0.0,
            'under_2pct': 0.0, 'under_5pct': 0.0, 'under_10pct': 0.0,
        }

    pct_errors = []
    abs_errors = []
    for (i, j) in edges:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        edge_key = (i, j) if (i, j) in reference_lengths else (j, i)
        ref_length = reference_lengths.get(edge_key, 0)
        if ref_length > 1e-6:
            current_length = np.linalg.norm(keypoints[i] - keypoints[j])
            abs_err = abs(current_length - ref_length)
            pct_err = abs_err / ref_length
            pct_errors.append(pct_err)
            abs_errors.append(abs_err)

    pct_errors = np.array(pct_errors)
    abs_errors = np.array(abs_errors)

    if len(pct_errors) == 0:
        return {
            'pct_errors': np.array([]), 'abs_errors': np.array([]),
            'pct_mean': 0.0, 'pct_std': 0.0, 'pct_max': 0.0, 'rmse_mm': 0.0,
            'under_2pct': 0.0, 'under_5pct': 0.0, 'under_10pct': 0.0,
        }

    return {
        'pct_errors': pct_errors,
        'abs_errors': abs_errors,
        'pct_mean': np.mean(pct_errors) * 100,
        'pct_std': np.std(pct_errors) * 100,
        'pct_max': np.max(pct_errors) * 100,
        'rmse_mm': np.sqrt(np.mean(abs_errors ** 2)),
        'under_2pct': np.mean(pct_errors < 0.02) * 100,
        'under_5pct': np.mean(pct_errors < 0.05) * 100,
        'under_10pct': np.mean(pct_errors < 0.10) * 100,
    }


def compute_position_metrics(keypoints, point_cloud, extra_gt_points=None):
    """Compute position RMSE (nearest neighbor to point cloud)."""
    if keypoints is None or len(keypoints) == 0 or point_cloud is None or len(point_cloud) == 0:
        return {
            'distances': np.array([]),
            'rmse_mm': 0.0,
            'under_2mm': 0.0, 'under_5mm': 0.0, 'under_10mm': 0.0,
        }

    gt_cloud = point_cloud
    if extra_gt_points is not None and len(extra_gt_points) > 0:
        gt_cloud = np.vstack([point_cloud, extra_gt_points])

    nn = NearestNeighbors(n_neighbors=1).fit(gt_cloud)
    distances, _ = nn.kneighbors(keypoints)
    distances = distances.flatten()

    return {
        'distances': distances,
        'rmse_mm': np.sqrt(np.mean(distances ** 2)),
        'under_2mm': np.mean(distances < 2.0) * 100,
        'under_5mm': np.mean(distances < 5.0) * 100,
        'under_10mm': np.mean(distances < 10.0) * 100,
    }


def sample_points_on_faces(keypoints, grid_rows, grid_cols, n_samples_per_face=10):
    """Sample points on grid faces (quads) for Chamfer distance."""
    if keypoints is None or len(keypoints) == 0:
        return np.empty((0, 3), dtype=np.float32)
    
    sampled_points = []
    
    # Each face is a quad formed by 4 neighboring grid nodes
    for r in range(grid_rows - 1):
        for c in range(grid_cols - 1):
            tl = r * grid_cols + c
            tr = r * grid_cols + c + 1
            bl = (r + 1) * grid_cols + c
            br = (r + 1) * grid_cols + c + 1
            
            if any(idx >= len(keypoints) for idx in [tl, tr, bl, br]):
                continue
            
            p_tl = keypoints[tl]
            p_tr = keypoints[tr]
            p_bl = keypoints[bl]
            p_br = keypoints[br]
            
            # Sample points on the quad using bilinear interpolation
            for _ in range(n_samples_per_face):
                u = np.random.random()
                v = np.random.random()
                p = (1 - u) * (1 - v) * p_tl + u * (1 - v) * p_tr + \
                    (1 - u) * v * p_bl + u * v * p_br
                sampled_points.append(p)
    
    if len(sampled_points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    
    return np.array(sampled_points, dtype=np.float32)


def compute_chamfer_metrics(pred_cloud, ref_cloud):
    """Compute Chamfer Distance metrics."""
    empty_result = {
        'pred2ref_avg': 0.0, 'ref2pred_avg': 0.0, 'cd': 0.0,
        'precision_2mm': 0.0, 'precision_5mm': 0.0, 'precision_10mm': 0.0,
        'recall_2mm': 0.0, 'recall_5mm': 0.0, 'recall_10mm': 0.0,
        'f_2mm': 0.0, 'f_5mm': 0.0, 'f_10mm': 0.0,
    }
    
    if pred_cloud is None or len(pred_cloud) == 0 or ref_cloud is None or len(ref_cloud) == 0:
        return empty_result
    
    nn_ref = NearestNeighbors(n_neighbors=1).fit(ref_cloud)
    pred2ref_dists, _ = nn_ref.kneighbors(pred_cloud)
    pred2ref_dists = pred2ref_dists.flatten()
    
    nn_pred = NearestNeighbors(n_neighbors=1).fit(pred_cloud)
    ref2pred_dists, _ = nn_pred.kneighbors(ref_cloud)
    ref2pred_dists = ref2pred_dists.flatten()
    
    pred2ref_avg = np.mean(pred2ref_dists)
    ref2pred_avg = np.mean(ref2pred_dists)
    cd = (pred2ref_avg + ref2pred_avg) / 2
    
    precision_2mm = np.mean(pred2ref_dists < 2.0) * 100
    precision_5mm = np.mean(pred2ref_dists < 5.0) * 100
    precision_10mm = np.mean(pred2ref_dists < 10.0) * 100
    
    recall_2mm = np.mean(ref2pred_dists < 2.0) * 100
    recall_5mm = np.mean(ref2pred_dists < 5.0) * 100
    recall_10mm = np.mean(ref2pred_dists < 10.0) * 100
    
    def f_score(p, r):
        if p + r < 1e-6:
            return 0.0
        return 2 * p * r / (p + r)
    
    return {
        'pred2ref_avg': pred2ref_avg,
        'ref2pred_avg': ref2pred_avg,
        'cd': cd,
        'precision_2mm': precision_2mm,
        'precision_5mm': precision_5mm,
        'precision_10mm': precision_10mm,
        'recall_2mm': recall_2mm,
        'recall_5mm': recall_5mm,
        'recall_10mm': recall_10mm,
        'f_2mm': f_score(precision_2mm, recall_2mm),
        'f_5mm': f_score(precision_5mm, recall_5mm),
        'f_10mm': f_score(precision_10mm, recall_10mm),
    }


# ============================================================================
# CUSTOM TRACKER WITH NO PROJECTION
# ============================================================================

class FabricTrackerNoProj(FabricTracker):
    """Fabric tracker with interior node projection disabled during geometry constraint."""
    
    def _joint_constraint_optimization_with_contour(
        self,
        keypoints: np.ndarray,
        point_cloud: np.ndarray,
        contour_3d: np.ndarray,
    ) -> np.ndarray:
        """
        Joint edge length + contour constraint WITHOUT interior projection.
        
        Constraints:
        - Corner nodes: Fixed
        - Border nodes: Snap to 3D contour
        - Interior nodes: Only edge correction (NO projection to point cloud)
        """
        keypoints = keypoints.copy().astype(np.float64)
        K = keypoints.shape[0]
        epsilon = 1e-8
        
        if len(point_cloud) == 0:
            return keypoints
        
        # Build NN for contour only (not for interior projection)
        contour_nn = None
        if contour_3d is not None and len(contour_3d) > 0:
            contour_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
            contour_nn.fit(contour_3d)
        
        for outer_iter in range(self.n_outer_iterations):
            # Edge correction
            for edge_iter in range(self.n_edge_iterations):
                for i, j in self.grid_edges:
                    # Skip if both are corners (fixed)
                    if i in self.CORNER_INDICES and j in self.CORNER_INDICES:
                        continue
                    
                    target_length = self.reference_lengths.get((i, j), 0)
                    if target_length < epsilon:
                        continue
                    
                    current_vec = keypoints[j] - keypoints[i]
                    current_length = np.linalg.norm(current_vec)
                    
                    if current_length < epsilon:
                        continue
                    
                    error = (current_length - target_length) / target_length
                    
                    if abs(error) > self.edge_tolerance:
                        direction = current_vec / current_length
                        correction = (current_length - target_length) * self.edge_weight / 2
                        
                        # Apply correction based on node type
                        i_is_corner = i in self.CORNER_INDICES
                        j_is_corner = j in self.CORNER_INDICES
                        
                        if not i_is_corner and not j_is_corner:
                            keypoints[i] += correction * direction
                            keypoints[j] -= correction * direction
                        elif not i_is_corner:
                            keypoints[i] += 2 * correction * direction
                        elif not j_is_corner:
                            keypoints[j] -= 2 * correction * direction
            
            # Snap border nodes to contour only (no interior projection)
            for i in range(K):
                if i in self.CORNER_INDICES:
                    continue  # Corners are fixed
                
                if i in self.BORDER_INDICES:
                    # Border nodes: snap to contour
                    if contour_nn is not None:
                        _, idx = contour_nn.kneighbors(keypoints[i:i+1])
                        keypoints[i] = contour_3d[idx[0, 0]]
                # Interior nodes: NO projection (skip entirely)
        
        return keypoints


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_method_panel(rgb, mask, keypoints_2d, edges, method_name, metrics, frame_idx,
                        grid_rows, grid_cols, traj_history_2d=None, tail_length=60):
    """Create visualization panel for a single method."""
    H, W = rgb.shape[:2]
    vis = rgb.copy()

    MASK_COLOR = [0, 191, 255]
    EDGE_COLOR = [50, 205, 50]
    CORNER_COLOR = [255, 0, 0]
    BORDER_COLOR = [255, 255, 0]
    INTERIOR_COLOR = [255, 165, 0]
    TAIL_COLOR = [100, 255, 100]

    # Overlay mask contour
    if mask is not None:
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, MASK_COLOR, 2)

    # Trajectory tails
    if traj_history_2d is not None and len(traj_history_2d) > 1:
        n_history = len(traj_history_2d)
        n_kp = traj_history_2d.shape[1]
        start_idx = max(0, n_history - tail_length)
        for k_idx in range(n_kp):
            for t in range(start_idx, n_history - 1):
                pt1 = traj_history_2d[t, k_idx]
                pt2 = traj_history_2d[t + 1, k_idx]
                if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
                    continue
                r1, c1 = int(pt1[0]), int(pt1[1])
                r2, c2 = int(pt2[0]), int(pt2[1])
                if not (0 <= r1 < H and 0 <= c1 < W and 0 <= r2 < H and 0 <= c2 < W):
                    continue
                age = n_history - 1 - t
                alpha = max(0.2, 1.0 - age / tail_length)
                color = [int(c * alpha) for c in TAIL_COLOR]
                cv2.line(vis, (c1, r1), (c2, r2), color, 2)

    # Draw edges and keypoints
    if keypoints_2d is not None and len(keypoints_2d) > 0:
        kp_int = keypoints_2d.astype(int)
        if edges is not None:
            for (i, j) in edges:
                if i < len(kp_int) and j < len(kp_int):
                    p1 = (kp_int[i, 1], kp_int[i, 0])
                    p2 = (kp_int[j, 1], kp_int[j, 0])
                    cv2.line(vis, p1, p2, EDGE_COLOR, 2)
        
        # Compute corner, border, interior indices
        n_cols = grid_cols
        corner_indices = [0, n_cols - 1, 
                          (grid_rows - 1) * n_cols, grid_rows * n_cols - 1]
        border_indices = (
            list(range(1, n_cols - 1)) +
            list(range((grid_rows - 1) * n_cols + 1, grid_rows * n_cols - 1)) +
            [r * n_cols for r in range(1, grid_rows - 1)] +
            [r * n_cols + n_cols - 1 for r in range(1, grid_rows - 1)]
        )
        
        for idx, (row, col) in enumerate(kp_int):
            if 0 <= row < H and 0 <= col < W:
                if idx in corner_indices:
                    color = CORNER_COLOR
                    radius = 7
                elif idx in border_indices:
                    color = BORDER_COLOR
                    radius = 5
                else:
                    color = INTERIOR_COLOR
                    radius = 4
                cv2.circle(vis, (col, row), radius, color, -1)

    # Text overlay
    cv2.putText(vis, method_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, f"Frame: {frame_idx}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"Edge: {metrics['edge_pct_mean']:.2f}% (max {metrics['edge_pct_max']:.2f}%)", (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(vis, f"Edge RMSE: {metrics['edge_rmse_mm']:.2f} mm", (10, 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(vis, f"Pos RMSE: {metrics['pos_rmse_mm']:.2f} mm", (10, 126),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


def create_ablation_grid(panels, frame_idx, shape_hw):
    """Create 1x2 grid for 2-method ablation."""
    H, W = shape_hw
    
    if len(panels) == 2:
        return np.concatenate([panels[0], panels[1]], axis=1)
    
    while len(panels) < 2:
        panels.append(np.zeros((H, W, 3), dtype=np.uint8))
    return np.concatenate([panels[0], panels[1]], axis=1)


# ============================================================================
# CLIP PROCESSING
# ============================================================================

def process_clip(data, transforms, ee_poses_3d, start_frame, end_frame, clip_idx,
                 output_dir, grid_rows, grid_cols, fps, tail_length=60):
    """Process a single clip with Full and NoProj methods."""
    
    clip_output_dir = output_dir / f'clip_{clip_idx}'
    clip_output_dir.mkdir(parents=True, exist_ok=True)
    
    n_frames = end_frame - start_frame
    clip_ee_poses = ee_poses_3d[start_frame:end_frame]
    
    # Construct intrinsics matrix explicitly
    K = transforms['K']
    intrinsics = np.array([
        [K[0, 0], 0, K[0, 2]],
        [0, K[1, 1], K[1, 2]],
        [0, 0, 1]
    ])
    
    # Base parameters
    base_params = {
        'intrinsics': intrinsics,
        'max_depth': 1250.0,
        'n_outer_iterations': 5,
        'n_edge_iterations': 20,
        'edge_weight': 0.5,
        'edge_tolerance': 0.15,
        'repulsion_iterations': 500,
        'repulsion_lr': 5.0,
        'ee_poses_3d': clip_ee_poses,
    }
    
    method_names = ['Full', 'NoProj']
    
    # Create trackers
    trackers = {
        'Full': FabricTracker(**base_params),
        'NoProj': FabricTrackerNoProj(**base_params),
    }
    
    # Video writer
    video_path = clip_output_dir / 'video_ablation.mp4'
    video_writer = None
    
    all_metrics = {m: [] for m in method_names}
    traj_histories = {m: [] for m in method_names}
    keypoints_3d_histories = {m: [] for m in method_names}
    stored_edges = None
    stored_reference_lengths = None
    
    for local_idx, global_idx in enumerate(tqdm(range(start_frame, end_frame), desc=f"    Clip {clip_idx}")):
        rgb = data['color'][global_idx]
        depth = data['depth'][global_idx].astype(np.float32)
        fg_mask = data['fg_masks'][global_idx] if data['fg_masks'] is not None else None
        
        panel_images = []
        
        for method in method_names:
            tracker = trackers[method]
            result = tracker.process_frame(
                mask=fg_mask,
                depth=depth,
                frame_idx=local_idx,
            )
            
            if result['success']:
                keypoints = result['keypoints']
                keypoints_2d = result['keypoints_2d']
                edges = result.get('edges', tracker.grid_edges)
                traj_histories[method].append(keypoints_2d.copy())
                keypoints_3d_histories[method].append(keypoints.copy())
                if stored_edges is None and edges is not None:
                    stored_edges = list(edges)
                if stored_reference_lengths is None and tracker.reference_lengths is not None:
                    stored_reference_lengths = dict(tracker.reference_lengths)
            else:
                keypoints = np.empty((0, 3))
                keypoints_2d = np.empty((0, 2))
                edges = []
                n_kp = grid_rows * grid_cols
                if len(traj_histories[method]) > 0:
                    traj_histories[method].append(np.full_like(traj_histories[method][-1], np.nan))
                else:
                    traj_histories[method].append(np.full((n_kp, 2), np.nan))
                keypoints_3d_histories[method].append(np.full((n_kp, 3), np.nan))
            
            # Compute metrics
            if result['success'] and tracker.reference_lengths is not None:
                point_cloud = result.get('point_cloud')
                ee_pos = clip_ee_poses[local_idx]
                if point_cloud is not None and len(point_cloud) > 0:
                    if ee_pos is not None and len(ee_pos) > 0:
                        ref_pc = np.vstack([point_cloud, np.array(ee_pos, dtype=np.float32).reshape(-1, 3)])
                    else:
                        ref_pc = point_cloud
                else:
                    ref_pc = np.array(ee_pos, dtype=np.float32).reshape(-1, 3) if ee_pos is not None else np.empty((0, 3))
                
                ref_lengths = stored_reference_lengths if stored_reference_lengths is not None else tracker.reference_lengths
                edge_m = compute_edge_metrics(keypoints, edges, ref_lengths)
                pos_m = compute_position_metrics(keypoints, ref_pc)
                
                # Sample on faces for Chamfer
                pred_cloud = sample_points_on_faces(keypoints, grid_rows, grid_cols, n_samples_per_face=20)
                cd_m = compute_chamfer_metrics(pred_cloud, ref_pc)
                
                metrics = {
                    'frame': local_idx,
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
            else:
                metrics = {
                    'frame': local_idx, 'global_frame': global_idx, 'success': False,
                    'edge_pct_mean': 0.0, 'edge_pct_std': 0.0, 'edge_pct_max': 0.0,
                    'edge_rmse_mm': 0.0, 'edge_under_2pct': 0.0, 'edge_under_5pct': 0.0,
                    'edge_under_10pct': 0.0, 'pos_rmse_mm': 0.0, 'pos_under_2mm': 0.0,
                    'pos_under_5mm': 0.0, 'pos_under_10mm': 0.0,
                    'cd': 0.0, 'cd_pred2ref': 0.0, 'cd_ref2pred': 0.0,
                    'precision_2mm': 0.0, 'precision_5mm': 0.0, 'precision_10mm': 0.0,
                    'recall_2mm': 0.0, 'recall_5mm': 0.0, 'recall_10mm': 0.0,
                    'f_2mm': 0.0, 'f_5mm': 0.0, 'f_10mm': 0.0,
                }
            
            all_metrics[method].append(metrics)
            
            panel = create_method_panel(
                rgb=rgb, mask=fg_mask, keypoints_2d=keypoints_2d,
                edges=edges, method_name=method, metrics=metrics, frame_idx=local_idx,
                grid_rows=grid_rows, grid_cols=grid_cols,
                traj_history_2d=np.array(traj_histories[method]) if traj_histories[method] else None,
                tail_length=tail_length,
            )
            panel_images.append(panel)
        
        grid = create_ablation_grid(panel_images, local_idx, rgb.shape[:2])
        
        if video_writer is None:
            H, W = grid.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (W, H))
        
        video_writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    
    if video_writer is not None:
        video_writer.release()
    
    # Save per-frame CSV
    per_frame_csv = clip_output_dir / 'per_frame.csv'
    with open(per_frame_csv, 'w') as f:
        f.write('Frame,GlobalFrame,Method,EdgePctMean,EdgePctStd,EdgePctMax,EdgeRMSE,PosRMSE,'
                'Edge<2%,Edge<5%,Edge<10%,Pos<2mm,Pos<5mm,Pos<10mm,'
                'CD,Pred2Ref,Ref2Pred,Prec@2mm,Prec@5mm,Prec@10mm,Rec@2mm,Rec@5mm,Rec@10mm,F@2mm,F@5mm,F@10mm\n')
        for local_idx in range(n_frames):
            for method in method_names:
                m = all_metrics[method][local_idx]
                f.write(f"{local_idx},{m['global_frame']},{method},{m['edge_pct_mean']:.6f},{m['edge_pct_std']:.6f},"
                        f"{m['edge_pct_max']:.6f},{m['edge_rmse_mm']:.6f},{m['pos_rmse_mm']:.6f},"
                        f"{m['edge_under_2pct']:.4f},{m['edge_under_5pct']:.4f},{m['edge_under_10pct']:.4f},"
                        f"{m['pos_under_2mm']:.4f},{m['pos_under_5mm']:.4f},{m['pos_under_10mm']:.4f},"
                        f"{m['cd']:.4f},{m['cd_pred2ref']:.4f},{m['cd_ref2pred']:.4f},"
                        f"{m['precision_2mm']:.4f},{m['precision_5mm']:.4f},{m['precision_10mm']:.4f},"
                        f"{m['recall_2mm']:.4f},{m['recall_5mm']:.4f},{m['recall_10mm']:.4f},"
                        f"{m['f_2mm']:.4f},{m['f_5mm']:.4f},{m['f_10mm']:.4f}\n")
    
    # Compute summary
    summary_rows = []
    for method in method_names:
        metrics_list = all_metrics[method][1:] if len(all_metrics[method]) > 1 else all_metrics[method]
        if len(metrics_list) == 0:
            continue
        
        valid_metrics = [m for m in metrics_list if m['success']]
        
        edge_pct_means = [m['edge_pct_mean'] for m in valid_metrics]
        edge_rmses = [m['edge_rmse_mm'] for m in valid_metrics]
        edge_under_2pct = [m['edge_under_2pct'] for m in valid_metrics]
        edge_under_5pct = [m['edge_under_5pct'] for m in valid_metrics]
        edge_under_10pct = [m['edge_under_10pct'] for m in valid_metrics]
        
        pos_rmses = [m['pos_rmse_mm'] for m in valid_metrics]
        pos_under_2mm = [m['pos_under_2mm'] for m in valid_metrics]
        pos_under_5mm = [m['pos_under_5mm'] for m in valid_metrics]
        pos_under_10mm = [m['pos_under_10mm'] for m in valid_metrics]
        
        cd_vals = [m['cd'] for m in valid_metrics]
        cd_pred2ref = [m['cd_pred2ref'] for m in valid_metrics]
        cd_ref2pred = [m['cd_ref2pred'] for m in valid_metrics]
        
        prec_2mm = [m['precision_2mm'] for m in valid_metrics]
        prec_5mm = [m['precision_5mm'] for m in valid_metrics]
        prec_10mm = [m['precision_10mm'] for m in valid_metrics]
        
        rec_2mm = [m['recall_2mm'] for m in valid_metrics]
        rec_5mm = [m['recall_5mm'] for m in valid_metrics]
        rec_10mm = [m['recall_10mm'] for m in valid_metrics]
        
        f_2mm = [m['f_2mm'] for m in valid_metrics]
        f_5mm = [m['f_5mm'] for m in valid_metrics]
        f_10mm = [m['f_10mm'] for m in valid_metrics]
        
        summary_rows.append({
            'method': method,
            'edge_pct_mean_avg': np.mean(edge_pct_means) if edge_pct_means else 0.0,
            'edge_pct_mean_std': np.std(edge_pct_means) if edge_pct_means else 0.0,
            'edge_rmse_avg': np.mean(edge_rmses) if edge_rmses else 0.0,
            'edge_rmse_std': np.std(edge_rmses) if edge_rmses else 0.0,
            'edge_under_2pct': np.mean(edge_under_2pct) if edge_under_2pct else 0.0,
            'edge_under_5pct': np.mean(edge_under_5pct) if edge_under_5pct else 0.0,
            'edge_under_10pct': np.mean(edge_under_10pct) if edge_under_10pct else 0.0,
            'pos_rmse_avg': np.mean(pos_rmses) if pos_rmses else 0.0,
            'pos_rmse_std': np.std(pos_rmses) if pos_rmses else 0.0,
            'pos_under_2mm': np.mean(pos_under_2mm) if pos_under_2mm else 0.0,
            'pos_under_5mm': np.mean(pos_under_5mm) if pos_under_5mm else 0.0,
            'pos_under_10mm': np.mean(pos_under_10mm) if pos_under_10mm else 0.0,
            'cd_avg': np.mean(cd_vals) if cd_vals else 0.0,
            'cd_std': np.std(cd_vals) if cd_vals else 0.0,
            'cd_pred2ref': np.mean(cd_pred2ref) if cd_pred2ref else 0.0,
            'cd_ref2pred': np.mean(cd_ref2pred) if cd_ref2pred else 0.0,
            'prec_2mm': np.mean(prec_2mm) if prec_2mm else 0.0,
            'prec_5mm': np.mean(prec_5mm) if prec_5mm else 0.0,
            'prec_10mm': np.mean(prec_10mm) if prec_10mm else 0.0,
            'rec_2mm': np.mean(rec_2mm) if rec_2mm else 0.0,
            'rec_5mm': np.mean(rec_5mm) if rec_5mm else 0.0,
            'rec_10mm': np.mean(rec_10mm) if rec_10mm else 0.0,
            'f_2mm': np.mean(f_2mm) if f_2mm else 0.0,
            'f_5mm': np.mean(f_5mm) if f_5mm else 0.0,
            'f_10mm': np.mean(f_10mm) if f_10mm else 0.0,
        })
    
    # Save summary
    summary_txt = clip_output_dir / 'summary.txt'
    with open(summary_txt, 'w') as f:
        f.write(f"Clip {clip_idx} Summary (frames {start_frame}-{end_frame}, {n_frames} frames)\n")
        f.write("=" * 100 + "\n\n")
        f.write("Methods: Full (with interior projection), NoProj (no interior projection).\n\n")
        
        # Edge Length Metrics
        f.write("Edge Length Metrics\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Method':<12} | {'Edge % Mean':<18} | {'Edge RMSE (mm)':<16} | {'<2%':<8} | {'<5%':<8} | {'<10%':<8}\n")
        f.write("-" * 100 + "\n")
        for s in summary_rows:
            f.write(f"{s['method']:<12} | {s['edge_pct_mean_avg']:>5.2f}% ± {s['edge_pct_mean_std']:>5.2f}% | "
                    f"{s['edge_rmse_avg']:>5.2f} ±{s['edge_rmse_std']:>4.2f} mm | "
                    f"{s['edge_under_2pct']:>5.1f}% | {s['edge_under_5pct']:>5.1f}% | {s['edge_under_10pct']:>5.1f}%\n")
        
        # Position RMSE Metrics
        f.write("\nPosition RMSE Metrics\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<12} | {'Pos RMSE (mm)':<18} | {'<2mm':<8} | {'<5mm':<8} | {'<10mm':<8}\n")
        f.write("-" * 80 + "\n")
        for s in summary_rows:
            f.write(f"{s['method']:<12} | {s['pos_rmse_avg']:>5.2f} ± {s['pos_rmse_std']:>5.2f} mm   | "
                    f"{s['pos_under_2mm']:>5.1f}% | {s['pos_under_5mm']:>5.1f}% | {s['pos_under_10mm']:>5.1f}%\n")
        
        # Chamfer Distance Metrics
        f.write("\nChamfer Distance Metrics \n")
        f.write("-" * 130 + "\n")
        f.write(f"{'Method':<12} | {'CD (mm)':<15} | {'Pred→Ref':<10} | {'Ref→Pred':<10} | "
                f"{'Prec@2mm':<8} | {'Prec@5mm':<8} | {'Prec@10mm':<9} | "
                f"{'Rec@2mm':<8} | {'Rec@5mm':<8} | {'Rec@10mm':<8}\n")
        f.write("-" * 130 + "\n")
        for s in summary_rows:
            f.write(f"{s['method']:<12} | {s['cd_avg']:>5.2f} ±{s['cd_std']:>4.2f} mm | "
                    f"{s['cd_pred2ref']:>7.2f} mm | {s['cd_ref2pred']:>7.2f} mm | "
                    f"{s['prec_2mm']:>5.1f}% | {s['prec_5mm']:>5.1f}% | {s['prec_10mm']:>6.1f}% | "
                    f"{s['rec_2mm']:>5.1f}% | {s['rec_5mm']:>5.1f}% | {s['rec_10mm']:>5.1f}%\n")
        
        # F-Scores
        f.write("\nF-Scores\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Method':<12} | {'F@2mm':<12} | {'F@5mm':<12} | {'F@10mm':<12}\n")
        f.write("-" * 60 + "\n")
        for s in summary_rows:
            f.write(f"{s['method']:<12} | {s['f_2mm']:>8.2f}% | {s['f_5mm']:>8.2f}% | {s['f_10mm']:>8.2f}%\n")
    
    # Save 3D keypoints (convert reference_lengths dict to array)
    keypoints_3d_path = clip_output_dir / '3d_keypoints.npz'
    
    # Convert edge dict to arrays
    if stored_edges:
        edges_array = np.array(stored_edges, dtype=np.int32)
    else:
        edges_array = np.array([], dtype=np.int32)
    
    if stored_reference_lengths:
        ref_lengths_array = np.array([stored_reference_lengths.get(tuple(e), 0.0) 
                                       for e in edges_array], dtype=np.float32)
    else:
        ref_lengths_array = np.array([], dtype=np.float32)
    
    np.savez(
        keypoints_3d_path,
        full=np.array(keypoints_3d_histories['Full']),
        noproj=np.array(keypoints_3d_histories['NoProj']),
        edge_connection=edges_array,
        reference_lengths=ref_lengths_array,
    )
    
    print(f"    Saved: {clip_output_dir}")
    
    return {
        'clip_idx': clip_idx,
        'all_metrics': all_metrics,
        'summary_rows': summary_rows,
    }


def aggregate_chunk_summary(output_dir, all_clip_results, method_names):
    """Aggregate results across all clips in a chunk."""
    chunk_summary_dir = output_dir / 'chunk_summary'
    chunk_summary_dir.mkdir(parents=True, exist_ok=True)
    
    all_clips_metrics = {m: [] for m in method_names}
    all_clips_3d_kpts = {m: [] for m in method_names}
    
    for clip_result in all_clip_results:
        for method in method_names:
            all_clips_metrics[method].extend(clip_result['all_metrics'][method])
        
        clip_dir = output_dir / f"clip_{clip_result['clip_idx']}"
        kpts_path = clip_dir / '3d_keypoints.npz'
        if kpts_path.exists():
            kpts_data = np.load(kpts_path)
            for method in method_names:
                key = method.lower()
                if key in kpts_data:
                    all_clips_3d_kpts[method].append(kpts_data[key])
    
    np.savez(
        chunk_summary_dir / 'all_clips_3d_keypoints.npz',
        **{m.lower(): np.concatenate(all_clips_3d_kpts[m], axis=0) if all_clips_3d_kpts[m] else np.array([])
           for m in method_names},
        reference_lengths_per_clip=np.array([len(r['all_metrics']['Full']) for r in all_clip_results]),
    )
    
    csv_path = chunk_summary_dir / 'all_clips_metrics.csv'
    with open(csv_path, 'w') as f:
        f.write('Clip,Frame,GlobalFrame,Method,EdgePctMean,EdgePctStd,EdgePctMax,EdgeRMSE,PosRMSE,'
                'CD,Pred2Ref,Ref2Pred,F@2mm,F@5mm,F@10mm\n')
        for clip_result in all_clip_results:
            clip_idx = clip_result['clip_idx']
            for method in method_names:
                for m in clip_result['all_metrics'][method]:
                    f.write(f"{clip_idx},{m['frame']},{m['global_frame']},{method},"
                            f"{m['edge_pct_mean']:.4f},{m['edge_pct_std']:.4f},{m['edge_pct_max']:.4f},"
                            f"{m['edge_rmse_mm']:.4f},{m['pos_rmse_mm']:.4f},"
                            f"{m['cd']:.4f},{m['cd_pred2ref']:.4f},{m['cd_ref2pred']:.4f},"
                            f"{m['f_2mm']:.4f},{m['f_5mm']:.4f},{m['f_10mm']:.4f}\n")
    
    summary_path = chunk_summary_dir / 'chunk_aggregate_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Fabric Tracking Projection Ablation - Chunk Aggregate Summary\n")
        f.write("=" * 100 + "\n\n")
        
        for method in method_names:
            metrics = all_clips_metrics[method]
            if not metrics:
                continue
            
            valid_metrics = [m for m in metrics if m['success']]
            
            edge_pct = [m['edge_pct_mean'] for m in valid_metrics]
            edge_rmse = [m['edge_rmse_mm'] for m in valid_metrics]
            pos_rmse = [m['pos_rmse_mm'] for m in valid_metrics]
            cd = [m['cd'] for m in valid_metrics]
            f10 = [m['f_10mm'] for m in valid_metrics]
            
            f.write(f"Method: {method}\n")
            f.write(f"  Total frames: {len(valid_metrics)}\n")
            f.write(f"  Edge % Mean: {np.mean(edge_pct):.2f}% +/- {np.std(edge_pct):.2f}%\n")
            f.write(f"  Edge RMSE: {np.mean(edge_rmse):.2f} +/- {np.std(edge_rmse):.2f} mm\n")
            f.write(f"  Pos RMSE: {np.mean(pos_rmse):.2f} +/- {np.std(pos_rmse):.2f} mm\n")
            f.write(f"  CD: {np.mean(cd):.2f} +/- {np.std(cd):.2f} mm\n")
            f.write(f"  F@10mm: {np.mean(f10):.1f}%\n\n")
    
    print(f"  Chunk summary saved to: {chunk_summary_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fabric Tracking Projection Ablation')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASETS.keys()),
                        help='Dataset name')
    parser.add_argument('--chunk', type=int, required=True, help='Chunk index')
    parser.add_argument('--clip_seconds', type=int, default=10, help='Clip duration (default: 10)')
    parser.add_argument('--fps', type=int, default=FPS, help='Frame rate (default: 30)')
    parser.add_argument('--grid_rows', type=int, default=GRID_ROWS, help='Grid rows')
    parser.add_argument('--grid_cols', type=int, default=GRID_COLS, help='Grid cols')
    args = parser.parse_args()
    
    chunk_dir = DATA_BASE / args.dataset / f'chunk_{args.chunk}'
    output_dir = OUTPUT_BASE / args.dataset / f'chunk_{args.chunk}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"FABRIC TRACKING PROJECTION ABLATION - {args.dataset} Chunk {args.chunk}")
    print("=" * 80)
    
    print(f"\nLoading chunk_{args.chunk} data...")
    data = load_chunk_data(chunk_dir)
    transforms = load_transforms(CALIB_DIR)
    
    print(f"  Color: {data['color'].shape}")
    print(f"  Depth: {data['depth'].shape}")
    print(f"  FG masks: {data['fg_masks'].shape if data['fg_masks'] is not None else 'None'}")
    print(f"  Total frames: {data['n_frames']}")
    
    if data['fg_masks'] is None:
        print("ERROR: No foreground masks found!")
        return
    
    print("\nComputing EE positions...")
    ee_poses_3d = np.zeros((data['n_frames'], 2, 3))
    for i in range(data['n_frames']):
        ee_poses_3d[i] = get_ee_positions_cam(
            data['left_poses'][i], data['right_poses'][i],
            transforms['T_left_base2cam'], transforms['T_right_base2cam'],
        )
    
    frames_per_clip = args.clip_seconds * args.fps
    n_clips = (data['n_frames'] + frames_per_clip - 1) // frames_per_clip
    
    print(f"\nClip configuration:")
    print(f"  Clip duration: {args.clip_seconds}s ({frames_per_clip} frames)")
    print(f"  Number of clips: {n_clips}")
    print(f"  Grid: {args.grid_rows}x{args.grid_cols} = {args.grid_rows * args.grid_cols} keypoints")
    
    method_names = ['Full', 'NoProj']
    all_clip_results = []
    
    for clip_idx in range(n_clips):
        start_frame = clip_idx * frames_per_clip
        end_frame = min(start_frame + frames_per_clip, data['n_frames'])
        
        clip_result = process_clip(
            data=data,
            transforms=transforms,
            ee_poses_3d=ee_poses_3d,
            start_frame=start_frame,
            end_frame=end_frame,
            clip_idx=clip_idx,
            output_dir=output_dir,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            fps=args.fps,
        )
        all_clip_results.append(clip_result)
    
    print("\nAggregating chunk summary...")
    aggregate_chunk_summary(output_dir, all_clip_results, method_names)
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
