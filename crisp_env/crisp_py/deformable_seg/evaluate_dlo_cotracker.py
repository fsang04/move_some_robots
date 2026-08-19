#!/usr/bin/env python3
"""
Evaluate CoTracker results on DLO dataset.

Uses the same metrics as dlo1_batch_experiment.py:
- Edge Length Metrics (Edge %, RMSE, <2%, <5%, <10%)
- Position RMSE (distance to skeleton point cloud)
- Chamfer Distance (CD, Precision, Recall, F-score at 2/5/10mm)

Handles invalid depth at CoTracker tracking points.

Usage:
    # Evaluate single chunk
    python evaluate_dlo_cotracker.py --chunk 0
    python evaluate_dlo_cotracker.py --chunk 5 --mode online
    
    # Aggregate all results
    python evaluate_dlo_cotracker.py --all --mode offline
"""

import argparse
import numpy as np
import cv2
import csv
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


# ============================================================================
# PATHS
# ============================================================================

# Raw data base path
DATA_BASE = Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/'
                 'roahm-deformable-objects/captured_data_double_arm/dlo1_first400')

# CoTracker results base path
COTRACKER_RESULTS = Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/deformable_seg/dlo1_cotracker_results')

# DLO evaluation results (for ground truth edge topology)
DLO_EVAL_BASE = Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/deformable_seg/dlo1_evaluation_results')

# Output base path
OUTPUT_BASE = Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/deformable_seg/dlo_cotracker_evaluation')

# Calibration path
CALIB_DIR = Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/'
                 'roahm-deformable-objects/captured_calibration_data/dlo1_cloth1_calibration')

# Default FPS
FPS = 30


# ============================================================================
# DATA LOADING
# ============================================================================

def load_calibration(calib_dir: Path) -> dict:
    """Load camera intrinsics and transforms from calibration directory."""
    tf = np.load(calib_dir / 'transform_ee_cam_world.npz')
    return {
        'K': tf['K'],
        'T_left_base2cam': tf['T_left_base2cam'],
        'T_right_base2cam': tf['T_right_base2cam'],
    }


def pose7_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert [x,y,z,qw,qx,qy,qz] to 4x4 matrix."""
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    quat = pose[3:]
    T[:3, :3] = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    return T


def get_ee_positions_cam(left_pose, right_pose, T_left_base2cam, T_right_base2cam):
    """Compute EE positions in camera frame from base poses."""
    T_left_ee_base = pose7_to_matrix(left_pose)
    T_right_ee_base = pose7_to_matrix(right_pose)
    
    T_left_ee_cam = T_left_base2cam @ T_left_ee_base
    T_right_ee_cam = T_right_base2cam @ T_right_ee_base
    
    left_pos_cam = T_left_ee_cam[:3, 3]
    right_pos_cam = T_right_ee_cam[:3, 3]
    
    # Return in mm (poses are in meters)
    return np.array([left_pos_cam * 1000, right_pos_cam * 1000])


def load_chunk_data(chunk_dir: Path) -> dict:
    """Load depth, dlo_masks, and EE poses from chunk directory.
    
    Uses masks/masks.npz (DLO-specific mask) like dlo1_batch_experiment.py.
    """
    print(f"Loading data from {chunk_dir}...")
    
    rgbd = np.load(chunk_dir / 'rgbd.npz')
    depth = rgbd['depth']
    
    # Load DLO masks (same as dlo1_batch_experiment.py)
    masks_path = chunk_dir / 'masks' / 'masks.npz'
    if masks_path.exists():
        dlo_masks = np.load(masks_path)['masks']
    else:
        dlo_masks = None
        print(f"  Warning: masks/masks.npz not found")
    
    # Load EE poses
    left_poses_npz = np.load(chunk_dir / 'left_arm_poses.npz')
    right_poses_npz = np.load(chunk_dir / 'right_arm_poses.npz')
    
    n_poses = len(left_poses_npz.files)
    left_poses = np.array([left_poses_npz[f'arr_{i}'] for i in range(n_poses)])
    right_poses = np.array([right_poses_npz[f'arr_{i}'] for i in range(n_poses)])
    
    print(f"  Loaded {len(depth)} frames")
    
    return {
        'depth': depth,
        'dlo_masks': dlo_masks,
        'left_poses': left_poses,
        'right_poses': right_poses,
        'n_frames': len(depth),
    }


def load_cotracker_results(cotracker_dir: Path, clip_idx: int, mode: str = 'offline') -> dict:
    """Load CoTracker tracking results."""
    clip_dir = cotracker_dir / f'clip_{clip_idx}'
    npz_path = clip_dir / f'keypoints_cotracker_{mode}.npz'
    
    if not npz_path.exists():
        raise FileNotFoundError(f"CoTracker results not found: {npz_path}")
    
    data = np.load(npz_path)
    return {
        'keypoints_2d': data['keypoints_2d'],  # (T, N, 2)
        'edge_connection': data['edge_connection'],  # (E, 2)
        'initial_keypoints_3d': data['initial_keypoints_3d'],  # (N, 3)
    }


def load_dlo_eval_reference(dlo_eval_dir: Path, clip_idx: int) -> dict:
    """Load reference edge lengths from DLO evaluation results."""
    clip_dir = dlo_eval_dir / f'clip_{clip_idx}'
    npz_path = clip_dir / '3d_keypoints.npz'
    
    if not npz_path.exists():
        raise FileNotFoundError(f"DLO evaluation reference not found: {npz_path}")
    
    data = np.load(npz_path)
    return {
        'edge_connection': data['edge_connection'],  # (E, 2)
        'reference_lengths': data['reference_lengths'],  # (E,)
    }


# ============================================================================
# 2D TO 3D LIFTING (with invalid depth handling)
# ============================================================================

def lift_2d_to_3d_with_depth(keypoints_2d: np.ndarray, depth: np.ndarray, K: np.ndarray,
                              search_radius: int = 5) -> tuple:
    """Lift 2D keypoints to 3D using depth image.
    
    Args:
        keypoints_2d: N × 2 array of (u, v) pixel coordinates
        depth: H × W depth image in mm
        K: 3 × 3 camera intrinsic matrix
        search_radius: Radius to search for valid depth if center is invalid
        
    Returns:
        keypoints_3d: N × 3 array of (x, y, z) in mm (NaN if invalid)
        valid_mask: N boolean array indicating valid depth
    """
    N = len(keypoints_2d)
    keypoints_3d = np.full((N, 3), np.nan, dtype=np.float32)
    valid_mask = np.zeros(N, dtype=bool)
    
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    for i, (u, v) in enumerate(keypoints_2d):
        u_int, v_int = int(round(u)), int(round(v))
        
        if u_int < 0 or u_int >= W or v_int < 0 or v_int >= H:
            continue
        
        # Try to get valid depth
        z = depth[v_int, u_int]
        
        # Check if depth is valid (foreground only: >0 and <2000mm)
        if z <= 0 or z > 2000:  # Invalid depth (foreground only)
            # Search neighborhood for valid depth
            y_min = max(0, v_int - search_radius)
            y_max = min(H, v_int + search_radius + 1)
            x_min = max(0, u_int - search_radius)
            x_max = min(W, u_int + search_radius + 1)
            
            neighborhood = depth[y_min:y_max, x_min:x_max]
            valid_depths = neighborhood[(neighborhood > 0) & (neighborhood < 2000)]
            
            if len(valid_depths) > 0:
                z = np.median(valid_depths)  # Use median for robustness
            else:
                continue  # No valid depth in neighborhood
        
        # Backproject to 3D
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        keypoints_3d[i] = [x, y, z]
        valid_mask[i] = True
    
    return keypoints_3d, valid_mask


# ============================================================================
# METRICS (same as dlo1_batch_experiment.py)
# ============================================================================

def compute_edge_metrics(keypoints_3d: np.ndarray, edges: np.ndarray,
                         reference_lengths: np.ndarray) -> dict:
    """Compute edge length metrics comparing predicted vs reference lengths."""
    if keypoints_3d is None or len(keypoints_3d) == 0:
        return {'pct_mean': np.nan, 'pct_std': np.nan, 'pct_max': np.nan,
                'rmse_mm': np.nan, 'under_2pct': 0.0, 'under_5pct': 0.0, 'under_10pct': 0.0}
    
    pct_errors = []
    abs_errors = []
    
    for edge_idx, (i, j) in enumerate(edges):
        if i >= len(keypoints_3d) or j >= len(keypoints_3d):
            continue
            
        pt_i, pt_j = keypoints_3d[i], keypoints_3d[j]
        if np.any(np.isnan(pt_i)) or np.any(np.isnan(pt_j)):
            continue
            
        pred_len = np.linalg.norm(pt_j - pt_i)
        ref_len = reference_lengths[edge_idx]
        
        if ref_len > 0:
            pct_err = 100.0 * abs(pred_len - ref_len) / ref_len
            pct_errors.append(pct_err)
            abs_errors.append(abs(pred_len - ref_len))
    
    if len(pct_errors) == 0:
        return {'pct_mean': np.nan, 'pct_std': np.nan, 'pct_max': np.nan,
                'rmse_mm': np.nan, 'under_2pct': 0.0, 'under_5pct': 0.0, 'under_10pct': 0.0}
    
    pct_errors = np.array(pct_errors)
    abs_errors = np.array(abs_errors)
    
    return {
        'pct_mean': np.mean(pct_errors),
        'pct_std': np.std(pct_errors),
        'pct_max': np.max(pct_errors),
        'rmse_mm': np.sqrt(np.mean(abs_errors ** 2)),
        'under_2pct': 100.0 * np.mean(pct_errors < 2.0),
        'under_5pct': 100.0 * np.mean(pct_errors < 5.0),
        'under_10pct': 100.0 * np.mean(pct_errors < 10.0),
    }


def compute_position_metrics(keypoints_3d: np.ndarray, ref_pc: np.ndarray) -> dict:
    """Compute position RMSE: distance from predicted keypoints to nearest ref point."""
    if keypoints_3d is None or len(keypoints_3d) == 0:
        return {'rmse_mm': np.nan, 'under_2mm': 0.0, 'under_5mm': 0.0, 'under_10mm': 0.0}
    if ref_pc is None or len(ref_pc) == 0:
        return {'rmse_mm': np.nan, 'under_2mm': 0.0, 'under_5mm': 0.0, 'under_10mm': 0.0}
    
    # Filter out NaN keypoints
    valid_kp = keypoints_3d[~np.any(np.isnan(keypoints_3d), axis=1)]
    if len(valid_kp) == 0:
        return {'rmse_mm': np.nan, 'under_2mm': 0.0, 'under_5mm': 0.0, 'under_10mm': 0.0}
    
    nn = NearestNeighbors(n_neighbors=1).fit(ref_pc)
    dists, _ = nn.kneighbors(valid_kp)
    dists = dists.flatten()
    
    return {
        'rmse_mm': np.sqrt(np.mean(dists ** 2)),
        'under_2mm': 100.0 * np.mean(dists < 2.0),
        'under_5mm': 100.0 * np.mean(dists < 5.0),
        'under_10mm': 100.0 * np.mean(dists < 10.0),
    }


def infer_endpoint_indices(edges: np.ndarray, n_keypoints: int) -> tuple:
    """Infer the two endpoint keypoint indices from chain graph connectivity."""
    if edges is None or len(edges) == 0 or n_keypoints <= 0:
        return None, None

    degree = np.zeros(n_keypoints, dtype=np.int32)
    for i, j in edges:
        if 0 <= i < n_keypoints:
            degree[i] += 1
        if 0 <= j < n_keypoints:
            degree[j] += 1

    endpoints = np.where(degree == 1)[0]
    if len(endpoints) >= 2:
        return int(endpoints[0]), int(endpoints[1])

    valid_nodes = np.where(degree > 0)[0]
    if len(valid_nodes) >= 2:
        return int(valid_nodes[0]), int(valid_nodes[-1])

    if n_keypoints >= 2:
        return 0, n_keypoints - 1
    return None, None


def assign_ee_endpoint_order(endpoint_pair: tuple, initial_keypoints_3d: np.ndarray,
                             ee_positions: np.ndarray) -> tuple:
    """Assign (left,right) EE to endpoint indices once per clip using nearest pairing."""
    left_idx, right_idx = endpoint_pair
    if left_idx is None or right_idx is None:
        return endpoint_pair

    if initial_keypoints_3d is None or len(initial_keypoints_3d) <= max(left_idx, right_idx):
        return endpoint_pair
    if ee_positions is None or len(ee_positions) < 2:
        return endpoint_pair

    p_left = initial_keypoints_3d[left_idx]
    p_right = initial_keypoints_3d[right_idx]
    ee_l = ee_positions[0]
    ee_r = ee_positions[1]

    if np.any(np.isnan(p_left)) or np.any(np.isnan(p_right)):
        return endpoint_pair

    cost_keep = np.linalg.norm(p_left - ee_l) + np.linalg.norm(p_right - ee_r)
    cost_swap = np.linalg.norm(p_left - ee_r) + np.linalg.norm(p_right - ee_l)

    if cost_swap < cost_keep:
        return right_idx, left_idx
    return endpoint_pair


def inject_ee_into_keypoints(keypoints_3d: np.ndarray, ee_positions: np.ndarray,
                             endpoint_order: tuple) -> np.ndarray:
    """Replace endpoint keypoints with EE positions (left->idx0, right->idx1)."""
    if keypoints_3d is None:
        return keypoints_3d
    if ee_positions is None or len(ee_positions) < 2:
        return keypoints_3d

    left_idx, right_idx = endpoint_order
    if left_idx is None or right_idx is None:
        return keypoints_3d
    if max(left_idx, right_idx) >= len(keypoints_3d):
        return keypoints_3d

    kp = keypoints_3d.copy()
    kp[left_idx] = ee_positions[0]
    kp[right_idx] = ee_positions[1]
    return kp


def sample_points_on_edges(keypoints_3d: np.ndarray, edges: np.ndarray,
                           n_target_points: int,
                           allocation_lengths: np.ndarray = None) -> np.ndarray:
    """Uniformly sample points along edges for Chamfer Distance.

    If allocation_lengths is provided, sample-count allocation per edge uses those
    fixed lengths (e.g., reference lengths) instead of per-frame predicted lengths.
    """
    if keypoints_3d is None or len(keypoints_3d) == 0:
        return np.empty((0, 3), dtype=np.float32)
    
    # Calculate allocation lengths (fixed reference lengths if provided)
    edge_lengths = []
    if allocation_lengths is not None and len(allocation_lengths) == len(edges):
        for edge_idx, (i, j) in enumerate(edges):
            if i >= len(keypoints_3d) or j >= len(keypoints_3d):
                edge_lengths.append(0.0)
                continue
            pt_i, pt_j = keypoints_3d[i], keypoints_3d[j]
            if np.any(np.isnan(pt_i)) or np.any(np.isnan(pt_j)):
                edge_lengths.append(0.0)
                continue
            edge_lengths.append(float(max(0.0, allocation_lengths[edge_idx])))
    else:
        for i, j in edges:
            if i >= len(keypoints_3d) or j >= len(keypoints_3d):
                edge_lengths.append(0.0)
                continue
            pt_i, pt_j = keypoints_3d[i], keypoints_3d[j]
            if np.any(np.isnan(pt_i)) or np.any(np.isnan(pt_j)):
                edge_lengths.append(0.0)
                continue
            edge_lengths.append(np.linalg.norm(pt_j - pt_i))

    total_length = float(np.sum(edge_lengths))
    
    if total_length <= 0:
        return np.empty((0, 3), dtype=np.float32)
    
    # Sample points proportionally along edges
    sampled_points = []
    for edge_idx, (i, j) in enumerate(edges):
        if edge_lengths[edge_idx] <= 0:
            continue
        n_samples = max(1, int(n_target_points * edge_lengths[edge_idx] / total_length))
        pt_i, pt_j = keypoints_3d[i], keypoints_3d[j]
        for t in np.linspace(0, 1, n_samples):
            sampled_points.append(pt_i * (1 - t) + pt_j * t)
    
    return np.array(sampled_points, dtype=np.float32) if sampled_points else np.empty((0, 3), dtype=np.float32)


def compute_chamfer_metrics(pred_cloud: np.ndarray, ref_cloud: np.ndarray) -> dict:
    """Compute Chamfer Distance and F-scores between predicted and reference point clouds."""
    empty_result = {'cd': np.nan, 'pred2ref_avg': np.nan, 'ref2pred_avg': np.nan,
                    'prec_10': 0.0, 'rec_10': 0.0,
                    'f2': 0.0, 'f5': 0.0, 'f10': 0.0, 'n_pred': 0, 'n_ref': 0}
    
    if pred_cloud is None or len(pred_cloud) == 0:
        empty_result['n_ref'] = len(ref_cloud) if ref_cloud is not None else 0
        return empty_result
    if ref_cloud is None or len(ref_cloud) == 0:
        return empty_result
    
    # Filter out NaN points
    valid_pred = pred_cloud[~np.any(np.isnan(pred_cloud), axis=1)]
    if len(valid_pred) == 0:
        empty_result['n_ref'] = len(ref_cloud)
        return empty_result
    
    # Pred -> Ref distances
    nn_ref = NearestNeighbors(n_neighbors=1).fit(ref_cloud)
    dists_pred2ref, _ = nn_ref.kneighbors(valid_pred)
    dists_pred2ref = dists_pred2ref.flatten()
    
    # Ref -> Pred distances
    nn_pred = NearestNeighbors(n_neighbors=1).fit(valid_pred)
    dists_ref2pred, _ = nn_pred.kneighbors(ref_cloud)
    dists_ref2pred = dists_ref2pred.flatten()
    
    # Chamfer Distance (symmetric average)
    cd = 0.5 * (np.mean(dists_pred2ref) + np.mean(dists_ref2pred))
    
    # Precision and Recall at various thresholds
    prec_10 = 100.0 * np.mean(dists_pred2ref < 10.0)  # % pred points near ref
    rec_10 = 100.0 * np.mean(dists_ref2pred < 10.0)   # % ref points near pred (coverage)
    
    # F-scores at various thresholds
    def f_score(threshold):
        precision = 100.0 * np.mean(dists_pred2ref < threshold)
        recall = 100.0 * np.mean(dists_ref2pred < threshold)
        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0
    
    return {
        'cd': cd,
        'pred2ref_avg': np.mean(dists_pred2ref),
        'ref2pred_avg': np.mean(dists_ref2pred),
        'prec_10': prec_10,
        'rec_10': rec_10,
        'f2': f_score(2.0),
        'f5': f_score(5.0),
        'f10': f_score(10.0),
        'n_pred': len(valid_pred),
        'n_ref': len(ref_cloud),
    }


def extract_skeleton_pc(dlo_mask, depth, K, ee_positions=None, dilate_pixels=1):
    """Extract 3D point cloud from skeletonized DLO mask.
    
    Same approach as dlo1_batch_experiment.py:
    1. Skeletonize DLO mask to get thin centerline
    2. Dilate slightly for robustness
    3. Extract 3D points from skeleton pixels
    4. Augment with EE positions
    
    Args:
        dlo_mask: H × W binary mask of DLO
        depth: H × W depth image in mm
        K: 3 × 3 camera intrinsic matrix
        ee_positions: optional 2 × 3 array of EE positions to append
        dilate_pixels: number of pixels to dilate skeleton (default: 1)
        
    Returns:
        N × 3 point cloud in mm (includes EE positions if provided)
    """
    if dlo_mask is None or depth is None:
        if ee_positions is not None and len(ee_positions) > 0:
            return np.array(ee_positions, dtype=np.float32)
        return np.empty((0, 3), dtype=np.float32)
    
    # Skeletonize the mask
    skeleton = skeletonize(dlo_mask > 0)
    
    # Dilate skeleton slightly for robustness
    if dilate_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_pixels+1, 2*dilate_pixels+1))
        skeleton = cv2.dilate(skeleton.astype(np.uint8), kernel, iterations=1)
    
    # Get pixel coordinates where skeleton is nonzero
    rows, cols = np.where(skeleton > 0)
    if len(rows) == 0:
        if ee_positions is not None and len(ee_positions) > 0:
            return np.array(ee_positions, dtype=np.float32)
        return np.empty((0, 3), dtype=np.float32)
    
    # Get depth values
    z_vals = depth[rows, cols].astype(np.float32)
    valid = (z_vals > 0) & (z_vals < 2000)
    rows, cols, z_vals = rows[valid], cols[valid], z_vals[valid]
    
    if len(z_vals) == 0:
        if ee_positions is not None and len(ee_positions) > 0:
            return np.array(ee_positions, dtype=np.float32)
        return np.empty((0, 3), dtype=np.float32)
    
    # Backproject to 3D
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x_vals = (cols - cx) * z_vals / fx
    y_vals = (rows - cy) * z_vals / fy
    
    pc = np.column_stack([x_vals, y_vals, z_vals]).astype(np.float32)
    
    # Append EE positions if provided
    if ee_positions is not None and len(ee_positions) > 0:
        ee_arr = np.array(ee_positions, dtype=np.float32).reshape(-1, 3)
        pc = np.vstack([pc, ee_arr])
    
    return pc


# ============================================================================
# CLIP PROCESSING
# ============================================================================

def process_clip(data: dict, cotracker_results: dict, dlo_ref: dict, K: np.ndarray,
                 T_left_base2cam: np.ndarray, T_right_base2cam: np.ndarray,
                 clip_idx: int, start_frame: int, end_frame: int, output_dir: Path,
                 mode: str = 'offline') -> dict:
    """Process a single clip and compute metrics."""
    
    keypoints_2d = cotracker_results['keypoints_2d']  # (T, N, 2)
    edge_connection = cotracker_results['edge_connection']
    reference_lengths = dlo_ref['reference_lengths']
    
    depth = data['depth']
    dlo_masks = data['dlo_masks']
    left_poses = data['left_poses']
    right_poses = data['right_poses']

    endpoint_pair = infer_endpoint_indices(edge_connection, keypoints_2d.shape[1])
    ee_pos_0 = get_ee_positions_cam(
        left_poses[start_frame], right_poses[start_frame],
        T_left_base2cam, T_right_base2cam
    )
    endpoint_order = assign_ee_endpoint_order(
        endpoint_pair,
        cotracker_results.get('initial_keypoints_3d', None),
        ee_pos_0,
    )
    
    n_frames = min(end_frame - start_frame, len(keypoints_2d))
    n_keypoints = keypoints_2d.shape[1]
    
    print(f"\n  Processing clip {clip_idx} (frames {start_frame}-{end_frame}, {n_frames} frames, {n_keypoints} keypoints)")
    
    # Process each frame
    all_keypoints_3d = []
    all_edge_metrics = []
    all_pos_metrics = []
    all_cd_metrics = []
    invalid_depth_counts = []
    
    for local_idx in tqdm(range(n_frames), desc=f"    Clip {clip_idx}", leave=False):
        frame_idx = start_frame + local_idx
        kp_2d = keypoints_2d[local_idx]
        
        kp_3d, valid_mask = lift_2d_to_3d_with_depth(kp_2d, depth[frame_idx], K, search_radius=5)
        
        # Compute EE positions for this frame
        ee_pos = get_ee_positions_cam(
            left_poses[frame_idx], right_poses[frame_idx],
            T_left_base2cam, T_right_base2cam
        )

        # Inject EE anchors by replacing corresponding endpoint keypoints
        kp_3d = inject_ee_into_keypoints(kp_3d, ee_pos, endpoint_order)
        left_idx, right_idx = endpoint_order
        if left_idx is not None and left_idx < len(valid_mask):
            valid_mask[left_idx] = True
        if right_idx is not None and right_idx < len(valid_mask):
            valid_mask[right_idx] = True

        all_keypoints_3d.append(kp_3d)

        n_invalid = np.sum(~valid_mask)
        invalid_depth_counts.append(n_invalid)
        
        # Extract reference skeleton PC (from dlo_masks with skeletonization, augmented with EE)
        if dlo_masks is not None:
            ref_pc = extract_skeleton_pc(dlo_masks[frame_idx], depth[frame_idx], K, 
                                         ee_positions=ee_pos, dilate_pixels=1)
        else:
            ref_pc = np.array(ee_pos, dtype=np.float32).reshape(-1, 3) if ee_pos is not None else np.empty((0, 3))
        
        # Edge metrics
        edge_m = compute_edge_metrics(kp_3d, edge_connection, reference_lengths)
        all_edge_metrics.append(edge_m)
        
        # Position metrics
        pos_m = compute_position_metrics(kp_3d, ref_pc)
        all_pos_metrics.append(pos_m)
        
        # Chamfer Distance metrics
        n_ref_points = len(ref_pc) if ref_pc is not None and len(ref_pc) > 0 else 100
        pred_cloud = sample_points_on_edges(
            kp_3d,
            edge_connection,
            n_ref_points,
            allocation_lengths=reference_lengths,
        )
        cd_m = compute_chamfer_metrics(pred_cloud, ref_pc)
        all_cd_metrics.append(cd_m)
    
    # Create output directory
    clip_dir = output_dir / f'clip_{clip_idx}'
    clip_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per_frame.csv
    csv_path = clip_dir / 'per_frame.csv'
    with open(csv_path, 'w', newline='') as f:
        f.write("Frame,Method,EdgePctMean,EdgePctStd,EdgeRMSE,PosRMSE,CD,")
        f.write("Pred2Ref,Ref2Pred,Prec@10mm,Rec@10mm,F@2mm,F@5mm,F@10mm,NPred,NRef,InvalidDepth\n")
        
        for i in range(n_frames):
            em = all_edge_metrics[i]
            pm = all_pos_metrics[i]
            cm = all_cd_metrics[i]
            
            f.write(f"{i},CoTracker_{mode},")
            f.write(f"{em['pct_mean']:.2f},{em['pct_std']:.2f},{em['rmse_mm']:.2f},")
            f.write(f"{pm['rmse_mm']:.2f},{cm['cd']:.2f},")
            f.write(f"{cm['pred2ref_avg']:.2f},{cm['ref2pred_avg']:.2f},")
            f.write(f"{cm['prec_10']:.1f},{cm['rec_10']:.1f},")
            f.write(f"{cm['f2']:.1f},{cm['f5']:.1f},{cm['f10']:.1f},")
            f.write(f"{cm['n_pred']},{cm['n_ref']},{invalid_depth_counts[i]}\n")
    
    # Compute summary statistics
    valid_edge = [m for m in all_edge_metrics if not np.isnan(m['pct_mean'])]
    valid_pos = [m for m in all_pos_metrics if not np.isnan(m['rmse_mm'])]
    valid_cd = [m for m in all_cd_metrics if not np.isnan(m['cd'])]
    
    edge_pct_mean = np.mean([m['pct_mean'] for m in valid_edge]) if valid_edge else np.nan
    edge_pct_std = np.mean([m['pct_std'] for m in valid_edge]) if valid_edge else np.nan
    edge_rmse_mean = np.mean([m['rmse_mm'] for m in valid_edge]) if valid_edge else np.nan
    edge_rmse_std = np.std([m['rmse_mm'] for m in valid_edge]) if valid_edge else np.nan
    edge_under_2 = np.mean([m['under_2pct'] for m in valid_edge]) if valid_edge else 0.0
    edge_under_5 = np.mean([m['under_5pct'] for m in valid_edge]) if valid_edge else 0.0
    edge_under_10 = np.mean([m['under_10pct'] for m in valid_edge]) if valid_edge else 0.0
    
    pos_rmse_mean = np.mean([m['rmse_mm'] for m in valid_pos]) if valid_pos else np.nan
    pos_rmse_std = np.std([m['rmse_mm'] for m in valid_pos]) if valid_pos else np.nan
    pos_under_2 = np.mean([m['under_2mm'] for m in valid_pos]) if valid_pos else 0.0
    pos_under_5 = np.mean([m['under_5mm'] for m in valid_pos]) if valid_pos else 0.0
    pos_under_10 = np.mean([m['under_10mm'] for m in valid_pos]) if valid_pos else 0.0
    
    cd_mean = np.mean([m['cd'] for m in valid_cd]) if valid_cd else np.nan
    cd_std = np.std([m['cd'] for m in valid_cd]) if valid_cd else np.nan
    pred2ref_mean = np.mean([m['pred2ref_avg'] for m in valid_cd]) if valid_cd else np.nan
    pred2ref_std = np.std([m['pred2ref_avg'] for m in valid_cd]) if valid_cd else np.nan
    ref2pred_mean = np.mean([m['ref2pred_avg'] for m in valid_cd]) if valid_cd else np.nan
    ref2pred_std = np.std([m['ref2pred_avg'] for m in valid_cd]) if valid_cd else np.nan
    prec10_mean = np.mean([m['prec_10'] for m in valid_cd]) if valid_cd else 0.0
    prec10_std = np.std([m['prec_10'] for m in valid_cd]) if valid_cd else 0.0
    rec10_mean = np.mean([m['rec_10'] for m in valid_cd]) if valid_cd else 0.0
    rec10_std = np.std([m['rec_10'] for m in valid_cd]) if valid_cd else 0.0
    f2_mean = np.mean([m['f2'] for m in valid_cd]) if valid_cd else 0.0
    f5_mean = np.mean([m['f5'] for m in valid_cd]) if valid_cd else 0.0
    f10_mean = np.mean([m['f10'] for m in valid_cd]) if valid_cd else 0.0
    
    avg_invalid = np.mean(invalid_depth_counts)
    
    # Save summary.txt
    summary_txt = clip_dir / 'summary.txt'
    with open(summary_txt, 'w') as f:
        f.write(f"Clip {clip_idx} Summary - CoTracker ({mode}) (frames {start_frame}-{end_frame}, {n_frames} frames)\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("Edge Length Metrics\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Method':<20} | {'Edge % Mean':<18} | {'Edge RMSE (mm)':<16} | {'<2%':<8} | {'<5%':<8} | {'<10%':<8}\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'CoTracker_' + mode:<20} | {edge_pct_mean:5.2f}% ± {edge_pct_std:5.2f}% | {edge_rmse_mean:5.2f} ±{edge_rmse_std:5.2f} mm | {edge_under_2:5.1f}% | {edge_under_5:5.1f}% | {edge_under_10:5.1f}%\n")
        f.write("\n")
        
        f.write("Position RMSE Metrics\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<20} | {'Pos RMSE (mm)':<18} | {'<2mm':<8} | {'<5mm':<8} | {'<10mm':<8}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'CoTracker_' + mode:<20} | {pos_rmse_mean:5.2f} ± {pos_rmse_std:5.2f} mm | {pos_under_2:5.1f}% | {pos_under_5:5.1f}% | {pos_under_10:5.1f}%\n")
        f.write("\n")
        
        f.write("Chamfer Distance Metrics\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Method':<20} | {'CD (mm)':<14} | {'P2R (mm)':<12} | {'R2P (mm)':<12} | {'Prec@10':<10} | {'Rec@10':<10} | {'F@10mm':<10}\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'CoTracker_' + mode:<20} | {cd_mean:5.2f} ±{cd_std:5.2f} mm | {pred2ref_mean:5.2f} ±{pred2ref_std:5.2f} | {ref2pred_mean:5.2f} ±{ref2pred_std:5.2f} | {prec10_mean:8.2f}% | {rec10_mean:8.2f}% | {f10_mean:8.2f}%\n")
        f.write(f"{'':<20} | {'':<14} | {'':<12} | {'':<12} | {'(pred→ref)':<10} | {'(ref→pred)':<10} | {'':<10}\n")
        f.write("\n")
        
        f.write(f"Average invalid depth keypoints per frame: {avg_invalid:.1f} / {n_keypoints}\n")
        f.write(f"Total keypoints: {n_keypoints}\n")
    
    # Save 3d_keypoints.npz
    np.savez(
        clip_dir / '3d_keypoints.npz',
        cotracker=np.array(all_keypoints_3d),  # (T, N, 3)
        edge_connection=edge_connection,
        reference_lengths=reference_lengths,
    )
    
    print(f"    Saved outputs to: {clip_dir}")
    print(f"    Edge RMSE: {edge_rmse_mean:.2f} ± {edge_rmse_std:.2f} mm")
    print(f"    CD: {cd_mean:.2f} ± {cd_std:.2f} mm, P2R: {pred2ref_mean:.2f} mm, R2P: {ref2pred_mean:.2f} mm")
    print(f"    Prec@10: {prec10_mean:.1f}%, Rec@10: {rec10_mean:.1f}%, F@10mm: {f10_mean:.1f}%")
    print(f"    Avg invalid depth: {avg_invalid:.1f} / {n_keypoints} keypoints")
    
    return {
        'clip_idx': clip_idx,
        'n_frames': n_frames,
        'edge_rmse': edge_rmse_mean,
        'edge_pct': edge_pct_mean,
        'cd': cd_mean,
        'f10': f10_mean,
        'avg_invalid_depth': avg_invalid,
    }


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate_all_results(mode: str = 'offline'):
    """Aggregate results from all evaluated chunks."""
    print("=" * 100)
    print(f"AGGREGATING ALL DLO COTRACKER RESULTS - Mode: {mode}")
    print("=" * 100)
    
    # Collect all per_frame data
    all_frames = []
    
    if not OUTPUT_BASE.exists():
        print(f"  ERROR: Output directory not found: {OUTPUT_BASE}")
        return
    
    # Find all chunk directories
    chunk_dirs = sorted([d for d in OUTPUT_BASE.iterdir() if d.is_dir() and d.name.startswith('chunk_')],
                       key=lambda x: int(x.name.split('_')[1]))
    
    if not chunk_dirs:
        print("  No evaluated chunks found.")
        return
    
    for chunk_dir in chunk_dirs:
        clip_dirs = sorted([d for d in chunk_dir.iterdir() if d.is_dir() and d.name.startswith('clip_')],
                          key=lambda x: int(x.name.split('_')[1]))
        
        for clip_dir in clip_dirs:
            csv_path = clip_dir / 'per_frame.csv'
            if not csv_path.exists():
                continue
            
            # Read per_frame.csv
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Only include rows for the requested mode
                    if f'CoTracker_{mode}' in row['Method']:
                        frame_data = {
                            'chunk': chunk_dir.name,
                            'clip': clip_dir.name,
                            'frame': int(row['Frame']),
                            'edge_pct': float(row['EdgePctMean']) if row['EdgePctMean'] != 'nan' else np.nan,
                            'edge_rmse': float(row['EdgeRMSE']) if row['EdgeRMSE'] != 'nan' else np.nan,
                            'pos_rmse': float(row['PosRMSE']) if row['PosRMSE'] != 'nan' else np.nan,
                            'cd': float(row['CD']) if row['CD'] != 'nan' else np.nan,
                            'pred2ref': float(row['Pred2Ref']) if 'Pred2Ref' in row and row['Pred2Ref'] != 'nan' else np.nan,
                            'ref2pred': float(row['Ref2Pred']) if 'Ref2Pred' in row and row['Ref2Pred'] != 'nan' else np.nan,
                            'f10': float(row['F@10mm']) if row['F@10mm'] != 'nan' else np.nan,
                            'invalid_depth': int(row['InvalidDepth']) if 'InvalidDepth' in row else 0,
                        }
                        # New columns (may not exist in old CSVs)
                        if 'Prec@10mm' in row:
                            frame_data['prec_10'] = float(row['Prec@10mm']) if row['Prec@10mm'] != 'nan' else np.nan
                            frame_data['rec_10'] = float(row['Rec@10mm']) if row['Rec@10mm'] != 'nan' else np.nan
                            frame_data['n_pred'] = int(row['NPred']) if row['NPred'] != 'nan' else 0
                            frame_data['n_ref'] = int(row['NRef']) if row['NRef'] != 'nan' else 0
                        all_frames.append(frame_data)
    
    if not all_frames:
        print("  No valid frame data found.")
        return
    
    print(f"  Found {len(all_frames)} frames across {len(chunk_dirs)} chunks")
    
    # Compute overall statistics
    valid_edge = [f for f in all_frames if not np.isnan(f['edge_rmse'])]
    valid_cd = [f for f in all_frames if not np.isnan(f['cd'])]
    
    edge_rmse_mean = np.mean([f['edge_rmse'] for f in valid_edge]) if valid_edge else np.nan
    edge_rmse_std = np.std([f['edge_rmse'] for f in valid_edge]) if valid_edge else np.nan
    edge_pct_mean = np.mean([f['edge_pct'] for f in valid_edge]) if valid_edge else np.nan
    edge_pct_std = np.std([f['edge_pct'] for f in valid_edge]) if valid_edge else np.nan
    
    # Compute edge error threshold metrics (% of frames where edge_pct < threshold)
    edge_under_2pct = 100.0 * np.mean([f['edge_pct'] < 2.0 for f in valid_edge]) if valid_edge else 0.0
    edge_under_5pct = 100.0 * np.mean([f['edge_pct'] < 5.0 for f in valid_edge]) if valid_edge else 0.0
    edge_under_10pct = 100.0 * np.mean([f['edge_pct'] < 10.0 for f in valid_edge]) if valid_edge else 0.0
    
    cd_mean = np.mean([f['cd'] for f in valid_cd]) if valid_cd else np.nan
    cd_std = np.std([f['cd'] for f in valid_cd]) if valid_cd else np.nan
    f10_mean = np.mean([f['f10'] for f in valid_cd]) if valid_cd else 0.0
    f10_std = np.std([f['f10'] for f in valid_cd]) if valid_cd else 0.0

    valid_p2r = [f for f in all_frames if not np.isnan(f.get('pred2ref', np.nan))]
    valid_r2p = [f for f in all_frames if not np.isnan(f.get('ref2pred', np.nan))]
    p2r_mean = np.mean([f['pred2ref'] for f in valid_p2r]) if valid_p2r else np.nan
    p2r_std = np.std([f['pred2ref'] for f in valid_p2r]) if valid_p2r else np.nan
    r2p_mean = np.mean([f['ref2pred'] for f in valid_r2p]) if valid_r2p else np.nan
    r2p_std = np.std([f['ref2pred'] for f in valid_r2p]) if valid_r2p else np.nan
    
    # Precision and Recall (if available in data)
    has_prec_rec = any('prec_10' in f for f in all_frames)
    if has_prec_rec:
        valid_prec = [f for f in all_frames if 'prec_10' in f and not np.isnan(f.get('prec_10', np.nan))]
        prec_mean = np.mean([f['prec_10'] for f in valid_prec]) if valid_prec else np.nan
        prec_std = np.std([f['prec_10'] for f in valid_prec]) if valid_prec else np.nan
        rec_mean = np.mean([f['rec_10'] for f in valid_prec]) if valid_prec else np.nan
        rec_std = np.std([f['rec_10'] for f in valid_prec]) if valid_prec else np.nan
        avg_n_pred = np.mean([f['n_pred'] for f in valid_prec]) if valid_prec else 0
        avg_n_ref = np.mean([f['n_ref'] for f in valid_prec]) if valid_prec else 0
    
    avg_invalid = np.mean([f['invalid_depth'] for f in all_frames])
    
    # Write aggregated summary
    output_file = OUTPUT_BASE / f'aggregated_summary_{mode}.txt'
    with open(output_file, 'w') as f:
        f.write(f"DLO CoTracker Aggregated Results - Mode: {mode}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total frames: {len(all_frames)}\n")
        f.write(f"Chunks evaluated: {len(chunk_dirs)}\n\n")
        
        f.write("Edge Length Metrics (all frames)\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Edge % Mean: {edge_pct_mean:.2f}% ± {edge_pct_std:.2f}%\n")
        f.write(f"  Edge RMSE:   {edge_rmse_mean:.2f} ± {edge_rmse_std:.2f} mm\n")
        f.write(f"  Edge <2%:    {edge_under_2pct:.1f}%\n")
        f.write(f"  Edge <5%:    {edge_under_5pct:.1f}%\n")
        f.write(f"  Edge <10%:   {edge_under_10pct:.1f}%\n\n")
        
        f.write("Chamfer Distance Metrics (all frames)\n")
        f.write("-" * 60 + "\n")
        f.write(f"  CD:          {cd_mean:.2f} ± {cd_std:.2f} mm\n")
        if valid_p2r:
            f.write(f"  Pred→Ref:    {p2r_mean:.2f} ± {p2r_std:.2f} mm\n")
        if valid_r2p:
            f.write(f"  Ref→Pred:    {r2p_mean:.2f} ± {r2p_std:.2f} mm\n")
        if has_prec_rec:
            f.write(f"  Prec@10mm:   {prec_mean:.1f}% ± {prec_std:.1f}%  (predicted pts near ref)\n")
            f.write(f"  Rec@10mm:    {rec_mean:.1f}% ± {rec_std:.1f}%  (ref pts covered by pred)\n")
        f.write(f"  F@10mm:      {f10_mean:.1f}% ± {f10_std:.1f}%\n")
        if has_prec_rec:
            f.write(f"\n  Avg pred cloud:  {avg_n_pred:.0f} points\n")
            f.write(f"  Avg ref cloud:   {avg_n_ref:.0f} points\n")
        f.write(f"\nAverage invalid depth keypoints per frame: {avg_invalid:.1f}\n")
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"DLO CoTracker Results Summary - Mode: {mode}")
    print("=" * 60)
    print(f"Total frames: {len(all_frames)}")
    print(f"\nEdge RMSE:    {edge_rmse_mean:.2f} ± {edge_rmse_std:.2f} mm")
    print(f"Edge %:       {edge_pct_mean:.2f}% ± {edge_pct_std:.2f}%")
    print(f"CD:           {cd_mean:.2f} ± {cd_std:.2f} mm")
    if valid_p2r:
        print(f"Pred->Ref:    {p2r_mean:.2f} ± {p2r_std:.2f} mm")
    if valid_r2p:
        print(f"Ref->Pred:    {r2p_mean:.2f} ± {r2p_std:.2f} mm")
    if has_prec_rec:
        print(f"Prec@10mm:    {prec_mean:.1f}% ± {prec_std:.1f}%  (pred near ref)")
        print(f"Rec@10mm:     {rec_mean:.1f}% ± {rec_std:.1f}%  (coverage)")
    print(f"F@10mm:       {f10_mean:.1f}% ± {f10_std:.1f}%")
    if has_prec_rec:
        print(f"\nAvg pred cloud: {avg_n_pred:.0f} pts, Avg ref cloud: {avg_n_ref:.0f} pts")
    print(f"Avg Invalid:  {avg_invalid:.1f} keypoints/frame")
    
    print(f"\nSaved aggregated summary to: {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate CoTracker results on DLO')
    parser.add_argument('--chunk', type=int, default=-1,
                        help='Chunk index (required unless --all)')
    parser.add_argument('--all', action='store_true',
                        help='Aggregate all evaluated results')
    parser.add_argument('--mode', type=str, default='offline', choices=['online', 'offline'],
                        help='CoTracker mode to evaluate (default: offline)')
    parser.add_argument('--clip_seconds', type=int, default=15,
                        help='Clip duration in seconds (default: 15)')
    args = parser.parse_args()

    # Handle aggregation
    if args.all:
        aggregate_all_results(mode=args.mode)
        return
    
    # Validate chunk argument for single chunk evaluation
    if args.chunk < 0:
        print("ERROR: --chunk is required for single chunk evaluation")
        print("Use --all to aggregate all results")
        return
    
    # Paths
    chunk_dir = DATA_BASE / f'chunk_{args.chunk}'
    cotracker_dir = COTRACKER_RESULTS / f'chunk_{args.chunk}'
    dlo_eval_dir = DLO_EVAL_BASE / f'chunk_{args.chunk}'
    output_dir = OUTPUT_BASE / f'chunk_{args.chunk}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"EVALUATE DLO COTRACKER - Chunk {args.chunk}, Mode: {args.mode}")
    print("=" * 80)

    # Check paths
    if not chunk_dir.exists():
        print(f"ERROR: Chunk directory not found: {chunk_dir}")
        return
    
    if not cotracker_dir.exists():
        print(f"ERROR: CoTracker results not found: {cotracker_dir}")
        print("Run dlo_cotracker.py first.")
        return
    
    if not dlo_eval_dir.exists():
        print(f"ERROR: DLO evaluation results not found: {dlo_eval_dir}")
        print("Run dlo1_batch_experiment.py first to get reference edge lengths.")
        return

    # Load calibration
    print(f"\nLoading calibration...")
    calib = load_calibration(CALIB_DIR)
    K = calib['K']

    # Load data
    print(f"\nLoading depth/mask data...")
    data = load_chunk_data(chunk_dir)
    total_frames = data['n_frames']

    # Calculate clips
    frames_per_clip = args.clip_seconds * FPS
    n_clips = (total_frames + frames_per_clip - 1) // frames_per_clip

    print(f"\nClip configuration:")
    print(f"  Total frames: {total_frames}")
    print(f"  Clip duration: {args.clip_seconds}s ({frames_per_clip} frames)")
    print(f"  Number of clips: {n_clips}")

    # Find available clips
    available_clips = []
    for clip_idx in range(n_clips):
        cotracker_clip_dir = cotracker_dir / f'clip_{clip_idx}'
        dlo_clip_dir = dlo_eval_dir / f'clip_{clip_idx}'
        
        if cotracker_clip_dir.exists() and dlo_clip_dir.exists():
            available_clips.append(clip_idx)
    
    print(f"  Available clips: {available_clips}")

    # Process each clip
    all_results = []
    for clip_idx in available_clips:
        start_frame = clip_idx * frames_per_clip
        end_frame = min(start_frame + frames_per_clip, total_frames)
        
        # Load CoTracker results
        cotracker_results = load_cotracker_results(cotracker_dir, clip_idx, mode=args.mode)
        
        # Load DLO reference (for edge topology and lengths)
        dlo_ref = load_dlo_eval_reference(dlo_eval_dir, clip_idx)
        
        result = process_clip(
            data=data,
            cotracker_results=cotracker_results,
            dlo_ref=dlo_ref,
            K=K,
            T_left_base2cam=calib['T_left_base2cam'],
            T_right_base2cam=calib['T_right_base2cam'],
            clip_idx=clip_idx,
            start_frame=start_frame,
            end_frame=end_frame,
            output_dir=output_dir,
            mode=args.mode,
        )
        all_results.append(result)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Chunk {args.chunk}, Mode: {args.mode}")
    print(f"Processed {len(all_results)} clips")
    print(f"\n{'Clip':<6} | {'EdgeRMSE':<10} | {'Edge%':<10} | {'CD':<10} | {'F@10mm':<10} | {'InvalidDepth':<12}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['clip_idx']:<6} | {r['edge_rmse']:8.2f} mm | {r['edge_pct']:8.2f}% | {r['cd']:8.2f} mm | {r['f10']:8.1f}% | {r['avg_invalid_depth']:10.1f}")
    
    # Overall average
    if all_results:
        avg_edge_rmse = np.mean([r['edge_rmse'] for r in all_results])
        avg_edge_pct = np.mean([r['edge_pct'] for r in all_results])
        avg_cd = np.mean([r['cd'] for r in all_results])
        avg_f10 = np.mean([r['f10'] for r in all_results])
        print("-" * 70)
        print(f"{'AVG':<6} | {avg_edge_rmse:8.2f} mm | {avg_edge_pct:8.2f}% | {avg_cd:8.2f} mm | {avg_f10:8.1f}%")
    
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
