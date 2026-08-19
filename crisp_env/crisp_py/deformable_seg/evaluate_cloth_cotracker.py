#!/usr/bin/env python3
"""
Evaluate CoTracker results on cloth datasets.

Uses the same metrics as fabric_batch_experiment.py:
- Edge Length Metrics (Edge %, RMSE, <2%, <5%, <10%)
- Position RMSE (distance to surface point cloud)
- Chamfer Distance (CD, Precision, Recall, F-score at 2/5/10mm)

Handles invalid depth at CoTracker tracking points by:
1. Searching neighborhood for valid depth
2. Using median of valid depths in neighborhood
3. Marking as NaN if no valid depth found

Usage:
    # Evaluate single dataset/chunk
    python evaluate_cloth_cotracker.py --dataset cloth_no_occlusion_back_3sec --chunk 0
    
    # Aggregate all results (after running individual evaluations)
    python evaluate_cloth_cotracker.py --dataset all --mode offline
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# PATHS
# ============================================================================

# Raw data base path
DATA_BASE = Path('/mnt/mydisk/captured_data_double_arm')

# CoTracker results base path
COTRACKER_RESULTS_BASE = Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/deformable_seg/fabric_cotracker_results')

# Fabric evaluation results (for ground truth edge topology)
FABRIC_EVAL_BASE = Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/deformable_seg/fabric_evaluation_results')

# Output base path
OUTPUT_BASE = Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/deformable_seg/fabric_cotracker_evaluation')

# Calibration path
CALIB_DIR = Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/'
                 'roahm-deformable-objects/captured_calibration_data/dlo1_cloth1_calibration')

# Grid size (6×6 = 36 keypoints)
GRID_ROWS = 6
GRID_COLS = 6

# Default FPS
FPS = 30


# ============================================================================
# DATA LOADING
# ============================================================================

def load_calibration(calib_dir: Path) -> dict:
    """Load camera intrinsics from calibration directory."""
    tf = np.load(calib_dir / 'transform_ee_cam_world.npz')
    K = tf['K']
    return {'K': K}


def load_chunk_data(chunk_dir: Path, max_frames: int = 10000) -> dict:
    """Load depth and fg_mask from chunk directory."""
    print(f"Loading data from {chunk_dir}...")
    
    rgbd = np.load(chunk_dir / 'rgbd.npz')
    n_total = rgbd['color'].shape[0]
    start_idx = max(0, n_total - max_frames)
    
    depth = rgbd['depth'][start_idx:]
    
    # Load foreground mask
    fg_mask_path = chunk_dir / 'fg_mask.npz'
    if fg_mask_path.exists():
        fg_mask = np.load(fg_mask_path)['fg_mask'][start_idx:]
    else:
        fg_mask = None
        print(f"  Warning: fg_mask.npz not found")
    
    print(f"  Loaded {len(depth)} frames")
    
    return {
        'depth': depth,
        'fg_mask': fg_mask,
        'n_frames': len(depth),
    }


def load_cotracker_results(cotracker_dir: Path, clip_idx: int, mode: str = 'offline') -> dict:
    """Load CoTracker tracking results."""
    clip_dir = cotracker_dir / f'clip_{clip_idx:02d}'
    npz_path = clip_dir / f'keypoints_cotracker_{mode}.npz'
    
    if not npz_path.exists():
        raise FileNotFoundError(f"CoTracker results not found: {npz_path}")
    
    data = np.load(npz_path)
    return {
        'keypoints_2d': data['keypoints_2d'],  # (T, N, 2)
        'edge_connections': data['edge_connections'],  # (E, 2)
        'initial_keypoints_3d': data['initial_keypoints_3d'],  # (N, 3)
    }


def load_fabric_eval_reference(fabric_eval_dir: Path, clip_idx: int) -> dict:
    """Load reference data from fabric evaluation (for edge lengths)."""
    clip_dir = fabric_eval_dir / f'clip_{clip_idx:02d}'
    npz_path = clip_dir / '3d_keypoints.npz'
    
    if not npz_path.exists():
        raise FileNotFoundError(f"Fabric eval results not found: {npz_path}")
    
    data = np.load(npz_path)
    return {
        'edge_connections': data['edge_connections'],
        'reference_lengths': data['reference_lengths'],
        'gt_keypoints': data['full'],  # (T, N, 3) ground truth from FabricTracker
    }


# ============================================================================
# 2D → 3D LIFTING WITH DEPTH
# ============================================================================

def lift_2d_to_3d_with_depth(
    keypoints_2d: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    search_radius: int = 5,
) -> tuple:
    """
    Lift 2D keypoints to 3D using depth map.
    
    Args:
        keypoints_2d: (N, 2) keypoint positions (x, y) = (col, row)
        depth: (H, W) depth map in mm
        K: 3×3 camera intrinsic matrix
        search_radius: Radius to search for valid depth if center is invalid
        
    Returns:
        keypoints_3d: (N, 3) 3D positions in mm (NaN if invalid)
        valid_mask: (N,) boolean mask of valid keypoints
    """
    N = len(keypoints_2d)
    H, W = depth.shape
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    keypoints_3d = np.full((N, 3), np.nan)
    valid_mask = np.zeros(N, dtype=bool)
    
    for i in range(N):
        x, y = keypoints_2d[i]  # x=col, y=row
        col, row = int(round(x)), int(round(y))
        
        # Check bounds
        if col < 0 or col >= W or row < 0 or row >= H:
            continue
        
        # Try to get valid depth
        z = depth[row, col]
        
        # If invalid (0 or very small), search neighborhood
        if z <= 0 or z > 2000:  # Invalid depth (foreground only)
            # Search in neighborhood
            r_min = max(0, row - search_radius)
            r_max = min(H, row + search_radius + 1)
            c_min = max(0, col - search_radius)
            c_max = min(W, col + search_radius + 1)
            
            neighborhood = depth[r_min:r_max, c_min:c_max]
            valid_depths = neighborhood[(neighborhood > 0) & (neighborhood < 2000)]
            
            if len(valid_depths) > 0:
                z = np.median(valid_depths)  # Use median for robustness
            else:
                continue  # No valid depth in neighborhood
        
        # Backproject to 3D
        X = (col - cx) * z / fx
        Y = (row - cy) * z / fy
        Z = z
        
        keypoints_3d[i] = [X, Y, Z]
        valid_mask[i] = True
    
    return keypoints_3d, valid_mask


# ============================================================================
# METRICS (same as fabric_batch_experiment.py)
# ============================================================================

def compute_edge_metrics(keypoints, edges, reference_lengths):
    """Compute edge length metrics."""
    if keypoints is None or len(keypoints) == 0 or edges is None or len(edges) == 0:
        return {
            'pct_mean': 0.0, 'pct_std': 0.0, 'pct_max': 0.0,
            'rmse_mm': 0.0, 'under_2pct': 0.0, 'under_5pct': 0.0, 'under_10pct': 0.0,
        }

    pct_errors = []
    abs_errors = []
    for edge_idx, (i, j) in enumerate(edges):
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        if np.any(np.isnan(keypoints[i])) or np.any(np.isnan(keypoints[j])):
            continue
        
        current_len = np.linalg.norm(keypoints[i] - keypoints[j])
        ref_len = reference_lengths[edge_idx] if edge_idx < len(reference_lengths) else current_len
        
        if ref_len > 0:
            pct_error = abs(current_len - ref_len) / ref_len
            abs_error = abs(current_len - ref_len)
            pct_errors.append(pct_error)
            abs_errors.append(abs_error)

    pct_errors = np.array(pct_errors)
    abs_errors = np.array(abs_errors)

    if len(pct_errors) == 0:
        return {
            'pct_mean': 0.0, 'pct_std': 0.0, 'pct_max': 0.0,
            'rmse_mm': 0.0, 'under_2pct': 0.0, 'under_5pct': 0.0, 'under_10pct': 0.0,
        }

    return {
        'pct_mean': np.mean(pct_errors) * 100,
        'pct_std': np.std(pct_errors) * 100,
        'pct_max': np.max(pct_errors) * 100,
        'rmse_mm': np.sqrt(np.mean(abs_errors ** 2)),
        'under_2pct': np.mean(pct_errors < 0.02) * 100,
        'under_5pct': np.mean(pct_errors < 0.05) * 100,
        'under_10pct': np.mean(pct_errors < 0.10) * 100,
    }


def compute_position_metrics(keypoints, point_cloud):
    """Compute position metrics (distance to nearest point in surface)."""
    if keypoints is None or point_cloud is None or len(point_cloud) == 0:
        return {
            'rmse_mm': 0.0, 'under_2mm': 0.0, 'under_5mm': 0.0, 'under_10mm': 0.0,
        }
    
    # Filter out NaN keypoints
    valid_kps = keypoints[~np.any(np.isnan(keypoints), axis=1)]
    if len(valid_kps) == 0:
        return {
            'rmse_mm': 0.0, 'under_2mm': 0.0, 'under_5mm': 0.0, 'under_10mm': 0.0,
        }

    nn = NearestNeighbors(n_neighbors=1).fit(point_cloud)
    distances, _ = nn.kneighbors(valid_kps)
    distances = distances.flatten()

    return {
        'rmse_mm': np.sqrt(np.mean(distances ** 2)),
        'under_2mm': np.mean(distances < 2.0) * 100,
        'under_5mm': np.mean(distances < 5.0) * 100,
        'under_10mm': np.mean(distances < 10.0) * 100,
    }


def extract_surface_point_cloud(fg_mask, depth, K, max_points=5000):
    """Extract 3D point cloud from foreground mask."""
    if fg_mask is None or depth is None:
        return np.empty((0, 3))
    
    rows, cols = np.where(fg_mask > 0)
    if len(rows) == 0:
        return np.empty((0, 3))
    
    z_vals = depth[rows, cols].astype(np.float32)
    valid = (z_vals > 0) & (z_vals < 2000)
    rows, cols, z_vals = rows[valid], cols[valid], z_vals[valid]
    
    if len(z_vals) == 0:
        return np.empty((0, 3))
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x_vals = (cols - cx) * z_vals / fx
    y_vals = (rows - cy) * z_vals / fy
    
    pc = np.column_stack([x_vals, y_vals, z_vals]).astype(np.float32)
    
    if len(pc) > max_points:
        indices = np.random.choice(len(pc), max_points, replace=False)
        pc = pc[indices]
    
    return pc


def sample_points_on_faces(keypoints, grid_rows, grid_cols, n_samples_per_face=10):
    """Sample points uniformly on quad faces for Chamfer distance."""
    if keypoints is None or len(keypoints) == 0:
        return np.empty((0, 3))
    
    # Filter out invalid keypoints
    if np.any(np.isnan(keypoints)):
        # Can't sample faces if any keypoint is invalid
        # Return valid keypoints as fallback
        valid_kps = keypoints[~np.any(np.isnan(keypoints), axis=1)]
        return valid_kps
    
    if len(keypoints) != grid_rows * grid_cols:
        return keypoints
    
    sampled_points = []
    
    for r in range(grid_rows - 1):
        for c in range(grid_cols - 1):
            # Quad corners: top-left, top-right, bottom-left, bottom-right
            tl = r * grid_cols + c
            tr = r * grid_cols + c + 1
            bl = (r + 1) * grid_cols + c
            br = (r + 1) * grid_cols + c + 1
            
            p_tl = keypoints[tl]
            p_tr = keypoints[tr]
            p_bl = keypoints[bl]
            p_br = keypoints[br]
            
            # Sample using bilinear interpolation
            for _ in range(n_samples_per_face):
                u, v = np.random.rand(2)
                p = (1-u)*(1-v)*p_tl + u*(1-v)*p_tr + (1-u)*v*p_bl + u*v*p_br
                sampled_points.append(p)
    
    if len(sampled_points) == 0:
        return np.empty((0, 3))
    
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
    
    # Pred → Ref distances
    nn_ref = NearestNeighbors(n_neighbors=1).fit(ref_cloud)
    pred2ref_dists, _ = nn_ref.kneighbors(pred_cloud)
    pred2ref_dists = pred2ref_dists.flatten()
    
    # Ref → Pred distances  
    nn_pred = NearestNeighbors(n_neighbors=1).fit(pred_cloud)
    ref2pred_dists, _ = nn_pred.kneighbors(ref_cloud)
    ref2pred_dists = ref2pred_dists.flatten()
    
    pred2ref_avg = np.mean(pred2ref_dists)
    ref2pred_avg = np.mean(ref2pred_dists)
    cd = (pred2ref_avg + ref2pred_avg) / 2
    
    def compute_pr_f(p2r, r2p, thresh):
        precision = np.mean(p2r < thresh) * 100
        recall = np.mean(r2p < thresh) * 100
        f_score = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f_score
    
    p2, r2, f2 = compute_pr_f(pred2ref_dists, ref2pred_dists, 2.0)
    p5, r5, f5 = compute_pr_f(pred2ref_dists, ref2pred_dists, 5.0)
    p10, r10, f10 = compute_pr_f(pred2ref_dists, ref2pred_dists, 10.0)
    
    return {
        'pred2ref_avg': pred2ref_avg,
        'ref2pred_avg': ref2pred_avg,
        'cd': cd,
        'precision_2mm': p2, 'precision_5mm': p5, 'precision_10mm': p10,
        'recall_2mm': r2, 'recall_5mm': r5, 'recall_10mm': r10,
        'f_2mm': f2, 'f_5mm': f5, 'f_10mm': f10,
    }


# ============================================================================
# CLIP PROCESSING
# ============================================================================

def process_clip(
    data: dict,
    cotracker_results: dict,
    fabric_ref: dict,
    K: np.ndarray,
    clip_idx: int,
    start_frame: int,
    end_frame: int,
    output_dir: Path,
    mode: str = 'offline',
):
    """Process a single clip and compute metrics."""
    clip_dir = output_dir / f'clip_{clip_idx:02d}'
    clip_dir.mkdir(parents=True, exist_ok=True)
    
    n_frames = end_frame - start_frame
    
    keypoints_2d = cotracker_results['keypoints_2d']  # (T, N, 2)
    edge_connections = fabric_ref['edge_connections']
    reference_lengths = fabric_ref['reference_lengths']
    
    depth = data['depth'][start_frame:end_frame]
    fg_mask = data['fg_mask'][start_frame:end_frame] if data['fg_mask'] is not None else None
    
    print(f"\n  Processing clip {clip_idx} ({mode}): frames {start_frame}-{end_frame} ({n_frames} frames)")
    
    # Storage
    all_keypoints_3d = []
    all_edge_metrics = []
    all_pos_metrics = []
    all_cd_metrics = []
    invalid_depth_counts = []
    
    for frame_idx in tqdm(range(n_frames), desc=f"  Clip {clip_idx}"):
        kp_2d = keypoints_2d[frame_idx]  # (N, 2)
        
        # Lift 2D to 3D using depth
        kp_3d, valid_mask = lift_2d_to_3d_with_depth(kp_2d, depth[frame_idx], K, search_radius=5)
        all_keypoints_3d.append(kp_3d)
        
        n_invalid = np.sum(~valid_mask)
        invalid_depth_counts.append(n_invalid)
        
        # Compute edge metrics
        edge_m = compute_edge_metrics(kp_3d, edge_connections, reference_lengths)
        all_edge_metrics.append(edge_m)
        
        # Extract surface point cloud
        if fg_mask is not None:
            pc = extract_surface_point_cloud(fg_mask[frame_idx], depth[frame_idx], K)
        else:
            pc = np.empty((0, 3))
        
        # Compute position metrics
        pos_m = compute_position_metrics(kp_3d, pc)
        all_pos_metrics.append(pos_m)
        
        # Compute Chamfer metrics
        pred_cloud = sample_points_on_faces(kp_3d, GRID_ROWS, GRID_COLS)
        cd_m = compute_chamfer_metrics(pred_cloud, pc)
        all_cd_metrics.append(cd_m)
    
    # Save per_frame.csv
    per_frame_csv = clip_dir / 'per_frame.csv'
    with open(per_frame_csv, 'w') as f:
        f.write("Frame,GlobalFrame,Method,EdgePctMean,EdgePctStd,EdgePctMax,EdgeRMSE,PosRMSE,")
        f.write("Edge<2%,Edge<5%,Edge<10%,Pos<2mm,Pos<5mm,Pos<10mm,")
        f.write("CD,Pred2Ref,Ref2Pred,Prec@2mm,Prec@5mm,Prec@10mm,Rec@2mm,Rec@5mm,Rec@10mm,")
        f.write("F@2mm,F@5mm,F@10mm,InvalidDepth\n")
        
        for i in range(n_frames):
            em = all_edge_metrics[i]
            pm = all_pos_metrics[i]
            cm = all_cd_metrics[i]
            f.write(f"{i},{start_frame + i},CoTracker_{mode},")
            f.write(f"{em['pct_mean']:.6f},{em['pct_std']:.6f},{em['pct_max']:.6f},{em['rmse_mm']:.6f},")
            f.write(f"{pm['rmse_mm']:.6f},")
            f.write(f"{em['under_2pct']:.4f},{em['under_5pct']:.4f},{em['under_10pct']:.4f},")
            f.write(f"{pm['under_2mm']:.4f},{pm['under_5mm']:.4f},{pm['under_10mm']:.4f},")
            f.write(f"{cm['cd']:.4f},{cm['pred2ref_avg']:.4f},{cm['ref2pred_avg']:.4f},")
            f.write(f"{cm['precision_2mm']:.4f},{cm['precision_5mm']:.4f},{cm['precision_10mm']:.4f},")
            f.write(f"{cm['recall_2mm']:.4f},{cm['recall_5mm']:.4f},{cm['recall_10mm']:.4f},")
            f.write(f"{cm['f_2mm']:.4f},{cm['f_5mm']:.4f},{cm['f_10mm']:.4f},")
            f.write(f"{invalid_depth_counts[i]}\n")
    
    # Compute summary statistics
    edge_pct_mean = np.mean([m['pct_mean'] for m in all_edge_metrics])
    edge_pct_std = np.std([m['pct_mean'] for m in all_edge_metrics])
    edge_rmse_mean = np.mean([m['rmse_mm'] for m in all_edge_metrics])
    edge_rmse_std = np.std([m['rmse_mm'] for m in all_edge_metrics])
    edge_under_2 = np.mean([m['under_2pct'] for m in all_edge_metrics])
    edge_under_5 = np.mean([m['under_5pct'] for m in all_edge_metrics])
    edge_under_10 = np.mean([m['under_10pct'] for m in all_edge_metrics])
    
    pos_rmse_mean = np.mean([m['rmse_mm'] for m in all_pos_metrics])
    pos_rmse_std = np.std([m['rmse_mm'] for m in all_pos_metrics])
    pos_under_2 = np.mean([m['under_2mm'] for m in all_pos_metrics])
    pos_under_5 = np.mean([m['under_5mm'] for m in all_pos_metrics])
    pos_under_10 = np.mean([m['under_10mm'] for m in all_pos_metrics])
    
    cd_mean = np.mean([m['cd'] for m in all_cd_metrics])
    cd_std = np.std([m['cd'] for m in all_cd_metrics])
    p2r_mean = np.mean([m['pred2ref_avg'] for m in all_cd_metrics])
    r2p_mean = np.mean([m['ref2pred_avg'] for m in all_cd_metrics])
    f2_mean = np.mean([m['f_2mm'] for m in all_cd_metrics])
    f5_mean = np.mean([m['f_5mm'] for m in all_cd_metrics])
    f10_mean = np.mean([m['f_10mm'] for m in all_cd_metrics])
    
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
        f.write(f"{'Method':<20} | {'CD (mm)':<14} | {'Pred→Ref':<10} | {'Ref→Pred':<10} | {'F@2mm':<10} | {'F@5mm':<10} | {'F@10mm':<10}\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'CoTracker_' + mode:<20} | {cd_mean:5.2f} ±{cd_std:5.2f} mm | {p2r_mean:8.2f} mm | {r2p_mean:8.2f} mm | {f2_mean:8.2f}% | {f5_mean:8.2f}% | {f10_mean:8.2f}%\n")
        f.write("\n")
        
        f.write(f"Average invalid depth keypoints per frame: {avg_invalid:.1f} / {GRID_ROWS * GRID_COLS}\n")
    
    # Save 3d_keypoints.npz
    np.savez(
        clip_dir / '3d_keypoints.npz',
        cotracker=np.array(all_keypoints_3d),  # (T, N, 3)
        edge_connections=edge_connections,
        reference_lengths=reference_lengths,
    )
    
    print(f"    Saved outputs to: {clip_dir}")
    print(f"    Edge RMSE: {edge_rmse_mean:.2f} ± {edge_rmse_std:.2f} mm")
    print(f"    CD: {cd_mean:.2f} ± {cd_std:.2f} mm, F@10mm: {f10_mean:.1f}%")
    print(f"    Avg invalid depth: {avg_invalid:.1f} / {GRID_ROWS * GRID_COLS} keypoints")
    
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
# DATASET CONFIGURATIONS
# ============================================================================

DATASET_CHUNKS = {
    'cloth_no_occlusion_back_3sec': [0, 3, 7, 12, 20],
    'cloth_no_occlusion_back_4sec': [8, 13],
    'cloth_no_occlusion_front_3sec': [2, 5, 6, 7, 11, 14, 17],
    'cloth_no_occlusion_front_4sec': [15, 21, 22, 23, 27, 28],
}


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate_all_results(mode: str = 'offline'):
    """Aggregate results from all evaluated datasets/chunks."""
    import csv
    
    print("=" * 100)
    print(f"AGGREGATING ALL COTRACKER RESULTS - Mode: {mode}")
    print("=" * 100)
    
    # Collect all per_frame data
    all_frames = []
    dataset_summaries = []
    
    for dataset, chunks in DATASET_CHUNKS.items():
        dataset_frames = []
        
        for chunk in chunks:
            chunk_dir = OUTPUT_BASE / dataset / f'chunk_{chunk}'
            
            # Find all clip directories
            if not chunk_dir.exists():
                print(f"  Skip: {dataset}/chunk_{chunk} (not evaluated)")
                continue
            
            clip_dirs = sorted(chunk_dir.glob('clip_*'))
            
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
                                'dataset': dataset,
                                'chunk': chunk,
                                'clip': clip_dir.name,
                                'frame': int(row['Frame']),
                                'edge_pct': float(row['EdgePctMean']),
                                'edge_rmse': float(row['EdgeRMSE']),
                                'pos_rmse': float(row['PosRMSE']),
                                'cd': float(row['CD']),
                                'edge_under_2': float(row['Edge<2%']),
                                'edge_under_5': float(row['Edge<5%']),
                                'edge_under_10': float(row['Edge<10%']),
                                'pos_under_2': float(row['Pos<2mm']),
                                'pos_under_5': float(row['Pos<5mm']),
                                'pos_under_10': float(row['Pos<10mm']),
                                'f_2mm': float(row['F@2mm']),
                                'f_5mm': float(row['F@5mm']),
                                'f_10mm': float(row['F@10mm']),
                                'invalid_depth': float(row['InvalidDepth']),
                            }
                            all_frames.append(frame_data)
                            dataset_frames.append(frame_data)
        
        # Dataset-level summary
        if dataset_frames:
            n_frames = len(dataset_frames)
            dataset_summaries.append({
                'dataset': dataset,
                'n_frames': n_frames,
                'edge_pct': np.mean([f['edge_pct'] for f in dataset_frames]),
                'edge_rmse': np.mean([f['edge_rmse'] for f in dataset_frames]),
                'pos_rmse': np.mean([f['pos_rmse'] for f in dataset_frames]),
                'cd': np.mean([f['cd'] for f in dataset_frames]),
                'edge_under_5': np.mean([f['edge_under_5'] for f in dataset_frames]),
                'f_10mm': np.mean([f['f_10mm'] for f in dataset_frames]),
            })
            print(f"  {dataset}: {n_frames} frames")
    
    if not all_frames:
        print("\nNo evaluation results found!")
        print("Run individual evaluations first:")
        print("  python evaluate_cloth_cotracker.py --dataset <name> --chunk <idx> --mode " + mode)
        return
    
    # Overall aggregation
    total_frames = len(all_frames)
    
    # Compute aggregated metrics
    edge_pct_mean = np.mean([f['edge_pct'] for f in all_frames])
    edge_pct_std = np.std([f['edge_pct'] for f in all_frames])
    edge_rmse_mean = np.mean([f['edge_rmse'] for f in all_frames])
    edge_rmse_std = np.std([f['edge_rmse'] for f in all_frames])
    edge_under_2 = np.mean([f['edge_under_2'] for f in all_frames])
    edge_under_5 = np.mean([f['edge_under_5'] for f in all_frames])
    edge_under_10 = np.mean([f['edge_under_10'] for f in all_frames])
    
    pos_rmse_mean = np.mean([f['pos_rmse'] for f in all_frames])
    pos_rmse_std = np.std([f['pos_rmse'] for f in all_frames])
    pos_under_2 = np.mean([f['pos_under_2'] for f in all_frames])
    pos_under_5 = np.mean([f['pos_under_5'] for f in all_frames])
    pos_under_10 = np.mean([f['pos_under_10'] for f in all_frames])
    
    cd_mean = np.mean([f['cd'] for f in all_frames])
    cd_std = np.std([f['cd'] for f in all_frames])
    f_2mm = np.mean([f['f_2mm'] for f in all_frames])
    f_5mm = np.mean([f['f_5mm'] for f in all_frames])
    f_10mm = np.mean([f['f_10mm'] for f in all_frames])
    
    avg_invalid = np.mean([f['invalid_depth'] for f in all_frames])
    
    # Print results
    print("\n" + "=" * 100)
    print(f"AGGREGATED RESULTS - CoTracker ({mode})")
    print(f"Total frames: {total_frames}")
    print("=" * 100)
    
    print("\nPer-Dataset Summary:")
    print("-" * 100)
    print(f"{'Dataset':<35} | {'Frames':<8} | {'Edge%':<8} | {'EdgeRMSE':<10} | {'CD':<10} | {'F@10mm':<10}")
    print("-" * 100)
    for s in dataset_summaries:
        print(f"{s['dataset']:<35} | {s['n_frames']:<8} | {s['edge_pct']:6.2f}% | {s['edge_rmse']:8.2f} mm | {s['cd']:8.2f} mm | {s['f_10mm']:8.1f}%")
    print("-" * 100)
    
    print("\n" + "=" * 100)
    print("OVERALL METRICS")
    print("=" * 100)
    
    print("\nEdge Length Metrics")
    print("-" * 100)
    print(f"{'Method':<20} | {'Edge % Mean':<18} | {'Edge RMSE (mm)':<16} | {'<2%':<8} | {'<5%':<8} | {'<10%':<8}")
    print("-" * 100)
    print(f"{'CoTracker_' + mode:<20} | {edge_pct_mean:5.2f}% ± {edge_pct_std:5.2f}% | {edge_rmse_mean:5.2f} ±{edge_rmse_std:5.2f} mm | {edge_under_2:5.1f}% | {edge_under_5:5.1f}% | {edge_under_10:5.1f}%")
    
    print("\nPosition RMSE Metrics")
    print("-" * 80)
    print(f"{'Method':<20} | {'Pos RMSE (mm)':<18} | {'<2mm':<8} | {'<5mm':<8} | {'<10mm':<8}")
    print("-" * 80)
    print(f"{'CoTracker_' + mode:<20} | {pos_rmse_mean:5.2f} ± {pos_rmse_std:5.2f} mm | {pos_under_2:5.1f}% | {pos_under_5:5.1f}% | {pos_under_10:5.1f}%")
    
    print("\nChamfer Distance Metrics")
    print("-" * 100)
    print(f"{'Method':<20} | {'CD (mm)':<14} | {'F@2mm':<10} | {'F@5mm':<10} | {'F@10mm':<10}")
    print("-" * 100)
    print(f"{'CoTracker_' + mode:<20} | {cd_mean:5.2f} ±{cd_std:5.2f} mm | {f_2mm:8.2f}% | {f_5mm:8.2f}% | {f_10mm:8.2f}%")
    
    print(f"\nAverage invalid depth keypoints per frame: {avg_invalid:.1f} / {GRID_ROWS * GRID_COLS}")
    
    # Save aggregated summary
    output_file = OUTPUT_BASE / f'aggregated_summary_{mode}.txt'
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(f"CoTracker Aggregated Results - Mode: {mode}\n")
        f.write(f"Total frames: {total_frames}\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("Per-Dataset Summary:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Dataset':<35} | {'Frames':<8} | {'Edge%':<8} | {'EdgeRMSE':<10} | {'CD':<10} | {'F@10mm':<10}\n")
        f.write("-" * 100 + "\n")
        for s in dataset_summaries:
            f.write(f"{s['dataset']:<35} | {s['n_frames']:<8} | {s['edge_pct']:6.2f}% | {s['edge_rmse']:8.2f} mm | {s['cd']:8.2f} mm | {s['f_10mm']:8.1f}%\n")
        f.write("-" * 100 + "\n\n")
        
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
        f.write(f"{'Method':<20} | {'CD (mm)':<14} | {'F@2mm':<10} | {'F@5mm':<10} | {'F@10mm':<10}\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'CoTracker_' + mode:<20} | {cd_mean:5.2f} ±{cd_std:5.2f} mm | {f_2mm:8.2f}% | {f_5mm:8.2f}% | {f_10mm:8.2f}%\n")
        f.write("\n")
        
        f.write(f"Average invalid depth keypoints per frame: {avg_invalid:.1f} / {GRID_ROWS * GRID_COLS}\n")
    
    print(f"\nSaved aggregated summary to: {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate CoTracker results on cloth')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., cloth_no_occlusion_back_3sec) or "all" for aggregation')
    parser.add_argument('--chunk', type=int, default=-1,
                        help='Chunk index (required unless --dataset all)')
    parser.add_argument('--mode', type=str, default='offline', choices=['online', 'offline'],
                        help='CoTracker mode to evaluate (default: offline)')
    parser.add_argument('--clip_seconds', type=int, default=10,
                        help='Clip duration in seconds (default: 10)')
    parser.add_argument('--max_frames', type=int, default=10000,
                        help='Max frames to load (default: 10000)')
    args = parser.parse_args()

    # Handle "all" dataset for aggregation
    if args.dataset == 'all':
        aggregate_all_results(mode=args.mode)
        return
    
    # Validate chunk argument for single dataset evaluation
    if args.chunk < 0:
        print("ERROR: --chunk is required for single dataset evaluation")
        print("Use --dataset all to aggregate all results")
        return

    # Paths
    chunk_dir = DATA_BASE / args.dataset / f'chunk_{args.chunk}'
    cotracker_dir = COTRACKER_RESULTS_BASE / args.dataset / f'chunk_{args.chunk}'
    fabric_eval_dir = FABRIC_EVAL_BASE / args.dataset / f'chunk_{args.chunk}'
    output_dir = OUTPUT_BASE / args.dataset / f'chunk_{args.chunk}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"EVALUATE COTRACKER - Dataset: {args.dataset}")
    print(f"Chunk {args.chunk}, Mode: {args.mode}")
    print("=" * 80)

    # Check paths
    if not chunk_dir.exists():
        print(f"ERROR: Chunk directory not found: {chunk_dir}")
        return
    
    if not cotracker_dir.exists():
        print(f"ERROR: CoTracker results not found: {cotracker_dir}")
        print("Run cloth_cotracker.py first.")
        return
    
    if not fabric_eval_dir.exists():
        print(f"ERROR: Fabric evaluation results not found: {fabric_eval_dir}")
        print("Run fabric_batch_experiment.py first to get reference edge lengths.")
        return

    # Load calibration
    print(f"\nLoading calibration...")
    calib = load_calibration(CALIB_DIR)
    K = calib['K']

    # Load data
    print(f"\nLoading depth/mask data...")
    data = load_chunk_data(chunk_dir, max_frames=args.max_frames)
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
        cotracker_clip_dir = cotracker_dir / f'clip_{clip_idx:02d}'
        fabric_clip_dir = fabric_eval_dir / f'clip_{clip_idx:02d}'
        if cotracker_clip_dir.exists() and fabric_clip_dir.exists():
            available_clips.append(clip_idx)
    
    print(f"  Available clips: {available_clips}")

    # Process each clip
    all_results = []
    for clip_idx in available_clips:
        start_frame = clip_idx * frames_per_clip
        end_frame = min(start_frame + frames_per_clip, total_frames)
        
        # Load CoTracker results
        cotracker_results = load_cotracker_results(cotracker_dir, clip_idx, mode=args.mode)
        
        # Load fabric reference (for edge topology and lengths)
        fabric_ref = load_fabric_eval_reference(fabric_eval_dir, clip_idx)
        
        result = process_clip(
            data=data,
            cotracker_results=cotracker_results,
            fabric_ref=fabric_ref,
            K=K,
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
    print(f"Dataset: {args.dataset}, Mode: {args.mode}")
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
