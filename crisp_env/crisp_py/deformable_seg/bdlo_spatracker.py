#!/usr/bin/env python3
"""
SpaTrackerV2 baseline for BDLO (Branched Deformable Linear Object) tracking experiment.

Supports two datasets:
  - 4sec (bdlo_no_contact_4sec): 49 chunks, slower motion
  - 2sec (bdlo_no_contact_2sec): 20 chunks, faster motion

Usage:
    python bdlo_spatracker.py --dataset 4sec --chunk 0
    python bdlo_spatracker.py --dataset 2sec --chunk 0
    python bdlo_spatracker.py --dataset 4sec --chunk 0 --segment_seconds 5

Note: Run from the deformable_seg directory or ensure SpaTrackerV2 is in sys.path.
"""

import sys
import os

# Add SpaTrackerV2 to path
SPATRACKER_DIR = "/home/roahmlab/move_some_robots/crisp_env/crisp_py/SpaTrackerV2"
sys.path.insert(0, SPATRACKER_DIR)

import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASET_CONFIG = {
    '4sec': {
        'data_dir': Path('/mnt/mydisk/captured_data_double_arm/bdlo_no_contact_4sec'),
        'eval_results_dir': Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/deformable_seg/bdlo1_evaluation_results'),
        'output_dir': Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/deformable_seg/bdlo1_spatracker_results'),
        'n_chunks': 49,
        'clip_seconds_default': 10,
        'description': 'slower motion (4sec between contact)',
    },
    '2sec': {
        'data_dir': Path('/mnt/mydisk/captured_data_double_arm/bdlo_no_contact_2sec'),
        'eval_results_dir': Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/deformable_seg/bdlo1_faster_evaluation_results'),
        'output_dir': Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/deformable_seg/bdlo1_faster_spatracker_results'),
        'n_chunks': 20,
        'clip_seconds_default': 10,
        'description': 'faster motion (2sec between contact)',
    },
}

# Calibration is shared (same for both datasets)
CALIB_DIR = Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/'
                 'roahm-deformable-objects/captured_calibration_data/test_0227')

# Depth parameters
DEPTH_SCALE = 1000.0  # depth_value / scale = meters
MAX_DEPTH = 2000.0    # mm, filter background

FPS = 30
DEFAULT_CLIP_SECONDS = 10  # 300 frames


# ============================================================================
# DATA LOADING
# ============================================================================

def load_calibration(calib_dir: Path) -> dict:
    """Load camera intrinsics from calibration directory."""
    tf = np.load(calib_dir / 'transform_ee_cam_world.npz')
    K = tf['K']
    intrinsics = np.array([
        [K[0, 0], 0, K[0, 2]],
        [0, K[1, 1], K[1, 2]],
        [0, 0, 1]
    ], dtype=np.float32)
    return {'K': K, 'intrinsics': intrinsics}


def project_3d_to_2d(keypoints_3d: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    Project 3D keypoints to 2D pixel coordinates.
    
    Args:
        keypoints_3d: (N, 3) array of 3D points in camera frame (mm)
        intrinsics: (3, 3) camera intrinsic matrix
        
    Returns:
        keypoints_2d: (N, 2) array of 2D pixel coordinates (x, y) = (col, row)
    """
    if keypoints_3d is None or len(keypoints_3d) == 0:
        return np.empty((0, 2))
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    X, Y, Z = keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2]
    Z = np.clip(Z, 1e-6, None)  # Avoid division by zero
    
    u = (fx * X / Z) + cx  # x coordinate (column)
    v = (fy * Y / Z) + cy  # y coordinate (row)
    
    return np.stack([u, v], axis=1).astype(np.float32)


def load_chunk_data(chunk_dir: Path) -> dict:
    """
    Load RGBD data from chunk (last 600 frames for BDLO).
    
    Args:
        chunk_dir: Path to chunk directory
        
    Returns:
        dict with color (T, H, W, 3) BGR uint8, depth (T, H, W) uint16 mm,
        dlo_mask (T, H, W) bool
    """
    rgbd_path = chunk_dir / 'rgbd.npz'
    if not rgbd_path.exists():
        raise FileNotFoundError(f"rgbd.npz not found: {rgbd_path}")
    
    print(f"  Loading {rgbd_path}...")
    rgbd = np.load(rgbd_path)
    
    # BDLO uses last 600 frames
    color = rgbd['color'][-600:]  # BGR format
    depth = rgbd['depth'][-600:]
    
    print(f"    Color: {color.shape}, Depth: {depth.shape}")
    
    result = {'color': color, 'depth': depth}
    
    # Load BDLO mask from masks/masks.npz (last 600 frames)
    mask_path = chunk_dir / 'masks' / 'masks.npz'
    if mask_path.exists():
        dlo_mask = np.load(mask_path)['masks'][-600:]
        result['dlo_mask'] = dlo_mask > 0
        print(f"    BDLO mask: {dlo_mask.shape}")
    else:
        print(f"    Warning: masks.npz not found at {mask_path}")
        result['dlo_mask'] = None
    
    result['n_frames'] = color.shape[0]
    return result


def load_initial_keypoints(eval_results_dir: Path, clip_idx: int) -> tuple:
    """
    Load initial 3D keypoints from evaluation results.
    
    Returns:
        keypoints_3d: (N, 3) initial 3D keypoints (frame 0, 'full' method)
        edge_connections: (E, 2) edge indices
        reference_lengths: (E,) reference edge lengths
    """
    # DLO uses clip_0, clip_1 format (not clip_00, clip_01)
    clip_dir = eval_results_dir / f'clip_{clip_idx}'
    kp_path = clip_dir / '3d_keypoints.npz'
    
    if not kp_path.exists():
        raise FileNotFoundError(f"3D keypoints not found: {kp_path}")
    
    data = np.load(kp_path)
    keypoints_3d = data['full'][0]  # First frame, 'full' method
    edge_connections = data['edge_connection']  # Note: 'edge_connection' (singular) for DLO
    reference_lengths = data['reference_lengths']
    
    return keypoints_3d, edge_connections, reference_lengths


def convert_to_spatracker_format(color: np.ndarray, depth: np.ndarray, 
                                  intrinsics: np.ndarray, mask: np.ndarray = None,
                                  max_depth: float = MAX_DEPTH,
                                  rgb_only: bool = False) -> dict:
    """
    Convert RGBD to SpaTrackerV2 format.
    
    SpaTrackerV2 expects:
    - video: (T, C, H, W) float32, values in [0, 1], RGB order
    - depths: (T, H, W) float32, in meters (or None for RGB-only mode)
    - intrinsics: (T, 3, 3) float32
    - extrinsics: (T, 4, 4) float32
    
    Args:
        color: (T, H, W, 3) uint8, BGR order
        depth: (T, H, W) uint16 in mm
        intrinsics: (3, 3) camera intrinsic matrix
        mask: (T, H, W) bool foreground mask (optional)
        max_depth: maximum depth threshold in mm
        rgb_only: if True, don't provide depth (SpaTracker will estimate)
        
    Returns:
        dict with video, depths, intrinsics, extrinsics
    """
    T, H, W, C = color.shape
    
    # Convert BGR to RGB
    color_rgb = color[..., ::-1].copy()
    
    if rgb_only:
        # RGB-only mode: SpaTracker will estimate depth using MoGe
        print(f"    RGB-only mode: SpaTracker will estimate depth")
        depths = None
    else:
        depth_out = depth.copy().astype(np.float32)
        
        # Apply max depth threshold
        if max_depth > 0:
            far_mask = depth_out > max_depth
            n_far = np.sum(far_mask)
            print(f"    Filtering depth > {max_depth}mm: {n_far}/{far_mask.size} pixels ({100*n_far/far_mask.size:.1f}%)")
            depth_out[far_mask] = 0
        
        # Apply foreground mask if provided
        if mask is not None:
            bg_mask = ~mask
            n_bg = np.sum(bg_mask)
            print(f"    Masking background: {n_bg}/{bg_mask.size} pixels ({100*n_bg/bg_mask.size:.1f}%)")
            color_rgb[bg_mask] = 0
            depth_out[bg_mask] = 0
        
        # Convert depth to meters
        depths = depth_out / DEPTH_SCALE
    
    # Convert color to (T, C, H, W) and normalize to [0, 1]
    video = color_rgb.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    
    # Create intrinsics (same for all frames)
    K = np.array([
        [intrinsics[0, 0], 0, intrinsics[0, 2]],
        [0, intrinsics[1, 1], intrinsics[1, 2]],
        [0, 0, 1]
    ], dtype=np.float32)
    intrinsics_arr = np.tile(K[None, :, :], (T, 1, 1))
    
    # Create extrinsics (identity for static camera)
    extrinsics = np.tile(np.eye(4, dtype=np.float32)[None, :, :], (T, 1, 1))
    
    return {
        'video': video,
        'depths': depths,
        'intrinsics': intrinsics_arr,
        'extrinsics': extrinsics,
    }


def create_queries_from_keypoints_2d(keypoints_2d: np.ndarray) -> np.ndarray:
    """
    Convert 2D keypoints (x, y) to query format (t, x, y).
    
    SpaTrackerV2 expects queries as (t, x, y) where:
    - t: frame index (0 for first frame)
    - x: column (horizontal)
    - y: row (vertical)
    
    Args:
        keypoints_2d: (N, 2) in (x, y) = (col, row) format
        
    Returns:
        queries: (N, 3) in (t, x, y) format
    """
    N = len(keypoints_2d)
    queries = np.zeros((N, 3), dtype=np.float32)
    queries[:, 0] = 0  # t = 0 (first frame)
    queries[:, 1] = keypoints_2d[:, 0]  # x = col
    queries[:, 2] = keypoints_2d[:, 1]  # y = row
    return queries


# ============================================================================
# SPATRACKER PROCESSING
# ============================================================================

def run_spatracker_on_segment(
    segment_data: dict,
    queries: np.ndarray,
    model,
) -> tuple:
    """
    Run SpaTrackerV2 on a single segment.
    
    Args:
        segment_data: dict with video, depths, intrinsics, extrinsics
        queries: (N, 3) query points in (t, x, y) format
        model: SpaTrackerV2 Predictor
        
    Returns:
        tracks_2d: (T, N, 2) tracked 2D positions (x, y)
        tracks_3d: (T, N, 3) tracked 3D positions
        visibility: (T, N) visibility scores
    """
    video_tensor = torch.from_numpy(segment_data['video'] * 255)
    depth_tensor = segment_data['depths']  # Can be None for RGB-only mode
    intrs = segment_data['intrinsics']
    extrs = np.linalg.inv(segment_data['extrinsics'])
    
    T = video_tensor.shape[0]
    
    print(f"      Video: {video_tensor.shape}, range=[{video_tensor.min():.1f}, {video_tensor.max():.1f}]")
    if depth_tensor is not None:
        print(f"      Depth: shape={depth_tensor.shape}, range=[{depth_tensor.min():.3f}, {depth_tensor.max():.3f}] m")
    else:
        print(f"      Depth: None (RGB-only, SpaTracker will estimate)")
    
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs_out, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model.forward(
            video_tensor, 
            depth=depth_tensor,
            intrs=intrs, 
            extrs=extrs, 
            queries=queries,
            fps=10, 
            full_point=False, 
            iters_track=4,
            query_no_BA=True, 
            fixed_cam=True,  # Static camera
            stage=1, 
            unc_metric=None,
            support_frame=T - 1, 
            replace_ratio=0.1
        )
    
    track2d_np = track2d_pred.cpu().numpy()[:, :, :2]  # (T, N, 2)
    track3d_np = track3d_pred.cpu().numpy()[:, :, :3]  # (T, N, 3)
    vis_np = vis_pred.cpu().numpy().squeeze(-1)  # (T, N)
    
    return track2d_np, track3d_np, vis_np


# ============================================================================
# METRICS
# ============================================================================

def sample_points_on_edges(keypoints, edges, n_samples_per_edge=10):
    """
    Sample points along edges (line segments) for Chamfer distance.
    
    Args:
        keypoints: (N, 3) array of 3D keypoints
        edges: (E, 2) edge indices
        n_samples_per_edge: Number of samples per edge (default: 10)
        
    Returns:
        (M, 3) sampled point cloud
    """
    if keypoints is None or len(keypoints) == 0:
        return np.empty((0, 3), dtype=np.float32)
    
    sampled_points = []
    
    for i, j in edges:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        
        p1 = keypoints[i]
        p2 = keypoints[j]
        
        # Sample points along the edge
        for t in np.linspace(0, 1, n_samples_per_edge):
            pt = p1 * (1 - t) + p2 * t
            sampled_points.append(pt)
    
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
    
    # Filter NaN values
    pred_valid = pred_cloud[~np.any(np.isnan(pred_cloud), axis=1)]
    ref_valid = ref_cloud[~np.any(np.isnan(ref_cloud), axis=1)]
    
    if len(pred_valid) == 0 or len(ref_valid) == 0:
        return empty_result
    
    nn_ref = NearestNeighbors(n_neighbors=1).fit(ref_valid)
    pred2ref_dists, _ = nn_ref.kneighbors(pred_valid)
    pred2ref_dists = pred2ref_dists.flatten()
    
    nn_pred = NearestNeighbors(n_neighbors=1).fit(pred_valid)
    ref2pred_dists, _ = nn_pred.kneighbors(ref_valid)
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


def compute_edge_metrics(keypoints_3d, edges, reference_lengths):
    """Compute edge length metrics."""
    if keypoints_3d is None or len(keypoints_3d) == 0:
        return {'pct_mean': 0.0, 'rmse_mm': 0.0}
    
    # Filter NaN keypoints
    valid_mask = ~np.any(np.isnan(keypoints_3d), axis=1)
    
    pct_errors = []
    abs_errors = []
    
    for edge_idx, (i, j) in enumerate(edges):
        if i >= len(keypoints_3d) or j >= len(keypoints_3d):
            continue
        if not valid_mask[i] or not valid_mask[j]:
            continue
        ref_len = reference_lengths[edge_idx] if edge_idx < len(reference_lengths) else 0
        if ref_len > 1e-6:
            current_len = np.linalg.norm(keypoints_3d[i] - keypoints_3d[j])
            abs_err = abs(current_len - ref_len)
            pct_err = abs_err / ref_len
            pct_errors.append(pct_err)
            abs_errors.append(abs_err)
    
    if len(pct_errors) == 0:
        return {'pct_mean': 0.0, 'rmse_mm': 0.0}
    
    return {
        'pct_mean': np.mean(pct_errors) * 100,
        'rmse_mm': np.sqrt(np.mean(np.array(abs_errors) ** 2)),
    }


def compute_position_metrics(keypoints_3d, point_cloud):
    """Compute position RMSE (distance to nearest point)."""
    if keypoints_3d is None or len(keypoints_3d) == 0 or point_cloud is None or len(point_cloud) == 0:
        return {'rmse_mm': 0.0}
    
    # Filter NaN keypoints
    valid_mask = ~np.any(np.isnan(keypoints_3d), axis=1)
    keypoints_valid = keypoints_3d[valid_mask]
    
    if len(keypoints_valid) == 0:
        return {'rmse_mm': 0.0}
    
    nn = NearestNeighbors(n_neighbors=1).fit(point_cloud)
    distances, _ = nn.kneighbors(keypoints_valid)
    distances = distances.flatten()
    
    return {'rmse_mm': np.sqrt(np.mean(distances ** 2))}


def extract_point_cloud(mask, depth, intrinsics, max_points=5000):
    """Extract 3D point cloud from foreground mask."""
    if mask is None or depth is None:
        return np.zeros((0, 3), dtype=np.float32)
    
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    z_vals = depth[rows, cols].astype(np.float32)
    valid = z_vals > 0
    rows, cols, z_vals = rows[valid], cols[valid], z_vals[valid]
    
    if len(z_vals) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x_vals = (cols - cx) * z_vals / fx
    y_vals = (rows - cy) * z_vals / fy
    
    pc = np.column_stack([x_vals, y_vals, z_vals]).astype(np.float32)
    
    if len(pc) > max_points:
        indices = np.random.choice(len(pc), max_points, replace=False)
        pc = pc[indices]
    
    return pc


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualization_frame(
    frame: np.ndarray,
    keypoints_2d: np.ndarray,
    edge_connections: np.ndarray,
    traj_history: list,
    frame_idx: int,
    tail_length: int = 60,
    label: str = 'SpaTracker',
) -> np.ndarray:
    """Create visualization frame with keypoints, edges, and trajectory tails."""
    H, W = frame.shape[:2]
    
    # Start with RGB frame
    vis = frame.copy()
    
    # DLO color scheme: endpoints and interior
    ENDPOINT_COLOR = (255, 0, 0)    # Red for endpoints
    INTERIOR_COLOR = (0, 255, 0)    # Green for interior
    EDGE_COLOR = (0, 200, 0)        # Slightly darker green for edges
    TRAJ_COLOR = (255, 165, 0)      # Orange for trajectory
    
    # Draw trajectory tails
    start_idx = max(0, len(traj_history) - tail_length)
    for t in range(start_idx, len(traj_history) - 1):
        alpha = (t - start_idx) / max(1, len(traj_history) - start_idx - 1)
        color = tuple(int(c * (0.3 + 0.7 * alpha)) for c in TRAJ_COLOR)
        
        for kp_idx in range(len(traj_history[t])):
            pt1 = traj_history[t][kp_idx]
            pt2 = traj_history[t + 1][kp_idx]
            
            if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
                continue
            
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]), int(pt2[1])
            
            if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                cv2.line(vis, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
    
    # Draw edges
    if edge_connections is not None:
        for edge in edge_connections:
            i, j = edge
            if i < len(keypoints_2d) and j < len(keypoints_2d):
                pt1 = keypoints_2d[i]
                pt2 = keypoints_2d[j]
                
                if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
                    continue
                
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]), int(pt2[1])
                
                if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                    cv2.line(vis, (x1, y1), (x2, y2), EDGE_COLOR, 2, lineType=cv2.LINE_AA)
    
    # Draw keypoints
    n_keypoints = len(keypoints_2d)
    for kp_idx, (x, y) in enumerate(keypoints_2d):
        if np.isnan(x) or np.isnan(y):
            continue
        if 0 <= x < W and 0 <= y < H:
            # Endpoints (first and last) in red, others in green
            if kp_idx == 0 or kp_idx == n_keypoints - 1:
                color = ENDPOINT_COLOR
                radius = 7
            else:
                color = INTERIOR_COLOR
                radius = 5
            cv2.circle(vis, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    
    # Add text
    cv2.putText(vis, f'Frame: {frame_idx}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, label, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    return vis


def save_tracking_video(
    video_frames: np.ndarray,
    tracks_2d: np.ndarray,
    edge_connections: np.ndarray,
    output_path: Path,
    fps: int = 30,
    tail_length: int = 60,
    label: str = 'SpaTracker',
):
    """Save tracking visualization video."""
    T_video = video_frames.shape[0]
    T_tracks = tracks_2d.shape[0]
    T = min(T_video, T_tracks)
    
    H, W = video_frames.shape[1:3]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    
    traj_history = []
    
    for t in tqdm(range(T), desc="Creating video", leave=False):
        # Convert BGR to RGB for visualization
        rgb = video_frames[t][..., ::-1]
        keypoints_2d = tracks_2d[t]
        traj_history.append(keypoints_2d.copy())
        
        vis = create_visualization_frame(
            rgb, keypoints_2d, edge_connections,
            traj_history, t, tail_length, label,
        )
        
        writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    writer.release()


# ============================================================================
# CLIP PROCESSING
# ============================================================================

def process_clip(
    chunk_data: dict,
    eval_results_dir: Path,
    output_dir: Path,
    clip_idx: int,
    start_frame: int,
    end_frame: int,
    intrinsics: np.ndarray,
    model,
    fps: int = 30,
    tail_length: int = 60,
    segment_seconds: int = 5,
    use_mask: bool = True,
    rgb_only: bool = False,
):
    """
    Process a single clip with SpaTrackerV2 using chained tracking.
    """
    clip_output_dir = output_dir / f'clip_{clip_idx}'
    clip_output_dir.mkdir(parents=True, exist_ok=True)
    
    n_frames = end_frame - start_frame
    
    # If segment_seconds is 0 or >= clip duration, use full clip (no splitting)
    clip_duration_seconds = n_frames / fps
    if segment_seconds <= 0 or segment_seconds >= clip_duration_seconds:
        frames_per_segment = n_frames
        n_segments = 1
        print(f"\n  Clip {clip_idx}: frames {start_frame}-{end_frame} ({n_frames} frames)")
        print(f"    No segment splitting (full clip tracking)")
    else:
        frames_per_segment = segment_seconds * fps
        n_segments = (n_frames + frames_per_segment - 1) // frames_per_segment
        print(f"\n  Clip {clip_idx}: frames {start_frame}-{end_frame} ({n_frames} frames)")
        print(f"    Chained tracking: {n_segments} segments of {segment_seconds}s")
    
    # Extract clip data
    clip_color = chunk_data['color'][start_frame:end_frame]
    clip_depth = chunk_data['depth'][start_frame:end_frame]
    
    # Always load mask for evaluation (foreground point cloud extraction)
    clip_mask_eval = chunk_data.get('dlo_mask', None)
    if clip_mask_eval is not None:
        clip_mask_eval = clip_mask_eval[start_frame:end_frame]
    
    # Mask for SpaTracker input (can be disabled with --no-mask)
    clip_mask_input = None
    if use_mask and not rgb_only:
        clip_mask_input = clip_mask_eval
        print(f"    Mask: ENABLED for SpaTracker input")
    elif not use_mask:
        print(f"    Mask: DISABLED for SpaTracker input (still used for evaluation)")
    
    if rgb_only:
        print(f"    Mode: RGB-only (SpaTracker will estimate depth)")
    
    # Load initial keypoints
    print(f"    Loading initial keypoints from evaluation results...")
    keypoints_3d, edge_connections, reference_lengths = load_initial_keypoints(eval_results_dir, clip_idx)
    
    initial_keypoints_2d = project_3d_to_2d(keypoints_3d, intrinsics)
    
    print(f"    Keypoints: {keypoints_3d.shape[0]} (branched Y-shape)")
    print(f"    Edges: {len(edge_connections)}")
    
    # Initialize accumulators
    all_tracks_2d = []
    all_tracks_3d = []
    all_visibility = []
    all_metrics = []
    
    # Current query points
    current_keypoints_2d = initial_keypoints_2d.copy()
    current_keypoints_3d = keypoints_3d.copy()
    
    # Process each segment
    for seg_idx in range(n_segments):
        seg_start = seg_idx * frames_per_segment
        seg_end = min(seg_start + frames_per_segment, n_frames)
        seg_len = seg_end - seg_start
        
        print(f"    Segment {seg_idx+1}/{n_segments}: frames {seg_start}-{seg_end} ({seg_len} frames)")
        
        # Extract segment data
        seg_color = clip_color[seg_start:seg_end]
        seg_depth = clip_depth[seg_start:seg_end]
        seg_mask_input = clip_mask_input[seg_start:seg_end] if clip_mask_input is not None else None
        seg_mask_eval = clip_mask_eval[seg_start:seg_end] if clip_mask_eval is not None else None
        
        # Convert to SpaTrackerV2 format (uses input mask for depth masking)
        segment_data = convert_to_spatracker_format(
            seg_color, seg_depth, intrinsics, mask=seg_mask_input, rgb_only=rgb_only
        )
        
        # Create queries
        queries = create_queries_from_keypoints_2d(current_keypoints_2d)
        
        # Run tracking
        tracks_2d_seg, tracks_3d_seg, vis_seg = run_spatracker_on_segment(
            segment_data, queries, model
        )
        
        # SpaTracker outputs 3D in meters, convert to mm
        tracks_3d_seg = tracks_3d_seg * 1000.0
        
        # Log zero-depth keypoints
        zero_mask = np.all(np.abs(tracks_3d_seg) < 1e-3, axis=-1)  # (T, N)
        n_zero_per_frame = zero_mask.sum(axis=1)
        n_zero_total = zero_mask.sum()
        if n_zero_total > 0:
            print(f"      WARNING: {n_zero_total} keypoint-frames at (0,0,0) (avg {n_zero_per_frame.mean():.1f}/frame)")
        
        # Force frame 0 to match query positions
        tracks_2d_seg[0] = current_keypoints_2d
        tracks_3d_seg[0] = current_keypoints_3d
        
        # Compute per-frame metrics for this segment
        for local_frame in range(seg_len):
            global_frame = seg_start + local_frame
            kp_3d = tracks_3d_seg[local_frame]
            
            # Extract point cloud for this frame (always use eval mask for GT)
            frame_mask = seg_mask_eval[local_frame] if seg_mask_eval is not None else None
            frame_depth = seg_depth[local_frame]
            point_cloud = extract_point_cloud(frame_mask, frame_depth, intrinsics)
            
            edge_m = compute_edge_metrics(kp_3d, edge_connections, reference_lengths)
            pos_m = compute_position_metrics(kp_3d, point_cloud)
            
            # Compute Chamfer metrics: sample points along edges
            n_gt_points = len(point_cloud) if point_cloud is not None else 500
            n_samples = max(10, n_gt_points // len(edge_connections))
            pred_cloud = sample_points_on_edges(kp_3d, edge_connections, n_samples_per_edge=n_samples)
            cd_m = compute_chamfer_metrics(pred_cloud, point_cloud)
            
            all_metrics.append({
                'frame': global_frame,
                'edge_pct_mean': edge_m['pct_mean'],
                'edge_rmse_mm': edge_m['rmse_mm'],
                'pos_rmse_mm': pos_m['rmse_mm'],
                'cd_mm': cd_m['cd'],
                'precision_2mm': cd_m['precision_2mm'],
                'precision_5mm': cd_m['precision_5mm'],
                'precision_10mm': cd_m['precision_10mm'],
                'recall_2mm': cd_m['recall_2mm'],
                'recall_5mm': cd_m['recall_5mm'],
                'recall_10mm': cd_m['recall_10mm'],
                'f_2mm': cd_m['f_2mm'],
                'f_5mm': cd_m['f_5mm'],
                'f_10mm': cd_m['f_10mm'],
            })
        
        # Accumulate tracks
        if seg_idx == 0:
            all_tracks_2d.append(tracks_2d_seg)
            all_tracks_3d.append(tracks_3d_seg)
            all_visibility.append(vis_seg)
        else:
            # Skip first frame (same as last of previous)
            all_tracks_2d.append(tracks_2d_seg[1:])
            all_tracks_3d.append(tracks_3d_seg[1:])
            all_visibility.append(vis_seg[1:])
        
        # Update for next segment
        current_keypoints_2d = tracks_2d_seg[-1].copy()
        current_keypoints_3d = tracks_3d_seg[-1].copy()
    
    # Concatenate all segments
    tracks_2d = np.concatenate(all_tracks_2d, axis=0)
    tracks_3d = np.concatenate(all_tracks_3d, axis=0)
    visibility = np.concatenate(all_visibility, axis=0)
    
    print(f"    Final tracked: {tracks_2d.shape[0]} frames")
    
    # Save keypoints
    npz_path = clip_output_dir / 'keypoints_spatracker.npz'
    np.savez(
        npz_path,
        keypoints_2d=tracks_2d,
        keypoints_3d=tracks_3d,
        visibility=visibility,
        edge_connections=edge_connections,
        reference_lengths=reference_lengths,
        initial_keypoints_2d=initial_keypoints_2d,
        initial_keypoints_3d=keypoints_3d,
    )
    print(f"    Saved: {npz_path}")
    
    # Save per-frame CSV with all metrics
    csv_path = clip_output_dir / 'per_frame.csv'
    with open(csv_path, 'w') as f:
        f.write('Frame,EdgePctMean,EdgeRMSE_mm,PosRMSE_mm,CD_mm,Prec_2mm,Prec_5mm,Prec_10mm,Rec_2mm,Rec_5mm,Rec_10mm,F_2mm,F_5mm,F_10mm\n')
        for m in all_metrics:
            f.write(f"{m['frame']},{m['edge_pct_mean']:.4f},{m['edge_rmse_mm']:.4f},{m['pos_rmse_mm']:.4f},"
                    f"{m['cd_mm']:.4f},{m['precision_2mm']:.4f},{m['precision_5mm']:.4f},{m['precision_10mm']:.4f},"
                    f"{m['recall_2mm']:.4f},{m['recall_5mm']:.4f},{m['recall_10mm']:.4f},"
                    f"{m['f_2mm']:.4f},{m['f_5mm']:.4f},{m['f_10mm']:.4f}\n")
    print(f"    Saved: {csv_path}")
    
    # Extract all metric arrays
    edge_pct_means = [m['edge_pct_mean'] for m in all_metrics]
    edge_rmses = [m['edge_rmse_mm'] for m in all_metrics]
    pos_rmses = [m['pos_rmse_mm'] for m in all_metrics]
    cd_mms = [m['cd_mm'] for m in all_metrics]
    prec_2mm = [m['precision_2mm'] for m in all_metrics]
    prec_5mm = [m['precision_5mm'] for m in all_metrics]
    prec_10mm = [m['precision_10mm'] for m in all_metrics]
    rec_2mm = [m['recall_2mm'] for m in all_metrics]
    rec_5mm = [m['recall_5mm'] for m in all_metrics]
    rec_10mm = [m['recall_10mm'] for m in all_metrics]
    f_2mm = [m['f_2mm'] for m in all_metrics]
    f_5mm = [m['f_5mm'] for m in all_metrics]
    f_10mm = [m['f_10mm'] for m in all_metrics]
    
    # Compute edge% < 5% rate
    edge_under_5pct = np.mean(np.array(edge_pct_means) < 5.0) * 100
    # Compute pos RMSE < 5mm rate
    pos_under_5mm = np.mean(np.array(pos_rmses) < 5.0) * 100
    
    # Save summary
    summary_path = clip_output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"SpaTrackerV2 BDLO Tracking - Clip {clip_idx}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Frames: {n_frames}\n")
        f.write(f"Keypoints: {len(initial_keypoints_2d)} (branched Y-shape)\n")
        f.write(f"Edges: {len(edge_connections)}\n\n")
        
        f.write("Edge Length Metrics:\n")
        f.write(f"  Edge % Mean:  {np.mean(edge_pct_means):6.2f}% ± {np.std(edge_pct_means):5.2f}%\n")
        f.write(f"  Edge RMSE:    {np.mean(edge_rmses):6.2f} ± {np.std(edge_rmses):5.2f} mm\n")
        f.write(f"  Edge <5%:     {edge_under_5pct:6.1f}%\n\n")
        
        f.write("Position Metrics:\n")
        f.write(f"  Pos RMSE:     {np.mean(pos_rmses):6.2f} ± {np.std(pos_rmses):5.2f} mm\n")
        f.write(f"  Pos <5mm:     {pos_under_5mm:6.1f}%\n\n")
        
        f.write("Chamfer Distance Metrics:\n")
        f.write(f"  CD:           {np.mean(cd_mms):6.2f} ± {np.std(cd_mms):5.2f} mm\n")
        f.write(f"  F@10mm:       {np.mean(f_10mm):6.1f}%\n\n")
        
        f.write("Precision/Recall/F-Score:\n")
        f.write(f"  Prec@2mm:     {np.mean(prec_2mm):6.1f}%\n")
        f.write(f"  Prec@5mm:     {np.mean(prec_5mm):6.1f}%\n")
        f.write(f"  Prec@10mm:    {np.mean(prec_10mm):6.1f}%\n")
        f.write(f"  Rec@2mm:      {np.mean(rec_2mm):6.1f}%\n")
        f.write(f"  Rec@5mm:      {np.mean(rec_5mm):6.1f}%\n")
        f.write(f"  Rec@10mm:     {np.mean(rec_10mm):6.1f}%\n")
        f.write(f"  F@2mm:        {np.mean(f_2mm):6.1f}%\n")
        f.write(f"  F@5mm:        {np.mean(f_5mm):6.1f}%\n")
        f.write(f"  F@10mm:       {np.mean(f_10mm):6.1f}%\n")
    print(f"    Saved: {summary_path}")
    
    # Save visualization video
    video_path = clip_output_dir / 'tracking_spatracker.mp4'
    print(f"    Creating visualization video...")
    save_tracking_video(
        clip_color, tracks_2d, edge_connections,
        video_path, fps=fps, tail_length=tail_length,
    )
    print(f"    Saved: {video_path}")
    
    return {
        'clip_idx': clip_idx,
        'n_frames': n_frames,
        'edge_pct_mean': np.mean(edge_pct_means),
        'edge_pct_std': np.std(edge_pct_means),
        'edge_rmse_mm': np.mean(edge_rmses),
        'edge_rmse_std': np.std(edge_rmses),
        'edge_under_5pct': edge_under_5pct,
        'pos_rmse_mm': np.mean(pos_rmses),
        'pos_rmse_std': np.std(pos_rmses),
        'pos_under_5mm': pos_under_5mm,
        'cd_mm': np.mean(cd_mms),
        'cd_std': np.std(cd_mms),
        'precision_2mm': np.mean(prec_2mm),
        'precision_5mm': np.mean(prec_5mm),
        'precision_10mm': np.mean(prec_10mm),
        'recall_2mm': np.mean(rec_2mm),
        'recall_5mm': np.mean(rec_5mm),
        'recall_10mm': np.mean(rec_10mm),
        'f_2mm': np.mean(f_2mm),
        'f_5mm': np.mean(f_5mm),
        'f_10mm': np.mean(f_10mm),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SpaTrackerV2 baseline for BDLO tracking')
    parser.add_argument('--dataset', type=str, required=True, choices=['4sec', '2sec'],
                        help='Dataset: 4sec (slower motion, 49 chunks) or 2sec (faster motion, 20 chunks)')
    parser.add_argument('--chunk', type=int, required=True, help='Chunk index')
    parser.add_argument('--clip_seconds', type=int, default=None,
                        help='Clip duration in seconds (default: 10)')
    parser.add_argument('--fps', type=int, default=FPS, help='Frame rate (default: 30)')
    parser.add_argument('--tail_length', type=int, default=60,
                        help='Trajectory tail length in frames (default: 60)')
    parser.add_argument('--segment_seconds', type=int, default=10,
                        help='Tracking segment duration for chained tracking (default: 5)')
    parser.add_argument('--no-mask', action='store_true',
                        help='Disable foreground masking (use raw images)')
    parser.add_argument('--rgb-only', action='store_true',
                        help='RGB-only mode: let SpaTracker estimate depth (no RGBD)')
    args = parser.parse_args()
    
    # Get dataset config
    config = DATASET_CONFIG[args.dataset]
    data_dir = config['data_dir']
    eval_results_base = config['eval_results_dir']
    output_base = config['output_dir']
    n_chunks = config['n_chunks']
    
    # Set clip duration
    clip_seconds = args.clip_seconds if args.clip_seconds else config['clip_seconds_default']
    
    # Validate chunk index
    if args.chunk < 0 or args.chunk >= n_chunks:
        print(f"ERROR: Chunk {args.chunk} is out of range for dataset {args.dataset} (0-{n_chunks - 1})")
        return
    
    chunk_dir = data_dir / f'chunk_{args.chunk}'
    eval_results_dir = eval_results_base / f'chunk_{args.chunk}'
    output_dir = output_base / f'chunk_{args.chunk}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"SPATRACKERV2 BDLO TRACKING - Dataset: {args.dataset} ({config['description']})")
    print(f"Chunk {args.chunk} / {n_chunks - 1}")
    print("=" * 80)
    
    # Load calibration
    print(f"\nLoading calibration from {CALIB_DIR}...")
    calib = load_calibration(CALIB_DIR)
    intrinsics = calib['intrinsics']
    print(f"  Intrinsics: fx={intrinsics[0,0]:.1f}, fy={intrinsics[1,1]:.1f}, cx={intrinsics[0,2]:.1f}, cy={intrinsics[1,2]:.1f}")
    
    # Load SpaTrackerV2 model
    print(f"\nLoading SpaTrackerV2-Offline model...")
    from models.SpaTrackV2.models.predictor import Predictor
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    model.eval()
    model.to("cuda")
    print(f"  Model loaded on: cuda")
    
    # Load chunk data (BDLO uses last 600 frames)
    print(f"\nLoading chunk data from {chunk_dir}...")
    chunk_data = load_chunk_data(chunk_dir)
    total_frames = chunk_data['n_frames']
    
    # Calculate clips
    frames_per_clip = clip_seconds * args.fps
    n_clips = (total_frames + frames_per_clip - 1) // frames_per_clip
    
    print(f"\nClip configuration:")
    print(f"  Clip duration: {clip_seconds}s ({frames_per_clip} frames)")
    print(f"  Number of clips: {n_clips}")
    print(f"  Segment duration: {args.segment_seconds}s (chained tracking)")
    print(f"  Use mask: {not getattr(args, 'no_mask', False)}")
    print(f"  RGB-only mode: {getattr(args, 'rgb_only', False)}")
    
    # Process each clip
    all_results = []
    for clip_idx in range(n_clips):
        start_frame = clip_idx * frames_per_clip
        end_frame = min(start_frame + frames_per_clip, total_frames)
        
        # Check if evaluation results exist
        clip_eval_dir = eval_results_dir / f'clip_{clip_idx}'
        if not clip_eval_dir.exists():
            print(f"\n  Clip {clip_idx}: Skipping (no evaluation results at {clip_eval_dir})")
            continue
        
        kp_path = clip_eval_dir / '3d_keypoints.npz'
        if not kp_path.exists():
            print(f"\n  Clip {clip_idx}: Skipping (no 3d_keypoints.npz)")
            continue
        
        result = process_clip(
            chunk_data=chunk_data,
            eval_results_dir=eval_results_dir,
            output_dir=output_dir,
            clip_idx=clip_idx,
            start_frame=start_frame,
            end_frame=end_frame,
            intrinsics=intrinsics,
            model=model,
            fps=args.fps,
            tail_length=args.tail_length,
            segment_seconds=args.segment_seconds,
            use_mask=not getattr(args, 'no_mask', False),
            rgb_only=getattr(args, 'rgb_only', False),
        )
        all_results.append(result)
    
    # Print and save summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Dataset: {args.dataset} ({config['description']})")
    print(f"Chunk: {args.chunk}")
    print(f"Processed {len(all_results)} clips")
    
    # Print per-clip summary
    print(f"\n{'Clip':<6} | {'Frames':<8} | {'Edge%':<10} | {'EdgeRMSE':<12} | {'PosRMSE':<12} | {'CD':<10} | {'F@10mm':<10}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['clip_idx']:<6} | {r['n_frames']:<8} | {r['edge_pct_mean']:<10.2f} | {r['edge_rmse_mm']:<12.2f} | {r['pos_rmse_mm']:<12.2f} | {r['cd_mm']:<10.2f} | {r['f_10mm']:<10.1f}")
    
    if all_results:
        # Compute weighted averages (by number of frames)
        total_frames = sum(r['n_frames'] for r in all_results)
        
        def weighted_avg(key):
            return sum(r[key] * r['n_frames'] for r in all_results) / total_frames
        
        avg_edge_pct = weighted_avg('edge_pct_mean')
        avg_edge_rmse = weighted_avg('edge_rmse_mm')
        avg_pos_rmse = weighted_avg('pos_rmse_mm')
        avg_cd = weighted_avg('cd_mm')
        avg_f_10mm = weighted_avg('f_10mm')
        
        print("-" * 90)
        print(f"{'AVG':<6} | {total_frames:<8} | {avg_edge_pct:<10.2f} | {avg_edge_rmse:<12.2f} | {avg_pos_rmse:<12.2f} | {avg_cd:<10.2f} | {avg_f_10mm:<10.1f}")
        
        # Save chunk summary file
        chunk_summary_path = output_dir / 'chunk_summary.txt'
        with open(chunk_summary_path, 'w') as f:
            f.write(f"SpaTrackerV2 BDLO Tracking - Dataset: {args.dataset}, Chunk {args.chunk}\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Total Frames: {total_frames}\n")
            f.write(f"Number of Clips: {len(all_results)}\n\n")
            
            f.write("FRAME-WEIGHTED SUMMARY\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Method':<12} | {'Edge%':<16} | {'EdgeRMSE (mm)':<16} | {'<5%':<8} | {'PosRMSE (mm)':<16} | {'<5mm':<8} | {'CD (mm)':<16} | {'F@10mm':<8}\n")
            f.write("-" * 100 + "\n")
            
            f.write(f"{'SpaTracker':<12} | {weighted_avg('edge_pct_mean'):5.2f} ± {weighted_avg('edge_pct_std'):5.2f}%  | "
                    f"{weighted_avg('edge_rmse_mm'):5.2f} ± {weighted_avg('edge_rmse_std'):5.2f} mm | "
                    f"{weighted_avg('edge_under_5pct'):5.1f}%  | "
                    f"{weighted_avg('pos_rmse_mm'):5.2f} ± {weighted_avg('pos_rmse_std'):5.2f} mm | "
                    f"{weighted_avg('pos_under_5mm'):5.1f}%  | "
                    f"{weighted_avg('cd_mm'):5.2f} ± {weighted_avg('cd_std'):5.2f} mm | "
                    f"{weighted_avg('f_10mm'):5.1f}%\n")
            
            f.write("\nPrecision/Recall/F-Score:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Method':<12} | {'Prec@2mm':<10} | {'Prec@5mm':<10} | {'Prec@10mm':<10} | {'Rec@2mm':<10} | {'Rec@5mm':<10} | {'Rec@10mm':<10} | {'F@2mm':<8} | {'F@5mm':<8} | {'F@10mm':<8}\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'SpaTracker':<12} | {weighted_avg('precision_2mm'):8.1f}% | {weighted_avg('precision_5mm'):8.1f}% | {weighted_avg('precision_10mm'):8.1f}% | "
                    f"{weighted_avg('recall_2mm'):8.1f}% | {weighted_avg('recall_5mm'):8.1f}% | {weighted_avg('recall_10mm'):8.1f}% | "
                    f"{weighted_avg('f_2mm'):6.1f}% | {weighted_avg('f_5mm'):6.1f}% | {weighted_avg('f_10mm'):6.1f}%\n")
        
        print(f"\nChunk summary saved to: {chunk_summary_path}")
    
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
