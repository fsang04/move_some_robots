"""
CDCPD2 Benchmark Script for Fabric Tracking

Compares CDCPD2 tracker against FabricTracker.
Uses the same metrics as fabric_tracker_ablation.py.

CDCPD2 is applied to fabric tracking by:
1. Using FabricTracker for initialization (same init as baseline)
2. Using CDCPD for subsequent frames with:
   - EE corner positions as anchor constraints (hard constraints in QP)
   - Edge length constraints from 5x5 grid topology
   - LLE regularization for smooth deformation

Usage:
    python fabric_tracking_cdcpd_benchmark.py
    python fabric_tracking_cdcpd_benchmark.py --n_frames 100

Author: Auto-generated
Date: 2026-02-23
"""

import numpy as np
import cv2
from pathlib import Path
import time
import argparse
import gc
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wire_tracking_cdcpd import CDCPDTracker
from fabric_tracker import FabricTracker


# ============================================================================
# GRID CONSTANTS (same as fabric_tracker.py)
# ============================================================================

GRID_ROWS = 5
GRID_COLS = 5
N_KEYPOINTS = GRID_ROWS * GRID_COLS  # 25

CORNER_INDICES = [0, 4, 20, 24]
BORDER_INDICES = [1, 2, 3, 5, 9, 10, 14, 15, 19, 21, 22, 23]
INTERIOR_INDICES = [6, 7, 8, 11, 12, 13, 16, 17, 18]

# Start frame (skip initial frames where fabric may not be fully visible)
START_FRAME = 18


# ============================================================================
# CAMERA INTRINSICS
# ============================================================================

INTRINSICS = np.array([
    [606.1124267578125, 0, 641.7578125],
    [0, 605.8821411132812, 365.6518859863281],
    [0, 0, 1]
], dtype=np.float64)


# ============================================================================
# METRICS (same as fabric_tracker_ablation.py)
# ============================================================================

def compute_edge_metrics(keypoints: np.ndarray, edges: list, reference_lengths: dict) -> dict:
    """Compute comprehensive edge length metrics."""
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
        current_length = np.linalg.norm(keypoints[i] - keypoints[j])
        ref_length = reference_lengths.get((i, j), current_length)
        if ref_length > 1e-6:
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


def compute_position_metrics(keypoints: np.ndarray, point_cloud: np.ndarray) -> dict:
    """Compute position accuracy metrics."""
    if keypoints is None or len(keypoints) == 0 or point_cloud is None or len(point_cloud) == 0:
        return {
            'distances': np.array([]),
            'rmse_mm': 0.0,
            'under_2mm': 0.0, 'under_5mm': 0.0, 'under_10mm': 0.0,
        }
    
    nn = NearestNeighbors(n_neighbors=1).fit(point_cloud)
    distances, _ = nn.kneighbors(keypoints)
    distances = distances.flatten()
    
    return {
        'distances': distances,
        'rmse_mm': np.sqrt(np.mean(distances ** 2)),
        'under_2mm': np.mean(distances < 2.0) * 100,
        'under_5mm': np.mean(distances < 5.0) * 100,
        'under_10mm': np.mean(distances < 10.0) * 100,
    }


def create_frame_visualization(
    rgb: np.ndarray,
    mask: np.ndarray,
    keypoints_2d: np.ndarray,
    edges: list,
    method_name: str,
    metrics: dict,
    frame_idx: int,
    traj_history: np.ndarray = None,
    tail_length: int = 60,
) -> np.ndarray:
    """Create visualization for a single method with trajectory tails."""
    H, W = rgb.shape[:2]
    
    # Colors
    MASK_COLOR = [0, 255, 0]        # Green for mask
    EDGE_COLOR = [255, 165, 0]      # Orange for edges
    CORNER_COLOR = [255, 0, 0]      # Red for corners (EE-held)
    BORDER_COLOR = [255, 255, 0]    # Yellow for border
    INTERIOR_COLOR = [0, 255, 255]  # Cyan for interior
    TAIL_COLOR = [255, 105, 180]    # Hot pink for trajectory tail
    
    KEYPOINT_RADIUS = 6
    EDGE_THICKNESS = 2
    TAIL_THICKNESS = 2
    
    vis = rgb.copy()
    
    # Draw mask contour
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(vis, contours, -1, MASK_COLOR, 2)
    
    # Draw trajectory tails
    if traj_history is not None and len(traj_history) > 1:
        n_history = len(traj_history)
        n_keypoints = traj_history.shape[1] if len(traj_history.shape) > 1 else 0
        
        for k_idx in range(n_keypoints):
            start_idx = max(0, n_history - tail_length)
            
            for t in range(start_idx, n_history - 1):
                pt1 = traj_history[t, k_idx]
                pt2 = traj_history[t + 1, k_idx]
                
                if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
                    continue
                if np.any(np.isinf(pt1)) or np.any(np.isinf(pt2)):
                    continue
                
                row1, col1 = int(pt1[0]), int(pt1[1])
                row2, col2 = int(pt2[0]), int(pt2[1])
                
                if not (0 <= row1 < H and 0 <= col1 < W):
                    continue
                if not (0 <= row2 < H and 0 <= col2 < W):
                    continue
                
                age = n_history - 1 - t
                alpha = max(0.2, 1.0 - age / tail_length)
                color = [int(c * alpha) for c in TAIL_COLOR]
                
                cv2.line(vis, (col1, row1), (col2, row2), color, TAIL_THICKNESS)
    
    # Draw edges and keypoints
    if keypoints_2d is not None and len(keypoints_2d) > 0 and edges is not None:
        kp_int = keypoints_2d.astype(int)
        for (i, j) in edges:
            if i < len(kp_int) and j < len(kp_int):
                pt1 = (kp_int[i, 1], kp_int[i, 0])
                pt2 = (kp_int[j, 1], kp_int[j, 0])
                cv2.line(vis, pt1, pt2, EDGE_COLOR, EDGE_THICKNESS)
        
        for idx, (row, col) in enumerate(kp_int):
            if 0 <= row < H and 0 <= col < W:
                if idx in CORNER_INDICES:
                    color = CORNER_COLOR
                elif idx in BORDER_INDICES:
                    color = BORDER_COLOR
                else:
                    color = INTERIOR_COLOR
                cv2.circle(vis, (col, row), KEYPOINT_RADIUS, color, -1)
                cv2.circle(vis, (col, row), KEYPOINT_RADIUS + 1, (255, 255, 255), 1)
    
    # Add text
    cv2.putText(vis, method_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, f"Frame: {frame_idx}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y_offset = 90
    if 'edge_pct_mean' in metrics:
        text = f"Edge: {metrics['edge_pct_mean']:.1f}% (max {metrics['edge_pct_max']:.1f}%)"
        cv2.putText(vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
    if 'pos_rmse_mm' in metrics:
        text = f"Pos RMSE: {metrics['pos_rmse_mm']:.2f} mm"
        cv2.putText(vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis


# ============================================================================
# CDCPD FABRIC TRACKER WRAPPER
# ============================================================================

class CDCPDFabricTracker:
    """
    Wrapper around CDCPDTracker for fabric tracking.
    
    Uses FabricTracker for initialization, then CDCPD for subsequent tracking.
    Uses EE poses as anchor constraints (like gripper constraints in CDCPD2 paper).
    """
    
    def __init__(self, intrinsics, ee_poses_3d=None, **kwargs):
        self.intrinsics = intrinsics
        self.ee_poses_3d = ee_poses_3d  # (n_frames, 2, 3) EE positions
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.cx = intrinsics[0, 2]
        self.cy = intrinsics[1, 2]
        
        # FabricTracker for initialization (same init as baseline)
        self.init_tracker = FabricTracker(
            intrinsics=intrinsics,
            ee_poses_3d=ee_poses_3d,
            **kwargs
        )
        
        # CDCPD tracker
        # Use parameters tuned for fabric scale:
        # - beta: motion coherence width (~grid spacing ~100mm)
        # - lambda: very low because it gets multiplied by sigma2
        # - anchor_hard=True for hard gripper constraints in QP
        self.cdcpd = CDCPDTracker(
            cpd_beta=50.0,       # Match fabric grid spacing
            cpd_lambda=0.1,      # Low - sigma2 amplifies this
            cpd_w=0.1,           # Outlier weight
            cpd_max_iter=200,
            cpd_tol=1e-4,
            lle_neighbors=4,     # Fabric has 4-connected neighbors
            lle_gamma=0.5,
            stretch_lambda=1.05,  # Allow 5% stretch for fabric (inextensible)
            use_qp_optimization=True,
            qp_max_iter=200,
            use_anchor_constraints=True,
            anchor_weight=100.0,  # Soft constraint weight in CPD
            anchor_hard=True,     # HARD constraints in QP
        )
        
        self.is_initialized = False
        self.prev_keypoints = None
        self.reference_keypoints = None
        self.reference_lengths = None
        self.grid_edges = None
        self.frame_count = 0
        self.ee_corner_indices = None  # Which corners correspond to EE
    
    def process_frame(self, depth: np.ndarray, mask: np.ndarray, frame_idx: int) -> dict:
        """Process a single frame using CDCPD with EE poses as anchors."""
        
        self.frame_count = frame_idx
        
        # First frame: use FabricTracker for initialization
        if not self.is_initialized:
            result = self.init_tracker.process_frame(depth, mask, frame_idx=frame_idx)
            
            if result['success'] and self.init_tracker.is_initialized:
                self.is_initialized = True
                self.prev_keypoints = self.init_tracker.prev_keypoints.copy()
                self.reference_keypoints = self.init_tracker.reference_keypoints.copy()
                self.reference_lengths = self.init_tracker.reference_lengths.copy()
                self.grid_edges = self.init_tracker.grid_edges
                
                # Get EE-to-corner mapping from init_tracker
                # init_tracker.ee_to_corner_mapping is {0: corner_idx, 1: corner_idx}
                if self.init_tracker.ee_to_corner_mapping is not None:
                    self.ee_corner_indices = [
                        self.init_tracker.ee_to_corner_mapping[0],
                        self.init_tracker.ee_to_corner_mapping[1]
                    ]
                else:
                    # Default: corners 0 (top-left) and 4 (top-right)
                    self.ee_corner_indices = [0, 4]
            
            return result
        
        # Subsequent frames: use CDCPD
        # Extract point cloud for CPD target
        point_cloud = self._extract_point_cloud(mask, depth)
        
        if len(point_cloud) < 500:
            return {
                'success': False,
                'reason': 'insufficient_points',
            }
        
        # Get EE poses for this frame as anchor constraints
        anchor_indices = []
        anchor_positions = []
        
        if self.ee_poses_3d is not None and frame_idx < len(self.ee_poses_3d):
            ee_pos = self.ee_poses_3d[frame_idx]  # (2, 3)
            for i, corner_idx in enumerate(self.ee_corner_indices):
                anchor_indices.append(corner_idx)
                anchor_positions.append(ee_pos[i])
        
        anchor_indices = np.array(anchor_indices) if anchor_indices else None
        anchor_positions = np.array(anchor_positions) if anchor_positions else None
        
        # Convert reference_lengths dict to list format for CDCPD
        ref_lengths_list = []
        for (i, j) in self.grid_edges:
            ref_lengths_list.append(self.reference_lengths.get((i, j), 100.0))
        ref_lengths_array = np.array(ref_lengths_list)
        
        # Run CDCPD tracking with EE anchors
        t_start = time.time()
        cdcpd_result = self.cdcpd.track_frame_with_anchors(
            prev_keypoints=self.prev_keypoints,
            skeleton_pc=point_cloud,
            anchor_indices=anchor_indices,
            anchor_positions=anchor_positions,
            reference_edges=self.grid_edges,
            reference_lengths=ref_lengths_array,
            cpd_downsample=3000,
        )
        track_time = time.time() - t_start
        
        keypoints = cdcpd_result['keypoints']
        self.prev_keypoints = keypoints.copy()
        
        # Project to 2D
        keypoints_2d = self._project_3d_to_2d(keypoints)
        
        return {
            'success': True,
            'mode': 'track',
            'keypoints': keypoints,
            'keypoints_2d': keypoints_2d,
            'edges': self.grid_edges,
            'timing': cdcpd_result.get('timing', {'total': track_time}),
        }
    
    def _extract_point_cloud(self, mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Extract 3D point cloud from mask and depth."""
        ys, xs = np.where(mask > 0)
        
        if len(xs) == 0:
            return np.empty((0, 3))
        
        zs = depth[ys, xs]
        valid = zs > 0
        xs, ys, zs = xs[valid], ys[valid], zs[valid]
        
        if len(xs) == 0:
            return np.empty((0, 3))
        
        X = (xs - self.cx) * zs / self.fx
        Y = (ys - self.cy) * zs / self.fy
        Z = zs
        
        return np.column_stack([X, Y, Z])
    
    def _project_3d_to_2d(self, keypoints_3d: np.ndarray) -> np.ndarray:
        """Project 3D keypoints to 2D image coordinates (row, col)."""
        if keypoints_3d is None or len(keypoints_3d) == 0:
            return np.array([])
        
        keypoints_2d = []
        for pt in keypoints_3d:
            x, y, z = pt
            if z > 0:
                col = self.fx * x / z + self.cx
                row = self.fy * y / z + self.cy
            else:
                col, row = 0, 0
            keypoints_2d.append([row, col])
        
        return np.array(keypoints_2d)


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark(args):
    """Run CDCPD vs FabricTracker benchmark."""
    
    # Paths
    data_path = Path("/home/yehengz/deformable_seg/data/full/tracking_fabric2_data.npy")
    masks_dir = Path("/home/yehengz/deformable_seg/data/arm_traj4_fabric/masks")
    ee_pose_path = Path("/home/yehengz/deformable_seg/data/arm_traj4_fabric/ee_pose_output/ee_poses_3d.npy")
    output_dir = Path("/home/yehengz/deformable_seg/data/arm_traj4_fabric/cdcpd_benchmark")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load EE poses
    ee_poses_3d = None
    if ee_pose_path.exists():
        ee_data = np.load(str(ee_pose_path), allow_pickle=True).item()
        ee_poses_3d_full = ee_data['ee_3d']
        # Slice EE poses to start from START_FRAME
        ee_poses_3d = ee_poses_3d_full[START_FRAME:]
        print(f"Loaded EE poses from: {ee_pose_path}")
        print(f"  Full shape: {ee_poses_3d_full.shape}, Sliced shape: {ee_poses_3d.shape}")
    else:
        print(f"No EE poses found at: {ee_pose_path}")
    
    # Base tracker parameters (same as fabric_tracking_main.py)
    base_params = {
        'intrinsics': INTRINSICS,
        'max_depth': 1080.0,
        # CPD parameters
        'cpd_beta': 50.0,
        'cpd_lambda': 0.1,
        'cpd_w': 0.1,
        'cpd_max_iter': 300,
        'cpd_tol': 1e-5,
        'cpd_downsample': 3000,
        # Geometry constraints
        'n_outer_iterations': 5,
        'n_edge_iterations': 20,
        'edge_weight': 0.5,
        'edge_tolerance': 0.05,
        # Repulsion
        'repulsion_iterations': 100,
        'repulsion_lr': 0.05,
        # End-effector poses
        'ee_poses_3d': ee_poses_3d,
    }
    
    # Methods to compare
    method_names = ['FabricTracker', 'CDCPD']
    
    print("=" * 70)
    print("CDCPD vs FabricTracker BENCHMARK")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    tracking_data = np.load(str(data_path), allow_pickle=True).item()
    
    frame_keys = sorted(tracking_data.keys())
    # Slice frame_keys to start from START_FRAME
    frame_keys = frame_keys[START_FRAME:]
    n_frames = len(frame_keys)
    print(f"Found {n_frames} frames (starting from frame {START_FRAME})")
    
    # Load masks
    print(f"\nLoading masks from: {masks_dir}")
    mask_files = sorted(masks_dir.glob("mask_frame_*.npy"))
    print(f"Found {len(mask_files)} mask files")
    
    n_frames = min(n_frames, len(mask_files) - START_FRAME)
    
    if args.n_frames is not None:
        n_frames = min(n_frames, args.n_frames)
    
    print(f"\nTotal frames to process: {n_frames}")
    
    # Initialize trackers
    print("\nInitializing trackers...")
    trackers = {
        'FabricTracker': FabricTracker(**base_params),
        'CDCPD': CDCPDFabricTracker(**base_params),
    }
    
    # Process frames
    print(f"\n{'='*70}")
    print("PROCESSING FRAMES")
    print("=" * 70)
    
    all_metrics = {name: [] for name in method_names}
    traj_histories = {name: [] for name in method_names}
    
    # Video writer
    video_writer = None
    fps = 30
    
    for i in range(n_frames):
        frame_start = time.time()
        
        frame_key = frame_keys[i]
        data = tracking_data[frame_key]
        
        rgb = data['color'][:, :, ::-1]  # BGR to RGB
        depth = data['transformed_depth']
        
        # Load mask using original frame index (accounting for START_FRAME)
        original_frame_idx = i + START_FRAME
        mask_path = masks_dir / f"mask_frame_{original_frame_idx:04d}.npy"
        mask_raw = np.load(str(mask_path))
        
        # Apply depth thresholding
        max_depth = base_params['max_depth']
        valid_depth = (depth > 0) & (depth < max_depth)
        mask = mask_raw & valid_depth
        
        metrics_dict = {}
        results_dict = {}
        point_cloud = None
        
        for name in method_names:
            tracker = trackers[name]
            result = tracker.process_frame(depth, mask, frame_idx=i)
            results_dict[name] = result
            
            # Extract point cloud for position metrics (once)
            if point_cloud is None and result['success']:
                if hasattr(tracker, '_extract_point_cloud'):
                    point_cloud = tracker._extract_point_cloud(mask, depth)
        
        # Verify both trackers have same initialization on frame 0
        if i == 0:
            ft = trackers['FabricTracker']
            cdcpd = trackers['CDCPD']
            if ft.is_initialized and cdcpd.is_initialized:
                print(f"\n  === Frame 0 Initialization Verification ===")
                print(f"  FabricTracker: {len(ft.grid_edges)} edges")
                print(f"  CDCPD:         {len(cdcpd.grid_edges)} edges")
                ref_sum_ft = sum(ft.reference_lengths.values())
                ref_sum_cd = sum(cdcpd.reference_lengths.values())
                print(f"  FabricTracker ref lengths sum: {ref_sum_ft:.2f} mm")
                print(f"  CDCPD ref lengths sum:         {ref_sum_cd:.2f} mm")
                kp_match = np.allclose(ft.reference_keypoints, cdcpd.reference_keypoints, atol=1e-3)
                print(f"  Keypoints match (atol=1e-3): {kp_match}")
                if not kp_match:
                    diff = np.linalg.norm(ft.reference_keypoints - cdcpd.reference_keypoints, axis=1)
                    print(f"  Max keypoint diff: {np.max(diff):.4f} mm")
                print(f"  ============================================\n")
        
        for name in method_names:
            result = results_dict[name]
            tracker = trackers[name]
            
            # Update trajectory history
            if result['success'] and 'keypoints_2d' in result:
                traj_histories[name].append(result['keypoints_2d'].copy())
            
            if result['success'] and tracker.is_initialized:
                keypoints = result['keypoints']
                edges = tracker.grid_edges
                ref_lengths = tracker.reference_lengths
                
                edge_metrics = compute_edge_metrics(keypoints, edges, ref_lengths)
                pos_metrics = compute_position_metrics(
                    keypoints, 
                    point_cloud if point_cloud is not None else np.empty((0, 3))
                )
                
                metrics = {
                    'edge_pct_mean': edge_metrics['pct_mean'],
                    'edge_pct_max': edge_metrics['pct_max'],
                    'edge_rmse_mm': edge_metrics['rmse_mm'],
                    'edge_under_2pct': edge_metrics['under_2pct'],
                    'edge_under_5pct': edge_metrics['under_5pct'],
                    'edge_under_10pct': edge_metrics['under_10pct'],
                    'pos_rmse_mm': pos_metrics['rmse_mm'],
                    'pos_under_2mm': pos_metrics['under_2mm'],
                    'pos_under_5mm': pos_metrics['under_5mm'],
                    'pos_under_10mm': pos_metrics['under_10mm'],
                }
            else:
                metrics = {
                    'edge_pct_mean': 0.0, 'edge_pct_max': 0.0, 'edge_rmse_mm': 0.0,
                    'edge_under_2pct': 0.0, 'edge_under_5pct': 0.0, 'edge_under_10pct': 0.0,
                    'pos_rmse_mm': 0.0, 'pos_under_2mm': 0.0, 'pos_under_5mm': 0.0, 'pos_under_10mm': 0.0,
                }
            
            metrics_dict[name] = metrics
            all_metrics[name].append(metrics)
        
        # Prepare trajectory histories for visualization
        traj_hist_arrays = {}
        for name in method_names:
            if len(traj_histories[name]) > 0:
                traj_hist_arrays[name] = np.array(traj_histories[name])
            else:
                traj_hist_arrays[name] = None
        
        # Create side-by-side visualization
        vis_frames = []
        for name in method_names:
            result = results_dict[name]
            metrics = metrics_dict[name]
            
            vis = create_frame_visualization(
                rgb, mask,
                result.get('keypoints_2d') if result['success'] else None,
                result.get('edges', []),
                name, metrics, i,
                traj_history=traj_hist_arrays.get(name),
                tail_length=60
            )
            vis_frames.append(vis)
        
        # Side-by-side
        comparison = np.concatenate(vis_frames, axis=1)
        
        # Initialize video writer
        if video_writer is None:
            H_vid, W_vid = comparison.shape[:2]
            video_path = str(output_dir / "cdcpd_comparison.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W_vid, H_vid))
        
        video_writer.write(cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        
        frame_time = time.time() - frame_start
        
        if i % 10 == 0:
            print(f"Frame {i:4d}: "
                  f"FT E={metrics_dict['FabricTracker']['edge_pct_mean']:.1f}% P={metrics_dict['FabricTracker']['pos_rmse_mm']:.1f} | "
                  f"CDCPD E={metrics_dict['CDCPD']['edge_pct_mean']:.1f}% P={metrics_dict['CDCPD']['pos_rmse_mm']:.1f} | "
                  f"{frame_time*1000:.0f}ms")
        
        if i % 20 == 0:
            gc.collect()
    
    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved to: {output_dir / 'cdcpd_comparison.mp4'}")
    
    # ========================================================================
    # Compute Summary Statistics
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    summary_data = []
    
    for name in method_names:
        metrics_list = all_metrics[name][1:]  # Skip frame 0
        
        if len(metrics_list) == 0:
            continue
        
        # Filter valid frames
        valid_metrics = [m for m in metrics_list if m['edge_pct_mean'] > 0]
        
        if len(valid_metrics) == 0:
            continue
        
        edge_pct_means = [m['edge_pct_mean'] for m in valid_metrics]
        edge_pct_maxes = [m['edge_pct_max'] for m in valid_metrics]
        edge_rmses = [m['edge_rmse_mm'] for m in valid_metrics]
        edge_under_2pct = [m['edge_under_2pct'] for m in valid_metrics]
        edge_under_5pct = [m['edge_under_5pct'] for m in valid_metrics]
        edge_under_10pct = [m['edge_under_10pct'] for m in valid_metrics]
        pos_rmses = [m['pos_rmse_mm'] for m in valid_metrics]
        pos_under_2mm = [m['pos_under_2mm'] for m in valid_metrics]
        pos_under_5mm = [m['pos_under_5mm'] for m in valid_metrics]
        pos_under_10mm = [m['pos_under_10mm'] for m in valid_metrics]
        
        summary = {
            'method': name,
            'edge_pct_mean_avg': np.mean(edge_pct_means),
            'edge_pct_mean_std': np.std(edge_pct_means),
            'edge_pct_max_avg': np.mean(edge_pct_maxes),
            'edge_pct_max_abs': np.max(edge_pct_maxes),
            'edge_rmse_avg': np.mean(edge_rmses),
            'edge_rmse_std': np.std(edge_rmses),
            'edge_under_2pct': np.mean(edge_under_2pct),
            'edge_under_5pct': np.mean(edge_under_5pct),
            'edge_under_10pct': np.mean(edge_under_10pct),
            'pos_rmse_avg': np.mean(pos_rmses),
            'pos_rmse_std': np.std(pos_rmses),
            'pos_under_2mm': np.mean(pos_under_2mm),
            'pos_under_5mm': np.mean(pos_under_5mm),
            'pos_under_10mm': np.mean(pos_under_10mm),
        }
        summary_data.append(summary)
    
    # Print summary
    print(f"\n{'Method':<14} | {'Edge%Mean':<14} | {'EdgeRMSE':<12} | {'Max%(Abs)':<9} | {'E<5%':<6} | {'PosRMSE':<14} | {'P<5mm':<6}")
    print("-" * 100)
    
    for s in summary_data:
        print(f"{s['method']:<14} | "
              f"{s['edge_pct_mean_avg']:>4.2f}%±{s['edge_pct_mean_std']:>4.2f}% | "
              f"{s['edge_rmse_avg']:>5.2f}±{s['edge_rmse_std']:>3.2f}mm | "
              f"{s['edge_pct_max_abs']:>7.2f}% | "
              f"{s['edge_under_5pct']:>5.1f}% | "
              f"{s['pos_rmse_avg']:>4.2f}±{s['pos_rmse_std']:>4.2f}mm | "
              f"{s['pos_under_5mm']:>5.1f}%")
    
    # Save to CSV
    csv_path = output_dir / "cdcpd_benchmark.csv"
    with open(csv_path, 'w') as f:
        f.write("Method,EdgePctMean_Avg,EdgePctMean_Std,EdgePctMax_Avg,EdgePctMax_Abs,"
                "EdgeRMSE_Avg,EdgeRMSE_Std,Edge<2%,Edge<5%,Edge<10%,"
                "PosRMSE_Avg,PosRMSE_Std,Pos<2mm,Pos<5mm,Pos<10mm\n")
        for s in summary_data:
            f.write(f"{s['method']},"
                    f"{s['edge_pct_mean_avg']:.4f},{s['edge_pct_mean_std']:.4f},"
                    f"{s['edge_pct_max_avg']:.4f},{s['edge_pct_max_abs']:.4f},"
                    f"{s['edge_rmse_avg']:.4f},{s['edge_rmse_std']:.4f},"
                    f"{s['edge_under_2pct']:.2f},{s['edge_under_5pct']:.2f},{s['edge_under_10pct']:.2f},"
                    f"{s['pos_rmse_avg']:.4f},{s['pos_rmse_std']:.4f},"
                    f"{s['pos_under_2mm']:.2f},{s['pos_under_5mm']:.2f},{s['pos_under_10mm']:.2f}\n")
    print(f"\nResults saved to: {csv_path}")
    
    # Generate plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    colors = {'FabricTracker': 'blue', 'CDCPD': 'red'}
    
    for name in method_names:
        frames = list(range(len(all_metrics[name])))
        edge_pct = [m['edge_pct_mean'] for m in all_metrics[name]]
        pos_rmse = [m['pos_rmse_mm'] for m in all_metrics[name]]
        
        ax1.plot(frames, edge_pct, label=name, color=colors[name], linewidth=1.5)
        ax2.plot(frames, pos_rmse, label=name, color=colors[name], linewidth=1.5)
    
    ax1.set_ylabel('Edge % Error (%)')
    ax1.set_title('CDCPD vs FabricTracker - Fabric Tracking')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    ax2.set_ylabel('Position RMSE (mm)')
    ax2.set_xlabel('Frame')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    plot_path = output_dir / "cdcpd_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {plot_path}")
    
    print(f"\nOutput directory: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CDCPD vs FabricTracker Benchmark")
    parser.add_argument('--n_frames', type=int, default=None,
                        help='Number of frames to process')
    
    args = parser.parse_args()
    run_benchmark(args)
