"""
Fabric Tracker Ablation Study

Evaluates the design components of FabricTracker:
    - Full: CPD → Corner/Border Snap → Geometry Constraint
    - NoSnap: CPD → Geometry Constraint (no corner/border snapping)
    - NoGeometry: CPD → Corner/Border Snap (no geometry constraint)
    - CPDOnly: CPD only (no snapping, no geometry)

Metrics:
    1. Edge Length Error (Mean % and Std %)
    2. Surface Distance RMSE (mm)

Output:
    - 2x2 grid video comparing all 4 methods (with trajectory tails)
    - CSV with per-frame metrics
    - Summary statistics

Usage:
    python fabric_tracker_ablation.py
    python fabric_tracker_ablation.py --n_frames 100

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

from fabric_tracker import FabricTracker


# ============================================================================
# GRID CONSTANTS
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
# ABLATION TRACKER CLASS
# ============================================================================

class FabricTrackerAblation(FabricTracker):
    """
    FabricTracker with ablation flags for component analysis.
    
    Extends FabricTracker with:
        - enable_snap: Enable/disable corner and border snapping
        - enable_geometry_constraint: Enable/disable geometry constraint optimization
        - enable_ee_constraint: Enable/disable EE pose constraint
    """
    
    def __init__(
        self,
        enable_snap: bool = True,
        enable_geometry_constraint: bool = True,
        enable_ee_constraint: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.enable_snap = enable_snap
        self.enable_geometry_constraint = enable_geometry_constraint
        self.enable_ee_constraint = enable_ee_constraint
    
    def track(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        frame_idx: int,
    ) -> dict:
        """
        Track with ablation flags.
        
        Args:
            mask: H × W binary mask (already depth-thresholded)
            depth: H × W depth image
            frame_idx: Current frame index
        
        Returns:
            dict with tracking results
        """
        t_start = time.time()
        
        # Extract point cloud
        point_cloud = self._extract_point_cloud(mask, depth)
        
        if len(point_cloud) < self.min_foreground_pixels:
            self.consecutive_skips += 1
            return {
                'success': False,
                'reason': 'insufficient_points',
                'mode': 'skip',
            }
        
        # Extract 3D contour for border constraint
        contour_3d = self._extract_contour_3d(mask, depth)
        
        # Detect current frame's corners
        corners_2d = self._find_mask_corners(mask, depth)
        corners_3d = self._pixel_to_3d(corners_2d, depth) if corners_2d is not None else None
        
        # CPD registration
        t_cpd_start = time.time()
        cpd_keypoints, _ = self._cpd_register(self.prev_keypoints, point_cloud)
        cpd_time = time.time() - t_cpd_start
        
        keypoints = cpd_keypoints.copy()
        
        # Corner and border snapping (if enabled)
        if self.enable_snap:
            # Snap corner nodes to detected corners (if valid)
            if corners_3d is not None and not np.any(np.isnan(corners_3d)):
                corner_mapping = {
                    0: 0,   # grid TL -> corners TL
                    4: 1,   # grid TR -> corners TR
                    24: 2,  # grid BR -> corners BR
                    20: 3,  # grid BL -> corners BL
                }
                for grid_idx, corner_idx in corner_mapping.items():
                    keypoints[grid_idx] = corners_3d[corner_idx]
            
            # Snap border nodes to 3D contour
            if len(contour_3d) > 0:
                for idx in self.BORDER_INDICES:
                    keypoints[idx] = self._snap_to_contour_3d(keypoints[idx], contour_3d)
        
        # Geometry constraint optimization (if enabled)
        t_geom_start = time.time()
        if self.enable_geometry_constraint:
            keypoints = self._joint_constraint_optimization_with_contour(
                keypoints, point_cloud, contour_3d
            )
        geom_time = time.time() - t_geom_start
        
        # Replace with EE poses (if enabled)
        if self.enable_ee_constraint:
            keypoints = self._replace_with_ee_poses(keypoints, frame_idx)
        
        # Final snap: ensure border nodes are still on contour after geometry optimization
        if self.enable_snap and len(contour_3d) > 0:
            for idx in self.BORDER_INDICES:
                keypoints[idx] = self._snap_to_contour_3d(keypoints[idx], contour_3d)
        
        # Update state
        self.prev_keypoints = keypoints.copy()
        self.frame_count += 1
        self.consecutive_skips = 0
        
        # Project to 2D
        keypoints_2d = self._project_3d_to_2d(keypoints)
        
        # Compute edge errors
        edge_errors = self._compute_edge_errors(keypoints)
        
        track_time = time.time() - t_start
        
        return {
            'success': True,
            'mode': 'track',
            'keypoints': keypoints,
            'keypoints_2d': keypoints_2d,
            'edges': self.grid_edges,
            'edge_errors': edge_errors,
            'timing': {
                'cpd': cpd_time,
                'geom': geom_time,
                'total': track_time,
            },
        }


# ============================================================================
# METRICS
# ============================================================================

def compute_edge_metrics(keypoints: np.ndarray, edges: list, reference_lengths: dict) -> dict:
    """
    Compute comprehensive edge length metrics.
    
    Args:
        keypoints: K × 3 keypoint positions
        edges: List of (i, j) edge tuples
        reference_lengths: Dict of (i, j) -> reference length (mm)
    
    Returns:
        dict with metrics
    """
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
        'pct_mean': np.mean(pct_errors) * 100,  # percentage
        'pct_std': np.std(pct_errors) * 100,    # percentage
        'pct_max': np.max(pct_errors) * 100,    # percentage
        'rmse_mm': np.sqrt(np.mean(abs_errors ** 2)),  # mm
        'under_2pct': np.mean(pct_errors < 0.02) * 100,   # percentage of edges
        'under_5pct': np.mean(pct_errors < 0.05) * 100,   # percentage of edges
        'under_10pct': np.mean(pct_errors < 0.10) * 100,  # percentage of edges
    }


def compute_position_metrics(keypoints: np.ndarray, point_cloud: np.ndarray) -> dict:
    """
    Compute position accuracy metrics (distance to nearest point in point cloud).
    
    Args:
        keypoints: K × 3 keypoint positions
        point_cloud: N × 3 point cloud
    
    Returns:
        dict with metrics
    """
    if keypoints is None or len(keypoints) == 0 or point_cloud is None or len(point_cloud) == 0:
        return {
            'distances': np.array([]),
            'rmse_mm': 0.0,
            'under_2mm': 0.0, 'under_5mm': 0.0, 'under_10mm': 0.0,
        }
    
    # Find nearest point cloud point for each keypoint
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


# ============================================================================
# VISUALIZATION WITH TRAJECTORY TAILS
# ============================================================================

def create_single_frame_visualization(
    rgb: np.ndarray,
    mask: np.ndarray,
    keypoints_2d: np.ndarray,
    edges: list,
    method_name: str,
    metrics: dict,
    frame_idx: int = 0,
    traj_history: np.ndarray = None,
    tail_length: int = 60,
) -> np.ndarray:
    """
    Create visualization for a single method with trajectory tails.
    """
    H, W = rgb.shape[:2]
    
    # Colors
    MASK_COLOR = [0, 255, 0]        # Green for mask
    EDGE_COLOR = [255, 165, 0]      # Orange for edges
    CORNER_COLOR = [255, 0, 0]      # Red for corners
    BORDER_COLOR = [255, 255, 0]    # Yellow for border
    INTERIOR_COLOR = [0, 255, 255]  # Cyan for interior
    TAIL_COLOR = [255, 105, 180]    # Hot pink for trajectory tail
    
    KEYPOINT_RADIUS = 6
    EDGE_THICKNESS = 2
    TAIL_THICKNESS = 2
    
    # Create overlay
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
    
    # Draw edges
    if keypoints_2d is not None and len(keypoints_2d) > 0 and edges is not None:
        kp_int = keypoints_2d.astype(int)
        for (i, j) in edges:
            if i < len(kp_int) and j < len(kp_int):
                pt1 = (kp_int[i, 1], kp_int[i, 0])  # (col, row)
                pt2 = (kp_int[j, 1], kp_int[j, 0])
                cv2.line(vis, pt1, pt2, EDGE_COLOR, EDGE_THICKNESS)
        
        # Draw keypoints
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
    
    # Add method name and metrics
    cv2.putText(vis, method_name, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, f"Frame: {frame_idx}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Metrics
    y_offset = 90
    if 'edge_pct_mean' in metrics:
        text = f"Edge: {metrics['edge_pct_mean']:.1f}% (max {metrics['edge_pct_max']:.1f}%)"
        cv2.putText(vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
    if 'edge_rmse_mm' in metrics:
        text = f"Edge RMSE: {metrics['edge_rmse_mm']:.2f} mm"
        cv2.putText(vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
    if 'pos_rmse_mm' in metrics:
        text = f"Pos RMSE: {metrics['pos_rmse_mm']:.2f} mm"
        cv2.putText(vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis


def create_2x2_grid_visualization(
    rgb: np.ndarray,
    mask: np.ndarray,
    results: dict,
    method_names: list,
    metrics_dict: dict,
    frame_idx: int,
    traj_histories: dict = None,
    tail_length: int = 60,
) -> np.ndarray:
    """
    Create 2x2 grid comparing 4 methods with trajectory tails.
    
    Layout:
        [Full]       [NoSnap]
        [NoGeometry] [CPDOnly]
    """
    H, W = rgb.shape[:2]
    
    # Create individual visualizations
    vis_list = []
    for method in method_names:
        result = results[method]
        metrics = metrics_dict[method]
        
        if result['success']:
            keypoints_2d = result['keypoints_2d']
            edges = result['edges']
        else:
            keypoints_2d = np.empty((0, 2))
            edges = []
        
        # Get trajectory history for this method
        traj_history = None
        if traj_histories is not None and method in traj_histories:
            traj_history = traj_histories[method]
        
        vis = create_single_frame_visualization(
            rgb, mask, keypoints_2d, edges,
            method, metrics, frame_idx,
            traj_history=traj_history, tail_length=tail_length
        )
        vis_list.append(vis)
    
    # Arrange in 2x2 grid
    row1 = np.concatenate([vis_list[0], vis_list[1]], axis=1)
    row2 = np.concatenate([vis_list[2], vis_list[3]], axis=1)
    grid = np.concatenate([row1, row2], axis=0)
    
    return grid


# ============================================================================
# MAIN ABLATION STUDY
# ============================================================================

def run_ablation_study(args):
    """Run the ablation study."""
    
    # Paths
    data_path = Path("/home/yehengz/deformable_seg/data/full/tracking_fabric2_data.npy")
    masks_dir = Path("/home/yehengz/deformable_seg/data/arm_traj4_fabric/masks")
    ee_pose_path = Path("/home/yehengz/deformable_seg/data/arm_traj4_fabric/ee_pose_output/ee_poses_3d.npy")
    output_dir = Path("/home/yehengz/deformable_seg/data/arm_traj4_fabric/ablation_output")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load EE poses if available
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
    
    # Base tracker parameters
    base_tracker_params = {
        'intrinsics': INTRINSICS,
        'max_depth': 1080.0,
        # CPD parameters
        # beta: motion coherence width (should be ~distance between keypoints)
        # lambda: regularization (VERY low = more flexible, the formula is lambda*sigma2 which amplifies)
        'cpd_beta': 50.0,      # Larger for fabric grid spacing (~100mm between nodes)
        'cpd_lambda': 0.1,     # Very low - gets multiplied by sigma2 (~10000) inside CPD
        'cpd_w': 0.1,           # Outlier weight
        'cpd_max_iter': 300,
        'cpd_tol': 1e-5,        # Tighter tolerance
        'cpd_downsample': 3000,
        # Geometry constraints
        'n_outer_iterations': 5,
        'n_edge_iterations': 20,
        'edge_weight': 0.5,
        'edge_tolerance': 0.05,
        # Repulsion
        'repulsion_iterations': 100,
        'repulsion_lr': 0.05,
        # EE poses
        'ee_poses_3d': ee_poses_3d,
    }
    
    # Ablation configurations
    # All methods use EE constraint since robot always knows gripper positions
    ablation_configs = {
        'Full': {
            'enable_snap': True,
            'enable_geometry_constraint': True,
            'enable_ee_constraint': True,
        },
        'NoSnap': {
            'enable_snap': False,
            'enable_geometry_constraint': True,
            'enable_ee_constraint': True,
        },
        'NoGeometry': {
            'enable_snap': True,
            'enable_geometry_constraint': False,
            'enable_ee_constraint': True,
        },
        'CPDOnly': {
            'enable_snap': False,
            'enable_geometry_constraint': False,
            'enable_ee_constraint': True,
        },
    }
    
    method_names = ['Full', 'NoSnap', 'NoGeometry', 'CPDOnly']
    
    # Trajectory tail parameters
    tail_length = 60
    
    # ========================================================================
    # Load Data
    # ========================================================================
    
    print("=" * 70)
    print("FABRIC TRACKER ABLATION STUDY")
    print("=" * 70)
    print(f"\nMethods: {method_names}")
    
    print(f"\nLoading tracking data from: {data_path}")
    tracking_data = np.load(str(data_path), allow_pickle=True).item()
    
    frame_keys = sorted(tracking_data.keys())
    total_frames = len(frame_keys)
    print(f"Found {total_frames} frames total")
    
    # Apply start frame offset
    frame_keys = frame_keys[START_FRAME:]
    n_frames = len(frame_keys)
    print(f"Starting from frame {START_FRAME}, {n_frames} frames remaining")
    
    # Load masks
    print(f"\nLoading masks from: {masks_dir}")
    mask_files = sorted(masks_dir.glob("mask_frame_*.npy"))
    print(f"Found {len(mask_files)} mask files")
    
    n_frames = min(n_frames, len(mask_files) - START_FRAME)
    
    if args.n_frames is not None:
        n_frames = min(n_frames, args.n_frames)
    
    print(f"\nTotal frames to process: {n_frames}")
    
    # ========================================================================
    # Initialize Trackers
    # ========================================================================
    
    print("\nInitializing trackers...")
    trackers = {}
    for name in method_names:
        tracker_params = {**base_tracker_params, **ablation_configs[name]}
        trackers[name] = FabricTrackerAblation(**tracker_params)
        print(f"  {name}: enable_snap={ablation_configs[name]['enable_snap']}, "
              f"enable_geometry_constraint={ablation_configs[name]['enable_geometry_constraint']}")
    
    # ========================================================================
    # Process Frames
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("PROCESSING FRAMES")
    print("=" * 70)
    
    # Storage for metrics
    all_metrics = {name: [] for name in method_names}
    
    # Trajectory history for each method
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
        
        # Load mask (use original frame index)
        original_frame_idx = i + START_FRAME
        mask_path = masks_dir / f"mask_frame_{original_frame_idx:04d}.npy"
        mask_raw = np.load(str(mask_path))
        
        # Apply depth thresholding
        max_depth = base_tracker_params['max_depth']
        valid_depth = (depth > 0) & (depth < max_depth)
        mask = mask_raw & valid_depth
        
        # Process frame with all methods
        results = {}
        metrics_dict = {}
        point_cloud = None
        
        for name in method_names:
            tracker = trackers[name]
            result = tracker.process_frame(depth, mask, frame_idx=i)
            results[name] = result
            
            # Extract point cloud for position metrics (once)
            if point_cloud is None and result['success']:
                point_cloud = tracker._extract_point_cloud(mask, depth)
            
            # Update trajectory history
            if result['success'] and 'keypoints_2d' in result:
                traj_histories[name].append(result['keypoints_2d'].copy())
            
            # Compute metrics
            if result['success'] and tracker.is_initialized:
                keypoints = result['keypoints']
                edges = tracker.grid_edges
                ref_lengths = tracker.reference_lengths
                
                edge_metrics = compute_edge_metrics(keypoints, edges, ref_lengths)
                pos_metrics = compute_position_metrics(keypoints, point_cloud if point_cloud is not None else np.empty((0, 3)))
                
                metrics = {
                    'edge_pct_mean': edge_metrics['pct_mean'],
                    'edge_pct_std': edge_metrics['pct_std'],
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
                    'edge_pct_mean': 0.0, 'edge_pct_std': 0.0, 'edge_pct_max': 0.0,
                    'edge_rmse_mm': 0.0,
                    'edge_under_2pct': 0.0, 'edge_under_5pct': 0.0, 'edge_under_10pct': 0.0,
                    'pos_rmse_mm': 0.0,
                    'pos_under_2mm': 0.0, 'pos_under_5mm': 0.0, 'pos_under_10mm': 0.0,
                }
            
            metrics_dict[name] = metrics
            all_metrics[name].append(metrics)
        
        # Prepare trajectory histories as arrays for visualization
        traj_hist_arrays = {}
        for name in method_names:
            if len(traj_histories[name]) > 0:
                traj_hist_arrays[name] = np.array(traj_histories[name])
            else:
                traj_hist_arrays[name] = None
        
        # Create 2x2 grid visualization
        grid = create_2x2_grid_visualization(
            rgb, mask, results, method_names,
            metrics_dict, i,
            traj_histories=traj_hist_arrays, tail_length=tail_length
        )
        
        # Initialize video writer
        if video_writer is None:
            H_grid, W_grid = grid.shape[:2]
            video_path = str(output_dir / "ablation_comparison.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W_grid, H_grid))
        
        # Write frame
        video_writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        
        frame_time = time.time() - frame_start
        
        # Print progress
        if i % 10 == 0:
            metrics_str = " | ".join([
                f"{name}: E={metrics_dict[name]['edge_pct_mean']:.1f}%"
                for name in method_names
            ])
            print(f"Frame {i:4d}: {metrics_str} | {frame_time*1000:.0f}ms")
        
        # Garbage collection
        if i % 20 == 0:
            gc.collect()
    
    # ========================================================================
    # Finalize
    # ========================================================================
    
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved to: {output_dir / 'ablation_comparison.mp4'}")
    
    # ========================================================================
    # Compute Summary Statistics
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    
    summary_data = []
    
    for name in method_names:
        metrics_list = all_metrics[name]
        
        # Skip frame 0 (initialization)
        metrics_list = metrics_list[1:] if len(metrics_list) > 1 else metrics_list
        
        if len(metrics_list) == 0:
            continue
        
        # Edge metrics
        edge_pct_means = [m['edge_pct_mean'] for m in metrics_list if m['edge_pct_mean'] > 0]
        edge_pct_maxes = [m['edge_pct_max'] for m in metrics_list if m['edge_pct_max'] > 0]
        edge_rmses = [m['edge_rmse_mm'] for m in metrics_list if m['edge_rmse_mm'] > 0]
        edge_under_2 = [m['edge_under_2pct'] for m in metrics_list]
        edge_under_5 = [m['edge_under_5pct'] for m in metrics_list]
        edge_under_10 = [m['edge_under_10pct'] for m in metrics_list]
        
        # Position metrics
        pos_rmses = [m['pos_rmse_mm'] for m in metrics_list if m['pos_rmse_mm'] > 0]
        pos_under_2 = [m['pos_under_2mm'] for m in metrics_list]
        pos_under_5 = [m['pos_under_5mm'] for m in metrics_list]
        pos_under_10 = [m['pos_under_10mm'] for m in metrics_list]
        
        summary = {
            'method': name,
            'edge_pct_mean_avg': np.mean(edge_pct_means) if edge_pct_means else 0,
            'edge_pct_mean_std': np.std(edge_pct_means) if edge_pct_means else 0,
            'edge_pct_max_avg': np.mean(edge_pct_maxes) if edge_pct_maxes else 0,
            'edge_pct_max_abs': np.max(edge_pct_maxes) if edge_pct_maxes else 0,
            'edge_rmse_avg': np.mean(edge_rmses) if edge_rmses else 0,
            'edge_rmse_std': np.std(edge_rmses) if edge_rmses else 0,
            'edge_under_2pct': np.mean(edge_under_2) if edge_under_2 else 0,
            'edge_under_5pct': np.mean(edge_under_5) if edge_under_5 else 0,
            'edge_under_10pct': np.mean(edge_under_10) if edge_under_10 else 0,
            'pos_rmse_avg': np.mean(pos_rmses) if pos_rmses else 0,
            'pos_rmse_std': np.std(pos_rmses) if pos_rmses else 0,
            'pos_under_2mm': np.mean(pos_under_2) if pos_under_2 else 0,
            'pos_under_5mm': np.mean(pos_under_5) if pos_under_5 else 0,
            'pos_under_10mm': np.mean(pos_under_10) if pos_under_10 else 0,
        }
        summary_data.append(summary)
    
    # Print summary table - Edge Metrics
    print(f"\n{'='*70}")
    print("EDGE LENGTH METRICS")
    print("=" * 70)
    print(f"{'Method':<12} | {'Edge % Mean':<14} | {'Edge RMSE':<14} | {'Max%(Avg)':<10} | {'Max%(Abs)':<10} | {'<5%':<6} | {'<10%':<6}")
    print("-" * 90)
    
    for s in summary_data:
        print(f"{s['method']:<12} | "
              f"{s['edge_pct_mean_avg']:>5.2f}% ±{s['edge_pct_mean_std']:>4.2f}% | "
              f"{s['edge_rmse_avg']:>5.2f} ±{s['edge_rmse_std']:>4.2f}mm | "
              f"{s['edge_pct_max_avg']:>7.2f}% | "
              f"{s['edge_pct_max_abs']:>7.2f}% | "
              f"{s['edge_under_5pct']:>5.1f}% | "
              f"{s['edge_under_10pct']:>5.1f}%")
    
    # Print summary table - Position Metrics
    print(f"\n{'='*70}")
    print("POSITION METRICS")
    print("=" * 70)
    print(f"{'Method':<12} | {'Pos RMSE (mm)':<18} | {'<2mm':<8} | {'<5mm':<8} | {'<10mm':<8}")
    print("-" * 70)
    
    for s in summary_data:
        print(f"{s['method']:<12} | "
              f"{s['pos_rmse_avg']:>6.2f} ±{s['pos_rmse_std']:>6.2f} mm | "
              f"{s['pos_under_2mm']:>6.1f}% | "
              f"{s['pos_under_5mm']:>6.1f}% | "
              f"{s['pos_under_10mm']:>6.1f}%")
    
    # Save summary to CSV
    csv_path = output_dir / "ablation_summary.csv"
    with open(csv_path, 'w') as f:
        f.write("Method,EdgePctMean_Avg,EdgePctMean_Std,EdgePctMax_Avg,EdgePctMax_Abs,EdgeRMSE_Avg,EdgeRMSE_Std,"
                "Edge<2%,Edge<5%,Edge<10%,PosRMSE_Avg,PosRMSE_Std,Pos<2mm,Pos<5mm,Pos<10mm\n")
        for s in summary_data:
            f.write(f"{s['method']},{s['edge_pct_mean_avg']:.4f},{s['edge_pct_mean_std']:.4f},"
                    f"{s['edge_pct_max_avg']:.4f},{s['edge_pct_max_abs']:.4f},"
                    f"{s['edge_rmse_avg']:.4f},{s['edge_rmse_std']:.4f},"
                    f"{s['edge_under_2pct']:.2f},{s['edge_under_5pct']:.2f},{s['edge_under_10pct']:.2f},"
                    f"{s['pos_rmse_avg']:.4f},{s['pos_rmse_std']:.4f},"
                    f"{s['pos_under_2mm']:.2f},{s['pos_under_5mm']:.2f},{s['pos_under_10mm']:.2f}\n")
    print(f"\nSummary saved to: {csv_path}")
    
    # Save per-frame metrics
    per_frame_path = output_dir / "ablation_per_frame.csv"
    with open(per_frame_path, 'w') as f:
        f.write("Frame,Method,EdgePctMean,EdgePctMax,EdgeRMSE,PosRMSE,Edge<5%,Edge<10%,Pos<5mm,Pos<10mm\n")
        for frame_idx in range(n_frames):
            for name in method_names:
                if frame_idx < len(all_metrics[name]):
                    m = all_metrics[name][frame_idx]
                    f.write(f"{frame_idx},{name},{m['edge_pct_mean']:.4f},{m['edge_pct_max']:.4f},"
                            f"{m['edge_rmse_mm']:.4f},{m['pos_rmse_mm']:.4f},"
                            f"{m['edge_under_5pct']:.2f},{m['edge_under_10pct']:.2f},"
                            f"{m['pos_under_5mm']:.2f},{m['pos_under_10mm']:.2f}\n")
    print(f"Per-frame metrics saved to: {per_frame_path}")
    
    # ========================================================================
    # Generate Metrics Over Time Plot
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("GENERATING METRICS PLOT")
    print("=" * 70)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    colors = {'Full': 'blue', 'NoSnap': 'orange', 'NoGeometry': 'green', 'CPDOnly': 'red'}
    
    for name in method_names:
        metrics_list = all_metrics[name]
        frames = list(range(len(metrics_list)))
        
        edge_pct = [m['edge_pct_mean'] for m in metrics_list]
        ax1.plot(frames, edge_pct, label=name, color=colors.get(name, 'gray'), linewidth=1.5)
        
        pos_rmse = [m['pos_rmse_mm'] for m in metrics_list]
        ax2.plot(frames, pos_rmse, label=name, color=colors.get(name, 'gray'), linewidth=1.5)
    
    ax1.set_ylabel('Edge % Error (%)')
    ax1.set_title('Edge Length Error Over Time')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    ax2.set_ylabel('Position RMSE (mm)')
    ax2.set_xlabel('Frame')
    ax2.set_title('Position RMSE Over Time')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    plot_path = output_dir / "metrics_over_time.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")
    
    # ========================================================================
    # Component Contribution Analysis
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("COMPONENT CONTRIBUTION ANALYSIS")
    print("=" * 70)
    
    idx_full = next((i for i, s in enumerate(summary_data) if s['method'] == 'Full'), None)
    idx_no_snap = next((i for i, s in enumerate(summary_data) if s['method'] == 'NoSnap'), None)
    idx_no_geom = next((i for i, s in enumerate(summary_data) if s['method'] == 'NoGeometry'), None)
    idx_cpd_only = next((i for i, s in enumerate(summary_data) if s['method'] == 'CPDOnly'), None)
    
    if all(idx is not None for idx in [idx_full, idx_no_snap, idx_no_geom, idx_cpd_only]):
        full = summary_data[idx_full]
        no_snap = summary_data[idx_no_snap]
        no_geom = summary_data[idx_no_geom]
        cpd_only = summary_data[idx_cpd_only]
        
        print("\n1. Corner/Border Snap Contribution (Full vs NoSnap):")
        print(f"   Edge % Error reduction: {no_snap['edge_pct_mean_avg'] - full['edge_pct_mean_avg']:.2f}%")
        print(f"   Position RMSE reduction: {no_snap['pos_rmse_avg'] - full['pos_rmse_avg']:.2f} mm")
        
        print("\n2. Geometry Constraint Contribution (Full vs NoGeometry):")
        print(f"   Edge % Error reduction: {no_geom['edge_pct_mean_avg'] - full['edge_pct_mean_avg']:.2f}%")
        print(f"   Position RMSE reduction: {no_geom['pos_rmse_avg'] - full['pos_rmse_avg']:.2f} mm")
        
        print("\n3. Combined Components vs CPD Only:")
        print(f"   Edge % Error reduction: {cpd_only['edge_pct_mean_avg'] - full['edge_pct_mean_avg']:.2f}%")
        print(f"   Position RMSE reduction: {cpd_only['pos_rmse_avg'] - full['pos_rmse_avg']:.2f} mm")
    
    print(f"\nOutput directory: {output_dir}")
    print("Done!")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fabric Tracker Ablation Study")
    parser.add_argument('--n_frames', type=int, default=None,
                        help='Number of frames to process (default: all)')
    
    args = parser.parse_args()
    run_ablation_study(args)
