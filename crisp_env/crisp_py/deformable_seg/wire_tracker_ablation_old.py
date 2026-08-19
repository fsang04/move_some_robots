"""
Wire Tracker Ablation Study

Evaluates the design components of WireTracker:
    - Full: CPD → Snap Node → Geometry Constraint
    - NoSnap: CPD → Geometry Constraint (no snap node)
    - NoGeometry: CPD → Snap Node (no geometry constraint)
    - CPDOnly: CPD only (no snap node, no geometry)

Metrics:
    1. Edge Length Error (Mean % and Std %)
    2. Surface Distance RMSE (mm)
    3. Chamfer Distance (mm²)

Output:
    - 2x2 grid video comparing all 4 methods
    - CSV with per-frame metrics
    - Summary statistics

Usage:
    python wire_tracker_ablation.py --trajectory traj2 --n_frames 100

Author: Auto-generated
Date: 2026-02-16
"""

import numpy as np
import cv2
from pathlib import Path
import time
import argparse
import gc
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import PIL.Image
import io

from wire_tracker import WireTracker


# ============================================================================
# METRICS
# ============================================================================

def compute_edge_length_error(keypoints: np.ndarray, edges: list, reference_lengths: np.ndarray) -> dict:
    """
    Compute edge length error statistics.
    
    For edge (i, j):
        Error = |L_curr - L_ref| / L_ref
    
    Args:
        keypoints: K × 3 keypoint positions
        edges: List of (i, j) edge tuples
        reference_lengths: Array of reference edge lengths
    
    Returns:
        dict with 'errors' (array), 'mean' (%), 'std' (%)
    """
    if keypoints is None or len(keypoints) == 0 or edges is None or len(edges) == 0:
        return {'errors': np.array([]), 'mean': 0.0, 'std': 0.0}
    
    errors = []
    for edge_idx, (i, j) in enumerate(edges):
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        current_length = np.linalg.norm(keypoints[i] - keypoints[j])
        ref_length = reference_lengths[edge_idx]
        if ref_length > 1e-6:
            error = abs(current_length - ref_length) / ref_length
            errors.append(error)
    
    errors = np.array(errors)
    return {
        'errors': errors,
        'mean': np.mean(errors) * 100 if len(errors) > 0 else 0.0,  # percentage
        'std': np.std(errors) * 100 if len(errors) > 0 else 0.0,    # percentage
    }


def compute_surface_rmse(keypoints: np.ndarray, skeleton_pc: np.ndarray) -> float:
    """
    Compute Surface Distance RMSE.
    
    For each keypoint p_k, find nearest skeleton point:
        d_k = min_{s in S} ||p_k - s||
    
    RMSE = sqrt(1/K * sum(d_k^2))
    
    Args:
        keypoints: K × 3 keypoint positions
        skeleton_pc: N × 3 skeleton point cloud
    
    Returns:
        RMSE in mm
    """
    if keypoints is None or len(keypoints) == 0 or skeleton_pc is None or len(skeleton_pc) == 0:
        return 0.0
    
    # Find nearest skeleton point for each keypoint
    nn = NearestNeighbors(n_neighbors=1).fit(skeleton_pc)
    distances, _ = nn.kneighbors(keypoints)
    distances = distances.flatten()
    
    # RMSE
    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse


def compute_chamfer_distance(keypoints: np.ndarray, skeleton_pc: np.ndarray) -> float:
    """
    Compute Chamfer Distance (bidirectional).
    
    CD(P, S) = (1/|P|) * sum_{p in P} min_{s in S} ||p - s||^2 
             + (1/|S|) * sum_{s in S} min_{p in P} ||s - p||^2
    
    Args:
        keypoints: K × 3 keypoint positions (P)
        skeleton_pc: N × 3 skeleton point cloud (S)
    
    Returns:
        Chamfer distance in mm²
    """
    if keypoints is None or len(keypoints) == 0 or skeleton_pc is None or len(skeleton_pc) == 0:
        return 0.0
    
    # P -> S: for each keypoint, find nearest skeleton point
    nn_s = NearestNeighbors(n_neighbors=1).fit(skeleton_pc)
    dist_p_to_s, _ = nn_s.kneighbors(keypoints)
    term1 = np.mean(dist_p_to_s.flatten() ** 2)
    
    # S -> P: for each skeleton point, find nearest keypoint
    nn_p = NearestNeighbors(n_neighbors=1).fit(keypoints)
    dist_s_to_p, _ = nn_p.kneighbors(skeleton_pc)
    term2 = np.mean(dist_s_to_p.flatten() ** 2)
    
    return term1 + term2


# ============================================================================
# ABLATION TRACKER VARIANTS
# ============================================================================

class AblationTracker:
    """
    Wrapper around WireTracker that supports different ablation modes.
    
    Modes:
        - 'full': CPD → Snap Node → Geometry (default WireTracker behavior)
        - 'no_snap': CPD → Geometry (skip snap node replacement)
        - 'no_geometry': CPD → Snap Node (skip geometry constraint)
        - 'cpd_only': CPD only (skip both snap node and geometry)
    """
    
    def __init__(self, mode: str, tracker_params: dict):
        """
        Initialize ablation tracker.
        
        Args:
            mode: One of ['full', 'no_snap', 'no_geometry', 'cpd_only']
            tracker_params: Parameters for WireTracker
        """
        self.mode = mode
        self.tracker = WireTracker(**tracker_params)
        
        # Access tracker internals
        self.is_initialized = False
        
    def process_frame(
        self,
        depth: np.ndarray,
        arm_depth: np.ndarray = None,
        rgb: np.ndarray = None,
        precomputed_arm_mask: np.ndarray = None,
    ) -> dict:
        """
        Process a frame with ablation-specific tracking.
        
        Frame 0: Always use full initialization
        Frame N>0: Use ablation-specific tracking
        """
        tracker = self.tracker
        
        # Segmentation (same for all methods)
        if not tracker.is_initialized:
            n_components = 1  # Frame 0: single largest component
        else:
            n_components = tracker.top_k_components
        
        seg_result = tracker.segment(
            depth, arm_depth, n_components=n_components,
            precomputed_arm_mask=precomputed_arm_mask
        )
        
        foreground_mask = seg_result['foreground_mask']
        skeleton_mask = seg_result['skeleton_mask']
        skeleton_pc = seg_result['skeleton_pc']
        
        # Check skeleton validity
        if np.sum(skeleton_mask > 0) < tracker.min_skeleton_pixels:
            tracker.skip_frame()
            return {
                'success': False,
                'reason': 'insufficient_skeleton',
                'mode': 'skip',
                'foreground_mask': foreground_mask,
                'skeleton_mask': skeleton_mask,
            }
        
        # Frame 0: Initialize (same for all methods)
        if not tracker.is_initialized:
            result = tracker.initialize(skeleton_mask, depth)
            result['foreground_mask'] = foreground_mask
            result['skeleton_mask'] = skeleton_mask
            result['skeleton_pc'] = skeleton_pc
            self.is_initialized = tracker.is_initialized
            return result
        
        # Frame N > 0: Ablation-specific tracking
        result = self._track_ablation(skeleton_mask, skeleton_pc, depth)
        result['foreground_mask'] = foreground_mask
        result['skeleton_mask'] = skeleton_mask
        result['skeleton_pc'] = skeleton_pc
        
        return result
    
    def _track_ablation(
        self,
        skeleton_mask: np.ndarray,
        skeleton_pc: np.ndarray,
        depth: np.ndarray,
    ) -> dict:
        """
        Ablation-specific tracking for Frame N > 0.
        
        Modes:
            - 'full': CPD → Snap Node → Geometry
            - 'no_snap': CPD → Geometry
            - 'no_geometry': CPD → Snap Node
            - 'cpd_only': CPD only
        """
        tracker = self.tracker
        timing = {}
        total_start = time.time()
        
        # Step 1: Node detection (for snap node methods)
        t0 = time.time()
        branch_2d, leaf_2d, _, _ = tracker._node_identification(skeleton_mask)
        detected_branch_3d = tracker._pixel_to_3d(branch_2d, depth)
        detected_leaf_3d = tracker._pixel_to_3d(leaf_2d, depth)
        timing['node_detection'] = time.time() - t0
        
        # Step 2: CPD registration (all methods)
        t0 = time.time()
        cpd_target = skeleton_pc
        if len(cpd_target) > tracker.cpd_downsample:
            indices = np.random.choice(len(cpd_target), tracker.cpd_downsample, replace=False)
            cpd_target = cpd_target[indices]
        
        cpd_keypoints, _ = tracker._cpd_register(tracker.prev_keypoints, cpd_target)
        timing['cpd'] = time.time() - t0
        
        # Step 3: Snap Node (Hungarian matching) - conditional
        t0 = time.time()
        if self.mode in ['full', 'no_geometry']:
            # Apply snap node
            adjusted = tracker._hungarian_replace_anchors(
                cpd_keypoints,
                detected_branch_3d,
                detected_leaf_3d,
                skeleton_pc,
            )
        else:
            # Skip snap node
            adjusted = cpd_keypoints
        timing['hungarian'] = time.time() - t0
        
        # Step 4: Geometry constraint - conditional
        t0 = time.time()
        if self.mode in ['full', 'no_snap']:
            # Apply geometry constraint
            keypoints = tracker._joint_constraint_optimization(adjusted, skeleton_pc)
        else:
            # Skip geometry constraint
            keypoints = adjusted
        timing['geometry'] = time.time() - t0
        
        # Compute edge errors
        edge_errors = tracker._compute_edge_errors(keypoints)
        
        # Update state
        tracker.prev_keypoints = keypoints.copy()
        tracker.consecutive_skips = 0
        
        keypoints_2d = tracker._project_3d_to_2d(keypoints)
        
        timing['total'] = time.time() - total_start
        
        return {
            'success': True,
            'keypoints': keypoints,
            'keypoints_2d': keypoints_2d,
            'edges': tracker.reference_edges,
            'edge_errors': edge_errors,
            'detected_branch': detected_branch_3d,
            'detected_leaf': detected_leaf_3d,
            'mode': 'track',
            'timing': timing,
        }
    
    @property
    def reference_edges(self):
        return self.tracker.reference_edges
    
    @property
    def reference_lengths(self):
        return self.tracker.reference_lengths
    
    @property
    def reference_n_branch(self):
        return self.tracker.reference_n_branch
    
    @property
    def reference_n_leaf(self):
        return self.tracker.reference_n_leaf


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_single_frame_visualization(
    rgb: np.ndarray,
    skeleton_mask: np.ndarray,
    keypoints_2d: np.ndarray,
    edges: list,
    method_name: str,
    metrics: dict,
    n_branch: int = 0,
    n_leaf: int = 0,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Create visualization for a single method.
    
    Shows:
        - RGB with skeleton overlay
        - Keypoints and edges
        - Metrics text
    """
    H, W = rgb.shape[:2]
    
    # Colors
    SKELETON_COLOR = [0, 191, 255]  # Deep sky blue
    EDGE_COLOR = [50, 205, 50]      # Lime green
    BRANCH_COLOR = [128, 0, 128]    # Purple
    LEAF_COLOR = [255, 255, 0]      # Yellow
    INTER_COLOR = [255, 165, 0]     # Orange
    
    # Create overlay
    vis = rgb.copy()
    
    # Draw skeleton
    skeleton_thick = cv2.dilate(skeleton_mask, np.ones((3, 3), np.uint8), iterations=1)
    vis[skeleton_thick > 0] = SKELETON_COLOR
    
    # Draw edges
    if keypoints_2d is not None and len(keypoints_2d) > 0 and edges is not None:
        kp_int = keypoints_2d.astype(int)
        for (i, j) in edges:
            if i < len(kp_int) and j < len(kp_int):
                pt1 = (kp_int[i, 1], kp_int[i, 0])  # (col, row)
                pt2 = (kp_int[j, 1], kp_int[j, 0])
                cv2.line(vis, pt1, pt2, EDGE_COLOR, 2)
        
        # Draw keypoints
        for idx, (row, col) in enumerate(kp_int):
            if 0 <= row < H and 0 <= col < W:
                if idx < n_branch:
                    color = BRANCH_COLOR
                elif idx < n_branch + n_leaf:
                    color = LEAF_COLOR
                else:
                    color = INTER_COLOR
                cv2.circle(vis, (col, row), 6, color, -1)
    
    # Add method name and metrics
    cv2.putText(vis, method_name, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, f"Frame: {frame_idx}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Metrics
    y_offset = 90
    if 'edge_err_mean' in metrics:
        text = f"Edge Err: {metrics['edge_err_mean']:.1f}% +/- {metrics['edge_err_std']:.1f}%"
        cv2.putText(vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
    if 'surface_rmse' in metrics:
        text = f"Surface RMSE: {metrics['surface_rmse']:.2f} mm"
        cv2.putText(vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
    if 'chamfer_dist' in metrics:
        text = f"Chamfer: {metrics['chamfer_dist']:.2f} mm^2"
        cv2.putText(vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis


def create_2x2_grid_visualization(
    rgb: np.ndarray,
    skeleton_mask: np.ndarray,
    results: dict,
    method_names: list,
    metrics_dict: dict,
    n_branch: int,
    n_leaf: int,
    frame_idx: int,
) -> np.ndarray:
    """
    Create 2x2 grid comparing 4 methods.
    
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
        
        vis = create_single_frame_visualization(
            rgb, skeleton_mask, keypoints_2d, edges,
            method, metrics, n_branch, n_leaf, frame_idx
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
    
    # ========================================================================
    # Configuration
    # ========================================================================
    
    # Camera intrinsics
    intrinsics = np.array([
        [606.1124267578125, 0, 641.7578125],
        [0, 605.8821411132812, 365.6518859863281],
        [0, 0, 1]
    ])
    
    # Trajectory configuration
    trajectory_configs = {
        'traj1': {
            'arm_data_path': Path("./data/arm_traj1/arm_traj1.npy"),
            'full_data_path': Path("./data/arm_traj1/arm_with_wires_traj1.npy"),
            'output_dir': Path("./data/arm_traj1/ablation_output"),
            'precomputed_mask_dir': None,
            'arm_green_frame': 66,
            'full_green_frame': 66,
        },
        'traj2': {
            'arm_data_path': Path("./data/arm_traj2/arm_traj2.npy"),
            'full_data_path': Path("./data/arm_traj2/arm_with_wires_traj2.npy"),
            'output_dir': Path("./data/arm_traj2/ablation_output"),
            'precomputed_mask_dir': Path("./data/arm_traj2/masks"),
            'arm_green_frame': 0,
            'full_green_frame': 0,
        },
        'traj3': {
            'arm_data_path': Path("./data/arm_traj3/arm_traj3_contact.npy"),
            'full_data_path': Path("./data/arm_traj3/arm_with_wires_traj3_contact.npy"),
            'output_dir': Path("./data/arm_traj3/ablation_output"),
            'precomputed_mask_dir': None,
            'arm_green_frame': 84,
            'full_green_frame': 100,
        },
    }
    
    config = trajectory_configs[args.trajectory]
    
    # Tracker parameters (shared across all methods)
    tracker_params = {
        'intrinsics': intrinsics,
        'n_keypoints': 21,
        'target_branch_nodes': 2,
        'target_leaf_nodes': 4,
        'bg_threshold': 80.0,
        'max_depth': 1000.0,
        'top_k_components': 5,
        'arm_dilation_pixels': 5,
        'cpd_beta': 10.0,
        'cpd_lambda': 2.0,
        'cpd_w': 0.1,
        'cpd_max_iter': 100,
        'cpd_tol': 1e-3,
        'n_outer_iterations': 5,
        'n_edge_iterations': 20,
        'edge_weight': 0.5,
        'edge_tolerance': 0.15,
        'repulsion_iterations': 200,
        'repulsion_lr': 10.0,
        'repulsion_k_neighbors': 3,
    }
    
    # Ablation methods
    method_names = ['Full', 'NoSnap', 'NoGeometry', 'CPDOnly']
    method_modes = ['full', 'no_snap', 'no_geometry', 'cpd_only']
    
    # ========================================================================
    # Load Data
    # ========================================================================
    
    print("=" * 70)
    print("WIRE TRACKER ABLATION STUDY")
    print("=" * 70)
    print(f"\nTrajectory: {args.trajectory}")
    print(f"Methods: {method_names}")
    
    print(f"\nLoading arm-only data from: {config['arm_data_path']}")
    arm_only_data = np.load(str(config['arm_data_path']), allow_pickle=True).item()
    
    print(f"Loading full scene data from: {config['full_data_path']}")
    full_scene_data = np.load(str(config['full_data_path']), allow_pickle=True).item()
    
    # Synchronize sequences
    arm_frame_keys = sorted(arm_only_data.keys())[config['arm_green_frame']:]
    full_frame_keys = sorted(full_scene_data.keys())[config['full_green_frame']:]
    
    n_frames = min(len(arm_frame_keys), len(full_frame_keys))
    if args.n_frames is not None:
        n_frames = min(n_frames, args.n_frames)
    
    arm_frame_keys = arm_frame_keys[:n_frames]
    full_frame_keys = full_frame_keys[:n_frames]
    
    print(f"\nTotal frames to process: {n_frames}")
    
    # Create output directory
    output_dir = config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Initialize Trackers
    # ========================================================================
    
    print("\nInitializing trackers...")
    trackers = {}
    for name, mode in zip(method_names, method_modes):
        trackers[name] = AblationTracker(mode, tracker_params)
        print(f"  {name}: mode={mode}")
    
    # ========================================================================
    # Process Frames
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("PROCESSING FRAMES")
    print("=" * 70)
    
    # Storage for metrics
    all_metrics = {name: [] for name in method_names}
    
    # Video writer
    video_writer = None
    fps = 30
    
    for i in range(n_frames):
        frame_start = time.time()
        
        arm_frame_key = arm_frame_keys[i]
        full_frame_key = full_frame_keys[i]
        
        # Load data
        arm_data = arm_only_data[arm_frame_key]
        arm_depth = arm_data['transformed_depth'].copy()
        
        full_data = full_scene_data[full_frame_key]
        full_rgb = full_data['color'][:, :, ::-1]  # BGR to RGB
        full_depth = full_data['transformed_depth'].copy()
        
        # Load precomputed arm mask if available
        precomputed_arm_mask = None
        if config['precomputed_mask_dir'] is not None:
            mask_path = config['precomputed_mask_dir'] / f"mask_frame_{i:04d}.npy"
            if mask_path.exists():
                precomputed_arm_mask = np.load(str(mask_path))
        
        # Process frame with all methods
        results = {}
        metrics_dict = {}
        skeleton_mask = None
        skeleton_pc = None
        
        for name in method_names:
            result = trackers[name].process_frame(
                full_depth, arm_depth, full_rgb, precomputed_arm_mask
            )
            results[name] = result
            
            # Get skeleton from first tracker (same for all)
            if skeleton_mask is None and 'skeleton_mask' in result:
                skeleton_mask = result['skeleton_mask']
            if skeleton_pc is None and 'skeleton_pc' in result:
                skeleton_pc = result['skeleton_pc']
            
            # Compute metrics
            if result['success'] and trackers[name].tracker.is_initialized:
                keypoints = result['keypoints']
                edges = trackers[name].reference_edges
                ref_lengths = trackers[name].reference_lengths
                
                # Edge length error
                edge_result = compute_edge_length_error(keypoints, edges, ref_lengths)
                
                # Surface RMSE (use skeleton_pc from result)
                skel_pc = result.get('skeleton_pc', skeleton_pc)
                surface_rmse = compute_surface_rmse(keypoints, skel_pc)
                
                # Chamfer distance
                chamfer_dist = compute_chamfer_distance(keypoints, skel_pc)
                
                metrics = {
                    'edge_err_mean': edge_result['mean'],
                    'edge_err_std': edge_result['std'],
                    'surface_rmse': surface_rmse,
                    'chamfer_dist': chamfer_dist,
                }
            else:
                metrics = {
                    'edge_err_mean': 0.0,
                    'edge_err_std': 0.0,
                    'surface_rmse': 0.0,
                    'chamfer_dist': 0.0,
                }
            
            metrics_dict[name] = metrics
            all_metrics[name].append(metrics)
        
        # Get reference topology info from first tracker
        n_branch = trackers['Full'].reference_n_branch
        n_leaf = trackers['Full'].reference_n_leaf
        
        # Create 2x2 grid visualization
        if skeleton_mask is None:
            skeleton_mask = np.zeros_like(full_depth, dtype=np.uint8)
        
        grid = create_2x2_grid_visualization(
            full_rgb, skeleton_mask, results, method_names,
            metrics_dict, n_branch, n_leaf, i
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
        metrics_str = " | ".join([
            f"{name}: E={metrics_dict[name]['edge_err_mean']:.1f}% R={metrics_dict[name]['surface_rmse']:.1f}"
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
    
    # Compute summary for each method
    summary_data = []
    
    for name in method_names:
        metrics_list = all_metrics[name]
        
        # Skip frame 0 (initialization - same for all methods)
        metrics_list = metrics_list[1:] if len(metrics_list) > 1 else metrics_list
        
        if len(metrics_list) == 0:
            continue
        
        edge_err_means = [m['edge_err_mean'] for m in metrics_list if m['edge_err_mean'] > 0]
        edge_err_stds = [m['edge_err_std'] for m in metrics_list if m['edge_err_std'] > 0]
        surface_rmses = [m['surface_rmse'] for m in metrics_list if m['surface_rmse'] > 0]
        chamfer_dists = [m['chamfer_dist'] for m in metrics_list if m['chamfer_dist'] > 0]
        
        summary = {
            'method': name,
            'edge_err_mean_avg': np.mean(edge_err_means) if edge_err_means else 0,
            'edge_err_mean_std': np.std(edge_err_means) if edge_err_means else 0,
            'edge_err_std_avg': np.mean(edge_err_stds) if edge_err_stds else 0,
            'surface_rmse_avg': np.mean(surface_rmses) if surface_rmses else 0,
            'surface_rmse_std': np.std(surface_rmses) if surface_rmses else 0,
            'chamfer_avg': np.mean(chamfer_dists) if chamfer_dists else 0,
            'chamfer_std': np.std(chamfer_dists) if chamfer_dists else 0,
        }
        summary_data.append(summary)
    
    # Print summary table
    print(f"\n{'Method':<12} | {'Edge Err Mean':<18} | {'Edge Err Std':<12} | {'Surface RMSE':<18} | {'Chamfer Dist':<18}")
    print("-" * 90)
    
    for s in summary_data:
        print(f"{s['method']:<12} | "
              f"{s['edge_err_mean_avg']:>6.2f}% +/- {s['edge_err_mean_std']:>5.2f}% | "
              f"{s['edge_err_std_avg']:>6.2f}%      | "
              f"{s['surface_rmse_avg']:>6.2f} +/- {s['surface_rmse_std']:>5.2f} mm | "
              f"{s['chamfer_avg']:>6.2f} +/- {s['chamfer_std']:>5.2f} mm²")
    
    # Save summary to CSV
    csv_path = output_dir / "ablation_summary.csv"
    with open(csv_path, 'w') as f:
        # Header
        f.write("Method,EdgeErrMean_Avg,EdgeErrMean_Std,EdgeErrStd_Avg,SurfaceRMSE_Avg,SurfaceRMSE_Std,Chamfer_Avg,Chamfer_Std\n")
        for s in summary_data:
            f.write(f"{s['method']},{s['edge_err_mean_avg']:.4f},{s['edge_err_mean_std']:.4f},"
                    f"{s['edge_err_std_avg']:.4f},{s['surface_rmse_avg']:.4f},{s['surface_rmse_std']:.4f},"
                    f"{s['chamfer_avg']:.4f},{s['chamfer_std']:.4f}\n")
    print(f"\nSummary saved to: {csv_path}")
    
    # Save per-frame metrics
    per_frame_path = output_dir / "ablation_per_frame.csv"
    with open(per_frame_path, 'w') as f:
        f.write("Frame,Method,EdgeErrMean,EdgeErrStd,SurfaceRMSE,ChamferDist\n")
        for frame_idx in range(n_frames):
            for name in method_names:
                if frame_idx < len(all_metrics[name]):
                    m = all_metrics[name][frame_idx]
                    f.write(f"{frame_idx},{name},{m['edge_err_mean']:.4f},{m['edge_err_std']:.4f},"
                            f"{m['surface_rmse']:.4f},{m['chamfer_dist']:.4f}\n")
    print(f"Per-frame metrics saved to: {per_frame_path}")
    
    # ========================================================================
    # Component Contribution Analysis
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("COMPONENT CONTRIBUTION ANALYSIS")
    print("=" * 70)
    
    # Find indices for comparison
    idx_full = next((i for i, s in enumerate(summary_data) if s['method'] == 'Full'), None)
    idx_no_snap = next((i for i, s in enumerate(summary_data) if s['method'] == 'NoSnap'), None)
    idx_no_geom = next((i for i, s in enumerate(summary_data) if s['method'] == 'NoGeometry'), None)
    idx_cpd_only = next((i for i, s in enumerate(summary_data) if s['method'] == 'CPDOnly'), None)
    
    if all(idx is not None for idx in [idx_full, idx_no_snap, idx_no_geom, idx_cpd_only]):
        full = summary_data[idx_full]
        no_snap = summary_data[idx_no_snap]
        no_geom = summary_data[idx_no_geom]
        cpd_only = summary_data[idx_cpd_only]
        
        print("\n1. Snap Node Contribution (Full vs NoSnap):")
        print(f"   Edge Err improvement: {no_snap['edge_err_mean_avg'] - full['edge_err_mean_avg']:.2f}%")
        print(f"   Surface RMSE improvement: {no_snap['surface_rmse_avg'] - full['surface_rmse_avg']:.2f} mm")
        print(f"   Chamfer improvement: {no_snap['chamfer_avg'] - full['chamfer_avg']:.2f} mm²")
        
        print("\n2. Geometry Constraint Contribution (Full vs NoGeometry):")
        print(f"   Edge Err improvement: {no_geom['edge_err_mean_avg'] - full['edge_err_mean_avg']:.2f}%")
        print(f"   Surface RMSE improvement: {no_geom['surface_rmse_avg'] - full['surface_rmse_avg']:.2f} mm")
        print(f"   Chamfer improvement: {no_geom['chamfer_avg'] - full['chamfer_avg']:.2f} mm²")
        
        print("\n3. Combined Components vs CPD Only:")
        print(f"   Edge Err improvement: {cpd_only['edge_err_mean_avg'] - full['edge_err_mean_avg']:.2f}%")
        print(f"   Surface RMSE improvement: {cpd_only['surface_rmse_avg'] - full['surface_rmse_avg']:.2f} mm")
        print(f"   Chamfer improvement: {cpd_only['chamfer_avg'] - full['chamfer_avg']:.2f} mm²")
    
    print(f"\nOutput directory: {output_dir}")
    print("Done!")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wire Tracker Ablation Study")
    parser.add_argument('--trajectory', type=str, default='traj2',
                        choices=['traj1', 'traj2', 'traj3'],
                        help='Trajectory to process')
    parser.add_argument('--n_frames', type=int, default=None,
                        help='Number of frames to process (default: all)')
    
    args = parser.parse_args()
    run_ablation_study(args)
