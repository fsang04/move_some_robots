"""
Wire Tracker Ablation Study

Evaluates the design components of WireTracker:
    - Full: CPD → Node Matching → Geometry Constraint
    - NoSnap: CPD → Geometry Constraint (no node matching)
    - NoGeometry: CPD → Node Matching (no geometry constraint)
    - CPDOnly: CPD only (no node matching, no geometry)
    - NoCPD: Node Matching → Geometry Constraint (no CPD)

Now uses WireTracker directly with enable_node_matching, enable_geometry_constraint, and enable_cpd flags.

Metrics:
    1. Edge Length Error (Mean % and Std %)
    2. Surface Distance RMSE (mm)
    3. Chamfer Distance (mm²)

Output:
    - 2x3 grid video comparing all 5 methods (with trajectory tails)
    - CSV with per-frame metrics
    - Summary statistics

Usage:
    python wire_tracker_ablation.py --trajectory traj2 --n_frames 100

Author: Auto-generated
Date: 2026-02-17
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

from wire_tracker import WireTracker


# ============================================================================
# METRICS
# ============================================================================

def compute_edge_metrics(keypoints: np.ndarray, edges: list, reference_lengths: np.ndarray) -> dict:
    """
    Compute comprehensive edge length metrics.
    
    Args:
        keypoints: K × 3 keypoint positions
        edges: List of (i, j) edge tuples
        reference_lengths: Array of reference edge lengths (mm)
    
    Returns:
        dict with:
            - 'pct_errors': array of percentage errors per edge
            - 'abs_errors': array of absolute errors (mm) per edge
            - 'pct_mean': mean percentage error
            - 'pct_std': std of percentage error
            - 'pct_max': max percentage error (this frame)
            - 'rmse_mm': RMSE of absolute edge length error (mm)
            - 'under_2pct': fraction of edges with <2% error
            - 'under_5pct': fraction of edges with <5% error
            - 'under_10pct': fraction of edges with <10% error
    """
    if keypoints is None or len(keypoints) == 0 or edges is None or len(edges) == 0:
        return {
            'pct_errors': np.array([]), 'abs_errors': np.array([]),
            'pct_mean': 0.0, 'pct_std': 0.0, 'pct_max': 0.0, 'rmse_mm': 0.0,
            'under_2pct': 0.0, 'under_5pct': 0.0, 'under_10pct': 0.0,
        }
    
    pct_errors = []
    abs_errors = []
    
    for edge_idx, (i, j) in enumerate(edges):
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        current_length = np.linalg.norm(keypoints[i] - keypoints[j])
        ref_length = reference_lengths[edge_idx]
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


def compute_position_metrics(keypoints: np.ndarray, skeleton_pc: np.ndarray) -> dict:
    """
    Compute position accuracy metrics.
    
    Args:
        keypoints: K × 3 keypoint positions
        skeleton_pc: N × 3 skeleton point cloud
    
    Returns:
        dict with:
            - 'distances': array of distances per keypoint (mm)
            - 'rmse_mm': Surface RMSE (mm)
            - 'under_2mm': fraction of keypoints within 2mm
            - 'under_5mm': fraction of keypoints within 5mm
            - 'under_10mm': fraction of keypoints within 10mm
    """
    if keypoints is None or len(keypoints) == 0 or skeleton_pc is None or len(skeleton_pc) == 0:
        return {
            'distances': np.array([]),
            'rmse_mm': 0.0,
            'under_2mm': 0.0, 'under_5mm': 0.0, 'under_10mm': 0.0,
        }
    
    # Find nearest skeleton point for each keypoint
    nn = NearestNeighbors(n_neighbors=1).fit(skeleton_pc)
    distances, _ = nn.kneighbors(keypoints)
    distances = distances.flatten()
    
    return {
        'distances': distances,
        'rmse_mm': np.sqrt(np.mean(distances ** 2)),
        'under_2mm': np.mean(distances < 2.0) * 100,   # percentage of keypoints
        'under_5mm': np.mean(distances < 5.0) * 100,   # percentage of keypoints
        'under_10mm': np.mean(distances < 10.0) * 100, # percentage of keypoints
    }


# ============================================================================
# VISUALIZATION WITH TRAJECTORY TAILS
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
    traj_history: np.ndarray = None,
    tail_length: int = 60,
) -> np.ndarray:
    """
    Create visualization for a single method with trajectory tails.
    
    Shows:
        - RGB with skeleton overlay
        - Trajectory tails (light green, fading)
        - Keypoints and edges
        - Metrics text
    
    Args:
        rgb: H × W × 3 RGB image
        skeleton_mask: H × W skeleton mask
        keypoints_2d: K × 2 keypoint positions (row, col)
        edges: List of (i, j) edge tuples
        method_name: Name of the method for display
        metrics: Dict of metrics to display
        n_branch: Number of branch nodes
        n_leaf: Number of leaf nodes
        frame_idx: Current frame index
        traj_history: T × K × 2 trajectory history (row, col)
        tail_length: Number of frames for trajectory tail
    """
    H, W = rgb.shape[:2]
    
    # Colors
    SKELETON_COLOR = [0, 191, 255]  # Deep sky blue
    EDGE_COLOR = [50, 205, 50]      # Lime green
    BRANCH_COLOR = [128, 0, 128]    # Purple
    LEAF_COLOR = [255, 255, 0]      # Yellow
    INTER_COLOR = [255, 165, 0]     # Orange
    TAIL_COLOR = [100, 255, 100]    # Light green for trajectory tail
    
    # Create overlay
    vis = rgb.copy()
    
    # Draw skeleton
    skeleton_thick = cv2.dilate(skeleton_mask, np.ones((3, 3), np.uint8), iterations=1)
    vis[skeleton_thick > 0] = SKELETON_COLOR
    
    # Draw trajectory tails
    if traj_history is not None and len(traj_history) > 1:
        n_history = len(traj_history)
        n_keypoints = traj_history.shape[1] if len(traj_history.shape) > 1 else 0
        
        # Draw trajectory for each keypoint
        for k_idx in range(n_keypoints):
            # Get valid history for this keypoint
            start_idx = max(0, n_history - tail_length)
            
            for t in range(start_idx, n_history - 1):
                pt1 = traj_history[t, k_idx]
                pt2 = traj_history[t + 1, k_idx]
                
                # Skip invalid points
                if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
                    continue
                if np.any(np.isinf(pt1)) or np.any(np.isinf(pt2)):
                    continue
                
                row1, col1 = int(pt1[0]), int(pt1[1])
                row2, col2 = int(pt2[0]), int(pt2[1])
                
                # Bounds check
                if not (0 <= row1 < H and 0 <= col1 < W):
                    continue
                if not (0 <= row2 < H and 0 <= col2 < W):
                    continue
                
                # Fade based on age
                age = n_history - 1 - t
                alpha = max(0.2, 1.0 - age / tail_length)
                color = [int(c * alpha) for c in TAIL_COLOR]
                
                cv2.line(vis, (col1, row1), (col2, row2), color, 2)
    
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


def create_2x3_grid_visualization(
    rgb: np.ndarray,
    skeleton_mask: np.ndarray,
    results: dict,
    method_names: list,
    metrics_dict: dict,
    n_branch: int,
    n_leaf: int,
    frame_idx: int,
    traj_histories: dict = None,
    tail_length: int = 60,
) -> np.ndarray:
    """
    Create 2x3 grid comparing 5 methods with trajectory tails.
    
    Layout:
        [Full]       [NoSnap]     [NoGeometry]
        [CPDOnly]    [NoCPD]      [Empty/Info]
    
    Args:
        rgb: H × W × 3 RGB image
        skeleton_mask: H × W skeleton mask
        results: Dict of method_name -> result dict
        method_names: List of method names in order
        metrics_dict: Dict of method_name -> metrics dict
        n_branch: Number of branch nodes
        n_leaf: Number of leaf nodes
        frame_idx: Current frame index
        traj_histories: Dict of method_name -> T × K × 2 trajectory history
        tail_length: Number of frames for trajectory tail
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
            rgb, skeleton_mask, keypoints_2d, edges,
            method, metrics, n_branch, n_leaf, frame_idx,
            traj_history=traj_history, tail_length=tail_length
        )
        vis_list.append(vis)
    
    # Create blank panel for 6th slot (or info panel)
    info_panel = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.putText(info_panel, "Ablation Study", (W//4, H//3),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(info_panel, f"Frame: {frame_idx}", (W//4, H//2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(info_panel, "5 Methods Comparison", (W//4, 2*H//3),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    vis_list.append(info_panel)
    
    # Arrange in 2x3 grid
    row1 = np.concatenate([vis_list[0], vis_list[1], vis_list[2]], axis=1)
    row2 = np.concatenate([vis_list[3], vis_list[4], vis_list[5]], axis=1)
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
            'ee_pose_path': Path("./data/arm_traj1/ee_pose_output/ee_poses_3d.npy"),
            'arm_green_frame': 66,
            'full_green_frame': 66,
        },
        'traj2': {
            'arm_data_path': Path("./data/arm_traj2/arm_traj2.npy"),
            'full_data_path': Path("./data/arm_traj2/arm_with_wires_traj2.npy"),
            'output_dir': Path("./data/arm_traj2/ablation_output"),
            'precomputed_mask_dir': Path("./data/arm_traj2/masks"),
            'ee_pose_path': Path("./data/arm_traj2/ee_pose_output/ee_poses_3d.npy"),
            'arm_green_frame': 0,
            'full_green_frame': 0,
        },
        'traj3': {
            'arm_data_path': Path("./data/arm_traj3/arm_traj3_contact.npy"),
            'full_data_path': Path("./data/arm_traj3/arm_with_wires_traj3_contact.npy"),
            'output_dir': Path("./data/arm_traj3/ablation_output"),
            'precomputed_mask_dir': None,
            'ee_pose_path': Path("./data/arm_traj3/ee_pose_output/ee_poses_3d.npy"),
            'arm_green_frame': 84,
            'full_green_frame': 100,
        },
    }
    
    config = trajectory_configs[args.trajectory]
    
    # Load EE poses if available
    ee_poses_3d = None
    if config['ee_pose_path'].exists():
        ee_data = np.load(str(config['ee_pose_path']), allow_pickle=True).item()
        ee_poses_3d = ee_data['ee_3d']  # Shape: (n_frames, 2, 3)
        print(f"Loaded EE poses from: {config['ee_pose_path']}")
        print(f"  Shape: {ee_poses_3d.shape}")
    else:
        print(f"No EE poses found at: {config['ee_pose_path']}")
    
    # Base tracker parameters (shared across all methods)
    base_tracker_params = {
        'intrinsics': intrinsics,
        'n_keypoints': 21,
        'target_branch_nodes': 2,
        'target_leaf_nodes': 4,
        'bg_threshold': 80.0,
        'max_depth': 1000.0,
        'top_k_components': 5,
        'arm_dilation_pixels': 5,

        'cpd_beta': 0.1,           # Reduced from 10.0 for less rigid deformation
        'cpd_lambda': 0.1,         # Reduced from 2.0 for less regularization
        'cpd_w': 0.05,
        'cpd_max_iter': 300,       # Increased from 100 for better convergence
        'cpd_tol': 1e-5,           # Tighter tolerance for better convergence

        'n_outer_iterations': 10,
        'n_edge_iterations': 30,
        'edge_weight': 0.8,
        'edge_tolerance': 0.05,

        'repulsion_iterations': 200,
        'repulsion_lr': 10.0,
        'repulsion_k_neighbors': 3,
        
        # End-effector poses
        'ee_poses_3d': ee_poses_3d,
    }
    
    # Ablation configurations using WireTracker flags
    ablation_configs = {
        'Full': {
            'enable_cpd': True,
            'enable_node_matching': True,
            'enable_geometry_constraint': True,
        },
        'NoSnap': {
            'enable_cpd': True,
            'enable_node_matching': False,
            'enable_geometry_constraint': True,
        },
        'NoGeometry': {
            'enable_cpd': True,
            'enable_node_matching': True,
            'enable_geometry_constraint': False,
        },
        'CPDOnly': {
            'enable_cpd': True,
            'enable_node_matching': False,
            'enable_geometry_constraint': False,
        },
        'NoCPD': {
            'enable_cpd': False,
            'enable_node_matching': True,
            'enable_geometry_constraint': True,
        },
    }
    
    method_names = ['Full', 'NoSnap', 'NoGeometry', 'CPDOnly', 'NoCPD']
    
    # Trajectory tail parameters
    tail_length = 60
    
    # ========================================================================
    # Load Data
    # ========================================================================
    
    print("=" * 70)
    print("WIRE TRACKER ABLATION STUDY")
    print("=" * 70)
    print(f"\nTrajectory: {args.trajectory}")
    print(f"Methods: {method_names}")
    print(f"Using WireTracker with ablation flags (enable_node_matching, enable_geometry_constraint)")
    
    print(f"\nLoading arm-only data from: {config['arm_data_path']}")
    arm_only_data = np.load(str(config['arm_data_path']), allow_pickle=True).item()
    
    print(f"Loading full scene data from: {config['full_data_path']}")
    full_scene_data = np.load(str(config['full_data_path']), allow_pickle=True).item()
    
    # Synchronize sequences
    arm_frame_keys = sorted(arm_only_data.keys())[config['arm_green_frame']:]
    full_frame_keys = sorted(full_scene_data.keys())[config['full_green_frame']:]
    
    n_frames = min(len(arm_frame_keys), len(full_frame_keys))
    
    # If using precomputed masks, limit to available masks (accounting for green_frame offset)
    # Masks are named by original frame index, so we need masks from green_frame onwards
    if config['precomputed_mask_dir'] is not None:
        available_mask_count = 0
        for mask_idx in range(config['full_green_frame'], config['full_green_frame'] + n_frames):
            mask_path = config['precomputed_mask_dir'] / f"mask_frame_{mask_idx:04d}.npy"
            if mask_path.exists():
                available_mask_count += 1
            else:
                break  # Stop at first missing mask
        
        if available_mask_count < n_frames:
            print(f"Warning: Only {available_mask_count} masks available (from frame {config['full_green_frame']}), limiting frames")
            n_frames = available_mask_count
    
    if args.n_frames is not None:
        n_frames = min(n_frames, args.n_frames)
    
    arm_frame_keys = arm_frame_keys[:n_frames]
    full_frame_keys = full_frame_keys[:n_frames]
    
    print(f"\nTotal frames to process: {n_frames}")
    
    # Create output directory
    output_dir = config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Initialize Trackers (using WireTracker directly with different flags)
    # ========================================================================
    
    print("\nInitializing trackers...")
    trackers = {}
    for name in method_names:
        # Combine base params with ablation-specific flags
        tracker_params = {**base_tracker_params, **ablation_configs[name]}
        trackers[name] = WireTracker(**tracker_params)
        print(f"  {name}: enable_cpd={ablation_configs[name]['enable_cpd']}, "
              f"enable_node_matching={ablation_configs[name]['enable_node_matching']}, "
              f"enable_geometry_constraint={ablation_configs[name]['enable_geometry_constraint']}")
    
    # ========================================================================
    # Process Frames
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("PROCESSING FRAMES")
    print("=" * 70)
    
    # Storage for metrics
    all_metrics = {name: [] for name in method_names}
    
    # Trajectory history for each method (for visualization)
    traj_histories = {name: [] for name in method_names}
    
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
        # Masks are named by original frame index, so add green_frame offset
        precomputed_arm_mask = None
        if config['precomputed_mask_dir'] is not None:
            original_frame_idx = i + config['full_green_frame']
            mask_path = config['precomputed_mask_dir'] / f"mask_frame_{original_frame_idx:04d}.npy"
            if mask_path.exists():
                precomputed_arm_mask = np.load(str(mask_path))
        
        # Process frame with all methods
        results = {}
        metrics_dict = {}
        skeleton_mask = None
        skeleton_pc = None
        
        for name in method_names:
            tracker = trackers[name]
            result = tracker.process_frame(
                full_depth, arm_depth, full_rgb, precomputed_arm_mask
            )
            results[name] = result
            
            # Get skeleton from first tracker (same for all)
            if skeleton_mask is None and 'skeleton_mask' in result:
                skeleton_mask = result['skeleton_mask']
            if skeleton_pc is None and 'skeleton_pc' in result:
                skeleton_pc = result['skeleton_pc']
            
            # Update trajectory history
            if result['success'] and 'keypoints_2d' in result:
                traj_histories[name].append(result['keypoints_2d'].copy())
            
            # Compute metrics
            if result['success'] and tracker.is_initialized:
                keypoints = result['keypoints']
                edges = tracker.reference_edges
                ref_lengths = tracker.reference_lengths
                
                # Edge metrics
                skel_pc = result.get('skeleton_pc', skeleton_pc)
                edge_metrics = compute_edge_metrics(keypoints, edges, ref_lengths)
                pos_metrics = compute_position_metrics(keypoints, skel_pc)
                
                metrics = {
                    # Edge metrics
                    'edge_pct_mean': edge_metrics['pct_mean'],
                    'edge_pct_std': edge_metrics['pct_std'],
                    'edge_pct_max': edge_metrics['pct_max'],
                    'edge_rmse_mm': edge_metrics['rmse_mm'],
                    'edge_under_2pct': edge_metrics['under_2pct'],
                    'edge_under_5pct': edge_metrics['under_5pct'],
                    'edge_under_10pct': edge_metrics['under_10pct'],
                    # Position metrics
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
        
        # Get reference topology info from first tracker
        n_branch = trackers['Full'].reference_n_branch
        n_leaf = trackers['Full'].reference_n_leaf
        
        # Prepare trajectory histories as arrays for visualization
        traj_hist_arrays = {}
        for name in method_names:
            if len(traj_histories[name]) > 0:
                traj_hist_arrays[name] = np.array(traj_histories[name])
            else:
                traj_hist_arrays[name] = None
        
        # Create 2x3 grid visualization with trajectory tails
        if skeleton_mask is None:
            skeleton_mask = np.zeros_like(full_depth, dtype=np.uint8)
        
        grid = create_2x3_grid_visualization(
            full_rgb, skeleton_mask, results, method_names,
            metrics_dict, n_branch, n_leaf, i,
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
        metrics_str = " | ".join([
            f"{name}: E={metrics_dict[name]['edge_pct_mean']:.1f}% P={metrics_dict[name]['pos_rmse_mm']:.1f}"
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
            # Edge % error
            'edge_pct_mean_avg': np.mean(edge_pct_means) if edge_pct_means else 0,
            'edge_pct_mean_std': np.std(edge_pct_means) if edge_pct_means else 0,
            # Max edge % error (average of per-frame max)
            'edge_pct_max_avg': np.mean(edge_pct_maxes) if edge_pct_maxes else 0,
            # Max edge % error (absolute max over all frames)
            'edge_pct_max_abs': np.max(edge_pct_maxes) if edge_pct_maxes else 0,
            # Edge RMSE (mm)
            'edge_rmse_avg': np.mean(edge_rmses) if edge_rmses else 0,
            'edge_rmse_std': np.std(edge_rmses) if edge_rmses else 0,
            # Edge success rates
            'edge_under_2pct': np.mean(edge_under_2) if edge_under_2 else 0,
            'edge_under_5pct': np.mean(edge_under_5) if edge_under_5 else 0,
            'edge_under_10pct': np.mean(edge_under_10) if edge_under_10 else 0,
            # Position RMSE
            'pos_rmse_avg': np.mean(pos_rmses) if pos_rmses else 0,
            'pos_rmse_std': np.std(pos_rmses) if pos_rmses else 0,
            # Position success rates
            'pos_under_2mm': np.mean(pos_under_2) if pos_under_2 else 0,
            'pos_under_5mm': np.mean(pos_under_5) if pos_under_5 else 0,
            'pos_under_10mm': np.mean(pos_under_10) if pos_under_10 else 0,
        }
        summary_data.append(summary)
    
    # Print summary table - Edge Metrics
    print(f"\n{'='*70}")
    print("EDGE LENGTH METRICS")
    print("=" * 70)
    print(f"{'Method':<12} | {'Edge % Mean':<14} | {'Edge RMSE':<14} | {'Max%(Avg)':<10} | {'Max%(Abs)':<10} | {'<2%':<6} | {'<5%':<6} | {'<10%':<6}")
    print("-" * 100)
    
    for s in summary_data:
        print(f"{s['method']:<12} | "
              f"{s['edge_pct_mean_avg']:>5.2f}% ±{s['edge_pct_mean_std']:>4.2f}% | "
              f"{s['edge_rmse_avg']:>5.2f} ±{s['edge_rmse_std']:>4.2f}mm | "
              f"{s['edge_pct_max_avg']:>7.2f}% | "
              f"{s['edge_pct_max_abs']:>7.2f}% | "
              f"{s['edge_under_2pct']:>5.1f}% | "
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
        # Header
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
        f.write("Frame,Method,EdgePctMean,EdgePctMax,EdgeRMSE,PosRMSE,Edge<2%,Edge<5%,Edge<10%,Pos<2mm,Pos<5mm,Pos<10mm\n")
        for frame_idx in range(n_frames):
            for name in method_names:
                if frame_idx < len(all_metrics[name]):
                    m = all_metrics[name][frame_idx]
                    f.write(f"{frame_idx},{name},{m['edge_pct_mean']:.4f},{m['edge_pct_max']:.4f},"
                            f"{m['edge_rmse_mm']:.4f},{m['pos_rmse_mm']:.4f},"
                            f"{m['edge_under_2pct']:.2f},{m['edge_under_5pct']:.2f},{m['edge_under_10pct']:.2f},"
                            f"{m['pos_under_2mm']:.2f},{m['pos_under_5mm']:.2f},{m['pos_under_10mm']:.2f}\n")
    print(f"Per-frame metrics saved to: {per_frame_path}")
    
    # ========================================================================
    # Generate Metrics Over Time Plot
    # ========================================================================
    
    import matplotlib.pyplot as plt
    
    print(f"\n{'='*70}")
    print("GENERATING METRICS PLOT")
    print("=" * 70)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    colors = {'Full': 'blue', 'NoSnap': 'orange', 'NoGeometry': 'green', 'CPDOnly': 'red', 'NoCPD': 'purple'}
    
    for name in method_names:
        metrics_list = all_metrics[name]
        frames = list(range(len(metrics_list)))
        
        # Edge % Error
        edge_pct = [m['edge_pct_mean'] for m in metrics_list]
        ax1.plot(frames, edge_pct, label=name, color=colors.get(name, 'gray'), linewidth=1.5)
        
        # Position RMSE
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
    
    # Find indices for comparison
    idx_full = next((i for i, s in enumerate(summary_data) if s['method'] == 'Full'), None)
    idx_no_snap = next((i for i, s in enumerate(summary_data) if s['method'] == 'NoSnap'), None)
    idx_no_geom = next((i for i, s in enumerate(summary_data) if s['method'] == 'NoGeometry'), None)
    idx_cpd_only = next((i for i, s in enumerate(summary_data) if s['method'] == 'CPDOnly'), None)
    idx_no_cpd = next((i for i, s in enumerate(summary_data) if s['method'] == 'NoCPD'), None)
    
    if all(idx is not None for idx in [idx_full, idx_no_snap, idx_no_geom, idx_cpd_only, idx_no_cpd]):
        full = summary_data[idx_full]
        no_snap = summary_data[idx_no_snap]
        no_geom = summary_data[idx_no_geom]
        cpd_only = summary_data[idx_cpd_only]
        no_cpd = summary_data[idx_no_cpd]
        
        print("\n1. Node Matching Contribution (Full vs NoSnap):")
        print(f"   Edge % Error reduction: {no_snap['edge_pct_mean_avg'] - full['edge_pct_mean_avg']:.2f}%")
        print(f"   Position RMSE reduction: {no_snap['pos_rmse_avg'] - full['pos_rmse_avg']:.2f} mm")
        
        print("\n2. Geometry Constraint Contribution (Full vs NoGeometry):")
        print(f"   Edge % Error reduction: {no_geom['edge_pct_mean_avg'] - full['edge_pct_mean_avg']:.2f}%")
        print(f"   Position RMSE reduction: {no_geom['pos_rmse_avg'] - full['pos_rmse_avg']:.2f} mm")
        
        print("\n3. CPD Registration Contribution (Full vs NoCPD):")
        print(f"   Edge % Error reduction: {no_cpd['edge_pct_mean_avg'] - full['edge_pct_mean_avg']:.2f}%")
        print(f"   Position RMSE reduction: {no_cpd['pos_rmse_avg'] - full['pos_rmse_avg']:.2f} mm")
        
        print("\n4. Combined Components vs CPD Only:")
        print(f"   Edge % Error reduction: {cpd_only['edge_pct_mean_avg'] - full['edge_pct_mean_avg']:.2f}%")
        print(f"   Position RMSE reduction: {cpd_only['pos_rmse_avg'] - full['pos_rmse_avg']:.2f} mm")
    
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
