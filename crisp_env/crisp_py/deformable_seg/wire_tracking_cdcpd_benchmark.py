"""
CDCPD2 Benchmark Script

Compares CDCPD2 tracker against WireTracker on trajectories 1, 2, 3.
Uses the same metrics as wire_tracker_ablation.py.

Usage:
    python wire_tracking_cdcpd_benchmark.py --trajectory traj1
    python wire_tracking_cdcpd_benchmark.py --all

Author: Auto-generated
Date: 2026-02-18
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
from wire_tracker import WireTracker


# ============================================================================
# METRICS (same as wire_tracker_ablation.py)
# ============================================================================

def compute_edge_metrics(keypoints: np.ndarray, edges: list, reference_lengths: np.ndarray) -> dict:
    """Compute comprehensive edge length metrics."""
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
        'pct_mean': np.mean(pct_errors) * 100,
        'pct_std': np.std(pct_errors) * 100,
        'pct_max': np.max(pct_errors) * 100,
        'rmse_mm': np.sqrt(np.mean(abs_errors ** 2)),
        'under_2pct': np.mean(pct_errors < 0.02) * 100,
        'under_5pct': np.mean(pct_errors < 0.05) * 100,
        'under_10pct': np.mean(pct_errors < 0.10) * 100,
    }


def compute_position_metrics(keypoints: np.ndarray, skeleton_pc: np.ndarray) -> dict:
    """Compute position accuracy metrics."""
    if keypoints is None or len(keypoints) == 0 or skeleton_pc is None or len(skeleton_pc) == 0:
        return {
            'distances': np.array([]),
            'rmse_mm': 0.0,
            'under_2mm': 0.0, 'under_5mm': 0.0, 'under_10mm': 0.0,
        }
    
    nn = NearestNeighbors(n_neighbors=1).fit(skeleton_pc)
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
    skeleton_mask: np.ndarray,
    keypoints_2d: np.ndarray,
    edges: list,
    method_name: str,
    metrics: dict,
    n_branch: int,
    n_leaf: int,
    frame_idx: int,
) -> np.ndarray:
    """Create visualization for a single method."""
    H, W = rgb.shape[:2]
    
    # Colors
    SKELETON_COLOR = [0, 191, 255]  # Deep sky blue
    EDGE_COLOR = [50, 205, 50]      # Lime green
    BRANCH_COLOR = [128, 0, 128]    # Purple
    LEAF_COLOR = [255, 255, 0]      # Yellow
    INTER_COLOR = [255, 165, 0]     # Orange
    
    vis = rgb.copy()
    
    # Draw skeleton
    if skeleton_mask is not None:
        skeleton_thick = cv2.dilate(skeleton_mask, np.ones((3, 3), np.uint8), iterations=1)
        vis[skeleton_thick > 0] = SKELETON_COLOR
    
    # Draw edges and keypoints
    if keypoints_2d is not None and len(keypoints_2d) > 0 and edges is not None:
        kp_int = keypoints_2d.astype(int)
        for (i, j) in edges:
            if i < len(kp_int) and j < len(kp_int):
                pt1 = (kp_int[i, 1], kp_int[i, 0])
                pt2 = (kp_int[j, 1], kp_int[j, 0])
                cv2.line(vis, pt1, pt2, EDGE_COLOR, 2)
        
        for idx, (row, col) in enumerate(kp_int):
            if 0 <= row < H and 0 <= col < W:
                if idx < n_branch:
                    color = BRANCH_COLOR
                elif idx < n_branch + n_leaf:
                    color = LEAF_COLOR
                else:
                    color = INTER_COLOR
                cv2.circle(vis, (col, row), 6, color, -1)
    
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
# CDCPD WRAPPER
# ============================================================================

class CDCPDWireTracker:
    """
    Wrapper around CDCPDTracker to match WireTracker interface.
    
    Uses WireTracker for initialization, then CDCPD for subsequent tracking.
    Uses EE poses as anchor constraints (like gripper constraints in CDCPD2 paper).
    """
    
    def __init__(self, intrinsics, n_keypoints=21, ee_poses_3d=None, **kwargs):
        self.intrinsics = intrinsics
        self.n_keypoints = n_keypoints
        self.ee_poses_3d = ee_poses_3d  # (n_frames, 2, 3) EE positions
        
        # Remove ee_poses_3d from kwargs to avoid passing twice
        kwargs_no_ee = {k: v for k, v in kwargs.items() if k != 'ee_poses_3d'}
        
        # WireTracker for initialization and skeleton extraction (WITH EE injection for same init)
        self.init_tracker = WireTracker(
            intrinsics=intrinsics,
            n_keypoints=n_keypoints,
            ee_poses_3d=ee_poses_3d,  # Use EE injection for identical initialization
            **kwargs_no_ee
        )
        
        # CDCPD tracker (use same beta/lambda as WireTracker CPD)
        # Use anchor_hard=True to match paper's hard gripper constraints in QP
        self.cdcpd = CDCPDTracker(
            cpd_beta=0.1,       # Same as WireTracker
            cpd_lambda=0.1,     # Same as WireTracker
            cpd_w=0.05,         # Same as WireTracker
            cpd_max_iter=100,
            cpd_tol=1e-3,
            lle_neighbors=6,
            lle_gamma=0.5,
            stretch_lambda=1.00,  # Allow 0% stretch
            use_qp_optimization=True,
            qp_max_iter=200,
            use_anchor_constraints=True,
            anchor_weight=100.0,  # Soft constraint weight in CPD
            anchor_hard=True,     # HARD constraints in QP (like paper)
        )
        
        self.is_initialized = False
        self.prev_keypoints = None
        self.reference_edges = None
        self.reference_lengths = None
        self.reference_n_branch = 0
        self.reference_n_leaf = 0
        self.frame_count = 0
        self.ee_leaf_indices = None  # Which leaf nodes correspond to EE
        
    def process_frame(self, full_depth, arm_depth, full_rgb, precomputed_arm_mask=None):
        """Process a single frame using CDCPD with EE poses as anchors."""
        
        frame_idx = self.frame_count
        self.frame_count += 1
        
        # First frame: use WireTracker for initialization
        # Note: init_tracker already has ee_poses_3d so it does EE injection automatically
        if not self.is_initialized:
            result = self.init_tracker.process_frame(
                full_depth, arm_depth, full_rgb, precomputed_arm_mask
            )
            
            if result['success'] and self.init_tracker.is_initialized:
                self.is_initialized = True
                self.prev_keypoints = result['keypoints'].copy()
                self.reference_edges = self.init_tracker.reference_edges
                self.reference_lengths = self.init_tracker.reference_lengths.copy()
                self.reference_n_branch = self.init_tracker.reference_n_branch
                self.reference_n_leaf = self.init_tracker.reference_n_leaf
                
                # Get ee_leaf_indices from init_tracker (already established during init)
                # init_tracker.ee_to_leaf_mapping is {0: kp_idx, 1: kp_idx}
                if self.init_tracker.ee_to_leaf_mapping is not None:
                    self.ee_leaf_indices = [
                        self.init_tracker.ee_to_leaf_mapping[0],
                        self.init_tracker.ee_to_leaf_mapping[1]
                    ]
                    # Note: keypoints already have EE injected by init_tracker
            
            return result
        
        # Subsequent frames: use CDCPD
        # Get skeleton from init_tracker (for preprocessing only)
        result = self.init_tracker.process_frame(
            full_depth, arm_depth, full_rgb, precomputed_arm_mask
        )
        
        if not result['success']:
            return result
        
        skeleton_pc = result.get('skeleton_pc')
        if skeleton_pc is None or len(skeleton_pc) == 0:
            return {'success': False, 'keypoints': None}
        
        # Get EE poses for this frame as anchor constraints
        anchor_indices = []
        anchor_positions = []
        
        if self.ee_poses_3d is not None and frame_idx < len(self.ee_poses_3d) and self.ee_leaf_indices is not None:
            ee_pos = self.ee_poses_3d[frame_idx]  # (2, 3)
            for i, ee_idx in enumerate(self.ee_leaf_indices):
                anchor_indices.append(ee_idx)
                anchor_positions.append(ee_pos[i])
        
        anchor_indices = np.array(anchor_indices) if anchor_indices else None
        anchor_positions = np.array(anchor_positions) if anchor_positions else None
        
        # Run CDCPD tracking with EE anchors
        cdcpd_result = self.cdcpd.track_frame_with_anchors(
            prev_keypoints=self.prev_keypoints,
            skeleton_pc=skeleton_pc,
            anchor_indices=anchor_indices,
            anchor_positions=anchor_positions,
            reference_edges=self.reference_edges,
            reference_lengths=self.reference_lengths,
        )
        
        keypoints = cdcpd_result['keypoints']
        self.prev_keypoints = keypoints.copy()
        
        # Project to 2D
        keypoints_2d = self._project_to_2d(keypoints)
        
        return {
            'success': True,
            'keypoints': keypoints,
            'keypoints_2d': keypoints_2d,
            'edges': self.reference_edges,
            'skeleton_pc': skeleton_pc,
            'skeleton_mask': result.get('skeleton_mask'),
            'timing': cdcpd_result.get('timing', {}),
        }
    
    def _project_to_2d(self, keypoints_3d):
        """Project 3D keypoints to 2D image coordinates."""
        if keypoints_3d is None or len(keypoints_3d) == 0:
            return np.array([])
        
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        
        keypoints_2d = []
        for pt in keypoints_3d:
            x, y, z = pt
            if z > 0:
                col = fx * x / z + cx
                row = fy * y / z + cy
            else:
                col, row = 0, 0
            keypoints_2d.append([row, col])
        
        return np.array(keypoints_2d)


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark(args):
    """Run CDCPD vs WireTracker benchmark."""
    
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
            'output_dir': Path("./data/arm_traj1/cdcpd_benchmark"),
            'precomputed_mask_dir': None,
            'ee_pose_path': Path("./data/arm_traj1/ee_pose_output/ee_poses_3d.npy"),
            'arm_green_frame': 66,
            'full_green_frame': 66,
        },
        'traj2': {
            'arm_data_path': Path("./data/arm_traj2/arm_traj2.npy"),
            'full_data_path': Path("./data/arm_traj2/arm_with_wires_traj2.npy"),
            'output_dir': Path("./data/arm_traj2/cdcpd_benchmark"),
            'precomputed_mask_dir': Path("./data/arm_traj2/masks"),
            'ee_pose_path': Path("./data/arm_traj2/ee_pose_output/ee_poses_3d.npy"),
            'arm_green_frame': 0,
            'full_green_frame': 0,
        },
        'traj3': {
            'arm_data_path': Path("./data/arm_traj3/arm_traj3_contact.npy"),
            'full_data_path': Path("./data/arm_traj3/arm_with_wires_traj3_contact.npy"),
            'output_dir': Path("./data/arm_traj3/cdcpd_benchmark"),
            'precomputed_mask_dir': None,
            'ee_pose_path': Path("./data/arm_traj3/ee_pose_output/ee_poses_3d.npy"),
            'arm_green_frame': 84,
            'full_green_frame': 100,
        },
    }
    
    config = trajectory_configs[args.trajectory]
    
    # Load EE poses
    ee_poses_3d = None
    if config['ee_pose_path'].exists():
        ee_data = np.load(str(config['ee_pose_path']), allow_pickle=True).item()
        ee_poses_3d = ee_data['ee_3d']
        print(f"Loaded EE poses from: {config['ee_pose_path']}")
    
    # Base tracker parameters (same as wire_tracking_main.py)
    base_params = {
        'intrinsics': intrinsics,
        'n_keypoints': 21,
        'target_branch_nodes': 2,
        'target_leaf_nodes': 4,
        # Segmentation
        'bg_threshold': 80.0,
        'max_depth': 1000.0,
        'top_k_components': 5,
        'arm_dilation_pixels': 5,
        # CPD
        # 'cpd_beta': 0.1,
        # 'cpd_lambda': 0.1,
        'cpd_beta': 50.0,
        'cpd_lambda': 0.1,
        'cpd_w': 0.05,
        'cpd_max_iter': 100,
        # Geometry constraints
        'n_outer_iterations': 10,
        'n_edge_iterations': 30,
        'edge_weight': 0.5,
        'edge_tolerance': 0.05,
        # Repulsion
        'repulsion_iterations': 200,
        'repulsion_lr': 10.0,
        'repulsion_k_neighbors': 3,
        # End-effector poses
        'ee_poses_3d': ee_poses_3d,
    }
    
    # Methods to compare
    method_names = ['WireTracker', 'CDCPD']
    
    print("=" * 70)
    print("CDCPD vs WireTracker BENCHMARK")
    print("=" * 70)
    print(f"\nTrajectory: {args.trajectory}")
    
    # Load data
    print(f"\nLoading data...")
    arm_only_data = np.load(str(config['arm_data_path']), allow_pickle=True).item()
    full_scene_data = np.load(str(config['full_data_path']), allow_pickle=True).item()
    
    arm_frame_keys = sorted(arm_only_data.keys())[config['arm_green_frame']:]
    full_frame_keys = sorted(full_scene_data.keys())[config['full_green_frame']:]
    
    n_frames = min(len(arm_frame_keys), len(full_frame_keys))
    
    if config['precomputed_mask_dir'] is not None:
        available_mask_count = 0
        for mask_idx in range(config['full_green_frame'], config['full_green_frame'] + n_frames):
            mask_path = config['precomputed_mask_dir'] / f"mask_frame_{mask_idx:04d}.npy"
            if mask_path.exists():
                available_mask_count += 1
            else:
                break
        if available_mask_count < n_frames:
            n_frames = available_mask_count
    
    if args.n_frames is not None:
        n_frames = min(n_frames, args.n_frames)
    
    arm_frame_keys = arm_frame_keys[:n_frames]
    full_frame_keys = full_frame_keys[:n_frames]
    
    print(f"Total frames to process: {n_frames}")
    
    # Create output directory
    output_dir = config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trackers
    print("\nInitializing trackers...")
    trackers = {
        'WireTracker': WireTracker(**base_params),
        'CDCPD': CDCPDWireTracker(**base_params),
    }
    
    # Process frames
    print(f"\n{'='*70}")
    print("PROCESSING FRAMES")
    print("=" * 70)
    
    all_metrics = {name: [] for name in method_names}
    all_results = {name: [] for name in method_names}  # Store results for video
    
    # Video writer
    video_writer = None
    fps = 30
    
    for i in range(n_frames):
        frame_start = time.time()
        
        arm_frame_key = arm_frame_keys[i]
        full_frame_key = full_frame_keys[i]
        
        arm_data = arm_only_data[arm_frame_key]
        arm_depth = arm_data['transformed_depth'].copy()
        
        full_data = full_scene_data[full_frame_key]
        full_rgb = full_data['color'][:, :, ::-1]
        full_depth = full_data['transformed_depth'].copy()
        
        precomputed_arm_mask = None
        if config['precomputed_mask_dir'] is not None:
            original_frame_idx = i + config['full_green_frame']
            mask_path = config['precomputed_mask_dir'] / f"mask_frame_{original_frame_idx:04d}.npy"
            if mask_path.exists():
                precomputed_arm_mask = np.load(str(mask_path))
        
        metrics_dict = {}
        results_dict = {}
        
        for name in method_names:
            tracker = trackers[name]
            result = tracker.process_frame(
                full_depth, arm_depth, full_rgb, precomputed_arm_mask
            )
            results_dict[name] = result
        
        # Verify both trackers have same initialization on frame 0
        if i == 0:
            wt = trackers['WireTracker']
            cdcpd = trackers['CDCPD']
            if wt.is_initialized and cdcpd.is_initialized:
                print(f"\n  === Frame 0 Initialization Verification ===")
                print(f"  WireTracker: {len(wt.reference_edges)} edges, lengths sum: {np.sum(wt.reference_lengths):.4f} mm")
                print(f"  CDCPD:       {len(cdcpd.reference_edges)} edges, lengths sum: {np.sum(cdcpd.reference_lengths):.4f} mm")
                print(f"  Edges match: {wt.reference_edges == cdcpd.reference_edges}")
                print(f"  Lengths match: {np.allclose(wt.reference_lengths, cdcpd.reference_lengths)}")
                kp_match = np.allclose(wt.reference_keypoints, cdcpd.prev_keypoints, atol=1e-3)
                print(f"  Keypoints match (atol=1e-3): {kp_match}")
                if not kp_match:
                    diff = np.linalg.norm(wt.reference_keypoints - cdcpd.prev_keypoints, axis=1)
                    print(f"  Max keypoint diff: {np.max(diff):.4f} mm")
                print(f"  ============================================\n")
        
        for name in method_names:
            result = results_dict[name]
            
            if result['success']:
                keypoints = result['keypoints']
                
                # Get reference from WireTracker (both use same init)
                if name == 'WireTracker':
                    ref_tracker = trackers[name]
                else:
                    ref_tracker = trackers[name].init_tracker
                
                if ref_tracker.is_initialized:
                    edges = ref_tracker.reference_edges
                    ref_lengths = ref_tracker.reference_lengths
                    skel_pc = result.get('skeleton_pc')
                    
                    edge_metrics = compute_edge_metrics(keypoints, edges, ref_lengths)
                    pos_metrics = compute_position_metrics(keypoints, skel_pc)
                    
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
            else:
                metrics = {
                    'edge_pct_mean': 0.0, 'edge_pct_max': 0.0, 'edge_rmse_mm': 0.0,
                    'edge_under_2pct': 0.0, 'edge_under_5pct': 0.0, 'edge_under_10pct': 0.0,
                    'pos_rmse_mm': 0.0, 'pos_under_2mm': 0.0, 'pos_under_5mm': 0.0, 'pos_under_10mm': 0.0,
                }
            
            metrics_dict[name] = metrics
            all_metrics[name].append(metrics)
        
        # Create side-by-side visualization
        skeleton_mask = results_dict['WireTracker'].get('skeleton_mask')
        if skeleton_mask is None:
            skeleton_mask = np.zeros_like(full_depth, dtype=np.uint8)
        
        # Get reference info
        ref_tracker = trackers['WireTracker']
        n_branch = ref_tracker.reference_n_branch if ref_tracker.is_initialized else 0
        n_leaf = ref_tracker.reference_n_leaf if ref_tracker.is_initialized else 0
        edges = ref_tracker.reference_edges if ref_tracker.is_initialized else []
        
        vis_frames = []
        for name in method_names:
            result = results_dict[name]
            metrics = metrics_dict[name]
            
            vis = create_frame_visualization(
                full_rgb, skeleton_mask,
                result.get('keypoints_2d') if result['success'] else None,
                edges, name, metrics, n_branch, n_leaf, i
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
        
        print(f"Frame {i:4d}: "
              f"WT E={metrics_dict['WireTracker']['edge_pct_mean']:.1f}% P={metrics_dict['WireTracker']['pos_rmse_mm']:.1f} | "
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
    print(f"\n{'Method':<12} | {'Edge%Mean':<12} | {'EdgeRMSE':<10} | {'Max%(Abs)':<9} | {'E<2%':<6} | {'E<5%':<6} | {'PosRMSE':<12} | {'P<2mm':<6} | {'P<5mm':<6}")
    print("-" * 110)
    
    for s in summary_data:
        print(f"{s['method']:<12} | "
              f"{s['edge_pct_mean_avg']:>4.2f}%±{s['edge_pct_mean_std']:>4.2f}% | "
              f"{s['edge_rmse_avg']:>5.2f}mm   | "
              f"{s['edge_pct_max_abs']:>7.2f}% | "
              f"{s['edge_under_2pct']:>5.1f}% | "
              f"{s['edge_under_5pct']:>5.1f}% | "
              f"{s['pos_rmse_avg']:>4.2f}±{s['pos_rmse_std']:>4.2f}mm | "
              f"{s['pos_under_2mm']:>5.1f}% | "
              f"{s['pos_under_5mm']:>5.1f}%")
    
    # Save to CSV with all metrics
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
    
    colors = {'WireTracker': 'blue', 'CDCPD': 'red'}
    
    for name in method_names:
        frames = list(range(len(all_metrics[name])))
        edge_pct = [m['edge_pct_mean'] for m in all_metrics[name]]
        pos_rmse = [m['pos_rmse_mm'] for m in all_metrics[name]]
        
        ax1.plot(frames, edge_pct, label=name, color=colors[name], linewidth=1.5)
        ax2.plot(frames, pos_rmse, label=name, color=colors[name], linewidth=1.5)
    
    ax1.set_ylabel('Edge % Error (%)')
    ax1.set_title(f'CDCPD vs WireTracker - {args.trajectory}')
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


def run_all_benchmarks():
    """Run benchmark on all trajectories."""
    for traj in ['traj1', 'traj2', 'traj3']:
        print(f"\n{'#'*70}")
        print(f"# TRAJECTORY: {traj}")
        print(f"{'#'*70}")
        
        args = argparse.Namespace(trajectory=traj, n_frames=None)
        try:
            run_benchmark(args)
        except Exception as e:
            print(f"Error on {traj}: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CDCPD vs WireTracker Benchmark")
    parser.add_argument('--trajectory', type=str, default='traj1',
                        choices=['traj1', 'traj2', 'traj3'],
                        help='Trajectory to process')
    parser.add_argument('--n_frames', type=int, default=None,
                        help='Number of frames to process')
    parser.add_argument('--all', action='store_true',
                        help='Run on all trajectories')
    
    args = parser.parse_args()
    
    if args.all:
        run_all_benchmarks()
    else:
        run_benchmark(args)
