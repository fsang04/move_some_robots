"""
BDLO Tracking Projection Ablation Study

Evaluates the impact of point cloud projection during geometry constraint:
    - Full: CPD disabled, projection enabled (blends keypoints toward point cloud)
    - NoProj: CPD disabled, projection disabled (only edge spring correction)

Both methods use EE anchor constraints. This isolates the effect of projection.

Dataset: bdlo_no_contact_4sec (same as bdlo1_batch_experiment.py)

Output (exactly matches existing evaluation format):
    - 3d_keypoints.npz: {full, noproj, edge_connection, reference_lengths}
    - per_frame.csv
    - summary.txt
    - video_ablation.mp4
    - chunk_summary/

Usage:
    python bdlo_tracking_ablation.py --chunk 0 --clip_seconds 20

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

from wire_tracker import WireTracker


# ============================================================================
# CONSTANTS
# ============================================================================

DATA_BASE = Path('/mnt/mydisk/captured_data_double_arm/bdlo_no_contact_4sec')
CALIB_DIR = Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/'
                 'roahm-deformable-objects/captured_calibration_data/test_0227')
OUTPUT_BASE = Path('./bdlo_tracking_ablation_results')

FPS = 30
N_KEYPOINTS = 25
TARGET_BRANCH_NODES = 2
TARGET_LEAF_NODES = 4


# ============================================================================
# DATA LOADING
# ============================================================================

def load_chunk_data(chunk_dir: Path) -> dict:
    """Load all data from a chunk directory (last 600 frames)."""
    rgbd = np.load(chunk_dir / 'rgbd.npz')
    color = rgbd['color'][-600:]
    depth = rgbd['depth'][-600:]
    
    masks_path = chunk_dir / 'masks' / 'masks.npz'
    if masks_path.exists():
        dlo_masks = np.load(masks_path)['masks'][-600:]
    else:
        dlo_masks = None
    
    left_poses_npz = np.load(chunk_dir / 'left_arm_poses.npz')
    right_poses_npz = np.load(chunk_dir / 'right_arm_poses.npz')
    
    n_frames_total = len(left_poses_npz.files)
    start_idx = max(0, n_frames_total - 600)
    n_frames = min(600, n_frames_total)
    left_poses = np.array([left_poses_npz[f'arr_{i}'] for i in range(start_idx, n_frames_total)])
    right_poses = np.array([right_poses_npz[f'arr_{i}'] for i in range(start_idx, n_frames_total)])
    
    return {
        'color': color,
        'depth': depth,
        'dlo_masks': dlo_masks,
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
    for edge_idx, (i, j) in enumerate(edges):
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        ref_length = reference_lengths[edge_idx]
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


def compute_position_metrics(keypoints, skeleton_pc, extra_gt_points=None):
    """Compute position metrics."""
    if keypoints is None or len(keypoints) == 0 or skeleton_pc is None or len(skeleton_pc) == 0:
        return {
            'distances': np.array([]),
            'rmse_mm': 0.0,
            'under_2mm': 0.0, 'under_5mm': 0.0, 'under_10mm': 0.0,
        }

    gt_cloud = skeleton_pc
    if extra_gt_points is not None and len(extra_gt_points) > 0:
        gt_cloud = np.vstack([skeleton_pc, extra_gt_points])

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


def sample_points_on_edges(keypoints, edges, n_target_points):
    """Uniformly sample points along predicted edges."""
    if keypoints is None or len(keypoints) == 0 or edges is None or len(edges) == 0:
        return np.empty((0, 3), dtype=np.float32)
    
    if n_target_points <= 0:
        return np.empty((0, 3), dtype=np.float32)
    
    edge_lengths = []
    for (i, j) in edges:
        if i < len(keypoints) and j < len(keypoints):
            length = np.linalg.norm(keypoints[i] - keypoints[j])
            edge_lengths.append(length)
        else:
            edge_lengths.append(0.0)
    
    total_length = sum(edge_lengths)
    if total_length < 1e-6:
        return np.empty((0, 3), dtype=np.float32)
    
    sampled_points = []
    for edge_idx, (i, j) in enumerate(edges):
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        
        n_edge = max(2, int(round(n_target_points * edge_lengths[edge_idx] / total_length)))
        t_vals = np.linspace(0, 1, n_edge)
        p_start = keypoints[i]
        p_end = keypoints[j]
        
        for t in t_vals:
            sampled_points.append(p_start + t * (p_end - p_start))
    
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
# CUSTOM TRACKER WITH CONFIGURABLE PROJECTION
# ============================================================================

class BDLOTrackerNoProj(WireTracker):
    """BDLO tracker with projection disabled during geometry constraint."""
    
    def _joint_constraint_optimization(
        self,
        keypoints: np.ndarray,
        wire_points: np.ndarray,
    ) -> np.ndarray:
        """
        BDLO geometry constraint with projection DISABLED.
        Only applies sequential edge correction (no projection to wire point cloud).
        
        This override skips the wire projection step that normally happens
        in each outer iteration, keeping only edge length constraints.
        """
        keypoints = keypoints.copy().astype(np.float64)
        K = keypoints.shape[0]
        
        # Use pre-computed anchor set, or fall back to old behavior
        if self.anchor_set is not None:
            anchor_set = self.anchor_set
        else:
            # Fallback: all branch + leaf nodes are anchors
            anchor_set = set(range(self.reference_n_branch + self.reference_n_leaf))
        
        # For NoSnap BDLO: branch/leaf nodes are projected but then frozen for edge correction
        n_anchors = self.reference_n_branch + self.reference_n_leaf
        is_nosnap_bdlo = (not self.enable_node_matching) and (self.reference_n_branch > 0)
        if is_nosnap_bdlo:
            edge_anchor_set = set(range(n_anchors))
        else:
            edge_anchor_set = anchor_set
        
        # Build edge index lookup for reference lengths
        edge_to_length = {}
        for edge_idx, (i, j) in enumerate(self.reference_edges):
            edge_to_length[(i, j)] = self.reference_lengths[edge_idx]
            edge_to_length[(j, i)] = self.reference_lengths[edge_idx]
        
        # NO PROJECTION - only edge length constraints
        for outer_iter in range(self.n_outer_iterations):
            # Skip wire projection step entirely
            
            # Edge length constraints only
            for edge_iter in range(self.n_edge_iterations):
                if self.segment_edges is not None:
                    for segment in self.segment_edges:
                        for (i, j) in segment:
                            self._apply_edge_correction(
                                keypoints, i, j, edge_to_length, edge_anchor_set
                            )
                else:
                    # Fallback: batch processing
                    corrections = np.zeros_like(keypoints)
                    correction_counts = np.zeros(K)
                    
                    for edge_idx, (i, j) in enumerate(self.reference_edges):
                        if i >= K or j >= K:
                            continue
                        
                        edge_vec = keypoints[j] - keypoints[i]
                        current_length = np.linalg.norm(edge_vec)
                        
                        if current_length < 1e-6:
                            continue
                        
                        target_length = self.reference_lengths[edge_idx]
                        length_ratio = current_length / target_length
                        
                        if 1.0 - self.edge_tolerance <= length_ratio <= 1.0 + self.edge_tolerance:
                            continue
                        
                        length_diff = target_length - current_length
                        correction_mag = length_diff * self.edge_weight * 0.5
                        correction_dir = edge_vec / current_length
                        
                        if i not in edge_anchor_set:
                            corrections[i] -= correction_dir * correction_mag
                            correction_counts[i] += 1
                        if j not in edge_anchor_set:
                            corrections[j] += correction_dir * correction_mag
                            correction_counts[j] += 1
                    
                    for k in range(K):
                        if correction_counts[k] > 0 and k not in edge_anchor_set:
                            keypoints[k] += corrections[k] / correction_counts[k]
        
        return keypoints


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_method_panel(rgb, skeleton_mask, keypoints_2d, edges, method_name, metrics, frame_idx,
                        traj_history_2d=None, tail_length=60):
    """Create visualization panel for a single method."""
    H, W = rgb.shape[:2]
    vis = rgb.copy()

    SKELETON_COLOR = [0, 191, 255]
    EDGE_COLOR = [50, 205, 50]
    BRANCH_COLOR = [128, 0, 128]  # Purple
    LEAF_COLOR = [255, 255, 0]
    INTER_COLOR = [255, 165, 0]
    TAIL_COLOR = [100, 255, 100]

    # Overlay skeleton with dilation for visibility
    if skeleton_mask is not None and np.any(skeleton_mask > 0):
        skeleton_thick = cv2.dilate(skeleton_mask, np.ones((5, 5), np.uint8), iterations=1)
        vis[skeleton_thick > 0] = SKELETON_COLOR

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

    if keypoints_2d is not None and len(keypoints_2d) > 0:
        kp_int = keypoints_2d.astype(int)
        if edges is not None:
            for (i, j) in edges:
                if i < len(kp_int) and j < len(kp_int):
                    p1 = (kp_int[i, 1], kp_int[i, 0])
                    p2 = (kp_int[j, 1], kp_int[j, 0])
                    cv2.line(vis, p1, p2, EDGE_COLOR, 2)
        # BDLO: [branch_0, branch_1, leaf_0..leaf_3, intermediate...]
        for idx, (row, col) in enumerate(kp_int):
            if 0 <= row < H and 0 <= col < W:
                if idx < 2:  # Branch nodes
                    color = BRANCH_COLOR
                elif idx < 6:  # Leaf nodes
                    color = LEAF_COLOR
                else:  # Intermediate
                    color = INTER_COLOR
                cv2.circle(vis, (col, row), 5, color, -1)
                # Draw keypoint index
                cv2.putText(vis, str(idx), (col + 6, row - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

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
                 output_dir, n_keypoints, fps, tail_length=60, keypoints_per_segment=None):
    """Process a single clip with Full and NoProj methods."""
    
    clip_output_dir = output_dir / f'clip_{clip_idx}'
    clip_output_dir.mkdir(parents=True, exist_ok=True)
    
    n_frames = end_frame - start_frame
    clip_ee_poses = ee_poses_3d[start_frame:end_frame]
    
    # Construct intrinsics matrix explicitly (same as bdlo1_batch_experiment.py)
    K = transforms['K']
    intrinsics = np.array([
        [K[0, 0], 0, K[0, 2]],
        [0, K[1, 1], K[1, 2]],
        [0, 0, 1]
    ])
    
    # Base parameters (CPD disabled in both methods)
    base_params = {
        'intrinsics': intrinsics,
        'n_keypoints': n_keypoints,
        'target_branch_nodes': TARGET_BRANCH_NODES,
        'target_leaf_nodes': TARGET_LEAF_NODES,
        'enable_cpd': False,  # CPD disabled
        'enable_node_matching': True,
        'enable_geometry_constraint': True,
        'enable_ee_injection': True,
        'ee_poses_3d': clip_ee_poses,
        'bg_threshold': 80.0,
        'max_depth': 2000.0,
        'top_k_components': 1,
        'arm_dilation_pixels': 5,
        'n_outer_iterations': 20,
        'n_edge_iterations': 15,
        'edge_weight': 0.5,
        'edge_tolerance': 0.02,
        'repulsion_iterations': 200,
        'repulsion_lr': 10.0,
        'repulsion_k_neighbors': 3,
        'keypoints_per_segment': keypoints_per_segment,
    }
    
    method_names = ['Full', 'NoProj']
    
    # Create trackers
    trackers = {
        'Full': WireTracker(**base_params),
        'NoProj': BDLOTrackerNoProj(**base_params),
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
        rgb = cv2.cvtColor(data['color'][global_idx], cv2.COLOR_BGR2RGB)  # Source is BGR
        depth = data['depth'][global_idx].astype(np.float32)
        dlo_mask = data['dlo_masks'][global_idx]
        exclude_mask = (1 - dlo_mask).astype(np.uint8)
        
        panel_images = []
        
        for method in method_names:
            tracker = trackers[method]
            result = tracker.process_frame(
                depth=depth, arm_depth=None, rgb=rgb,
                precomputed_arm_mask=exclude_mask,
            )
            
            if result['success']:
                keypoints = result['keypoints']
                keypoints_2d = result['keypoints_2d']
                edges = result['edges']
                traj_histories[method].append(keypoints_2d.copy())
                keypoints_3d_histories[method].append(keypoints.copy())
                if stored_edges is None and edges is not None:
                    stored_edges = list(edges)
                if stored_reference_lengths is None and tracker.reference_lengths is not None:
                    stored_reference_lengths = tracker.reference_lengths.copy()
            else:
                keypoints = np.empty((0, 3))
                keypoints_2d = np.empty((0, 2))
                edges = []
                if len(traj_histories[method]) > 0:
                    traj_histories[method].append(np.full_like(traj_histories[method][-1], np.nan))
                else:
                    traj_histories[method].append(np.full((n_keypoints, 2), np.nan))
                keypoints_3d_histories[method].append(np.full((n_keypoints, 3), np.nan))
            
            # Compute metrics
            if result['success'] and tracker.reference_lengths is not None:
                skeleton_pc = result.get('skeleton_pc')
                ee_pos = clip_ee_poses[local_idx]
                if skeleton_pc is not None and len(skeleton_pc) > 0:
                    if ee_pos is not None and len(ee_pos) > 0:
                        ref_pc = np.vstack([skeleton_pc, np.array(ee_pos, dtype=np.float32).reshape(-1, 3)])
                    else:
                        ref_pc = skeleton_pc
                else:
                    ref_pc = np.array(ee_pos, dtype=np.float32).reshape(-1, 3) if ee_pos is not None else np.empty((0, 3))
                
                ref_lengths = stored_reference_lengths if stored_reference_lengths is not None else tracker.reference_lengths
                edge_m = compute_edge_metrics(keypoints, edges, ref_lengths)
                pos_m = compute_position_metrics(keypoints, ref_pc)
                
                n_ref_points = len(ref_pc) if ref_pc is not None and len(ref_pc) > 0 else 100
                pred_cloud = sample_points_on_edges(keypoints, edges, n_ref_points)
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
            
            # Use skeleton_mask from tracker (skeletonized mask used for tracking) - matches reference
            skeleton_mask = result.get('skeleton_mask', np.zeros_like(depth, dtype=np.uint8))
            
            traj_hist = np.array(traj_histories[method]) if len(traj_histories[method]) > 0 else None
            panel = create_method_panel(
                rgb=rgb, skeleton_mask=skeleton_mask, keypoints_2d=keypoints_2d,
                edges=edges, method_name=method, metrics=metrics, frame_idx=local_idx,
                traj_history_2d=traj_hist, tail_length=tail_length,
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
        f.write("Methods: Full (with projection), NoProj (no projection). CPD disabled in both.\n\n")
        
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
    
    # Save 3D keypoints
    keypoints_3d_path = clip_output_dir / '3d_keypoints.npz'
    np.savez(
        keypoints_3d_path,
        full=np.array(keypoints_3d_histories['Full']),
        noproj=np.array(keypoints_3d_histories['NoProj']),
        edge_connection=np.array(stored_edges) if stored_edges else np.array([]),
        reference_lengths=np.array(stored_reference_lengths) if stored_reference_lengths is not None else np.array([]),
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
    
    # Combine metrics
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
    
    # Save combined 3d_keypoints
    np.savez(
        chunk_summary_dir / 'all_clips_3d_keypoints.npz',
        **{m.lower(): np.concatenate(all_clips_3d_kpts[m], axis=0) if all_clips_3d_kpts[m] else np.array([])
           for m in method_names},
        reference_lengths_per_clip=np.array([len(r['all_metrics']['Full']) for r in all_clip_results]),
    )
    
    # Save combined CSV
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
    
    # Compute aggregate summary
    summary_path = chunk_summary_dir / 'chunk_aggregate_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("BDLO Tracking Projection Ablation - Chunk Aggregate Summary\n")
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
            f.write(f"  Edge % Mean: {np.mean(edge_pct):.2f}% ±{np.std(edge_pct):.2f}%\n")
            f.write(f"  Edge RMSE: {np.mean(edge_rmse):.2f} ±{np.std(edge_rmse):.2f} mm\n")
            f.write(f"  Pos RMSE: {np.mean(pos_rmse):.2f} ±{np.std(pos_rmse):.2f} mm\n")
            f.write(f"  CD: {np.mean(cd):.2f} ±{np.std(cd):.2f} mm\n")
            f.write(f"  F@10mm: {np.mean(f10):.1f}%\n\n")
    
    print(f"  Chunk summary saved to: {chunk_summary_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='BDLO Tracking Projection Ablation')
    parser.add_argument('--chunk', type=int, required=True, help='Chunk index')
    parser.add_argument('--clip_seconds', type=int, default=20, help='Clip duration (default: 20)')
    parser.add_argument('--fps', type=int, default=FPS, help='Frame rate (default: 30)')
    parser.add_argument('--n_keypoints', type=int, default=N_KEYPOINTS, help='Number of keypoints')
    parser.add_argument('--keypoints_per_segment', type=int, nargs=5, default=None,
                        help='Intermediate keypoints per segment: [ee0, ee1, free0, free1, trunk] (default: auto)')
    args = parser.parse_args()
    
    # Compute n_keypoints from keypoints_per_segment if provided
    n_keypoints = args.n_keypoints
    if args.keypoints_per_segment is not None:
        # BDLO: 2 branch + 4 leaf + sum(intermediate)
        computed_n = 2 + 4 + sum(args.keypoints_per_segment)
        if n_keypoints != computed_n:
            print(f"Auto-computed n_keypoints={computed_n} from keypoints_per_segment={args.keypoints_per_segment}")
            n_keypoints = computed_n
    
    chunk_dir = DATA_BASE / f'chunk_{args.chunk}'
    output_dir = OUTPUT_BASE / f'chunk_{args.chunk}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"BDLO TRACKING PROJECTION ABLATION - Chunk {args.chunk}")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading chunk_{args.chunk} data...")
    data = load_chunk_data(chunk_dir)
    transforms = load_transforms(CALIB_DIR)
    
    print(f"  Color: {data['color'].shape}")
    print(f"  Depth: {data['depth'].shape}")
    print(f"  DLO masks: {data['dlo_masks'].shape if data['dlo_masks'] is not None else 'None'}")
    print(f"  Total frames: {data['n_frames']}")
    
    if data['dlo_masks'] is None:
        print("ERROR: No DLO masks found!")
        return
    
    # Compute EE positions
    print("\nComputing EE positions...")
    ee_poses_3d = np.zeros((data['n_frames'], 2, 3))
    for i in range(data['n_frames']):
        ee_poses_3d[i] = get_ee_positions_cam(
            data['left_poses'][i], data['right_poses'][i],
            transforms['T_left_base2cam'], transforms['T_right_base2cam'],
        )
    
    # Calculate clips
    frames_per_clip = args.clip_seconds * args.fps
    n_clips = (data['n_frames'] + frames_per_clip - 1) // frames_per_clip
    
    print(f"\nClip configuration:")
    print(f"  Clip duration: {args.clip_seconds}s ({frames_per_clip} frames)")
    print(f"  Number of clips: {n_clips}")
    
    # Process clips
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
            n_keypoints=n_keypoints,
            fps=args.fps,
            keypoints_per_segment=args.keypoints_per_segment,
        )
        all_clip_results.append(clip_result)
    
    # Aggregate chunk summary
    print("\nAggregating chunk summary...")
    aggregate_chunk_summary(output_dir, all_clip_results, method_names)
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
