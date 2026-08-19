"""
Test script for DLO tracking on dlo1_first400 data.

Runs WireTracker on first 300 frames of chunk_0 with visualization.

Usage:
    cd deformable_seg
    python test_dlo1_tracking.py

Author: Auto-generated
Date: 2026-02-27
"""

import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wire_tracker import WireTracker
from wire_tracking_cdcpd import CDCPDTracker


# ============================================================================
# DATA LOADING
# ============================================================================

def load_chunk_data(chunk_dir: Path) -> dict:
    """Load all data from a chunk directory."""
    # Load RGBD
    rgbd = np.load(chunk_dir / 'rgbd.npz')
    color = rgbd['color']  # (N, H, W, 3)
    depth = rgbd['depth']  # (N, H, W) uint16, mm
    
    # Load DLO masks (from SAM2)
    masks_path = chunk_dir / 'masks' / 'masks.npz'
    if masks_path.exists():
        dlo_masks = np.load(masks_path)['masks']  # (N, H, W) binary, 1=DLO
    else:
        dlo_masks = None
    
    # Load EE poses
    left_poses_npz = np.load(chunk_dir / 'left_arm_poses.npz')
    right_poses_npz = np.load(chunk_dir / 'right_arm_poses.npz')
    
    n_frames = len(left_poses_npz.files)
    left_poses = np.array([left_poses_npz[f'arr_{i}'] for i in range(n_frames)])  # (N, 7)
    right_poses = np.array([right_poses_npz[f'arr_{i}'] for i in range(n_frames)])  # (N, 7)
    
    return {
        'color': color,
        'depth': depth,
        'dlo_masks': dlo_masks,  # DLO mask from SAM2
        'left_poses': left_poses,   # (N, 7) [x,y,z,qw,qx,qy,qz] in left base frame
        'right_poses': right_poses, # (N, 7) [x,y,z,qw,qx,qy,qz] in right base frame
        'n_frames': n_frames,
    }


def load_transforms(calib_dir: Path) -> dict:
    """Load camera-robot transforms."""
    tf = np.load(calib_dir / 'transform_ee_cam_world.npz')
    return {
        'T_left_base2cam': tf['T_left_base2cam'],   # (4, 4)
        'T_right_base2cam': tf['T_right_base2cam'], # (4, 4)
        'K': tf['K'],  # (3, 3)
    }


def pose7_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert [x,y,z,qw,qx,qy,qz] to 4x4 matrix."""
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    # scipy uses [qx, qy, qz, qw] but our data is [qw, qx, qy, qz]
    quat = pose[3:]  # [qw, qx, qy, qz]
    T[:3, :3] = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    return T


def get_ee_positions_cam(left_pose: np.ndarray, right_pose: np.ndarray,
                          T_left_base2cam: np.ndarray, T_right_base2cam: np.ndarray) -> np.ndarray:
    """
    Convert EE poses from robot base frames to camera frame.
    
    Args:
        left_pose: (7,) [x,y,z,qw,qx,qy,qz] in left base frame
        right_pose: (7,) [x,y,z,qw,qx,qy,qz] in right base frame
        T_left_base2cam: (4,4) left base to camera
        T_right_base2cam: (4,4) right base to camera
    
    Returns:
        ee_3d: (2, 3) = [[left_x, left_y, left_z], [right_x, right_y, right_z]] in camera frame (mm)
    """
    # Left EE in base frame
    T_left_ee = pose7_to_matrix(left_pose)
    # Transform to camera frame: T_cam_ee = T_base2cam @ T_base_ee
    left_pos_cam = (T_left_base2cam @ T_left_ee)[:3, 3]
    
    # Right EE in base frame
    T_right_ee = pose7_to_matrix(right_pose)
    right_pos_cam = (T_right_base2cam @ T_right_ee)[:3, 3]
    
    # Convert from meters to mm (depth is in mm)
    return np.array([left_pos_cam * 1000, right_pos_cam * 1000])


# ============================================================================
# METRICS
# ============================================================================

def compute_edge_metrics(keypoints: np.ndarray, edges: list, reference_lengths: np.ndarray) -> dict:
    """Compute comprehensive edge metrics (same style as wire_tracker_ablation.py)."""
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


def compute_position_metrics(
    keypoints: np.ndarray,
    skeleton_pc: np.ndarray,
    extra_gt_points: np.ndarray = None,
) -> dict:
    """Compute position metrics (same style as wire_tracker_ablation.py)."""
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


class CDCPDDLOTracker:
    """CDCPD wrapper for single-DLO tracking with EE anchor constraints."""

    def __init__(self, intrinsics, n_keypoints=10, ee_poses_3d=None, **kwargs):
        self.intrinsics = intrinsics
        self.n_keypoints = n_keypoints
        self.ee_poses_3d = ee_poses_3d

        kwargs_no_ee = {k: v for k, v in kwargs.items() if k != 'ee_poses_3d'}

        cdcpd_defaults = {
            'cpd_beta': 2.0,
            'cpd_lambda': 1.0,
            'cpd_w': 0.1,
            'cpd_max_iter': 100,
            'cpd_tol': 1e-3,
            'lle_neighbors': 2,
            'lle_gamma': 0.5,
            'stretch_lambda': 1.02,
            'use_qp_optimization': True,
            'qp_max_iter': 200,
            'use_anchor_constraints': True,
            'anchor_weight': 100.0,
            'anchor_hard': True,
        }
        cdcpd_cfg = {}
        for key, default_value in cdcpd_defaults.items():
            cdcpd_cfg[key] = kwargs_no_ee.pop(key, default_value)

        self.init_tracker = WireTracker(
            intrinsics=intrinsics,
            n_keypoints=n_keypoints,
            ee_poses_3d=ee_poses_3d,
            **kwargs_no_ee,
        )

        self.cdcpd = CDCPDTracker(**cdcpd_cfg)

        self.is_initialized = False
        self.frame_count = 0
        self.prev_keypoints = None
        self.reference_edges = None
        self.reference_lengths = None
        self.ee_leaf_indices = None

    def process_frame(
        self,
        depth: np.ndarray,
        arm_depth: np.ndarray = None,
        rgb: np.ndarray = None,
        precomputed_arm_mask: np.ndarray = None,
    ) -> dict:
        frame_idx = self.frame_count
        self.frame_count += 1

        if not self.is_initialized:
            result = self.init_tracker.process_frame(
                depth=depth,
                arm_depth=arm_depth,
                rgb=rgb,
                precomputed_arm_mask=precomputed_arm_mask,
            )

            if result.get('success', False) and self.init_tracker.is_initialized:
                self.is_initialized = True
                self.prev_keypoints = result['keypoints'].copy()
                self.reference_edges = list(self.init_tracker.reference_edges or [])
                self.reference_lengths = None if self.init_tracker.reference_lengths is None else self.init_tracker.reference_lengths.copy()

                if self.init_tracker.ee_to_leaf_mapping is not None:
                    self.ee_leaf_indices = [
                        self.init_tracker.ee_to_leaf_mapping[0],
                        self.init_tracker.ee_to_leaf_mapping[1],
                    ]

            return result

        result = self.init_tracker.process_frame(
            depth=depth,
            arm_depth=arm_depth,
            rgb=rgb,
            precomputed_arm_mask=precomputed_arm_mask,
        )

        if not result.get('success', False):
            return result

        skeleton_pc = result.get('skeleton_pc')
        if skeleton_pc is None or len(skeleton_pc) == 0:
            return {'success': False, 'reason': 'empty_skeleton'}

        anchor_indices = []
        anchor_positions = []
        if self.ee_poses_3d is not None and self.ee_leaf_indices is not None and frame_idx < len(self.ee_poses_3d):
            ee_pos = self.ee_poses_3d[frame_idx]
            for i, leaf_idx in enumerate(self.ee_leaf_indices):
                anchor_indices.append(leaf_idx)
                anchor_positions.append(ee_pos[i])

        anchor_indices = np.array(anchor_indices, dtype=int) if anchor_indices else None
        anchor_positions = np.array(anchor_positions, dtype=np.float64) if anchor_positions else None

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

        return {
            'success': True,
            'keypoints': keypoints,
            'keypoints_2d': self._project_to_2d(keypoints),
            'edges': self.reference_edges,
            'skeleton_pc': skeleton_pc,
            'skeleton_mask': result.get('skeleton_mask'),
            'clean_path_mask': result.get('clean_path_mask'),
            'mode': 'track_cdcpd',
            'timing': cdcpd_result.get('timing', {}),
        }

    def _project_to_2d(self, keypoints_3d: np.ndarray) -> np.ndarray:
        if keypoints_3d is None or len(keypoints_3d) == 0:
            return np.empty((0, 2), dtype=np.float32)

        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]

        keypoints_2d = []
        for x, y, z in keypoints_3d:
            if z > 0:
                col = fx * x / z + cx
                row = fy * y / z + cy
            else:
                row, col = 0.0, 0.0
            keypoints_2d.append([row, col])
        return np.array(keypoints_2d, dtype=np.float32)


def create_method_panel(
    rgb: np.ndarray,
    skeleton_mask: np.ndarray,
    keypoints_2d: np.ndarray,
    edges: list,
    method_name: str,
    metrics: dict,
    frame_idx: int,
    traj_history_2d: np.ndarray = None,
    tail_length: int = 60,
) -> np.ndarray:
    """Compact single-method panel for ablation video."""
    H, W = rgb.shape[:2]
    vis = rgb.copy()

    SKELETON_COLOR = [0, 191, 255]
    EDGE_COLOR = [50, 205, 50]
    LEAF_COLOR = [255, 255, 0]
    INTER_COLOR = [255, 165, 0]
    TAIL_COLOR = [100, 255, 100]

    skeleton_thick = cv2.dilate(skeleton_mask, np.ones((3, 3), np.uint8), iterations=1)
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
        for idx, (row, col) in enumerate(kp_int):
            if 0 <= row < H and 0 <= col < W:
                color = LEAF_COLOR if (idx == 0 or idx == len(kp_int)-1) else INTER_COLOR
                cv2.circle(vis, (col, row), 5, color, -1)

    cv2.putText(vis, method_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, f"Frame: {frame_idx}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"Edge: {metrics['edge_pct_mean']:.2f}% (max {metrics['edge_pct_max']:.2f}%)", (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(vis, f"Edge RMSE: {metrics['edge_rmse_mm']:.2f} mm", (10, 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(vis, f"Pos RMSE: {metrics['pos_rmse_mm']:.2f} mm", (10, 126),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


def create_ablation_grid(panels: list, frame_idx: int, shape_hw: tuple, method_names: list = None) -> np.ndarray:
    """Create grid visualization for ablation methods."""
    H, W = shape_hw

    if len(panels) == 4:
        row1 = np.concatenate([panels[0], panels[1]], axis=1)
        row2 = np.concatenate([panels[2], panels[3]], axis=1)
        return np.concatenate([row1, row2], axis=0)

    if len(panels) > 4:
        panels = panels[:4]
        row1 = np.concatenate([panels[0], panels[1]], axis=1)
        row2 = np.concatenate([panels[2], panels[3]], axis=1)
        return np.concatenate([row1, row2], axis=0)

    info = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.putText(info, "DLO Ablation", (W//4, H//3), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(info, f"Frame: {frame_idx}", (W//4, H//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    methods_text = " / ".join(method_names) if method_names is not None else "Full / NoSnap / NoGeometry"
    cv2.putText(info, methods_text, (W//12, 2*H//3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 2)

    while len(panels) < 3:
        panels.append(np.zeros((H, W, 3), dtype=np.uint8))

    row1 = np.concatenate([panels[0], panels[1]], axis=1)
    row2 = np.concatenate([panels[2], info], axis=1)
    return np.concatenate([row1, row2], axis=0)


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualization(
    rgb: np.ndarray,
    foreground_mask: np.ndarray,
    skeleton_mask: np.ndarray,
    keypoints_2d: np.ndarray,
    edges: list,
    n_branch: int = 0,
    n_leaf: int = 2,
    frame_idx: int = 0,
    mode: str = 'track',
    traj_history_2d: np.ndarray = None,
    tail_length: int = 60,
) -> np.ndarray:
    """Create 2x2 visualization grid."""
    H, W = rgb.shape[:2]
    
    # Colors (BGR for OpenCV)
    SKELETON_COLOR = (255, 191, 0)   # Deep sky blue
    EDGE_COLOR = (50, 205, 50)       # Lime green
    LEAF_COLOR = (0, 255, 255)       # Yellow
    INTER_COLOR = (0, 165, 255)      # Orange
    TAIL_COLOR = (100, 255, 100)     # Light green

    def draw_trajectory_tail(canvas, traj_history, tail_len):
        if traj_history is None or len(traj_history) < 2:
            return

        T_hist, K, _ = traj_history.shape
        actual_tail = min(tail_len, T_hist)

        for idx in range(K):
            traj = traj_history[-actual_tail:, idx, :]
            for t in range(len(traj) - 1):
                pt1 = traj[t]
                pt2 = traj[t + 1]

                if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
                    continue

                row1, col1 = int(pt1[0]), int(pt1[1])
                row2, col2 = int(pt2[0]), int(pt2[1])

                if not (0 <= row1 < H and 0 <= col1 < W):
                    continue
                if not (0 <= row2 < H and 0 <= col2 < W):
                    continue

                alpha = (t + 1) / len(traj)
                color = tuple(int(c * alpha) for c in TAIL_COLOR)
                cv2.line(canvas, (col1, row1), (col2, row2), color, 2)
    
    # Panel 1: RGB
    panel1 = rgb.copy()
    cv2.putText(panel1, f"Frame {frame_idx} - {mode}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Panel 2: Binary mask overlay on RGB
    panel2 = rgb.copy()
    mask_overlay = np.zeros((H, W, 3), dtype=np.uint8)
    mask_overlay[foreground_mask > 0] = (0, 255, 0)  # Green for DLO
    panel2 = cv2.addWeighted(panel2, 0.7, mask_overlay, 0.3, 0)
    cv2.putText(panel2, "RGB + Mask", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Panel 3: Skeleton with keypoints
    panel3 = np.zeros((H, W, 3), dtype=np.uint8)
    skeleton_thick = cv2.dilate(skeleton_mask, np.ones((3, 3), np.uint8), iterations=1)
    panel3[skeleton_thick > 0] = SKELETON_COLOR
    
    # Draw keypoints on skeleton panel too
    if keypoints_2d is not None and len(keypoints_2d) > 0:
        kp_int = keypoints_2d.astype(int)
        n_kp = len(kp_int)
        for idx, (row, col) in enumerate(kp_int):
            if 0 <= row < H and 0 <= col < W:
                if idx == 0 or idx == n_kp - 1:
                    color = LEAF_COLOR
                else:
                    color = INTER_COLOR
                cv2.circle(panel3, (col, row), 5, color, -1)
    
    cv2.putText(panel3, "Clean Path + Keypoints", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Panel 4: Keypoints overlay on RGB
    panel4 = rgb.copy()

    draw_trajectory_tail(panel3, traj_history_2d, tail_length)
    draw_trajectory_tail(panel4, traj_history_2d, tail_length)
    
    if keypoints_2d is not None and len(keypoints_2d) > 0:
        kp_int = keypoints_2d.astype(int)
        n_kp = len(kp_int)
        
        # Draw edges FIRST (so keypoints are on top)
        if edges is not None:
            for edge_idx, (i, j) in enumerate(edges):
                if i < n_kp and j < n_kp:
                    row_i, col_i = kp_int[i]
                    row_j, col_j = kp_int[j]
                    if (0 <= row_i < H and 0 <= col_i < W and 
                        0 <= row_j < H and 0 <= col_j < W):
                        pt1 = (col_i, row_i)  # (x, y) for cv2
                        pt2 = (col_j, row_j)
                        cv2.line(panel4, pt1, pt2, EDGE_COLOR, 3)
        
        # Draw keypoints with labels
        for idx, (row, col) in enumerate(kp_int):
            if 0 <= row < H and 0 <= col < W:
                if idx == 0:
                    color = LEAF_COLOR
                    label = f"{idx}:L"  # Left EE
                elif idx == n_kp - 1:
                    color = LEAF_COLOR
                    label = f"{idx}:R"  # Right EE
                else:
                    color = INTER_COLOR
                    label = str(idx)
                
                cv2.circle(panel4, (col, row), 8, color, -1)
                cv2.circle(panel4, (col, row), 8, (255, 255, 255), 2)
                cv2.putText(panel4, label, (col + 10, row + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.putText(panel4, f"Chain: 0(L)->...->9(R)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(panel4, f"Edges: {len(edges) if edges else 0}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Combine into 2x2 grid
    row1 = np.concatenate([panel1, panel2], axis=1)
    row2 = np.concatenate([panel3, panel4], axis=1)
    grid = np.concatenate([row1, row2], axis=0)
    
    return grid


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Paths
    data_dir = Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects/captured_data_double_arm/dlo1_first400')
    calib_dir = Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects/captured_calibration_data/dlo1_cloth1_calibration')
    output_dir = Path('./test_output')
    output_dir.mkdir(exist_ok=True)
    
    chunk_dir = data_dir / 'chunk_0'
    
    # Load data
    print("Loading data...")
    data = load_chunk_data(chunk_dir)
    transforms = load_transforms(calib_dir)
    
    print(f"  Color: {data['color'].shape}")
    print(f"  Depth: {data['depth'].shape}")
    print(f"  DLO masks: {data['dlo_masks'].shape if data['dlo_masks'] is not None else 'None'}")
    print(f"  Frames: {data['n_frames']}")
    
    # Test parameters
    n_test_frames = min(600, data['n_frames'])
    n_keypoints = 10
    tail_length = 60
    
    # Camera intrinsics from calibration
    K = transforms['K']
    intrinsics = np.array([
        [K[0, 0], 0, K[0, 2]],
        [0, K[1, 1], K[1, 2]],
        [0, 0, 1]
    ])
    
    print(f"\nCamera intrinsics:\n{intrinsics}")
    
    # Precompute EE positions in camera frame for all frames
    print("\nConverting EE poses to camera frame...")
    ee_poses_3d = np.zeros((data['n_frames'], 2, 3))
    for i in range(data['n_frames']):
        ee_poses_3d[i] = get_ee_positions_cam(
            data['left_poses'][i],
            data['right_poses'][i],
            transforms['T_left_base2cam'],
            transforms['T_right_base2cam'],
        )
    print(f"  EE poses shape: {ee_poses_3d.shape}")
    print(f"  Sample EE at frame 0 (mm): {ee_poses_3d[0]}")
    
    # Check DLO masks available
    if data['dlo_masks'] is None:
        print("ERROR: No DLO masks found!")
        return
    
    # Ablation setup (per your requirement)
    print("\nInitializing DLO ablation trackers...")
    base_params = {
        'intrinsics': intrinsics,
        'n_keypoints': n_keypoints,
        'target_branch_nodes': 0,
        'target_leaf_nodes': 2,
        'bg_threshold': 80.0,
        'max_depth': 2000.0,
        'top_k_components': 1,
        'arm_dilation_pixels': 5,
        'enable_cpd': False,
        'n_outer_iterations': 20,
        'n_edge_iterations': 15,
        'edge_weight': 0.4,
        'edge_tolerance': 0.02,
        'repulsion_iterations': 200,
        'repulsion_lr': 10.0,
        'repulsion_k_neighbors': 3,
    }

    method_configs = {
        'Full': {
            'enable_node_matching': True,
            'enable_geometry_constraint': True,
            'enable_ee_injection': True,
            'ee_poses_3d': ee_poses_3d,
        },
        'NoSnap': {
            # For single DLO, node matching is bypassed in tracker.track(),
            # so NoSnap is intentionally identical to Full.
            'enable_node_matching': True,
            'enable_geometry_constraint': True,
            'enable_ee_injection': True,
            'ee_poses_3d': ee_poses_3d,
        },
        'NoGeometry': {
            'enable_node_matching': True,
            'enable_geometry_constraint': False,
            'enable_ee_injection': True,
            'ee_poses_3d': ee_poses_3d,
        },
        'CDCPD': {
            'enable_node_matching': True,
            'enable_geometry_constraint': True,
            'enable_ee_injection': True,
            'ee_poses_3d': ee_poses_3d,
            # Match wire_tracking_cdcpd_benchmark.py
            'cpd_beta': 2.0,
            'cpd_lambda': 0.05,
            'cpd_w': 0.05,
            'cpd_max_iter': 100,
            'cpd_tol': 1e-3,
            'lle_neighbors': 2,
            'lle_gamma': 0.1,
            'stretch_lambda': 1.02,
            'use_qp_optimization': True,
            'qp_max_iter': 200,
            'use_anchor_constraints': True,
            'anchor_weight': 100.0,
            'anchor_hard': True,
        },
    }
    method_names = ['Full', 'NoSnap', 'NoGeometry', 'CDCPD']

    trackers = {}
    for method in method_names:
        if method == 'CDCPD':
            trackers[method] = CDCPDDLOTracker(**{**base_params, **method_configs[method]})
        else:
            trackers[method] = WireTracker(**{**base_params, **method_configs[method]})
        print(f"  {method}: node_matching={method_configs[method]['enable_node_matching']}, "
              f"geometry={method_configs[method]['enable_geometry_constraint']}, "
              f"ee_injection={'on' if method_configs[method]['enable_ee_injection'] else 'off'}")

    # Outputs
    video_path = output_dir / 'test_chunk0_600frames_ablation.mp4'
    video_writer = None
    tracking_video_path = output_dir / 'test_chunk0_600frames_tracking_full.mp4'
    tracking_video_writer = None
    fps = 30

    all_metrics = {m: [] for m in method_names}
    traj_histories = {m: [] for m in method_names}
    keypoints_3d_histories = {m: [] for m in method_names}  # Store 3D keypoints for all methods
    stored_edges = None  # Will store edge connections (same for all methods)

    print(f"\nProcessing {n_test_frames} frames...")

    for frame_idx in tqdm(range(n_test_frames)):
        rgb = data['color'][frame_idx]
        depth = data['depth'][frame_idx].astype(np.float32)
        dlo_mask = data['dlo_masks'][frame_idx]
        exclude_mask = (1 - dlo_mask).astype(np.uint8)

        panel_images = []
        progress_items = []
        full_result = None

        for method in method_names:
            tracker = trackers[method]
            result = tracker.process_frame(
                depth=depth,
                arm_depth=None,
                rgb=rgb,
                precomputed_arm_mask=exclude_mask,
            )

            if method == 'Full':
                full_result = result

            if result['success']:
                keypoints = result['keypoints']
                keypoints_2d = result['keypoints_2d']
                edges = result['edges']
                traj_histories[method].append(keypoints_2d.copy())
                keypoints_3d_histories[method].append(keypoints.copy())
                # Store edges once (same for all methods)
                if stored_edges is None and edges is not None:
                    stored_edges = list(edges)
            else:
                keypoints = np.empty((0, 3))
                keypoints_2d = np.empty((0, 2))
                edges = []
                if len(traj_histories[method]) > 0:
                    traj_histories[method].append(np.full_like(traj_histories[method][-1], np.nan))
                else:
                    traj_histories[method].append(np.full((n_keypoints, 2), np.nan))
                keypoints_3d_histories[method].append(np.full((n_keypoints, 3), np.nan))

            if result['success'] and tracker.reference_lengths is not None:
                edge_m = compute_edge_metrics(keypoints, edges, tracker.reference_lengths)
                pos_m = compute_position_metrics(
                    keypoints,
                    result.get('skeleton_pc', np.empty((0, 3))),
                    extra_gt_points=ee_poses_3d[frame_idx],
                )
                metrics = {
                    'frame': frame_idx,
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
                }
            else:
                metrics = {
                    'frame': frame_idx,
                    'success': False,
                    'edge_pct_mean': 0.0,
                    'edge_pct_std': 0.0,
                    'edge_pct_max': 0.0,
                    'edge_rmse_mm': 0.0,
                    'edge_under_2pct': 0.0,
                    'edge_under_5pct': 0.0,
                    'edge_under_10pct': 0.0,
                    'pos_rmse_mm': 0.0,
                    'pos_under_2mm': 0.0,
                    'pos_under_5mm': 0.0,
                    'pos_under_10mm': 0.0,
                }

            all_metrics[method].append(metrics)
            progress_items.append(f"{method}: E={metrics['edge_pct_mean']:.1f}% P={metrics['pos_rmse_mm']:.1f}")

            skeleton_mask = result.get('clean_path_mask', result.get('skeleton_mask', np.zeros_like(depth, dtype=np.uint8)))
            traj_hist = np.array(traj_histories[method]) if len(traj_histories[method]) > 0 else None
            panel = create_method_panel(
                rgb=rgb,
                skeleton_mask=skeleton_mask,
                keypoints_2d=keypoints_2d,
                edges=edges,
                method_name=method,
                metrics=metrics,
                frame_idx=frame_idx,
                traj_history_2d=traj_hist,
                tail_length=tail_length,
            )
            panel_images.append(panel)

        grid = create_ablation_grid(panel_images, frame_idx, rgb.shape[:2], method_names=method_names)

        if frame_idx == 0:
            frame0_path = output_dir / 'frame0_init_ablation.png'
            cv2.imwrite(str(frame0_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print(f"\n  Frame 0 saved to: {frame0_path}")

        if video_writer is None:
            H, W = grid.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (W, H))

        video_writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

        # Also save Full-method tracking video using existing 2x2 grid visualization
        if full_result is not None:
            full_keypoints_2d = full_result.get('keypoints_2d', np.empty((0, 2)))
            full_edges = full_result.get('edges', [])
            full_mode = full_result.get('mode', 'failed')
            full_skeleton_mask = full_result.get(
                'clean_path_mask',
                full_result.get('skeleton_mask', np.zeros_like(depth, dtype=np.uint8))
            )

            full_traj_hist = np.array(traj_histories['Full']) if len(traj_histories['Full']) > 0 else None
            full_vis = create_visualization(
                rgb=rgb,
                foreground_mask=dlo_mask,
                skeleton_mask=full_skeleton_mask,
                keypoints_2d=full_keypoints_2d,
                edges=full_edges,
                n_branch=0,
                n_leaf=2,
                frame_idx=frame_idx,
                mode=full_mode,
                traj_history_2d=full_traj_hist,
                tail_length=tail_length,
            )

            if tracking_video_writer is None:
                Ht, Wt = full_vis.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                tracking_video_writer = cv2.VideoWriter(str(tracking_video_path), fourcc, fps, (Wt, Ht))

            tracking_video_writer.write(full_vis)

        if frame_idx % 20 == 0 or frame_idx == n_test_frames - 1:
            print(f"Frame {frame_idx:4d}: " + " | ".join(progress_items))

    if video_writer is not None:
        video_writer.release()
    if tracking_video_writer is not None:
        tracking_video_writer.release()

    # Save per-frame CSV
    per_frame_csv = output_dir / 'test_ablation_per_frame.csv'
    with open(per_frame_csv, 'w') as f:
        f.write('Frame,Method,EdgePctMean,EdgePctStd,EdgePctMax,EdgeRMSE,PosRMSE,Edge<2%,Edge<5%,Edge<10%,Pos<2mm,Pos<5mm,Pos<10mm\n')
        for frame_idx in range(n_test_frames):
            for method in method_names:
                m = all_metrics[method][frame_idx]
                f.write(f"{frame_idx},{method},{m['edge_pct_mean']:.6f},{m['edge_pct_std']:.6f},{m['edge_pct_max']:.6f},"
                        f"{m['edge_rmse_mm']:.6f},{m['pos_rmse_mm']:.6f},{m['edge_under_2pct']:.4f},"
                        f"{m['edge_under_5pct']:.4f},{m['edge_under_10pct']:.4f},{m['pos_under_2mm']:.4f},"
                        f"{m['pos_under_5mm']:.4f},{m['pos_under_10mm']:.4f}\n")

    # Summary table (same style fields as wire_tracker_ablation)
    summary_rows = []
    for method in method_names:
        metrics_list = all_metrics[method][1:] if len(all_metrics[method]) > 1 else all_metrics[method]
        if len(metrics_list) == 0:
            continue

        edge_pct_means = [m['edge_pct_mean'] for m in metrics_list if m['edge_pct_mean'] > 0]
        edge_pct_maxes = [m['edge_pct_max'] for m in metrics_list if m['edge_pct_max'] > 0]
        edge_rmses = [m['edge_rmse_mm'] for m in metrics_list if m['edge_rmse_mm'] > 0]
        edge_under_2 = [m['edge_under_2pct'] for m in metrics_list]
        edge_under_5 = [m['edge_under_5pct'] for m in metrics_list]
        edge_under_10 = [m['edge_under_10pct'] for m in metrics_list]
        pos_rmses = [m['pos_rmse_mm'] for m in metrics_list if m['pos_rmse_mm'] > 0]
        pos_under_2 = [m['pos_under_2mm'] for m in metrics_list]
        pos_under_5 = [m['pos_under_5mm'] for m in metrics_list]
        pos_under_10 = [m['pos_under_10mm'] for m in metrics_list]

        summary_rows.append({
            'method': method,
            'edge_pct_mean_avg': np.mean(edge_pct_means) if edge_pct_means else 0.0,
            'edge_pct_mean_std': np.std(edge_pct_means) if edge_pct_means else 0.0,
            'edge_pct_max_avg': np.mean(edge_pct_maxes) if edge_pct_maxes else 0.0,
            'edge_pct_max_abs': np.max(edge_pct_maxes) if edge_pct_maxes else 0.0,
            'edge_rmse_avg': np.mean(edge_rmses) if edge_rmses else 0.0,
            'edge_rmse_std': np.std(edge_rmses) if edge_rmses else 0.0,
            'edge_under_2pct': np.mean(edge_under_2) if edge_under_2 else 0.0,
            'edge_under_5pct': np.mean(edge_under_5) if edge_under_5 else 0.0,
            'edge_under_10pct': np.mean(edge_under_10) if edge_under_10 else 0.0,
            'pos_rmse_avg': np.mean(pos_rmses) if pos_rmses else 0.0,
            'pos_rmse_std': np.std(pos_rmses) if pos_rmses else 0.0,
            'pos_under_2mm': np.mean(pos_under_2) if pos_under_2 else 0.0,
            'pos_under_5mm': np.mean(pos_under_5) if pos_under_5 else 0.0,
            'pos_under_10mm': np.mean(pos_under_10) if pos_under_10 else 0.0,
        })

    lines = []
    lines.append("=" * 90)
    lines.append("DLO ABLATION SUMMARY")
    lines.append("=" * 90)
    lines.append(f"Processed frames: {n_test_frames}")
    lines.append("")
    lines.append("EDGE LENGTH METRICS")
    lines.append("-" * 90)
    lines.append(f"{'Method':<12} | {'Edge % Mean':<14} | {'Edge RMSE':<14} | {'Max%(Avg)':<10} | {'Max%(Abs)':<10} | {'<2%':<6} | {'<5%':<6} | {'<10%':<6}")
    lines.append("-" * 110)
    for s in summary_rows:
        lines.append(f"{s['method']:<12} | "
                     f"{s['edge_pct_mean_avg']:>5.2f}% ±{s['edge_pct_mean_std']:>4.2f}% | "
                     f"{s['edge_rmse_avg']:>5.2f} ±{s['edge_rmse_std']:>4.2f}mm | "
                     f"{s['edge_pct_max_avg']:>7.2f}% | "
                     f"{s['edge_pct_max_abs']:>7.2f}% | "
                     f"{s['edge_under_2pct']:>5.1f}% | "
                     f"{s['edge_under_5pct']:>5.1f}% | "
                     f"{s['edge_under_10pct']:>5.1f}%")

    lines.append("")
    lines.append("POSITION METRICS")
    lines.append("-" * 90)
    lines.append(f"{'Method':<12} | {'Pos RMSE (mm)':<18} | {'<2mm':<8} | {'<5mm':<8} | {'<10mm':<8}")
    lines.append("-" * 90)
    for s in summary_rows:
        lines.append(f"{s['method']:<12} | "
                     f"{s['pos_rmse_avg']:>6.2f} ±{s['pos_rmse_std']:>6.2f} mm | "
                     f"{s['pos_under_2mm']:>6.1f}% | "
                     f"{s['pos_under_5mm']:>6.1f}% | "
                     f"{s['pos_under_10mm']:>6.1f}%")

    summary_txt_path = output_dir / 'test_ablation_summary.txt'
    with open(summary_txt_path, 'w') as f:
        f.write("\n".join(lines) + "\n")

    # Save metrics-over-time plot (Edge RMSE + Pos RMSE for all methods)
    plot_path = output_dir / 'test_ablation_metrics_over_time.png'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = {'Full': 'blue', 'NoSnap': 'orange', 'NoGeometry': 'green', 'CDCPD': 'red'}

    for method in method_names:
        method_metrics = all_metrics[method]
        frames = list(range(len(method_metrics)))
        edge_rmse = [m['edge_rmse_mm'] for m in method_metrics]
        pos_rmse = [m['pos_rmse_mm'] for m in method_metrics]

        ax1.plot(frames, edge_rmse, label=method, color=colors.get(method, 'gray'), linewidth=1.5)
        ax2.plot(frames, pos_rmse, label=method, color=colors.get(method, 'gray'), linewidth=1.5)

    ax1.set_ylabel('Edge RMSE (mm)')
    ax1.set_title('Edge RMSE Over Time')
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
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save 3D keypoints for all methods to single npz file
    keypoints_3d_path = output_dir / '3d_keypoints.npz'
    np.savez(
        keypoints_3d_path,
        full=np.array(keypoints_3d_histories['Full']),
        nosnap=np.array(keypoints_3d_histories['NoSnap']),
        noGeometry=np.array(keypoints_3d_histories['NoGeometry']),
        cdcpd2=np.array(keypoints_3d_histories['CDCPD']),
        edge_connection=np.array(stored_edges) if stored_edges else np.array([]),
    )

    print("\n" + "\n".join(lines))
    print(f"\nVideo saved: {video_path}")
    print(f"Tracking video saved: {tracking_video_path}")
    print(f"Per-frame CSV saved: {per_frame_csv}")
    print(f"Summary TXT saved: {summary_txt_path}")
    print(f"Metrics plot saved: {plot_path}")
    print(f"3D keypoints saved: {keypoints_3d_path}")


if __name__ == "__main__":
    main()
