"""
Wire Tracking Main Script

Uses WireTracker class with data format from seg_with_arms.py.

Pipeline:
    1. Load arm trajectory + full scene trajectory data
    2. Synchronize sequences
    3. Run WireTracker on each frame
    4. Visualize and save results

Usage:
    python wire_tracking_main.py --traj 1
    python wire_tracking_main.py --traj 2
    python wire_tracking_main.py --traj 3

Author: Auto-generated
Date: 2026-02-14
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
import time
import PIL.Image
import io
import gc
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from wire_tracker import WireTracker


def create_visualization(
    rgb: np.ndarray,
    foreground_mask: np.ndarray,
    skeleton_mask: np.ndarray,
    keypoints_2d: np.ndarray,
    edges: list,
    detected_branch_2d: np.ndarray = None,
    detected_leaf_2d: np.ndarray = None,
    mode: str = 'track',
    n_branch: int = 0,
    n_leaf: int = 0,
    frame_idx: int = 0,
    traj_history_2d: np.ndarray = None,
    tail_length: int = 60,
) -> np.ndarray:
    """
    Create 2x2 visualization grid with trajectory tails.
    
    Layout:
        [Skeleton]         [Skeleton Overlay]
        [Keypoints Only]   [Keypoints Overlay]
    
    Args:
        rgb: H × W × 3 RGB image
        foreground_mask: H × W binary mask
        skeleton_mask: H × W skeleton mask
        keypoints_2d: K × 2 keypoint pixel coords (row, col)
        edges: List of (i, j) edge tuples
        detected_branch_2d: B × 2 detected branch coords (optional)
        detected_leaf_2d: L × 2 detected leaf coords (optional)
        mode: 'init', 'track', 'restart', 'skip'
        n_branch: Number of branch nodes (first n_branch keypoints)
        n_leaf: Number of leaf nodes (next n_leaf keypoints after branch)
        frame_idx: Current frame index
        traj_history_2d: T × K × 2 trajectory history for all keypoints
        tail_length: Number of frames for trajectory tail
    
    Returns:
        grid: Visualization grid
    """
    H, W = rgb.shape[:2]
    
    # Colors
    SKELETON_COLOR = [0, 191, 255]  # Deep sky blue
    EDGE_COLOR = [50, 205, 50]      # Lime green for edges
    BRANCH_COLOR = [128, 0, 128]    # Purple for branch nodes
    LEAF_COLOR = [255, 255, 0]      # Yellow for leaf nodes
    INTER_COLOR = [255, 165, 0]     # Orange for intermediate nodes
    TAIL_COLOR = [100, 255, 100]    # Light green for trajectory tail
    
    # Visualization parameters
    KEYPOINT_RADIUS = 8             # Increased from 5
    EDGE_THICKNESS = 3              # Increased from 2
    DETECTED_NODE_RADIUS = 6
    TAIL_THICKNESS = 2
    
    def draw_trajectory_tail(canvas, traj_history, tail_length):
        """Draw trajectory tails for all keypoints."""
        if traj_history is None or len(traj_history) == 0:
            return
        
        T_hist, K, _ = traj_history.shape
        actual_tail = min(tail_length, T_hist)
        
        for idx in range(K):
            # Get trajectory for this keypoint
            traj = traj_history[-actual_tail:, idx, :]  # Last `actual_tail` frames
            
            # Draw trajectory as connected line segments with fading alpha
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
                
                # Fade color based on position in tail
                alpha = (t + 1) / len(traj)
                color = [int(c * alpha) for c in TAIL_COLOR]
                cv2.line(canvas, (col1, row1), (col2, row2), color, TAIL_THICKNESS)
    
    # Row 1: Skeleton + Skeleton overlay
    skeleton_vis = np.zeros((H, W, 3), dtype=np.uint8)
    skeleton_thick = cv2.dilate(skeleton_mask, np.ones((3, 3), np.uint8), iterations=1)
    skeleton_vis[skeleton_thick > 0] = SKELETON_COLOR
    
    skeleton_overlay = rgb.copy()
    skeleton_thick_overlay = cv2.dilate(skeleton_mask, np.ones((5, 5), np.uint8), iterations=1)
    skeleton_overlay[skeleton_thick_overlay > 0] = SKELETON_COLOR
    
    # Draw detected nodes on skeleton
    if detected_branch_2d is not None:
        for coord in detected_branch_2d:
            row, col = int(coord[0]), int(coord[1])
            if 0 <= row < H and 0 <= col < W:
                cv2.circle(skeleton_vis, (col, row), DETECTED_NODE_RADIUS, BRANCH_COLOR, -1)
                cv2.circle(skeleton_overlay, (col, row), DETECTED_NODE_RADIUS, BRANCH_COLOR, -1)
    
    if detected_leaf_2d is not None:
        for coord in detected_leaf_2d:
            row, col = int(coord[0]), int(coord[1])
            if 0 <= row < H and 0 <= col < W:
                cv2.circle(skeleton_vis, (col, row), DETECTED_NODE_RADIUS, LEAF_COLOR, -1)
                cv2.circle(skeleton_overlay, (col, row), DETECTED_NODE_RADIUS, LEAF_COLOR, -1)
    
    # Add mode and frame label to Row 1 (skeleton overlay)
    cv2.putText(skeleton_overlay, f"Mode: {mode}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(skeleton_overlay, f"Frame: {frame_idx}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Row 2: Keypoints + Keypoints overlay
    keypoint_vis = np.zeros((H, W, 3), dtype=np.uint8)
    keypoint_vis[skeleton_thick > 0] = [50, 50, 50]  # Dim skeleton background
    
    keypoint_overlay = rgb.copy()
    
    # Draw trajectory tails first (so keypoints are on top)
    draw_trajectory_tail(keypoint_vis, traj_history_2d, tail_length)
    draw_trajectory_tail(keypoint_overlay, traj_history_2d, tail_length)
    
    # Draw edges (so keypoints are on top)
    if keypoints_2d is not None and len(keypoints_2d) > 0 and edges is not None:
        kp_int = keypoints_2d.astype(int)
        for (i, j) in edges:
            if i < len(kp_int) and j < len(kp_int):
                pt1 = (kp_int[i, 1], kp_int[i, 0])  # (col, row)
                pt2 = (kp_int[j, 1], kp_int[j, 0])
                cv2.line(keypoint_vis, pt1, pt2, EDGE_COLOR, EDGE_THICKNESS)
                cv2.line(keypoint_overlay, pt1, pt2, EDGE_COLOR, EDGE_THICKNESS)
        
        # Draw keypoints with different colors based on type
        for idx, (row, col) in enumerate(kp_int):
            if 0 <= row < H and 0 <= col < W:
                # Determine keypoint type and color
                if idx < n_branch:
                    # Branch node - Purple
                    color = BRANCH_COLOR
                elif idx < n_branch + n_leaf:
                    # Leaf node - Yellow
                    color = LEAF_COLOR
                else:
                    # Intermediate node - Orange
                    color = INTER_COLOR
                
                cv2.circle(keypoint_vis, (col, row), KEYPOINT_RADIUS, color, -1)
                cv2.circle(keypoint_overlay, (col, row), KEYPOINT_RADIUS, color, -1)
                
                # Add index label
                cv2.putText(keypoint_vis, str(idx), (col + 10, row - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Add mode and frame label to Row 2 (keypoint overlay)
    cv2.putText(keypoint_overlay, f"Mode: {mode}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(keypoint_overlay, f"Frame: {frame_idx}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add legend
    legend_y = 90
    cv2.circle(keypoint_overlay, (20, legend_y), 6, BRANCH_COLOR, -1)
    cv2.putText(keypoint_overlay, "Branch", (35, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(keypoint_overlay, (20, legend_y + 25), 6, LEAF_COLOR, -1)
    cv2.putText(keypoint_overlay, "Leaf", (35, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(keypoint_overlay, (20, legend_y + 50), 6, INTER_COLOR, -1)
    cv2.putText(keypoint_overlay, "Intermediate", (35, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Create grid (2x2 now, removed row 1)
    row1 = np.concatenate([skeleton_vis, skeleton_overlay], axis=1)
    row2 = np.concatenate([keypoint_vis, keypoint_overlay], axis=1)
    
    grid = np.concatenate([row1, row2], axis=0)
    
    return grid


def create_3d_visualization(
    keypoints_3d: np.ndarray,
    edges: list,
    full_pc: np.ndarray,
    pc_colors: np.ndarray,
    n_branch: int = 0,
    n_leaf: int = 0,
    frame_idx: int = 0,
    mode: str = 'track',
    pc_downsample: int = 500,
    figsize: tuple = (16, 16),
    fixed_xlim: tuple = None,
    fixed_ylim: tuple = None,
    fixed_zlim: tuple = None,
    traj_history_3d: np.ndarray = None,
    tail_length: int = 60,
) -> np.ndarray:
    """
    Create 2x2 3D visualization grid with trajectory tails.
    
    Row 1: View 1 (elev=100, azim=-90)
        Left: Keypoints + Edges only
        Right: Keypoints + Edges + Downsampled full point cloud (colored)
    
    Row 2: View 2 (elev=180, azim=-90)
        Left: Keypoints + Edges only
        Right: Keypoints + Edges + Downsampled full point cloud (colored)
    
    Args:
        keypoints_3d: K × 3 keypoint positions
        edges: List of (i, j) edge tuples
        full_pc: N × 3 full scene point cloud (unmasked)
        pc_colors: N × 3 RGB colors for point cloud (0-255)
        n_branch: Number of branch nodes
        n_leaf: Number of leaf nodes
        frame_idx: Current frame index
        mode: 'init', 'track', 'restart', 'skip'
        pc_downsample: Number of points to keep from point cloud
        figsize: Figure size
        fixed_xlim: Fixed (min, max) for X axis
        fixed_ylim: Fixed (min, max) for Y axis
        fixed_zlim: Fixed (min, max) for Z axis
        traj_history_3d: T × K × 3 trajectory history for all keypoints
        tail_length: Number of frames for trajectory tail
    
    Returns:
        img: RGB image as numpy array
    """
    # Colors (normalized to 0-1 for matplotlib)
    BRANCH_COLOR = np.array([128, 0, 128]) / 255.0    # Purple
    LEAF_COLOR = np.array([255, 255, 0]) / 255.0      # Yellow
    INTER_COLOR = np.array([255, 165, 0]) / 255.0     # Orange
    EDGE_COLOR = np.array([50, 205, 50]) / 255.0      # Lime green
    TAIL_COLOR = np.array([100, 255, 100]) / 255.0    # Light green for trajectory tail
    
    fig = plt.figure(figsize=figsize)
    
    # Downsample point cloud (keep same indices for colors)
    if len(full_pc) > pc_downsample:
        indices = np.random.choice(len(full_pc), pc_downsample, replace=False)
        pc_downsampled = full_pc[indices]
        colors_downsampled = pc_colors[indices] / 255.0  # Normalize to 0-1
    else:
        pc_downsampled = full_pc
        colors_downsampled = pc_colors / 255.0 if len(pc_colors) > 0 else np.empty((0, 3))
    
    def setup_ax(ax, title, show_pc=False, elev=100, azim=-90):
        """Setup 3D axis with keypoints, edges, and trajectory tails."""
        ax.set_title(f"{title}\nFrame: {frame_idx} | Mode: {mode}", fontsize=10)
        
        # Draw point cloud first (lower zorder so keypoints are on top)
        if show_pc and len(pc_downsampled) > 0:
            ax.scatter(pc_downsampled[:, 0], pc_downsampled[:, 1], pc_downsampled[:, 2],
                      c=colors_downsampled, s=2, alpha=0.7, zorder=1, depthshade=False)
        
        # Draw trajectory tails
        if traj_history_3d is not None and len(traj_history_3d) > 0:
            T_hist = traj_history_3d.shape[0]
            actual_tail = min(tail_length, T_hist)
            K = traj_history_3d.shape[1]
            
            for idx in range(K):
                traj = traj_history_3d[-actual_tail:, idx, :]
                valid = ~np.any(np.isnan(traj), axis=1)
                
                if np.sum(valid) > 1:
                    traj_valid = traj[valid]
                    # Draw as single line for efficiency
                    ax.plot(traj_valid[:, 0], traj_valid[:, 1], traj_valid[:, 2],
                           color=TAIL_COLOR, linewidth=1.5, alpha=0.7, zorder=5)
        
        # Draw edges (higher zorder to be on top of point cloud)
        if len(keypoints_3d) > 0 and edges is not None:
            for (i, j) in edges:
                if i < len(keypoints_3d) and j < len(keypoints_3d):
                    pts = keypoints_3d[[i, j], :]
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 
                           color=EDGE_COLOR, linewidth=3, alpha=1.0, zorder=10)
        
        # Draw keypoints with different colors (highest zorder, no outline)
        if len(keypoints_3d) > 0:
            for idx in range(len(keypoints_3d)):
                if idx < n_branch:
                    color = BRANCH_COLOR
                    size = 60
                elif idx < n_branch + n_leaf:
                    color = LEAF_COLOR
                    size = 60
                else:
                    color = INTER_COLOR
                    size = 60
                
                ax.scatter(keypoints_3d[idx, 0], keypoints_3d[idx, 1], keypoints_3d[idx, 2],
                          c=[color], s=size, zorder=20, depthshade=False)
        
        # Set fixed axis limits
        if fixed_xlim is not None:
            ax.set_xlim(fixed_xlim)
        if fixed_ylim is not None:
            ax.set_ylim(fixed_ylim)
        if fixed_zlim is not None:
            ax.set_zlim(fixed_zlim)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Invert Y axis so positive Y points downward (image convention)
        ax.invert_yaxis()

    
    # Row 1: View 1 (elev=100, azim=-90)
    ax1 = fig.add_subplot(221, projection='3d')
    setup_ax(ax1, "View 1: Keypoints + Edges", show_pc=False, elev=100, azim=-90)
    
    ax2 = fig.add_subplot(222, projection='3d')
    setup_ax(ax2, "View 1: Keypoints + Edges + Point Cloud", show_pc=True, elev=100, azim=-90)
    
    # Row 2: View 2 (elev=180, azim=-90)
    ax3 = fig.add_subplot(223, projection='3d')
    setup_ax(ax3, "View 2: Keypoints + Edges", show_pc=False, elev=24, azim=-90)
    
    ax4 = fig.add_subplot(224, projection='3d')
    setup_ax(ax4, "View 2: Keypoints + Edges + Point Cloud", show_pc=True, elev=24, azim=-90)
    
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    
    # Read image from buffer
    img = np.array(PIL.Image.open(buf))[:, :, :3]  # Remove alpha channel if present
    
    plt.close(fig)
    
    return img


def main():
    """Main tracking loop."""
    
    # ================================================================
    # Parse Arguments
    # ================================================================
    
    parser = argparse.ArgumentParser(description='Wire Tracking with CPD + Geometry Constraints')
    parser.add_argument('--traj', type=int, required=True, choices=[1, 2, 3],
                        help='Trajectory number to run (1, 2, or 3)')
    args = parser.parse_args()
    
    # ================================================================
    # Configuration
    # ================================================================
    
    # Camera intrinsics
    intrinsics = np.array([
        [606.1124267578125, 0, 641.7578125],
        [0, 605.8821411132812, 365.6518859863281],
        [0, 0, 1]
    ])
    
    # Set data paths and synchronization parameters based on trajectory
    if args.traj == 1:
        # Data paths traj1
        arm_data_path = Path("./data/arm_traj1/arm_traj1.npy")
        full_data_path = Path("./data/arm_traj1/arm_with_wires_traj1.npy")
        output_dir = Path("./data/arm_traj1/wire_tracking_output")
        precomputed_mask_dir = None  # No precomputed masks for traj1
        ee_pose_path = Path("./data/arm_traj1/ee_pose_output/ee_poses_3d.npy")
        # Synchronization parameters for traj1 (from seg_with_arms.py)
        arm_green_frame = 66
        full_green_frame = 66
    elif args.traj == 2:
        # Data paths traj2
        arm_data_path = Path("./data/arm_traj2/arm_traj2.npy")
        full_data_path = Path("./data/arm_traj2/arm_with_wires_traj2.npy")
        output_dir = Path("./data/arm_traj2/wire_tracking_output")
        precomputed_mask_dir = Path("./data/arm_traj2/masks")  # Precomputed arm masks for traj2
        ee_pose_path = Path("./data/arm_traj2/ee_pose_output/ee_poses_3d.npy")
        # Synchronization parameters for traj2
        # Masks are named 0000-0224 corresponding to original frames 0-224
        arm_green_frame = 0
        full_green_frame = 0
    elif args.traj == 3:
        # Data paths traj3
        arm_data_path = Path("./data/arm_traj3/arm_traj3_contact.npy")
        full_data_path = Path("./data/arm_traj3/arm_with_wires_traj3_contact.npy")
        output_dir = Path("./data/arm_traj3/wire_tracking_output")
        precomputed_mask_dir = None  # No precomputed masks for traj3
        ee_pose_path = Path("./data/arm_traj3/ee_pose_output/ee_poses_3d.npy")
        # Synchronization parameters for traj3 (contact)
        arm_green_frame = 84
        full_green_frame = 100
    
    # Load EE poses if available
    ee_poses_3d = None
    if ee_pose_path.exists():
        ee_data = np.load(str(ee_pose_path), allow_pickle=True).item()
        ee_poses_3d = ee_data['ee_3d']  # Shape: (n_frames, 2, 3)
        print(f"Loaded EE poses from: {ee_pose_path}")
        print(f"  Shape: {ee_poses_3d.shape}")
    else:
        print(f"No EE poses found at: {ee_pose_path}")

    # Tracker parameters
    tracker_params = {
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
        'cpd_beta': 0.1,
        'cpd_lambda': 0.1,
        'cpd_w': 0.05,
        'cpd_max_iter': 100,
        # Geometry constraints
        'n_outer_iterations': 10,
        'n_edge_iterations': 30,
        'edge_weight': 0.5,
        'edge_tolerance': 0.02,
        # Repulsion
        'repulsion_iterations': 200,
        'repulsion_lr': 10.0,
        'repulsion_k_neighbors': 3,
        # End-effector poses
        'ee_poses_3d': ee_poses_3d,
    }
    
    # Video parameters
    fps = 30
    
    # ================================================================
    # Load Data
    # ================================================================
    
    print("=" * 60)
    print("WIRE TRACKING WITH CPD + GEOMETRY CONSTRAINTS")
    print(f"Trajectory: {args.traj}")
    print("=" * 60)
    
    print(f"\nLoading arm-only data from: {arm_data_path}")
    arm_only_data = np.load(str(arm_data_path), allow_pickle=True).item()
    
    print(f"Loading full scene data from: {full_data_path}")
    full_scene_data = np.load(str(full_data_path), allow_pickle=True).item()
    
    # Get sorted frame keys
    arm_frame_keys = sorted(arm_only_data.keys())
    full_frame_keys = sorted(full_scene_data.keys())
    
    # Synchronize sequences
    arm_frame_keys = arm_frame_keys[arm_green_frame:]
    full_frame_keys = full_frame_keys[full_green_frame:]
    
    n_frames = min(len(arm_frame_keys), len(full_frame_keys))
    
    # If using precomputed masks, limit to available masks (accounting for green_frame offset)
    # Masks are named by original frame index, so we need masks from green_frame onwards
    if precomputed_mask_dir is not None:
        # Count available masks starting from green_frame
        available_mask_count = 0
        for mask_idx in range(full_green_frame, full_green_frame + n_frames):
            mask_path = precomputed_mask_dir / f"mask_frame_{mask_idx:04d}.npy"
            if mask_path.exists():
                available_mask_count += 1
            else:
                break  # Stop at first missing mask
        
        if available_mask_count < n_frames:
            print(f"Warning: Only {available_mask_count} masks available (from frame {full_green_frame}), limiting frames")
            n_frames = available_mask_count
    
    arm_frame_keys = arm_frame_keys[:n_frames]
    full_frame_keys = full_frame_keys[:n_frames]
    
    print(f"\nSynchronized sequences:")
    print(f"  Arm-only: starting from frame {arm_green_frame}")
    print(f"  Full scene: starting from frame {full_green_frame}")
    print(f"  Total frames: {n_frames}")
    if precomputed_mask_dir is not None:
        print(f"  Using precomputed masks from: {precomputed_mask_dir}")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    frames_3d_dir = output_dir / "frames_3d"
    frames_3d_dir.mkdir(exist_ok=True)
    
    # ================================================================
    # Initialize Tracker
    # ================================================================
    
    print(f"\nInitializing WireTracker...")
    tracker = WireTracker(**tracker_params)
    
    print(f"  Keypoints: {tracker.n_keypoints}")
    print(f"  Target topology: {tracker.target_branch_nodes} branch, {tracker.target_leaf_nodes} leaf")
    print(f"  CPD beta: {tracker.cpd_beta}, lambda: {tracker.cpd_lambda}")
    print(f"  Edge tolerance: {tracker.edge_tolerance * 100:.0f}%")
    if tracker.ee_poses_3d is not None:
        print(f"  EE poses shape: {tracker.ee_poses_3d.shape}")
    
    # ================================================================
    # Compute Fixed Axis Limits - Cropped to Robot Arm/Wire Region
    # ================================================================
    
    print(f"\nComputing fixed axis limits (cropped to arm/wire region)...")
    first_full_data = full_scene_data[full_frame_keys[0]]
    first_depth = first_full_data['transformed_depth']
    
    # Crop region: focus on robot arm and wire area
    # Depth range: 400mm to 1200mm (captures arm workspace)
    # This filters out far background
    crop_z_min = 400
    crop_z_max = 1600
    crop_x_min = -700
    crop_x_max = 850
    crop_y_min = -700
    crop_y_max = 400
    
    # Extract cropped point cloud from first frame
    valid = (first_depth > 0) & (first_depth < crop_z_max) & (first_depth > crop_z_min)
    rows, cols = np.where(valid)
    if len(rows) > 0:
        z = first_depth[rows, cols].astype(np.float64)
        x = (cols - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (rows - intrinsics[1, 2]) * z / intrinsics[1, 1]
        
        # Further crop by X and Y
        crop_mask = (x >= crop_x_min) & (x <= crop_x_max) & (y >= crop_y_min) & (y <= crop_y_max)
        x, y, z = x[crop_mask], y[crop_mask], z[crop_mask]
        
        if len(x) > 0:
            # Set fixed limits based on cropped region with padding
            fixed_xlim = (crop_x_min, crop_x_max)
            fixed_ylim = (crop_y_min, crop_y_max)
            fixed_zlim = (crop_z_min, crop_z_max)
        else:
            fixed_xlim = (-600, 600)
            fixed_ylim = (-400, 800)
            fixed_zlim = (400, 1200)
    else:
        fixed_xlim = (-600, 600)
        fixed_ylim = (-400, 800)
        fixed_zlim = (400, 1200)
    
    print(f"  Crop region:")
    print(f"  X: [{fixed_xlim[0]:.0f}, {fixed_xlim[1]:.0f}]")
    print(f"  Y: [{fixed_ylim[0]:.0f}, {fixed_ylim[1]:.0f}]")
    print(f"  Z: [{fixed_zlim[0]:.0f}, {fixed_zlim[1]:.0f}]")
    
    # ================================================================
    # Process Frames
    # ================================================================
    
    print(f"\n{'='*60}")
    print("PROCESSING FRAMES")
    print("=" * 60)
    
    # Storage
    all_results = []
    video_writer = None
    video_writer_3d = None
    
    # Trajectory history for visualization
    traj_history_2d = []  # List of K x 2 arrays
    traj_history_3d = []  # List of K x 3 arrays
    tail_length = 60  # Number of frames for trajectory tail
    
    # Timing
    total_time = 0.0
    
    for i in range(n_frames):
        frame_start = time.time()
        
        arm_frame_key = arm_frame_keys[i]
        full_frame_key = full_frame_keys[i]
        
        # Load arm data
        arm_data = arm_only_data[arm_frame_key]
        arm_depth = arm_data['transformed_depth'].copy()
        
        # Load full scene data
        full_data = full_scene_data[full_frame_key]
        full_rgb = full_data['color'][:, :, ::-1]  # BGR to RGB
        full_depth = full_data['transformed_depth'].copy()
        
        # Load precomputed arm mask if available
        # Masks are named by original frame index, so add green_frame offset
        precomputed_arm_mask = None
        if precomputed_mask_dir is not None:
            original_frame_idx = i + full_green_frame
            mask_path = precomputed_mask_dir / f"mask_frame_{original_frame_idx:04d}.npy"
            if mask_path.exists():
                precomputed_arm_mask = np.load(str(mask_path))
        
        # Process frame
        result = tracker.process_frame(full_depth, arm_depth, full_rgb, 
                                        precomputed_arm_mask=precomputed_arm_mask)
        
        frame_time = time.time() - frame_start
        total_time += frame_time
        
        # Extract results
        success = result.get('success', False)
        mode = result.get('mode', 'unknown')
        foreground_mask = result.get('foreground_mask', np.zeros_like(full_depth, dtype=np.uint8))
        skeleton_mask = result.get('skeleton_mask', np.zeros_like(full_depth, dtype=np.uint8))
        
        if success:
            keypoints = result['keypoints']
            keypoints_2d = result['keypoints_2d']
            edges = result['edges']
            
            # Get detected nodes for visualization
            detected_branch_3d = result.get('detected_branch', np.empty((0, 3)))
            detected_leaf_3d = result.get('detected_leaf', np.empty((0, 3)))
            
            # Project detected nodes to 2D
            detected_branch_2d = tracker._project_3d_to_2d(detected_branch_3d) if len(detected_branch_3d) > 0 else None
            detected_leaf_2d = tracker._project_3d_to_2d(detected_leaf_3d) if len(detected_leaf_3d) > 0 else None
            
            # Compute edge error stats
            edge_errors = result.get('edge_errors', np.array([]))
            if len(edge_errors) > 0:
                err_mean = np.mean(edge_errors) * 100
                err_max = np.max(edge_errors) * 100
            else:
                err_mean, err_max = 0.0, 0.0
            edge_rmse_mm = result.get('edge_rmse_mm', 0.0)
            
            # Format timing info
            timing = result.get('timing', {})
            timing_str = ""
            if timing:
                seg_time = timing.get('segmentation', 0) * 1000
                if mode == 'init':
                    timing_str = (f" | seg:{seg_time:.1f}ms "
                                  f"node:{timing.get('node_detection', 0)*1000:.1f}ms "
                                  f"prune:{timing.get('pruning', 0)*1000:.1f}ms "
                                  f"fps:{timing.get('fps', 0)*1000:.1f}ms "
                                  f"repul:{timing.get('repulsion', 0)*1000:.1f}ms "
                                  f"topo:{timing.get('topology', 0)*1000:.1f}ms "
                                  f"total:{timing.get('total', 0)*1000:.1f}ms")
                else:
                    timing_str = (f" | seg:{seg_time:.1f}ms "
                                  f"node:{timing.get('node_detection', 0)*1000:.1f}ms "
                                  f"cpd:{timing.get('cpd', 0)*1000:.1f}ms "
                                  f"hung:{timing.get('hungarian', 0)*1000:.1f}ms "
                                  f"geom:{timing.get('geometry', 0)*1000:.1f}ms "
                                  f"total:{timing.get('total', 0)*1000:.1f}ms")
            
            print(f"Frame {i:4d}: {mode:8s} | {len(keypoints):2d} keypoints | "
                f"edge_err: mean={err_mean:5.1f}%, max={err_max:5.1f}%, "
                f"rmse={edge_rmse_mm:5.2f}mm{timing_str}")
        else:
            keypoints_2d = np.empty((0, 2))
            edges = []
            detected_branch_2d = None
            detected_leaf_2d = None
            edge_rmse_mm = 0.0
            
            reason = result.get('reason', 'unknown')
            print(f"Frame {i:4d}: {mode:8s} | FAILED ({reason}) | frame_time: {frame_time*1000:.1f}ms")
        
        # Store result
        all_results.append({
            'frame_idx': i,
            'success': success,
            'mode': mode,
            'keypoints_3d': result.get('keypoints', np.empty((0, 3))),
            'keypoints_2d': keypoints_2d,
            'edges': edges,
            'edge_errors': result.get('edge_errors', np.array([])),
            'edge_rmse_mm': edge_rmse_mm,
        })
        
        # Update trajectory history
        keypoints_3d = result.get('keypoints', np.empty((0, 3)))
        if success and len(keypoints_3d) > 0:
            traj_history_3d.append(keypoints_3d.copy())
            traj_history_2d.append(keypoints_2d.copy())
        else:
            # Use NaN for failed frames to maintain alignment
            if len(traj_history_3d) > 0:
                nan_3d = np.full_like(traj_history_3d[-1], np.nan)
                nan_2d = np.full_like(traj_history_2d[-1], np.nan)
            else:
                nan_3d = np.full((tracker.n_keypoints, 3), np.nan)
                nan_2d = np.full((tracker.n_keypoints, 2), np.nan)
            traj_history_3d.append(nan_3d)
            traj_history_2d.append(nan_2d)
        
        # Convert trajectory history to numpy arrays for visualization
        traj_hist_2d_arr = np.array(traj_history_2d) if len(traj_history_2d) > 0 else None
        traj_hist_3d_arr = np.array(traj_history_3d) if len(traj_history_3d) > 0 else None
        
        # Create visualization
        viz = create_visualization(
            full_rgb,
            foreground_mask,
            skeleton_mask,
            keypoints_2d,
            edges,
            detected_branch_2d,
            detected_leaf_2d,
            mode,
            n_branch=tracker.reference_n_branch,
            n_leaf=tracker.reference_n_leaf,
            frame_idx=i,
            traj_history_2d=traj_hist_2d_arr,
            tail_length=tail_length,
        )
        
        # Initialize video writer
        if video_writer is None:
            H_viz, W_viz = viz.shape[:2]
            video_path = str(output_dir / "wire_tracking_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W_viz, H_viz))
        
        # Write frame
        video_writer.write(cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
        
        # Save frame image
        frame_path = frames_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
        
        # Create 3D visualization with cropped point cloud (arm/wire region)
        edges_for_viz = edges if success else []
        
        # Extract cropped point cloud with RGB colors
        valid = (full_depth > crop_z_min) & (full_depth < crop_z_max)
        rows, cols = np.where(valid)
        if len(rows) > 0:
            z = full_depth[rows, cols].astype(np.float64)
            x = (cols - intrinsics[0, 2]) * z / intrinsics[0, 0]
            y = (rows - intrinsics[1, 2]) * z / intrinsics[1, 1]
            colors = full_rgb[rows, cols]  # N x 3 RGB
            
            # Crop by X and Y
            crop_mask = (x >= crop_x_min) & (x <= crop_x_max) & (y >= crop_y_min) & (y <= crop_y_max)
            x, y, z = x[crop_mask], y[crop_mask], z[crop_mask]
            colors = colors[crop_mask]
            
            full_pc = np.stack([x, y, z], axis=1)
            pc_colors = colors
        else:
            full_pc = np.empty((0, 3))
            pc_colors = np.empty((0, 3))
        
        viz_3d = create_3d_visualization(
            keypoints_3d=keypoints_3d,
            edges=edges_for_viz,
            full_pc=full_pc,
            pc_colors=pc_colors,
            n_branch=tracker.reference_n_branch,
            n_leaf=tracker.reference_n_leaf,
            frame_idx=i,
            mode=mode,
            pc_downsample=8000,  # Max 8K points for faster visualization
            fixed_xlim=fixed_xlim,
            fixed_ylim=fixed_ylim,
            fixed_zlim=fixed_zlim,
            traj_history_3d=traj_hist_3d_arr,
            tail_length=tail_length,
        )
        
        # Initialize 3D video writer
        if video_writer_3d is None:
            H_3d, W_3d = viz_3d.shape[:2]
            video_path_3d = str(output_dir / "wire_tracking_3d.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer_3d = cv2.VideoWriter(video_path_3d, fourcc, fps, (W_3d, H_3d))
        
        # Write 3D frame
        video_writer_3d.write(cv2.cvtColor(viz_3d, cv2.COLOR_RGB2BGR))
        
        # Save 3D frame image
        frame_path_3d = frames_3d_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(frame_path_3d), cv2.cvtColor(viz_3d, cv2.COLOR_RGB2BGR))
        
        # Free memory periodically
        del viz, viz_3d, full_pc, pc_colors
        if i % 10 == 0:
            gc.collect()
    
    # ================================================================
    # Finalize
    # ================================================================
    
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved to: {output_dir / 'wire_tracking_video.mp4'}")
    
    if video_writer_3d is not None:
        video_writer_3d.release()
        print(f"3D video saved to: {output_dir / 'wire_tracking_3d.mp4'}")
    
    # Save results
    results_path = output_dir / "tracking_results.npy"
    np.save(str(results_path), all_results, allow_pickle=True)
    print(f"Results saved to: {results_path}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print("TRACKING STATISTICS")
    print("=" * 60)
    
    n_success = sum(1 for r in all_results if r['success'])
    n_init = sum(1 for r in all_results if r['mode'] == 'init')
    n_track = sum(1 for r in all_results if r['mode'] == 'track')
    n_restart = sum(1 for r in all_results if r['mode'] == 'restart')
    n_skip = sum(1 for r in all_results if r['mode'] == 'skip')
    
    print(f"Total frames: {n_frames}")
    print(f"Successful:   {n_success} ({n_success/n_frames*100:.1f}%)")
    print(f"  Init:       {n_init}")
    print(f"  Track:      {n_track}")
    print(f"  Restart:    {n_restart}")
    print(f"  Skip:       {n_skip}")
    print(f"Average time: {total_time/n_frames:.4f}s per frame")
    
    # Edge error statistics
    all_edge_errors = [r['edge_errors'] for r in all_results if len(r['edge_errors']) > 0]
    if all_edge_errors:
        all_errors = np.concatenate(all_edge_errors)
        all_edge_rmse_mm = [r['edge_rmse_mm'] for r in all_results if r['edge_rmse_mm'] > 0]
        print(f"\nEdge Length Errors:")
        print(f"  Mean:   {np.mean(all_errors)*100:.2f}%")
        print(f"  Std:    {np.std(all_errors)*100:.2f}%")
        print(f"  Max:    {np.max(all_errors)*100:.2f}%")
        if len(all_edge_rmse_mm) > 0:
            print(f"  RMSE:   {np.mean(all_edge_rmse_mm):.3f} mm")
        print(f"  Within tolerance ({tracker.edge_tolerance*100:.0f}%): "
              f"{np.mean(all_errors <= tracker.edge_tolerance)*100:.1f}%")
    
    print(f"\nOutput directory: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
