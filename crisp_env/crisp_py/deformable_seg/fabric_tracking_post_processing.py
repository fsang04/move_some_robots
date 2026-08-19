"""
Fabric Tracking Post-Processing Script

Loads tracking results, applies trajectory smoothing, and creates visualizations
showing before/after smoothing with trajectory tails.

Outputs:
    - smooth_traj_2d.mp4: 2D visualization with raw (pink) vs smooth (white) trajectories
    - smooth_traj_3d.mp4: 3D visualization with raw (pink) vs smooth (white) trajectories

Usage:
    python fabric_tracking_post_processing.py
    python fabric_tracking_post_processing.py --sigma 3.0 --tail_length 60

Author: Auto-generated
Date: 2026-02-23
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
import PIL.Image
import io
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d


# ============================================================
# CAMERA INTRINSICS
# ============================================================
INTRINSICS = np.array([
    [606.1124267578125, 0, 641.7578125],
    [0, 605.8821411132812, 365.6518859863281],
    [0, 0, 1]
], dtype=np.float64)


# ============================================================
# FABRIC GRID CONFIGURATION
# ============================================================
GRID_ROWS = 5
GRID_COLS = 5
N_KEYPOINTS = GRID_ROWS * GRID_COLS  # 25

CORNER_INDICES = [0, 4, 20, 24]
BORDER_INDICES = [1, 2, 3, 5, 9, 10, 14, 15, 19, 21, 22, 23]
INTERIOR_INDICES = [6, 7, 8, 11, 12, 13, 16, 17, 18]

# Start frame (must match fabric_tracking_main.py)
START_FRAME = 18


# ============================================================
# PROJECTION UTILITIES
# ============================================================

def project_3d_to_2d(points_3d: np.ndarray, intrinsics: np.ndarray = INTRINSICS) -> np.ndarray:
    """Project 3D points to 2D pixel coordinates (row, col)."""
    if len(points_3d) == 0:
        return np.empty((0, 2), dtype=np.float64)
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    z_safe = np.maximum(z, 1e-6)
    
    u = (x * fx) / z_safe + cx  # col
    v = (y * fy) / z_safe + cy  # row
    
    return np.stack([v, u], axis=1)  # row, col


# ============================================================
# TRAJECTORY SMOOTHING
# ============================================================

def smooth_trajectories(keypoints_3d_seq: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to keypoint trajectories.
    
    Args:
        keypoints_3d_seq: T x K x 3 array of keypoints over time
        sigma: Gaussian filter sigma
    
    Returns:
        smoothed: T x K x 3 smoothed keypoints
    """
    T, K, D = keypoints_3d_seq.shape
    smoothed = np.zeros_like(keypoints_3d_seq)
    
    for k in range(K):
        for d in range(D):
            traj = keypoints_3d_seq[:, k, d]
            # Handle NaN values by interpolating
            valid = ~np.isnan(traj)
            if np.sum(valid) > 2:
                # Interpolate NaN values
                indices = np.arange(T)
                traj_interp = np.interp(indices, indices[valid], traj[valid])
                # Apply Gaussian smoothing
                smoothed[:, k, d] = gaussian_filter1d(traj_interp, sigma=sigma, mode='nearest')
            else:
                smoothed[:, k, d] = traj
    
    return smoothed


# ============================================================
# 2D VISUALIZATION
# ============================================================

def create_2d_visualization(
    rgb: np.ndarray,
    keypoints_2d_raw: np.ndarray,
    keypoints_2d_smooth: np.ndarray,
    traj_raw: np.ndarray,  # T_hist x K x 2 trajectory history (raw)
    traj_smooth: np.ndarray,  # T_hist x K x 2 trajectory history (smooth)
    edges: list,
    frame_idx: int = 0,
    tail_length: int = 60,
) -> np.ndarray:
    """
    Create 2x2 2D visualization grid.
    
    Layout:
        [Raw Keypoints + Tail]       [Raw Overlay on RGB]
        [Smooth Keypoints + Tail]    [Smooth Overlay on RGB]
    """
    H, W = rgb.shape[:2]
    
    # Colors
    EDGE_COLOR = [255, 165, 0]       # Orange for edges
    CORNER_COLOR = [255, 0, 0]       # Red for corners
    BORDER_COLOR = [255, 255, 0]    # Yellow for border
    INTERIOR_COLOR = [0, 255, 255]  # Cyan for interior
    RAW_TAIL_COLOR = [255, 105, 180]    # Hot pink for raw trajectory
    SMOOTH_TAIL_COLOR = [255, 255, 255]  # White for smooth trajectory
    
    KEYPOINT_RADIUS = 6
    EDGE_THICKNESS = 2
    TAIL_THICKNESS = 2
    
    def get_keypoint_color(idx):
        if idx in CORNER_INDICES:
            return CORNER_COLOR
        elif idx in BORDER_INDICES:
            return BORDER_COLOR
        else:
            return INTERIOR_COLOR
    
    def draw_keypoints_and_edges(canvas, keypoints_2d, edges, draw_edges=True):
        """Draw keypoints and edges on canvas."""
        if keypoints_2d is None or len(keypoints_2d) == 0:
            return
        
        kp_int = keypoints_2d.astype(int)
        
        # Draw edges
        if draw_edges and edges is not None:
            for (i, j) in edges:
                if i < len(kp_int) and j < len(kp_int):
                    pt1 = (kp_int[i, 1], kp_int[i, 0])  # (col, row)
                    pt2 = (kp_int[j, 1], kp_int[j, 0])
                    cv2.line(canvas, pt1, pt2, EDGE_COLOR, EDGE_THICKNESS)
        
        # Draw keypoints
        for idx in range(len(kp_int)):
            row, col = kp_int[idx]
            if 0 <= row < H and 0 <= col < W:
                color = get_keypoint_color(idx)
                cv2.circle(canvas, (col, row), KEYPOINT_RADIUS, color, -1)
                cv2.circle(canvas, (col, row), KEYPOINT_RADIUS + 1, (255, 255, 255), 1)
    
    def draw_trajectory_tail(canvas, traj_history, tail_color, tail_length):
        """Draw trajectory tails for keypoints."""
        if traj_history is None or len(traj_history) == 0:
            return
        
        T_hist, K, _ = traj_history.shape
        actual_tail = min(tail_length, T_hist)
        
        for idx in range(K):
            # Get trajectory for this keypoint
            traj = traj_history[-actual_tail:, idx, :]  # Last `actual_tail` frames
            
            # Draw trajectory as connected line segments with fading alpha
            for t in range(len(traj) - 1):
                pt1 = traj[t].astype(int)
                pt2 = traj[t + 1].astype(int)
                
                if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
                    continue
                
                row1, col1 = pt1
                row2, col2 = pt2
                
                if not (0 <= row1 < H and 0 <= col1 < W):
                    continue
                if not (0 <= row2 < H and 0 <= col2 < W):
                    continue
                
                # Fade color based on position in tail
                alpha = (t + 1) / len(traj)
                color = [int(c * alpha) for c in tail_color]
                cv2.line(canvas, (col1, row1), (col2, row2), color, TAIL_THICKNESS)
    
    # Row 1: Raw keypoints
    raw_vis = np.zeros((H, W, 3), dtype=np.uint8)
    draw_trajectory_tail(raw_vis, traj_raw, RAW_TAIL_COLOR, tail_length)
    draw_keypoints_and_edges(raw_vis, keypoints_2d_raw, edges)
    
    raw_overlay = rgb.copy()
    draw_trajectory_tail(raw_overlay, traj_raw, RAW_TAIL_COLOR, tail_length)
    draw_keypoints_and_edges(raw_overlay, keypoints_2d_raw, edges)
    
    # Add labels
    cv2.putText(raw_overlay, "Raw", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(raw_overlay, f"Frame: {frame_idx}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Row 2: Smooth keypoints
    smooth_vis = np.zeros((H, W, 3), dtype=np.uint8)
    draw_trajectory_tail(smooth_vis, traj_smooth, SMOOTH_TAIL_COLOR, tail_length)
    draw_keypoints_and_edges(smooth_vis, keypoints_2d_smooth, edges)
    
    smooth_overlay = rgb.copy()
    draw_trajectory_tail(smooth_overlay, traj_smooth, SMOOTH_TAIL_COLOR, tail_length)
    draw_keypoints_and_edges(smooth_overlay, keypoints_2d_smooth, edges)
    
    # Add labels
    cv2.putText(smooth_overlay, "Smoothed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(smooth_overlay, f"Frame: {frame_idx}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add legend
    legend_y = 90
    cv2.circle(smooth_overlay, (20, legend_y), 6, CORNER_COLOR, -1)
    cv2.putText(smooth_overlay, "Corner", (35, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(smooth_overlay, (20, legend_y + 25), 6, BORDER_COLOR, -1)
    cv2.putText(smooth_overlay, "Border", (35, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(smooth_overlay, (20, legend_y + 50), 6, INTERIOR_COLOR, -1)
    cv2.putText(smooth_overlay, "Interior", (35, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.line(smooth_overlay, (15, legend_y + 75), (40, legend_y + 75), RAW_TAIL_COLOR, 2)
    cv2.putText(smooth_overlay, "Raw Traj", (50, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.line(smooth_overlay, (15, legend_y + 100), (40, legend_y + 100), SMOOTH_TAIL_COLOR, 2)
    cv2.putText(smooth_overlay, "Smooth Traj", (50, legend_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Create grid
    row1 = np.concatenate([raw_vis, raw_overlay], axis=1)
    row2 = np.concatenate([smooth_vis, smooth_overlay], axis=1)
    grid = np.concatenate([row1, row2], axis=0)
    
    return grid


# ============================================================
# 3D VISUALIZATION
# ============================================================

def create_3d_visualization(
    keypoints_3d_raw: np.ndarray,
    keypoints_3d_smooth: np.ndarray,
    traj_raw_3d: np.ndarray,  # T_hist x K x 3
    traj_smooth_3d: np.ndarray,  # T_hist x K x 3
    edges: list,
    frame_idx: int = 0,
    tail_length: int = 60,
    fixed_xlim: tuple = None,
    fixed_ylim: tuple = None,
    fixed_zlim: tuple = None,
    figsize: tuple = (16, 16),
) -> np.ndarray:
    """
    Create 2x2 3D visualization grid.
    
    Layout:
        [Raw View 1]        [Smooth View 1]
        [Raw View 2]        [Smooth View 2]
    """
    # Colors (normalized to 0-1 for matplotlib)
    CORNER_COLOR = np.array([255, 0, 0]) / 255.0
    BORDER_COLOR = np.array([255, 255, 0]) / 255.0
    INTERIOR_COLOR = np.array([0, 255, 255]) / 255.0
    EDGE_COLOR = np.array([255, 165, 0]) / 255.0
    RAW_TAIL_COLOR = np.array([255, 105, 180]) / 255.0  # Hot pink
    SMOOTH_TAIL_COLOR = np.array([255, 255, 255]) / 255.0  # White
    
    fig = plt.figure(figsize=figsize)
    
    def get_keypoint_color(idx):
        if idx in CORNER_INDICES:
            return CORNER_COLOR
        elif idx in BORDER_INDICES:
            return BORDER_COLOR
        else:
            return INTERIOR_COLOR
    
    def setup_ax(ax, keypoints_3d, traj_3d, edges, 
                 tail_color, title, tail_length,
                 elev=100, azim=-90):
        """Setup 3D axis with keypoints and trajectory tails."""
        ax.set_title(f"{title}\nFrame: {frame_idx}", fontsize=10)
        
        # Draw trajectory tails
        if traj_3d is not None and len(traj_3d) > 0:
            T_hist = traj_3d.shape[0]
            actual_tail = min(tail_length, T_hist)
            K = traj_3d.shape[1]
            
            for idx in range(K):
                traj = traj_3d[-actual_tail:, idx, :]
                valid = ~np.any(np.isnan(traj), axis=1)
                
                if np.sum(valid) > 1:
                    traj_valid = traj[valid]
                    ax.plot(traj_valid[:, 0], traj_valid[:, 1], traj_valid[:, 2],
                           color=tail_color, linewidth=1.5, alpha=0.7, zorder=5)
        
        # Draw edges
        if keypoints_3d is not None and len(keypoints_3d) > 0 and edges is not None:
            for (i, j) in edges:
                if i < len(keypoints_3d) and j < len(keypoints_3d):
                    pts = keypoints_3d[[i, j], :]
                    if not np.any(np.isnan(pts)):
                        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                               color=EDGE_COLOR, linewidth=2, alpha=1.0, zorder=10)
        
        # Draw keypoints
        if keypoints_3d is not None and len(keypoints_3d) > 0:
            for idx in range(len(keypoints_3d)):
                pt = keypoints_3d[idx]
                if np.any(np.isnan(pt)):
                    continue
                color = get_keypoint_color(idx)
                size = 80 if idx in CORNER_INDICES else 60 if idx in BORDER_INDICES else 50
                ax.scatter(pt[0], pt[1], pt[2], c=[color], s=size, zorder=20, depthshade=False)
        
        # Set axis limits
        if fixed_xlim is not None:
            ax.set_xlim(fixed_xlim)
        if fixed_ylim is not None:
            ax.set_ylim(fixed_ylim)
        if fixed_zlim is not None:
            ax.set_zlim(fixed_zlim)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.view_init(elev=elev, azim=azim)
        ax.invert_yaxis()
    
    # Row 1: View 1 (elev=100, azim=-90)
    ax1 = fig.add_subplot(221, projection='3d')
    setup_ax(ax1, keypoints_3d_raw, traj_raw_3d, edges,
             RAW_TAIL_COLOR, "Raw (View 1)", tail_length, elev=100, azim=-90)
    
    ax2 = fig.add_subplot(222, projection='3d')
    setup_ax(ax2, keypoints_3d_smooth, traj_smooth_3d, edges,
             SMOOTH_TAIL_COLOR, "Smoothed (View 1)", tail_length, elev=100, azim=-90)
    
    # Row 2: View 2 (elev=24, azim=-90)
    ax3 = fig.add_subplot(223, projection='3d')
    setup_ax(ax3, keypoints_3d_raw, traj_raw_3d, edges,
             RAW_TAIL_COLOR, "Raw (View 2)", tail_length, elev=24, azim=-90)
    
    ax4 = fig.add_subplot(224, projection='3d')
    setup_ax(ax4, keypoints_3d_smooth, traj_smooth_3d, edges,
             SMOOTH_TAIL_COLOR, "Smoothed (View 2)", tail_length, elev=24, azim=-90)
    
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = np.array(PIL.Image.open(buf))[:, :, :3]
    plt.close(fig)
    
    return img


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Fabric Tracking Post-Processing')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='Gaussian smoothing sigma (default: 2.0)')
    parser.add_argument('--tail_length', type=int, default=60,
                        help='Trajectory tail length in frames (default: 60)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video FPS (default: 30)')
    args = parser.parse_args()
    
    # Paths
    tracking_results_path = Path('/home/yehengz/deformable_seg/data/arm_traj4_fabric/fabric_tracking_output/tracking_results.npy')
    full_data_path = Path('/home/yehengz/deformable_seg/data/full/tracking_fabric2_data.npy')
    output_dir = Path('/home/yehengz/deformable_seg/data/arm_traj4_fabric/fabric_tracking_output')
    
    print("=" * 60)
    print("FABRIC TRACKING POST-PROCESSING")
    print("=" * 60)
    
    # Load tracking results
    print(f"\nLoading tracking results from: {tracking_results_path}")
    tracking_results = np.load(str(tracking_results_path), allow_pickle=True)
    
    n_frames = len(tracking_results)
    print(f"  Total frames: {n_frames}")
    print(f"  Keypoints per frame: {N_KEYPOINTS}")
    
    # Build keypoints array (T x K x 3)
    keypoints_3d_raw = np.full((n_frames, N_KEYPOINTS, 3), np.nan, dtype=np.float64)
    edges_list = [None] * n_frames
    
    for i, r in enumerate(tracking_results):
        if r['success'] and len(r['keypoints_3d']) == N_KEYPOINTS:
            keypoints_3d_raw[i] = r['keypoints_3d']
            edges_list[i] = r['edges']
    
    # Count successful frames
    n_valid = np.sum(~np.any(np.isnan(keypoints_3d_raw), axis=(1, 2)))
    print(f"  Valid frames: {n_valid}/{n_frames}")
    
    # Smooth trajectories
    print(f"\nSmoothing trajectories with sigma={args.sigma}...")
    keypoints_3d_smooth = smooth_trajectories(keypoints_3d_raw, sigma=args.sigma)
    
    # Project to 2D
    print("Projecting to 2D...")
    keypoints_2d_raw = np.full((n_frames, N_KEYPOINTS, 2), np.nan, dtype=np.float64)
    keypoints_2d_smooth = np.full((n_frames, N_KEYPOINTS, 2), np.nan, dtype=np.float64)
    
    for i in range(n_frames):
        if not np.any(np.isnan(keypoints_3d_raw[i])):
            keypoints_2d_raw[i] = project_3d_to_2d(keypoints_3d_raw[i])
        if not np.any(np.isnan(keypoints_3d_smooth[i])):
            keypoints_2d_smooth[i] = project_3d_to_2d(keypoints_3d_smooth[i])
    
    # Load RGB images
    print(f"\nLoading RGB data from: {full_data_path}")
    full_scene_data = np.load(str(full_data_path), allow_pickle=True).item()
    # Start from START_FRAME to match fabric_tracking_main.py
    all_frame_keys = sorted(full_scene_data.keys())
    full_frame_keys = all_frame_keys[START_FRAME:START_FRAME + n_frames]
    print(f"  Using frames {START_FRAME} to {START_FRAME + n_frames - 1} from full data")
    
    # Compute fixed 3D axis limits
    valid_pts = keypoints_3d_raw[~np.isnan(keypoints_3d_raw).any(axis=2)]
    if len(valid_pts) > 0:
        margin = 100
        fixed_xlim = (valid_pts[:, 0].min() - margin, valid_pts[:, 0].max() + margin)
        fixed_ylim = (valid_pts[:, 1].min() - margin, valid_pts[:, 1].max() + margin)
        fixed_zlim = (valid_pts[:, 2].min() - margin, valid_pts[:, 2].max() + margin)
    else:
        fixed_xlim = (-600, 600)
        fixed_ylim = (-400, 800)
        fixed_zlim = (600, 1400)
    
    print(f"\n3D axis limits:")
    print(f"  X: [{fixed_xlim[0]:.0f}, {fixed_xlim[1]:.0f}]")
    print(f"  Y: [{fixed_ylim[0]:.0f}, {fixed_ylim[1]:.0f}]")
    print(f"  Z: [{fixed_zlim[0]:.0f}, {fixed_zlim[1]:.0f}]")
    
    # Initialize video writers
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_paths = {
        '2d': output_dir / f'smooth_traj_2d_sigma{args.sigma}.mp4',
        '3d': output_dir / f'smooth_traj_3d_sigma{args.sigma}.mp4',
    }
    video_writers = {'2d': None, '3d': None}
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    for i in range(n_frames):
        if i % 10 == 0:
            print(f"Processing frame {i}/{n_frames}...")
        
        # Load RGB
        full_frame_key = full_frame_keys[i]
        full_data = full_scene_data[full_frame_key]
        rgb = full_data['color'][:, :, ::-1]  # BGR to RGB
        
        # Get edges for this frame
        edges = edges_list[i] if edges_list[i] is not None else []
        
        # Trajectory history up to current frame
        traj_raw_2d = keypoints_2d_raw[:i+1]
        traj_smooth_2d = keypoints_2d_smooth[:i+1]
        traj_raw_3d = keypoints_3d_raw[:i+1]
        traj_smooth_3d = keypoints_3d_smooth[:i+1]
        
        # Current keypoints
        kp_raw_2d = keypoints_2d_raw[i]
        kp_smooth_2d = keypoints_2d_smooth[i]
        kp_raw_3d = keypoints_3d_raw[i]
        kp_smooth_3d = keypoints_3d_smooth[i]
        
        # ---- 2D Visualization ----
        viz_2d = create_2d_visualization(
            rgb, kp_raw_2d, kp_smooth_2d, traj_raw_2d, traj_smooth_2d, edges,
            frame_idx=i, tail_length=args.tail_length,
        )
        
        if video_writers['2d'] is None:
            H, W = viz_2d.shape[:2]
            video_writers['2d'] = cv2.VideoWriter(str(video_paths['2d']), fourcc, args.fps, (W, H))
        video_writers['2d'].write(cv2.cvtColor(viz_2d, cv2.COLOR_RGB2BGR))
        
        # ---- 3D Visualization ----
        viz_3d = create_3d_visualization(
            kp_raw_3d, kp_smooth_3d, traj_raw_3d, traj_smooth_3d, edges,
            frame_idx=i, tail_length=args.tail_length,
            fixed_xlim=fixed_xlim, fixed_ylim=fixed_ylim, fixed_zlim=fixed_zlim,
        )
        
        if video_writers['3d'] is None:
            H, W = viz_3d.shape[:2]
            video_writers['3d'] = cv2.VideoWriter(str(video_paths['3d']), fourcc, args.fps, (W, H))
        video_writers['3d'].write(cv2.cvtColor(viz_3d, cv2.COLOR_RGB2BGR))
        
        # Free memory
        del viz_2d, viz_3d
        if i % 10 == 0:
            gc.collect()
    
    # Release video writers
    for key, writer in video_writers.items():
        if writer is not None:
            writer.release()
            print(f"Video saved: {video_paths[key]}")
    
    # Save smoothed trajectories
    smoothed_results_path = output_dir / f'smoothed_keypoints_3d_sigma{args.sigma}.npy'
    np.save(str(smoothed_results_path), keypoints_3d_smooth)
    print(f"\nSmoothed keypoints saved: {smoothed_results_path}")
    
    print(f"\n{'='*60}")
    print("DONE!")
    print("=" * 60)
    print(f"\nOutput files:")
    for key, path in video_paths.items():
        print(f"  {path}")


if __name__ == "__main__":
    main()
