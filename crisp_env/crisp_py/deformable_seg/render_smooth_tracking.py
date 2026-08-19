#!/usr/bin/env python3
"""
Render tracking_smooth.mp4 from pre-computed 3d_keypoints.npz.

Loads smoothed 3D keypoints (with temporal + spatial Laplacian smoothing),
projects them to 2D, and renders a single-panel video matching the style of
tracking_full.mp4 (RGB + trajectory tail + edges + colored keypoints).

Usage:
    python render_smooth_tracking.py --faster --chunk-idx 0 --clip-idx 0
    python render_smooth_tracking.py --faster --chunk-idx 1 --clip-idx 1 --keypoint-key nosnap
    python render_smooth_tracking.py --eval-dir path/to/clip --chunk-dir path/to/chunk
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R


# ============================================================================
# PATHS
# ============================================================================

BDLO_EVAL_ROOT        = Path(__file__).resolve().parent / 'bdlo1_evaluation_results'
BDLO_FASTER_EVAL_ROOT = Path(__file__).resolve().parent / 'bdlo1_faster_evaluation_results'
BDLO_DATA_ROOT        = Path('/mnt/mydisk/captured_data_double_arm/bdlo_no_contact_4sec')
BDLO_FASTER_DATA_ROOT = Path('/mnt/mydisk/captured_data_double_arm/bdlo_no_contact_2sec')

CALIB_PATH = Path(__file__).resolve().parent.parent / (
    'hand_to_eye_calibration/roahm-deformable-objects/'
    'captured_calibration_data/test_0227/transform_ee_cam_world.npz'
)

FRAMES_PER_CLIP = 300
N_BRANCH = 2
N_LEAF   = 4

# Match tracking_full.mp4 colors (RGB)
BRANCH_COLOR = (128,   0, 128)   # Purple
LEAF_COLOR   = (255, 255,   0)   # Yellow
INTER_COLOR  = (255, 165,   0)   # Orange
EDGE_COLOR   = ( 50, 205,  50)   # Green
TAIL_COLOR   = (100, 255, 100)   # Light green


# ============================================================================
# SMOOTHING
# ============================================================================

def temporal_smooth(keypoints_3d: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along time axis with NaN interpolation."""
    T, K, D = keypoints_3d.shape
    out = np.empty_like(keypoints_3d, dtype=np.float64)
    indices = np.arange(T)
    for k in range(K):
        for d in range(D):
            traj = keypoints_3d[:, k, d].astype(np.float64)
            valid = ~np.isnan(traj)
            if valid.sum() > 2:
                traj = np.interp(indices, indices[valid], traj[valid])
            out[:, k, d] = gaussian_filter1d(traj, sigma=sigma, mode='nearest')
    return out


def spatial_laplacian_smooth(
    keypoints_3d: np.ndarray,
    edge_connections: np.ndarray,
    iters: int = 3,
    weight: float = 0.4,
) -> np.ndarray:
    """Blend each node toward the mean of its graph neighbors."""
    N = keypoints_3d.shape[1]
    neighbors = [[] for _ in range(N)]
    for i, j in edge_connections:
        neighbors[i].append(j)
        neighbors[j].append(i)

    kp = keypoints_3d.copy()
    for _ in range(iters):
        new_kp = kp.copy()
        for node in range(N):
            nb = neighbors[node]
            if nb:
                new_kp[:, node, :] = (
                    (1 - weight) * kp[:, node, :] +
                    weight * kp[:, nb, :].mean(axis=1)
                )
        kp = new_kp
    return kp


# ============================================================================
# PROJECTION
# ============================================================================

def project_to_2d_rowcol(points_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project (N, 3) mm → (N, 2) as (row, col). Returns NaN for z<=0."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    pts = points_3d / 1000.0
    out = np.full((len(pts), 2), np.nan)
    z = pts[:, 2]
    valid = z > 0
    out[valid, 0] = fy * pts[valid, 1] / z[valid] + cy  # row
    out[valid, 1] = fx * pts[valid, 0] / z[valid] + cx  # col
    return out


# ============================================================================
# RENDERING
# ============================================================================

def render_frame(
    rgb: np.ndarray,
    kp2d: np.ndarray,
    edges: np.ndarray,
    traj_history: np.ndarray,
    frame_idx: int,
    tail_length: int = 60,
) -> np.ndarray:
    """
    Render a single frame: RGB + trajectory tail + edges + colored keypoints.

    Args:
        rgb:          (H, W, 3) RGB uint8
        kp2d:         (N, 2) as (row, col)
        edges:        (E, 2) edge index pairs
        traj_history: (T_hist, N, 2) past keypoint positions (row, col)
        frame_idx:    frame number for text overlay
    """
    H, W = rgb.shape[:2]
    vis = rgb.copy()

    # --- Trajectory tail ---
    if traj_history is not None and len(traj_history) > 1:
        T_hist = len(traj_history)
        K_nodes = traj_history.shape[1]
        start = max(0, T_hist - tail_length)
        for k in range(K_nodes):
            for t in range(start, T_hist - 1):
                pt1 = traj_history[t, k]
                pt2 = traj_history[t + 1, k]
                if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
                    continue
                r1, c1 = int(pt1[0]), int(pt1[1])
                r2, c2 = int(pt2[0]), int(pt2[1])
                if not (0 <= r1 < H and 0 <= c1 < W and 0 <= r2 < H and 0 <= c2 < W):
                    continue
                age = T_hist - 1 - t
                alpha = max(0.15, 1.0 - age / tail_length)
                color = tuple(int(c * alpha) for c in TAIL_COLOR)
                cv2.line(vis, (c1, r1), (c2, r2), color, 1)

    # --- Edges ---
    kp_int = np.round(kp2d).astype(int)
    for i, j in edges:
        if i >= len(kp_int) or j >= len(kp_int):
            continue
        r_i, c_i = kp_int[i]
        r_j, c_j = kp_int[j]
        if not (0 <= r_i < H and 0 <= c_i < W and 0 <= r_j < H and 0 <= c_j < W):
            continue
        if np.any(np.isnan(kp2d[i])) or np.any(np.isnan(kp2d[j])):
            continue
        cv2.line(vis, (c_i, r_i), (c_j, r_j), EDGE_COLOR, 2)

    # --- Keypoints ---
    for idx in range(len(kp_int)):
        if np.any(np.isnan(kp2d[idx])):
            continue
        r, c = kp_int[idx]
        if not (0 <= r < H and 0 <= c < W):
            continue
        if idx < N_BRANCH:
            color, radius = BRANCH_COLOR, 7
        elif idx < N_BRANCH + N_LEAF:
            color, radius = LEAF_COLOR, 6
        else:
            color, radius = INTER_COLOR, 4
        cv2.circle(vis, (c, r), radius, color, -1)

    cv2.putText(vis, f"Smooth  Frame {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Render tracking_smooth.mp4 from smoothed keypoints')
    parser.add_argument('--chunk-idx', type=int, default=1)
    parser.add_argument('--clip-idx', type=int, default=0)
    parser.add_argument('--faster', action='store_true')
    parser.add_argument('--eval-dir', type=str, default=None)
    parser.add_argument('--chunk-dir', type=str, default=None)
    parser.add_argument('--keypoint-key', type=str, default='full',
                        choices=['full', 'nosnap', 'noGeometry', 'cdcpd2'])
    parser.add_argument('--time-smooth-sigma', type=float, default=25.0)
    parser.add_argument('--spatial-smooth-iters', type=int, default=3)
    parser.add_argument('--spatial-smooth-weight', type=float, default=0.4)
    parser.add_argument('--tail-length', type=int, default=60)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--output-name', type=str, default='tracking_smooth.mp4')
    args = parser.parse_args()

    eval_root = BDLO_FASTER_EVAL_ROOT if args.faster else BDLO_EVAL_ROOT
    data_root = BDLO_FASTER_DATA_ROOT if args.faster else BDLO_DATA_ROOT

    eval_dir  = Path(args.eval_dir)  if args.eval_dir  else eval_root  / f'chunk_{args.chunk_idx}' / f'clip_{args.clip_idx}'
    chunk_dir = Path(args.chunk_dir) if args.chunk_dir else data_root  / f'chunk_{args.chunk_idx}'

    for p, label in [(eval_dir, 'eval-dir'), (chunk_dir, 'chunk-dir')]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}")
            return

    print("=" * 60)
    print("RENDER SMOOTH TRACKING VIDEO")
    print("=" * 60)
    print(f"Eval dir:  {eval_dir}")
    print(f"Chunk dir: {chunk_dir}")
    print(f"Key:       {args.keypoint_key}")

    # Load keypoints
    kp_data = np.load(eval_dir / '3d_keypoints.npz')
    keypoints_3d  = kp_data[args.keypoint_key].astype(np.float64)   # (T, N, 3) mm
    edge_connections = kp_data['edge_connection']                    # (E, 2)
    print(f"Keypoints: {keypoints_3d.shape}  edges: {edge_connections.shape}")

    # Smooth
    if args.time_smooth_sigma > 0:
        print(f"Temporal smoothing sigma={args.time_smooth_sigma}...")
        keypoints_3d = temporal_smooth(keypoints_3d, args.time_smooth_sigma)

    if args.spatial_smooth_iters > 0:
        print(f"Spatial Laplacian smoothing iters={args.spatial_smooth_iters} weight={args.spatial_smooth_weight}...")
        keypoints_3d = spatial_laplacian_smooth(
            keypoints_3d, edge_connections,
            iters=args.spatial_smooth_iters,
            weight=args.spatial_smooth_weight,
        )

    # Load intrinsics
    if CALIB_PATH.exists():
        tf = np.load(CALIB_PATH)
        K_raw = tf['K']
        K = np.array([[K_raw[0, 0], 0, K_raw[0, 2]],
                      [0, K_raw[1, 1], K_raw[1, 2]],
                      [0, 0, 1]], dtype=np.float64)
        print("Intrinsics: loaded from calibration file")
    else:
        K = np.array([[606.1124, 0, 641.7578],
                      [0, 605.8821, 365.6519],
                      [0, 0, 1]], dtype=np.float64)
        print("Intrinsics: using fallback ZED2 values")

    # Project all keypoints to 2D
    T = len(keypoints_3d)
    N = keypoints_3d.shape[1]
    kp2d_all = np.zeros((T, N, 2))
    for t in range(T):
        kp2d_all[t] = project_to_2d_rowcol(keypoints_3d[t], K)

    # Load RGB
    print("Loading RGB...")
    rgbd = np.load(chunk_dir / 'rgbd.npz', mmap_mode='r')
    color_all = rgbd['color'][-600:]                                     # (600, H, W, 3) BGR
    clip_start = args.clip_idx * FRAMES_PER_CLIP
    color = color_all[clip_start:clip_start + FRAMES_PER_CLIP]           # (300, H, W, 3) BGR
    n_frames = min(len(color), T)
    print(f"Frames to render: {n_frames}")

    # Write video
    out_path = eval_dir / args.output_name
    writer = None
    traj_history = []

    print(f"\nRendering {out_path.name} ...")
    for frame_idx in range(n_frames):
        rgb = cv2.cvtColor(color[frame_idx], cv2.COLOR_BGR2RGB)
        kp2d = kp2d_all[frame_idx]
        traj_history.append(kp2d.copy())
        traj_arr = np.array(traj_history)

        vis = render_frame(
            rgb=rgb,
            kp2d=kp2d,
            edges=edge_connections,
            traj_history=traj_arr,
            frame_idx=frame_idx,
            tail_length=args.tail_length,
        )

        bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        if writer is None:
            H, W = bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (W, H))

        writer.write(bgr)

    if writer is not None:
        writer.release()

    print(f"\nDone! Saved: {out_path}")


if __name__ == '__main__':
    main()
