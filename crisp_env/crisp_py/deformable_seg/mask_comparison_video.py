"""
Generate a mask comparison video showing 4 panels:
  1. Raw BDLO mask (green) + EE0/EE1 projected positions
  2. After depth filtering (cyan) - pixels lost shown in red
  3. After top-1 connected component (orange) - pixels lost shown in red
  4. Summary: final mask + all losses + detected skeleton nodes
"""
import numpy as np
import cv2
from pathlib import Path
from scipy import ndimage
from scipy.spatial.transform import Rotation as R


def get_top_k_components(mask, k=1):
    labeled, n_labels = ndimage.label(mask)
    if n_labels <= k:
        return mask.copy()
    sizes = ndimage.sum(mask, labeled, range(1, n_labels + 1))
    top_k_labels = np.argsort(sizes)[-k:] + 1
    result = np.zeros_like(mask)
    for label in top_k_labels:
        result[labeled == label] = 1
    return result


def apply_depth_threshold(mask, depth, max_depth=2000.0):
    filtered = mask.copy()
    filtered[depth > max_depth] = 0
    filtered[depth <= 0] = 0
    filtered[np.isnan(depth)] = 0
    filtered[np.isinf(depth)] = 0
    return filtered


def make_overlay(rgb, mask, color=(0, 255, 0), alpha=0.4):
    out = rgb.copy()
    overlay = np.zeros_like(rgb)
    overlay[mask > 0] = color
    out = cv2.addWeighted(out, 1.0, overlay, alpha, 0)
    return out


def pose7_to_matrix(pose):
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    quat = pose[3:]
    T[:3, :3] = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    return T


def project_3d_to_2d(pos_3d_mm, K):
    """Project 3D position (mm) to 2D pixel (col, row)."""
    pos_m = pos_3d_mm / 1000.0
    if pos_m[2] <= 0:
        return None
    px = K @ pos_m
    col = int(px[0] / px[2])
    row = int(px[1] / px[2])
    return (col, row)


def detect_skeleton_nodes(skeleton_mask):
    """Detect branch and leaf nodes on skeleton."""
    skel = (skeleton_mask > 0).astype(np.uint8)
    # Count neighbors for each skeleton pixel
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_count = cv2.filter2D(skel, -1, kernel)
    neighbor_count = neighbor_count * skel

    # Branch nodes: 3+ neighbors, leaf nodes: 1 neighbor
    branch_mask = (neighbor_count >= 3) & (skel > 0)
    leaf_mask = (neighbor_count == 1) & (skel > 0)

    # Cluster nearby branch pixels
    branch_points = []
    if branch_mask.any():
        labeled, n = ndimage.label(ndimage.binary_dilation(branch_mask, iterations=3))
        for i in range(1, n + 1):
            ys, xs = np.where((labeled == i) & branch_mask)
            branch_points.append((int(ys.mean()), int(xs.mean())))

    # Cluster nearby leaf pixels
    leaf_points = []
    if leaf_mask.any():
        labeled, n = ndimage.label(ndimage.binary_dilation(leaf_mask, iterations=5))
        for i in range(1, n + 1):
            ys, xs = np.where((labeled == i) & leaf_mask)
            leaf_points.append((int(ys.mean()), int(xs.mean())))

    return branch_points, leaf_points


def draw_ee_and_nodes(panel, ee0_2d, ee1_2d, branch_pts, leaf_pts):
    """Draw EE positions and detected nodes on a panel."""
    # Draw EE0 (red circle)
    if ee0_2d is not None:
        cv2.circle(panel, ee0_2d, 10, (255, 0, 0), 3)
        cv2.putText(panel, "EE0", (ee0_2d[0] + 12, ee0_2d[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Draw EE1 (blue circle)
    if ee1_2d is not None:
        cv2.circle(panel, ee1_2d, 10, (0, 100, 255), 3)
        cv2.putText(panel, "EE1", (ee1_2d[0] + 12, ee1_2d[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

    # Draw detected branch nodes (magenta filled)
    for idx, (r, c) in enumerate(branch_pts):
        cv2.circle(panel, (c, r), 6, (200, 0, 200), -1)
        cv2.putText(panel, f"B{idx}", (c + 8, r - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 0, 200), 1)

    # Draw detected leaf nodes (yellow filled)
    for idx, (r, c) in enumerate(leaf_pts):
        cv2.circle(panel, (c, r), 6, (255, 255, 0), -1)
        cv2.putText(panel, f"L{idx}", (c + 8, r - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)


def main():
    base = Path('/mnt/mydisk/captured_data_double_arm/bdlo_yellow_2sec/chunk_0')
    calib_dir = Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects/captured_calibration_data/test_0228')
    max_depth = 2000.0

    print("Loading data...")
    rgbd = np.load(base / 'rgbd.npz')
    color = rgbd['color'][-600:]
    depth = rgbd['depth'][-600:]
    masks = np.load(base / 'bdlo_masks' / 'masks.npz')['masks'][-600:]

    # Load EE poses
    left_npz = np.load(base / 'left_arm_poses.npz')
    right_npz = np.load(base / 'right_arm_poses.npz')
    n_total = len(left_npz.files)
    start_idx = max(0, n_total - 600)
    left_poses = [left_npz[f'arr_{i}'] for i in range(start_idx, n_total)]
    right_poses = [right_npz[f'arr_{i}'] for i in range(start_idx, n_total)]

    # Load calibration
    tf = np.load(calib_dir / 'transform_ee_cam_world.npz')
    T_left = tf['T_left_base2cam']
    T_right = tf['T_right_base2cam']
    K = tf['K']

    n_frames = len(color)
    h, w = depth[0].shape
    out_w = w * 2
    out_h = h * 2

    out_path = base / 'bdlo_masks' / 'mask_comparison.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, 15, (out_w, out_h))

    print(f"Generating {n_frames} frames...")
    for i in range(n_frames):
        rgb = color[i]
        d = depth[i]
        raw_mask = masks[i]

        # Compute EE positions in camera frame (mm)
        T_left_ee = pose7_to_matrix(left_poses[i])
        ee0_cam = (T_left @ T_left_ee)[:3, 3] * 1000
        T_right_ee = pose7_to_matrix(right_poses[i])
        ee1_cam = (T_right @ T_right_ee)[:3, 3] * 1000

        # Project to 2D
        ee0_2d = project_3d_to_2d(ee0_cam, K)
        ee1_2d = project_3d_to_2d(ee1_cam, K)

        # Stage 1: Raw mask
        mask_raw = (raw_mask > 0).astype(np.uint8)

        # Stage 2: After depth filtering
        mask_depth = apply_depth_threshold(mask_raw, d, max_depth)
        lost_by_depth = (mask_raw > 0) & (mask_depth == 0)

        # Stage 3: After top-1
        mask_topk = get_top_k_components(mask_depth, k=1)
        lost_by_topk = (mask_depth > 0) & (mask_topk == 0)

        # Detect skeleton nodes on the depth-filtered mask
        skel = cv2.ximgproc.thinning(mask_depth * 255) if hasattr(cv2, 'ximgproc') else np.zeros_like(mask_depth)
        skel = (skel > 0).astype(np.uint8)
        branch_pts, leaf_pts = detect_skeleton_nodes(skel)

        # Panel 1: Raw mask (green) + EE
        panel_raw = make_overlay(rgb, mask_raw, color=(0, 255, 0))
        draw_ee_and_nodes(panel_raw, ee0_2d, ee1_2d, [], [])

        # Panel 2: After depth (cyan kept, red lost) + EE
        panel_depth = make_overlay(rgb, mask_depth, color=(0, 200, 255))
        panel_depth = make_overlay(panel_depth, lost_by_depth, color=(255, 0, 0), alpha=0.6)
        draw_ee_and_nodes(panel_depth, ee0_2d, ee1_2d, [], [])

        # Panel 3: After top-1 (orange kept, red lost) + EE
        panel_topk = make_overlay(rgb, mask_topk, color=(255, 160, 0))
        panel_topk = make_overlay(panel_topk, lost_by_topk, color=(255, 0, 0), alpha=0.6)
        draw_ee_and_nodes(panel_topk, ee0_2d, ee1_2d, [], [])

        # Panel 4: Summary + detected nodes + EE
        panel_loss = rgb.copy()
        panel_loss = make_overlay(panel_loss, mask_topk, color=(0, 255, 0), alpha=0.3)
        panel_loss = make_overlay(panel_loss, lost_by_depth, color=(255, 0, 0), alpha=0.6)
        panel_loss = make_overlay(panel_loss, lost_by_topk, color=(0, 0, 255), alpha=0.6)
        draw_ee_and_nodes(panel_loss, ee0_2d, ee1_2d, branch_pts, leaf_pts)

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        raw_total = int(mask_raw.sum())
        depth_lost = int(lost_by_depth.sum())
        topk_lost = int(lost_by_topk.sum())
        final = int(mask_topk.sum())

        cv2.putText(panel_raw, f"[1] Raw mask: {raw_total} px", (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(panel_depth, f"[2] After depth: {int(mask_depth.sum())} px (lost {depth_lost})", (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(panel_topk, f"[3] After top-1: {final} px (lost {topk_lost})", (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(panel_loss, f"[4] Nodes: {len(branch_pts)}B {len(leaf_pts)}L | {final}/{raw_total} px", (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(panel_loss, "RED=depth lost, BLUE=top-k lost, Magenta=B, Yellow=L", (10, 60), font, 0.5, (200, 200, 200), 1)

        for panel in [panel_raw, panel_depth, panel_topk, panel_loss]:
            cv2.putText(panel, f"Frame {i}", (w - 150, 30), font, 0.7, (200, 200, 200), 2)

        # Compose 2x2 grid
        top = np.hstack([panel_raw, panel_depth])
        bottom = np.hstack([panel_topk, panel_loss])
        frame_out = np.vstack([top, bottom])

        writer.write(cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR))

        if i % 50 == 0:
            print(f"  Frame {i}/{n_frames}")

    writer.release()
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
