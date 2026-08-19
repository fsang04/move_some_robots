"""
Create skeleton + nodes + keypoints visualization using saved keypoint data.
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from seg_utils import (
    compute_point_cloud_mask,
    filter_pcd_mask_dbscan,
    skelentonize,
    node_identification,
    prune_leaf_segments,
    mask_from_mst,
)

DEPTH_THRESHOLD = 1000


def remove_small_components(mask, min_size=100):
    """Remove small connected components from binary mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_mask[labels == i] = 1
    return cleaned_mask


def create_video_from_frames(frames, output_path, fps=30):
    """Create video from list of RGB frames."""
    if len(frames) == 0:
        print(f"No frames to write for {output_path}")
        return
    
    H, W = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Saved video: {output_path}")


def project_points_to_2d(points_3d, intrinsics):
    """Project 3D points to 2D pixel coordinates using camera intrinsics."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    u = (x * fx) / z + cx
    v = (y * fy) / z + cy
    return np.stack([v, u], axis=1)  # (row, col)


def main():
    print("Loading tracking data...")
    tracking_data = np.load("./data/full/tracking_BDLO_data.npy", allow_pickle=True).item()
    bg_data = np.load("./data/bg/tracking_BDLO_background_data.npy", allow_pickle=True).item()
    
    # Load saved keypoints
    keypoints_data = np.load("./tracking_output/all_keypoints_3d.npy", allow_pickle=True).item()
    all_keypoints_3d = keypoints_data["keypoints"]
    all_edges = keypoints_data["edges"]
    print(f"Loaded keypoints for {len(all_keypoints_3d)} frames")
    
    intrinsics = np.array([
        [606.1124267578125, 0, 641.7578125],
        [0, 605.8821411132812, 365.6518859863281],
        [0, 0, 1]
    ])
    
    output_dir = Path("./tracking_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_keys = sorted(tracking_data.keys())
    frame_keys = frame_keys[:225]  # Process only first 225 frames
    print(f"Processing {len(frame_keys)} frames")
    
    # Get background depth and apply threshold
    bg_depth = bg_data[0]['transformed_depth'].copy()
    bg_depth[bg_depth >= DEPTH_THRESHOLD] = 0
    
    arm_mask_dir = Path("/home/yehengz/deformable_seg/data/wire_tracking_arm_masks")

    # Get dual_arm_mask from first frame
    first_data = tracking_data[frame_keys[0]]
    first_depth = first_data['transformed_depth'].copy()
    first_pc_mask = compute_point_cloud_mask(
        bg_depth,
        first_depth,
        intrinsics,
        distance_threshold=18
    )
    first_depth_mask = ((first_depth > 0) & (first_depth < DEPTH_THRESHOLD)).astype(np.uint8)
    dual_arm_mask = ((1 - (first_pc_mask > 0).astype(np.uint8)) * first_depth_mask).astype(np.uint8)

    expected_leaf_nodes = 4

    print("\n" + "=" * 60)
    print("CREATING SKELETON + NODES + KEYPOINTS VISUALIZATION")
    print("=" * 60)

    skel_frames_dir = output_dir / "skeleton_nodes_keypoints_frames"
    skel_frames_dir.mkdir(parents=True, exist_ok=True)
    skel_video_frames = []

    for i, frame_key in enumerate(frame_keys):
        data = tracking_data[frame_key]
        rgb_image = data['color'][:, :, ::-1]  # BGR to RGB
        curr_depth = data['transformed_depth'].copy()

        # Recompute mask and skeleton for this frame
        arm_mask_path = arm_mask_dir / f"mask_frame_{i:04d}.npy"
        if arm_mask_path.exists():
            arm_mask = np.load(str(arm_mask_path))
            aug_arm_mask = ((arm_mask > 0) | (dual_arm_mask > 0)).astype(np.uint8)
        else:
            aug_arm_mask = dual_arm_mask.copy()
        depth_mask = ((curr_depth > 0) & (curr_depth < DEPTH_THRESHOLD)).astype(np.uint8)
        pc_mask = ((1 - aug_arm_mask) * depth_mask).astype(np.uint8)
        pc_mask[0:150, :] = 0
        pc_mask = filter_pcd_mask_dbscan(pc_mask, curr_depth, intrinsics, eps=30.0, min_samples=58)
        pc_mask = remove_small_components(pc_mask, min_size=1000)

        skeleton_pc_mask = skelentonize(pc_mask)
        branch_nodes, end_nodes, adjacency, coords = node_identification(skeleton_pc_mask, return_graph=True)

        if adjacency is not None and coords is not None:
            pruning_result = prune_leaf_segments(adjacency, coords, expected_num_leaf_nodes=expected_leaf_nodes)
            branch_nodes = pruning_result["branch_coords"]
            end_nodes = pruning_result["leaf_coords"]
            skeleton_pc_mask = mask_from_mst(pruning_result["adjacency"], pruning_result["coords"], skeleton_pc_mask.shape)

        # Get keypoints and edges for this frame
        keypoints = all_keypoints_3d[i]
        wire_edges = all_edges[i]

        # Project keypoints to 2D
        if keypoints.shape[0] > 0:
            keypoints_2d = project_points_to_2d(keypoints, intrinsics)
            keypoints_2d_int = np.round(keypoints_2d).astype(int)
        else:
            keypoints_2d_int = np.zeros((0, 2), dtype=int)

        # Create binary visualization with skeleton and nodes (no edges)
        H, W = rgb_image.shape[:2]
        
        # Start with black background
        binary_img = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Draw skeleton in white
        skeleton_mask_bool = skeleton_pc_mask > 0
        binary_img[skeleton_mask_bool] = [255, 255, 255]
        
        fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
        ax.imshow(binary_img)

        # Draw branch nodes (gold, larger radius=8) - first so keypoints appear on top
        for node in branch_nodes:
            row, col = int(node[0]), int(node[1])
            circ = Circle((col, row), radius=8, color='gold', fill=True, alpha=1.0)
            ax.add_patch(circ)

        # Draw leaf nodes (purple, larger radius=8) - first so keypoints appear on top
        for node in end_nodes:
            row, col = int(node[0]), int(node[1])
            circ = Circle((col, row), radius=8, color='purple', fill=True, alpha=1.0)
            ax.add_patch(circ)

        # Draw keypoints (red, smaller radius=4) - last so they appear on top
        for (row, col) in keypoints_2d_int:
            circ = Circle((col, row), radius=4, color='red', fill=True, alpha=1.0)
            ax.add_patch(circ)

        ax.axis('off')
        plt.tight_layout()
        skel_frame_path = skel_frames_dir / f"frame_{i:04d}.png"
        plt.savefig(str(skel_frame_path), dpi=100)
        plt.close(fig)

        # Read saved frame for video
        skel_frame_img = cv2.imread(str(skel_frame_path))
        skel_frame_img_rgb = cv2.cvtColor(skel_frame_img, cv2.COLOR_BGR2RGB)
        skel_video_frames.append(skel_frame_img_rgb)

        if (i + 1) % 50 == 0:
            print(f"  Skeleton visualization: {i + 1}/{len(frame_keys)} frames")

    print(f"\nCreating skeleton_nodes_keypoints.mp4 with {len(skel_video_frames)} frames...")
    create_video_from_frames(skel_video_frames, output_dir / "skeleton_nodes_keypoints.mp4", fps=30)
    print(f"Saved skeleton visualization frames to {skel_frames_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
