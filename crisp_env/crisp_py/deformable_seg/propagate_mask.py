"""
Geometry-constrained wire tracking using CPD + edge length preservation.

Pipeline:
Frame 0: Independent detection → FPS + Repulsion → Extract reference geometry (edges, lengths)
Frame N>0: CPD initialization → Correct branch/leaf → Edge length constraint optimization

Key features:
- CPD (Coherent Point Drift) for temporal coherence
- Hungarian matching for branch/leaf node correction
- Edge length constraints to preserve wire topology
"""

import numpy as np
import cv2
import time
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from scipy.optimize import linear_sum_assignment
from seg_utils import (
    compute_point_cloud_mask,
    filter_pcd_mask_dbscan,
    remove_small_components,
    skelentonize,
    node_identification,
    prune_leaf_segments,
    mask_from_mst,
)


DEPTH_THRESHOLD = 1000
MAX_THICKNESS = 20  # Maximum thickness in pixels for wire (adjust as needed)


# ============================================================
# CPD (Coherent Point Drift) Registration
# ============================================================

def cpd_register(Y, X, beta=2.0, lmbda=2.0, w=0.1, max_iter=100, tol=1e-5):
    """
    Non-rigid Coherent Point Drift registration.
    
    Aligns template point set Y to target point set X using a Gaussian mixture
    model with motion coherence regularization.
    
    Args:
        Y: M x D template point set (previous frame keypoints)
        X: N x D target point set (current frame foreground points)
        beta: Gaussian kernel width for motion coherence (larger = more rigid)
        lmbda: Regularization weight (larger = smoother deformation)
        w: Outlier weight in [0, 1] (expected fraction of outliers)
        max_iter: Maximum EM iterations
        tol: Convergence tolerance
    
    Returns:
        T_Y: M x D transformed template points
        P: M x N correspondence matrix (posterior probabilities)
    """
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    
    M, D = Y.shape
    N = X.shape[0]
    
    if M == 0 or N == 0:
        return Y.copy(), np.zeros((M, N))
    
    # Initialize
    T_Y = Y.copy()
    W = np.zeros((M, D))
    
    # Compute Gaussian kernel matrix G for motion coherence
    diff_Y = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
    G = np.exp(-np.sum(diff_Y ** 2, axis=2) / (2 * beta ** 2))
    
    # Initialize sigma^2
    diff_init = X[np.newaxis, :, :] - Y[:, np.newaxis, :]
    sigma2 = np.sum(diff_init ** 2) / (M * N * D)
    
    for iteration in range(max_iter):
        # E-step: Compute posterior probabilities
        diff = X[np.newaxis, :, :] - T_Y[:, np.newaxis, :]
        dist2 = np.sum(diff ** 2, axis=2)
        
        P_num = np.exp(-dist2 / (2 * sigma2))
        c = (w / (1 - w)) * (M / N) * ((2 * np.pi * sigma2) ** (D / 2))
        P_den = np.sum(P_num, axis=0, keepdims=True) + c
        P = P_num / (P_den + 1e-10)
        
        # M-step: Update W and sigma^2
        P1 = np.sum(P, axis=1)
        Np = np.sum(P1)
        
        P1_safe = np.maximum(P1, 1e-10)
        D_inv = np.diag(1.0 / P1_safe)
        
        A = G + lmbda * sigma2 * D_inv
        B = D_inv @ P @ X - Y
        
        try:
            W = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            W = np.linalg.lstsq(A, B, rcond=None)[0]
        
        T_Y_new = Y + G @ W
        
        # Update sigma^2
        diff_new = X[np.newaxis, :, :] - T_Y_new[:, np.newaxis, :]
        dist2_new = np.sum(diff_new ** 2, axis=2)
        sigma2_new = np.sum(P * dist2_new) / (Np * D)
        sigma2_new = max(sigma2_new, 1e-10)
        
        # Check convergence
        if np.linalg.norm(T_Y_new - T_Y) < tol:
            T_Y = T_Y_new
            break
        
        T_Y = T_Y_new
        sigma2 = sigma2_new
    
    return T_Y, P


# ============================================================
# Edge Length Constraint Optimization
# ============================================================

def apply_edge_length_constraints(keypoints, reference_edges, reference_lengths, 
                                   anchor_indices, n_iterations=50, 
                                   edge_weight=0.5, tolerance=0.1):
    """
    Apply edge length constraints to preserve wire topology.
    
    Uses iterative projection to enforce edge lengths while keeping anchor nodes fixed.
    
    Args:
        keypoints: N x 3 array of current keypoint positions
        reference_edges: List of (i, j) tuples defining edge connectivity
        reference_lengths: Array of reference edge lengths (same order as edges)
        anchor_indices: List of keypoint indices to keep fixed (e.g., branch nodes)
        n_iterations: Number of constraint iterations
        edge_weight: How strongly to enforce edge constraints (0-1)
        tolerance: Allowed deviation from reference length (fraction, e.g., 0.1 = ±10%)
    
    Returns:
        corrected_keypoints: N x 3 array with edge constraints applied
    """
    keypoints = keypoints.copy().astype(np.float64)
    n_keypoints = keypoints.shape[0]
    anchor_set = set(anchor_indices)
    
    for iteration in range(n_iterations):
        # Accumulate corrections for each keypoint
        corrections = np.zeros_like(keypoints)
        correction_counts = np.zeros(n_keypoints)
        
        for edge_idx, (i, j) in enumerate(reference_edges):
            if i >= n_keypoints or j >= n_keypoints:
                continue
                
            # Current edge vector and length
            edge_vec = keypoints[j] - keypoints[i]
            current_length = np.linalg.norm(edge_vec)
            
            if current_length < 1e-6:
                continue
            
            target_length = reference_lengths[edge_idx]
            
            # Check if within tolerance
            length_ratio = current_length / target_length
            if 1.0 - tolerance <= length_ratio <= 1.0 + tolerance:
                continue  # Already within tolerance
            
            # Compute correction to move toward target length
            length_diff = target_length - current_length
            correction_magnitude = length_diff * edge_weight * 0.5
            correction_dir = edge_vec / current_length
            
            # Apply correction (move both endpoints toward each other or apart)
            if i not in anchor_set:
                corrections[i] -= correction_dir * correction_magnitude
                correction_counts[i] += 1
            if j not in anchor_set:
                corrections[j] += correction_dir * correction_magnitude
                correction_counts[j] += 1
        
        # Average and apply corrections
        for k in range(n_keypoints):
            if correction_counts[k] > 0 and k not in anchor_set:
                keypoints[k] += corrections[k] / correction_counts[k]
    
    return keypoints


def correct_branch_leaf_nodes(cpd_keypoints, detected_branch, detected_leaf, 
                               n_branch, n_leaf, foreground_points):
    """
    Correct branch and leaf node positions using Hungarian matching.
    
    The first n_branch keypoints are branch nodes, next n_leaf are leaf nodes.
    Snap them to the nearest detected branch/leaf candidates.
    
    Args:
        cpd_keypoints: N x 3 CPD-deformed keypoints
        detected_branch: M1 x 3 detected branch node candidates
        detected_leaf: M2 x 3 detected leaf node candidates  
        n_branch: Number of branch nodes (first n_branch keypoints)
        n_leaf: Number of leaf nodes (next n_leaf keypoints)
        foreground_points: Dense foreground point cloud for fallback
    
    Returns:
        corrected_keypoints: N x 3 with branch/leaf nodes corrected
    """
    from scipy.spatial.distance import cdist
    
    corrected = cpd_keypoints.copy()
    
    # Correct branch nodes (indices 0 to n_branch-1)
    if n_branch > 0 and len(detected_branch) > 0:
        cpd_branch = cpd_keypoints[:n_branch]
        cost_matrix = cdist(cpd_branch, detected_branch)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            corrected[r] = detected_branch[c]
    elif n_branch > 0:
        # No detected branch nodes - snap to nearest foreground point
        from sklearn.neighbors import NearestNeighbors
        if len(foreground_points) > 0:
            nn = NearestNeighbors(n_neighbors=1).fit(foreground_points)
            _, indices = nn.kneighbors(cpd_keypoints[:n_branch])
            for r, idx in enumerate(indices.flatten()):
                corrected[r] = foreground_points[idx]
    
    # Correct leaf nodes (indices n_branch to n_branch+n_leaf-1)
    if n_leaf > 0 and len(detected_leaf) > 0:
        cpd_leaf = cpd_keypoints[n_branch:n_branch + n_leaf]
        cost_matrix = cdist(cpd_leaf, detected_leaf)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            corrected[n_branch + r] = detected_leaf[c]
    elif n_leaf > 0:
        # No detected leaf nodes - snap to nearest foreground point
        from sklearn.neighbors import NearestNeighbors
        if len(foreground_points) > 0:
            nn = NearestNeighbors(n_neighbors=1).fit(foreground_points)
            _, indices = nn.kneighbors(cpd_keypoints[n_branch:n_branch + n_leaf])
            for r, idx in enumerate(indices.flatten()):
                corrected[n_branch + r] = foreground_points[idx]
    
    return corrected


def project_keypoints_to_foreground(keypoints, foreground_points, max_distance=50.0):
    """
    Project keypoints to nearest foreground point cloud points.
    
    Args:
        keypoints: N x 3 keypoint positions
        foreground_points: M x 3 foreground point cloud
        max_distance: Maximum snap distance (mm)
    
    Returns:
        projected_keypoints: N x 3 projected positions
    """
    from sklearn.neighbors import NearestNeighbors
    
    if len(foreground_points) == 0 or len(keypoints) == 0:
        return keypoints.copy()
    
    nn = NearestNeighbors(n_neighbors=1).fit(foreground_points)
    distances, indices = nn.kneighbors(keypoints)
    
    projected = keypoints.copy()
    for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
        if dist < max_distance:
            projected[i] = foreground_points[idx]
    
    return projected


def create_overlay_frame(rgb_image, mask, alpha=0.4, mask_color=(255, 0, 0)):
    """Create an overlay of mask on RGB image."""
    overlay = rgb_image.copy().astype(np.float32)
    mask_bool = mask > 0
    
    for c in range(3):
        overlay[:, :, c][mask_bool] = (
            (1 - alpha) * overlay[:, :, c][mask_bool] + 
            alpha * mask_color[c]
        )
    
    return overlay.astype(np.uint8)


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


def main():
    print("Loading tracking data...")
    tracking_data = np.load("./data/full/tracking_BDLO_data.npy", allow_pickle=True).item()
    bg_data = np.load("./data/bg/tracking_BDLO_background_data.npy", allow_pickle=True).item()
    
    intrinsics = np.array([
        [606.1124267578125, 0, 641.7578125],
        [0, 605.8821411132812, 365.6518859863281],
        [0, 0, 1]
    ])
    
    output_dir = Path("./tracking_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_keys = sorted(tracking_data.keys())
    frame_keys = frame_keys[:225]  # Process only first 240 frames (8s)
    print(f"Found {len(frame_keys)} frames")
    print(f"Depth threshold: {DEPTH_THRESHOLD}mm")
    
    # Get background depth and apply threshold
    bg_depth = bg_data[0]['transformed_depth'].copy()
    bg_depth[bg_depth >= DEPTH_THRESHOLD] = 0
    
    arm_mask_dir = Path("/home/yehengz/deformable_seg/data/wire_tracking_arm_masks")

    # Step 1: get pc_mask and depth mask for the first frame
    first_data = tracking_data[frame_keys[0]]
    first_depth = first_data['transformed_depth'].copy()
    first_pc_mask = compute_point_cloud_mask(
        bg_depth,
        first_depth,
        intrinsics,
        distance_threshold=18
    )
    first_depth_mask = ((first_depth > 0) & (first_depth < DEPTH_THRESHOLD)).astype(np.uint8)
    # dual_arm_mask = (1 - pc_mask) * depth_mask for first frame
    dual_arm_mask = ((1 - (first_pc_mask > 0).astype(np.uint8)) * first_depth_mask).astype(np.uint8)
    
    # Create directories for saving arm masks and overlay visualizations
    arm_mask_save_dir = Path("/home/yehengz/deformable_seg/data/arm_traj2/masks")
    arm_mask_save_dir.mkdir(parents=True, exist_ok=True)
    arm_mask_viz_dir = arm_mask_save_dir / "masks_viz_overlay"
    arm_mask_viz_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # Keypoint extraction for every frame (following tracking_wire.py procedure)
    # ============================================================
    from viz_utils import create_color_point_cloud
    from repulsion_wires_utils import repulsion_relaxation_wire, compute_spacing_stats
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    def pixel_to_3d(pixel_coords, depth, intrinsics):
        """Convert 2D pixel coordinates to 3D points using depth."""
        if pixel_coords.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        points_3d = []
        for coord in pixel_coords:
            row, col = int(coord[0]), int(coord[1])
            if 0 <= row < depth.shape[0] and 0 <= col < depth.shape[1]:
                z = depth[row, col]
                if z > 0:
                    x = (col - cx) * z / fx
                    y = (row - cy) * z / fy
                    points_3d.append([x, y, z])
        return np.array(points_3d, dtype=np.float64) if points_3d else np.empty((0, 3), dtype=np.float64)

    def project_points_to_2d(points_3d, intrinsics):
        """Project 3D points to 2D pixel coordinates using camera intrinsics."""
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
        u = (x * fx) / z + cx
        v = (y * fy) / z + cy
        return np.stack([v, u], axis=1)  # (row, col)

    n_keypoints = 21
    expected_leaf_nodes = 4
    tracking_frames_dir = output_dir / "tracking_all_frames"
    tracking_frames_dir.mkdir(parents=True, exist_ok=True)
    tracking_video_frames = []
    all_keypoints_3d = []  # Store all frame keypoints
    all_edges = []  # Store all frame edges

    # For skeleton recovery across frames
    prev_pc_mask = None  # Store previous frame's pc_mask (before skeletonization)
    prev_branch_nodes_3d = None
    prev_end_nodes_3d = None
    NODE_JUMP_THRESHOLD = 60.0  # mm - threshold for detecting sharp 3D distance change
    MIN_NODES_REQUIRED = 6  # Minimum number of nodes required

    # ============================================================
    # Reference geometry for tracking (extracted from frame 0)
    # ============================================================
    reference_keypoints = None  # N x 3 reference keypoint positions
    reference_edges = None      # List of (i, j) edge tuples
    reference_lengths = None    # Array of reference edge lengths
    reference_n_branch = 0      # Number of branch nodes
    reference_n_leaf = 0        # Number of leaf nodes
    prev_keypoints_3d = None    # Previous frame's keypoints for CPD

    # CPD and constraint parameters (increased iterations for better accuracy)
    CPD_BETA = 10.0      # Motion coherence (larger = more rigid)
    CPD_LAMBDA = 2.0     # Regularization weight
    CPD_W = 0.1          # Outlier weight
    CPD_MAX_ITER = 100   # Increased from 50 for better convergence
    EDGE_CONSTRAINT_ITERATIONS = 100  # Increased from 50 for better edge preservation
    EDGE_WEIGHT = 0.5
    EDGE_TOLERANCE = 0.15  # Allow ±15% edge length deviation

    # Warm restart parameters
    consecutive_skips = 0
    MAX_SKIPS_BEFORE_RESTART = 3  # Trigger warm restart if more than 3 frames skipped
    last_valid_frame_idx = -1

    print("\n" + "=" * 60)
    print("GEOMETRY-CONSTRAINED KEYPOINT TRACKING")
    print("=" * 60)

    total_extraction_time = 0.0
    for i, frame_key in enumerate(frame_keys):
        frame_start = time.time()
        data = tracking_data[frame_key]
        rgb_image = data['color'][:, :, ::-1]  # BGR to RGB
        curr_depth = data['transformed_depth'].copy()

        # ============================================================
        # Step 1: Loading data for segmentation (already done above)
        # ============================================================
        if i == 0:
            print("Loading data for segmentation...")

        # ============================================================
        # Step 2: Creating segmentation masks
        # ============================================================
        if i == 0:
            print("Creating segmentation masks...")

        arm_mask_path = arm_mask_dir / f"mask_frame_{i:04d}.npy"
        if arm_mask_path.exists():
            arm_mask = np.load(str(arm_mask_path))
            # Dilate arm_mask by 3 pixels to make it wider
            kernel = np.ones((2, 2), np.uint8)  # 2x2 kernel for ~1 pixel dilation
            arm_mask = cv2.dilate((arm_mask > 0).astype(np.uint8), kernel, iterations=1)
            aug_arm_mask = ((arm_mask > 0) | (dual_arm_mask > 0)).astype(np.uint8)
        else:
            aug_arm_mask = dual_arm_mask.copy()
        
        # Save aug_arm_mask (dual_arm_mask) for this frame
        mask_save_path = arm_mask_save_dir / f"mask_frame_{i:04d}.npy"
        np.save(str(mask_save_path), aug_arm_mask)
        
        # Save overlay visualization
        overlay_viz = create_overlay_frame(rgb_image, aug_arm_mask, alpha=0.4, mask_color=(255, 0, 0))
        overlay_save_path = arm_mask_viz_dir / f"mask_frame_{i:04d}.png"
        cv2.imwrite(str(overlay_save_path), cv2.cvtColor(overlay_viz, cv2.COLOR_RGB2BGR))
        
        start = time.time()
        depth_mask = ((curr_depth > 0) & (curr_depth < DEPTH_THRESHOLD)).astype(np.uint8)
        pc_mask = ((1 - aug_arm_mask) * depth_mask).astype(np.uint8)
        pc_mask[0:150, :] = 0
        pc_mask_bg_subtraction = pc_mask.copy()  # Save for visualization

        # ============================================================
        # Step 3: Filtering point cloud mask using DBSCAN
        # ============================================================
        if i == 0:
            print("Filtering point cloud mask using DBSCAN...")

        pc_mask = filter_pcd_mask_dbscan(
            pc_mask,
            curr_depth,
            intrinsics,
            eps=30.0,
            min_samples=58,
        )
        pc_mask_after_dbscan = pc_mask.copy()  # Save for visualization

        if i == 0:
            print("point cloud masks created successfully")

        pc_mask = remove_small_components(pc_mask, min_size=100)
        pc_mask_after_remove = pc_mask.copy()  # Save for visualization
        end_mask = time.time()
        print(f"  Mask creation and filtering completed in {end_mask - start:.3f}s")

        # ============================================================
        # Skeletonization + Node identification (2D)
        # ============================================================
        start = time.time()
        skeleton_pc_mask = skelentonize(pc_mask)
        branch_nodes, end_nodes, adjacency, coords = node_identification(
            skeleton_pc_mask,
            return_graph=True,
        )

        if adjacency is not None and coords is not None:
            pruning_result = prune_leaf_segments(
                adjacency,
                coords,
                expected_num_leaf_nodes=expected_leaf_nodes,
            )
            branch_nodes = pruning_result["branch_coords"]
            end_nodes = pruning_result["leaf_coords"]
            skeleton_pc_mask = mask_from_mst(
                pruning_result["adjacency"],
                pruning_result["coords"],
                skeleton_pc_mask.shape,
            )
        end = time.time()
        print(f"  Skeletonization and node identification completed in {end - start:.3f}s")



        # ============================================================
        # Save 2x2 mask visualization
        # ============================================================
        mask_viz_dir = Path("/home/yehengz/deformable_seg/tracking_output/mask_viz")
        mask_viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 2x2 visualization
        H, W = pc_mask.shape
        viz_2x2 = np.zeros((H * 2, W * 2, 3), dtype=np.uint8)
        
        # Top-left: pc_mask after DBSCAN
        viz_2x2[:H, :W, :] = np.stack([(pc_mask_bg_subtraction > 0) * 255] * 3, axis=-1)
        
        # Top-right: pc_mask after remove_small_components
        viz_2x2[:H, W:, :] = np.stack([pc_mask_after_dbscan] * 3, axis=-1)
        
        # Bottom-left: skeletonized mask
        viz_2x2[H:, :W, :] = np.stack([pc_mask_after_remove] * 3, axis=-1)
        
        # Bottom-right: skeleton with nodes overlay
        skeleton_with_nodes = np.stack([(skeleton_pc_mask > 0).astype(np.uint8) * 255] * 3, axis=-1)
        # Draw branch nodes in gold (BGR: 0, 215, 255)
        for node in branch_nodes:
            row, col = int(node[0]), int(node[1])
            if 0 <= row < H and 0 <= col < W:
                cv2.circle(skeleton_with_nodes, (col, row), 5, (0, 215, 255), -1)
        # Draw leaf nodes in purple (BGR: 128, 0, 128)
        for node in end_nodes:
            row, col = int(node[0]), int(node[1])
            if 0 <= row < H and 0 <= col < W:
                cv2.circle(skeleton_with_nodes, (col, row), 5, (128, 0, 128), -1)
        viz_2x2[H:, W:, :] = skeleton_with_nodes
        
        # Add labels
        cv2.putText(viz_2x2, "BG Subtraction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(viz_2x2, "After DBSCAN", (W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(viz_2x2, "After Small Remove", (10, H + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(viz_2x2, f"Skeleton + Nodes (B:{len(branch_nodes)} L:{len(end_nodes)})", (W + 10, H + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imwrite(str(mask_viz_dir / f"frame_{i:04d}.png"), viz_2x2)

        # ============================================================
        # Check if skeleton is valid (skip frame if not)
        # ============================================================
        total_nodes = len(branch_nodes) + len(end_nodes)
        if total_nodes < MIN_NODES_REQUIRED:
            print(f"  Frame {i}: Skipping - insufficient nodes ({total_nodes} < {MIN_NODES_REQUIRED})")
            all_keypoints_3d.append(np.zeros((0, 3)))
            all_edges.append([])
            consecutive_skips += 1
            continue

        # Store current pc_mask for next frame
        prev_pc_mask = pc_mask.copy()

        # Convert nodes to 3D
        branch_nodes_3d = pixel_to_3d(branch_nodes, curr_depth, intrinsics)
        end_nodes_3d = pixel_to_3d(end_nodes, curr_depth, intrinsics)
        n_branch = branch_nodes_3d.shape[0]
        n_leaf = end_nodes_3d.shape[0]

        if i == 0:
            print(f"  Branch nodes: {branch_nodes.shape}, Leaf nodes: {end_nodes.shape}")


        # Store current 3D nodes for next frame's recovery check
        prev_branch_nodes_3d = branch_nodes_3d.copy() if branch_nodes_3d.shape[0] > 0 else None
        prev_end_nodes_3d = end_nodes_3d.copy() if end_nodes_3d.shape[0] > 0 else None

        # Create foreground point cloud from pc_mask
        points, colors, valid_mask = create_color_point_cloud(
            rgb_image,
            curr_depth,
            intrinsics,
            return_valid_mask=True,
        )
        valid_flat = valid_mask.reshape(-1)
        valid_indices = np.flatnonzero(valid_flat)
        pc_mask_flat = (skeleton_pc_mask > 0).reshape(-1)
        if valid_indices.size:
            foreground_mask_points = pc_mask_flat[valid_indices].astype(bool)
        else:
            foreground_mask_points = np.zeros(points.shape[0], dtype=bool)

        foreground_points = points[foreground_mask_points]
        foreground_colors = colors[foreground_mask_points]

        # # Downsample foreground points by factor of 4
        # if foreground_points.shape[0] > 4:
        #     foreground_points = foreground_points[::4]
        #     foreground_colors = foreground_colors[::4]

        if i == 0:
            print(f"Foreground points available for keypoint extraction: {foreground_points.shape[0]:,} (from {points.shape[0]:,} valid points)")

        if foreground_points.shape[0] < n_keypoints:
            print(f"  Frame {i}: Not enough foreground points ({foreground_points.shape[0]}), skipping")
            all_keypoints_3d.append(np.zeros((0, 3)))
            all_edges.append([])
            consecutive_skips += 1
            continue

        # ============================================================
        # Check if warm restart is needed (too many consecutive skips)
        # ============================================================
        need_warm_restart = (consecutive_skips > MAX_SKIPS_BEFORE_RESTART and 
                             reference_keypoints is not None and 
                             prev_keypoints_3d is not None)
        
        if need_warm_restart:
            print(f"  Frame {i}: WARM RESTART triggered ({consecutive_skips} consecutive skips)")

        # ============================================================
        # FRAME 0: Independent detection (FPS + Repulsion)
        # WARM RESTART: Re-detect and match to reference topology
        # FRAME N>0: CPD tracking with geometry constraints
        # ============================================================
        
        if i == 0 or reference_keypoints is None or prev_keypoints_3d is None or need_warm_restart:
            # ============================================================
            # FRAME 0: Independent frame processing - FPS with anchors + repulsion
            # ============================================================
            
            # Combine branch and end nodes as anchor points
            if branch_nodes_3d.size > 0 or end_nodes_3d.size > 0:
                anchor_points_3d = np.vstack([branch_nodes_3d, end_nodes_3d])
            else:
                anchor_points_3d = np.empty((0, 3))

            n_anchors = anchor_points_3d.shape[0]
            n_fps_additional = max(0, n_keypoints - n_anchors)

            print(f"Frame {i}: Independent detection mode")
            print(f"  Anchor points (branch + leaf): {n_anchors} (branch: {n_branch}, leaf: {n_leaf})")
            print(f"  Stage 1 (FPS): Initializing {n_keypoints} keypoints ({n_anchors} anchors + {n_fps_additional} FPS)...")

            fps_start = time.time()

            if n_anchors > 0 and foreground_points.shape[0] > 0:
                # Find closest points in foreground to each anchor
                nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
                nn.fit(foreground_points)
                _, anchor_indices = nn.kneighbors(anchor_points_3d)
                anchor_indices = anchor_indices.flatten()

                chosen = list(anchor_indices)
                chosen_set = set(chosen)

                distances = np.full(foreground_points.shape[0], np.inf)
                for idx in chosen:
                    new_distances = np.linalg.norm(foreground_points - foreground_points[idx], axis=1)
                    distances = np.minimum(distances, new_distances)

                for k in range(n_fps_additional):
                    masked_distances = distances.copy()
                    for idx in chosen_set:
                        masked_distances[idx] = -np.inf
                    next_idx = int(np.argmax(masked_distances))
                    chosen.append(next_idx)
                    chosen_set.add(next_idx)
                    new_distances = np.linalg.norm(foreground_points - foreground_points[next_idx], axis=1)
                    distances = np.minimum(distances, new_distances)

                chosen = np.array(chosen, dtype=np.int64)
            else:
                # No anchors, standard FPS
                centroid = np.mean(foreground_points, axis=0)
                dists_to_centroid = np.linalg.norm(foreground_points - centroid, axis=1)
                start_idx = int(np.argmax(dists_to_centroid))

                chosen = np.empty(n_keypoints, dtype=np.int64)
                chosen[0] = start_idx
                distances = np.linalg.norm(foreground_points - foreground_points[chosen[0]], axis=1)

                for k in range(1, n_keypoints):
                    next_idx = int(np.argmax(distances))
                    chosen[k] = next_idx
                    new_distances = np.linalg.norm(foreground_points - foreground_points[next_idx], axis=1)
                    distances = np.minimum(distances, new_distances)

            fps_keypoints = foreground_points[chosen]
            fps_time = time.time() - fps_start

            print(f"  FPS completed in {fps_time:.3f}s: {fps_keypoints.shape[0]} keypoints")

            # Fixed mask: branch and leaf nodes should not move
            fixed_mask = np.zeros(fps_keypoints.shape[0], dtype=bool)
            fixed_mask[:n_branch + n_leaf] = True

            print(f"  Fixed nodes (branch + leaf): {n_branch + n_leaf}")
            fps_stats = compute_spacing_stats(fps_keypoints, n_neighbors=1)
            print(f"  FPS spacing → min: {fps_stats['min_dist']:.2f}, max: {fps_stats['max_dist']:.2f}, uniformity: {fps_stats['uniformity']:.3f}")

            # Stage 2: Repulsion relaxation
            print("  Stage 2 (Repulsion): Relaxing keypoint positions...")

            relax_start = time.time()
            relaxation_result = repulsion_relaxation_wire(
                fps_keypoints,
                foreground_points,
                skeleton_pc_mask,
                intrinsics,
                fixed_mask=fixed_mask,
                n_iterations=40,
                learning_rate=5.0,
                k_neighbors=3,
                target_edge_length=None,
                epsilon=1e-8,
                project_each_step=True,
                rebuild_neighbors_every=20,
                return_debug=True,
            )
            keypoints = relaxation_result["keypoints"]
            degrees = relaxation_result["degrees"]
            edges = relaxation_result["edges"]
            relax_time = time.time() - relax_start
            print(f"  Repulsion completed in {relax_time:.3f}s")

            # Extract wire edges
            wire_edges = [(int(e[0]), int(e[1])) for e in edges]

            # ============================================================
            # Store reference geometry from frame 0
            # ============================================================
            if i == 0:
                reference_keypoints = keypoints.copy()
                reference_edges = wire_edges.copy()
                reference_n_branch = n_branch
                reference_n_leaf = n_leaf
                
                # Compute reference edge lengths
                reference_lengths = []
                for (ei, ej) in reference_edges:
                    if ei < keypoints.shape[0] and ej < keypoints.shape[0]:
                        length = np.linalg.norm(keypoints[ei] - keypoints[ej])
                        reference_lengths.append(length)
                    else:
                        reference_lengths.append(0.0)
                reference_lengths = np.array(reference_lengths)
                
                print(f"\n  === REFERENCE GEOMETRY STORED ===")
                print(f"  Reference keypoints: {reference_keypoints.shape}")
                print(f"  Reference edges: {len(reference_edges)}")
                print(f"  Reference edge lengths: min={reference_lengths.min():.2f}, max={reference_lengths.max():.2f}, mean={reference_lengths.mean():.2f}")
                print(f"  Branch nodes: {reference_n_branch}, Leaf nodes: {reference_n_leaf}")
            
            elif need_warm_restart:
                # ============================================================
                # WARM RESTART: Match newly detected keypoints to reference topology
                # ============================================================
                from scipy.spatial.distance import cdist
                
                print(f"  === WARM RESTART: Matching to reference topology ===")
                
                # Create DENSE wire point cloud (from pc_mask, not skeleton) for final projection
                # This ensures restart keypoints lie on the actual wire, not just the thin skeleton
                pc_mask_flat_dense = (pc_mask > 0).reshape(-1)
                if valid_indices.size:
                    dense_wire_mask = pc_mask_flat_dense[valid_indices].astype(bool)
                else:
                    dense_wire_mask = np.zeros(points.shape[0], dtype=bool)
                dense_wire_points = points[dense_wire_mask]
                print(f"  Dense wire point cloud: {dense_wire_points.shape[0]} points (vs skeleton: {foreground_points.shape[0]})")
                
                # We need to reorder the newly detected keypoints to match reference ordering
                # Reference ordering: [branch_0, branch_1, ..., leaf_0, leaf_1, ..., intermediate_0, ...]
                
                new_keypoints = keypoints.copy()
                matched_keypoints = np.zeros_like(reference_keypoints)
                
                n_ref_branch = reference_n_branch
                n_ref_leaf = reference_n_leaf
                n_ref_intermediate = reference_keypoints.shape[0] - n_ref_branch - n_ref_leaf
                
                # Current keypoints ordering from detection:
                # [branch_0..n_branch-1, leaf_0..n_leaf-1, intermediate_0..]
                n_curr_branch = n_branch
                n_curr_leaf = n_leaf
                n_curr_intermediate = new_keypoints.shape[0] - n_curr_branch - n_curr_leaf
                
                # Match branch nodes: new branch → reference branch
                if n_ref_branch > 0 and n_curr_branch > 0:
                    ref_branch = reference_keypoints[:n_ref_branch]
                    curr_branch = new_keypoints[:n_curr_branch]
                    cost = cdist(ref_branch, curr_branch)
                    row_ind, col_ind = linear_sum_assignment(cost)
                    for r, c in zip(row_ind, col_ind):
                        matched_keypoints[r] = curr_branch[c]
                    # Fill unmatched reference branch nodes with nearest current branch
                    for r in range(n_ref_branch):
                        if r not in row_ind:
                            dists = np.linalg.norm(curr_branch - ref_branch[r], axis=1)
                            matched_keypoints[r] = curr_branch[np.argmin(dists)]
                
                # Match leaf nodes: new leaf → reference leaf
                if n_ref_leaf > 0 and n_curr_leaf > 0:
                    ref_leaf = reference_keypoints[n_ref_branch:n_ref_branch + n_ref_leaf]
                    curr_leaf = new_keypoints[n_curr_branch:n_curr_branch + n_curr_leaf]
                    cost = cdist(ref_leaf, curr_leaf)
                    row_ind, col_ind = linear_sum_assignment(cost)
                    for r, c in zip(row_ind, col_ind):
                        matched_keypoints[n_ref_branch + r] = curr_leaf[c]
                    # Fill unmatched
                    for r in range(n_ref_leaf):
                        if r not in row_ind:
                            dists = np.linalg.norm(curr_leaf - ref_leaf[r], axis=1)
                            matched_keypoints[n_ref_branch + r] = curr_leaf[np.argmin(dists)]
                
                # Match intermediate nodes: new intermediate → reference intermediate
                if n_ref_intermediate > 0 and n_curr_intermediate > 0:
                    ref_inter = reference_keypoints[n_ref_branch + n_ref_leaf:]
                    curr_inter = new_keypoints[n_curr_branch + n_curr_leaf:]
                    cost = cdist(ref_inter, curr_inter)
                    row_ind, col_ind = linear_sum_assignment(cost)
                    for r, c in zip(row_ind, col_ind):
                        matched_keypoints[n_ref_branch + n_ref_leaf + r] = curr_inter[c]
                    # Fill unmatched
                    for r in range(n_ref_intermediate):
                        if r not in row_ind:
                            dists = np.linalg.norm(curr_inter - ref_inter[r], axis=1)
                            matched_keypoints[n_ref_branch + n_ref_leaf + r] = curr_inter[np.argmin(dists)]
                
                # Apply edge length constraints to enforce reference topology
                anchor_indices = list(range(reference_n_branch))  # Fix branch nodes
                matched_keypoints = apply_edge_length_constraints(
                    matched_keypoints, reference_edges, reference_lengths,
                    anchor_indices=anchor_indices,
                    n_iterations=EDGE_CONSTRAINT_ITERATIONS,
                    edge_weight=EDGE_WEIGHT,
                    tolerance=EDGE_TOLERANCE
                )
                
                # FORCE all keypoints onto the DENSE wire point cloud (not skeleton)
                # This ensures every restart keypoint lies exactly on the wire
                from sklearn.neighbors import NearestNeighbors
                if len(dense_wire_points) > 0:
                    nn = NearestNeighbors(n_neighbors=1).fit(dense_wire_points)
                    distances, indices = nn.kneighbors(matched_keypoints)
                    for k in range(len(matched_keypoints)):
                        matched_keypoints[k] = dense_wire_points[indices[k, 0]]
                    
                    # Print debug info about projection distances
                    print(f"  Projection distances to dense wire: min={distances.min():.2f}, max={distances.max():.2f}, mean={distances.mean():.2f}")
                
                keypoints = matched_keypoints
                wire_edges = reference_edges  # Use reference topology
                
                print(f"  Matched {keypoints.shape[0]} keypoints to reference topology (all on dense wire point cloud)")
                
                # Compute edge length error after matching
                current_lengths = []
                for (ei, ej) in wire_edges:
                    if ei < keypoints.shape[0] and ej < keypoints.shape[0]:
                        length = np.linalg.norm(keypoints[ei] - keypoints[ej])
                        current_lengths.append(length)
                current_lengths = np.array(current_lengths) if current_lengths else np.array([0])
                length_errors = np.abs(current_lengths - reference_lengths) / (reference_lengths + 1e-6)
                print(f"  Edge length error after restart: mean={length_errors.mean()*100:.1f}%, max={length_errors.max()*100:.1f}%")
        
        else:
            # ============================================================
            # FRAME N > 0: CPD Tracking with Geometry Constraints
            # ============================================================
            tracking_start = time.time()
            
            print(f"Frame {i}: CPD tracking mode")
            
            # Create DENSE wire point cloud for projection (from pc_mask, not skeleton)
            pc_mask_flat_dense = (pc_mask > 0).reshape(-1)
            if valid_indices.size:
                dense_wire_mask = pc_mask_flat_dense[valid_indices].astype(bool)
            else:
                dense_wire_mask = np.zeros(points.shape[0], dtype=bool)
            dense_wire_points = points[dense_wire_mask]
            
            # Step 1: CPD - Deform previous keypoints toward current DENSE wire
            cpd_start = time.time()
            
            # Downsample dense wire for CPD (too many points slows it down)
            if dense_wire_points.shape[0] > 500:
                downsample_indices = np.random.choice(dense_wire_points.shape[0], 500, replace=False)
                cpd_target = dense_wire_points[downsample_indices]
            else:
                cpd_target = dense_wire_points
            
            cpd_keypoints, _ = cpd_register(
                prev_keypoints_3d, cpd_target,
                beta=CPD_BETA, lmbda=CPD_LAMBDA, w=CPD_W, max_iter=CPD_MAX_ITER
            )
            cpd_time = time.time() - cpd_start
            
            # Step 2: Correct branch/leaf nodes using Hungarian matching
            correct_start = time.time()
            corrected_keypoints = correct_branch_leaf_nodes(
                cpd_keypoints, branch_nodes_3d, end_nodes_3d,
                reference_n_branch, reference_n_leaf, dense_wire_points
            )
            correct_time = time.time() - correct_start
            
            # Step 3: Apply edge length constraints
            constraint_start = time.time()
            
            # Anchor indices: branch nodes (most stable)
            anchor_indices = list(range(reference_n_branch))  # Fix branch nodes
            
            keypoints = apply_edge_length_constraints(
                corrected_keypoints, reference_edges, reference_lengths,
                anchor_indices=anchor_indices,
                n_iterations=EDGE_CONSTRAINT_ITERATIONS,
                edge_weight=EDGE_WEIGHT,
                tolerance=EDGE_TOLERANCE
            )
            constraint_time = time.time() - constraint_start
            
            # Step 4: FORCE all keypoints onto dense wire point cloud
            from sklearn.neighbors import NearestNeighbors
            if len(dense_wire_points) > 0:
                nn = NearestNeighbors(n_neighbors=1).fit(dense_wire_points)
                distances, indices = nn.kneighbors(keypoints)
                for k in range(len(keypoints)):
                    keypoints[k] = dense_wire_points[indices[k, 0]]
            
            tracking_time = time.time() - tracking_start
            
            # Use reference edges (topology is preserved)
            wire_edges = reference_edges
            
            # Compute edge length stats for this frame
            current_lengths = []
            for (ei, ej) in wire_edges:
                if ei < keypoints.shape[0] and ej < keypoints.shape[0]:
                    length = np.linalg.norm(keypoints[ei] - keypoints[ej])
                    current_lengths.append(length)
            current_lengths = np.array(current_lengths) if current_lengths else np.array([0])
            
            length_errors = np.abs(current_lengths - reference_lengths) / (reference_lengths + 1e-6)
            
            print(f"  CPD: {cpd_time:.3f}s, Correct: {correct_time:.3f}s, Constraint: {constraint_time:.3f}s")
            print(f"  Edge length error: mean={length_errors.mean()*100:.1f}%, max={length_errors.max()*100:.1f}%")

        # Update previous keypoints for next frame
        prev_keypoints_3d = keypoints.copy()
        
        # Reset skip counter after successful processing
        consecutive_skips = 0
        last_valid_frame_idx = i

        frame_time = time.time() - frame_start
        total_extraction_time += frame_time

        print(f"  Frame {i} total time: {frame_time:.3f}s")

        all_keypoints_3d.append(keypoints)
        all_edges.append(wire_edges)

        # Project keypoints to 2D
        if keypoints.shape[0] > 0:
            keypoints_2d = project_points_to_2d(keypoints, intrinsics)
            keypoints_2d_int = np.round(keypoints_2d).astype(int)
        else:
            keypoints_2d_int = np.zeros((0, 2), dtype=int)

        # Overlay keypoints and edges on the image
        fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
        ax.imshow(rgb_image)
        # Draw edges
        for (ei, ej) in wire_edges:
            if ei < keypoints_2d_int.shape[0] and ej < keypoints_2d_int.shape[0]:
                pt1 = keypoints_2d_int[ei]
                pt2 = keypoints_2d_int[ej]
                ax.plot([pt1[1], pt2[1]], [pt1[0], pt2[0]], color='red', linewidth=2)
        # Draw keypoints
        for (row, col) in keypoints_2d_int:
            circ = Circle((col, row), radius=5, color='red', fill=True, alpha=1.0)
            ax.add_patch(circ)
        ax.axis('off')
        plt.tight_layout()
        frame_path = tracking_frames_dir / f"frame_{i:04d}.png"
        plt.savefig(str(frame_path), dpi=100)
        plt.close(fig)

        # Read saved frame for video
        frame_img = cv2.imread(str(frame_path))
        frame_img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        tracking_video_frames.append(frame_img_rgb)

        if (i + 1) % 50 == 0:
            print(f"  Keypoint extraction: {i + 1}/{len(frame_keys)} frames, total time so far: {total_extraction_time:.2f}s")


    print(f"\nTotal extraction time for all frames: {total_extraction_time:.3f}s, avg per frame: {total_extraction_time/len(frame_keys):.4f}s")
    print(f"\nCreating tracking.mp4 with {len(tracking_video_frames)} frames...")
    create_video_from_frames(tracking_video_frames, output_dir / "tracking.mp4", fps=30)
    print(f"Saved tracking frames to {tracking_frames_dir}")

    # Save keypoints data
    keypoints_save_path = output_dir / "all_keypoints_3d.npy"
    np.save(str(keypoints_save_path), {"keypoints": all_keypoints_3d, "edges": all_edges}, allow_pickle=True)
    print(f"Saved keypoints data to {keypoints_save_path}")

    # ============================================================
    # Compute edge length statistics using frame 0's topology
    # ============================================================
    print("\n" + "=" * 60)
    print("EDGE LENGTH STATISTICS (Using Frame 0 Topology)")
    print("=" * 60)

    # Get frame 0's edges as the reference topology
    if len(all_edges) > 0 and len(all_edges[0]) > 0:
        reference_edges = all_edges[0]  # List of (i, j) tuples
        n_edges = len(reference_edges)
        print(f"Reference topology from frame 0: {n_edges} edges")

        # Collect edge lengths for each edge across all valid frames
        edge_lengths_per_edge = {edge_idx: [] for edge_idx in range(n_edges)}
        valid_frame_count = 0

        for frame_idx, (keypoints, edges) in enumerate(zip(all_keypoints_3d, all_edges)):
            # Only use frames with valid keypoints (same number as frame 0)
            if keypoints.shape[0] == all_keypoints_3d[0].shape[0] and keypoints.shape[0] > 0:
                valid_frame_count += 1
                for edge_idx, (i, j) in enumerate(reference_edges):
                    if i < keypoints.shape[0] and j < keypoints.shape[0]:
                        edge_length = np.linalg.norm(keypoints[i] - keypoints[j])
                        edge_lengths_per_edge[edge_idx].append(edge_length)

        print(f"Valid frames with matching topology: {valid_frame_count}/{len(all_keypoints_3d)}")

        # Prepare data for boxplot
        edge_data = []
        edge_labels = []
        for edge_idx in range(n_edges):
            lengths = edge_lengths_per_edge[edge_idx]
            if len(lengths) > 0:
                edge_data.append(lengths)
                i, j = reference_edges[edge_idx]
                edge_labels.append(f"{i}-{j}")

        if len(edge_data) > 0:
            # Create boxplot
            fig, ax = plt.subplots(figsize=(max(12, n_edges * 0.5), 6))
            bp = ax.boxplot(edge_data, labels=edge_labels, patch_artist=True)
            
            # Color the boxes
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            
            ax.set_xlabel('Edge (keypoint i - keypoint j)')
            ax.set_ylabel('Edge Length (mm)')
            ax.set_title(f'Edge Length Distribution Across {valid_frame_count} Frames')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            
            boxplot_path = output_dir / "edge_lengths_boxplot.png"
            plt.savefig(str(boxplot_path), dpi=150)
            plt.close()
            print(f"Saved edge length boxplot to {boxplot_path}")

            # Print summary statistics
            print("\nEdge Length Statistics (mm):")
            print(f"{'Edge':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
            print("-" * 50)
            all_lengths = []
            for edge_idx in range(n_edges):
                lengths = edge_lengths_per_edge[edge_idx]
                if len(lengths) > 0:
                    all_lengths.extend(lengths)
                    i, j = reference_edges[edge_idx]
                    print(f"{i}-{j:<7} {np.mean(lengths):<10.2f} {np.std(lengths):<10.2f} {np.min(lengths):<10.2f} {np.max(lengths):<10.2f}")
            
            print("-" * 50)
            print(f"{'Overall':<10} {np.mean(all_lengths):<10.2f} {np.std(all_lengths):<10.2f} {np.min(all_lengths):<10.2f} {np.max(all_lengths):<10.2f}")
    else:
        print("No valid edges found in frame 0")

    print("Done!")


if __name__ == "__main__":
    main()
