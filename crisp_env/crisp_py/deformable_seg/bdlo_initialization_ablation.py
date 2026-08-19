#!/usr/bin/env python3
"""BDLO initialization-only ablation.

Protocol (per clip):
- Clip length: 300 frames
- Use first 31 frames only
- Frame 0 initialization is the reference
- For frames 1..30, run initialization from scratch for each ablation method
- Compute metrics against frame-0 reference topology/lengths
- Save per-frame ablation images (no tracking videos)
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors

from wire_tracker import WireTracker
from wire_initializer_combined import WireInitializer
from bdlo1_batch_experiment import (
    load_chunk_data,
    load_transforms,
    get_ee_positions_cam,
    compute_edge_metrics,
    compute_position_metrics,
    sample_points_on_edges,
    compute_chamfer_metrics,
    create_method_panel,
    create_ablation_grid,
)


class AblationInitializer(WireInitializer):
    """WireInitializer with ablation flags for projection and edge repulsion."""

    def __init__(
        self,
        *args,
        use_projection=True,
        use_edge_repulsion=True,
        reference_edge_lengths=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_projection = use_projection
        self.use_edge_repulsion = use_edge_repulsion
        self.reference_edge_lengths = reference_edge_lengths  # Dict: (i,j) -> length from frame 0

    def _repulsion_relaxation_with_topology(
        self, 
        keypoints: np.ndarray, 
        target_points: np.ndarray, 
        fixed_mask: np.ndarray,
        edges: List[Tuple[int, int]],
        segment_edges: List[List[Tuple[int, int]]] = None,
        segment_lengths: List[float] = None,
    ) -> np.ndarray:
        """
        Gauss-Seidel relaxation with ablation support for projection and edge repulsion.
        Uses reference_edge_lengths from frame 0 if available, otherwise falls back to segment_lengths.
        """
        keypoints = keypoints.copy().astype(np.float64)
        K = keypoints.shape[0]
        epsilon = 1e-8

        anchor_positions_before = keypoints[fixed_mask].copy()
        n_anchors = np.sum(fixed_mask)
        print(f"  Anchors: {n_anchors} keypoints (indices: {np.where(fixed_mask)[0].tolist()})")

        if K <= 1 or len(target_points) == 0:
            return keypoints

        # Build NN index for projection (if enabled)
        cloud_nn = None
        if self.use_projection:
            cloud_nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
            cloud_nn.fit(target_points)

        # Compute per-edge target lengths
        # Priority: 1) reference_edge_lengths from frame 0, 2) segment_lengths from skeleton
        edge_to_target: Dict[Tuple[int, int], float] = {}

        if self.reference_edge_lengths is not None and len(self.reference_edge_lengths) > 0:
            # Use reference edge lengths from frame 0
            edge_to_target = self.reference_edge_lengths.copy()
            print(f"  Using reference edge lengths from frame 0 ({len(edge_to_target)//2} edges)")
        elif segment_edges is not None and segment_lengths is not None:
            # Fallback to segment-based lengths
            for seg_idx, seg_edge_list in enumerate(segment_edges):
                if seg_idx < len(segment_lengths) and len(seg_edge_list) > 0:
                    seg_len = segment_lengths[seg_idx]
                    n_edges = len(seg_edge_list)
                    target_edge_len = seg_len / n_edges

                    for edge in seg_edge_list:
                        i, j = edge
                        edge_to_target[(i, j)] = target_edge_len
                        edge_to_target[(j, i)] = target_edge_len

            print(f"  Per-segment target edge lengths (fallback):")
            for seg_idx, seg_edge_list in enumerate(segment_edges):
                if seg_idx < len(segment_lengths) and len(seg_edge_list) > 0:
                    seg_len = segment_lengths[seg_idx]
                    n_edges = len(seg_edge_list)
                    print(f"    Segment {seg_idx}: {seg_len:.1f}mm / {n_edges} edges = {seg_len/n_edges:.1f}mm per edge")

        use_global_target = len(edge_to_target) == 0
        if use_global_target:
            all_lens = [np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in edges]
            global_tgt = np.mean(all_lens) if all_lens else 50.0
            print(f"  Using global target edge length: {global_tgt:.1f}mm")

        if segment_edges is not None and len(segment_edges) > 0:
            edge_order = [(i, j) for seg in segment_edges for (i, j) in seg]
        else:
            edge_order = edges

        print(f"  Gauss-Seidel relaxation: {self.repulsion_iterations} iterations, lr={self.repulsion_lr}")
        print(f"    use_projection={self.use_projection}, use_edge_repulsion={self.use_edge_repulsion}")

        def compute_edge_error():
            errors = []
            for (i, j) in edge_order:
                d = np.linalg.norm(keypoints[j] - keypoints[i])
                if use_global_target:
                    tgt = global_tgt
                else:
                    tgt = edge_to_target.get((i, j), edge_to_target.get((j, i), 50.0))
                errors.append(abs(d - tgt))
            return np.mean(errors), np.max(errors)

        # NoEdge special case: just project to surface and done (no iteration needed)
        if not self.use_edge_repulsion:
            print(f"  NoEdge mode: skipping edge optimization, just projecting to surface")
            if self.use_projection and cloud_nn is not None:
                _, proj_indices = cloud_nn.kneighbors(keypoints)
                for k in range(K):
                    if not fixed_mask[k]:
                        keypoints[k] = target_points[proj_indices[k, 0]]
                print(f"    Projected all free keypoints to surface")
            mean_err, max_err = compute_edge_error()
            print(f"    Final: edge_err={mean_err:.2f}mm, max={max_err:.2f}mm, surf_rmse=0.00mm")
            return keypoints

        for iteration in range(self.repulsion_iterations):

            # Edge corrections (only if enabled)
            if self.use_edge_repulsion:
                for (i, j) in edge_order:
                    i_free = not fixed_mask[i]
                    j_free = not fixed_mask[j]

                    if not i_free and not j_free:
                        continue

                    v = keypoints[j] - keypoints[i]
                    d = np.linalg.norm(v)

                    if d < epsilon:
                        continue

                    unit_v = v / d

                    if use_global_target:
                        tgt = global_tgt
                    else:
                        tgt = edge_to_target.get((i, j), edge_to_target.get((j, i), 50.0))

                    force_mag = (tgt - d) / tgt

                    if i_free and j_free:
                        weight_i, weight_j = 0.5, 0.5
                    elif i_free:
                        weight_i, weight_j = 1.0, 0.0
                    else:
                        weight_i, weight_j = 0.0, 1.0

                    correction = force_mag * self.repulsion_lr * unit_v
                    if i_free:
                        keypoints[i] -= correction * weight_i
                    if j_free:
                        keypoints[j] += correction * weight_j

            # Projection (only if enabled) - every 10 iterations with stronger blending
            if self.use_projection and cloud_nn is not None and (iteration + 1) % 10 == 0:
                _, target_indices = cloud_nn.kneighbors(keypoints)
                proj_strength = 0.5  # Strong blend toward surface to keep keypoints on cloud
                for k in range(K):
                    if not fixed_mask[k]:
                        cloud_pt = target_points[target_indices[k, 0]]
                        keypoints[k] = (1 - proj_strength) * keypoints[k] + proj_strength * cloud_pt

            if iteration == 0 or iteration == self.repulsion_iterations - 1 or (iteration + 1) % 50 == 0:
                mean_err, max_err = compute_edge_error()
                if cloud_nn is not None:
                    _, tmp_indices = cloud_nn.kneighbors(keypoints)
                    tmp_dists = np.linalg.norm(keypoints - target_points[tmp_indices[:, 0]], axis=1)
                    surf_rmse = np.sqrt(np.mean(tmp_dists[~fixed_mask] ** 2)) if np.sum(~fixed_mask) > 0 else 0.0
                else:
                    surf_rmse = 0.0
                print(f"    Iter {iteration+1:4d}: edge_err={mean_err:.2f}mm (max={max_err:.2f}), surf_rmse={surf_rmse:.2f}mm")

        # Final hard projection: snap all free keypoints to nearest cloud point
        if self.use_projection and cloud_nn is not None:
            _, final_proj_indices = cloud_nn.kneighbors(keypoints)
            for k in range(K):
                if not fixed_mask[k]:
                    keypoints[k] = target_points[final_proj_indices[k, 0]]
            print(f"    Final projection: snapped all free keypoints to surface")

        mean_err_final, max_err_final = compute_edge_error()
        if cloud_nn is not None:
            _, final_indices = cloud_nn.kneighbors(keypoints)
            final_dists = np.linalg.norm(keypoints - target_points[final_indices[:, 0]], axis=1)
            final_surf_rmse = np.sqrt(np.mean(final_dists[~fixed_mask] ** 2)) if np.sum(~fixed_mask) > 0 else 0.0
        else:
            final_surf_rmse = 0.0
        print(f"    Final: edge_err={mean_err_final:.2f}mm, max={max_err_final:.2f}mm, surf_rmse={final_surf_rmse:.2f}mm")

        anchor_positions_after = keypoints[fixed_mask]
        anchor_drift = np.linalg.norm(anchor_positions_after - anchor_positions_before, axis=1)
        max_anchor_drift = np.max(anchor_drift) if len(anchor_drift) > 0 else 0.0
        if max_anchor_drift > 1e-6:
            print(f"  WARNING: Anchors moved! Max drift = {max_anchor_drift:.6f} mm")
        else:
            print(f"  Anchors verified: no drift (max={max_anchor_drift:.2e})")

        if cloud_nn is not None:
            dists, _ = cloud_nn.kneighbors(keypoints)
            free_dists = dists[~fixed_mask, 0]
            surface_rmse = np.sqrt(np.mean(free_dists ** 2)) if len(free_dists) > 0 else 0.0
        else:
            surface_rmse = 0.0
        print(f"  Surface RMSE (free keypoints to cloud): {surface_rmse:.3f} mm")

        return keypoints


class InitAblationTracker(WireTracker):
    """WireTracker variant that uses AblationInitializer for ablation experiments."""

    def __init__(
        self,
        *args,
        use_branch_leaf_anchors=True,
        use_projection=True,
        use_edge_repulsion=True,
        reference_edge_lengths=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_branch_leaf_anchors = use_branch_leaf_anchors
        self.use_projection = use_projection
        self.use_edge_repulsion = use_edge_repulsion
        self.reference_edge_lengths = reference_edge_lengths  # Dict: (i,j) -> length from frame 0

    def _create_initializer(self):
        """Create AblationInitializer with ablation flags."""
        return AblationInitializer(
            intrinsics=self.intrinsics,
            n_keypoints=self.n_keypoints,
            target_branch_nodes=self.target_branch_nodes,
            target_leaf_nodes=self.target_leaf_nodes,
            max_depth=self.max_depth,
            repulsion_iterations=self.repulsion_iterations,
            repulsion_lr=self.repulsion_lr,
            ee_poses_3d=self.ee_poses_3d,
            use_projection=self.use_projection,
            use_edge_repulsion=self.use_edge_repulsion,
            reference_edge_lengths=self.reference_edge_lengths,
        )

    def _initialize_with_segment_allocation(self, skeleton_mask, depth):
        """Override to use AblationInitializer and apply anchor ablation."""
        import time

        timing = {}
        total_start = time.time()

        # Extract skeleton point cloud
        skeleton_pc = self._extract_point_cloud(skeleton_mask, depth)

        # Step 1: Node identification (same as parent)
        t0 = time.time()
        branch_2d, leaf_2d, adjacency, coords = self._node_identification(skeleton_mask)
        timing['node_detection'] = time.time() - t0

        # Step 2: Topology pruning
        t0 = time.time()
        if adjacency is not None:
            pruned = self._prune_to_target_topology(adjacency, coords)
            branch_2d = pruned["branch_coords"]
            leaf_2d = pruned["leaf_coords"]
        timing['pruning'] = time.time() - t0

        # Step 3: 2D → 3D conversion
        t0 = time.time()
        branch_3d = self._pixel_to_3d(branch_2d, depth)
        leaf_3d = self._pixel_to_3d(leaf_2d, depth)
        n_branch = len(branch_3d)
        n_leaf = len(leaf_3d)
        timing['2d_to_3d'] = time.time() - t0

        self.reference_n_branch = n_branch
        self.reference_n_leaf = n_leaf

        # Step 4: Create AblationInitializer
        t0 = time.time()
        initializer = self._create_initializer()

        # Step 5: EE mapping
        ee_to_leaf_kp = initializer._establish_ee_mapping_from_leaves(leaf_3d, n_branch)

        if self.ee_poses_3d is not None and ee_to_leaf_kp:
            initializer.ee_to_leaf_mapping = ee_to_leaf_kp
            self.ee_to_leaf_mapping = ee_to_leaf_kp
            ee_positions = self.ee_poses_3d[0]
            for ee_idx, kp_idx in ee_to_leaf_kp.items():
                leaf_local_idx = kp_idx - n_branch
                old_pos = leaf_3d[leaf_local_idx].copy()
                leaf_3d[leaf_local_idx] = ee_positions[ee_idx]
                dist = np.linalg.norm(old_pos - ee_positions[ee_idx])
                print(f"  EE {ee_idx} -> Leaf {kp_idx} (shift={dist:.1f} mm)")
        else:
            initializer.ee_to_leaf_mapping = ee_to_leaf_kp if ee_to_leaf_kp else None
            self.ee_to_leaf_mapping = ee_to_leaf_kp
        timing['ee_mapping'] = time.time() - t0

        leaf_2d_updated = leaf_2d.copy().astype(np.float64)
        if self.ee_poses_3d is not None and ee_to_leaf_kp:
            for ee_idx, kp_idx in ee_to_leaf_kp.items():
                leaf_local_idx = kp_idx - n_branch
                ee_2d = self._project_3d_to_2d(self.ee_poses_3d[0][ee_idx:ee_idx+1])
                leaf_2d_updated[leaf_local_idx] = ee_2d[0]

        # Step 6: Build topology
        t0 = time.time()
        try:
            keypoints, edges, segment_edges, ordered_segments = initializer._build_topology(
                branch_3d, leaf_3d, branch_2d, leaf_2d_updated,
                skeleton_mask, depth, ee_to_leaf_kp,
                n_keypoints_per_segment=self.keypoints_per_segment,
            )
        except ValueError as e:
            return {'success': False, 'reason': str(e)}
        timing['build_topology'] = time.time() - t0
        
        # Save initial keypoints BEFORE relaxation for ablation comparison
        initial_keypoints = keypoints.copy()

        self.reference_edges = edges
        self.reference_lengths = np.array([
            np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in edges
        ])
        self.segment_edges = segment_edges
        self.anchor_set = set(range(n_branch + n_leaf))

        self.free_leaf_indices = []
        for seg in ordered_segments:
            if seg['type'] == 'free_leaf':
                self.free_leaf_indices.append(seg['start_kp'])

        # Step 7: Repulsion refinement with ablation
        t0 = time.time()
        skeleton_segment_lengths = [seg.get('estimated_length', 0) for seg in ordered_segments]
        
        # Save segment lengths for ablation
        self.skeleton_segment_lengths = skeleton_segment_lengths

        # Build fixed_mask with anchor ablation
        fixed_mask = np.zeros(len(keypoints), dtype=bool)
        if self.use_branch_leaf_anchors:
            # Full/NoProj/NoEdge: all branch+leaf nodes are anchored
            fixed_mask[:n_branch + n_leaf] = True
        else:
            # NoAnchor: ONLY EE-attached leaf nodes are anchored
            # Branch nodes and non-EE leaf nodes are FREE
            if ee_to_leaf_kp:
                for kp_idx in ee_to_leaf_kp.values():
                    fixed_mask[kp_idx] = True

        keypoints = initializer._repulsion_relaxation_with_topology(
            keypoints, skeleton_pc, fixed_mask, edges, segment_edges,
            segment_lengths=skeleton_segment_lengths,
        )
        timing['repulsion'] = time.time() - t0

        self.reference_keypoints = keypoints.copy()
        self.reference_lengths = np.array([
            np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in edges
        ])

        self.prev_keypoints = keypoints.copy()
        self.is_initialized = True
        self.consecutive_skips = 0

        keypoints_2d = self._project_3d_to_2d(keypoints)

        timing['total'] = time.time() - total_start

        seg_lengths = []
        for seg in segment_edges:
            seg_len = sum(np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in seg)
            seg_lengths.append(seg_len)
        print(f"  Segment lengths: {[f'{l:.1f}' for l in seg_lengths]} mm")
        print(f"  Total wire length: {sum(seg_lengths):.1f} mm")
        print(f"  keypoints_per_segment used: {self.keypoints_per_segment}")

        return {
            'success': True,
            'keypoints': keypoints,
            'initial_keypoints': initial_keypoints,  # Pre-relaxation positions
            'keypoints_2d': keypoints_2d,
            'edges': edges,
            'segment_edges': segment_edges,
            'skeleton_segment_lengths': skeleton_segment_lengths,
            'n_branch': n_branch,
            'n_leaf': n_leaf,
            'ee_to_leaf_kp': ee_to_leaf_kp,
            'skeleton_mask': skeleton_mask,
            'skeleton_pc': skeleton_pc,
            'timing': timing,
        }


def build_init_tracker(intrinsics, n_keypoints, ee_pose_single, keypoints_per_segment, method_name,
                       reference_edge_lengths=None):
    """Build tracker with optional reference edge lengths from frame 0.
    
    Args:
        reference_edge_lengths: Dict mapping (i,j) -> target length in mm from frame 0.
                               If provided, used as target in relaxation instead of skeleton length.
    """
    ee_poses = np.array([ee_pose_single], dtype=np.float32)

    base_params = {
        'intrinsics': intrinsics,
        'n_keypoints': n_keypoints,
        'target_branch_nodes': 2,
        'target_leaf_nodes': 4,
        'bg_threshold': 80.0,
        'max_depth': 2000.0,
        'top_k_components': 1,
        'arm_dilation_pixels': 5,
        'enable_cpd': False,
        'enable_node_matching': True,
        'enable_geometry_constraint': False,
        'enable_ee_injection': True,
        'n_outer_iterations': 1,
        'n_edge_iterations': 1,
        'edge_weight': 0.5,
        'edge_tolerance': 0.02,
        'repulsion_iterations': 200,
        'repulsion_lr': 10.0,
        'repulsion_k_neighbors': 3,
        'keypoints_per_segment': keypoints_per_segment,
        'ee_poses_3d': ee_poses,
        'reference_edge_lengths': reference_edge_lengths,
    }

    flags = {
        'Full': dict(use_branch_leaf_anchors=True, use_projection=True, use_edge_repulsion=True),
        'NoProj': dict(use_branch_leaf_anchors=True, use_projection=False, use_edge_repulsion=True),
        'NoEdge': dict(use_branch_leaf_anchors=True, use_projection=True, use_edge_repulsion=False),
        'NoAnchor': dict(use_branch_leaf_anchors=False, use_projection=True, use_edge_repulsion=True),
    }

    return InitAblationTracker(**base_params, **flags[method_name])


def run_initialize_once(tracker, data, frame_idx):
    depth = data['depth'][frame_idx].astype(np.float32)
    rgb = cv2.cvtColor(data['color'][frame_idx], cv2.COLOR_BGR2RGB)
    dlo_mask = data['dlo_masks'][frame_idx]
    exclude_mask = (1 - dlo_mask).astype(np.uint8)

    return tracker.process_frame(
        depth=depth,
        arm_depth=None,
        rgb=rgb,
        precomputed_arm_mask=exclude_mask,
    )


def run_relaxation_only(initial_keypoints, skeleton_pc, edges, segment_edges, segment_lengths,
                        n_branch, n_leaf, ee_to_leaf_kp, method_name, reference_edge_lengths=None):
    """
    Run only the relaxation step with specified ablation flags.
    All methods share the same initial_keypoints and edges from topology building.
    
    Args:
        initial_keypoints: K×3 array of keypoint positions from FPS (on skeleton)
        skeleton_pc: N×3 skeleton point cloud
        edges: List of (i,j) edge tuples
        segment_edges: List of lists of edges per segment
        segment_lengths: List of estimated segment lengths
        n_branch: Number of branch nodes
        n_leaf: Number of leaf nodes
        ee_to_leaf_kp: Dict mapping EE index to leaf keypoint index
        method_name: 'Full', 'NoProj', 'NoEdge', 'NoAnchor'
        reference_edge_lengths: Dict (i,j) -> target length from frame 0
        
    Returns:
        Relaxed keypoints K×3
    """
    flags = {
        'Full': dict(use_branch_leaf_anchors=True, use_projection=True, use_edge_repulsion=True),
        'NoProj': dict(use_branch_leaf_anchors=True, use_projection=False, use_edge_repulsion=True),
        'NoEdge': dict(use_branch_leaf_anchors=True, use_projection=True, use_edge_repulsion=False),
        'NoAnchor': dict(use_branch_leaf_anchors=False, use_projection=True, use_edge_repulsion=True),
    }
    
    use_branch_leaf_anchors = flags[method_name]['use_branch_leaf_anchors']
    use_projection = flags[method_name]['use_projection']
    use_edge_repulsion = flags[method_name]['use_edge_repulsion']
    
    # Build fixed_mask
    K = len(initial_keypoints)
    fixed_mask = np.zeros(K, dtype=bool)
    if use_branch_leaf_anchors:
        fixed_mask[:n_branch + n_leaf] = True
    else:
        if ee_to_leaf_kp:
            for kp_idx in ee_to_leaf_kp.values():
                fixed_mask[kp_idx] = True
    
    # Create a temporary initializer just for relaxation
    initializer = AblationInitializer(
        intrinsics=np.eye(3),  # Not used in relaxation
        n_keypoints=K,
        target_branch_nodes=n_branch,
        target_leaf_nodes=n_leaf,
        max_depth=2000.0,
        repulsion_iterations=200,
        repulsion_lr=10.0,
        ee_poses_3d=None,
        use_projection=use_projection,
        use_edge_repulsion=use_edge_repulsion,
        reference_edge_lengths=reference_edge_lengths,
    )
    
    keypoints = initializer._repulsion_relaxation_with_topology(
        initial_keypoints.copy(),
        skeleton_pc,
        fixed_mask,
        edges,
        segment_edges,
        segment_lengths=segment_lengths,
    )
    
    return keypoints


def process_clip(data, transforms, ee_poses_3d, clip_idx, start_frame, end_frame, output_dir,
                 n_keypoints=21, keypoints_per_segment=None, eval_frames=31):
    clip_output_dir = output_dir / f'clip_{clip_idx}'
    clip_output_dir.mkdir(parents=True, exist_ok=True)

    frame_end = min(start_frame + eval_frames, end_frame)
    if frame_end - start_frame < 2:
        return None

    method_names = ['Full', 'NoProj', 'NoEdge', 'NoAnchor']
    K = transforms['K']
    intrinsics = np.array([[K[0, 0], 0, K[0, 2]], [0, K[1, 1], K[1, 2]], [0, 0, 1]])

    # Frame 0 reference initialization (Full)
    ref_frame = start_frame
    ref_tracker = build_init_tracker(
        intrinsics,
        n_keypoints,
        ee_poses_3d[ref_frame],
        keypoints_per_segment,
        'Full',
    )
    ref_result = run_initialize_once(ref_tracker, data, ref_frame)
    if not ref_result.get('success', False):
        print(f"  Clip {clip_idx}: reference init failed at frame {ref_frame}")
        return None

    reference_edges = list(ref_tracker.reference_edges or ref_result.get('edges', []))
    reference_lengths = ref_tracker.reference_lengths.copy() if ref_tracker.reference_lengths is not None else np.array([])
    
    # Store frame 0's segment lengths and segment_edges for scaling
    ref_segment_lengths = ref_result.get('skeleton_segment_lengths', None)
    ref_segment_edges = ref_result.get('segment_edges', None)
    
    # Build edge-to-segment mapping for scaling
    edge_to_segment = {}
    if ref_segment_edges is not None:
        for seg_idx, seg_edge_list in enumerate(ref_segment_edges):
            for (i, j) in seg_edge_list:
                edge_to_segment[(i, j)] = seg_idx
                edge_to_segment[(j, i)] = seg_idx
    
    # Build reference_edge_lengths dict to pass to subsequent frames
    reference_edge_lengths_dict = {}
    for idx, (i, j) in enumerate(reference_edges):
        if idx < len(reference_lengths):
            reference_edge_lengths_dict[(i, j)] = reference_lengths[idx]
            reference_edge_lengths_dict[(j, i)] = reference_lengths[idx]
    print(f"  Reference edge lengths from frame 0: {len(reference_edge_lengths_dict)//2} edges")
    if ref_segment_lengths is not None:
        print(f"  Reference segment lengths: {[f'{l:.1f}' for l in ref_segment_lengths]}")

    all_metrics = {m: [] for m in method_names}
    keypoints_3d_histories = {m: [] for m in method_names}

    frames_dir = clip_output_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    for global_idx in range(start_frame + 1, frame_end):
        rgb = cv2.cvtColor(data['color'][global_idx], cv2.COLOR_BGR2RGB)
        panel_images = []

        # Step 1: Run Full method ONCE to get shared topology (initial keypoints, edges, skeleton_pc)
        # All ablation methods will share this topology and only differ in relaxation
        # Don't pass reference_edge_lengths - let it use current frame's segment_lengths
        full_tracker = build_init_tracker(
            intrinsics,
            n_keypoints,
            ee_poses_3d[global_idx],
            keypoints_per_segment,
            'Full',  # Always use Full for topology building
            reference_edge_lengths=None,  # Use current frame's segment_lengths
        )
        full_result = run_initialize_once(full_tracker, data, global_idx)
        
        if not full_result.get('success', False):
            print(f"  Frame {global_idx}: topology building failed")
            for method in method_names:
                keypoints_3d_histories[method].append(np.full((n_keypoints, 3), np.nan))
                all_metrics[method].append({
                    'frame': global_idx - (start_frame + 1),
                    'global_frame': global_idx,
                    'success': False,
                })
            continue
        
        # Extract shared topology from Full result
        initial_keypoints = full_result.get('initial_keypoints', full_result['keypoints'])  # Pre-relaxation
        shared_edges = full_result.get('edges', reference_edges)
        segment_edges = full_result.get('segment_edges', None)
        segment_lengths = full_result.get('skeleton_segment_lengths', None)
        n_branch = full_result.get('n_branch', 2)
        n_leaf = full_result.get('n_leaf', 4)
        ee_to_leaf_kp = full_result.get('ee_to_leaf_kp', {})
        skeleton_pc = full_result.get('skeleton_pc')
        ee_pos = ee_poses_3d[global_idx]
        
        print(f"  Frame {global_idx}: shared initial_keypoints from FPS (pre-relaxation)")
        
        # Build GT reference point cloud
        if skeleton_pc is not None and len(skeleton_pc) > 0:
            ref_pc = np.vstack([skeleton_pc, np.array(ee_pos, dtype=np.float32).reshape(-1, 3)])
        else:
            ref_pc = np.array(ee_pos, dtype=np.float32).reshape(-1, 3)

        # Compute scale factors per segment (current frame / frame 0)
        # This accounts for sensor noise in skeleton length measurements
        segment_scale_factors = {}
        if (ref_segment_lengths is not None and segment_lengths is not None 
            and len(ref_segment_lengths) == len(segment_lengths)):
            for seg_idx in range(len(ref_segment_lengths)):
                if ref_segment_lengths[seg_idx] > 1e-6:
                    segment_scale_factors[seg_idx] = segment_lengths[seg_idx] / ref_segment_lengths[seg_idx]
                else:
                    segment_scale_factors[seg_idx] = 1.0
        
        # Scale reference_lengths for edge error computation
        scaled_reference_lengths = reference_lengths.copy()
        for edge_idx, (i, j) in enumerate(reference_edges):
            if edge_idx < len(scaled_reference_lengths):
                seg_idx = edge_to_segment.get((i, j), None)
                if seg_idx is not None and seg_idx in segment_scale_factors:
                    scaled_reference_lengths[edge_idx] *= segment_scale_factors[seg_idx]

        # Step 2: For each method, run relaxation with SAME initial keypoints from FPS
        for method in method_names:
            print(f"    Running {method} relaxation...")
            
            # Run relaxation with method-specific flags on SHARED initial keypoints
            # Don't pass reference_edge_lengths - let it use current frame's segment_lengths
            # This automatically scales targets based on current skeleton measurements
            keypoints = run_relaxation_only(
                initial_keypoints=initial_keypoints,
                skeleton_pc=skeleton_pc,
                edges=shared_edges,
                segment_edges=segment_edges,
                segment_lengths=segment_lengths,  # Current frame's lengths (already scaled by sensor noise)
                n_branch=n_branch,
                n_leaf=n_leaf,
                ee_to_leaf_kp=ee_to_leaf_kp,
                method_name=method,
                reference_edge_lengths=None,  # Use fallback to segment_lengths
            )
            
            keypoints_2d = full_tracker._project_3d_to_2d(keypoints) if hasattr(full_tracker, '_project_3d_to_2d') else np.empty((0, 2))
            keypoints_3d_histories[method].append(keypoints.copy())

            # All methods use SAME shared_edges for fair comparison
            # Use scaled_reference_lengths to account for sensor noise in segment lengths
            edge_m = compute_edge_metrics(keypoints, shared_edges, scaled_reference_lengths)
            pos_m = compute_position_metrics(keypoints, ref_pc, extra_gt_points=None)

            n_ref_points = len(ref_pc) if ref_pc is not None and len(ref_pc) > 0 else 100
            pred_cloud = sample_points_on_edges(keypoints, shared_edges, n_ref_points)
            cd_m = compute_chamfer_metrics(pred_cloud, ref_pc)

            metrics = {
                'frame': global_idx - (start_frame + 1),
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

            all_metrics[method].append(metrics)

            # Create MST skeleton mask from keypoints_2d and edges
            # This draws the MST topology as lines between keypoints
            H, W = rgb.shape[:2]
            mst_skeleton_mask = np.zeros((H, W), dtype=np.uint8)
            if keypoints_2d is not None and len(keypoints_2d) > 0 and shared_edges is not None:
                kp_int = keypoints_2d.astype(int)
                for (i, j) in shared_edges:
                    if i < len(kp_int) and j < len(kp_int):
                        r1, c1 = kp_int[i, 0], kp_int[i, 1]
                        r2, c2 = kp_int[j, 0], kp_int[j, 1]
                        if 0 <= r1 < H and 0 <= c1 < W and 0 <= r2 < H and 0 <= c2 < W:
                            cv2.line(mst_skeleton_mask, (c1, r1), (c2, r2), 255, 1)
            
            panel = create_method_panel(
                rgb=rgb,
                skeleton_mask=mst_skeleton_mask,
                keypoints_2d=keypoints_2d,
                edges=shared_edges,
                method_name=method,
                metrics=metrics,
                frame_idx=global_idx,
                traj_history_2d=None,
                tail_length=0,
            )
            panel_images.append(panel)

        grid = create_ablation_grid(panel_images, global_idx, rgb.shape[:2], method_names=method_names)
        cv2.imwrite(str(frames_dir / f'frame_{global_idx:04d}_ablation.png'), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    per_frame_csv = clip_output_dir / 'per_frame.csv'
    with open(per_frame_csv, 'w') as f:
        f.write('Frame,GlobalFrame,Method,EdgePctMean,EdgePctStd,EdgePctMax,EdgeRMSE,PosRMSE,'
                'Edge<2%,Edge<5%,Edge<10%,Pos<2mm,Pos<5mm,Pos<10mm,'
                'CD,Pred2Ref,Ref2Pred,Prec@2mm,Prec@5mm,Prec@10mm,Rec@2mm,Rec@5mm,Rec@10mm,F@2mm,F@5mm,F@10mm\n')
        for method in method_names:
            for m in all_metrics[method]:
                f.write(f"{m['frame']},{m['global_frame']},{method},{m['edge_pct_mean']:.6f},{m['edge_pct_std']:.6f},"
                        f"{m['edge_pct_max']:.6f},{m['edge_rmse_mm']:.6f},{m['pos_rmse_mm']:.6f},"
                        f"{m['edge_under_2pct']:.4f},{m['edge_under_5pct']:.4f},{m['edge_under_10pct']:.4f},"
                        f"{m['pos_under_2mm']:.4f},{m['pos_under_5mm']:.4f},{m['pos_under_10mm']:.4f},"
                        f"{m['cd']:.4f},{m['cd_pred2ref']:.4f},{m['cd_ref2pred']:.4f},"
                        f"{m['precision_2mm']:.4f},{m['precision_5mm']:.4f},{m['precision_10mm']:.4f},"
                        f"{m['recall_2mm']:.4f},{m['recall_5mm']:.4f},{m['recall_10mm']:.4f},"
                        f"{m['f_2mm']:.4f},{m['f_5mm']:.4f},{m['f_10mm']:.4f}\n")

    summary_rows = []
    for method in method_names:
        metrics_list = all_metrics[method]
        if len(metrics_list) == 0:
            continue

        edge_pct_means = [m['edge_pct_mean'] for m in metrics_list if m['edge_pct_mean'] > 0]
        edge_rmses = [m['edge_rmse_mm'] for m in metrics_list if m['edge_rmse_mm'] > 0]
        pos_rmses = [m['pos_rmse_mm'] for m in metrics_list if m['pos_rmse_mm'] > 0]
        edge_under_2 = [m['edge_under_2pct'] for m in metrics_list if m['success']]
        edge_under_5 = [m['edge_under_5pct'] for m in metrics_list if m['success']]
        edge_under_10 = [m['edge_under_10pct'] for m in metrics_list if m['success']]
        pos_under_2 = [m['pos_under_2mm'] for m in metrics_list if m['success']]
        pos_under_5 = [m['pos_under_5mm'] for m in metrics_list if m['success']]
        pos_under_10 = [m['pos_under_10mm'] for m in metrics_list if m['success']]

        cd_vals = [m['cd'] for m in metrics_list if m['success']]
        cd_pred2ref_vals = [m['cd_pred2ref'] for m in metrics_list if m['success']]
        cd_ref2pred_vals = [m['cd_ref2pred'] for m in metrics_list if m['success']]
        precision_2 = [m['precision_2mm'] for m in metrics_list if m['success']]
        precision_5 = [m['precision_5mm'] for m in metrics_list if m['success']]
        precision_10 = [m['precision_10mm'] for m in metrics_list if m['success']]
        recall_2 = [m['recall_2mm'] for m in metrics_list if m['success']]
        recall_5 = [m['recall_5mm'] for m in metrics_list if m['success']]
        recall_10 = [m['recall_10mm'] for m in metrics_list if m['success']]
        f_2 = [m['f_2mm'] for m in metrics_list if m['success']]
        f_5 = [m['f_5mm'] for m in metrics_list if m['success']]
        f_10 = [m['f_10mm'] for m in metrics_list if m['success']]

        summary_rows.append({
            'method': method,
            'edge_pct_mean_avg': np.mean(edge_pct_means) if edge_pct_means else 0.0,
            'edge_pct_mean_std': np.std(edge_pct_means) if edge_pct_means else 0.0,
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
            'cd_avg': np.mean(cd_vals) if cd_vals else 0.0,
            'cd_std': np.std(cd_vals) if cd_vals else 0.0,
            'cd_pred2ref_avg': np.mean(cd_pred2ref_vals) if cd_pred2ref_vals else 0.0,
            'cd_ref2pred_avg': np.mean(cd_ref2pred_vals) if cd_ref2pred_vals else 0.0,
            'precision_2mm': np.mean(precision_2) if precision_2 else 0.0,
            'precision_5mm': np.mean(precision_5) if precision_5 else 0.0,
            'precision_10mm': np.mean(precision_10) if precision_10 else 0.0,
            'recall_2mm': np.mean(recall_2) if recall_2 else 0.0,
            'recall_5mm': np.mean(recall_5) if recall_5 else 0.0,
            'recall_10mm': np.mean(recall_10) if recall_10 else 0.0,
            'f_2mm': np.mean(f_2) if f_2 else 0.0,
            'f_5mm': np.mean(f_5) if f_5 else 0.0,
            'f_10mm': np.mean(f_10) if f_10 else 0.0,
        })

    summary_txt = clip_output_dir / 'summary.txt'
    with open(summary_txt, 'w') as f:
        f.write(f"Clip {clip_idx} Init Summary (reference frame {start_frame}, eval frames {start_frame+1}-{frame_end-1})\n")
        f.write("=" * 100 + "\n\n")

        f.write("Edge Length Metrics\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Method':<12} | {'Edge % Mean':<18} | {'Edge RMSE (mm)':<15} | {'<2%':<8} | {'<5%':<8} | {'<10%':<8}\n")
        f.write("-" * 100 + "\n")
        for s in summary_rows:
            f.write(f"{s['method']:<12} | {s['edge_pct_mean_avg']:>5.2f}% ±{s['edge_pct_mean_std']:>5.2f}% | "
                    f"{s['edge_rmse_avg']:>5.2f} ±{s['edge_rmse_std']:>4.2f} mm | "
                    f"{s['edge_under_2pct']:>5.1f}% | {s['edge_under_5pct']:>5.1f}% | {s['edge_under_10pct']:>5.1f}%\n")

        f.write("\n")
        f.write("Position RMSE Metrics\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<12} | {'Pos RMSE (mm)':<18} | {'<2mm':<8} | {'<5mm':<8} | {'<10mm':<8}\n")
        f.write("-" * 80 + "\n")
        for s in summary_rows:
            f.write(f"{s['method']:<12} | {s['pos_rmse_avg']:>5.2f} ±{s['pos_rmse_std']:>5.2f} mm   | "
                    f"{s['pos_under_2mm']:>5.1f}% | {s['pos_under_5mm']:>5.1f}% | {s['pos_under_10mm']:>5.1f}%\n")

        f.write("\n")
        f.write("Chamfer Distance Metrics \n")
        f.write("-" * 130 + "\n")
        f.write(f"{'Method':<12} | {'CD (mm)':<15} | {'Pred→Ref':<10} | {'Ref→Pred':<10} | {'Prec@2mm':<8} | {'Prec@5mm':<8} | {'Prec@10mm':<8} | {'Rec@2mm':<8} | {'Rec@5mm':<8} | {'Rec@10mm':<8}\n")
        f.write("-" * 130 + "\n")
        for s in summary_rows:
            f.write(f"{s['method']:<12} | {s['cd_avg']:>5.2f} ±{s['cd_std']:>4.2f} mm | "
                    f"{s['cd_pred2ref_avg']:>7.2f} mm | {s['cd_ref2pred_avg']:>7.2f} mm | "
                    f"{s['precision_2mm']:>5.1f}% | {s['precision_5mm']:>5.1f}% | {s['precision_10mm']:>5.1f}% | "
                    f"{s['recall_2mm']:>5.1f}% | {s['recall_5mm']:>5.1f}% | {s['recall_10mm']:>5.1f}%\n")

        f.write("\n")
        f.write("F-Scores\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Method':<12} | {'F@2mm':<12} | {'F@5mm':<12} | {'F@10mm':<12}\n")
        f.write("-" * 60 + "\n")
        for s in summary_rows:
            f.write(f"{s['method']:<12} | {s['f_2mm']:>8.2f}% | {s['f_5mm']:>8.2f}% | {s['f_10mm']:>8.2f}%\n")

    np.savez(
        clip_output_dir / '3d_keypoints.npz',
        full=np.array(keypoints_3d_histories['Full']),
        no_proj=np.array(keypoints_3d_histories['NoProj']),
        no_edge=np.array(keypoints_3d_histories['NoEdge']),
        no_anchor=np.array(keypoints_3d_histories['NoAnchor']),
        edge_connection=np.array(reference_edges) if reference_edges else np.array([]),
        reference_lengths=np.array(reference_lengths) if reference_lengths is not None else np.array([]),
    )

    return {
        'clip_idx': clip_idx,
        'summary_rows': summary_rows,
    }


def main():
    parser = argparse.ArgumentParser(description='BDLO initialization-only ablation experiment')
    parser.add_argument('--chunk', type=int, required=True, help='Chunk index')
    parser.add_argument('--dataset', type=str, default='batch', choices=['batch', 'faster'],
                        help='batch=bdlo_no_contact_4sec, faster=bdlo_no_contact_2sec')
    parser.add_argument('--clip_frames', type=int, default=300, help='Frames per clip (default: 300)')
    parser.add_argument('--eval_frames', type=int, default=31, help='Use first N frames per clip (default: 31)')
    parser.add_argument('--use_last_n_frames', type=int, default=600, 
                        help='Use only the last N frames from each chunk (default: 600)')
    parser.add_argument('--n_keypoints', type=int, default=21)
    parser.add_argument('--keypoints_per_segment', type=int, nargs=5, default=None)
    args = parser.parse_args()

    if args.dataset == 'batch':
        data_base = Path('/mnt/mydisk/captured_data_double_arm/bdlo_no_contact_4sec')
        output_base = Path('./bdlo1_init_ablation_results')
    else:
        data_base = Path('/mnt/mydisk/captured_data_double_arm/bdlo_no_contact_2sec')
        output_base = Path('./bdlo1_faster_init_ablation_results')

    calib_dir = Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/'
                     'roahm-deformable-objects/captured_calibration_data/test_0227')

    chunk_dir = data_base / f'chunk_{args.chunk}'
    output_dir = output_base / f'chunk_{args.chunk}'
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_chunk_data(chunk_dir)
    transforms = load_transforms(calib_dir)
    
    # Use only last N frames
    total_frames = data['n_frames']
    if args.use_last_n_frames > 0 and args.use_last_n_frames < total_frames:
        frame_offset = total_frames - args.use_last_n_frames
        usable_frames = args.use_last_n_frames
    else:
        frame_offset = 0
        usable_frames = total_frames

    ee_poses_3d = np.zeros((data['n_frames'], 2, 3))
    for i in range(data['n_frames']):
        ee_poses_3d[i] = get_ee_positions_cam(
            data['left_poses'][i], data['right_poses'][i],
            transforms['T_left_base2cam'], transforms['T_right_base2cam'],
        )

    n_keypoints = args.n_keypoints
    if args.keypoints_per_segment is not None:
        n_keypoints = 2 + 4 + sum(args.keypoints_per_segment)

    print('=' * 80)
    print('BDLO Initialization-Only Ablation')
    print('=' * 80)
    print(f'Dataset preset: {args.dataset}')
    print(f'Total frames in chunk: {total_frames}')
    print(f'Using last {usable_frames} frames (offset={frame_offset})')
    print(f'clip_frames: {args.clip_frames}')
    print(f'eval_frames (first): {args.eval_frames}')
    print('Reference frame: frame 0 in each clip (edge lengths passed to subsequent frames)')
    print(f'keypoints_per_segment: {args.keypoints_per_segment}')
    print(f'n_keypoints: {n_keypoints}')
    print('Methods: Full / NoProj / NoEdge / NoAnchor')

    n_clips = (usable_frames + args.clip_frames - 1) // args.clip_frames
    for clip_idx in range(n_clips):
        # Calculate frame range within the usable frames (last N)
        clip_start_in_usable = clip_idx * args.clip_frames
        clip_end_in_usable = min(clip_start_in_usable + args.clip_frames, usable_frames)
        
        # Map to actual frame indices (with offset)
        start_frame = frame_offset + clip_start_in_usable
        end_frame = frame_offset + clip_end_in_usable
        
        print(f"\nProcessing clip {clip_idx}: frames {start_frame}-{end_frame} (usable range: {clip_start_in_usable}-{clip_end_in_usable})")
        process_clip(
            data=data,
            transforms=transforms,
            ee_poses_3d=ee_poses_3d,
            clip_idx=clip_idx,
            start_frame=start_frame,
            end_frame=end_frame,
            output_dir=output_dir,
            n_keypoints=n_keypoints,
            keypoints_per_segment=args.keypoints_per_segment,
            eval_frames=args.eval_frames,
        )

    print(f'\nSaved initialization ablation outputs to: {output_dir}')


if __name__ == '__main__':
    main()
