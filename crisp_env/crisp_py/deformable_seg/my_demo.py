from typing import Dict, List, Optional, Tuple

import math
import numpy as np  # type: ignore[import]
import open3d as o3d  # type: ignore[import]

from seg_utils import (
    compute_rgb_mask,
    compute_point_cloud_mask,
    refine_rgb_mask_dbscan,
    filter_pcd_mask_dbscan,
    remove_small_components,
    filter_point_cloud_radius,
)
from gmm_utils import (
    extract_gmm_keypoints,
    classify_deformable_topology,
    build_wire_connections,
    refine_keypoints_uniform_edges,
)
from constrained_gmm_utils import UniformKeypoints
from repulsion_utils import (
    repulsion_relaxation,
    compute_spacing_stats,
)
from viz_utils import (
    create_color_point_cloud,
    create_mask_visualization,
    visualize_foreground_background_gmm,
    visualize_masked_point_clouds,
)



if __name__ == "__main__":
    print("Loading data for segmentation...")
    
    # Load sample data
    data = np.load("/home/yehengz/deformable_seg/data/full/test_colored_fabric.npy", allow_pickle=True).item()
    # data = np.load("/home/yehengz/deformable_seg/data/full/test_wire.npy", allow_pickle=True).item()
    data = np.load("/home/yehengz/deformable_seg/data/full/tracking_BDLO_data.npy", allow_pickle=True).item()
    sampled_points = data[0]

    bg_data = np.load("/home/yehengz/deformable_seg/data/bg/test_fabric_bg.npy", allow_pickle=True).item()
    # bg_data = np.load("/home/yehengz/deformable_seg/data/bg/test_wire_bg.npy", allow_pickle=True).item()
    bg_data = np.load("/home/yehengz/deformable_seg/data/bg/tracking_BDLO_background_data.npy", allow_pickle=True).item()
    sampled_points_bg = bg_data[0]

    # convert BGR to RGB for visualization and refinement colouring
    rgb_image = sampled_points['color'][:, :, ::-1]
    
    print("Creating segmentation masks...")
    
    # Camera intrinsics
    intrinsics = np.array([
        [606.1124267578125, 0, 641.7578125],
        [0, 605.8821411132812, 365.6518859863281],
        [0, 0, 1]
    ])
    
    # create rgb mask
    rgb_mask_initial = compute_rgb_mask(
        sampled_points_bg['color'], 
        sampled_points['color'], 
        threshold_percentile=90
        # threshold_percentile=99
    )

    print("Refining RGB mask using DBSCAN + neighbourhood propagation...")
    rgb_mask = refine_rgb_mask_dbscan(
        rgb_mask_initial,
        sampled_points['transformed_depth'],
        intrinsics,
        eps=40.0,
        min_samples=18,
        propagation_radius=40.0,
        propagation_min_neighbors=1,
        depth_tolerance=10.0,
        max_iterations=5,
        max_pixel_distance=5.0,
    )

    # compute point cloud mask
    pc_mask_initial = compute_point_cloud_mask(
        sampled_points_bg['transformed_depth'],
        sampled_points['transformed_depth'],
        intrinsics,
        distance_threshold=18
    )
    print("Filtering point cloud mask using DBSCAN...")
    pc_mask = filter_pcd_mask_dbscan(
        pc_mask_initial,
        sampled_points['transformed_depth'],
        intrinsics,
        eps=30.0,
        min_samples=18,
    )

    pc_mask = remove_small_components(pc_mask, min_size=800, depth_image=sampled_points['transformed_depth'], intrinsics=intrinsics, eps=30.0)
    
    print("All masks created successfully")
    
    # create 2D mask visualization
    print("Creating 2D mask visualization...")
    mask_save_path = "/home/yehengz/deformable_seg/mask_comparison.png"
    create_mask_visualization(
        rgb_image,
        rgb_mask,
        pc_mask,
        mask_save_path,
        rgb_mask_initial=rgb_mask_initial,
        pcd_mask_initial=pc_mask_initial,
    )
    print(f"Saved 2D mask visualization: {mask_save_path}")
    
    # Create color point cloud
    print("Creating color point cloud...")
    depth_data = sampled_points['transformed_depth']
    points, colors, valid_mask = create_color_point_cloud(
        rgb_image, depth_data, intrinsics, return_valid_mask=True
    )
    print(f"Point cloud created: {len(points):,} points")

    # Derive foreground mask aligned with unprojected points
    valid_flat = valid_mask.reshape(-1)
    valid_indices = np.flatnonzero(valid_flat)
    pc_mask_flat = (pc_mask > 128).reshape(-1)
    foreground_mask_points = pc_mask_flat[valid_indices] if valid_indices.size else np.zeros(0, dtype=bool)
    foreground_mask_points = foreground_mask_points.astype(bool)
    foreground_points = points[foreground_mask_points]
    foreground_colors = colors[foreground_mask_points]
    foreground_ratio = (len(foreground_points) / len(points)) if len(points) else 0.0


    # Print mask statistics
    print("Mask statistics:")
    rgb_initial_count = np.sum(rgb_mask_initial > 128)
    rgb_count = np.sum(rgb_mask > 128)
    pc_count = np.sum(pc_mask > 128)
    total_pixels = rgb_mask.size

    print(f"  RGB mask (initial): {rgb_initial_count:,} pixels ({rgb_initial_count/total_pixels*100:.1f}%)")
    print(f"  RGB mask (refined): {rgb_count:,} pixels ({rgb_count/total_pixels*100:.1f}%)")
    print(f"  PC mask: {pc_count:,} pixels ({pc_count/total_pixels*100:.1f}%)")
    print(
        f"  Point cloud foreground (PC mask): {len(foreground_points):,} / {len(points):,} "
        f"points ({foreground_ratio*100:.1f}%)"
    )

    # Filter foreground point cloud in 3D before GMM
    print("Filtering foreground point cloud in 3D...")
    foreground_points, foreground_colors = filter_point_cloud_radius(
        foreground_points, 
        foreground_colors, 
        radius=20.0, 
        min_neighbors=100
    )
    print(f"  Filtered foreground points: {len(foreground_points):,}")

    gmm_result = None
    gmm_viz_keypoints = None
    gmm_viz_mesh_vertices = None
    gmm_viz_mesh_triangles = None
    gmm_wire_edges = None
    topology_result = None
    if len(foreground_points) >= 3:
        print("Extracting uniform keypoints via FPS + repulsion...")
        try:
            # ============================================================
            # Stage 1: Farthest Point Sampling (FPS) initialization
            # ============================================================
            n_keypoints = 37
            uniform_kp = UniformKeypoints(
                n_keypoints=n_keypoints,
                n_neighbors=6,
                target_distance=None,
                delta=5.0,
                learning_rate=0.5,
                n_iterations=0,  # Skip internal relaxation, we'll use repulsion_utils
                covariance_reg=1e-5,
                run_em=False,  # Skip EM, just get FPS initialization
                em_iterations=0,
                random_state=0,
            )
            uniform_kp.fit(foreground_points)
            fps_keypoints = uniform_kp.means_.copy()
            print(f"  Stage 1 (FPS): {fps_keypoints.shape[0]} keypoints initialized")

            # Compute spacing stats after FPS
            fps_stats = compute_spacing_stats(fps_keypoints, n_neighbors=1)
            print(
                f"    FPS spacing → min: {fps_stats['min_dist']:.2f}, "
                f"max: {fps_stats['max_dist']:.2f}, "
                f"uniformity: {fps_stats['uniformity']:.3f}"
            )

            # ============================================================
            # Stage 2: Repulsion-based relaxation (blue-noise polishing)
            # ============================================================
            print("  Stage 2 (Repulsion): relaxing keypoint positions...")
            relaxation_result = repulsion_relaxation(
                fps_keypoints,
                foreground_points,
                n_neighbors=6,
                n_iterations=15,
                learning_rate=0.5,
                p=2.0,  # inverse-power repulsion
                epsilon=1e-8,
                project_each_step=True,
                return_debug=True,
            )
            keypoints = relaxation_result["keypoints"]
            relax_debug = relaxation_result.get("debug")

            # Compute spacing stats after relaxation
            final_stats = compute_spacing_stats(keypoints, n_neighbors=1)
            print(
                f"    Relaxed spacing → min: {final_stats['min_dist']:.2f}, "
                f"max: {final_stats['max_dist']:.2f}, "
                f"uniformity: {final_stats['uniformity']:.3f}"
            )

            if relax_debug:
                init_min = relax_debug.get("initial_min_dist", 0)
                final_min = relax_debug.get("final_min_dist", 0)
                if init_min > 0:
                    improvement = (final_min - init_min) / init_min * 100
                    print(f"    Min-dist improvement: {init_min:.2f} → {final_min:.2f} ({improvement:+.1f}%)")

            print(f"  Uniform keypoints extracted: {keypoints.shape[0]}")

            # Wrap into a stub for downstream visualization compatibility
            from types import SimpleNamespace
            gmm_result = SimpleNamespace(
                keypoints=keypoints,
                covariances=np.tile(np.eye(3) * 10.0, (keypoints.shape[0], 1, 1)),  # placeholder
                metadata={
                    "covariance_type": "diag",
                    "fps_stats": fps_stats,
                    "relaxed_stats": final_stats,
                    "relaxation_debug": relax_debug,
                },
                weights=np.ones(keypoints.shape[0]) / keypoints.shape[0],
                filtered_points=foreground_points,
                filtered_colors=foreground_colors,
            )
            meta = gmm_result.metadata
        except Exception as e:
            print(f"  Uniform keypoint extraction failed: {e}")
            import traceback
            traceback.print_exc()
            gmm_result = None
            meta = {}

        # Legacy GMM path (commented out)
        # try:
        #     gmm_result = extract_gmm_keypoints(
        #         foreground_points,
        #         colors=foreground_colors,
        #         nb_neighbors=30,
        #         std_ratio=3.0,
        #         downsample_method='fps',  # "voxel", "fps", "poisson"
        #         target_count=2000,
        #         n_components=37,
        #         covariance_type='spherical',
        #         random_state=0,
        #         reg_covar=1e-5,
        #         max_iter=500,
        #         tol=1e-3,
        #         n_init=8,
        #     )
        #     meta = gmm_result.metadata
        #     print(
        #         "  GMM converged: {converged}, components: {components}, iterations: {iters}".format(
        #             converged=meta.get("converged", True),
        #             components=meta.get("effective_components"),
        #             iters=meta.get("n_iter"),
        #         )
        #     )
        # except Exception as e:
        #     print(f"  GMM fitting failed: {e}")
        #     gmm_result = None

        if gmm_result is not None:

            def _run_uniform_edge_refinement(
                edge_list,
                keypoints_subset,
                *,
                branch_label: str,
                update_indices=None,
                refine_kwargs: Optional[Dict[str, object]] = None,
            ):
                if not edge_list:
                    print(f"    {branch_label} refinement skipped (no edges)")
                    return keypoints_subset, None
                if keypoints_subset is None or keypoints_subset.shape[0] < 2:
                    print(f"    {branch_label} refinement skipped (insufficient keypoints)")
                    return keypoints_subset, None

                refine_params = dict(
                    n_iters=400,
                    lr=0.05,
                    lambda_var=100.0,
                    lambda_anchor_b=0.0,
                    lambda_move=0.0,
                    project_each_step=True,
                    max_step=None,
                    max_total_move=None,
                    normalize_var=True,
                    return_debug=True,
                )
                if refine_kwargs:
                    refine_params.update(refine_kwargs)

                try:
                    refinement_result = refine_keypoints_uniform_edges(
                        keypoints_subset,
                        edges=edge_list,
                        foreground_points=foreground_points,
                        **refine_params,
                    )
                except Exception as refine_err:
                    print(f"    {branch_label} refinement skipped (error): {refine_err}")
                    return keypoints_subset, None

                refined_keypoints = refinement_result.get("keypoints_refined")
                if not isinstance(refined_keypoints, np.ndarray) or refined_keypoints.shape != keypoints_subset.shape:
                    print(f"    {branch_label} refinement skipped (unexpected output shape)")
                    return keypoints_subset, None

                debug_info = refinement_result.get("debug", {}) or {}
                initial_stats = debug_info.get("initial_edge_stats", {}) or {}
                final_stats = debug_info.get("final_edge_stats", {}) or {}

                def _to_float(value):
                    if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                        return float(value)
                    return float("nan")

                norm_start = _to_float(initial_stats.get("normalized_var"))
                norm_end = _to_float(final_stats.get("normalized_var"))
                print(
                    f"    {branch_label} refinement normalized variance: {norm_start:.4f} → {norm_end:.4f}"
                )

                mean_start = _to_float(initial_stats.get("mean"))
                mean_end = _to_float(final_stats.get("mean"))
                if not math.isnan(mean_start) and not math.isnan(mean_end):
                    print(
                        f"    {branch_label} refinement mean edge length: {mean_start:.3f} → {mean_end:.3f}"
                    )

                max_movement = _to_float(debug_info.get("max_movement"))
                if not math.isnan(max_movement):
                    print(f"    {branch_label} refinement max keypoint displacement: {max_movement:.3f}")

                if update_indices is None:
                    gmm_result.keypoints = refined_keypoints.copy()
                else:
                    update_indices = np.asarray(update_indices, dtype=int)
                    if update_indices.shape[0] != refined_keypoints.shape[0]:
                        print(
                            f"    {branch_label} refinement skipped (index mismatch: {update_indices.shape[0]} vs {refined_keypoints.shape[0]})"
                        )
                        return keypoints_subset, debug_info
                    gmm_result.keypoints[update_indices] = refined_keypoints

                post_lengths = []
                for edge in edge_list:
                    if edge is None or len(edge) < 2:
                        continue
                    i, j = int(edge[0]), int(edge[1])
                    if 0 <= i < refined_keypoints.shape[0] and 0 <= j < refined_keypoints.shape[0]:
                        post_lengths.append(
                            float(np.linalg.norm(refined_keypoints[i] - refined_keypoints[j]))
                        )
                if post_lengths:
                    post_lengths_np = np.asarray(post_lengths, dtype=np.float64)
                    print(
                        "    {label} post-refinement edge stats → mean: {mean:.3f}, variance: {var:.3f}".format(
                            label=branch_label,
                            mean=float(np.mean(post_lengths_np)),
                            var=float(np.var(post_lengths_np)),
                        )
                    )

                return refined_keypoints, debug_info

            try:
                topology_result = classify_deformable_topology(gmm_result, foreground_points)
            except ValueError as topo_err:
                print(f"  Skipping topology classification: {topo_err}")
            else:
                meta["topology"] = topology_result
                scores = topology_result.get("scores", {})
                counts = topology_result.get("counts", {})

                print(f"  Topology classification: {topology_result.get('topology', 'unknown')}")
                if scores:
                    linear_fraction = scores.get("linear_fraction")
                    planar_fraction = scores.get("planar_fraction")
                    valid_fraction = scores.get("valid_keypoint_fraction")
                    print(
                        "    Scores -> linear: {linear:.2f}, planar: {planar:.2f}, valid keypoints: {valid:.2f}".format(
                            linear=float(linear_fraction) if linear_fraction is not None else float('nan'),
                            planar=float(planar_fraction) if planar_fraction is not None else float('nan'),
                            valid=float(valid_fraction) if valid_fraction is not None else float('nan'),
                        )
                    )
                if counts:
                    print(
                        "    Keypoint counts -> linear: {linear}, planar: {planar}, volumetric: {volumetric}, "
                        "insufficient: {insufficient}, degenerate: {degenerate}".format(
                            linear=counts.get("linear", 0),
                            planar=counts.get("planar", 0),
                            volumetric=counts.get("volumetric", 0),
                            insufficient=counts.get("insufficient", 0),
                            degenerate=counts.get("degenerate", 0),
                        )
                    )

            is_wire_like = (
                topology_result is not None
                and topology_result.get("topology") == "wireharness_1d"
                and gmm_result.keypoints.size > 0
            )

            if is_wire_like:
                print("  Detected wire-like topology — computing skeleton wiring...")
                wiring_result = build_wire_connections(
                    gmm_result.keypoints,
                    foreground_points=foreground_points,
                    intrinsics=intrinsics,
                    image_shape=rgb_image.shape[:2],
                )
                raw_edges = wiring_result.get("edges", [])
                gmm_wire_edges = [(int(edge[0]), int(edge[1])) for edge in raw_edges if len(edge) >= 2]
                meta["wire_skeleton_graph"] = wiring_result
                print(
                    "    Skeleton edges: {edge_count}, total length: {length:.2f}".format(
                        edge_count=len(raw_edges),
                        length=float(wiring_result.get("total_length", 0.0)),
                    )
                )
                degree_stats = wiring_result.get("degrees", [])
                if degree_stats:
                    print(
                        "    Degree stats → min: {min_deg}, max: {max_deg}, avg: {avg_deg:.2f}".format(
                            min_deg=int(np.min(degree_stats)),
                            max_deg=int(np.max(degree_stats)),
                            avg_deg=float(np.mean(degree_stats)),
                        )
                    )
                summary = wiring_result.get("filter_summary", {})
                if summary:
                    print(
                        "    Candidate pairs: {pairs}, unreachable after constraints: {unreachable}".format(
                            pairs=int(summary.get("candidate_pairs", 0)),
                            unreachable=int(summary.get("unreachable_pairs", 0)),
                        )
                    )
                    valid_paths = summary.get("valid_paths")
                    if isinstance(valid_paths, int):
                        print(f"    Valid skeleton paths: {valid_paths}")
                    unreachable_details = summary.get("unreachable_details")
                    if unreachable_details:
                        print(f"    Unreachable pair details: {unreachable_details}")
                if raw_edges:
                    path_lengths = []
                    euclid_lengths = []
                    keypoints_arr = gmm_result.keypoints
                    for edge in raw_edges:
                        if edge is None or len(edge) < 2:
                            continue
                        i, j = int(edge[0]), int(edge[1])
                        if 0 <= i < keypoints_arr.shape[0] and 0 <= j < keypoints_arr.shape[0]:
                            euclid_lengths.append(
                                float(np.linalg.norm(keypoints_arr[i] - keypoints_arr[j]))
                            )
                        if len(edge) >= 3:
                            val = edge[2]
                            if isinstance(val, (int, float, np.floating)) and np.isfinite(val):
                                path_lengths.append(float(val))
                    if euclid_lengths:
                        euclid_np = np.asarray(euclid_lengths, dtype=np.float64)
                        print(
                            "    Edge length stats (Euclidean) → mean: {mean:.3f}, variance: {var:.3f}".format(
                                mean=float(np.mean(euclid_np)),
                                var=float(np.var(euclid_np)),
                            )
                        )
                    if path_lengths:
                        path_np = np.asarray(path_lengths, dtype=np.float64)
                        print(
                            "    Edge length stats (skeleton path) → mean: {mean:.3f}, variance: {var:.3f}".format(
                                mean=float(np.mean(path_np)),
                                var=float(np.var(path_np)),
                            )
                        )
                if raw_edges and len(gmm_result.keypoints) >= 2:
                    refined_wire, wire_debug = _run_uniform_edge_refinement(
                        raw_edges,
                        gmm_result.keypoints.copy(),
                        branch_label="Wire",
                        update_indices=None,
                        refine_kwargs={
                            "project_each_step": True,
                            "max_step": 150.0,
                            "max_total_move": 6000.0,
                            "lambda_var": 100,
                            "lambda_anchor_b": 10.0,
                            "lambda_move": 0.0,
                            "n_iters": 100,
                            "lr": 0.03,
                        },
                    )
                    if isinstance(refined_wire, np.ndarray):
                        gmm_viz_keypoints = refined_wire.copy()
                        if wire_debug is not None:
                            meta.setdefault("wire_skeleton_graph", {})["refinement"] = wire_debug
                            target_mean = wire_debug.get("target_mean_edge_length")
                            if target_mean is not None and np.isfinite(float(target_mean)):
                                print(f"    Wire target mean edge length: {float(target_mean):.3f}")
                gmm_viz_keypoints = gmm_result.keypoints.copy()

            elif gmm_result.keypoints.size > 0:
                print("Preparing GMM keypoints for Ball Pivoting reconstruction...")
                gmm_keypoints = gmm_result.keypoints.copy()
                keep_mask = np.ones(gmm_keypoints.shape[0], dtype=bool)
                gmm_colors = None
                if gmm_result.filtered_colors is not None:
                    gmm_colors = gmm_result.filtered_colors.copy()

                if gmm_keypoints.shape[0] > 2: # this is hacky to remove hanging points
                    y_mask = gmm_keypoints[:, 1] < -250.0
                    candidate_indices = np.flatnonzero(y_mask)
                    if candidate_indices.size > 0:
                        candidate_z = gmm_keypoints[candidate_indices, 2]
                        sorted_idx = np.argsort(candidate_z)[::-1]
                        remove_indices = candidate_indices[sorted_idx[:1]]
                        keep_mask[remove_indices] = False
                if not np.all(keep_mask):
                    gmm_keypoints = gmm_keypoints[keep_mask]
                    if gmm_colors is not None and len(gmm_colors) == keep_mask.shape[0]:
                        gmm_colors = gmm_colors[keep_mask]
                    else:
                        gmm_colors = None

                keep_indices = np.flatnonzero(keep_mask)
                fabric_refinement_edges: List[Tuple[int, int]] = []

                print(f"  Number of GMM keypoints after transform: {len(gmm_keypoints):,}")
                gmm_viz_keypoints = gmm_keypoints

                keypoint_pcd = o3d.geometry.PointCloud()
                keypoint_pcd.points = o3d.utility.Vector3dVector(gmm_keypoints)
                if gmm_colors is not None and len(gmm_colors) == len(gmm_keypoints):
                    keypoint_pcd.colors = o3d.utility.Vector3dVector(
                        np.clip(gmm_colors.astype(np.float32), 0.0, 1.0)
                    )

                keypoint_pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20.0, max_nn=30)
                )
                try:
                    keypoint_pcd.orient_normals_consistent_tangent_plane(30)
                except RuntimeError:
                    keypoint_pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))

                if len(gmm_keypoints) >= 3:
                    print("Running Ball Pivoting reconstruction on GMM keypoints...")
                    neighbor_distances = keypoint_pcd.compute_nearest_neighbor_distance()
                    if len(neighbor_distances) == 0:
                        avg_dist = float(np.linalg.norm(np.std(gmm_keypoints, axis=0)))
                    else:
                        avg_dist = float(np.mean(neighbor_distances))
                    if not np.isfinite(avg_dist) or avg_dist <= 0:
                        avg_dist = 1.0

                    radii = o3d.utility.DoubleVector([avg_dist, avg_dist * 2.0, avg_dist * 3.0])
                    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                        keypoint_pcd, radii
                    )

                    bpa_mesh.remove_unreferenced_vertices()
                    bpa_mesh.remove_degenerate_triangles()
                    bpa_mesh.remove_duplicated_triangles()
                    bpa_mesh.remove_duplicated_vertices()
                    bpa_mesh.remove_non_manifold_edges()

                    bpa_vertices_np = np.asarray(bpa_mesh.vertices)
                    bpa_triangles_np = np.asarray(bpa_mesh.triangles)
                    if bpa_vertices_np.size > 0 and bpa_triangles_np.size > 0:
                        gmm_viz_mesh_vertices = bpa_vertices_np
                        gmm_viz_mesh_triangles = bpa_triangles_np

                        # Compute edge length statistics for BPA mesh
                        edge_pairs = set()
                        for tri in bpa_triangles_np:
                            if len(tri) != 3:
                                continue
                            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                            edge_pairs.add(tuple(sorted((a, b))))
                            edge_pairs.add(tuple(sorted((b, c))))
                            edge_pairs.add(tuple(sorted((c, a))))
                        if edge_pairs:
                            edge_lengths = []
                            for i, j in edge_pairs:
                                if i >= len(bpa_vertices_np) or j >= len(bpa_vertices_np):
                                    continue
                                p_i = bpa_vertices_np[i]
                                p_j = bpa_vertices_np[j]
                                edge_lengths.append(float(np.linalg.norm(p_i - p_j)))
                            if edge_lengths:
                                edge_lengths_np = np.asarray(edge_lengths, dtype=np.float64)
                                mean_edge = float(np.mean(edge_lengths_np))
                                var_edge = float(np.var(edge_lengths_np))
                                print(
                                    "  BPA edge stats → mean: {mean:.3f}, variance: {var:.3f}".format(
                                        mean=mean_edge,
                                        var=var_edge,
                                    )
                                )

                            if bpa_vertices_np.shape[0] > 0 and gmm_keypoints.shape[0] > 0:
                                diff = bpa_vertices_np[:, None, :] - gmm_keypoints[None, :, :]
                                dist_sq = np.sum(diff * diff, axis=2)
                                nearest_indices = np.argmin(dist_sq, axis=1)
                                vertex_to_local = {
                                    mesh_idx: int(nearest_indices[mesh_idx])
                                    for mesh_idx in range(bpa_vertices_np.shape[0])
                                }
                                edge_set = set()
                                for i_mesh, j_mesh in edge_pairs:
                                    local_i = vertex_to_local.get(i_mesh)
                                    local_j = vertex_to_local.get(j_mesh)
                                    if local_i is None or local_j is None or local_i == local_j:
                                        continue
                                    pair = (local_i, local_j) if local_i < local_j else (local_j, local_i)
                                    edge_set.add(pair)
                                fabric_refinement_edges = [tuple(edge) for edge in edge_set]

                    print(f"  Number of BPA mesh vertices: {len(bpa_mesh.vertices):,}")
                    print(f"  Number of BPA mesh faces: {len(bpa_mesh.triangles):,}")

                if (
                    fabric_refinement_edges
                    and keep_indices.size == gmm_keypoints.shape[0]
                    and gmm_keypoints.shape[0] >= 2
                ):
                    print("  Refining surface keypoints using BPA edges...")
                    refined_surface, surface_debug = _run_uniform_edge_refinement(
                        fabric_refinement_edges,
                        gmm_keypoints.copy(),
                        branch_label="Surface",
                        update_indices=keep_indices,
                        refine_kwargs={
                            "project_each_step": False,
                            "max_step": 8.0,
                            "max_total_move": 40.0,
                            "lambda_move": 4.0,
                            "lambda_var": 0.5,
                            "lr": 0.02,
                            "n_iters": 30,
                        },
                    )
                    if isinstance(refined_surface, np.ndarray):
                        gmm_keypoints = refined_surface
                        gmm_viz_keypoints = refined_surface.copy()
                        meta["surface_refinement"] = {
                            "debug": surface_debug,
                            "kept_indices": keep_indices.tolist(),
                            "edge_type": "bpa_edges",
                            "edge_count": len(fabric_refinement_edges),
                        }
                        if surface_debug is not None:
                            target_mean = surface_debug.get("target_mean_edge_length")
                            if target_mean is not None and np.isfinite(float(target_mean)):
                                print(f"  Surface target mean edge length: {float(target_mean):.3f}")
                else:
                    gmm_viz_keypoints = gmm_keypoints

    else:
        print("Skipping GMM fitting — not enough foreground points after masking.")

    # Create 3D point cloud visualization
    print("Creating 3D point cloud visualization...")
    pc_save_path = "/home/yehengz/deformable_seg/point_cloud_comparison.html"
    visualize_masked_point_clouds(
        rgb_image,
        depth_data,
        intrinsics,
        rgb_mask,
        pc_mask,
        pc_save_path,
        point_size=2,
        downsample_factor=2,
        max_points=90000,
        axis_limits=None,
        percentile_clip=99,
        rgb_mask_initial=rgb_mask_initial,
        pcd_mask_initial=pc_mask_initial,
    )
    print(f"Saved interactive 3D point cloud visualization: {pc_save_path}")

    if gmm_result is not None:
        print("Creating combined GMM visualization...")
        gmm_visual_path = "/home/yehengz/deformable_seg/point_cloud_gmm.html"
        visualize_foreground_background_gmm(
            points,
            colors,
            foreground_mask_points,
            gmm_viz_keypoints if gmm_viz_keypoints is not None else gmm_result.keypoints,
            gmm_visual_path,
            point_size=2,
            keypoint_size=9,
            max_background_points=60_000,
            max_foreground_points=30_000,
            percentile_clip=99.0,
            mesh_vertices=gmm_viz_mesh_vertices,
            mesh_triangles=gmm_viz_mesh_triangles,
            gmm_result=gmm_result,
            ellipsoid_scale=2.0,
            ellipsoid_n_phi=25,
            ellipsoid_n_theta=36,
            graph_edges=gmm_wire_edges,
            downsampled_points=foreground_points,
        )
        print(f"Saved combined GMM visualization: {gmm_visual_path}")

    print("Segmentation complete!")