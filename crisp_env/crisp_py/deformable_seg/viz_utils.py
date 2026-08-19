import numpy as np  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
from matplotlib.patches import Circle  # type: ignore[import]

try:
    import plotly.graph_objects as go  # type: ignore[import]
except ImportError:  # pragma: no cover - runtime dependency check
    go = None


__all__ = [
    "create_color_point_cloud",
    "visualize_masked_point_clouds",
    "visualize_foreground_background_gmm",
    "create_mask_visualization",
    "create_skel_mask_viz",
]


def create_color_point_cloud(rgb_image, depth_image, intrinsics, return_valid_mask=False):
    """Lift an RGB-D frame into a colored 3D point cloud."""

    H, W = depth_image.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))

    valid = (depth_image > 0) & np.isfinite(depth_image)

    x = (u - cx) / fx * depth_image
    y = (v - cy) / fy * depth_image
    z = depth_image

    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    colors = rgb_image[valid] / 255.0

    if return_valid_mask:
        return points, colors, valid
    return points, colors


def visualize_masked_point_clouds(
    rgb_image,
    depth_image,
    intrinsics,
    rgb_mask,
    pc_mask,
    save_path,
    point_size=3,
    downsample_factor=4,
    max_points=80000,
    axis_limits=None,
    percentile_clip=99.5,
    rgb_mask_initial=None,
    pcd_mask_initial=None,
):
    """Render point clouds with masked pixels recolored red before lifting to 3D.

    For each mask (RGB, point-cloud), we recolor the masked pixels to
    pure red in the RGB image, lift the full depth map to 3D, and plot the
    resulting point cloud with per-point colors. The result is saved as an
    interactive Plotly HTML file for inspection.

    Args:
        rgb_image: (H, W, 3) RGB image (uint8 or float in [0, 255])
        depth_image: (H, W) depth map aligned with `rgb_image`
        intrinsics: (3, 3) camera intrinsic matrix
        rgb_mask / pc_mask: binary masks with value >128 marking foreground
        save_path: output path for the visualization figure
        point_size: plotly marker size (applied to every point)
        downsample_factor: stride used to thin the image/depth grid before lifting
        max_points: cap on the number of points plotted per subplot (random subsample)
    """

    if go is None:
        raise ImportError(
            "Plotly is required for interactive point cloud visualization. "
            "Install it with `pip install plotly`."
        )

    rgb_binary = rgb_mask > 128
    pc_binary = pc_mask > 128
    rgb_initial_binary = None
    pcd_initial_binary = None
    if rgb_mask_initial is not None:
        rgb_initial_binary = rgb_mask_initial > 128
    if pcd_mask_initial is not None:
        pcd_initial_binary = pcd_mask_initial > 128

    rgb_down = rgb_image[::downsample_factor, ::downsample_factor]
    depth_down = depth_image[::downsample_factor, ::downsample_factor]
    rgb_mask_down = rgb_binary[::downsample_factor, ::downsample_factor]
    pc_mask_down = pc_binary[::downsample_factor, ::downsample_factor]
    rgb_mask_initial_down = None
    pcd_mask_initial_down = None
    if rgb_initial_binary is not None:
        rgb_mask_initial_down = rgb_initial_binary[::downsample_factor, ::downsample_factor]
    if pcd_initial_binary is not None:
        pcd_mask_initial_down = pcd_initial_binary[::downsample_factor, ::downsample_factor]

    def random_subsample(points, colors):
        if len(points) <= max_points:
            return points, colors
        idx = np.random.choice(len(points), max_points, replace=False)
        return points[idx], colors[idx]

    def recolor_and_lift(mask_down):
        colored_rgb = rgb_down.copy()
        colored_rgb[mask_down] = np.array([255, 0, 0], dtype=colored_rgb.dtype)
        points, colors = create_color_point_cloud(colored_rgb, depth_down, intrinsics)
        return random_subsample(points, colors)

    full_points, full_colors = recolor_and_lift(np.zeros_like(rgb_mask_down, dtype=bool))
    rgb_points, rgb_colors = recolor_and_lift(rgb_mask_down)
    if rgb_mask_initial_down is not None:
        rgb_initial_points, rgb_initial_colors = recolor_and_lift(rgb_mask_initial_down)
    else:
        rgb_initial_points, rgb_initial_colors = None, None
    if pcd_mask_initial_down is not None:
        pcd_initial_points, pcd_initial_colors = recolor_and_lift(pcd_mask_initial_down)
    else:
        pcd_initial_points, pcd_initial_colors = None, None
    pc_points, pc_colors = recolor_and_lift(pc_mask_down)

    scenes = [
        ("Original RGB", full_points, full_colors),
    ]

    if rgb_initial_points is not None:
        scenes.append(("RGB Mask (initial) → Red", rgb_initial_points, rgb_initial_colors))
    if pcd_initial_points is not None:
        scenes.append(("PC Mask (initial) → Red", pcd_initial_points, pcd_initial_colors))

    scenes.extend([
        ("RGB Mask (refined) → Red", rgb_points, rgb_colors),
        ("PC Mask → Red", pc_points, pc_colors),
    ])

    def compute_axis_limits(points_list):
        concat_points = np.concatenate([pts for pts in points_list if len(pts) > 0], axis=0)
        if concat_points.size == 0:
            return (-1, 1), (-1, 1), (-1, 1)

        if percentile_clip is not None:
            lower = (100 - percentile_clip) / 2.0
            upper = 100 - lower
            x_min, x_max = np.percentile(concat_points[:, 0], [lower, upper])
            y_min, y_max = np.percentile(concat_points[:, 1], [lower, upper])
            z_min, z_max = np.percentile(concat_points[:, 2], [lower, upper])
        else:
            x_min, x_max = concat_points[:, 0].min(), concat_points[:, 0].max()
            y_min, y_max = concat_points[:, 1].min(), concat_points[:, 1].max()
            z_min, z_max = concat_points[:, 2].min(), concat_points[:, 2].max()

        if axis_limits:
            x_min = axis_limits.get("x_min", x_min)
            x_max = axis_limits.get("x_max", x_max)
            y_min = axis_limits.get("y_min", y_min)
            y_max = axis_limits.get("y_max", y_max)
            z_min = axis_limits.get("z_min", z_min)
            z_max = axis_limits.get("z_max", z_max)

        margin = 0.01
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        eps = 1e-6
        return (
            (x_min - margin * max(x_range, eps), x_max + margin * max(x_range, eps)),
            (y_min - margin * max(y_range, eps), y_max + margin * max(y_range, eps)),
            (z_min - margin * max(z_range, eps), z_max + margin * max(z_range, eps)),
        )

    x_range, y_range, z_range = compute_axis_limits([p for _, p, _ in scenes])

    fig = go.Figure()

    def to_rgb_strings(color_array):
        if len(color_array) == 0:
            return []
        color_uint8 = np.clip(color_array * 255.0, 0, 255).astype(np.uint8)
        return [f"rgb({r},{g},{b})" for r, g, b in color_uint8]

    for idx, (title, pts, cols) in enumerate(scenes):
        if len(pts) == 0:
            fig.add_trace(
                go.Scatter3d(
                    x=[], y=[], z=[],
                    mode="markers",
                    name=title,
                    visible=(idx == 0),
                    marker=dict(size=point_size)
                )
            )
            continue

        colors_rgb = to_rgb_strings(cols)
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                name=title,
                visible=(idx == 0),
                marker=dict(size=point_size, color=colors_rgb)
            )
        )

    num_traces = len(scenes)
    buttons = []
    for idx, (title, _, _) in enumerate(scenes):
        visibility = [False] * num_traces
        visibility[idx] = True
        buttons.append(
            dict(
                label=title,
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"{title} Point Cloud"},
                ],
            )
        )

    # Add a button to show all traces simultaneously
    buttons.insert(
        0,
        dict(
            label="All",
            method="update",
            args=[
                {"visible": [True] * num_traces},
                {"title": "All Point Clouds"},
            ],
        ),
    )

    fig.update_layout(
        title="Original RGB Point Cloud",
        scene=dict(
            xaxis=dict(title="X", range=x_range),
            yaxis=dict(title="Y", range=y_range),
            zaxis=dict(title="Z", range=z_range),
            aspectmode="data",
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=0, r=0, b=0, t=40),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=buttons,
                showactive=True,
                x=0.5,
                xanchor="center",
                y=1.12,
                yanchor="top",
            )
        ],
    )

    fig.write_html(save_path, include_plotlyjs="cdn")

    fig.update_layout(
        title="Original RGB Point Cloud",
        scene=dict(
            xaxis=dict(title="X", range=x_range),
            yaxis=dict(title="Y", range=y_range),
            zaxis=dict(title="Z", range=z_range),
            aspectmode="data",
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=0, r=0, b=0, t=40),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=buttons,
                showactive=True,
                x=0.5,
                xanchor="center",
                y=1.12,
                yanchor="top",
            )
        ],
    )

    fig.write_html(save_path, include_plotlyjs="cdn")


def visualize_foreground_background_gmm(
    points,
    colors,
    foreground_mask,
    keypoints,
    save_path,
    *,
    point_size=2,
    keypoint_size=7,
    max_background_points=150_000,
    max_foreground_points=80_000,
    max_downsampled_points=50_000,
    percentile_clip=99.0,
    mesh_vertices=None,
    mesh_triangles=None,
    gmm_result=None,
    ellipsoid_scale=2.0,
    ellipsoid_n_phi=25,
    ellipsoid_n_theta=36,
    graph_edges=None,
    graph_edge_color="rgba(0, 0, 255, 0.95)",
    graph_edge_width=5,
    graph_edge_marker_size=4,
    downsampled_points=None,
    downsampled_color="rgba(50,205,50,0.9)",
    downsampled_name="Downsampled (GMM input)",
    downsampled_point_size=None,
    foreground_color="rgba(255,0,0,0.9)",
    keypoint_color=None,
):
    """Create an interactive Plotly visualization of background, foreground, GMM keypoints, and ellipsoids.

    When ``graph_edges`` is provided (e.g., MST connections for wire-like topologies), the
    function overlays the corresponding line segments on top of the keypoints.
    ``graph_edge_color``, ``graph_edge_width``, and ``graph_edge_marker_size`` control
    the styling of that overlay.
    """

    if go is None:
        raise ImportError(
            "Plotly is required for interactive point cloud visualization. Install it with `pip install plotly`."
        )

    points = np.asarray(points)
    colors = None if colors is None else np.asarray(colors)
    foreground_mask = np.asarray(foreground_mask, dtype=bool)
    keypoints = np.asarray(keypoints) if keypoints is not None else np.empty((0, 3))
    mesh_vertices = (
        np.asarray(mesh_vertices)
        if mesh_vertices is not None and len(mesh_vertices) > 0
        else None
    )
    mesh_triangles = (
        np.asarray(mesh_triangles, dtype=np.int64)
        if mesh_triangles is not None and len(mesh_triangles) > 0
        else None
    )
    downsampled_points = (
        np.asarray(downsampled_points, dtype=np.float64)
        if downsampled_points is not None and len(downsampled_points) > 0
        else None
    )

    ellipsoid_centers = None
    ellipsoid_covariance_type = None
    ellipsoid_weights = None
    if gmm_result is not None:
        ellipsoid_centers = np.asarray(gmm_result.keypoints, dtype=np.float64)
        ellipsoid_covariances = gmm_result.covariances
        ellipsoid_covariance_type = gmm_result.metadata.get("covariance_type", "full")
        ellipsoid_weights = np.asarray(gmm_result.weights, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("`points` must be of shape (N, 3).")
    if colors is not None and colors.shape != points.shape:
        raise ValueError("`colors` must match the shape of `points`.")
    if foreground_mask.shape[0] != points.shape[0]:
        raise ValueError("`foreground_mask` must be length N matching the number of points.")
    if downsampled_points is not None:
        if downsampled_points.ndim != 2 or downsampled_points.shape[1] != 3:
            raise ValueError("`downsampled_points` must be of shape (M, 3).")

    background_mask = ~foreground_mask
    foreground_points = points[foreground_mask]
    background_points = points[background_mask]
    background_colors = None if colors is None else colors[background_mask]

    def subsample(arr_points, arr_colors, limit):
        if limit is None or arr_points.shape[0] <= limit:
            return arr_points, arr_colors
        idx = np.random.choice(arr_points.shape[0], limit, replace=False)
        colors_sub = None if arr_colors is None else arr_colors[idx]
        return arr_points[idx], colors_sub

    background_points, background_colors = subsample(
        background_points, background_colors, max_background_points
    )
    foreground_points, _ = subsample(foreground_points, None, max_foreground_points)

    downsampled_points_plot = None
    if downsampled_points is not None:
        downsampled_points_plot, _ = subsample(downsampled_points, None, max_downsampled_points)

    ellipsoid_points_flat = []

    def to_rgb_strings(color_array):
        if color_array is None or len(color_array) == 0:
            return []
        color_uint8 = np.clip(color_array * 255.0, 0, 255).astype(np.uint8)
        return [f"rgb({r},{g},{b})" for r, g, b in color_uint8]

    background_colors_plotly = (
        to_rgb_strings(background_colors) if background_colors is not None else "rgb(180,180,180)"
    )

    fig = go.Figure()
    edge_trace_data = None

    if background_points.shape[0] > 0:
        fig.add_trace(
            go.Scatter3d(
                x=background_points[:, 0],
                y=background_points[:, 1],
                z=background_points[:, 2],
                mode="markers",
                name="Background",
                marker=dict(
                    size=point_size,
                    color=background_colors_plotly,
                    opacity=0.65,
                ),
            )
        )

    if foreground_points.shape[0] > 0:
        fig.add_trace(
            go.Scatter3d(
                x=foreground_points[:, 0],
                y=foreground_points[:, 1],
                z=foreground_points[:, 2],
                mode="markers",
                name="Foreground",
                marker=dict(
                    size=point_size,
                    color=foreground_color,
                ),
            )
        )

    if downsampled_points_plot is not None and downsampled_points_plot.shape[0] > 0:
        ds_marker_size = (
            downsampled_point_size
            if downsampled_point_size is not None
            else max(point_size + 1, 2)
        )
        fig.add_trace(
            go.Scatter3d(
                x=downsampled_points_plot[:, 0],
                y=downsampled_points_plot[:, 1],
                z=downsampled_points_plot[:, 2],
                mode="markers",
                name=downsampled_name,
                marker=dict(
                    size=ds_marker_size*3,
                    color=downsampled_color,
                    opacity=1,
                ),
            )
        )

    if keypoints.shape[0] > 0:
        keypoints_plot = keypoints.copy()
        # Use keypoint_color if provided, otherwise match graph_edge_color
        kp_color = keypoint_color if keypoint_color is not None else graph_edge_color
        fig.add_trace(
            go.Scatter3d(
                x=keypoints_plot[:, 0],
                y=keypoints_plot[:, 1],
                z=keypoints_plot[:, 2],
                mode="markers",
                name="GMM Keypoints",
                marker=dict(
                    size=keypoint_size,
                    color=kp_color,
                    symbol="diamond",
                ),
            )
        )

    if graph_edges is not None and len(graph_edges) > 0 and keypoints.shape[0] > 0:
        edge_x, edge_y, edge_z = [], [], []
        for edge in graph_edges:
            if edge is None:
                continue
            if len(edge) < 2:
                continue
            i, j = int(edge[0]), int(edge[1])
            if i < 0 or j < 0 or i >= keypoints.shape[0] or j >= keypoints.shape[0]:
                continue
            p_i = keypoints[i]
            p_j = keypoints[j]
            edge_x.extend([p_i[0], p_j[0], None])
            edge_y.extend([p_i[1], p_j[1], None])
            edge_z.extend([p_i[2], p_j[2], None])

        if edge_x:
            edge_trace_data = (edge_x, edge_y, edge_z)

    if ellipsoid_centers is not None and ellipsoid_centers.size > 0:
        phi = np.linspace(0.0, np.pi, ellipsoid_n_phi)
        theta = np.linspace(0.0, 2.0 * np.pi, ellipsoid_n_theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x_unit = np.outer(sin_phi, cos_theta)
        y_unit = np.outer(sin_phi, sin_theta)
        z_unit = np.outer(cos_phi, np.ones_like(theta))

        weights_array = ellipsoid_weights if ellipsoid_weights is not None else None
        max_weight = float(np.max(weights_array)) if weights_array is not None and weights_array.size > 0 else 1.0

        for idx in range(ellipsoid_centers.shape[0]):
            cov_matrix = _get_component_covariance(
                ellipsoid_covariances,
                ellipsoid_covariance_type,
                idx,
            )
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)
            eigvals = np.clip(eigvals, 1e-10, None)
            radii = np.sqrt(eigvals) * ellipsoid_scale
            transform = eigvecs @ np.diag(radii)

            sphere_stack = np.stack([x_unit, y_unit, z_unit], axis=-1)
            transformed = sphere_stack @ transform.T
            ellipsoid_world = transformed + ellipsoid_centers[idx]

            ellipsoid_plot = ellipsoid_world.copy()
            ellipsoid_points_flat.append(ellipsoid_plot.reshape(-1, 3))

            if weights_array is not None and idx < len(weights_array):
                comp_weight = float(weights_array[idx])
            else:
                comp_weight = 1.0
            denom = max_weight + 1e-12
            opacity = float(np.clip(0.25 + 0.5 * comp_weight / denom, 0.2, 0.85))

            legend_is_first = idx == 0
            fig.add_trace(
                go.Surface(
                    x=ellipsoid_plot[:, :, 0],
                    y=ellipsoid_plot[:, :, 1],
                    z=ellipsoid_plot[:, :, 2],
                    opacity=opacity,
                    showscale=False,
                    name="GMM Ellipsoids" if legend_is_first else f"Gaussian #{idx+1}",
                    legendgroup="gmm-ellipsoids",
                    legendgrouptitle_text="GMM Ellipsoids" if legend_is_first else None,
                    showlegend=legend_is_first,
                    surfacecolor=np.full_like(x_unit, comp_weight),
                    colorscale=[[0, "#4682B4"], [1, "#1E90FF"]],
                )
            )

    if mesh_vertices is not None and mesh_triangles is not None:
        edge_pairs = set()
        for tri in mesh_triangles:
            if len(tri) != 3:
                continue
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            edge_pairs.add(tuple(sorted((a, b))))
            edge_pairs.add(tuple(sorted((b, c))))
            edge_pairs.add(tuple(sorted((c, a))))

        if edge_pairs:
            edge_x, edge_y, edge_z = [], [], []
            verts = mesh_vertices
            for i, j in edge_pairs:
                if i >= len(verts) or j >= len(verts):
                    continue
                p_i, p_j = verts[i], verts[j]
                edge_x.extend([p_i[0], p_j[0], None])
                edge_y.extend([p_i[1], p_j[1], None])
                edge_z.extend([p_i[2], p_j[2], None])

            fig.add_trace(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode="lines",
                    line=dict(color="blue", width=4),
                    name="BPA edges",
                    hoverinfo="none",
                )
            )

    if edge_trace_data is not None:
        edge_x, edge_y, edge_z = edge_trace_data
        fig.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines+markers",
                name="Wire MST",
                line=dict(color=graph_edge_color, width=graph_edge_width),
                marker=dict(size=graph_edge_marker_size, color=graph_edge_color),
                hoverinfo="none",
                legendgroup="wire-mst",
                legendgrouptitle_text="Wire Connectivity",
            )
        )

    all_sets = []
    for arr in (background_points, foreground_points, downsampled_points_plot, keypoints):
        if arr is not None and arr.size > 0:
            all_sets.append(arr)
    if mesh_vertices is not None and mesh_vertices.size > 0:
        all_sets.append(mesh_vertices)
    if graph_edges is not None and len(graph_edges) > 0 and keypoints.shape[0] > 0:
        segments = []
        for edge in graph_edges:
            if edge is None or len(edge) < 2:
                continue
            i, j = int(edge[0]), int(edge[1])
            if i < 0 or j < 0 or i >= keypoints.shape[0] or j >= keypoints.shape[0]:
                continue
            segments.append(keypoints[[i, j]])
        if segments:
            all_sets.append(np.concatenate(segments, axis=0))
    for ellipsoid_pts in ellipsoid_points_flat:
        if ellipsoid_pts.size > 0:
            all_sets.append(ellipsoid_pts)

    if all_sets:
        concatenated = np.concatenate(all_sets, axis=0)
        if percentile_clip is not None:
            lower = (100 - percentile_clip) / 2.0
            upper = 100 - lower
            x_min, x_max = np.percentile(concatenated[:, 0], [lower, upper])
            y_min, y_max = np.percentile(concatenated[:, 1], [lower, upper])
            z_min, z_max = np.percentile(concatenated[:, 2], [lower, upper])
        else:
            x_min, x_max = concatenated[:, 0].min(), concatenated[:, 0].max()
            y_min, y_max = concatenated[:, 1].min(), concatenated[:, 1].max()
            z_min, z_max = concatenated[:, 2].min(), concatenated[:, 2].max()
        margin = 0.02
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        eps = 1e-6
        x_bounds = (x_min - margin * max(x_range, eps), x_max + margin * max(x_range, eps))
        y_bounds = (y_min - margin * max(y_range, eps), y_max + margin * max(y_range, eps))
        z_bounds = (z_min - margin * max(z_range, eps), z_max + margin * max(z_range, eps))
    else:
        x_bounds = y_bounds = z_bounds = (-1, 1)

    fig.update_layout(
        title="Foreground / Background / GMM Overlays",
        scene=dict(
            xaxis=dict(title="X", range=x_bounds),
            yaxis=dict(title="Y", range=y_bounds),
            zaxis=dict(title="Z", range=z_bounds),
            aspectmode="data",
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.write_html(save_path, include_plotlyjs="cdn")


def _get_component_covariance(covariances, covariance_type, index):
    if covariance_type == "full":
        return np.asarray(covariances[index], dtype=np.float64)
    if covariance_type == "tied":
        return np.asarray(covariances, dtype=np.float64)
    if covariance_type == "diag":
        return np.diag(np.asarray(covariances[index], dtype=np.float64))
    if covariance_type == "spherical":
        cov_value = float(covariances[index])
        return np.eye(3, dtype=np.float64) * cov_value
    raise ValueError(f"Unsupported covariance_type: {covariance_type}")


def create_mask_visualization(
    rgb_image,
    rgb_mask,
    pc_mask,
    save_path,
    rgb_mask_initial=None,
    pcd_mask_initial=None,
):
    """Create 3x2 visualization: binary masks and RGB overlays for each mask type."""

    # Convert masks to binary
    rgb_binary = rgb_mask > 128
    pc_binary = pc_mask > 128
    if rgb_mask_initial is not None:
        rgb_initial_binary = rgb_mask_initial > 128
    else:
        rgb_initial_binary = None
    if pcd_mask_initial is not None:
        pcd_initial_binary = pcd_mask_initial > 128
    else:
        pcd_initial_binary = None

    # Create RGB overlays with red highlighting
    def create_overlay(image, mask_binary, alpha=0.4):
        overlay = image.copy().astype(np.float32) / 255.0
        red_overlay = np.zeros_like(overlay)
        red_overlay[mask_binary, 0] = 1.0  # Red channel
        blended = (1 - alpha) * overlay + alpha * red_overlay
        return np.clip(blended, 0, 1)

    rgb_overlay = create_overlay(rgb_image, rgb_binary)
    pc_overlay = create_overlay(rgb_image, pc_binary)
    if rgb_initial_binary is not None:
        rgb_initial_overlay = create_overlay(rgb_image, rgb_initial_binary)
    else:
        rgb_initial_overlay = None
    if pcd_initial_binary is not None:
        pcd_initial_overlay = create_overlay(rgb_image, pcd_initial_binary)
    else:
        pcd_initial_overlay = None

    rows = []
    if rgb_initial_binary is not None:
        rows.append([
            (rgb_mask_initial, 'RGB Mask (initial)', 'gray'),
            (rgb_initial_overlay, 'RGB Overlay (initial)', None),
            (rgb_mask, 'RGB Mask (refined)', 'gray'),
            (rgb_overlay, 'RGB Overlay (refined)', None),
        ])
    else:
        rows.append([
            (rgb_mask, 'RGB Binary Mask', 'gray'),
            (rgb_overlay, 'RGB Mask Overlay', None),
        ])

    if pcd_initial_binary is not None:
        rows.append([
            (pcd_mask_initial, 'PC Mask (initial)', 'gray'),
            (pcd_initial_overlay, 'PC Overlay (initial)', None),
            (pc_mask, 'PC Mask (refined)', 'gray'),
            (pc_overlay, 'PC Overlay (refined)', None),
        ])
    else:
        rows.append([
            (pc_mask, 'Point Cloud Binary Mask', 'gray'),
            (pc_overlay, 'Point Cloud Mask Overlay', None),
        ])

    ncols = max(len(r) for r in rows)
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 6 * nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, row in enumerate(rows):
        for col_idx in range(ncols):
            ax = axes[row_idx, col_idx]
            if col_idx < len(row):
                data, title, cmap = row[col_idx]
                if cmap == 'gray':
                    ax.imshow(data, cmap='gray', vmin=0, vmax=255)
                else:
                    ax.imshow(data)
                ax.set_title(title)
                ax.axis('off')
            else:
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_skel_mask_viz(
    rgb_image,
    pc_mask,
    skeleton_pc_mask,
    branch_nodes,
    end_nodes,
    save_path,
    *,
    radius=3,
    mask_color=(255, 0, 0),
    mask_alpha=0.4,
    skeleton_color=(0, 255, 0),
):
    """Create a 2x2 grid showing RGB, mask overlay, skeleton overlay, and annotated nodes overlay."""

    rgb_image = np.asarray(rgb_image)
    if rgb_image.ndim != 3 or rgb_image.shape[2] not in (3, 4):
        raise ValueError("`rgb_image` must have shape (H, W, 3) or (H, W, 4).")

    pc_mask = np.asarray(pc_mask)
    skeleton_pc_mask = np.asarray(skeleton_pc_mask)
    if pc_mask.shape != skeleton_pc_mask.shape[:2]:
        raise ValueError("`pc_mask` and `skeleton_pc_mask` must have matching spatial dimensions.")

    if rgb_image.shape[:2] != pc_mask.shape:
        raise ValueError("`rgb_image` must align spatially with the masks.")

    branch_nodes = np.asarray(branch_nodes, dtype=np.int64)
    end_nodes = np.asarray(end_nodes, dtype=np.int64)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Prepare RGB as float
    rgb_float = rgb_image.astype(np.float32) / 255.0 if rgb_image.max() > 1 else rgb_image.astype(np.float32)

    # Top-left: RGB Image
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title("RGB Image")
    axes[0, 0].axis("off")

    # Top-right: PC Mask overlay on RGB (before skeletonization)
    mask_overlay = rgb_float.copy()
    mask_bool = pc_mask > 0
    mask_color_normalized = np.array(mask_color, dtype=np.float32) / 255.0
    mask_overlay[mask_bool] = (
        mask_overlay[mask_bool] * (1 - mask_alpha) + 
        mask_color_normalized * mask_alpha
    )
    axes[0, 1].imshow(np.clip(mask_overlay, 0, 1))
    axes[0, 1].set_title("PC Mask Overlay")
    axes[0, 1].axis("off")

    # Bottom-left: Skeleton overlay on RGB (opacity=1 for skeleton pixels)
    skeleton_overlay = rgb_float.copy()
    skeleton_bool = skeleton_pc_mask > 0
    skeleton_color_normalized = np.array(skeleton_color, dtype=np.float32) / 255.0
    skeleton_overlay[skeleton_bool] = skeleton_color_normalized  # opacity=1
    axes[1, 0].imshow(np.clip(skeleton_overlay, 0, 1))
    axes[1, 0].set_title("Skeleton Overlay")
    axes[1, 0].axis("off")

    # Bottom-right: Skeleton with nodes overlay on RGB
    ax_nodes = axes[1, 1]
    nodes_overlay = skeleton_overlay.copy()  # Start with skeleton overlay
    ax_nodes.imshow(np.clip(nodes_overlay, 0, 1))
    ax_nodes.set_title("Skeleton Nodes Overlay")
    ax_nodes.axis("off")

    for nodes, color in ((branch_nodes, "red"), (end_nodes, "blue")):
        if nodes.size == 0:
            continue
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError("Node arrays must have shape (N, 2).")
        for row, col in nodes:
            circ = Circle((col, row), radius=radius, color=color, fill=True, alpha=1.0)
            ax_nodes.add_patch(circ)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
