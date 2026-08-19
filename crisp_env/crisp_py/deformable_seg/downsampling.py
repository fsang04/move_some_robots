import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

try:
    import open3d as o3d  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    o3d = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import plotly.graph_objects as go  # type: ignore[import]
except ImportError:
    go = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sklearn.neighbors import NearestNeighbors  # type: ignore[import]
except ImportError:
    NearestNeighbors = None  # type: ignore

from viz_utils import create_color_point_cloud
from seg_utils import filter_pcd_mask_dbscan


def compute_point_cloud_mask(
    bg_depth: np.ndarray,
    fg_depth: np.ndarray,
    intrinsics: np.ndarray,
    distance_threshold: float = 18.0,
) -> np.ndarray:
    """Compute a foreground mask by comparing two depth frames in 3D space."""

    bg_depth = np.asarray(bg_depth, dtype=np.float32)
    fg_depth = np.asarray(fg_depth, dtype=np.float32)
    if bg_depth.shape != fg_depth.shape:
        raise ValueError("Background and foreground depth maps must have the same shape")

    h, w = bg_depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    def lift(depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        valid = (depth > 0) & np.isfinite(depth)
        x = (u - cx) / fx * depth
        y = (v - cy) / fy * depth
        z = depth
        return x, y, z, valid

    bg_x, bg_y, bg_z, bg_valid = lift(bg_depth)
    fg_x, fg_y, fg_z, fg_valid = lift(fg_depth)

    valid_mask = bg_valid & fg_valid
    if not np.any(valid_mask):
        return np.zeros_like(bg_depth, dtype=np.uint8)

    bg_points = np.stack([bg_x, bg_y, bg_z], axis=-1)
    fg_points = np.stack([fg_x, fg_y, fg_z], axis=-1)
    delta = fg_points - bg_points
    distance = np.linalg.norm(delta, axis=-1)

    change_mask = distance > float(distance_threshold)
    mask = (valid_mask & change_mask).astype(np.uint8) * 255
    mask = change_mask.astype(np.uint8) * 255
    return mask


def estimate_uniform_spacing(points: np.ndarray, target_count: int) -> float:
    if target_count <= 0 or len(points) == 0:
        return 1.0

    if NearestNeighbors is not None and len(points) > 8:
        nn = NearestNeighbors(n_neighbors=9)
        nn.fit(points)
        distances, _ = nn.kneighbors(points, return_distance=True)
        local_spacing = np.median(distances[:, 1])
        if np.isfinite(local_spacing) and local_spacing > 0:
            scale = (float(len(points)) / float(target_count)) ** (1.0 / 3.0)
            return float(max(local_spacing * scale, 1e-3))

    extent = np.ptp(points, axis=0)
    extent = np.maximum(extent, 1e-3)
    volume = float(np.prod(extent))
    if not math.isfinite(volume) or volume <= 0:
        return 1.0

    density = float(target_count) / volume
    if density <= 0:
        return 1.0

    spacing = (1.0 / density) ** (1.0 / 3.0)
    return float(max(spacing, 1e-3))


def _voxel_downsample_numpy(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    voxel_size: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if voxel_size is None or voxel_size <= 0:
        return points.copy(), None if colors is None else colors.copy()

    grid_indices = np.floor(points / voxel_size).astype(np.int64)
    unique_voxels, inverse = np.unique(grid_indices, axis=0, return_inverse=True)

    counts = np.bincount(inverse)
    down_points = np.zeros((len(unique_voxels), 3), dtype=np.float64)
    np.add.at(down_points, inverse, points)
    down_points /= counts[:, None]

    if colors is None:
        down_colors = None
    else:
        down_colors = np.zeros((len(unique_voxels), 3), dtype=np.float64)
        np.add.at(down_colors, inverse, colors)
        down_colors /= counts[:, None]

    return down_points.astype(points.dtype), None if down_colors is None else down_colors.astype(colors.dtype)


def voxel_downsample(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    voxel_size: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if len(points) == 0:
        return points.copy(), None if colors is None else colors.copy()

    if o3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        down = pcd.voxel_down_sample(float(voxel_size))
        down_points = np.asarray(down.points)
        down_colors = None
        if colors is not None and down.has_colors():
            down_colors = np.asarray(down.colors)
        return down_points, down_colors

    return _voxel_downsample_numpy(points, colors, voxel_size)


def farthest_point_sampling(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    num_samples: int,
    seed: int = 0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    num_points = len(points)
    if num_points == 0 or num_samples is None or num_samples <= 0 or num_samples >= num_points:
        return points.copy(), None if colors is None else colors.copy()

    rng = np.random.default_rng(seed)
    chosen = np.empty(num_samples, dtype=np.int64)

    if o3d is not None and hasattr(o3d.geometry.PointCloud, "farthest_point_down_sample"):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        down = pcd.farthest_point_down_sample(num_samples)
        down_points = np.asarray(down.points)
        down_colors = None
        if colors is not None and down.has_colors():
            down_colors = np.asarray(down.colors)
        return down_points, down_colors

    chosen[0] = int(rng.integers(num_points))
    distances = np.linalg.norm(points - points[chosen[0]], axis=1)
    for i in range(1, num_samples):
        next_idx = int(np.argmax(distances))
        chosen[i] = next_idx
        new_distances = np.linalg.norm(points - points[next_idx], axis=1)
        distances = np.minimum(distances, new_distances)

    sampled_points = points[chosen]
    sampled_colors = None if colors is None else colors[chosen]
    return sampled_points, sampled_colors


def poisson_disk_sampling(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    radius: float,
    *,
    seed: int = 0,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if len(points) == 0 or radius is None or radius <= 0:
        return points.copy(), None if colors is None else colors.copy()

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(points))

    cell_size = float(radius) / math.sqrt(3.0)
    if cell_size <= 0:
        cell_size = float(radius)
    inv_cell = 1.0 / cell_size
    radius_sq = float(radius) ** 2

    grid: Dict[Tuple[int, int, int], int] = {}
    selected_indices = []

    for idx in order:
        p = points[idx]
        cell = tuple(np.floor(p * inv_cell).astype(np.int64))

        accept = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    neighbor_idx = grid.get(neighbor)
                    if neighbor_idx is None:
                        continue
                    if np.sum((points[neighbor_idx] - p) ** 2) < radius_sq:
                        accept = False
                        break
                if not accept:
                    break
            if not accept:
                break

        if not accept:
            continue

        grid[cell] = idx
        selected_indices.append(idx)
        if max_samples is not None and len(selected_indices) >= max_samples:
            break

    sampled_points = points[selected_indices]
    sampled_colors = None if colors is None else colors[selected_indices]
    return sampled_points, sampled_colors


def adaptive_voxel_downsample(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    target_count: int,
    initial_size: float,
    *,
    tolerance: float = 0.05,
    max_iter: int = 12,
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    if target_count <= 0 or len(points) == 0 or target_count >= len(points):
        return points.copy(), None if colors is None else colors.copy(), float(max(initial_size, 1e-3))

    voxel_size = float(max(initial_size, 1e-3))
    min_size = 1e-3
    max_size = float(np.max(np.ptp(points, axis=0)) + 1e-3)
    tol_count = max(1, int(target_count * tolerance))

    best_pts = points
    best_cols = colors
    best_size = voxel_size
    best_err = abs(len(points) - target_count)

    for _ in range(max_iter):
        sampled_pts, sampled_cols = voxel_downsample(points, colors, voxel_size)
        count = len(sampled_pts)
        err = abs(count - target_count)
        if err < best_err:
            best_pts, best_cols = sampled_pts, sampled_cols
            best_err = err
            best_size = voxel_size

        if err <= tol_count:
            return sampled_pts, sampled_cols, voxel_size

        if count == 0:
            voxel_size = max(voxel_size * 0.5, min_size)
            continue

        if count > target_count:
            voxel_size = min(voxel_size * 1.35, max_size)
        else:
            voxel_size = max(voxel_size * 0.75, min_size)

    return best_pts, best_cols, best_size


def adaptive_poisson_disk_sampling(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    target_count: int,
    initial_radius: float,
    *,
    max_samples: Optional[int] = None,
    tolerance: float = 0.05,
    max_iter: int = 16,
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    if target_count <= 0 or len(points) == 0 or target_count >= len(points):
        return points.copy(), None if colors is None else colors.copy(), float(max(initial_radius, 1e-3))

    radius = float(max(initial_radius, 1e-3))
    min_radius = radius * 0.05
    max_radius = max(radius * 5.0, radius + 1e-3)
    tol_count = max(1, int(target_count * tolerance))

    if max_samples is None:
        max_samples = min(int(target_count * 2), len(points))

    best_pts = points
    best_cols = colors
    best_radius = radius
    best_err = abs(len(points) - target_count)

    for _ in range(max_iter):
        sampled_pts, sampled_cols = poisson_disk_sampling(
            points,
            colors,
            radius,
            max_samples=max_samples,
        )

        count = len(sampled_pts)
        err = abs(count - target_count)
        if err < best_err and count > 0:
            best_pts, best_cols = sampled_pts, sampled_cols
            best_radius = radius
            best_err = err

        if err <= tol_count and count > 0:
            return sampled_pts, sampled_cols, radius

        if count == 0:
            radius = max(radius * 0.6, min_radius)
            continue

        if count > target_count:
            radius = min(radius * 1.35, max_radius)
        else:
            radius = max(radius * 0.75, min_radius)

    return best_pts, best_cols, best_radius


def compute_density_metrics(points: np.ndarray, *, k: int = 8) -> Dict[str, object]:
    """Compute simple density metrics based on nearest-neighbour spacing."""

    metrics: Dict[str, object] = {
        "count": int(len(points)),
        "valid": False,
        "k_neighbors": int(max(1, min(k, max(0, len(points) - 1)))),
    }

    if len(points) <= 1:
        metrics["reason"] = "insufficient-points"
        return metrics

    extent = np.ptp(points, axis=0)
    volume = float(np.prod(np.maximum(extent, 1e-6)))
    metrics["aabb_volume"] = volume
    metrics["points_per_unit_volume"] = float(metrics["count"]) / volume if volume > 0 else float("inf")

    if NearestNeighbors is None:
        metrics["reason"] = "sklearn-not-installed"
        return metrics

    k_use = int(metrics["k_neighbors"])
    if k_use < 1:
        metrics["reason"] = "insufficient-neighbours"
        return metrics

    nn = NearestNeighbors(n_neighbors=k_use + 1)
    nn.fit(points)
    distances, _ = nn.kneighbors(points, return_distance=True)
    nn_distances = distances[:, 1]

    metrics["mean_nn_distance"] = float(np.mean(nn_distances))
    metrics["median_nn_distance"] = float(np.median(nn_distances))
    metrics["std_nn_distance"] = float(np.std(nn_distances))
    metrics["valid"] = True
    return metrics


def write_point_cloud_comparison(
    output_path: Path,
    point_sets: Sequence[Tuple[str, np.ndarray, Optional[np.ndarray]]],
    *,
    point_size: int = 3,
    max_points: int = 80_000,
) -> None:
    """Render multiple point clouds into a single Plotly HTML visualization."""

    if go is None:
        print("Plotly is unavailable; skipping interactive visualization.")
        return

    fig = go.Figure()

    palette = {
        "Voxel downsample": "rgb(0, 200, 70)",  # green
        "Farthest point": "rgb(255, 105, 180)",  # pink
        "Poisson disk": "rgb(30, 144, 255)",  # blue
    }

    for label, pts, cols in point_sets:
        if pts is None or len(pts) == 0:
            continue

        if max_points is not None and len(pts) > max_points:
            idx = np.random.choice(len(pts), max_points, replace=False)
            pts = pts[idx]
            if cols is not None:
                cols = cols[idx]

        if label == "Raw foreground" and cols is not None and cols.size:
            if cols.max() > 1.0:
                cols = cols / 255.0
            cols = np.clip(cols, 0.0, 1.0)
            color_strings = [
                f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in cols
            ]
        else:
            color_strings = palette.get(label, "rgb(200,200,200)")

        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                name=label,
                marker=dict(size=point_size, color=color_strings, opacity=0.85),
            )
        )

    fig.update_layout(
        title="Wire Downsampling Comparison",
        scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn")


def main() -> None:
    data_root = Path(__file__).resolve().parent
    full_data_path = data_root / "data/full/test_wire.npy"
    bg_data_path = data_root / "data/bg/test_wire_bg.npy"

    if not full_data_path.exists():
        raise FileNotFoundError(f"Foreground wire data not found at {full_data_path}")
    if not bg_data_path.exists():
        raise FileNotFoundError(f"Background wire data not found at {bg_data_path}")

    print("Loading wire foreground and background frames...")
    full_data = np.load(full_data_path, allow_pickle=True).item()
    bg_data = np.load(bg_data_path, allow_pickle=True).item()

    frame = full_data[0]
    bg_frame = bg_data[0]

    intrinsics = np.array(
        [
            [606.1124267578125, 0.0, 641.7578125],
            [0.0, 605.8821411132812, 365.6518859863281],
            [0.0, 0.0, 1.0],
        ]
    )

    print("Computing depth-based foreground mask...")
    pc_mask = compute_point_cloud_mask(
        bg_frame["transformed_depth"],
        frame["transformed_depth"],
        intrinsics,
        distance_threshold=18.0,
    )

    print("Refining point-cloud mask with DBSCAN clustering...")
    pc_mask = filter_pcd_mask_dbscan(
        pc_mask,
        frame["transformed_depth"],
        intrinsics,
        eps=30.0,
        min_samples=18,
    )

    rgb_image = frame["color"][:, :, ::-1]
    depth_image = frame["transformed_depth"]

    print("Lifting RGB-D frame to 3D point cloud...")
    points, colors, valid_mask = create_color_point_cloud(
        rgb_image,
        depth_image,
        intrinsics,
        return_valid_mask=True,
    )

    valid_flat = valid_mask.reshape(-1)
    valid_indices = np.flatnonzero(valid_flat)
    mask_flat = (pc_mask > 128).reshape(-1)

    if valid_indices.size == 0:
        raise RuntimeError("No valid depth pixels were found in the frame")

    foreground_mask = mask_flat[valid_indices].astype(bool)
    foreground_points = points[foreground_mask]
    foreground_colors = colors[foreground_mask]

    print(
        "Foreground extraction complete: {fg:,} of {total:,} lifted points".format(
            fg=len(foreground_points), total=len(points)
        )
    )

    if len(foreground_points) == 0:
        raise RuntimeError("No foreground points detected after masking")

    target_points = min(1600, len(foreground_points))
    uniform_spacing = estimate_uniform_spacing(foreground_points, target_points)

    print("Estimating adaptive voxel size...")
    voxel_points, voxel_colors, voxel_size = adaptive_voxel_downsample(
        foreground_points,
        foreground_colors,
        target_points,
        uniform_spacing,
        tolerance=0.05,
        max_iter=14,
    )

    fps_target = target_points
    print(f"Running farthest point sampling (target={fps_target})...")
    fps_points, fps_colors = farthest_point_sampling(foreground_points, foreground_colors, fps_target)

    poisson_max_samples = min(target_points * 2, len(foreground_points))

    print("Estimating adaptive Poisson radius...")
    poisson_points, poisson_colors, poisson_radius = adaptive_poisson_disk_sampling(
        foreground_points,
        foreground_colors,
        target_points,
        max(uniform_spacing * 0.9, 1e-3),
        max_samples=poisson_max_samples,
        tolerance=0.05,
        max_iter=18,
    )

    output_dir = data_root / "wire_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    density_metrics = {
        "raw": compute_density_metrics(foreground_points),
        "voxel": compute_density_metrics(voxel_points),
        "farthest_point": compute_density_metrics(fps_points),
        "poisson_disk": compute_density_metrics(poisson_points),
        "_config": {
            "target_points": int(target_points),
            "initial_spacing": float(uniform_spacing),
            "voxel_size": float(voxel_size),
            "fps_target": int(fps_target),
            "poisson_radius": float(poisson_radius),
            "poisson_max_samples": int(poisson_max_samples),
        },
    }

    density_metrics["_notes"] = {
        "mean_nn_distance": "Average distance to each point's nearest neighbour; lower implies denser sampling.",
        "points_per_unit_volume": "Number of points divided by the axis-aligned bounding-box volume.",
    }

    baseline_mean = density_metrics["raw"].get("mean_nn_distance")

    metrics_path = output_dir / "wire_downsampling_metrics.json"
    metrics_path.write_text(json.dumps(density_metrics, indent=2), encoding="utf-8")

    np.savez_compressed(
        output_dir / "wire_downsampling_results.npz",
        original_points=foreground_points.astype(np.float32),
        original_colors=foreground_colors.astype(np.float32),
        voxel_points=voxel_points.astype(np.float32),
        voxel_colors=(
            np.zeros((0, 3), dtype=np.float32)
            if voxel_colors is None
            else voxel_colors.astype(np.float32)
        ),
        fps_points=fps_points.astype(np.float32),
        fps_colors=(
            np.zeros((0, 3), dtype=np.float32)
            if fps_colors is None
            else fps_colors.astype(np.float32)
        ),
        poisson_points=poisson_points.astype(np.float32),
        poisson_colors=(
            np.zeros((0, 3), dtype=np.float32)
            if poisson_colors is None
            else poisson_colors.astype(np.float32)
        ),
    )

    plot_path = output_dir / "wire_downsampling_comparison.html"
    write_point_cloud_comparison(
        plot_path,
        [
            ("Raw foreground", foreground_points, foreground_colors),
            ("Voxel downsample", voxel_points, voxel_colors),
            ("Farthest point", fps_points, fps_colors),
            ("Poisson disk", poisson_points, poisson_colors),
        ],
        point_size=3,
        max_points=90_000,
    )

    summary_lines = [
        "Wire downsampling summary:",
        f"  Original foreground points: {len(foreground_points):,}",
        f"  Voxel downsampled points:  {len(voxel_points):,} (voxel_size={voxel_size:.3f})",
        f"  Farthest point sample:     {len(fps_points):,} (target={fps_target})",
        f"  Poisson-disk sample:       {len(poisson_points):,} (radius={poisson_radius:.3f}, max={poisson_max_samples})",
    ]

    summary_lines.append("Density metrics (mean NN distance; lower = denser):")
    for label, key in (
        ("  Raw foreground", "raw"),
        ("  Voxel downsample", "voxel"),
        ("  Farthest point", "farthest_point"),
        ("  Poisson disk", "poisson_disk"),
    ):
        metrics = density_metrics[key]
        if metrics.get("valid") and baseline_mean and baseline_mean > 0:
            ratio = float(metrics["mean_nn_distance"]) / float(baseline_mean)
            density = metrics.get("points_per_unit_volume")
            density_str = (
                f", density={density:.2f}/vol"
                if isinstance(density, (float, int)) and math.isfinite(density)
                else ""
            )
            summary_lines.append(
                f"{label}: mean={metrics['mean_nn_distance']:.3f}, ratio vs raw={ratio:.2f}x{density_str}"
            )
        elif metrics.get("valid"):
            density = metrics.get("points_per_unit_volume")
            density_str = (
                f", density={density:.2f}/vol"
                if isinstance(density, (float, int)) and math.isfinite(density)
                else ""
            )
            summary_lines.append(
                f"{label}: mean={metrics['mean_nn_distance']:.3f}{density_str}"
            )
        else:
            reason = metrics.get("reason", "unavailable")
            summary_lines.append(f"{label}: metric unavailable ({reason})")

    summary_lines.append(
        "  mean_nn_distance = average nearest-neighbour spacing; points_per_unit_volume = density over bounding box volume."
    )

    summary_path = output_dir / "wire_downsampling_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"Metrics saved to {metrics_path}")
    if go is not None:
        print(f"Interactive comparison saved to {plot_path}")
    else:
        print("Plotly not installed; skipped interactive comparison.")
    print(f"Results saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
