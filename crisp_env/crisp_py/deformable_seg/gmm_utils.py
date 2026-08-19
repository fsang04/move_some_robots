"""Utilities for extracting keypoints from point clouds via Gaussian mixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Set, List

import heapq
import math
import numpy as np  # type: ignore[import]

try:  # pragma: no cover - optional dependency
	import open3d as o3d  # type: ignore[import]
except ImportError:  # pragma: no cover - handled gracefully at runtime
	o3d = None  # type: ignore

try:  # pragma: no cover - optional dependency
	from sklearn.mixture import GaussianMixture  # type: ignore[import]
	from sklearn.neighbors import KDTree  # type: ignore[import]
except ImportError:  # pragma: no cover - handled gracefully at runtime
	GaussianMixture = None  # type: ignore
	KDTree = None  # type: ignore

# from seg_utils import (
#     compute_rgb_mask,
#     compute_point_cloud_mask,
#     refine_rgb_mask_dbscan,
#     filter_pcd_mask_dbscan,
#     remove_small_components,
#     filter_point_cloud_radius,
# )

__all__ = [
	"GMMExtractionResult",
	"extract_gmm_keypoints",
	"classify_deformable_topology",
	"build_wire_connections",
	"refine_keypoints_uniform_edges",
]


@dataclass
class GMMExtractionResult:
	"""Container for Gaussian mixture keypoint extraction output."""

	keypoints: np.ndarray
	gmm: Any
	filtered_points: np.ndarray
	filtered_colors: Optional[np.ndarray]
	labels: np.ndarray
	responsibilities: np.ndarray
	weights: np.ndarray
	covariances: np.ndarray
	metadata: Dict[str, Any]


def _ensure_points_array(points: np.ndarray) -> np.ndarray:
	arr = np.asarray(points, dtype=np.float64)
	if arr.ndim != 2 or arr.shape[1] != 3:
		raise ValueError(
			"`points` must be an array-like of shape (N, 3) representing XYZ coordinates."
		)
	return arr


def _ensure_colors_array(points: np.ndarray, colors: Optional[np.ndarray]) -> Optional[np.ndarray]:
	if colors is None:
		return None
	arr = np.asarray(colors, dtype=np.float64)
	if arr.shape != points.shape:
		raise ValueError(
			"When provided, `colors` must have the same shape as `points` (N, 3)."
		)
	# Normalize to [0, 1] if values look like 0-255 uint8s
	if arr.max() > 1.0:
		arr = arr / 255.0
	return arr


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


def _estimate_uniform_spacing(points: np.ndarray, target_count: int) -> float:
	if target_count <= 0 or points.size == 0:
		return 1.0

	if KDTree is not None and len(points) > 8:
		tree = KDTree(points)
		k = min(9, len(points))
		distances, _ = tree.query(points, k=k)
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


def _farthest_point_sampling(
	points: np.ndarray,
	colors: Optional[np.ndarray],
	num_samples: int,
	*,
	seed: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
	num_points = len(points)
	if num_points == 0 or num_samples is None or num_samples <= 0 or num_samples >= num_points:
		return points.copy(), None if colors is None else colors.copy()

	rng = np.random.default_rng(seed)
	chosen = np.empty(num_samples, dtype=np.int64)
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


def _poisson_disk_sampling(
	points: np.ndarray,
	colors: Optional[np.ndarray],
	radius: float,
	*,
	seed: Optional[int] = None,
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
	selected: List[int] = []

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
		selected.append(idx)
		if max_samples is not None and len(selected) >= max_samples:
			break

	sampled_points = points[selected]
	sampled_colors = None if colors is None else colors[selected]
	return sampled_points, sampled_colors


def _downsample_with_method(
	points: np.ndarray,
	colors: Optional[np.ndarray],
	*,
	method: str,
	target_count: Optional[int],
	seed: Optional[int],
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
	method_name = method.lower().strip()
	metadata: Dict[str, Any] = {
		"downsample_method": method_name,
	}

	if target_count is None or target_count <= 0:
		target = len(points)
		requested_target: Optional[int] = None
	else:
		target = int(np.clip(target_count, 1, len(points)))
		requested_target = target

	if method_name == "voxel":
		estimated = _estimate_uniform_spacing(points, max(target, 1))
		effective_voxel = float(max(estimated, 1e-6))
		max_iterations = 1 if requested_target is None else 8
		best_points: Optional[np.ndarray] = None
		best_colors: Optional[np.ndarray] = None
		best_voxel = effective_voxel
		best_error = float("inf")
		iterations_performed = 0
		for _ in range(max_iterations):
			iterations_performed += 1
			down_points, down_colors = _voxel_downsample_numpy(points, colors, effective_voxel)
			current_count = len(down_points)
			if requested_target is None:
				best_points = down_points
				best_colors = down_colors
				best_voxel = effective_voxel
				best_error = 0.0
				break
			if requested_target <= 0:
				best_points = down_points
				best_colors = down_colors
				best_voxel = effective_voxel
				best_error = 0.0
				break
			error = abs(current_count - requested_target)
			if error < best_error:
				best_points = down_points
				best_colors = down_colors
				best_voxel = effective_voxel
				best_error = error
			if current_count == 0:
				effective_voxel = max(effective_voxel * 0.5, 1e-6)
				continue
			ratio = current_count / float(requested_target)
			if 0.9 <= ratio <= 1.1:
				break
			adjust = float(np.clip(ratio ** (1.0 / 3.0), 0.25, 4.0))
			effective_voxel = max(effective_voxel * adjust, 1e-6)
		if best_points is None:
			best_points, best_colors = down_points, down_colors
			best_voxel = effective_voxel
		metadata["voxel_size"] = float(best_voxel)
		metadata["downsampled_point_count"] = len(best_points)
		metadata["voxel_iterations"] = iterations_performed
		if requested_target is not None:
			metadata["target_count"] = requested_target
		return best_points, best_colors, metadata

	if requested_target is not None:
		metadata["target_count"] = requested_target

	if method_name in {"farthest_point", "fps"}:
		down_points, down_colors = _farthest_point_sampling(points, colors, target, seed=seed)
		metadata["downsampled_point_count"] = len(down_points)
		return down_points, down_colors, metadata

	if method_name == "poisson":
		radius = _estimate_uniform_spacing(points, max(target, 1))
		effective_radius = float(max(radius * 0.9, 1e-6))
		max_iterations = 1 if requested_target is None else 8
		best_points: Optional[np.ndarray] = None
		best_colors: Optional[np.ndarray] = None
		best_radius = effective_radius
		best_error = float("inf")
		iterations_performed = 0
		for _ in range(max_iterations):
			iterations_performed += 1
			down_points, down_colors = _poisson_disk_sampling(
				points,
				colors,
				effective_radius,
				seed=seed,
			)
			current_count = len(down_points)
			if requested_target is None:
				best_points = down_points
				best_colors = down_colors
				best_radius = effective_radius
				best_error = 0.0
				break
			if requested_target <= 0:
				best_points = down_points
				best_colors = down_colors
				best_radius = effective_radius
				best_error = 0.0
				break
			error = abs(current_count - requested_target)
			if error < best_error:
				best_points = down_points
				best_colors = down_colors
				best_radius = effective_radius
				best_error = error
			if current_count == 0:
				effective_radius = max(effective_radius * 0.5, 1e-6)
				continue
			ratio = current_count / float(requested_target)
			if 0.9 <= ratio <= 1.1:
				break
			adjust = float(np.clip(ratio ** (1.0 / 3.0), 0.25, 4.0))
			# For Poisson disk sampling, increase radius when we have too many points (ratio>1)
			# and decrease when we need more points (ratio<1).
			if ratio > 1.0:
				effective_radius = max(effective_radius * adjust, 1e-6)
			else:
				effective_radius = max(effective_radius / max(adjust, 1e-6), 1e-6)
		if best_points is None:
			best_points, best_colors = down_points, down_colors
			best_radius = effective_radius
		if requested_target is not None and requested_target > 0 and len(best_points) > requested_target:
			rng = np.random.default_rng(seed)
			chosen = rng.choice(len(best_points), size=requested_target, replace=False)
			best_points = best_points[chosen]
			if best_colors is not None:
				best_colors = best_colors[chosen]
		metadata.update(
			{
				"poisson_initial_radius": float(radius * 0.9),
				"poisson_final_radius": float(best_radius),
				"poisson_iterations": iterations_performed,
				"downsampled_point_count": len(best_points),
			}
		)
		if requested_target is not None:
			metadata["target_count"] = requested_target
		return best_points, best_colors, metadata

	raise ValueError(f"Unsupported downsampling method: {method}")


def _statistical_outlier_removal_numpy(
	points: np.ndarray,
	colors: Optional[np.ndarray],
	nb_neighbors: int,
	std_ratio: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
	if nb_neighbors is None or nb_neighbors < 1 or std_ratio is None or std_ratio <= 0:
		inlier_mask = np.ones(points.shape[0], dtype=bool)
		return points, colors, inlier_mask

	if KDTree is None:  # pragma: no cover - dependency missing
		raise ImportError(
			"scikit-learn is required for statistical outlier removal when Open3D is unavailable."
		)

	nb_neighbors = min(nb_neighbors, points.shape[0] - 1)
	if nb_neighbors < 1:
		inlier_mask = np.ones(points.shape[0], dtype=bool)
		return points, colors, inlier_mask

	tree = KDTree(points)
	distances, _ = tree.query(points, k=nb_neighbors + 1)
	neighbor_distances = distances[:, 1:]  # exclude distance to self
	mean_dist = np.mean(neighbor_distances, axis=1)

	distance_mean = float(np.mean(mean_dist))
	distance_std = float(np.std(mean_dist))
	threshold = distance_mean + std_ratio * distance_std
	inlier_mask = mean_dist <= threshold

	filtered_points = points[inlier_mask]
	filtered_colors = None if colors is None else colors[inlier_mask]
	return filtered_points, filtered_colors, inlier_mask


def _preprocess_point_cloud(
	points: np.ndarray,
	colors: Optional[np.ndarray],
	nb_neighbors: int,
	std_ratio: float,
	*,
	downsample_method: str = "voxel",
	target_count: Optional[int] = None,
	random_state: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
	metadata: Dict[str, Any] = {
		"nb_neighbors": nb_neighbors,
		"std_ratio": std_ratio,
		"downsample_method": downsample_method,
	}

	down_points, down_colors, ds_metadata = _downsample_with_method(
		points,
		colors,
		method=downsample_method,
		target_count=target_count,
		seed=random_state,
	)
	metadata.update(ds_metadata)

	if o3d is not None:
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(down_points)
		if down_colors is not None:
			pcd.colors = o3d.utility.Vector3dVector(down_colors)

		if nb_neighbors is not None and nb_neighbors > 0 and std_ratio is not None and std_ratio > 0:
			pcd, inlier_indices = pcd.remove_statistical_outlier(
				nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio)
			)
			metadata["inlier_count"] = len(inlier_indices)
		else:
			metadata["inlier_count"] = len(pcd.points)

		filtered_points = np.asarray(pcd.points, dtype=np.float64)
		if pcd.has_colors():
			filtered_colors = np.asarray(pcd.colors, dtype=np.float64)
		else:
			filtered_colors = down_colors
		metadata["preprocess_backend"] = "open3d"
		return filtered_points, filtered_colors, metadata

	filtered_points, filtered_colors, inlier_mask = _statistical_outlier_removal_numpy(
		down_points, down_colors, nb_neighbors, std_ratio
	)
	metadata["inlier_count"] = int(np.sum(inlier_mask))
	metadata["preprocess_backend"] = "numpy"
	return filtered_points, filtered_colors, metadata


def extract_gmm_keypoints(
	points: np.ndarray,
	colors: Optional[np.ndarray] = None,
	*,
	nb_neighbors: int = 30,
	std_ratio: float = 2.0,
	downsample_method: str = "voxel",
	target_count: Optional[int] = None,
	n_components: int = 20,
	covariance_type: str = "full",
	random_state: Optional[int] = None,
	reg_covar: float = 1e-6,
	max_iter: int = 200,
	tol: float = 1e-3,
	n_init: int = 1,
) -> GMMExtractionResult:
	"""Preprocess a point cloud and fit a Gaussian mixture to extract keypoints.

	Args:
		points: (N, 3) array-like of XYZ coordinates.
		colors: Optional (N, 3) array-like of RGB values aligned with ``points``.
		nb_neighbors: Number of neighbors used for statistical outlier removal.
		std_ratio: Threshold in standard deviations for outlier rejection.
		downsample_method: Strategy for the initial downsampling step (``"voxel"``, ``"fps"``, or ``"poisson"``).
		target_count: Desired number of points to retain during downsampling. If omitted, the entire set is used.
		n_components: Desired number of Gaussian components (keypoints).
		covariance_type: Covariance model passed to :class:`GaussianMixture`.
		random_state: RNG seed for deterministic fitting.
		reg_covar: Non-negative regularization added to the diagonal of covariances.
		max_iter: Maximum EM iterations for GMM fitting.
		tol: Convergence tolerance for the EM algorithm.
		n_init: Number of initializations for GMM fitting.

	Returns:
		``GMMExtractionResult`` containing the fitted model, keypoints, and metadata.
	"""

	if GaussianMixture is None:  # pragma: no cover - dependency missing
		raise ImportError(
			"scikit-learn is required to fit a Gaussian Mixture Model. Install it via `pip install scikit-learn`."
		)

	points_arr = _ensure_points_array(points)
	colors_arr = _ensure_colors_array(points_arr, colors)

	if points_arr.shape[0] == 0:
		raise ValueError("No points provided for GMM extraction.")

	filtered_points, filtered_colors, metadata = _preprocess_point_cloud(
		points_arr,
		colors_arr,
		nb_neighbors,
		std_ratio,
		downsample_method=downsample_method,
		target_count=target_count,
		random_state=random_state,
	)
	# filter_point_cloud_radius(
    #     filtered_points, 
    #     filtered_colors, 
    #     radius=10.0, 
    #     min_neighbors=100
    # )

	if filtered_points.shape[0] == 0:
		raise ValueError(
			"No points remain after preprocessing. Consider relaxing the downsampling or outlier parameters."
		)

	effective_components = int(np.clip(n_components, 1, filtered_points.shape[0]))
	if effective_components != n_components:
		metadata["effective_components"] = effective_components
	else:
		metadata["effective_components"] = n_components

	gmm = GaussianMixture(
		n_components=effective_components,
		covariance_type=covariance_type,
		random_state=random_state,
		reg_covar=reg_covar,
		max_iter=max_iter,
		tol=tol,
		n_init=n_init,
	)
	gmm.fit(filtered_points)

	responsibilities = gmm.predict_proba(filtered_points)
	labels = responsibilities.argmax(axis=1)

	metadata.update(
		{
			"n_original_points": int(points_arr.shape[0]),
			"n_filtered_points": int(filtered_points.shape[0]),
			"converged": bool(getattr(gmm, "converged_", True)),
			"lower_bound": float(getattr(gmm, "lower_bound_", 0.0)),
			"n_iter": int(getattr(gmm, "n_iter_", 0)),
			"covariance_type": covariance_type,
		}
	)

	return GMMExtractionResult(
		keypoints=gmm.means_.copy(),
		gmm=gmm,
		filtered_points=filtered_points,
		filtered_colors=filtered_colors,
		labels=labels,
		responsibilities=responsibilities,
		weights=gmm.weights_.copy(),
		covariances=gmm.covariances_.copy(),
		metadata=metadata,
	)


def _get_component_covariance(
	covariances: np.ndarray,
	covariance_type: str,
	index: int,
) -> np.ndarray:
	"""Return the covariance matrix for a specific GMM component."""

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


def classify_deformable_topology(
	gmm_result: GMMExtractionResult,
	foreground_points: np.ndarray,
	*,
	k_neighbors: int = 64,
	neighbor_radius: Optional[float] = None,
	linear_ratio_threshold: float = 0.15,
	planar_ratio_threshold: float = 0.3,
	planar_thickness_threshold: float = 0.1,
	min_neighbors: int = 12,
	ellipsoid_std_multiplier: float = 2.5,
) -> Dict[str, Any]:
	"""Classify whether a deformable object is wire-like (1D) or fabric-like (2D).

	This utility inspects the local covariance structure of the foreground point
	cloud around each GMM keypoint. For each component, it gathers points lying
	inside the corresponding Gaussian ellipsoid (within
	``ellipsoid_std_multiplier`` standard deviations), recomputes a local
	covariance via PCA, and compares the relative magnitudes of its eigenvalues to
	determine whether the neighborhood is predominantly linear or planar.

	Args:
		gmm_result: Output from :func:`extract_gmm_keypoints` containing keypoints.
		foreground_points: (N, 3) array-like of XYZ points representing the
			foreground object.
		k_neighbors: Number of nearest neighbours to include when a radius query is
			not supplied or yields too few samples.
		neighbor_radius: Optional radius (same units as the point cloud) used to
			gather neighbours. If provided, the radius neighbourhood is attempted
			first before falling back to ``k_neighbors``.
		linear_ratio_threshold: Upper bound on the ratio of the second to first
			eigenvalue for a neighbourhood to be considered 1D.
		planar_ratio_threshold: Lower bound on the ratio of the second to first
			eigenvalue for planar (2D) structures.
		planar_thickness_threshold: Upper bound on the ratio of the third to first
			eigenvalue for planar structures (controls surface "thickness").
		min_neighbors: Minimum number of neighbours required to attempt a PCA.
		ellipsoid_std_multiplier: Multiplier applied to the Mahalanobis distance for
			selecting inlier points inside each Gaussian ellipsoid. Larger values
			include more points from the component; smaller values tighten the
			neighborhood.

	Returns:
		Dictionary containing the overall topology label (``wireharness_1d``,
		``fabric_2d``, or ``ambiguous``) along with per-keypoint diagnostics.

	Raises:
		ValueError: If the provided foreground point cloud is empty or malformed.
	"""

	points_arr = _ensure_points_array(foreground_points)
	if points_arr.shape[0] == 0:
		raise ValueError("Foreground point cloud must contain at least one point.")

	keypoints = _ensure_points_array(gmm_result.keypoints)
	if keypoints.shape[0] == 0:
		return {
			"topology": "ambiguous",
			"reason": "no-keypoints",
			"keypoint_labels": [],
			"eigenvalues": [],
			"parameters": {
				"k_neighbors": k_neighbors,
				"neighbor_radius": neighbor_radius,
				"linear_ratio_threshold": linear_ratio_threshold,
				"planar_ratio_threshold": planar_ratio_threshold,
				"planar_thickness_threshold": planar_thickness_threshold,
				"min_neighbors": min_neighbors,
				"ellipsoid_std_multiplier": ellipsoid_std_multiplier,
			},
		}

	if k_neighbors is None or k_neighbors < 3:
		k_neighbors = 3
	else:
		k_neighbors = int(k_neighbors)

	if min_neighbors < 3:
		min_neighbors = 3

	if ellipsoid_std_multiplier is None or ellipsoid_std_multiplier <= 0:
		raise ValueError("ellipsoid_std_multiplier must be positive.")

	points_source = getattr(gmm_result, "filtered_points", None)
	if points_source is None or np.asarray(points_source).size == 0:
		points_source = points_arr
	else:
		points_source = _ensure_points_array(points_source)

	labels = getattr(gmm_result, "labels", None)
	if labels is not None:
		labels = np.asarray(labels)
		if labels.shape[0] != points_source.shape[0]:
			labels = None

	covariances = np.asarray(gmm_result.covariances)
	covariance_type = str(gmm_result.metadata.get("covariance_type", "full"))

	tree = KDTree(points_arr) if KDTree is not None else None

	keypoint_labels = []
	eigenvalues_list = []
	neighbor_counts = []
	neighbor_sources = []

	def _fallback_neighbors(point: np.ndarray) -> np.ndarray:
		indices = np.array([], dtype=np.int64)
		if neighbor_radius is not None and neighbor_radius > 0:
			if tree is not None:
				indices = tree.query_radius(point.reshape(1, -1), r=float(neighbor_radius))[0]
			else:
				dists = np.linalg.norm(points_arr - point[None, :], axis=1)
				indices = np.where(dists <= float(neighbor_radius))[0]
		if indices.size < min_neighbors:
			k = min(k_neighbors, points_arr.shape[0])
			if tree is not None:
				_, idx = tree.query(point.reshape(1, -1), k=k)
				indices = idx[0]
			else:
				dists = np.linalg.norm(points_arr - point[None, :], axis=1)
				indices = np.argsort(dists)[:k]
		return points_arr[indices]

	for comp_idx, kp in enumerate(keypoints):
		source_points = points_source
		if labels is not None:
			mask = labels == comp_idx
			if np.any(mask):
				source_points = source_points[mask]
			else:
				source_points = np.empty((0, 3), dtype=np.float64)

		neighbors = np.empty((0, 3), dtype=np.float64)
		source_used = "none"

		if source_points.shape[0] > 0:
			cov_matrix = _get_component_covariance(covariances, covariance_type, comp_idx)
			cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
			if cov_matrix.shape != (3, 3):
				raise ValueError("Component covariance must be a 3x3 matrix.")
			cov_matrix = (cov_matrix + cov_matrix.T) * 0.5
			cov_matrix += np.eye(3, dtype=np.float64) * 1e-8
			try:
				inv_cov = np.linalg.inv(cov_matrix)
			except np.linalg.LinAlgError:
				inv_cov = np.linalg.pinv(cov_matrix)

			diff = source_points - kp
			dist_sq = np.einsum("ni,ij,nj->n", diff, inv_cov, diff)
			inside_mask = dist_sq <= float(ellipsoid_std_multiplier) ** 2
			neighbors = source_points[inside_mask]
			source_used = "ellipsoid"

			if neighbors.shape[0] < min_neighbors and source_points.shape[0] > 0:
				neighbors = source_points
				source_used = "component"

		if neighbors.shape[0] < min_neighbors:
			fallback = _fallback_neighbors(kp)
			if fallback.size > 0:
				neighbors = fallback
				source_used = "fallback"

		neighbor_counts.append(int(neighbors.shape[0]))
		neighbor_sources.append(source_used)
		if neighbors.shape[0] < min_neighbors:
			keypoint_labels.append("insufficient")
			eigenvalues_list.append(np.array([np.nan, np.nan, np.nan]))
			continue

		centered = neighbors - np.mean(neighbors, axis=0, keepdims=True)
		cov = centered.T @ centered / max(neighbors.shape[0] - 1, 1)
		cov = (cov + cov.T) * 0.5
		eigvals = np.linalg.eigvalsh(cov)
		eigvals = np.sort(np.clip(eigvals, 0.0, None))[::-1]
		eigenvalues_list.append(eigvals)

		if eigvals[0] <= 0:
			keypoint_labels.append("degenerate")
			continue

		ratio2 = eigvals[1] / eigvals[0]
		ratio3 = eigvals[2] / eigvals[0]
		# print(f"the ratio2 is {ratio2}, the ratio3 is {ratio3}")

		if ratio2 < linear_ratio_threshold and ratio3 < linear_ratio_threshold:
			keypoint_labels.append("linear")
		elif ratio2 >= planar_ratio_threshold and ratio3 < planar_thickness_threshold:
			keypoint_labels.append("planar")
		else:
			keypoint_labels.append("volumetric")

	linear_count = sum(label == "linear" for label in keypoint_labels)
	planar_count = sum(label == "planar" for label in keypoint_labels)
	volumetric_count = sum(label == "volumetric" for label in keypoint_labels)
	valid_count = linear_count + planar_count + volumetric_count

	linear_fraction = (linear_count / valid_count) if valid_count else 0.0
	planar_fraction = (planar_count / valid_count) if valid_count else 0.0

	if valid_count == 0:
		topology = "ambiguous"
	elif linear_fraction >= 0.6 and linear_fraction > planar_fraction:
		topology = "wireharness_1d"
	elif planar_fraction >= 0.6 and planar_fraction >= linear_fraction:
		topology = "fabric_2d"
	else:
		topology = "ambiguous"

	return {
		"topology": topology,
		"scores": {
			"linear_fraction": linear_fraction,
			"planar_fraction": planar_fraction,
			"valid_keypoint_fraction": valid_count / len(keypoints) if len(keypoints) else 0.0,
		},
		"counts": {
			"linear": linear_count,
			"planar": planar_count,
			"volumetric": volumetric_count,
			"insufficient": sum(label == "insufficient" for label in keypoint_labels),
			"degenerate": sum(label == "degenerate" for label in keypoint_labels),
		},
		"neighbor_counts": neighbor_counts,
		"neighbor_sources": neighbor_sources,
		"keypoint_labels": keypoint_labels,
		"eigenvalues": eigenvalues_list,
		"parameters": {
			"k_neighbors": k_neighbors,
			"neighbor_radius": neighbor_radius,
			"linear_ratio_threshold": linear_ratio_threshold,
			"planar_ratio_threshold": planar_ratio_threshold,
			"planar_thickness_threshold": planar_thickness_threshold,
			"min_neighbors": min_neighbors,
			"ellipsoid_std_multiplier": ellipsoid_std_multiplier,
		},
	}



def build_wire_connections(
	keypoints: np.ndarray,
	*,
	foreground_points: np.ndarray,
	intrinsics: np.ndarray,
	image_shape: Tuple[int, int],
	visibility_tolerance: float = 1e-6,
	skeleton_dilation_iterations: int = 2,
	skeleton_block_radius: float = 2.5,
) -> Dict[str, Any]:
	"""Construct wire connections between keypoints using a 2D skeleton mask.

	Workflow (requested by the user):

	1. Anchor each Gaussian keypoint to its nearest foreground point in 3D.
	2. Project the anchored keypoints and the foreground cloud onto the image plane
	   using the provided camera intrinsics. The projected foreground points are
	   rasterised into a dense binary mask.
	3. Apply morphological thinning (Zhang–Suen) to the 2D mask to produce a
	   single-pixel-wide skeleton representing the wire harness.
	4. Snap every keypoint to the closest skeleton pixel.
	5. Connect pairs of keypoints when a skeleton path exists between their snapped
	   pixels that does not traverse pixels claimed by any other keypoint.

	The resulting graph is returned as a minimum spanning tree over the admissible
	connections, weighted by the skeleton path length. Diagnostic information is
	provided to aid downstream visualisation.

	Args:
		keypoints: (M, 3) keypoint array output by the GMM.
		foreground_points: (N, 3) array of foreground point-cloud samples.
		intrinsics: 3x3 camera intrinsics matrix used to generate the point cloud.
		image_shape: Tuple ``(height, width)`` describing the target pixel grid.
		visibility_tolerance: Slack for the Gabriel test that prevents edges whose
			diameter sphere contains another keypoint.
		skeleton_dilation_iterations: Number of binary dilation passes applied before
			skeletonisation to close small gaps in the projected foreground mask.
		skeleton_block_radius: Radius (in pixels) around snapped skeleton pixels that
		is reserved for the owning keypoint when testing candidate paths.

	Returns:
		Dictionary containing:
			``edges`` – list of ``(i, j, length)`` tuples defining the extracted tree.
			``total_length`` – total skeleton-path length of the tree.
			``degrees`` – per-keypoint degree array.
			``skeleton`` – binary skeleton mask for debugging/visualisation.
			``pixel_anchors`` – integer pixel locations for each keypoint.
			``filter_summary`` – statistics about candidate evaluation stages.
	"""

	keypoints_arr = _ensure_points_array(keypoints)
	foreground_arr = _ensure_points_array(foreground_points)
	intrinsics_arr = np.asarray(intrinsics, dtype=np.float64)
	if intrinsics_arr.shape != (3, 3):
		raise ValueError("intrinsics must be a 3x3 matrix")
	if len(image_shape) != 2:
		raise ValueError("image_shape must be a (height, width) tuple")
	img_height, img_width = int(image_shape[0]), int(image_shape[1])
	if img_height <= 0 or img_width <= 0:
		raise ValueError("image_shape entries must be positive")

	if keypoints_arr.shape[0] == 0:
		return {
			"edges": [],
			"total_length": 0.0,
			"degrees": [],
			"skeleton": np.zeros((img_height, img_width), dtype=np.uint8),
			"pixel_anchors": [],
			"filter_summary": {"candidate_pairs": 0},
		}
	if keypoints_arr.shape[0] == 1:
		return {
			"edges": [],
			"total_length": 0.0,
			"degrees": [0],
			"skeleton": np.zeros((img_height, img_width), dtype=np.uint8),
			"pixel_anchors": [(0, 0)],
			"filter_summary": {"candidate_pairs": 0},
		}

	visibility_tolerance = float(max(visibility_tolerance, 0.0))
	skeleton_block_radius = float(max(skeleton_block_radius, 0.0))
	skeleton_dilation_iterations = int(max(skeleton_dilation_iterations, 0))

	def _anchor_to_foreground(points3d: np.ndarray, cloud: np.ndarray) -> np.ndarray:
		if cloud.shape[0] == 0:
			return points3d.copy()
		if KDTree is not None:
			tree = KDTree(cloud)
			dists, indices = tree.query(points3d, k=1)
			indices = np.asarray(indices).reshape(-1)
			return cloud[indices]
		anchored = np.empty_like(points3d)
		for idx, point in enumerate(points3d):
			diffs = cloud - point
			dists_sq = np.sum(diffs * diffs, axis=1)
			anchored[idx] = cloud[np.argmin(dists_sq)]
		return anchored

	def _project_points(points3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		pts = np.asarray(points3d, dtype=np.float64)
		x = pts[:, 0]
		y = pts[:, 1]
		z = pts[:, 2]
		fx, fy = intrinsics_arr[0, 0], intrinsics_arr[1, 1]
		cx, cy = intrinsics_arr[0, 2], intrinsics_arr[1, 2]
		valid = z > 1e-6
		inv_z = np.zeros_like(z)
		inv_z[valid] = 1.0 / z[valid]
		u = fx * x * inv_z + cx
		v = fy * y * inv_z + cy
		pixels = np.stack([v, u], axis=1)
		return pixels, valid

	def _rasterise_mask(points2d: np.ndarray, mask_shape: Tuple[int, int]) -> np.ndarray:
		mask = np.zeros(mask_shape, dtype=np.uint8)
		if points2d.size == 0:
			return mask
		rows = np.clip(np.round(points2d[:, 0]).astype(int), 0, mask_shape[0] - 1)
		cols = np.clip(np.round(points2d[:, 1]).astype(int), 0, mask_shape[1] - 1)
		mask[rows, cols] = 1
		return mask

	def _binary_dilation(mask: np.ndarray, iterations: int) -> np.ndarray:
		if iterations <= 0:
			return mask.astype(bool)
		result = mask.astype(bool)
		for _ in range(iterations):
			padded = np.pad(result, 1, mode="constant", constant_values=False)
			dilated = (
				padded[1:-1, 1:-1]
				| padded[:-2, 1:-1]
				| padded[2:, 1:-1]
				| padded[1:-1, :-2]
				| padded[1:-1, 2:]
				| padded[:-2, :-2]
				| padded[:-2, 2:]
				| padded[2:, :-2]
				| padded[2:, 2:]
			)
			result = dilated
		return result

	def _zhang_suen_thinning(mask: np.ndarray) -> np.ndarray:
		img = mask.astype(np.uint8)
		if img.shape[0] < 3 or img.shape[1] < 3:
			return img.astype(bool)
		prev = np.zeros_like(img)
		img = img.copy()
		changed = True
		while changed:
			changed = False
			for step in range(2):
				sub = img[1:-1, 1:-1]
				P2 = img[:-2, 1:-1]
				P3 = img[:-2, 2:]
				P4 = img[1:-1, 2:]
				P5 = img[2:, 2:]
				P6 = img[2:, 1:-1]
				P7 = img[2:, :-2]
				P8 = img[1:-1, :-2]
				P9 = img[:-2, :-2]
				neighbors = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
				transitions = (
					((P2 == 0) & (P3 == 1)).astype(np.uint8)
					+ ((P3 == 0) & (P4 == 1)).astype(np.uint8)
					+ ((P4 == 0) & (P5 == 1)).astype(np.uint8)
					+ ((P5 == 0) & (P6 == 1)).astype(np.uint8)
					+ ((P6 == 0) & (P7 == 1)).astype(np.uint8)
					+ ((P7 == 0) & (P8 == 1)).astype(np.uint8)
					+ ((P8 == 0) & (P9 == 1)).astype(np.uint8)
					+ ((P9 == 0) & (P2 == 1)).astype(np.uint8)
				)
				if step == 0:
					cond = (
						sub == 1
						& (neighbors >= 2)
						& (neighbors <= 6)
						& (transitions == 1)
						& ((P2 * P4 * P6) == 0)
						& ((P4 * P6 * P8) == 0)
					)
				else:
					cond = (
						sub == 1
						& (neighbors >= 2)
						& (neighbors <= 6)
						& (transitions == 1)
						& ((P2 * P4 * P8) == 0)
						& ((P2 * P6 * P8) == 0)
					)
				if np.any(cond):
					sub[cond] = 0
					changed = True
			if np.array_equal(img, prev):
				break
			prev = img.copy()
		return img.astype(bool)

	anchored_keypoints = _anchor_to_foreground(keypoints_arr, foreground_arr)
	keypoint_pixels, keypoint_valid = _project_points(anchored_keypoints)
	foreground_pixels, foreground_valid = _project_points(foreground_arr)
	foreground_pixels = foreground_pixels[foreground_valid]

	mask = _rasterise_mask(foreground_pixels, (img_height, img_width))
	if np.any(mask):
		mask = _binary_dilation(mask, skeleton_dilation_iterations)
	else:
		mask = mask.astype(bool)

	skeleton_mask = _zhang_suen_thinning(mask.astype(np.uint8))
	if not np.any(skeleton_mask):
		skeleton_mask = mask.copy()

	skeleton_coords = np.argwhere(skeleton_mask)
	if skeleton_coords.size == 0:
		return {
			"edges": [],
			"total_length": 0.0,
			"degrees": [0] * keypoints_arr.shape[0],
			"skeleton": skeleton_mask.astype(np.uint8),
			"pixel_anchors": [],
			"filter_summary": {
				"candidate_pairs": 0,
				"reason": "empty-skeleton",
			},
		}

	if KDTree is not None and skeleton_coords.shape[0] > 1:
		skeleton_tree = KDTree(skeleton_coords)
	else:
		skeleton_tree = None

	def _snap_pixels(pixels: np.ndarray, valid_mask: np.ndarray) -> Tuple[List[Optional[int]], List[Tuple[int, int]]]:
		anchor_ids: List[Optional[int]] = []
		pixel_locations: List[Tuple[int, int]] = []
		for pixel, is_valid in zip(pixels, valid_mask):
			if not is_valid:
				anchor_ids.append(None)
				pixel_locations.append((-1, -1))
				continue
			row = int(np.clip(round(pixel[0]), 0, img_height - 1))
			col = int(np.clip(round(pixel[1]), 0, img_width - 1))
			if skeleton_tree is not None:
				dist, idx = skeleton_tree.query([[row, col]], k=1)
				skeleton_idx = int(np.asarray(idx).reshape(-1)[0])
			else:
				diffs = skeleton_coords - np.array([row, col])
				dists = np.sum(diffs * diffs, axis=1)
				skeleton_idx = int(np.argmin(dists))
			anchor_ids.append(skeleton_idx)
			pixel_locations.append((row, col))
		return anchor_ids, pixel_locations

	anchor_indices, pixel_anchors = _snap_pixels(keypoint_pixels, keypoint_valid)

	valid_anchor_mask = np.array([idx is not None for idx in anchor_indices], dtype=bool)
	if not np.any(valid_anchor_mask):
		return {
			"edges": [],
			"total_length": 0.0,
			"degrees": [0] * keypoints_arr.shape[0],
			"skeleton": skeleton_mask.astype(np.uint8),
			"pixel_anchors": pixel_anchors,
			"filter_summary": {
				"candidate_pairs": 0,
				"reason": "no-valid-anchors",
			},
		}

	n_keypoints = keypoints_arr.shape[0]
	anchor_indices_array = np.array([idx if idx is not None else -1 for idx in anchor_indices], dtype=int)

	# Pairwise distance matrix (3D) to retain Gabriel filtering in metric space.
	diff = keypoints_arr[:, None, :] - keypoints_arr[None, :, :]
	dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))

	# Build skeleton adjacency graph.
	n_nodes = skeleton_coords.shape[0]
	index_map = {tuple(coord.tolist()): idx for idx, coord in enumerate(skeleton_coords)}
	adjacency: List[List[Tuple[int, float]]] = [list() for _ in range(n_nodes)]
	neighbor_offsets = [
		(-1, -1), (-1, 0), (-1, 1),
		(0, -1), (0, 1),
		(1, -1), (1, 0), (1, 1),
	]
	for idx, (row, col) in enumerate(skeleton_coords):
		for dr, dc in neighbor_offsets:
			neighbor = (row + dr, col + dc)
			neighbor_idx = index_map.get(neighbor)
			if neighbor_idx is None:
				continue
			distance = math.hypot(dr, dc)
			if distance <= 0:
				continue
			adjacency[idx].append((neighbor_idx, distance))

	skeleton_coords_float = skeleton_coords.astype(np.float64)
	block_radius_sq = skeleton_block_radius * skeleton_block_radius
	keypoint_block_nodes: List[Set[int]] = []
	for kp_idx, anchor_idx in enumerate(anchor_indices_array):
		if anchor_idx < 0:
			keypoint_block_nodes.append(set())
			continue
		anchor_coord = skeleton_coords_float[anchor_idx]
		deltas = skeleton_coords_float - anchor_coord
		dists_sq = np.sum(deltas * deltas, axis=1)
		mask = dists_sq <= block_radius_sq
		indices = set(np.nonzero(mask)[0].tolist())
		indices.add(anchor_idx)
		keypoint_block_nodes.append(indices)

	def _passes_gabriel(i: int, j: int) -> bool:
		d_ij = float(dist_matrix[i, j])
		if not np.isfinite(d_ij) or d_ij <= 0.0:
			return False
		d_ij_sq = d_ij * d_ij
		for k in range(n_keypoints):
			if k == i or k == j:
				continue
			d_ik = float(dist_matrix[i, k])
			d_jk = float(dist_matrix[j, k])
			if not np.isfinite(d_ik) or not np.isfinite(d_jk):
				continue
			if d_ik == 0.0 or d_jk == 0.0:
				return False
			if (d_ik * d_ik + d_jk * d_jk) <= d_ij_sq * (1.0 + visibility_tolerance):
				return False
		return True

	def _shortest_skeleton_path(start: int, goal: int, blocked: Set[int]) -> Optional[float]:
		if start == goal:
			return 0.0
		heap: List[Tuple[float, int]] = [(0.0, start)]
		visited: Set[int] = set()
		blocked_local = set(blocked)
		blocked_local.discard(start)
		blocked_local.discard(goal)
		while heap:
			dist_so_far, node = heapq.heappop(heap)
			if node in visited or node in blocked_local:
				continue
			visited.add(node)
			if node == goal:
				return dist_so_far
			for nbr, weight in adjacency[node]:
				if nbr in visited or nbr in blocked_local:
					continue
				heapq.heappush(heap, (dist_so_far + weight, nbr))
		return None

	# Collect candidate pairs by checking all Gabriel-valid anchor pairs.
	candidate_pairs: List[Tuple[int, int]] = []
	for i in range(n_keypoints):
		if anchor_indices_array[i] < 0:
			continue
		for j in range(i + 1, n_keypoints):
			if anchor_indices_array[j] < 0:
				continue
			if not _passes_gabriel(i, j):
				continue
			candidate_pairs.append((i, j))

	filter_summary: Dict[str, Any] = {
		"candidate_pairs": len(candidate_pairs),
		"unreachable_pairs": 0,
	}

	candidate_edges: List[Tuple[int, int, float]] = []
	unreachable_details: List[Tuple[int, int, str]] = []
	for i, j in candidate_pairs:
		blocked_nodes = set()
		for k in range(n_keypoints):
			if k == i or k == j:
				continue
			blocked_nodes.update(keypoint_block_nodes[k])
		path_length = _shortest_skeleton_path(anchor_indices_array[i], anchor_indices_array[j], blocked_nodes)
		if path_length is None:
			filter_summary["unreachable_pairs"] += 1
			unreachable_details.append((i, j, "no-path"))
			continue
		if not np.isfinite(path_length) or path_length <= 0.0:
			filter_summary["unreachable_pairs"] += 1
			unreachable_details.append((i, j, "degenerate-path"))
			continue
		candidate_edges.append((i, j, float(path_length)))

	filter_summary["valid_paths"] = len(candidate_edges)
	if unreachable_details:
		filter_summary["unreachable_details"] = unreachable_details

	if not candidate_edges:
		return {
			"edges": [],
			"total_length": 0.0,
			"degrees": [0] * n_keypoints,
			"skeleton": skeleton_mask.astype(np.uint8),
			"pixel_anchors": pixel_anchors,
			"filter_summary": filter_summary,
		}

	candidate_edges.sort(key=lambda item: item[2])
	parent = list(range(n_keypoints))
	rank = [0] * n_keypoints

	def find(x: int) -> int:
		while parent[x] != x:
			parent[x] = parent[parent[x]]
			x = parent[x]
		return x

	def union(a: int, b: int) -> bool:
		root_a = find(a)
		root_b = find(b)
		if root_a == root_b:
			return False
		if rank[root_a] < rank[root_b]:
			parent[root_a] = root_b
		elif rank[root_a] > rank[root_b]:
			parent[root_b] = root_a
		else:
			parent[root_b] = root_a
			rank[root_a] += 1
		return True

	edges: List[Tuple[int, int, float]] = []
	degrees = np.zeros(n_keypoints, dtype=int)
	total_length = 0.0

	for i, j, distance in candidate_edges:
		if union(i, j):
			edges.append((i, j, distance))
			degrees[i] += 1
			degrees[j] += 1
			total_length += distance
			if len(edges) == n_keypoints - 1:
				break

	return {
		"edges": edges,
		"total_length": float(total_length),
		"degrees": degrees.tolist(),
		"skeleton": skeleton_mask.astype(np.uint8),
		"pixel_anchors": pixel_anchors,
		"filter_summary": filter_summary,
	}


def refine_keypoints_uniform_edges(
	keypoints: np.ndarray,
	*,
	edges: List[Tuple[int, int]] | List[Tuple[int, int, float]],
	foreground_points: np.ndarray,
	n_iters: int = 20,
	lr: float = 0.05,
	lambda_var: float = 1.0,
	lambda_anchor_b: float = 5.0,
	lambda_move: float = 0.0,
	normalize_var: bool = True,
	project_each_step: bool = True,
	max_step: Optional[float] = None,
	max_total_move: Optional[float] = None,
	eps: float = 1e-8,
	return_debug: bool = True,
) -> Dict[str, Any]:
	"""Refine GMM keypoints to encourage more uniform edge lengths.

	Args:
		keypoints: (K, 3) array of current keypoint locations that will be refined.
		edges: Connectivity list referencing keypoint indices. Entries may include an
			optional third value (e.g., path length); only the first two indices are used.
		foreground_points: (N, 3) point cloud used for optional projection back to the
			observed surface after each update step.
		n_iters: Number of gradient descent iterations.
		lr: Learning rate for gradient descent updates.
		lambda_var: Weight for the edge-length variance term.
		lambda_anchor_b: Weight for the anchor that preserves the global mean
			edge length relative to the original configuration.
		lambda_move: Weight for penalizing large deviations from the original
			keypoints.
		normalize_var: When ``True`` use the normalized variance objective; otherwise
			use the unnormalized form.
		project_each_step: Project updated keypoints back onto the foreground cloud
			after each iteration.
		max_step: Optional clamp on the per-iteration displacement magnitude per
			keypoint.
		max_total_move: Optional clamp on the overall displacement relative to the
			initial keypoints.
		eps: Numerical stability constant.
		return_debug: Whether to return diagnostic statistics.

	Returns:
		Dictionary containing the refined keypoints and optional debug information.
	"""

	mu0 = np.asarray(keypoints, dtype=np.float64)
	if mu0.ndim != 2 or mu0.shape[1] != 3:
		raise ValueError("`keypoints` must be of shape (K, 3).")

	if n_iters <= 0 or mu0.shape[0] == 0:
		return {
			"keypoints_refined": mu0.copy(),
			"debug": {
				"reason": "no-iterations" if n_iters <= 0 else "empty-keypoints",
				"iterations": int(max(n_iters, 0)),
			},
		}

	foreground_arr = np.asarray(foreground_points, dtype=np.float64)
	edge_pairs: List[Tuple[int, int]] = []
	seen: Set[Tuple[int, int]] = set()
	n_keypoints = mu0.shape[0]

	for edge in edges:
		if edge is None:
			continue
		if len(edge) < 2:
			continue
		i = int(edge[0])
		j = int(edge[1])
		if i == j:
			continue
		if i < 0 or j < 0 or i >= n_keypoints or j >= n_keypoints:
			continue
		pair = (i, j) if i < j else (j, i)
		if pair in seen:
			continue
		seen.add(pair)
		edge_pairs.append(pair)

	m = len(edge_pairs)
	if m == 0:
		return {
			"keypoints_refined": mu0.copy(),
			"debug": {
				"reason": "no-valid-edges",
				"edge_count": 0,
			},
		}

	edges_arr = np.asarray(edge_pairs, dtype=np.int64)
	mu = mu0.copy()

	diff0 = mu0[edges_arr[:, 0]] - mu0[edges_arr[:, 1]]
	l0 = np.linalg.norm(diff0, axis=1)
	l0 = np.maximum(l0, eps)

	mask_valid_move = max_total_move is not None and max_total_move > 0
	max_step_val = max_step if max_step is not None and max_step > 0 else None
	max_total_move_val = max_total_move if mask_valid_move else None

	tree = None
	if project_each_step and foreground_arr.shape[0] >= 1:
		try:
			tree = KDTree(foreground_arr)
		except Exception:
			tree = None

	def _project_to_cloud(points: np.ndarray) -> np.ndarray:
		if foreground_arr.shape[0] == 0:
			return points
		if KDTree is not None and tree is not None:
			_, idx = tree.query(points, k=1)
			idx = np.asarray(idx).reshape(-1)
			return foreground_arr[idx].copy()
		projected = np.empty_like(points)
		for idx_pt, pt in enumerate(points):
			diffs = foreground_arr - pt
			dists_sq = np.sum(diffs * diffs, axis=1)
			nearest = int(np.argmin(dists_sq))
			projected[idx_pt] = foreground_arr[nearest]
		return projected

	def _edge_stats(lengths: np.ndarray) -> Dict[str, float]:
		s = float(np.sum(lengths))
		q = float(np.sum(lengths * lengths))
		count = float(len(lengths))
		N = count * q - s * s
		if normalize_var:
			D = s * s + eps
			norm_var = N / D if D != 0.0 else 0.0
		else:
			denom = max(count * count, 1.0)
			norm_var = N / denom
		return {
			"mean": s / count if count > 0 else 0.0,
			"var": N / max(count, 1.0) if count > 0 else 0.0,
			"normalized_var": norm_var,
		}

	initial_lengths = l0.copy()
	stats_start = _edge_stats(initial_lengths)
	target_mean_length = stats_start.get("mean", float(np.mean(initial_lengths) if initial_lengths.size else 0.0))
	h_target = float(target_mean_length)
	history: List[Dict[str, float]] = []

	for iteration in range(int(n_iters)):
		diffs = mu[edges_arr[:, 0]] - mu[edges_arr[:, 1]]
		lengths = np.linalg.norm(diffs, axis=1)
		lengths = np.maximum(lengths, eps)

		s = float(np.sum(lengths))
		q = float(np.sum(lengths * lengths))
		count_edges = float(m)
		N = count_edges * q - s * s
		if normalize_var:
			D = s * s + eps
			dN = 2.0 * count_edges * lengths - 2.0 * s
			dD = 2.0 * s
			dR_var = (dN * D - N * dD) / (D * D)
		else:
			denom = max(count_edges * count_edges, 1.0)
			dN = 2.0 * count_edges * lengths - 2.0 * s
			dR_var = dN / denom

		if count_edges > 0:
			mean_length = s / count_edges
			anchor_scalar = 2.0 * (mean_length - h_target) / count_edges
		else:
			anchor_scalar = 0.0
		anchor_term = np.full_like(lengths, anchor_scalar)
		grad_lengths = lambda_var * dR_var + lambda_anchor_b * anchor_term

		unit_dirs = (diffs / lengths[:, None])
		grad = np.zeros_like(mu)
		for edge_idx, (i, j) in enumerate(edge_pairs):
			direction = unit_dirs[edge_idx]
			scale = grad_lengths[edge_idx]
			grad[ i ] += scale * direction
			grad[ j ] -= scale * direction

		grad += 2.0 * lambda_move * (mu - mu0)

		step = lr * grad
		if max_step_val is not None:
			norms = np.linalg.norm(step, axis=1)
			over = norms > max_step_val
			if np.any(over):
				scale = max_step_val / np.maximum(norms[over], eps)
				step[over] *= scale[:, None]

		mu_candidate = mu - step

		if max_total_move_val is not None:
			disp = mu_candidate - mu0
			disp_norms = np.linalg.norm(disp, axis=1)
			overall = disp_norms > max_total_move_val
			if np.any(overall):
				scale = max_total_move_val / np.maximum(disp_norms[overall], eps)
				mu_candidate[overall] = mu0[overall] + disp[overall] * scale[:, None]

		if project_each_step:
			mu_candidate = _project_to_cloud(mu_candidate)

		mu = mu_candidate
		history.append(_edge_stats(lengths))

	if not project_each_step:
		mu = _project_to_cloud(mu)

	final_diffs = mu[edges_arr[:, 0]] - mu[edges_arr[:, 1]]
	final_lengths = np.linalg.norm(final_diffs, axis=1)
	final_lengths = np.maximum(final_lengths, eps)
	stats_end = _edge_stats(final_lengths)

	debug: Dict[str, Any] = {
		"edge_count": int(m),
		"iterations": int(n_iters),
		"initial_edge_stats": stats_start,
		"final_edge_stats": stats_end,
		"max_movement": float(np.max(np.linalg.norm(mu - mu0, axis=1))),
		"target_mean_edge_length": h_target,
	}
	if return_debug:
		debug["history"] = history

	return {
		"keypoints_refined": mu,
		"debug": debug if return_debug else None,
	}

