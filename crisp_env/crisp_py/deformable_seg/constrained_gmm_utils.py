"""Constrained Gaussian Mixture utilities with uniform-spacing regularization.

This module implements a simple Maximum-A-Posteriori (MAP) Gaussian Mixture
Model optimiser that augments the standard log-likelihood with a topology-free
uniformity prior on the component means. The regulariser penalises local
deviations from a desired spacing using a robust (Huber) loss evaluated on a
k-nearest-neighbour graph constructed over the current means each iteration.

The resulting objective encourages the GMM components to cover the foreground
points evenly without relying on an explicit edge topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def _huber_loss_and_grad(residual: np.ndarray, delta: float) -> Tuple[np.ndarray, np.ndarray]:
	"""Return Huber loss value and derivative with respect to ``residual``."""

	delta = max(float(delta), 1e-8)
	abs_res = np.abs(residual)
	quadratic = abs_res <= delta

	loss = np.where(
		quadratic,
		0.5 * residual**2,
		delta * (abs_res - 0.5 * delta),
	)

	grad = np.where(
		quadratic,
		residual,
		delta * np.sign(residual),
	)

	return loss, grad


def _build_neighbor_pairs(
	means: np.ndarray,
	n_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Return unique neighbour pairs (i, j) and their distances."""

	n_components = means.shape[0]
	if n_components <= 1 or n_neighbors <= 0:
		return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float64)

	k = min(n_neighbors + 1, n_components)
	nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(means)
	distances, indices = nbrs.kneighbors(means)

	pair_set = set()
	pair_list = []
	dist_list = []

	for i in range(n_components):
		for dist, j in zip(distances[i, 1:], indices[i, 1:]):  # skip self neighbour
			if i == j:
				continue
			key = (i, j) if i < j else (j, i)
			if key in pair_set:
				continue
			pair_set.add(key)
			pair_list.append(key)
			dist_list.append(float(dist))

	if not pair_list:
		return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float64)

	return np.asarray(pair_list, dtype=np.int64), np.asarray(dist_list, dtype=np.float64)


def _uniform_penalty(
	means: np.ndarray,
	n_neighbors: int,
	target_distance: Optional[float],
	delta: float,
) -> Tuple[float, np.ndarray, float]:
	"""Compute the local uniform-spacing penalty and its gradient."""

	if means.shape[0] <= 1:
		return 0.0, np.zeros_like(means), float(target_distance or 0.0)

	pairs, pair_distances = _build_neighbor_pairs(means, n_neighbors)
	if pairs.size == 0:
		return 0.0, np.zeros_like(means), float(target_distance or 0.0)

	if target_distance is None:
		target = float(np.median(pair_distances)) if pair_distances.size else 0.0
	else:
		target = float(target_distance)

	penalty = 0.0
	gradient = np.zeros_like(means, dtype=np.float64)

	eps = 1e-9
	residuals = pair_distances - target
	loss_vals, loss_grads = _huber_loss_and_grad(residuals, delta)

	for (i, j), distance, grad_scalar in zip(pairs, pair_distances, loss_grads):
		if distance < eps:
			# Degenerate pair: push the points apart slightly.
			direction = np.random.default_rng().normal(size=means.shape[1])
			direction /= np.linalg.norm(direction) + eps
		else:
			direction = (means[i] - means[j]) / distance

		gradient[i] += grad_scalar * direction
		gradient[j] -= grad_scalar * direction

	penalty = float(np.sum(loss_vals))

	return penalty, gradient, target


def _log_gaussian_diag(
	X: np.ndarray,
	means: np.ndarray,
	covariances: np.ndarray,
) -> np.ndarray:
	"""Log probability of each point under each diagonal Gaussian component."""

	n_samples, n_features = X.shape
	n_components = means.shape[0]

	precisions = 1.0 / covariances
	log_det = np.sum(np.log(covariances), axis=1)

	log_prob = np.empty((n_samples, n_components), dtype=np.float64)
	for k in range(n_components):
		diff = X - means[k]
		log_prob[:, k] = -0.5 * (
			np.sum(diff**2 * precisions[k], axis=1)
			+ log_det[k]
			+ n_features * np.log(2.0 * np.pi)
		)

	return log_prob


def _logsumexp(a: np.ndarray, axis: int = 1) -> Tuple[np.ndarray, np.ndarray]:
	max_vals = np.max(a, axis=axis, keepdims=True)
	stable = a - max_vals
	exp_vals = np.exp(stable)
	sum_exp = np.sum(exp_vals, axis=axis, keepdims=True)
	log_sum = np.log(sum_exp) + max_vals
	probs = exp_vals / sum_exp
	return log_sum.squeeze(axis), probs


@dataclass
class ConstrainedGMM:
	"""Gaussian Mixture with a local uniformity regulariser on the means."""

	n_components: int
	lambda_uniform: float = 1.0
	uniform_neighbors: int = 6
	uniform_target: Optional[float] = None
	uniform_delta: float = 5.0
	covariance_reg: float = 1e-6
	max_iter: int = 100
	tol: float = 1e-4
	mean_learning_rate: float = 0.25
	mean_gradient_steps: int = 3
	random_state: Optional[int] = None

	weights_: Optional[np.ndarray] = None
	means_: Optional[np.ndarray] = None
	covariances_: Optional[np.ndarray] = None
	log_likelihood_: Optional[float] = None
	uniform_penalty_: Optional[float] = None
	target_distance_: Optional[float] = None

	def _initialise(self, X: np.ndarray) -> None:
		rng = np.random.default_rng(self.random_state)
		kmeans = KMeans(n_clusters=self.n_components, n_init=10, random_state=self.random_state)
		labels = kmeans.fit_predict(X)
		means = kmeans.cluster_centers_

		weights = np.bincount(labels, minlength=self.n_components).astype(np.float64)
		weights = np.maximum(weights, 1.0)
		weights /= weights.sum()

		covariances = np.empty((self.n_components, X.shape[1]), dtype=np.float64)
		for k in range(self.n_components):
			cluster_points = X[labels == k]
			if cluster_points.size == 0:
				covariances[k] = np.var(X, axis=0) + self.covariance_reg
				means[k] = X[rng.integers(0, X.shape[0])]
			else:
				covariances[k] = np.var(cluster_points, axis=0) + self.covariance_reg

		self.weights_ = weights
		self.means_ = means
		self.covariances_ = covariances

	def _e_step(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
		log_prob = _log_gaussian_diag(X, self.means_, self.covariances_)
		log_prob += np.log(self.weights_)[None, :]
		log_sum, resp = _logsumexp(log_prob, axis=1)
		total_log_likelihood = float(np.sum(log_sum))
		return resp, total_log_likelihood

	def _m_step(self, X: np.ndarray, resp: np.ndarray) -> None:
		Nk = resp.sum(axis=0) + 1e-10
		weighted_sum = resp.T @ X

		# Standard updates before applying the uniform regulariser.
		means = weighted_sum / Nk[:, None]
		covariances = np.empty_like(self.covariances_)
		for k in range(self.n_components):
			diff = X - means[k]
			covariances[k] = (resp[:, k][:, None] * diff**2).sum(axis=0) / Nk[k]
		covariances += self.covariance_reg

		weights = Nk / Nk.sum()

		if self.lambda_uniform > 0.0 and self.uniform_neighbors > 0:
			for _ in range(self.mean_gradient_steps):
				penalty, grad, target = _uniform_penalty(
					means,
					self.uniform_neighbors,
					self.uniform_target,
					self.uniform_delta,
				)
				means -= self.mean_learning_rate * self.lambda_uniform * grad / (Nk[:, None])
				self.uniform_penalty_ = penalty
				self.target_distance_ = target

		self.weights_ = weights
		self.means_ = means
		self.covariances_ = covariances

	def fit(self, X: ArrayLike) -> "ConstrainedGMM":
		X = np.asarray(X, dtype=np.float64)
		if X.ndim != 2:
			raise ValueError("Input data X must be a 2D array")
		if X.shape[0] < self.n_components:
			raise ValueError("Number of samples must exceed number of components")

		self._initialise(X)

		prev_objective = -np.inf
		for iteration in range(self.max_iter):
			resp, log_likelihood = self._e_step(X)
			self._m_step(X, resp)

			penalty = 0.0
			if self.lambda_uniform > 0.0 and self.uniform_neighbors > 0:
				penalty, _, target = _uniform_penalty(
					self.means_,
					self.uniform_neighbors,
					self.uniform_target,
					self.uniform_delta,
				)
				self.uniform_penalty_ = penalty
				self.target_distance_ = target

			objective = log_likelihood - self.lambda_uniform * penalty
			improvement = objective - prev_objective
			if improvement < self.tol:
				break
			prev_objective = objective

		self.log_likelihood_ = log_likelihood
		return self

	def score_samples(self, X: ArrayLike) -> np.ndarray:
		X = np.asarray(X, dtype=np.float64)
		log_prob = _log_gaussian_diag(X, self.means_, self.covariances_)
		log_prob += np.log(self.weights_)[None, :]
		log_sum, _ = _logsumexp(log_prob, axis=1)
		return log_sum

	def predict_proba(self, X: ArrayLike) -> np.ndarray:
		X = np.asarray(X, dtype=np.float64)
		log_prob = _log_gaussian_diag(X, self.means_, self.covariances_)
		log_prob += np.log(self.weights_)[None, :]
		_, resp = _logsumexp(log_prob, axis=1)
		return resp

	def predict(self, X: ArrayLike) -> np.ndarray:
		resp = self.predict_proba(X)
		return np.argmax(resp, axis=1)


__all__ = [
	"ConstrainedGMM",
	"fps_repulsion_keypoints",
	"UniformKeypoints",
]


def _farthest_point_sampling(points: np.ndarray, n_samples: int, seed: int = 0) -> np.ndarray:
    """Sample n_samples from points using farthest point sampling."""
    rng = np.random.default_rng(seed)
    n_points = points.shape[0]
    if n_samples >= n_points:
        return points.copy()

    idx = rng.integers(0, n_points)
    selected = [idx]
    min_dist = np.full(n_points, np.inf, dtype=np.float64)

    for _ in range(n_samples - 1):
        last = points[selected[-1]]
        dist = np.linalg.norm(points - last, axis=1)
        min_dist = np.minimum(min_dist, dist)
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)

    return points[selected]


def _project_to_cloud(means: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    """Project each mean to its nearest neighbour in the cloud."""
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(cloud)
    _, indices = nbrs.kneighbors(means)
    return cloud[indices.squeeze()]


def _repulsion_gradient(
    means: np.ndarray,
    n_neighbors: int,
    target_distance: Optional[float],
    delta: float,
) -> Tuple[np.ndarray, float]:
    """Compute repulsion gradient pushing means towards uniform spacing."""
    if means.shape[0] <= 1:
        return np.zeros_like(means), 0.0

    pairs, pair_distances = _build_neighbor_pairs(means, n_neighbors)
    if pairs.size == 0:
        return np.zeros_like(means), 0.0

    target = target_distance if target_distance else float(np.median(pair_distances))
    residuals = pair_distances - target
    _, loss_grads = _huber_loss_and_grad(residuals, delta)

    gradient = np.zeros_like(means, dtype=np.float64)
    eps = 1e-9
    for (i, j), distance, grad_scalar in zip(pairs, pair_distances, loss_grads):
        if distance < eps:
            direction = np.random.default_rng().normal(size=means.shape[1])
            direction /= np.linalg.norm(direction) + eps
        else:
            direction = (means[i] - means[j]) / distance
        gradient[i] += grad_scalar * direction
        gradient[j] -= grad_scalar * direction

    return gradient, target


def fps_repulsion_keypoints(
    points: np.ndarray,
    n_keypoints: int,
    *,
    n_neighbors: int = 6,
    target_distance: Optional[float] = None,
    delta: float = 5.0,
    learning_rate: float = 0.5,
    n_iterations: int = 15,
    seed: int = 0,
) -> np.ndarray:
    """Extract uniformly-spaced keypoints via FPS + repulsion refinement.

    Parameters
    ----------
    points : ndarray (N, D)
        Input point cloud.
    n_keypoints : int
        Number of keypoints to extract.
    n_neighbors : int
        Number of neighbours for repulsion gradient.
    target_distance : float or None
        Desired spacing; if None, uses median of current neighbour distances.
    delta : float
        Huber loss transition threshold.
    learning_rate : float
        Step size for gradient descent.
    n_iterations : int
        Number of repulsion + projection iterations.
    seed : int
        Random seed for FPS initialisation.

    Returns
    -------
    ndarray (n_keypoints, D)
        Keypoint coordinates lying on the input cloud.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must be 2D")
    if n_keypoints <= 0 or n_keypoints > points.shape[0]:
        raise ValueError("Invalid n_keypoints")

    means = _farthest_point_sampling(points, n_keypoints, seed=seed)

    for _ in range(n_iterations):
        grad, _ = _repulsion_gradient(means, n_neighbors, target_distance, delta)
        means = means - learning_rate * grad
        means = _project_to_cloud(means, points)

    return means


@dataclass
class UniformKeypoints:
    """Lightweight keypoint extractor: FPS + repulsion, optional EM for soft assignments."""

    n_keypoints: int
    n_neighbors: int = 6
    target_distance: Optional[float] = None
    delta: float = 5.0
    learning_rate: float = 0.5
    n_iterations: int = 15
    covariance_reg: float = 1e-6
    run_em: bool = False
    em_iterations: int = 5
    random_state: int = 0

    means_: Optional[np.ndarray] = None
    weights_: Optional[np.ndarray] = None
    covariances_: Optional[np.ndarray] = None
    responsibilities_: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike) -> "UniformKeypoints":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[0] < self.n_keypoints:
            raise ValueError("Not enough points")

        means = fps_repulsion_keypoints(
            X,
            self.n_keypoints,
            n_neighbors=self.n_neighbors,
            target_distance=self.target_distance,
            delta=self.delta,
            learning_rate=self.learning_rate,
            n_iterations=self.n_iterations,
            seed=self.random_state,
        )
        self.means_ = means

        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(means)
        _, assignments = nbrs.kneighbors(X)
        assignments = assignments.squeeze()

        counts = np.bincount(assignments, minlength=self.n_keypoints).astype(np.float64) + 1e-10
        self.weights_ = counts / counts.sum()

        covariances = np.empty((self.n_keypoints, X.shape[1]), dtype=np.float64)
        for k in range(self.n_keypoints):
            cluster = X[assignments == k]
            if cluster.size == 0:
                covariances[k] = np.var(X, axis=0) + self.covariance_reg
            else:
                covariances[k] = np.var(cluster, axis=0) + self.covariance_reg
        self.covariances_ = covariances

        if self.run_em:
            for _ in range(self.em_iterations):
                log_prob = _log_gaussian_diag(X, self.means_, self.covariances_)
                log_prob += np.log(self.weights_)[None, :]
                _, resp = _logsumexp(log_prob, axis=1)
                Nk = resp.sum(axis=0) + 1e-10
                self.weights_ = Nk / Nk.sum()
                for k in range(self.n_keypoints):
                    diff = X - self.means_[k]
                    self.covariances_[k] = (resp[:, k][:, None] * diff**2).sum(axis=0) / Nk[k] + self.covariance_reg
            log_prob = _log_gaussian_diag(X, self.means_, self.covariances_)
            log_prob += np.log(self.weights_)[None, :]
            _, resp = _logsumexp(log_prob, axis=1)
            self.responsibilities_ = resp
        else:
            resp = np.zeros((X.shape[0], self.n_keypoints), dtype=np.float64)
            resp[np.arange(X.shape[0]), assignments] = 1.0
            self.responsibilities_ = resp

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(self.means_)
        _, indices = nbrs.kneighbors(X)
        return indices.squeeze()


