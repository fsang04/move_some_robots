"""
Repulsion-based relaxation utilities for uniform keypoint spacing.

Stage 2 of the mu-only uniform keypoints algorithm:
- Build kNN graph among keypoints
- Apply repulsive potential to push close keypoints apart
- Project back onto the point cloud after each step
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Dict, Any


def build_knn_graph(
    keypoints: np.ndarray,
    n_neighbors: int = 6,
) -> np.ndarray:
    """Build k-nearest neighbor graph among keypoints."""
    K = keypoints.shape[0]
    actual_neighbors = min(n_neighbors + 1, K)
    
    nn = NearestNeighbors(n_neighbors=actual_neighbors, algorithm="auto")
    nn.fit(keypoints)
    _, indices = nn.kneighbors(keypoints)
    
    neighbor_indices = indices[:, 1:]
    return neighbor_indices


def compute_repulsion_gradient(
    keypoints: np.ndarray,
    neighbor_indices: np.ndarray,
    min_distance: Optional[float] = None,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Compute pure repulsion gradient to push close keypoints apart."""
    K = keypoints.shape[0]
    gradient = np.zeros((K, 3), dtype=np.float64)
    
    all_dists = []
    for i in range(K):
        for j in neighbor_indices[i]:
            v = keypoints[i] - keypoints[j]
            d = np.linalg.norm(v)
            all_dists.append(d)
    
    if min_distance is None:
        min_distance = np.max(all_dists) * 1.1 if all_dists else 1.0
    
    for i in range(K):
        for j in neighbor_indices[i]:
            v = keypoints[i] - keypoints[j]
            d = np.linalg.norm(v)
            
            if d < epsilon:
                v = np.random.randn(3)
                d = np.linalg.norm(v) + epsilon
                v = v / d
                d = epsilon
            
            if d >= min_distance:
                continue
            
            unit_v = v / d
            strength = (min_distance / d) ** 2 - 1.0
            strength = max(strength, 0.0)
            gradient[i] += strength * unit_v
    
    return gradient


def project_to_cloud(
    keypoints: np.ndarray,
    point_cloud: np.ndarray,
    nn_index: Optional[NearestNeighbors] = None,
) -> np.ndarray:
    """Project each keypoint to its nearest point in the cloud."""
    if nn_index is None:
        nn_index = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn_index.fit(point_cloud)
    
    _, indices = nn_index.kneighbors(keypoints)
    projected = point_cloud[indices.flatten()]
    return projected


def repulsion_relaxation(
    keypoints: np.ndarray,
    point_cloud: np.ndarray,
    *,
    n_neighbors: int = 6,
    n_iterations: int = 15,
    learning_rate: float = 1.0,
    epsilon: float = 1e-8,
    project_each_step: bool = True,
    return_debug: bool = False,
) -> Dict[str, Any]:
    """Apply repulsion-based relaxation to make keypoint spacing more uniform."""
    K = keypoints.shape[0]
    if K < 2:
        return {"keypoints": keypoints.copy(), "debug": None}
    
    cloud_nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    cloud_nn.fit(point_cloud)
    
    mu = keypoints.copy().astype(np.float64)
    
    debug_info = {
        "iterations": [],
        "initial_min_dist": None,
        "final_min_dist": None,
    } if return_debug else None
    
    if return_debug:
        nn_graph = build_knn_graph(mu, n_neighbors=1)
        dists = np.linalg.norm(mu - mu[nn_graph[:, 0]], axis=1)
        debug_info["initial_min_dist"] = float(np.min(dists))
        debug_info["initial_mean_dist"] = float(np.mean(dists))
    
    for iteration in range(n_iterations):
        neighbor_indices = build_knn_graph(mu, n_neighbors=n_neighbors)
        
        repulsion = compute_repulsion_gradient(
            mu, neighbor_indices, min_distance=None, epsilon=epsilon
        )
        
        nn_graph = build_knn_graph(mu, n_neighbors=1)
        mean_dist = np.mean([np.linalg.norm(mu[i] - mu[nn_graph[i, 0]]) for i in range(K)])
        
        max_rep = np.max(np.linalg.norm(repulsion, axis=1))
        if max_rep > epsilon:
            repulsion = repulsion / max_rep * learning_rate * mean_dist * 0.1
        
        mu = mu + repulsion
        
        if project_each_step:
            mu = project_to_cloud(mu, point_cloud, nn_index=cloud_nn)
        
        if return_debug:
            nn_graph = build_knn_graph(mu, n_neighbors=1)
            dists = np.linalg.norm(mu - mu[nn_graph[:, 0]], axis=1)
            debug_info["iterations"].append({
                "iteration": iteration,
                "min_neighbor_dist": float(np.min(dists)),
                "mean_neighbor_dist": float(np.mean(dists)),
            })
    
    mu = project_to_cloud(mu, point_cloud, nn_index=cloud_nn)
    
    if return_debug:
        nn_graph = build_knn_graph(mu, n_neighbors=1)
        dists = np.linalg.norm(mu - mu[nn_graph[:, 0]], axis=1)
        debug_info["final_min_dist"] = float(np.min(dists))
        debug_info["final_mean_dist"] = float(np.mean(dists))
    
    return {
        "keypoints": mu,
        "debug": debug_info,
    }


def compute_spacing_stats(keypoints: np.ndarray, n_neighbors: int = 1) -> Dict[str, float]:
    """Compute statistics about keypoint spacing."""
    if keypoints.shape[0] < 2:
        return {
            "min_dist": 0.0,
            "max_dist": 0.0,
            "mean_dist": 0.0,
            "std_dist": 0.0,
            "uniformity": 1.0,
        }
    
    neighbor_indices = build_knn_graph(keypoints, n_neighbors=n_neighbors)
    
    dists = np.linalg.norm(
        keypoints[:, np.newaxis, :] - keypoints[neighbor_indices],
        axis=2
    )
    min_dists = dists[:, 0]
    
    min_dist = float(np.min(min_dists))
    max_dist = float(np.max(min_dists))
    
    return {
        "min_dist": min_dist,
        "max_dist": max_dist,
        "mean_dist": float(np.mean(min_dists)),
        "std_dist": float(np.std(min_dists)),
        "uniformity": min_dist / max_dist if max_dist > 0 else 1.0,
    }
