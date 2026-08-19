"""
Pure CPD Tracking for Wire Tracking (Baseline)

This implements pure Coherent Point Drift without any additional constraints.
Used to visualize what CPD alone produces, as a baseline comparison.

Loss: L = L_cpd only (no anchor, edge, or wire terms)

Author: Auto-generated
Date: 2026-02-15
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple, List, Dict
import time


class PureCPDTracker:
    """
    Pure CPD tracker without any additional constraints.
    
    This is the simplest baseline - just CPD registration from
    previous keypoints to current skeleton point cloud.
    """
    
    def __init__(
        self,
        # CPD parameters
        cpd_beta: float = 10.0,
        cpd_lambda: float = 2.0,
        cpd_w: float = 0.1,
        cpd_max_iter: int = 100,
        cpd_tol: float = 1e-3,  # 1mm convergence tolerance for meter-scale data
    ):
        """
        Initialize Pure CPD Tracker.
        
        Args:
            cpd_beta: Gaussian kernel width multiplier (scaled by data)
            cpd_lambda: Regularization weight
            cpd_w: Outlier weight [0, 1]
            cpd_max_iter: Maximum EM iterations
            cpd_tol: Convergence tolerance (change in T_Y norm)
        """
        self.cpd_beta = cpd_beta
        self.cpd_lambda = cpd_lambda
        self.cpd_w = cpd_w
        self.cpd_max_iter = cpd_max_iter
        self.cpd_tol = cpd_tol
    
    def cpd_register(
        self,
        Y: np.ndarray,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Pure CPD registration.
        
        Args:
            Y: M × D template (previous keypoints)
            X: N × D target (current skeleton points)
        
        Returns:
            T_Y: M × D transformed template
            P: M × N correspondence matrix
            info: Dictionary with convergence info
        """
        Y = np.asarray(Y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        
        M, D = Y.shape
        N = X.shape[0]
        
        if M == 0 or N == 0:
            return Y.copy(), np.zeros((M, N)), {'iterations': 0}
        
        # Initialize
        T_Y = Y.copy()
        W = np.zeros((M, D))
        
        # Gaussian kernel for motion coherence
        # Use pairwise distances in Y to set appropriate scale
        diff_Y = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
        dist_Y = np.sqrt(np.sum(diff_Y ** 2, axis=2))
        # Set beta based on median neighbor distance (more adaptive)
        np.fill_diagonal(dist_Y, np.inf)
        median_dist = np.median(np.min(dist_Y, axis=1))
        beta = max(self.cpd_beta * median_dist, 0.01)  # Scale beta by data
        
        G = np.exp(-np.sum(diff_Y ** 2, axis=2) / (2 * beta ** 2))
        
        # Initialize sigma^2 - use a smaller initial value based on data scale
        # This is critical: large sigma2 makes all correspondences uniform
        diff_init = X[np.newaxis, :, :] - Y[:, np.newaxis, :]
        dist2_init = np.sum(diff_init ** 2, axis=2)
        # Use median of minimum distances (not mean of all) for better initialization
        min_dists = np.min(dist2_init, axis=1)  # For each Y, find closest X
        sigma2 = np.median(min_dists) / 2.0  # Start smaller to get sharper correspondences
        sigma2 = max(sigma2, 1e-6)
        
        info = {
            'iterations': 0,
            'sigma2_history': [],
            'cpd_loss': [],
            'beta_used': beta,
        }
        
        for iteration in range(self.cpd_max_iter):
            # ============================================================
            # E-step: Compute posterior probabilities
            # ============================================================
            diff = X[np.newaxis, :, :] - T_Y[:, np.newaxis, :]
            dist2 = np.sum(diff ** 2, axis=2)  # M x N
            
            # Numerically stable softmax-style normalization
            # Subtract max to prevent overflow
            log_p = -dist2 / (2 * sigma2)
            log_p_max = np.max(log_p, axis=0, keepdims=True)
            P_num = np.exp(log_p - log_p_max)
            
            c = (self.cpd_w / (1 - self.cpd_w + 1e-10)) * (M / N) * np.exp(-log_p_max)
            P_den = np.sum(P_num, axis=0, keepdims=True) + c
            P = P_num / (P_den + 1e-10)
            
            # ============================================================
            # M-step: Solve for W (pure CPD, no constraints)
            # ============================================================
            P1 = np.sum(P, axis=1)  # M
            Np = np.sum(P1)
            
            P1_safe = np.maximum(P1, 1e-10)
            D_inv = np.diag(1.0 / P1_safe)
            
            # Standard CPD linear system: (G + λσ²D⁻¹)W = D⁻¹PX - Y
            A = G + self.cpd_lambda * sigma2 * D_inv
            B = D_inv @ P @ X - Y
            
            try:
                W = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                W = np.linalg.lstsq(A, B, rcond=None)[0]
            
            T_Y_new = Y + G @ W
            
            # Update sigma^2
            diff_new = X[np.newaxis, :, :] - T_Y_new[:, np.newaxis, :]
            dist2_new = np.sum(diff_new ** 2, axis=2)
            sigma2_new = np.sum(P * dist2_new) / (Np * D + 1e-10)
            sigma2_new = max(sigma2_new, 1e-6)
            
            # Compute loss for monitoring
            cpd_loss = np.sum(P * dist2_new)
            
            info['sigma2_history'].append(sigma2_new)
            info['cpd_loss'].append(cpd_loss)
            
            # Convergence check
            change = np.linalg.norm(T_Y_new - T_Y)
            if change < self.cpd_tol:
                T_Y = T_Y_new
                info['iterations'] = iteration + 1
                break
            
            T_Y = T_Y_new
            sigma2 = sigma2_new
            info['iterations'] = iteration + 1
        
        return T_Y, P, info
    
    def track_frame(
        self,
        prev_keypoints: np.ndarray,
        skeleton_pc: np.ndarray,
        detected_branch_3d: np.ndarray,  # Unused, but kept for API compatibility
        detected_leaf_3d: np.ndarray,     # Unused, but kept for API compatibility
        reference_edges: List[Tuple[int, int]],  # Unused
        reference_lengths: np.ndarray,    # Unused
        n_branch: int,                    # Unused
        n_leaf: int,                      # Unused
        cpd_downsample: int = 500,
    ) -> Dict:
        """
        Track keypoints using Pure CPD.
        
        Args:
            prev_keypoints: K × 3 previous frame keypoints
            skeleton_pc: N × 3 current skeleton point cloud
            detected_branch_3d: Unused (for API compatibility)
            detected_leaf_3d: Unused (for API compatibility)
            reference_edges: Unused (for API compatibility)
            reference_lengths: Unused (for API compatibility)
            n_branch: Unused (for API compatibility)
            n_leaf: Unused (for API compatibility)
            cpd_downsample: Max points for CPD target
        
        Returns:
            Dict with keypoints, timing, convergence info
        """
        timing = {}
        t_total = time.time()
        
        # Input validation
        if len(prev_keypoints) == 0:
            return {
                'keypoints': prev_keypoints.copy(),
                'timing': {'total': 0},
                'cpd_info': {'iterations': 0, 'error': 'empty prev_keypoints'},
            }
        
        if len(skeleton_pc) == 0:
            return {
                'keypoints': prev_keypoints.copy(),
                'timing': {'total': 0},
                'cpd_info': {'iterations': 0, 'error': 'empty skeleton_pc'},
            }
        
        # Downsample target
        cpd_target = skeleton_pc
        if len(cpd_target) > cpd_downsample:
            indices = np.random.choice(len(cpd_target), cpd_downsample, replace=False)
            cpd_target = cpd_target[indices]
        
        # Pure CPD (no constraints)
        t0 = time.time()
        keypoints, P, cpd_info = self.cpd_register(
            prev_keypoints,
            cpd_target,
        )
        timing['cpd'] = time.time() - t0
        
        timing['total'] = time.time() - t_total
        
        return {
            'keypoints': keypoints,
            'timing': timing,
            'cpd_info': cpd_info,
        }


def create_tracker(config: Dict = None) -> PureCPDTracker:
    """Factory function to create PureCPDTracker with optional config."""
    if config is None:
        config = {}
    return PureCPDTracker(**config)
