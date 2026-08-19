"""
CDCPD2-style Tracking for Wire Tracking

Constrained Deformable Coherent Point Drift (CDCPD2) implementation.
Based on the paper "Tracking Partially-Occluded Deformable Objects while Enforcing Geometric Constraints"
by Wang, McConachie, and Berenson (2020).

Paper: https://arxiv.org/abs/2011.00627
Reference: https://github.com/UM-ARM-Lab/cdcpd

Key differences from pure CPD:
1. LLE (Locally Linear Embedding) regularization for smoothness
2. Post-CPD quadratic programming to enforce edge length constraints
3. Fixed point (anchor) constraints as hard constraints

The algorithm:
    Phase 1: CPD with LLE regularization
        (G + λσ²D⁻¹ + σ²γL_lle)W = D⁻¹PX - Y - γL_lle·Y
        
    Phase 2: QP optimization
        min ||Y_opt - T_Y||²
        s.t. ||Y_opt[i] - Y_opt[j]||² ≤ (stretch_λ · l_ij)² for all edges (i,j)
             Y_opt[anchor_idx] = anchor_pos  (optional hard constraints)

Author: Auto-generated
Date: 2026-02-15
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Optional
import time
from sklearn.neighbors import NearestNeighbors


class CDCPDTracker:
    """
    CDCPD2-style tracker with LLE regularization and edge constraints.
    
    This implements the CDCPD2 algorithm:
    1. CPD with LLE for smooth deformation
    2. QP post-optimization for edge length preservation
    """
    
    def __init__(
        self,
        # CPD parameters
        cpd_beta: float = 2.0,
        cpd_lambda: float = 1.0,
        cpd_w: float = 0.1,
        cpd_max_iter: int = 100,
        cpd_tol: float = 1e-3,  # 1mm tolerance for meter-scale data
        # LLE parameters
        lle_neighbors: int = 6,
        lle_gamma: float = 0.5,
        lle_reg: float = 1e-3,
        # Edge constraint parameters
        stretch_lambda: float = 1.3,  # Allow 30% stretch
        use_qp_optimization: bool = True,
        qp_max_iter: int = 200,
        # Anchor parameters
        use_anchor_constraints: bool = True,
        anchor_weight: float = 100.0,  # Weight for soft anchor (in CPD)
        anchor_hard: bool = False,  # Hard constraint in QP
    ):
        """
        Initialize CDCPD Tracker.
        
        Args:
            cpd_beta: Gaussian kernel width multiplier (scaled by data)
            cpd_lambda: CPD regularization weight (α in paper)
            cpd_w: Outlier weight [0, 1]
            cpd_max_iter: Maximum EM iterations
            cpd_tol: Convergence tolerance (1mm for meter-scale data)
            lle_neighbors: Number of neighbors for LLE
            lle_gamma: LLE regularization weight (γ in paper)
            lle_reg: LLE regularization for numerical stability
            stretch_lambda: Maximum allowed stretch ratio for edges
            use_qp_optimization: Whether to run QP after CPD
            qp_max_iter: Max iterations for QP solver
            use_anchor_constraints: Whether to use anchor constraints
            anchor_weight: Weight for soft anchor in CPD
            anchor_hard: Use hard anchor constraints in QP
        """
        self.cpd_beta = cpd_beta
        self.cpd_lambda = cpd_lambda
        self.cpd_w = cpd_w
        self.cpd_max_iter = cpd_max_iter
        self.cpd_tol = cpd_tol
        
        self.lle_neighbors = lle_neighbors
        self.lle_gamma = lle_gamma
        self.lle_reg = lle_reg
        
        self.stretch_lambda = stretch_lambda
        self.use_qp_optimization = use_qp_optimization
        self.qp_max_iter = qp_max_iter
        
        self.use_anchor_constraints = use_anchor_constraints
        self.anchor_weight = anchor_weight
        self.anchor_hard = anchor_hard
        
        # Cache for LLE matrix
        self._L_lle = None
        self._lle_template = None
    
    def compute_lle_matrix(self, Y: np.ndarray) -> np.ndarray:
        """
        Compute Locally Linear Embedding matrix L = (I - W)^T (I - W).
        
        This enforces that each point is reconstructed as a linear
        combination of its neighbors, preserving local structure.
        
        Args:
            Y: M × D template points
        
        Returns:
            L_lle: M × M LLE regularization matrix
        """
        M = len(Y)
        k = min(self.lle_neighbors, M - 1)
        
        if k <= 0:
            return np.eye(M)
        
        # Find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(Y)
        distances, indices = nbrs.kneighbors(Y)
        
        # Compute reconstruction weights
        W = np.zeros((M, M))
        
        for i in range(M):
            neighbors = indices[i, 1:]  # Exclude self
            
            # Local covariance matrix
            Z = Y[neighbors] - Y[i]
            C = Z @ Z.T
            
            # Regularization for numerical stability
            trace = np.trace(C)
            if trace > 0:
                C += self.lle_reg * trace * np.eye(k)
            else:
                C += self.lle_reg * np.eye(k)
            
            # Solve for weights
            try:
                w = np.linalg.solve(C, np.ones(k))
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(C, np.ones(k), rcond=None)[0]
            
            w = w / (np.sum(w) + 1e-10)
            W[i, neighbors] = w
        
        # L = (I - W)^T (I - W)
        IW = np.eye(M) - W
        L_lle = IW.T @ IW
        
        return L_lle
    
    def cpd_with_lle(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        anchor_indices: Optional[np.ndarray] = None,
        anchor_positions: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        CPD registration with LLE regularization (Phase 1 of CDCPD2).
        
        Solves: (G + λσ²D⁻¹ + σ²γL_lle)W = D⁻¹PX - Y - γL_lle·Y
        
        If anchors provided, adds anchor term to the linear system.
        
        Args:
            Y: M × D template (previous keypoints)
            X: N × D target (current skeleton points)
            anchor_indices: Indices of anchor points (optional)
            anchor_positions: 3D positions of anchors (optional)
        
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
        # Set beta based on median neighbor distance
        np.fill_diagonal(dist_Y, np.inf)
        median_dist = np.median(np.min(dist_Y, axis=1))
        beta = max(self.cpd_beta * median_dist, 0.01)
        
        G = np.exp(-np.sum(diff_Y ** 2, axis=2) / (2 * beta ** 2))
        
        # LLE matrix (computed once and cached)
        if self._L_lle is None or self._lle_template is None or not np.allclose(self._lle_template, Y):
            self._L_lle = self.compute_lle_matrix(Y)
            self._lle_template = Y.copy()
        L_lle = self._L_lle
        
        # Initialize sigma^2 - critical for good convergence
        diff_init = X[np.newaxis, :, :] - Y[:, np.newaxis, :]
        dist2_init = np.sum(diff_init ** 2, axis=2)
        min_dists = np.min(dist2_init, axis=1)
        sigma2 = np.median(min_dists) / 2.0
        sigma2 = max(sigma2, 1e-6)
        
        # Anchor setup
        use_anchors = (
            self.use_anchor_constraints 
            and anchor_indices is not None 
            and anchor_positions is not None
            and len(anchor_indices) > 0
        )
        if use_anchors:
            anchor_indices = np.asarray(anchor_indices).astype(int)
            anchor_positions = np.asarray(anchor_positions)
        
        info = {
            'iterations': 0,
            'sigma2_history': [],
            'cpd_loss': [],
            'beta_used': beta,
        }
        
        for iteration in range(self.cpd_max_iter):
            # ============================================================
            # E-step: Compute posterior probabilities (numerically stable)
            # ============================================================
            diff = X[np.newaxis, :, :] - T_Y[:, np.newaxis, :]
            dist2 = np.sum(diff ** 2, axis=2)  # M x N
            
            # Numerically stable softmax-style normalization
            log_p = -dist2 / (2 * sigma2)
            log_p_max = np.max(log_p, axis=0, keepdims=True)
            P_num = np.exp(log_p - log_p_max)
            
            c = (self.cpd_w / (1 - self.cpd_w + 1e-10)) * (M / N) * np.exp(-log_p_max)
            P_den = np.sum(P_num, axis=0, keepdims=True) + c
            P = P_num / (P_den + 1e-10)
            
            # ============================================================
            # M-step: Solve for W with LLE regularization
            # ============================================================
            P1 = np.sum(P, axis=1)
            Np = np.sum(P1)
            
            P1_safe = np.maximum(P1, 1e-10)
            D_inv = np.diag(1.0 / P1_safe)
            
            # CDCPD linear system with LLE:
            # (G + λσ²D⁻¹ + σ²γL_lle)W = D⁻¹PX - Y - γL_lle·Y
            A = G + self.cpd_lambda * sigma2 * D_inv + sigma2 * self.lle_gamma * L_lle
            B = D_inv @ P @ X - Y - self.lle_gamma * (L_lle @ Y)
            
            # Add soft anchor constraint to linear system
            if use_anchors:
                # Add anchor term: λ_anchor * I_anchor * (W) = λ_anchor * (anchor_pos - Y[anchor])
                # This modifies A and B for anchor indices
                for idx, pos in zip(anchor_indices, anchor_positions):
                    if 0 <= idx < M:
                        A[idx, idx] += self.anchor_weight
                        B[idx] += self.anchor_weight * (pos - Y[idx])
            
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
            if np.linalg.norm(T_Y_new - T_Y) < self.cpd_tol:
                T_Y = T_Y_new
                info['iterations'] = iteration + 1
                break
            
            T_Y = T_Y_new
            sigma2 = sigma2_new
            info['iterations'] = iteration + 1
        
        return T_Y, P, info
    
    def qp_edge_optimization(
        self,
        T_Y: np.ndarray,
        edges: List[Tuple[int, int]],
        reference_lengths: np.ndarray,
        anchor_indices: Optional[np.ndarray] = None,
        anchor_positions: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        QP optimization to enforce edge constraints (Phase 2 of CDCPD2).
        
        Solves:
            min ||Y_opt - T_Y||²
            s.t. ||Y_opt[i] - Y_opt[j]||² ≤ (stretch_λ · l_ij)² for edges
                 Y_opt[anchor] = anchor_pos (if hard constraints)
        
        Uses scipy.optimize since we don't have Gurobi.
        For inequality constraints, we use SLSQP.
        
        Args:
            T_Y: M × D CPD result
            edges: List of (i, j) edge tuples
            reference_lengths: Reference length for each edge
            anchor_indices: Anchor point indices (optional)
            anchor_positions: Anchor positions (optional)
        
        Returns:
            Y_opt: M × D optimized points
            info: Optimization info
        """
        M, D = T_Y.shape
        
        # Handle anchors
        use_hard_anchors = (
            self.anchor_hard 
            and anchor_indices is not None 
            and anchor_positions is not None
            and len(anchor_indices) > 0
        )
        
        if use_hard_anchors:
            anchor_indices = np.asarray(anchor_indices).astype(int)
            anchor_positions = np.asarray(anchor_positions)
        
        # Objective: min ||Y_opt - T_Y||²
        def objective(y_flat):
            y = y_flat.reshape(M, D)
            return np.sum((y - T_Y) ** 2)
        
        def grad_objective(y_flat):
            y = y_flat.reshape(M, D)
            grad = 2 * (y - T_Y)
            return grad.flatten()
        
        # Edge constraints: ||y[i] - y[j]||² ≤ (stretch_λ · l_ij)²
        # Reformulated as: (stretch_λ · l_ij)² - ||y[i] - y[j]||² ≥ 0
        constraints = []
        
        for edge_idx, (i, j) in enumerate(edges):
            if i >= M or j >= M:
                continue
            max_len_sq = (self.stretch_lambda * reference_lengths[edge_idx]) ** 2
            
            def edge_constraint(y_flat, i=i, j=j, max_len_sq=max_len_sq):
                y = y_flat.reshape(M, D)
                return max_len_sq - np.sum((y[i] - y[j]) ** 2)
            
            constraints.append({
                'type': 'ineq',
                'fun': edge_constraint
            })
        
        # Hard anchor constraints (equality)
        if use_hard_anchors:
            for idx, pos in zip(anchor_indices, anchor_positions):
                if 0 <= idx < M:
                    for d in range(D):
                        def anchor_constraint(y_flat, idx=idx, d=d, val=pos[d]):
                            y = y_flat.reshape(M, D)
                            return y[idx, d] - val
                        
                        constraints.append({
                            'type': 'eq',
                            'fun': anchor_constraint
                        })
        
        # Initial guess
        y0 = T_Y.flatten()
        
        # Solve QP using SLSQP
        t0 = time.time()
        result = minimize(
            objective,
            y0,
            method='SLSQP',
            jac=grad_objective,
            constraints=constraints,
            options={
                'maxiter': self.qp_max_iter,
                'ftol': 1e-6,
                'disp': False,
            }
        )
        
        Y_opt = result.x.reshape(M, D)
        
        info = {
            'success': result.success,
            'iterations': result.nit,
            'objective': result.fun,
            'time': time.time() - t0,
            'message': result.message,
        }
        
        return Y_opt, info
    
    def track_frame(
        self,
        prev_keypoints: np.ndarray,
        skeleton_pc: np.ndarray,
        detected_branch_3d: np.ndarray,
        detected_leaf_3d: np.ndarray,
        reference_edges: List[Tuple[int, int]],
        reference_lengths: np.ndarray,
        n_branch: int,
        n_leaf: int,
        cpd_downsample: int = 500,
    ) -> Dict:
        """
        Track keypoints using CDCPD2 method.
        
        Phase 1: CPD with LLE regularization
        Phase 2: QP optimization for edge constraints
        
        Args:
            prev_keypoints: K × 3 previous frame keypoints
            skeleton_pc: N × 3 current skeleton point cloud
            detected_branch_3d: B × 3 detected branch points (used as anchors)
            detected_leaf_3d: L × 3 detected leaf points (used as anchors)
            reference_edges: List of (i, j) edge tuples
            reference_lengths: Reference length for each edge
            n_branch: Number of branch nodes
            n_leaf: Number of leaf nodes  
            cpd_downsample: Max points for CPD target
        
        Returns:
            Dict with keypoints, timing, convergence info
        """
        timing = {}
        t_total = time.time()
        
        # Downsample target
        cpd_target = skeleton_pc
        if len(cpd_target) > cpd_downsample:
            indices = np.random.choice(len(cpd_target), cpd_downsample, replace=False)
            cpd_target = cpd_target[indices]
        
        # ========================================
        # Setup anchors (branch and leaf points)
        # ========================================
        anchor_indices = []
        anchor_positions = []
        K = len(prev_keypoints)
        
        # Match detected branch points to branch keypoints
        if len(detected_branch_3d) > 0 and n_branch > 0:
            branch_kp = prev_keypoints[:n_branch]
            for det in detected_branch_3d:
                dists = np.linalg.norm(branch_kp - det, axis=1)
                best_idx = np.argmin(dists)
                if dists[best_idx] < 0.1:  # 10cm threshold
                    anchor_indices.append(best_idx)
                    anchor_positions.append(det)
        
        # Match detected leaf points to leaf keypoints
        if len(detected_leaf_3d) > 0 and n_leaf > 0:
            leaf_kp = prev_keypoints[n_branch:n_branch + n_leaf]
            for det in detected_leaf_3d:
                dists = np.linalg.norm(leaf_kp - det, axis=1)
                best_idx = np.argmin(dists)
                if dists[best_idx] < 0.1:  # 10cm threshold
                    anchor_indices.append(n_branch + best_idx)
                    anchor_positions.append(det)
        
        anchor_indices = np.array(anchor_indices) if anchor_indices else None
        anchor_positions = np.array(anchor_positions) if anchor_positions else None
        
        # ========================================
        # Phase 1: CPD with LLE
        # ========================================
        t0 = time.time()
        T_Y, P, cpd_info = self.cpd_with_lle(
            prev_keypoints,
            cpd_target,
            anchor_indices,
            anchor_positions,
        )
        timing['cpd_lle'] = time.time() - t0
        
        # ========================================
        # Phase 2: QP optimization (optional)
        # ========================================
        if self.use_qp_optimization and len(reference_edges) > 0:
            t0 = time.time()
            keypoints, qp_info = self.qp_edge_optimization(
                T_Y,
                reference_edges,
                reference_lengths,
                anchor_indices,
                anchor_positions,
            )
            timing['qp'] = time.time() - t0
        else:
            keypoints = T_Y
            qp_info = {'success': True, 'iterations': 0}
        
        timing['total'] = time.time() - t_total
        
        # Compute edge errors for diagnostics
        edge_errors = []
        for idx, (i, j) in enumerate(reference_edges):
            if i < len(keypoints) and j < len(keypoints):
                curr_len = np.linalg.norm(keypoints[i] - keypoints[j])
                ref_len = reference_lengths[idx]
                edge_errors.append(abs(curr_len - ref_len) / (ref_len + 1e-6))
        
        return {
            'keypoints': keypoints,
            'timing': timing,
            'cpd_info': cpd_info,
            'qp_info': qp_info,
            'T_Y_cpd': T_Y,  # CPD output before QP
            'edge_errors': np.array(edge_errors) if edge_errors else np.array([0.0]),
            'n_anchors': len(anchor_indices) if anchor_indices is not None else 0,
        }
    
    def track_frame_with_anchors(
        self,
        prev_keypoints: np.ndarray,
        skeleton_pc: np.ndarray,
        anchor_indices: Optional[np.ndarray],
        anchor_positions: Optional[np.ndarray],
        reference_edges: List[Tuple[int, int]],
        reference_lengths: np.ndarray,
        cpd_downsample: int = 500,
    ) -> Dict:
        """
        Track keypoints using CDCPD2 with explicit anchor constraints.
        
        This is a simplified interface that directly takes anchor indices/positions
        (e.g., from robot EE poses) instead of detecting them from the skeleton.
        
        Args:
            prev_keypoints: K × 3 previous frame keypoints
            skeleton_pc: N × 3 current skeleton point cloud
            anchor_indices: Indices of anchor points (e.g., EE leaf nodes)
            anchor_positions: 3D positions of anchors (e.g., EE poses)
            reference_edges: List of (i, j) edge tuples
            reference_lengths: Reference length for each edge
            cpd_downsample: Max points for CPD target
        
        Returns:
            Dict with keypoints, timing, convergence info
        """
        timing = {}
        t_total = time.time()
        
        # Downsample target
        cpd_target = skeleton_pc
        if len(cpd_target) > cpd_downsample:
            indices = np.random.choice(len(cpd_target), cpd_downsample, replace=False)
            cpd_target = cpd_target[indices]
        
        # Phase 1: CPD with LLE
        t0 = time.time()
        T_Y, P, cpd_info = self.cpd_with_lle(
            prev_keypoints,
            cpd_target,
            anchor_indices,
            anchor_positions,
        )
        timing['cpd_lle'] = time.time() - t0
        
        # Phase 2: QP optimization (optional)
        if self.use_qp_optimization and len(reference_edges) > 0:
            t0 = time.time()
            keypoints, qp_info = self.qp_edge_optimization(
                T_Y,
                reference_edges,
                reference_lengths,
                anchor_indices,
                anchor_positions,
            )
            timing['qp'] = time.time() - t0
        else:
            keypoints = T_Y
            qp_info = {'success': True, 'iterations': 0}
        
        timing['total'] = time.time() - t_total
        
        # Compute edge errors for diagnostics
        edge_errors = []
        for idx, (i, j) in enumerate(reference_edges):
            if i < len(keypoints) and j < len(keypoints):
                curr_len = np.linalg.norm(keypoints[i] - keypoints[j])
                ref_len = reference_lengths[idx]
                edge_errors.append(abs(curr_len - ref_len) / (ref_len + 1e-6))
        
        return {
            'keypoints': keypoints,
            'timing': timing,
            'cpd_info': cpd_info,
            'qp_info': qp_info,
            'T_Y_cpd': T_Y,
            'edge_errors': np.array(edge_errors) if edge_errors else np.array([0.0]),
            'n_anchors': len(anchor_indices) if anchor_indices is not None else 0,
        }


def create_tracker(config: Dict = None) -> CDCPDTracker:
    """Factory function to create CDCPDTracker with optional config."""
    if config is None:
        config = {}
    return CDCPDTracker(**config)
