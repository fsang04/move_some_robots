"""
Wire Initialization Main Script

Uses WireInitializer class to perform Frame 0 initialization.

Outputs:
    - Initialization results (.npy)
    - 2D visualization (skeleton + keypoints)
    - Segment length boxplot

Usage:
    python wire_init_main.py --traj 1
    python wire_init_main.py --traj 2
    python wire_init_main.py --traj 3

Author: Auto-generated
Date: 2026-02-21
"""

import argparse
import time
import io
import json
import numpy as np
import cv2
import PIL.Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

from wire_initializer_combined import WireInitializer

# Segment colors (different for each segment)
SEGMENT_COLORS = [
    [255, 0, 0],      # Red - Segment 0 (ee_leaf to b0)
    [0, 0, 255],      # Blue - Segment 1 (ee_leaf to b1)
    [0, 255, 0],      # Green - Segment 2 (free_leaf from b0)
    [255, 165, 0],    # Orange - Segment 3 (free_leaf from b1)
    [128, 0, 128],    # Purple - Segment 4 (trunk)
    [0, 255, 255],    # Cyan - Extra segments if any
    [255, 0, 255],    # Magenta
]


def create_2d_visualization(
    rgb: np.ndarray,
    foreground_mask: np.ndarray,
    skeleton_mask: np.ndarray,
    keypoints_2d: np.ndarray,
    edges: list,
    segment_edges: list = None,
    n_branch: int = 0,
    n_leaf: int = 0,
    frame_idx: int = 0,
    mst_skeleton_mask: np.ndarray = None,
    skeleton_mask_raw: np.ndarray = None,
    segment_3d_lengths: list = None,
    skeleton_segment_lengths: list = None,
) -> np.ndarray:
    """
    Create 3x2 2D visualization grid with segment-colored edges.
    
    Layout:
        [Skeleton]         [Skeleton Overlay]
        [MST Skeleton]     [MST Overlay]
        [Keypoints Only]   [Keypoints Overlay]
    
    Args:
        rgb: H × W × 3 RGB image
        foreground_mask: H × W binary mask
        skeleton_mask: H × W skeleton mask
        keypoints_2d: K × 2 keypoint pixel coords (row, col)
        edges: List of (i, j) edge tuples
        segment_edges: List of segment edge lists (for coloring)
        n_branch: Number of branch nodes
        n_leaf: Number of leaf nodes
        frame_idx: Frame index for label
        mst_skeleton_mask: H × W MST skeleton mask (subset of skeleton used for FPS)
    
    Returns:
        grid: Visualization grid
    """
    H, W = rgb.shape[:2]
    
    # Colors
    SKELETON_COLOR = [0, 191, 255]  # Deep sky blue
    MST_COLOR = [0, 255, 128]       # Green for MST skeleton
    PRUNED_COLOR = [80, 80, 80]     # Dim gray for pruned skeleton pixels
    BRANCH_COLOR = [128, 0, 128]    # Purple for branch nodes
    LEAF_COLOR = [255, 255, 0]      # Yellow for leaf nodes
    INTER_COLOR = [255, 165, 0]     # Orange for intermediate nodes
    
    # Visualization parameters
    KEYPOINT_RADIUS = 8
    EDGE_THICKNESS = 3
    
    # Build edge to segment mapping for coloring
    edge_to_segment = {}
    if segment_edges is not None:
        for seg_idx, seg in enumerate(segment_edges):
            for e in seg:
                edge_key = (min(e[0], e[1]), max(e[0], e[1]))
                edge_to_segment[edge_key] = seg_idx
    
    def get_keypoint_color(idx):
        """Get color based on keypoint type."""
        if idx < n_branch:
            return BRANCH_COLOR
        elif idx < n_branch + n_leaf:
            return LEAF_COLOR
        else:
            return INTER_COLOR
    
    def get_edge_color(i, j):
        """Get color based on segment membership."""
        edge_key = (min(i, j), max(i, j))
        seg_idx = edge_to_segment.get(edge_key, 0)
        return SEGMENT_COLORS[seg_idx % len(SEGMENT_COLORS)]
    
    # Row 1: Skeleton + Skeleton overlay
    # Row 1: Raw skeleton (before pruning) to show spurs
    raw_skel = skeleton_mask_raw if skeleton_mask_raw is not None else skeleton_mask
    raw_skel_thick = cv2.dilate(raw_skel, np.ones((3, 3), np.uint8), iterations=1)
    
    skeleton_vis = np.zeros((H, W, 3), dtype=np.uint8)
    skeleton_vis[raw_skel_thick > 0] = SKELETON_COLOR
    # Overlay pruned skeleton in green to show what was kept
    pruned_thick = cv2.dilate(skeleton_mask, np.ones((3, 3), np.uint8), iterations=1)
    skeleton_vis[pruned_thick > 0] = MST_COLOR  # Green = kept
    # Remaining blue = pruned spurs
    
    skeleton_overlay = rgb.copy()
    raw_skel_thick_overlay = cv2.dilate(raw_skel, np.ones((5, 5), np.uint8), iterations=1)
    pruned_thick_overlay = cv2.dilate(skeleton_mask, np.ones((5, 5), np.uint8), iterations=1)
    skeleton_overlay[raw_skel_thick_overlay > 0] = PRUNED_COLOR  # Dim gray = pruned
    skeleton_overlay[pruned_thick_overlay > 0] = SKELETON_COLOR  # Blue = kept
    
    n_raw = int(np.sum(raw_skel > 0))
    n_pruned = int(np.sum(skeleton_mask > 0))
    cv2.putText(skeleton_vis, f"Skeleton (green={n_pruned}px kept, blue={n_raw - n_pruned}px pruned)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(skeleton_overlay, f"Frame {frame_idx}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Row 2: MST Skeleton + MST Overlay
    mst_vis = np.zeros((H, W, 3), dtype=np.uint8)
    mst_overlay = rgb.copy()
    
    if mst_skeleton_mask is not None:
        # Show pruned skeleton dimmed, MST skeleton bright
        mst_vis[pruned_thick > 0] = PRUNED_COLOR
        mst_thick = cv2.dilate(mst_skeleton_mask, np.ones((3, 3), np.uint8), iterations=1)
        mst_vis[mst_thick > 0] = MST_COLOR
        
        mst_overlay_thick = cv2.dilate(mst_skeleton_mask, np.ones((5, 5), np.uint8), iterations=1)
        # Dim pruned skeleton on overlay
        skel_only = (pruned_thick_overlay > 0) & (mst_overlay_thick == 0)
        mst_overlay[skel_only] = (mst_overlay[skel_only] * 0.5 + np.array(PRUNED_COLOR) * 0.5).astype(np.uint8)
        mst_overlay[mst_overlay_thick > 0] = MST_COLOR
        
        # Draw keypoints on MST panels to show anchor positions
        if keypoints_2d is not None and len(keypoints_2d) > 0:
            kp_int = keypoints_2d.astype(int)
            for idx, (row, col) in enumerate(kp_int):
                if 0 <= row < H and 0 <= col < W:
                    color = get_keypoint_color(idx)
                    cv2.circle(mst_vis, (col, row), KEYPOINT_RADIUS, color, -1)
                    cv2.circle(mst_overlay, (col, row), KEYPOINT_RADIUS, color, -1)
                    cv2.putText(mst_vis, str(idx), (col + 10, row),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        n_skel = np.sum(skeleton_mask > 0)
        n_mst = np.sum(mst_skeleton_mask > 0)
        cv2.putText(mst_vis, f"MST Skeleton ({n_mst}/{n_skel} px)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        mst_vis[pruned_thick > 0] = SKELETON_COLOR
        cv2.putText(mst_vis, "MST Skeleton (N/A)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(mst_overlay, f"Frame {frame_idx} - MST", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Row 3: Keypoints + Keypoints overlay
    keypoint_vis = np.zeros((H, W, 3), dtype=np.uint8)
    keypoint_vis[pruned_thick > 0] = [50, 50, 50]  # Dim skeleton background
    
    keypoint_overlay = rgb.copy()
    
    # Draw edges and keypoints
    if keypoints_2d is not None and len(keypoints_2d) > 0 and edges is not None:
        kp_int = keypoints_2d.astype(int)
        
        # Draw edges with segment colors
        for (i, j) in edges:
            if i < len(kp_int) and j < len(kp_int):
                pt1 = (kp_int[i, 1], kp_int[i, 0])  # (col, row)
                pt2 = (kp_int[j, 1], kp_int[j, 0])
                edge_color = get_edge_color(i, j)
                cv2.line(keypoint_vis, pt1, pt2, edge_color, EDGE_THICKNESS)
                cv2.line(keypoint_overlay, pt1, pt2, edge_color, EDGE_THICKNESS)
        
        # Draw keypoints
        for idx, (row, col) in enumerate(kp_int):
            if 0 <= row < H and 0 <= col < W:
                color = get_keypoint_color(idx)
                cv2.circle(keypoint_vis, (col, row), KEYPOINT_RADIUS, color, -1)
                cv2.circle(keypoint_overlay, (col, row), KEYPOINT_RADIUS, color, -1)
                
                # Add keypoint index label
                cv2.putText(keypoint_vis, str(idx), (col + 10, row),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(keypoint_overlay, str(idx), (col + 10, row),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.putText(keypoint_vis, "Keypoints", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(keypoint_overlay, f"Frame {frame_idx}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add segment legend on keypoint overlay with lengths
    legend_y = 60
    seg_names = ["Seg0: ee→b0", "Seg1: ee→b1", "Seg2: b0→free", "Seg3: b1→free", "Seg4: trunk"]
    for seg_idx, seg_name in enumerate(seg_names):
        color = SEGMENT_COLORS[seg_idx % len(SEGMENT_COLORS)]
        cv2.line(keypoint_overlay, (10, legend_y + seg_idx * 20), (30, legend_y + seg_idx * 20), color, 3)
        
        # Build label with lengths if available
        label = seg_name
        if segment_3d_lengths is not None and seg_idx < len(segment_3d_lengths):
            edge_len = segment_3d_lengths[seg_idx]
            label += f" [{edge_len:.0f}mm"
            if skeleton_segment_lengths is not None and seg_idx < len(skeleton_segment_lengths):
                skel_len = skeleton_segment_lengths[seg_idx]
                label += f"/{skel_len:.0f}mm"
            label += "]"
        
        cv2.putText(keypoint_overlay, label, (35, legend_y + seg_idx * 20 + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Create 3×2 grid
    row1 = np.concatenate([skeleton_vis, skeleton_overlay], axis=1)
    row2 = np.concatenate([mst_vis, mst_overlay], axis=1)
    row3 = np.concatenate([keypoint_vis, keypoint_overlay], axis=1)
    
    grid = np.concatenate([row1, row2, row3], axis=0)
    
    return grid


def compute_segment_lengths(keypoints: np.ndarray, segment_edges: list) -> list:
    """Compute the length of each segment (sum of edge lengths)."""
    seg_lengths = []
    for seg in segment_edges:
        total_len = 0.0
        for (i, j) in seg:
            edge_len = np.linalg.norm(keypoints[i] - keypoints[j])
            total_len += edge_len
        seg_lengths.append(total_len)
    return seg_lengths


def get_edge_lengths(keypoints: np.ndarray, edges: list) -> np.ndarray:
    """Get length of each edge. Returns array of shape (n_edges,)."""
    return np.array([np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in edges])


def get_segment_edge_lengths(keypoints: np.ndarray, segment_edges: list) -> list:
    """
    Get per-segment per-position edge lengths.
    
    Returns:
        List of lists: segment_edge_lengths[seg_idx][edge_pos] = length in mm.
        Edge position is consistent across frames (0th edge from start anchor, etc.)
    """
    result = []
    for seg in segment_edges:
        seg_lens = [np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in seg]
        result.append(seg_lens)
    return result


def compute_boxplot_stats(data: np.ndarray, name: str) -> dict:
    """Compute boxplot statistics for a 1D array."""
    return {
        'name': name,
        'n': len(data),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'Q1_25': float(np.percentile(data, 25)),
        'median_50': float(np.percentile(data, 50)),
        'Q3_75': float(np.percentile(data, 75)),
        'p2_5': float(np.percentile(data, 2.5)),
        'p97_5': float(np.percentile(data, 97.5)),
        'p5': float(np.percentile(data, 5)),
        'p95': float(np.percentile(data, 95)),
    }


def save_boxplot_stats(stats_list: list, output_path: Path) -> None:
    """Save boxplot statistics to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(stats_list, f, indent=2)
    print(f"Saved boxplot stats to: {output_path}")


def create_segment_length_boxplot(
    all_segment_lengths: dict,
    output_path: Path,
    traj: int,
) -> None:
    """
    Create boxplot of segment lengths across all frames (no reference).
    Also saves stats to a JSON file alongside the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    seg_names = ["Seg0\nee→b0", "Seg1\nee→b1", "Seg2\nb0→free", "Seg3\nb1→free", "Seg4\ntrunk"]
    colors = ['#FF0000', '#0000FF', '#00FF00', '#FFA500', '#800080']
    
    data = []
    positions = []
    stats_list = []
    for seg_idx in range(5):
        if seg_idx in all_segment_lengths and len(all_segment_lengths[seg_idx]) > 0:
            arr = np.array(all_segment_lengths[seg_idx])
            data.append(arr)
            positions.append(seg_idx)
            stats_list.append(compute_boxplot_stats(arr, f"Seg{seg_idx}"))
    
    if data:
        bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    
    ax.set_xticks(range(5))
    ax.set_xticklabels(seg_names)
    ax.set_ylabel('Segment Length (mm)')
    ax.set_title(f'Trajectory {traj}: Segment Length Distribution ({len(data[0]) if data else 0} frames)')
    ax.grid(True, alpha=0.3)
    
    # Add stats text
    stats_text = ""
    for s in stats_list:
        stats_text += f"{s['name']}: μ={s['mean']:.1f}, σ={s['std']:.1f}\n"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved segment length boxplot to: {output_path}")
    
    # Save stats JSON
    stats_path = output_path.with_suffix('.json')
    save_boxplot_stats(stats_list, stats_path)


def create_edge_length_boxplot(
    topology_match_seg_edge_lengths: dict,
    ref_seg_edge_lengths: list,
    output_path: Path,
    traj: int,
) -> None:
    """
    Create boxplot of per-edge lengths across topology-matching frames,
    traced by segment position (Seg0 edge0, Seg0 edge1, ..., Seg1 edge0, ...).
    One box per edge position, colored by segment.
    Also saves stats to a JSON file.
    
    Args:
        topology_match_seg_edge_lengths: {seg_idx: {edge_pos: [lengths across frames]}}
        ref_seg_edge_lengths: list of lists, reference per-segment per-position lengths
        output_path: Path to save the plot
        traj: Trajectory number
    """
    # Flatten to ordered list: Seg0-pos0, Seg0-pos1, ..., Seg1-pos0, ...
    data = []
    labels = []
    seg_indices = []
    for seg_idx in sorted(topology_match_seg_edge_lengths.keys()):
        for edge_pos in sorted(topology_match_seg_edge_lengths[seg_idx].keys()):
            lengths = topology_match_seg_edge_lengths[seg_idx][edge_pos]
            data.append(lengths)
            labels.append(f"S{seg_idx}E{edge_pos}")
            seg_indices.append(seg_idx)
    
    n_edges = len(data)
    if n_edges == 0:
        print("No edge data — skipping edge length boxplot.")
        return
    
    n_frames = len(data[0])
    seg_colors = ['#FF0000', '#0000FF', '#00FF00', '#FFA500', '#800080', '#00FFFF', '#FF00FF']
    
    fig, ax = plt.subplots(figsize=(max(14, n_edges * 0.7), 6))
    
    bp = ax.boxplot(data, patch_artist=True, widths=0.6)
    
    stats_list = []
    for e in range(n_edges):
        seg_idx = seg_indices[e]
        color = seg_colors[seg_idx % len(seg_colors)]
        bp['boxes'][e].set_facecolor(color)
        bp['boxes'][e].set_alpha(0.5)
        stats_list.append(compute_boxplot_stats(np.array(data[e]), f"{labels[e]}"))
    
    # X-axis labels
    ax.set_xticks(range(1, n_edges + 1))
    ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.set_ylabel('Edge Length (mm)')
    ax.set_title(f'Trajectory {traj}: Per-Edge Length Distribution ({n_frames} topology-matching frames)')
    ax.grid(True, alpha=0.3)
    
    # Add segment color legend
    seg_names = ["Seg0: ee→b0", "Seg1: ee→b1", "Seg2: b0→free", "Seg3: b1→free", "Seg4: trunk"]
    legend_patches = []
    for seg_idx, seg_name in enumerate(seg_names):
        from matplotlib.patches import Patch
        legend_patches.append(Patch(facecolor=seg_colors[seg_idx], alpha=0.5, label=seg_name))
    ax.legend(handles=legend_patches, loc='upper right', fontsize=7)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved edge length boxplot to: {output_path}")
    
    # Save stats JSON
    stats_path = output_path.with_suffix('.json')
    save_boxplot_stats(stats_list, stats_path)


def compute_edge_length_error_metrics(
    seg_edge_lengths: dict,
    ref_seg_edge_lengths: list,
    output_path: Path,
    traj: int,
) -> dict:
    """
    Compute edge length error metrics (RMSE, %error) overall and per-segment.
    Also computes metrics after removing outliers using IQR method.
    
    Args:
        seg_edge_lengths: {seg_idx: {edge_pos: [lengths across frames]}}
        ref_seg_edge_lengths: list of lists, reference per-segment per-position lengths
        output_path: Path to save the JSON metrics
        traj: Trajectory number
    
    Returns:
        dict with all computed metrics
    """
    seg_names = ["ee→b0", "ee→b1", "b0→free", "b1→free", "trunk"]
    
    # Flatten data structure: collect errors per segment and overall
    # seg_errors[seg_idx] = list of (error, ref_len) tuples for all edges in segment
    seg_errors = {i: [] for i in range(5)}
    all_errors_flat = []  # (error, ref_len) for all edges
    all_lengths_flat = []  # raw lengths for outlier detection
    
    n_frames = 0
    for seg_idx in sorted(seg_edge_lengths.keys()):
        for edge_pos in sorted(seg_edge_lengths[seg_idx].keys()):
            lengths = seg_edge_lengths[seg_idx][edge_pos]  # [frame0, frame1, ...]
            n_frames = max(n_frames, len(lengths))
            ref_len = lengths[0]  # Frame 0 reference
            
            for frame_len in lengths[1:]:  # exclude Frame 0
                error = frame_len - ref_len
                seg_errors[seg_idx].append((error, ref_len))
                all_errors_flat.append((error, ref_len))
                all_lengths_flat.append(frame_len)
    
    if len(all_errors_flat) == 0:
        print("No edge length data — cannot compute error metrics.")
        return {}
    
    n_frames_excl0 = n_frames - 1
    n_total_points = len(all_errors_flat)
    
    # --- Compute metrics WITH all data ---
    def compute_rmse_pct(error_ref_list):
        """Compute RMSE and mean |%error| from list of (error, ref_len) tuples."""
        if len(error_ref_list) == 0:
            return 0.0, 0.0
        errors = np.array([e for e, r in error_ref_list])
        refs = np.array([r for e, r in error_ref_list])
        rmse = np.sqrt(np.mean(errors ** 2))
        refs_safe = np.maximum(refs, 1e-6)
        pct_errors = np.abs(errors) / refs_safe * 100
        mean_pct = np.mean(pct_errors)
        return float(rmse), float(mean_pct)
    
    overall_rmse, overall_pct = compute_rmse_pct(all_errors_flat)
    
    per_segment_metrics = []
    for seg_idx in range(5):
        if seg_idx in seg_errors and len(seg_errors[seg_idx]) > 0:
            rmse, pct = compute_rmse_pct(seg_errors[seg_idx])
            per_segment_metrics.append({
                'segment': seg_idx,
                'name': seg_names[seg_idx],
                'n_data_points': len(seg_errors[seg_idx]),
                'rmse_mm': rmse,
                'mean_pct_error': pct,
            })
    
    # --- Outlier detection using IQR on raw lengths ---
    all_lengths_arr = np.array(all_lengths_flat)
    Q1 = np.percentile(all_lengths_arr, 25)
    Q3 = np.percentile(all_lengths_arr, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out outliers
    all_errors_no_outliers = []
    seg_errors_no_outliers = {i: [] for i in range(5)}
    n_outliers = 0
    
    idx = 0
    for seg_idx in sorted(seg_edge_lengths.keys()):
        for edge_pos in sorted(seg_edge_lengths[seg_idx].keys()):
            lengths = seg_edge_lengths[seg_idx][edge_pos]
            ref_len = lengths[0]
            
            for frame_len in lengths[1:]:
                is_outlier = (frame_len < lower_bound) or (frame_len > upper_bound)
                if is_outlier:
                    n_outliers += 1
                else:
                    error = frame_len - ref_len
                    seg_errors_no_outliers[seg_idx].append((error, ref_len))
                    all_errors_no_outliers.append((error, ref_len))
                idx += 1
    
    outlier_pct = (n_outliers / n_total_points * 100) if n_total_points > 0 else 0.0
    
    # --- Compute metrics WITHOUT outliers ---
    overall_rmse_no_outliers, overall_pct_no_outliers = compute_rmse_pct(all_errors_no_outliers)
    
    for seg_metric in per_segment_metrics:
        seg_idx = seg_metric['segment']
        if len(seg_errors_no_outliers[seg_idx]) > 0:
            rmse, pct = compute_rmse_pct(seg_errors_no_outliers[seg_idx])
            seg_metric['n_data_points_no_outliers'] = len(seg_errors_no_outliers[seg_idx])
            seg_metric['rmse_mm_no_outliers'] = rmse
            seg_metric['mean_pct_error_no_outliers'] = pct
        else:
            seg_metric['n_data_points_no_outliers'] = 0
            seg_metric['rmse_mm_no_outliers'] = 0.0
            seg_metric['mean_pct_error_no_outliers'] = 0.0
    
    # Build final metrics dict
    metrics = {
        'trajectory': traj,
        'n_frames': n_frames,
        'n_frames_excl_ref': n_frames_excl0,
        'n_total_data_points': n_total_points,
        'overall': {
            'rmse_mm': overall_rmse,
            'mean_pct_error': overall_pct,
        },
        'outlier_detection': {
            'method': 'IQR',
            'Q1': float(Q1),
            'Q3': float(Q3),
            'IQR': float(IQR),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'n_outliers': n_outliers,
            'n_total_points': n_total_points,
            'outlier_pct': outlier_pct,
        },
        'overall_no_outliers': {
            'rmse_mm': overall_rmse_no_outliers,
            'mean_pct_error': overall_pct_no_outliers,
            'n_data_points': len(all_errors_no_outliers),
        },
        'per_segment': per_segment_metrics,
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved edge length error metrics to: {output_path}")
    
    # Print summary
    print(f"\nEdge length error metrics (n_frames={n_frames_excl0}, n_data_points={n_total_points}):")
    print(f"  Overall RMSE: {overall_rmse:.2f} mm")
    print(f"  Overall Mean |%error|: {overall_pct:.1f}%")
    print(f"\n  Per-segment (with all data):")
    for seg_metric in per_segment_metrics:
        print(f"    {seg_metric['name']}: RMSE={seg_metric['rmse_mm']:.2f} mm, "
              f"|%error|={seg_metric['mean_pct_error']:.1f}%")
    
    print(f"\n  Outliers (IQR method): {n_outliers}/{n_total_points} ({outlier_pct:.1f}%)")
    print(f"    Bounds: [{lower_bound:.1f}, {upper_bound:.1f}] mm")
    
    print(f"\n  After removing outliers:")
    print(f"    Overall RMSE: {overall_rmse_no_outliers:.2f} mm")
    print(f"    Overall Mean |%error|: {overall_pct_no_outliers:.1f}%")
    print(f"    Per-segment:")
    for seg_metric in per_segment_metrics:
        print(f"      {seg_metric['name']}: RMSE={seg_metric['rmse_mm_no_outliers']:.2f} mm, "
              f"|%error|={seg_metric['mean_pct_error_no_outliers']:.1f}%")
    
    return metrics


def create_video_from_frames(
    output_dir: Path,
    output_path: Path,
    fps: int = 10,
) -> None:
    """
    Create a video from saved 2D visualization frames.
    
    Args:
        output_dir: Directory containing the init_2d_*.png frames
        output_path: Path to save the output video
        fps: Frames per second for the video
    """
    # Find all 2D visualization frames
    frame_files = sorted(output_dir.glob("init_2d_*.png"))
    
    if len(frame_files) == 0:
        print("No frames found to create video.")
        return
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        print(f"Could not read first frame: {frame_files[0]}")
        return
    
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Write frames to video
    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        if frame is not None:
            video_writer.write(frame)
    
    video_writer.release()
    print(f"Saved video ({len(frame_files)} frames, {fps} fps) to: {output_path}")


def main():
    """Main initialization script."""
    
    # ================================================================
    # Parse Arguments
    # ================================================================
    
    parser = argparse.ArgumentParser(description='Wire Initialization (all frames)')
    parser.add_argument('--traj', type=int, required=True, choices=[1, 2, 3, 7],
                        help='Trajectory number (1, 2, 3, or 7)')
    parser.add_argument('--pipeline', type=str, default='original', choices=['original', 'streamlined'],
                        help='Pipeline to use: original (9-phase) or streamlined (7-phase)')
    parser.add_argument('--method', type=str, default='fps', choices=['fps', 'gmm'],
                        help='Keypoint placement method: fps or gmm')
    parser.add_argument('--n_keypoints', type=int, default=21,
                        help='Number of keypoints to extract')
    parser.add_argument('--target_branch', type=int, default=2,
                        help='Target number of branch nodes')
    parser.add_argument('--target_leaf', type=int, default=4,
                        help='Target number of leaf nodes')
    parser.add_argument('--frame_step', type=int, default=1,
                        help='Process every N frames (default: 1 = all frames)')
    parser.add_argument('--start', type=int, default=0,
                        help='Start frame index (default: 0)')
    parser.add_argument('--end', type=int, default=-1,
                        help='End frame index, -1 for all frames (default: -1)')
    parser.add_argument('--min_mst_pixels', type=int, default=850,
                        help='Minimum MST skeleton pixels to accept a frame (default: 960)')
    parser.add_argument('--keypoints_per_segment', type=int, nargs=5, default=None,
                        help='Intermediate keypoints per segment: [ee0, ee1, free0, free1, trunk] (default: auto)')
    parser.add_argument('--no_repulsion', action='store_true',
                        help='Disable repulsion relaxation (for ablation study)')
    args = parser.parse_args()
    
    # ================================================================
    # Configuration
    # ================================================================
    
    # Camera intrinsics
    intrinsics = np.array([
        [606.1124267578125, 0, 641.7578125],
        [0, 605.8821411132812, 365.6518859863281],
        [0, 0, 1]
    ])
    
    # Set data paths based on trajectory
    repulsion_suffix = "_no_repulsion" if args.no_repulsion else ""
    if args.traj == 1:
        arm_data_path = Path("./data/arm_traj1/arm_traj1.npy")
        full_data_path = Path("./data/arm_traj1/arm_with_wires_traj1.npy")
        output_dir = Path(f"./data/arm_traj1/wire_init_output_{args.pipeline}_{args.method}{repulsion_suffix}")
        precomputed_mask_dir = None
        ee_pose_path = Path("./data/arm_traj1/ee_pose_output/ee_poses_3d.npy")
        arm_green_frame = 71
        full_green_frame = 71
    elif args.traj == 2:
        arm_data_path = Path("./data/arm_traj2/arm_traj2.npy")
        full_data_path = Path("./data/arm_traj2/arm_with_wires_traj2.npy")
        output_dir = Path(f"./data/arm_traj2/wire_init_output_{args.pipeline}_{args.method}{repulsion_suffix}")
        precomputed_mask_dir = Path("./data/arm_traj2/masks")
        ee_pose_path = Path("./data/arm_traj2/ee_pose_output/ee_poses_3d.npy")
        arm_green_frame = 0
        full_green_frame = 0
    elif args.traj == 3:
        arm_data_path = Path("./data/arm_traj3/arm_traj3_contact.npy")
        full_data_path = Path("./data/arm_traj3/arm_with_wires_traj3_contact.npy")
        output_dir = Path(f"./data/arm_traj3/wire_init_output_{args.pipeline}_{args.method}{repulsion_suffix}")
        precomputed_mask_dir = None
        ee_pose_path = Path("./data/arm_traj3/ee_pose_output/ee_poses_3d.npy")
        arm_green_frame = 84
        full_green_frame = 100

    
    # Load EE poses if available
    ee_poses_3d = None
    if ee_pose_path.exists():
        ee_data = np.load(str(ee_pose_path), allow_pickle=True).item()
        ee_poses_3d = ee_data['ee_3d']
        print(f"Loaded EE poses from: {ee_pose_path}")
        print(f"  Shape: {ee_poses_3d.shape}")
    else:
        print(f"No EE poses found at: {ee_pose_path}")
    
    # ================================================================
    # Load Data
    # ================================================================
    
    print("=" * 60)
    print("WIRE INITIALIZATION (All Frames)")
    print(f"Trajectory: {args.traj}")
    print(f"Pipeline: {args.pipeline}")
    print(f"Method: {args.method}")
    print(f"Repulsion: {'DISABLED' if args.no_repulsion else 'ENABLED'}")
    print(f"Min MST pixels: {args.min_mst_pixels}")
    print("=" * 60)
    
    print(f"\nLoading arm-only data from: {arm_data_path}")
    arm_only_data = np.load(str(arm_data_path), allow_pickle=True).item()
    
    print(f"Loading full scene data from: {full_data_path}")
    full_scene_data = np.load(str(full_data_path), allow_pickle=True).item()
    
    # Get frame keys (after synchronization)
    arm_frame_keys = sorted(arm_only_data.keys())
    full_frame_keys = sorted(full_scene_data.keys())
    
    # Calculate total frames after green frame sync
    n_arm_frames = len(arm_frame_keys) - arm_green_frame
    n_full_frames = len(full_frame_keys) - full_green_frame
    n_frames_total = min(n_arm_frames, n_full_frames)
    
    # Apply start/end frame limits
    start_frame = args.start
    end_frame = args.end if args.end >= 0 else n_frames_total
    end_frame = min(end_frame, n_frames_total)
    n_frames = end_frame - start_frame
    
    print(f"\nTotal synchronized frames: {n_frames_total}")
    print(f"Processing frames {start_frame} to {end_frame} (step={args.frame_step})")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Process First Frame (Reference)
    # ================================================================
    
    print(f"\n{'#'*60}")
    print(f"# FRAME {start_frame} (Reference)")
    print(f"{'#'*60}")
    
    # Load first frame data
    arm_frame_key = arm_frame_keys[arm_green_frame + start_frame]
    full_frame_key = full_frame_keys[full_green_frame + start_frame]
    
    arm_data = arm_only_data[arm_frame_key]
    arm_depth = arm_data['transformed_depth'].copy()
    
    full_data = full_scene_data[full_frame_key]
    full_rgb = full_data['color'][:, :, ::-1]  # BGR to RGB
    full_depth = full_data['transformed_depth'].copy()
    
    # Load precomputed arm mask if available
    precomputed_arm_mask = None
    if precomputed_mask_dir is not None:
        mask_path = precomputed_mask_dir / f"mask_frame_{full_green_frame + start_frame:04d}.npy"
        if mask_path.exists():
            precomputed_arm_mask = np.load(str(mask_path))
    
    # Get EE poses for first frame
    frame_ee_poses = None
    if ee_poses_3d is not None and start_frame < len(ee_poses_3d):
        frame_ee_poses = ee_poses_3d[start_frame:start_frame+1]  # Shape: (1, 2, 3)

    # Repulsion iterations: 0 if disabled, 500 otherwise (with interleaved projection)
    repulsion_iters = 0 if args.no_repulsion else 500
    
    # Create initializer for first frame
    initializer = WireInitializer(
        intrinsics=intrinsics,
        n_keypoints=args.n_keypoints,
        target_branch_nodes=args.target_branch,
        target_leaf_nodes=args.target_leaf,
        bg_threshold=80.0,
        max_depth=1000.0,
        top_k_components=5,
        arm_dilation_pixels=5,
        repulsion_iterations=repulsion_iters,
        repulsion_lr=2.0,
        ee_poses_3d=frame_ee_poses,
        placement_method=args.method,
        min_skeleton_pixels=args.min_mst_pixels,
    )
    
    # Run initialization for first frame
    if args.pipeline == 'streamlined':
        result = initializer.initialize_streamlined(full_depth, arm_depth, precomputed_arm_mask=precomputed_arm_mask,
                                                    n_keypoints_per_segment=args.keypoints_per_segment)
    else:
        result = initializer.initialize(full_depth, arm_depth, precomputed_arm_mask=precomputed_arm_mask)
    
    if not result['success']:
        print(f"Frame {start_frame} FAILED: {result.get('reason', 'unknown')}")
        print("Cannot continue without reference frame.")
        return
    
    # Reference frame stats
    ref_mst_pixels = int(np.sum(result.get('mst_skeleton_mask', np.zeros(1)) > 0))
    total_edges = len(result['edges'])
    ref_edge_counts = [len(seg) for seg in result['segment_edges']]
    ref_seg_edge_lengths = get_segment_edge_lengths(result['keypoints'], result['segment_edges'])
    ref_seg_lengths = compute_segment_lengths(result['keypoints'], result['segment_edges'])
    
    # Get skeleton-based segment lengths (target for repulsion)
    skeleton_segment_lengths = result.get('skeleton_segment_lengths', None)
    
    print(f"\nFrame {start_frame} MST pixels: {ref_mst_pixels}")
    print(f"Total edges: {total_edges}")
    print(f"Edges per segment (ref): {ref_edge_counts}")
    print(f"Segment lengths (edge-sum): {[f'{l:.1f}' for l in ref_seg_lengths]} mm")
    if skeleton_segment_lengths:
        print(f"Segment lengths (skeleton): {[f'{l:.1f}' for l in skeleton_segment_lengths]} mm")
    print(f"Min MST pixels threshold: {args.min_mst_pixels}")
    
    # Create frames subdirectory for PNG files
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Save reference frame visualization
    viz_2d = create_2d_visualization(
        full_rgb,
        result['foreground_mask'],
        result['skeleton_mask'],
        result['keypoints_2d'],
        result['edges'],
        segment_edges=result['segment_edges'],
        n_branch=result['n_branch'],
        n_leaf=result['n_leaf'],
        frame_idx=start_frame,
        mst_skeleton_mask=result.get('mst_skeleton_mask'),
        skeleton_mask_raw=result.get('skeleton_mask_raw'),
        segment_3d_lengths=ref_seg_lengths,
        skeleton_segment_lengths=skeleton_segment_lengths,
    )
    viz_2d_path = frames_dir / f"init_2d_{start_frame:04d}.png"
    cv2.imwrite(str(viz_2d_path), cv2.cvtColor(viz_2d, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {viz_2d_path.name}")
    
    # Initialize tracking
    all_segment_lengths = {i: [ref_seg_lengths[i]] for i in range(len(ref_seg_lengths))}
    # Store per-segment per-position edge lengths for all frames
    # seg_edge_lengths[seg_idx][edge_pos] = [lengths across frames]
    seg_edge_lengths = {}
    for seg_idx, seg_lens in enumerate(ref_seg_edge_lengths):
        seg_edge_lengths[seg_idx] = {}
        for edge_pos, length in enumerate(seg_lens):
            seg_edge_lengths[seg_idx][edge_pos] = [length]  # Reference frame
    processed_frames = [start_frame]
    skipped_frames = []
    
    # ================================================================
    # Process Remaining Frames
    # ================================================================
    
    frame_indices = list(range(start_frame + args.frame_step, end_frame, args.frame_step))
    print(f"\nProcessing {len(frame_indices)} additional frames...")
    
    for frame_idx in frame_indices:
        # Get frame keys
        arm_frame_key = arm_frame_keys[arm_green_frame + frame_idx]
        full_frame_key = full_frame_keys[full_green_frame + frame_idx]
        
        # Load frame data
        arm_data = arm_only_data[arm_frame_key]
        arm_depth = arm_data['transformed_depth'].copy()
        
        full_data = full_scene_data[full_frame_key]
        full_rgb = full_data['color'][:, :, ::-1]  # BGR to RGB
        full_depth = full_data['transformed_depth'].copy()
        
        # Load precomputed arm mask if available
        precomputed_arm_mask = None
        if precomputed_mask_dir is not None:
            mask_path = precomputed_mask_dir / f"mask_frame_{full_green_frame + frame_idx:04d}.npy"
            if mask_path.exists():
                precomputed_arm_mask = np.load(str(mask_path))
        
        # Get EE poses for this frame
        frame_ee_poses = None
        if ee_poses_3d is not None and frame_idx < len(ee_poses_3d):
            frame_ee_poses = ee_poses_3d[frame_idx:frame_idx+1]
        
        # Create initializer for this frame
        initializer = WireInitializer(
            intrinsics=intrinsics,
            n_keypoints=args.n_keypoints,
            target_branch_nodes=args.target_branch,
            target_leaf_nodes=args.target_leaf,
            bg_threshold=80.0,
            max_depth=1000.0,
            top_k_components=5,
            arm_dilation_pixels=5,
            repulsion_iterations=repulsion_iters,
            repulsion_lr=5.0,
            ee_poses_3d=frame_ee_poses,
            placement_method=args.method,
            min_skeleton_pixels=args.min_mst_pixels,
        )
        
        # Run initialization
        if args.pipeline == 'streamlined':
            result = initializer.initialize_streamlined(full_depth, arm_depth, precomputed_arm_mask=precomputed_arm_mask,
                                                        n_keypoints_per_segment=args.keypoints_per_segment)
        else:
            result = initializer.initialize(full_depth, arm_depth, precomputed_arm_mask=precomputed_arm_mask)
        
        if not result['success']:
            print(f"  Frame {frame_idx}: FAILED ({result.get('reason', 'unknown')})")
            skipped_frames.append((frame_idx, 'init_failed'))
            continue
        
        # Check MST pixel count
        mst_pixels = int(np.sum(result.get('mst_skeleton_mask', np.zeros(1)) > 0))
        
        if mst_pixels < args.min_mst_pixels:
            print(f"  Frame {frame_idx}: SKIPPED (mst_pixels={mst_pixels} < {args.min_mst_pixels})")
            skipped_frames.append((frame_idx, f'mst_pixels_{mst_pixels}'))
            continue
        
        # Check segment lengths: skip if any segment < 70% of reference
        seg_lengths = compute_segment_lengths(result['keypoints'], result['segment_edges'])
        short_seg = False
        for seg_idx in range(min(len(seg_lengths), len(ref_seg_lengths))):
            if ref_seg_lengths[seg_idx] > 0 and seg_lengths[seg_idx] < 0.7 * ref_seg_lengths[seg_idx]:
                print(f"  Frame {frame_idx}: SKIPPED (seg{seg_idx} length={seg_lengths[seg_idx]:.1f}mm "
                      f"< 70% of ref {ref_seg_lengths[seg_idx]:.1f}mm)")
                skipped_frames.append((frame_idx, f'short_seg{seg_idx}_{seg_lengths[seg_idx]:.0f}mm'))
                short_seg = True
                break
        if short_seg:
            continue
        
        # Store segment lengths
        for seg_idx in range(min(len(seg_lengths), 5)):
            if seg_idx in all_segment_lengths:
                all_segment_lengths[seg_idx].append(seg_lengths[seg_idx])
        
        # Store per-edge lengths (topology is fixed, so all frames should match)
        frame_seg_edge_lens = get_segment_edge_lengths(result['keypoints'], result['segment_edges'])
        for seg_idx, seg_lens in enumerate(frame_seg_edge_lens):
            for edge_pos, length in enumerate(seg_lens):
                if seg_idx in seg_edge_lengths and edge_pos in seg_edge_lengths[seg_idx]:
                    seg_edge_lengths[seg_idx][edge_pos].append(length)
        
        processed_frames.append(frame_idx)
        
        # Get skeleton-based segment lengths for this frame
        frame_skeleton_lengths = result.get('skeleton_segment_lengths', None)
        
        # Save 2D visualization
        viz_2d = create_2d_visualization(
            full_rgb,
            result['foreground_mask'],
            result['skeleton_mask'],
            result['keypoints_2d'],
            result['edges'],
            segment_edges=result['segment_edges'],
            n_branch=result['n_branch'],
            n_leaf=result['n_leaf'],
            frame_idx=frame_idx,
            mst_skeleton_mask=result.get('mst_skeleton_mask'),
            skeleton_mask_raw=result.get('skeleton_mask_raw'),
            segment_3d_lengths=seg_lengths,
            skeleton_segment_lengths=frame_skeleton_lengths,
        )
        viz_2d_path = frames_dir / f"init_2d_{frame_idx:04d}.png"
        cv2.imwrite(str(viz_2d_path), cv2.cvtColor(viz_2d, cv2.COLOR_RGB2BGR))
        
        if frame_idx % 20 == 0:
            print(f"  Frame {frame_idx}: OK (mst_pixels={mst_pixels})")
    
    # ================================================================
    # Generate Plots
    # ================================================================
    
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print("=" * 60)
    
    # 1. Segment length boxplot (distribution only, no reference)
    boxplot_path = output_dir / "segment_length_boxplot.png"
    create_segment_length_boxplot(all_segment_lengths, boxplot_path, args.traj)
    
    # 2. Edge length boxplot (per-edge by segment position)
    edge_length_path = output_dir / "edge_length_boxplot.png"
    create_edge_length_boxplot(
        seg_edge_lengths, ref_seg_edge_lengths,
        edge_length_path, args.traj,
    )
    
    # ================================================================
    # Generate Video from 2D Frames
    # ================================================================
    
    print(f"\n{'='*60}")
    print("GENERATING 2D VISUALIZATION VIDEO")
    print("=" * 60)
    
    video_path = output_dir / "init_2d_video.mp4"
    create_video_from_frames(frames_dir, video_path, fps=10)
    
    # ================================================================
    # Evaluation: Edge Length Error Metrics
    # ================================================================
    
    print(f"\n{'='*60}")
    print("EVALUATION")
    print("=" * 60)
    
    n_processed = len(processed_frames)
    
    print(f"Total frames: {n_frames}")
    print(f"Processed frames: {n_processed}")
    print(f"Skipped frames: {len(skipped_frames)}")
    print(f"Edges per segment (fixed): {ref_edge_counts}")
    
    if skipped_frames:
        print(f"\nSkipped frames details:")
        for frame_idx, reason in skipped_frames[:20]:
            print(f"  Frame {frame_idx}: {reason}")
        if len(skipped_frames) > 20:
            print(f"  ... and {len(skipped_frames) - 20} more")
    
    # Compute and save edge length error metrics (overall, per-segment, with/without outliers)
    edge_metrics_path = output_dir / "edge_length_error_metrics.json"
    compute_edge_length_error_metrics(
        seg_edge_lengths, ref_seg_edge_lengths,
        edge_metrics_path, args.traj,
    )
    
    print(f"\nOutput directory: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()