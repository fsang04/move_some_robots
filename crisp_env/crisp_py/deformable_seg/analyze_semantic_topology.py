"""
Analyze edge topology with fixed semantic indexing.

Topology structure:
- Left branch node (smaller x) and Right branch node (larger x)
- 4 leaf nodes: top-left, top-right, bottom-left, bottom-right

Edge indexing:
- Edges 1-4: Left branch → Top-left leaf (4 edges)
- Edges 5-9: Left branch → Right branch (5 edges)
- Edges 10-13: Right branch → Top-right leaf (4 edges)
- Edges 14-16: Left branch → Bottom-left leaf (3 edges)
- Edges 17-20: Right branch → Bottom-right leaf (4 edges)

Total: 20 edges, 21 keypoints
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def identify_semantic_nodes(keypoints, edges):
    """
    Identify branch nodes and leaf nodes semantically.
    
    Returns:
        dict with keys: 'left_branch', 'right_branch', 
                        'top_left', 'top_right', 'bottom_left', 'bottom_right'
    """
    n_keypoints = keypoints.shape[0]
    
    # Compute degree of each node
    degrees = np.zeros(n_keypoints, dtype=int)
    for (i, j) in edges:
        degrees[i] += 1
        degrees[j] += 1
    
    # Branch nodes have degree >= 3, leaf nodes have degree == 1
    branch_indices = np.where(degrees >= 3)[0]
    leaf_indices = np.where(degrees == 1)[0]
    
    if len(branch_indices) < 2:
        return None
    if len(leaf_indices) < 4:
        return None
    
    # Identify left and right branch by x-coordinate
    branch_coords = keypoints[branch_indices]
    sorted_by_x = np.argsort(branch_coords[:, 0])
    left_branch_idx = branch_indices[sorted_by_x[0]]
    right_branch_idx = branch_indices[sorted_by_x[-1]]
    
    # Identify leaf nodes by position
    leaf_coords = keypoints[leaf_indices]
    
    # Split by x-coordinate first (left vs right)
    leaf_x = leaf_coords[:, 0]
    median_x = np.median(leaf_x)
    
    left_leaves = leaf_indices[leaf_x < median_x]
    right_leaves = leaf_indices[leaf_x >= median_x]
    
    # For left leaves, identify top and bottom by y-coordinate (smaller y = top in image coords)
    if len(left_leaves) >= 2:
        left_leaf_y = keypoints[left_leaves, 1]
        left_sorted = left_leaves[np.argsort(left_leaf_y)]
        top_left = left_sorted[0]
        bottom_left = left_sorted[-1]
    else:
        return None
    
    # For right leaves, identify top and bottom by y-coordinate
    if len(right_leaves) >= 2:
        right_leaf_y = keypoints[right_leaves, 1]
        right_sorted = right_leaves[np.argsort(right_leaf_y)]
        top_right = right_sorted[0]
        bottom_right = right_sorted[-1]
    else:
        return None
    
    return {
        'left_branch': left_branch_idx,
        'right_branch': right_branch_idx,
        'top_left': top_left,
        'top_right': top_right,
        'bottom_left': bottom_left,
        'bottom_right': bottom_right,
    }


def find_path_between_nodes(start, end, edges, n_keypoints):
    """
    Find the path between two nodes using BFS.
    Returns list of node indices from start to end (inclusive).
    """
    from collections import deque
    
    # Build adjacency list
    adj = {i: [] for i in range(n_keypoints)}
    for (i, j) in edges:
        adj[i].append(j)
        adj[j].append(i)
    
    # BFS
    visited = {start}
    parent = {start: None}
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        if node == end:
            break
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                queue.append(neighbor)
    
    if end not in parent:
        return None
    
    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    
    return path


def path_to_edges(path):
    """Convert a path (list of nodes) to edges (list of (i, j) tuples)."""
    edges = []
    for i in range(len(path) - 1):
        edges.append((path[i], path[i + 1]))
    return edges


def main():
    # Load saved keypoints data
    data_path = Path("/home/yehengz/deformable_seg/tracking_output/all_keypoints_3d.npy")
    print(f"Loading data from {data_path}...")
    data = np.load(str(data_path), allow_pickle=True).item()
    
    all_keypoints_3d = data["keypoints"]
    all_edges = data["edges"]
    
    n_frames = len(all_keypoints_3d)
    print(f"Loaded {n_frames} frames")
    
    # ============================================================
    # Analyze semantic structure for specific frames
    # ============================================================
    print("\n" + "=" * 60)
    print("SEMANTIC NODE IDENTIFICATION")
    print("=" * 60)
    
    test_frames = [0, 15, 30, 45, 60, 128]
    
    for frame_idx in test_frames:
        if frame_idx >= n_frames:
            continue
        
        keypoints = all_keypoints_3d[frame_idx]
        edges = all_edges[frame_idx]
        
        if keypoints.shape[0] == 0 or len(edges) == 0:
            print(f"\nFrame {frame_idx}: No valid data")
            continue
        
        semantic_nodes = identify_semantic_nodes(keypoints, edges)
        
        if semantic_nodes is None:
            print(f"\nFrame {frame_idx}: Could not identify semantic structure")
            continue
        
        print(f"\nFrame {frame_idx}:")
        print(f"  Left branch node:  index {semantic_nodes['left_branch']}, pos: {keypoints[semantic_nodes['left_branch']]}")
        print(f"  Right branch node: index {semantic_nodes['right_branch']}, pos: {keypoints[semantic_nodes['right_branch']]}")
        print(f"  Top-left leaf:     index {semantic_nodes['top_left']}, pos: {keypoints[semantic_nodes['top_left']]}")
        print(f"  Top-right leaf:    index {semantic_nodes['top_right']}, pos: {keypoints[semantic_nodes['top_right']]}")
        print(f"  Bottom-left leaf:  index {semantic_nodes['bottom_left']}, pos: {keypoints[semantic_nodes['bottom_left']]}")
        print(f"  Bottom-right leaf: index {semantic_nodes['bottom_right']}, pos: {keypoints[semantic_nodes['bottom_right']]}")
    
    # ============================================================
    # Compute edge lengths using semantic topology
    # ============================================================
    print("\n" + "=" * 60)
    print("EDGE LENGTH ANALYSIS WITH SEMANTIC TOPOLOGY")
    print("=" * 60)
    
    # Expected number of edges per segment
    expected_edges = {
        'left_branch_to_top_left': 4,      # Edges 1-4
        'left_branch_to_right_branch': 5,  # Edges 5-9
        'right_branch_to_top_right': 4,    # Edges 10-13
        'left_branch_to_bottom_left': 3,   # Edges 14-16
        'right_branch_to_bottom_right': 4, # Edges 17-20
    }
    
    # Collect edge lengths per semantic edge (1-20)
    semantic_edge_lengths = {i: [] for i in range(1, 21)}
    valid_frames = []
    
    for frame_idx in range(n_frames):
        keypoints = all_keypoints_3d[frame_idx]
        edges = all_edges[frame_idx]
        
        if keypoints.shape[0] == 0 or len(edges) == 0:
            continue
        
        semantic_nodes = identify_semantic_nodes(keypoints, edges)
        if semantic_nodes is None:
            continue
        
        lb = semantic_nodes['left_branch']
        rb = semantic_nodes['right_branch']
        tl = semantic_nodes['top_left']
        tr = semantic_nodes['top_right']
        bl = semantic_nodes['bottom_left']
        br = semantic_nodes['bottom_right']
        
        n_kp = keypoints.shape[0]
        
        # Find paths
        path_lb_tl = find_path_between_nodes(lb, tl, edges, n_kp)
        path_lb_rb = find_path_between_nodes(lb, rb, edges, n_kp)
        path_rb_tr = find_path_between_nodes(rb, tr, edges, n_kp)
        path_lb_bl = find_path_between_nodes(lb, bl, edges, n_kp)
        path_rb_br = find_path_between_nodes(rb, br, edges, n_kp)
        
        # Check if paths have expected number of edges
        if path_lb_tl is None or len(path_lb_tl) - 1 != expected_edges['left_branch_to_top_left']:
            continue
        if path_lb_rb is None or len(path_lb_rb) - 1 != expected_edges['left_branch_to_right_branch']:
            continue
        if path_rb_tr is None or len(path_rb_tr) - 1 != expected_edges['right_branch_to_top_right']:
            continue
        if path_lb_bl is None or len(path_lb_bl) - 1 != expected_edges['left_branch_to_bottom_left']:
            continue
        if path_rb_br is None or len(path_rb_br) - 1 != expected_edges['right_branch_to_bottom_right']:
            continue
        
        valid_frames.append(frame_idx)
        
        # Compute edge lengths for each semantic edge
        edge_idx = 1
        
        # Edges 1-4: Left branch → Top-left
        for i in range(len(path_lb_tl) - 1):
            length = np.linalg.norm(keypoints[path_lb_tl[i]] - keypoints[path_lb_tl[i + 1]])
            semantic_edge_lengths[edge_idx].append(length)
            edge_idx += 1
        
        # Edges 5-9: Left branch → Right branch
        for i in range(len(path_lb_rb) - 1):
            length = np.linalg.norm(keypoints[path_lb_rb[i]] - keypoints[path_lb_rb[i + 1]])
            semantic_edge_lengths[edge_idx].append(length)
            edge_idx += 1
        
        # Edges 10-13: Right branch → Top-right
        for i in range(len(path_rb_tr) - 1):
            length = np.linalg.norm(keypoints[path_rb_tr[i]] - keypoints[path_rb_tr[i + 1]])
            semantic_edge_lengths[edge_idx].append(length)
            edge_idx += 1
        
        # Edges 14-16: Left branch → Bottom-left
        for i in range(len(path_lb_bl) - 1):
            length = np.linalg.norm(keypoints[path_lb_bl[i]] - keypoints[path_lb_bl[i + 1]])
            semantic_edge_lengths[edge_idx].append(length)
            edge_idx += 1
        
        # Edges 17-20: Right branch → Bottom-right
        for i in range(len(path_rb_br) - 1):
            length = np.linalg.norm(keypoints[path_rb_br[i]] - keypoints[path_rb_br[i + 1]])
            semantic_edge_lengths[edge_idx].append(length)
            edge_idx += 1
    
    print(f"\nValid frames with correct topology: {len(valid_frames)}/{n_frames}")
    print(f"Valid frame indices: {valid_frames[:20]}..." if len(valid_frames) > 20 else f"Valid frame indices: {valid_frames}")
    
    if len(valid_frames) == 0:
        print("No valid frames found!")
        return
    
    # ============================================================
    # Create boxplot
    # ============================================================
    print("\n" + "=" * 60)
    print("EDGE LENGTH BOXPLOT")
    print("=" * 60)
    
    # Prepare data for boxplot
    edge_data = []
    edge_labels = []
    segment_colors = []
    
    # Define colors for each segment
    colors = {
        'LB→TL': 'lightblue',     # Edges 1-4
        'LB→RB': 'lightgreen',    # Edges 5-9
        'RB→TR': 'lightyellow',   # Edges 10-13
        'LB→BL': 'lightcoral',    # Edges 14-16
        'RB→BR': 'plum',          # Edges 17-20
    }
    
    for edge_idx in range(1, 21):
        lengths = semantic_edge_lengths[edge_idx]
        if len(lengths) > 0:
            edge_data.append(lengths)
            edge_labels.append(str(edge_idx))
            
            # Assign color based on segment
            if 1 <= edge_idx <= 4:
                segment_colors.append(colors['LB→TL'])
            elif 5 <= edge_idx <= 9:
                segment_colors.append(colors['LB→RB'])
            elif 10 <= edge_idx <= 13:
                segment_colors.append(colors['RB→TR'])
            elif 14 <= edge_idx <= 16:
                segment_colors.append(colors['LB→BL'])
            else:
                segment_colors.append(colors['RB→BR'])
    
    # Create boxplot
    fig, ax = plt.subplots(figsize=(16, 8))
    bp = ax.boxplot(edge_data, labels=edge_labels, patch_artist=True)
    
    # Color the boxes by segment
    for patch, color in zip(bp['boxes'], segment_colors):
        patch.set_facecolor(color)
    
    # Add segment labels
    ax.axvline(4.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(9.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(13.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(16.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add segment annotations
    ax.text(2.5, ax.get_ylim()[1] * 0.95, 'LB→TL\n(4 edges)', ha='center', fontsize=9, color='blue')
    ax.text(7, ax.get_ylim()[1] * 0.95, 'LB→RB\n(5 edges)', ha='center', fontsize=9, color='green')
    ax.text(11.5, ax.get_ylim()[1] * 0.95, 'RB→TR\n(4 edges)', ha='center', fontsize=9, color='orange')
    ax.text(15, ax.get_ylim()[1] * 0.95, 'LB→BL\n(3 edges)', ha='center', fontsize=9, color='red')
    ax.text(18.5, ax.get_ylim()[1] * 0.95, 'RB→BR\n(4 edges)', ha='center', fontsize=9, color='purple')
    
    ax.set_xlabel('Semantic Edge Index', fontsize=12)
    ax.set_ylabel('Edge Length (mm)', fontsize=12)
    ax.set_title(f'Edge Length Distribution by Semantic Index ({len(valid_frames)} Frames)', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_dir = Path("/home/yehengz/deformable_seg/tracking_output")
    boxplot_path = output_dir / "semantic_edge_lengths_boxplot.png"
    plt.savefig(str(boxplot_path), dpi=150)
    plt.close()
    print(f"Saved semantic edge boxplot to {boxplot_path}")
    
    # ============================================================
    # Create histogram for each edge in one figure
    # ============================================================
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    # Define colors for each segment
    hist_colors = {
        'LB→TL': 'steelblue',     # Edges 1-4
        'LB→RB': 'forestgreen',   # Edges 5-9
        'RB→TR': 'goldenrod',     # Edges 10-13
        'LB→BL': 'indianred',     # Edges 14-16
        'RB→BR': 'mediumpurple',  # Edges 17-20
    }
    
    segment_names = {
        1: 'LB→TL', 2: 'LB→TL', 3: 'LB→TL', 4: 'LB→TL',
        5: 'LB→RB', 6: 'LB→RB', 7: 'LB→RB', 8: 'LB→RB', 9: 'LB→RB',
        10: 'RB→TR', 11: 'RB→TR', 12: 'RB→TR', 13: 'RB→TR',
        14: 'LB→BL', 15: 'LB→BL', 16: 'LB→BL',
        17: 'RB→BR', 18: 'RB→BR', 19: 'RB→BR', 20: 'RB→BR',
    }
    
    for i, edge_idx in enumerate(range(1, 21)):
        ax = axes[i]
        lengths = semantic_edge_lengths[edge_idx]
        segment = segment_names[edge_idx]
        color = hist_colors[segment]
        
        if len(lengths) > 0:
            ax.hist(lengths, bins=20, edgecolor='black', alpha=0.7, color=color)
            ax.axvline(np.mean(lengths), color='red', linestyle='--', linewidth=1.5)
            ax.set_title(f'Edge {edge_idx} ({segment})\nμ={np.mean(lengths):.1f}, σ={np.std(lengths):.1f}', fontsize=10)
        else:
            ax.set_title(f'Edge {edge_idx} (no data)', fontsize=10)
        
        ax.set_xlabel('Length (mm)', fontsize=8)
        ax.set_ylabel('Freq', fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
    
    plt.suptitle(f'Edge Length Histograms ({len(valid_frames)} frames)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    hist_path = output_dir / "semantic_edge_lengths_histogram.png"
    plt.savefig(str(hist_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved edge length histogram to {hist_path}")
    
    # ============================================================
    # Create 2D projection with edge indices
    # ============================================================
    # Use first valid frame for visualization
    if len(valid_frames) > 0:
        frame_idx = valid_frames[0]
        keypoints = all_keypoints_3d[frame_idx]
        edges = all_edges[frame_idx]
        
        # Project to 2D using camera intrinsics
        fx, fy = 606.1124, 605.8821
        cx, cy = 641.7578, 365.6519
        
        pts_2d = np.zeros((len(keypoints), 2))
        for i, pt in enumerate(keypoints):
            if pt[2] > 0:
                pts_2d[i, 0] = fx * pt[0] / pt[2] + cx
                pts_2d[i, 1] = fy * pt[1] / pt[2] + cy
        
        # Get semantic nodes for this frame
        semantic_nodes = identify_semantic_nodes(keypoints, edges)
        if semantic_nodes is None:
            print("Could not identify semantic nodes for 2D visualization")
        else:
            lb_idx = semantic_nodes['left_branch']
            rb_idx = semantic_nodes['right_branch']
            tl_idx = semantic_nodes['top_left']
            tr_idx = semantic_nodes['top_right']
            bl_idx = semantic_nodes['bottom_left']
            br_idx = semantic_nodes['bottom_right']
            
            n_keypoints = len(keypoints)
            
            # Find all paths
            path_lb_tl = find_path_between_nodes(lb_idx, tl_idx, edges, n_keypoints)
            path_lb_rb = find_path_between_nodes(lb_idx, rb_idx, edges, n_keypoints)
            path_rb_tr = find_path_between_nodes(rb_idx, tr_idx, edges, n_keypoints)
            path_lb_bl = find_path_between_nodes(lb_idx, bl_idx, edges, n_keypoints)
            path_rb_br = find_path_between_nodes(rb_idx, br_idx, edges, n_keypoints)
        
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Define colors for each segment
            edge_colors = {
                'LB→TL': 'blue',
                'LB→RB': 'green',
                'RB→TR': 'orange',
                'LB→BL': 'red',
                'RB→BR': 'purple',
            }
            
            # Draw edges with semantic indices
            edge_idx = 1
            
            def draw_edge_with_label(path, segment_name, start_idx):
                nonlocal edge_idx
                color = edge_colors[segment_name]
                for i in range(len(path) - 1):
                    p1 = pts_2d[path[i]]
                    p2 = pts_2d[path[i + 1]]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=3, alpha=0.8)
                    # Label at midpoint
                    mid_x = (p1[0] + p2[0]) / 2
                    mid_y = (p1[1] + p2[1]) / 2
                    ax.text(mid_x, mid_y, str(edge_idx), fontsize=10, fontweight='bold',
                           color='white', ha='center', va='center',
                           bbox=dict(boxstyle='circle,pad=0.3', facecolor=color, edgecolor='black', alpha=0.9))
                    edge_idx += 1
            
            # Draw all edges with labels
            draw_edge_with_label(path_lb_tl, 'LB→TL', 1)
            draw_edge_with_label(path_lb_rb, 'LB→RB', 5)
            draw_edge_with_label(path_rb_tr, 'RB→TR', 10)
            draw_edge_with_label(path_lb_bl, 'LB→BL', 14)
            draw_edge_with_label(path_rb_br, 'RB→BR', 17)
            
            # Draw nodes
            for i, pt in enumerate(pts_2d):
                if i == lb_idx:
                    ax.scatter(pt[0], pt[1], c='gold', s=300, edgecolors='black', linewidths=2, zorder=5, marker='s')
                    ax.text(pt[0], pt[1] + 20, 'LB', fontsize=12, fontweight='bold', ha='center', color='darkgoldenrod')
                elif i == rb_idx:
                    ax.scatter(pt[0], pt[1], c='gold', s=300, edgecolors='black', linewidths=2, zorder=5, marker='s')
                    ax.text(pt[0], pt[1] + 20, 'RB', fontsize=12, fontweight='bold', ha='center', color='darkgoldenrod')
                elif i == tl_idx:
                    ax.scatter(pt[0], pt[1], c='cyan', s=200, edgecolors='black', linewidths=2, zorder=5)
                    ax.text(pt[0], pt[1] - 20, 'TL', fontsize=11, fontweight='bold', ha='center', color='darkcyan')
                elif i == tr_idx:
                    ax.scatter(pt[0], pt[1], c='cyan', s=200, edgecolors='black', linewidths=2, zorder=5)
                    ax.text(pt[0], pt[1] - 20, 'TR', fontsize=11, fontweight='bold', ha='center', color='darkcyan')
                elif i == bl_idx:
                    ax.scatter(pt[0], pt[1], c='magenta', s=200, edgecolors='black', linewidths=2, zorder=5)
                    ax.text(pt[0], pt[1] + 20, 'BL', fontsize=11, fontweight='bold', ha='center', color='darkmagenta')
                elif i == br_idx:
                    ax.scatter(pt[0], pt[1], c='magenta', s=200, edgecolors='black', linewidths=2, zorder=5)
                    ax.text(pt[0], pt[1] + 20, 'BR', fontsize=11, fontweight='bold', ha='center', color='darkmagenta')
                else:
                    ax.scatter(pt[0], pt[1], c='white', s=80, edgecolors='gray', linewidths=1, zorder=4)
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], color='blue', linewidth=3, label='LB→TL (edges 1-4)'),
                plt.Line2D([0], [0], color='green', linewidth=3, label='LB→RB (edges 5-9)'),
                plt.Line2D([0], [0], color='orange', linewidth=3, label='RB→TR (edges 10-13)'),
                plt.Line2D([0], [0], color='red', linewidth=3, label='LB→BL (edges 14-16)'),
                plt.Line2D([0], [0], color='purple', linewidth=3, label='RB→BR (edges 17-20)'),
                plt.scatter([], [], c='gold', s=100, marker='s', edgecolors='black', label='Branch nodes'),
                plt.scatter([], [], c='cyan', s=100, edgecolors='black', label='Top leaf nodes'),
                plt.scatter([], [], c='magenta', s=100, edgecolors='black', label='Bottom leaf nodes'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            ax.set_xlabel('X (pixels)', fontsize=12)
            ax.set_ylabel('Y (pixels)', fontsize=12)
            ax.set_title(f'Semantic Topology with Edge Indices (Frame {frame_idx})', fontsize=14)
            ax.set_aspect('equal')
            ax.invert_yaxis()  # Image coordinates
            plt.tight_layout()
            
            topo_path = output_dir / "semantic_topology_2d.png"
            plt.savefig(str(topo_path), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved 2D topology visualization to {topo_path}")
    
    # Print statistics
    print("\nEdge Length Statistics (mm):")
    print(f"{'Edge':<6} {'Segment':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'CV(%)':<10}")
    print("-" * 70)
    
    segment_names = {
        1: 'LB→TL', 2: 'LB→TL', 3: 'LB→TL', 4: 'LB→TL',
        5: 'LB→RB', 6: 'LB→RB', 7: 'LB→RB', 8: 'LB→RB', 9: 'LB→RB',
        10: 'RB→TR', 11: 'RB→TR', 12: 'RB→TR', 13: 'RB→TR',
        14: 'LB→BL', 15: 'LB→BL', 16: 'LB→BL',
        17: 'RB→BR', 18: 'RB→BR', 19: 'RB→BR', 20: 'RB→BR',
    }
    
    all_lengths = []
    for edge_idx in range(1, 21):
        lengths = semantic_edge_lengths[edge_idx]
        if len(lengths) > 0:
            all_lengths.extend(lengths)
            mean_len = np.mean(lengths)
            std_len = np.std(lengths)
            cv = (std_len / mean_len * 100) if mean_len > 0 else 0
            segment = segment_names[edge_idx]
            print(f"{edge_idx:<6} {segment:<10} {mean_len:<10.2f} {std_len:<10.2f} {np.min(lengths):<10.2f} {np.max(lengths):<10.2f} {cv:<10.1f}")
    
    print("-" * 70)
    mean_all = np.mean(all_lengths)
    std_all = np.std(all_lengths)
    cv_all = (std_all / mean_all * 100) if mean_all > 0 else 0
    print(f"{'All':<6} {'':<10} {mean_all:<10.2f} {std_all:<10.2f} {np.min(all_lengths):<10.2f} {np.max(all_lengths):<10.2f} {cv_all:<10.1f}")
    
    # ============================================================
    # Edge Length Difference Analysis (Inextensibility Assumption)
    # Use frame 0 as reference
    # ============================================================
    print("\n" + "=" * 60)
    print("EDGE LENGTH DIFFERENCE ANALYSIS (Frame 0 as Reference)")
    print("=" * 60)
    
    # First, compute reference edge lengths from frame 0
    if 0 not in valid_frames:
        print("Frame 0 is not a valid frame. Using first valid frame as reference.")
        ref_frame = valid_frames[0]
    else:
        ref_frame = 0
    
    ref_keypoints = all_keypoints_3d[ref_frame]
    ref_edges = all_edges[ref_frame]
    n_keypoints = len(ref_keypoints)
    
    ref_semantic_nodes = identify_semantic_nodes(ref_keypoints, ref_edges)
    if ref_semantic_nodes is None:
        print("Cannot identify semantic nodes in reference frame!")
    else:
        ref_lb = ref_semantic_nodes['left_branch']
        ref_rb = ref_semantic_nodes['right_branch']
        ref_tl = ref_semantic_nodes['top_left']
        ref_tr = ref_semantic_nodes['top_right']
        ref_bl = ref_semantic_nodes['bottom_left']
        ref_br = ref_semantic_nodes['bottom_right']
        
        # Find paths in reference frame
        ref_path_lb_tl = find_path_between_nodes(ref_lb, ref_tl, ref_edges, n_keypoints)
        ref_path_lb_rb = find_path_between_nodes(ref_lb, ref_rb, ref_edges, n_keypoints)
        ref_path_rb_tr = find_path_between_nodes(ref_rb, ref_tr, ref_edges, n_keypoints)
        ref_path_lb_bl = find_path_between_nodes(ref_lb, ref_bl, ref_edges, n_keypoints)
        ref_path_rb_br = find_path_between_nodes(ref_rb, ref_br, ref_edges, n_keypoints)
        
        # Compute reference edge lengths
        ref_edge_lengths = {}  # edge_idx -> length
        edge_idx = 1
        
        for path in [ref_path_lb_tl, ref_path_lb_rb, ref_path_rb_tr, ref_path_lb_bl, ref_path_rb_br]:
            if path is not None:
                for i in range(len(path) - 1):
                    length = np.linalg.norm(ref_keypoints[path[i]] - ref_keypoints[path[i + 1]])
                    ref_edge_lengths[edge_idx] = length
                    edge_idx += 1
        
        print(f"Reference frame: {ref_frame}")
        print(f"Reference edge lengths computed for {len(ref_edge_lengths)} edges")
        
        # Compute differences for all valid frames
        edge_differences = {i: [] for i in range(1, 21)}  # edge_idx -> list of differences
        
        for frame_idx in valid_frames:
            if frame_idx == ref_frame:
                continue
                
            keypoints = all_keypoints_3d[frame_idx]
            edges = all_edges[frame_idx]
            n_kp = len(keypoints)
            
            semantic_nodes = identify_semantic_nodes(keypoints, edges)
            if semantic_nodes is None:
                continue
            
            lb = semantic_nodes['left_branch']
            rb = semantic_nodes['right_branch']
            tl = semantic_nodes['top_left']
            tr = semantic_nodes['top_right']
            bl = semantic_nodes['bottom_left']
            br = semantic_nodes['bottom_right']
            
            path_lb_tl = find_path_between_nodes(lb, tl, edges, n_kp)
            path_lb_rb = find_path_between_nodes(lb, rb, edges, n_kp)
            path_rb_tr = find_path_between_nodes(rb, tr, edges, n_kp)
            path_lb_bl = find_path_between_nodes(lb, bl, edges, n_kp)
            path_rb_br = find_path_between_nodes(rb, br, edges, n_kp)
            
            paths = [path_lb_tl, path_lb_rb, path_rb_tr, path_lb_bl, path_rb_br]
            expected_lens = [4, 5, 4, 3, 4]  # Expected number of edges
            
            all_valid = True
            for path, exp_len in zip(paths, expected_lens):
                if path is None or len(path) - 1 != exp_len:
                    all_valid = False
                    break
            
            if not all_valid:
                continue
            
            # Compute edge length differences
            edge_idx = 1
            for path in paths:
                for i in range(len(path) - 1):
                    length = np.linalg.norm(keypoints[path[i]] - keypoints[path[i + 1]])
                    if edge_idx in ref_edge_lengths:
                        diff = length - ref_edge_lengths[edge_idx]
                        edge_differences[edge_idx].append(diff)
                    edge_idx += 1
        
        # Print statistics
        print("\nEdge Length Difference Statistics (mm):")
        print(f"{'Edge':<6} {'Segment':<10} {'Mean Diff':<12} {'Std Diff':<12} {'Min Diff':<12} {'Max Diff':<12} {'Ref Len':<10} {'%Error':<10}")
        print("-" * 90)
        
        all_diffs = []
        percent_errors = {}
        for edge_idx in range(1, 21):
            diffs = edge_differences[edge_idx]
            if len(diffs) > 0:
                all_diffs.extend(diffs)
                mean_diff = np.mean(diffs)
                std_diff = np.std(diffs)
                ref_len = ref_edge_lengths.get(edge_idx, 0)
                segment = segment_names[edge_idx]
                pct_error = (abs(mean_diff) / ref_len * 100) if ref_len > 0 else 0
                percent_errors[edge_idx] = pct_error
                print(f"{edge_idx:<6} {segment:<10} {mean_diff:<12.3f} {std_diff:<12.3f} {np.min(diffs):<12.3f} {np.max(diffs):<12.3f} {ref_len:<10.2f} {pct_error:<10.2f}")
        
        print("-" * 90)
        if len(all_diffs) > 0:
            total_ref_len = sum(ref_edge_lengths.values())
            total_pct_error = (abs(np.mean(all_diffs)) / (total_ref_len / 20) * 100) if total_ref_len > 0 else 0
            print(f"{'All':<6} {'':<10} {np.mean(all_diffs):<12.3f} {np.std(all_diffs):<12.3f} {np.min(all_diffs):<12.3f} {np.max(all_diffs):<12.3f} {'':<10} {total_pct_error:<10.2f}")
        
        # Print sorted by percent error
        print("\nEdges sorted by percent error (highest first):")
        sorted_edges = sorted(percent_errors.items(), key=lambda x: x[1], reverse=True)
        for edge_idx, pct_err in sorted_edges:
            segment = segment_names[edge_idx]
            ref_len = ref_edge_lengths.get(edge_idx, 0)
            print(f"  Edge {edge_idx:2d} ({segment:<8}): {pct_err:6.2f}% error (ref: {ref_len:.2f} mm)")
        
        # ============================================================
        # Visualization 1: Boxplot of edge length differences
        # ============================================================
        fig, ax = plt.subplots(figsize=(16, 8))
        
        diff_data = []
        diff_labels = []
        diff_colors = []
        
        hist_colors = {
            'LB→TL': 'steelblue',
            'LB→RB': 'forestgreen',
            'RB→TR': 'goldenrod',
            'LB→BL': 'indianred',
            'RB→BR': 'mediumpurple',
        }
        
        for edge_idx in range(1, 21):
            diffs = edge_differences[edge_idx]
            if len(diffs) > 0:
                diff_data.append(diffs)
                diff_labels.append(str(edge_idx))
                diff_colors.append(hist_colors[segment_names[edge_idx]])
        
        bp = ax.boxplot(diff_data, labels=diff_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], diff_colors):
            patch.set_facecolor(color)
        
        ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero difference (inextensible)')
        ax.axvline(4.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(9.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(13.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(16.5, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Semantic Edge Index', fontsize=12)
        ax.set_ylabel('Edge Length Difference from Frame 0 (mm)', fontsize=12)
        ax.set_title(f'Edge Length Differences (Inextensibility Analysis, {len(valid_frames)-1} Frames vs Frame {ref_frame})', fontsize=14)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        diff_boxplot_path = output_dir / "edge_length_differences_boxplot.png"
        plt.savefig(str(diff_boxplot_path), dpi=150)
        plt.close()
        print(f"\nSaved edge length difference boxplot to {diff_boxplot_path}")
        
        # ============================================================
        # Visualization 2: Histogram of all differences
        # ============================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(all_diffs, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero (perfect inextensibility)')
        ax.axvline(np.mean(all_diffs), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_diffs):.3f} mm')
        ax.set_xlabel('Edge Length Difference (mm)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Distribution of Edge Length Differences from Frame {ref_frame}', fontsize=14)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        diff_hist_path = output_dir / "edge_length_differences_histogram.png"
        plt.savefig(str(diff_hist_path), dpi=150)
        plt.close()
        print(f"Saved edge length difference histogram to {diff_hist_path}")
        
        # ============================================================
        # Visualization 3: Per-edge difference histograms (4x5 grid)
        # ============================================================
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, edge_idx in enumerate(range(1, 21)):
            ax = axes[i]
            diffs = edge_differences[edge_idx]
            segment = segment_names[edge_idx]
            color = hist_colors[segment]
            
            if len(diffs) > 0:
                ax.hist(diffs, bins=20, edgecolor='black', alpha=0.7, color=color)
                ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
                ax.axvline(np.mean(diffs), color='orange', linestyle='--', linewidth=1.5)
                ax.set_title(f'Edge {edge_idx} ({segment})\nμ={np.mean(diffs):.2f}, σ={np.std(diffs):.2f}', fontsize=10)
            else:
                ax.set_title(f'Edge {edge_idx} (no data)', fontsize=10)
            
            ax.set_xlabel('Diff (mm)', fontsize=8)
            ax.set_ylabel('Freq', fontsize=8)
            ax.tick_params(axis='both', labelsize=7)
        
        plt.suptitle(f'Per-Edge Length Difference Histograms (vs Frame {ref_frame})', fontsize=14, y=1.02)
        plt.tight_layout()
        
        diff_per_edge_path = output_dir / "edge_length_differences_per_edge.png"
        plt.savefig(str(diff_per_edge_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved per-edge difference histograms to {diff_per_edge_path}")
        
        # ============================================================
        # Visualization 4: Time series of total length difference
        # ============================================================
        frame_total_diffs = []
        frame_indices = []
        
        for frame_idx in valid_frames:
            if frame_idx == ref_frame:
                continue
            
            # Check if all edges have data for this frame
            total_diff = 0
            valid = True
            for edge_idx in range(1, 21):
                diffs = edge_differences[edge_idx]
                frame_offset = valid_frames.index(frame_idx) - (1 if ref_frame in valid_frames and valid_frames.index(ref_frame) < valid_frames.index(frame_idx) else 0)
                if frame_offset < len(diffs):
                    total_diff += abs(diffs[frame_offset - (1 if ref_frame == 0 else 0)] if frame_offset > 0 else 0)
        
        # Simpler approach: compute mean absolute difference per frame
        mean_abs_diffs_per_frame = []
        for frame_offset in range(len(edge_differences[1])):
            frame_diffs = []
            for edge_idx in range(1, 21):
                if frame_offset < len(edge_differences[edge_idx]):
                    frame_diffs.append(abs(edge_differences[edge_idx][frame_offset]))
            if len(frame_diffs) > 0:
                mean_abs_diffs_per_frame.append(np.mean(frame_diffs))
        
        if len(mean_abs_diffs_per_frame) > 0:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(mean_abs_diffs_per_frame, 'b-', linewidth=1, alpha=0.7)
            ax.axhline(np.mean(mean_abs_diffs_per_frame), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(mean_abs_diffs_per_frame):.3f} mm')
            ax.set_xlabel('Frame Index (relative)', fontsize=12)
            ax.set_ylabel('Mean Absolute Edge Length Difference (mm)', fontsize=12)
            ax.set_title('Mean Absolute Edge Length Difference Over Time', fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            
            time_series_path = output_dir / "edge_length_diff_time_series.png"
            plt.savefig(str(time_series_path), dpi=150)
            plt.close()
            print(f"Saved time series plot to {time_series_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
