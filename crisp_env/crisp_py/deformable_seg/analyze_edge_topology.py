"""
Analyze edge topology across frames.
Load saved keypoints and edges, check topology consistency, and compute edge length statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def edges_to_set(edges):
    """Convert edge list to a set of sorted tuples for comparison."""
    return set(tuple(sorted(e)) for e in edges)


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
    # Step 1: Analyze topology consistency
    # ============================================================
    print("\n" + "=" * 60)
    print("TOPOLOGY ANALYSIS")
    print("=" * 60)
    
    # Find frame 0's topology as reference
    reference_edges = None
    reference_frame = None
    for i, edges in enumerate(all_edges):
        if len(edges) > 0:
            reference_edges = edges
            reference_frame = i
            break
    
    if reference_edges is None:
        print("ERROR: No valid edges found in any frame!")
        return
    
    reference_edge_set = edges_to_set(reference_edges)
    n_reference_edges = len(reference_edges)
    print(f"Reference topology from frame {reference_frame}: {n_reference_edges} edges")
    print(f"Reference edges: {reference_edges[:10]}..." if len(reference_edges) > 10 else f"Reference edges: {reference_edges}")
    
    # Check each frame's topology
    matching_frames = []
    mismatched_frames = []
    empty_frames = []
    
    for frame_idx, edges in enumerate(all_edges):
        if len(edges) == 0:
            empty_frames.append(frame_idx)
            continue
        
        edge_set = edges_to_set(edges)
        
        if edge_set == reference_edge_set:
            matching_frames.append(frame_idx)
        else:
            # Find differences
            missing_edges = reference_edge_set - edge_set
            extra_edges = edge_set - reference_edge_set
            mismatched_frames.append({
                'frame': frame_idx,
                'n_edges': len(edges),
                'missing': missing_edges,
                'extra': extra_edges,
            })
    
    print(f"\nTopology Summary:")
    print(f"  Matching frames: {len(matching_frames)}")
    print(f"  Mismatched frames: {len(mismatched_frames)}")
    print(f"  Empty frames (skipped): {len(empty_frames)}")
    
    # Print all matching frames
    print(f"\nMatching frames: {matching_frames}")
    
    if empty_frames:
        print(f"\nEmpty frames: {empty_frames[:20]}..." if len(empty_frames) > 20 else f"\nEmpty frames: {empty_frames}")
    
    if mismatched_frames:
        print(f"\nMismatched frames details:")
        for info in mismatched_frames[:10]:  # Show first 10
            print(f"  Frame {info['frame']}: {info['n_edges']} edges")
            if info['missing']:
                print(f"    Missing edges: {list(info['missing'])[:5]}...")
            if info['extra']:
                print(f"    Extra edges: {list(info['extra'])[:5]}...")
        if len(mismatched_frames) > 10:
            print(f"  ... and {len(mismatched_frames) - 10} more mismatched frames")
    
    # ============================================================
    # Step 2: Compute edge length statistics for matching frames only
    # ============================================================
    print("\n" + "=" * 60)
    print("EDGE LENGTH STATISTICS (Matching Topology Frames Only)")
    print("=" * 60)
    
    if len(matching_frames) == 0:
        print("No frames with matching topology found!")
        return
    
    # Collect edge lengths for each edge across matching frames
    edge_lengths_per_edge = {edge_idx: [] for edge_idx in range(n_reference_edges)}
    
    for frame_idx in matching_frames:
        keypoints = all_keypoints_3d[frame_idx]
        if keypoints.shape[0] == 0:
            continue
        
        for edge_idx, (i, j) in enumerate(reference_edges):
            if i < keypoints.shape[0] and j < keypoints.shape[0]:
                edge_length = np.linalg.norm(keypoints[i] - keypoints[j])
                edge_lengths_per_edge[edge_idx].append(edge_length)
    
    print(f"Using {len(matching_frames)} frames with matching topology")
    
    # Prepare data for boxplot
    edge_data = []
    edge_labels = []
    for edge_idx in range(n_reference_edges):
        lengths = edge_lengths_per_edge[edge_idx]
        if len(lengths) > 0:
            edge_data.append(lengths)
            i, j = reference_edges[edge_idx]
            edge_labels.append(f"{i}-{j}")
    
    if len(edge_data) > 0:
        # Create boxplot
        fig, ax = plt.subplots(figsize=(max(14, n_reference_edges * 0.6), 7))
        bp = ax.boxplot(edge_data, labels=edge_labels, patch_artist=True)
        
        # Color the boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_xlabel('Edge (keypoint i - keypoint j)', fontsize=12)
        ax.set_ylabel('Edge Length (mm)', fontsize=12)
        ax.set_title(f'Edge Length Distribution Across {len(matching_frames)} Frames (Matching Topology Only)', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        output_dir = Path("/home/yehengz/deformable_seg/tracking_output")
        boxplot_path = output_dir / "edge_lengths_boxplot.png"
        plt.savefig(str(boxplot_path), dpi=150)
        plt.close()
        print(f"\nSaved edge length boxplot to {boxplot_path}")
        
        # Print summary statistics
        print("\nEdge Length Statistics (mm):")
        print(f"{'Edge':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'CV(%)':<10}")
        print("-" * 60)
        all_lengths = []
        for edge_idx in range(n_reference_edges):
            lengths = edge_lengths_per_edge[edge_idx]
            if len(lengths) > 0:
                all_lengths.extend(lengths)
                i, j = reference_edges[edge_idx]
                mean_len = np.mean(lengths)
                std_len = np.std(lengths)
                cv = (std_len / mean_len * 100) if mean_len > 0 else 0
                print(f"{i}-{j:<7} {mean_len:<10.2f} {std_len:<10.2f} {np.min(lengths):<10.2f} {np.max(lengths):<10.2f} {cv:<10.1f}")
        
        print("-" * 60)
        mean_all = np.mean(all_lengths)
        std_all = np.std(all_lengths)
        cv_all = (std_all / mean_all * 100) if mean_all > 0 else 0
        print(f"{'Overall':<10} {mean_all:<10.2f} {std_all:<10.2f} {np.min(all_lengths):<10.2f} {np.max(all_lengths):<10.2f} {cv_all:<10.1f}")
        
        # Also create a histogram of all edge lengths
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(all_lengths, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(mean_all, color='red', linestyle='--', label=f'Mean: {mean_all:.2f}mm')
        ax.axvline(mean_all - std_all, color='orange', linestyle=':', label=f'±1 Std: {std_all:.2f}mm')
        ax.axvline(mean_all + std_all, color='orange', linestyle=':')
        ax.set_xlabel('Edge Length (mm)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Distribution of All Edge Lengths', fontsize=14)
        ax.legend()
        plt.tight_layout()
        
        hist_path = output_dir / "edge_lengths_histogram.png"
        plt.savefig(str(hist_path), dpi=150)
        plt.close()
        print(f"Saved edge length histogram to {hist_path}")
    
    # ============================================================
    # Step 3: Visualize randomly selected frames with keypoint indices
    # ============================================================
    print("\n" + "=" * 60)
    print("VISUALIZING RANDOM FRAMES WITH TOPOLOGY")
    print("=" * 60)
    
    # Randomly select 10 frames (or fewer if not enough matching frames)
    n_viz = min(10, len(matching_frames))
    np.random.seed(42)  # For reproducibility
    selected_frames = np.random.choice(matching_frames, size=n_viz, replace=False)
    selected_frames = sorted(selected_frames)
    print(f"Randomly selected {n_viz} frames for visualization: {list(selected_frames)}")
    
    # Create a 2x5 grid of 3D plots
    n_rows = 2
    n_cols = 5
    fig = plt.figure(figsize=(20, 10))
    
    for plot_idx, frame_idx in enumerate(selected_frames):
        ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1, projection='3d')
        
        keypoints = all_keypoints_3d[frame_idx]
        
        # Plot edges
        for (i, j) in reference_edges:
            if i < keypoints.shape[0] and j < keypoints.shape[0]:
                xs = [keypoints[i, 0], keypoints[j, 0]]
                ys = [keypoints[i, 1], keypoints[j, 1]]
                zs = [keypoints[i, 2], keypoints[j, 2]]
                ax.plot(xs, ys, zs, 'b-', linewidth=1.5, alpha=0.7)
        
        # Plot keypoints
        ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], 
                   c='red', s=50, depthshade=True)
        
        # Add keypoint indices as text labels
        for kp_idx, kp in enumerate(keypoints):
            ax.text(kp[0], kp[1], kp[2], str(kp_idx), fontsize=8, color='black',
                    ha='center', va='bottom')
        
        ax.set_title(f'Frame {frame_idx}', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.suptitle(f'Keypoint Topology Visualization ({n_viz} Random Frames)', fontsize=14)
    plt.tight_layout()
    
    topology_viz_path = output_dir / "topology_visualization.png"
    plt.savefig(str(topology_viz_path), dpi=150)
    plt.close()
    print(f"Saved topology visualization to {topology_viz_path}")
    
    # Also create a single detailed view of frame 0
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    frame_idx = reference_frame
    keypoints = all_keypoints_3d[frame_idx]
    
    # Plot edges with labels
    for edge_idx, (i, j) in enumerate(reference_edges):
        if i < keypoints.shape[0] and j < keypoints.shape[0]:
            xs = [keypoints[i, 0], keypoints[j, 0]]
            ys = [keypoints[i, 1], keypoints[j, 1]]
            zs = [keypoints[i, 2], keypoints[j, 2]]
            ax.plot(xs, ys, zs, 'b-', linewidth=2, alpha=0.7)
            # Add edge label at midpoint
            mid = (keypoints[i] + keypoints[j]) / 2
            ax.text(mid[0], mid[1], mid[2], f'{i}-{j}', fontsize=7, color='blue', alpha=0.8)
    
    # Plot keypoints
    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], 
               c='red', s=100, depthshade=True)
    
    # Add keypoint indices
    for kp_idx, kp in enumerate(keypoints):
        ax.text(kp[0], kp[1], kp[2], f'  {kp_idx}', fontsize=10, color='black', fontweight='bold')
    
    ax.set_title(f'Reference Topology (Frame {frame_idx}) - {len(reference_edges)} Edges, {keypoints.shape[0]} Keypoints', fontsize=14)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    
    reference_viz_path = output_dir / "reference_topology.png"
    plt.savefig(str(reference_viz_path), dpi=150)
    plt.close()
    print(f"Saved reference topology visualization to {reference_viz_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
