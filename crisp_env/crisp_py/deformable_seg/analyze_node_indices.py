"""
Analyze node indices across frames.
Identify branch nodes and leaf nodes, and classify them by position.
"""

import numpy as np
from pathlib import Path


def main():
    # Load saved keypoints data
    data_path = Path("/home/yehengz/deformable_seg/tracking_output/all_keypoints_3d.npy")
    print(f"Loading data from {data_path}...")
    data = np.load(str(data_path), allow_pickle=True).item()
    
    all_keypoints_3d = data["keypoints"]
    all_edges = data["edges"]
    
    n_frames = len(all_keypoints_3d)
    print(f"Loaded {n_frames} frames")
    
    # Frames to analyze
    frames_to_analyze = [0, 15, 30, 45, 60, 128]
    
    print("\n" + "=" * 80)
    print("NODE CLASSIFICATION ANALYSIS")
    print("=" * 80)
    
    for frame_idx in frames_to_analyze:
        if frame_idx >= len(all_keypoints_3d):
            print(f"\nFrame {frame_idx}: Not available (only {len(all_keypoints_3d)} frames)")
            continue
        
        keypoints = all_keypoints_3d[frame_idx]
        edges = all_edges[frame_idx]
        
        if keypoints.shape[0] == 0:
            print(f"\nFrame {frame_idx}: Empty (skipped frame)")
            continue
        
        print(f"\n{'='*80}")
        print(f"Frame {frame_idx}: {keypoints.shape[0]} keypoints, {len(edges)} edges")
        print("="*80)
        
        # Compute degree of each node
        degrees = np.zeros(keypoints.shape[0], dtype=int)
        for (i, j) in edges:
            degrees[i] += 1
            degrees[j] += 1
        
        # Identify branch nodes (degree >= 3) and leaf nodes (degree == 1)
        branch_indices = np.where(degrees >= 3)[0]
        leaf_indices = np.where(degrees == 1)[0]
        
        print(f"\nBranch nodes (degree >= 3): {list(branch_indices)}")
        print(f"Leaf nodes (degree == 1): {list(leaf_indices)}")
        
        # Get 3D positions
        if len(branch_indices) >= 2:
            branch_positions = keypoints[branch_indices]
            
            # Sort by X coordinate (smaller X = left, larger X = right)
            sorted_by_x = np.argsort(branch_positions[:, 0])
            left_branch_idx = branch_indices[sorted_by_x[0]]
            right_branch_idx = branch_indices[sorted_by_x[-1]]
            
            print(f"\nBranch node positions (X, Y, Z):")
            for idx in branch_indices:
                pos = keypoints[idx]
                label = ""
                if idx == left_branch_idx:
                    label = " <- LEFT BRANCH (smaller X)"
                elif idx == right_branch_idx:
                    label = " <- RIGHT BRANCH (larger X)"
                print(f"  Node {idx}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}){label}")
        else:
            print(f"\nNot enough branch nodes to classify (found {len(branch_indices)})")
            left_branch_idx = None
            right_branch_idx = None
        
        if len(leaf_indices) >= 4:
            leaf_positions = keypoints[leaf_indices]
            
            # Classify leaves by X and Y coordinates
            # In camera coordinates: X is horizontal, Y is vertical (typically down is positive)
            # We'll use: smaller X = left, larger X = right
            #            smaller Y = top, larger Y = bottom
            
            print(f"\nLeaf node positions (X, Y, Z):")
            for idx in leaf_indices:
                pos = keypoints[idx]
                print(f"  Node {idx}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
            
            # Find the 4 corners based on X and Y
            # Top-left: small X, small Y
            # Top-right: large X, small Y
            # Bottom-left: small X, large Y
            # Bottom-right: large X, large Y
            
            # Method: Use quadrant assignment
            x_coords = leaf_positions[:, 0]
            y_coords = leaf_positions[:, 1]
            
            # Compute median to separate left/right and top/bottom
            x_median = np.median(x_coords)
            y_median = np.median(y_coords)
            
            top_left_idx = None
            top_right_idx = None
            bottom_left_idx = None
            bottom_right_idx = None
            
            for i, idx in enumerate(leaf_indices):
                x, y = x_coords[i], y_coords[i]
                is_left = x < x_median
                is_top = y < y_median
                
                if is_left and is_top:
                    top_left_idx = idx
                elif not is_left and is_top:
                    top_right_idx = idx
                elif is_left and not is_top:
                    bottom_left_idx = idx
                elif not is_left and not is_top:
                    bottom_right_idx = idx
            
            print(f"\nLeaf node classification (based on X-Y position):")
            print(f"  X median: {x_median:.1f}, Y median: {y_median:.1f}")
            print(f"  TOP-LEFT:     Node {top_left_idx}")
            print(f"  TOP-RIGHT:    Node {top_right_idx}")
            print(f"  BOTTOM-LEFT:  Node {bottom_left_idx}")
            print(f"  BOTTOM-RIGHT: Node {bottom_right_idx}")
            
            print(f"\nSummary for Frame {frame_idx}:")
            print(f"  Left Branch:   Node {left_branch_idx}")
            print(f"  Right Branch:  Node {right_branch_idx}")
            print(f"  Top-Left:      Node {top_left_idx}")
            print(f"  Top-Right:     Node {top_right_idx}")
            print(f"  Bottom-Left:   Node {bottom_left_idx}")
            print(f"  Bottom-Right:  Node {bottom_right_idx}")
        else:
            print(f"\nNot enough leaf nodes to classify (found {len(leaf_indices)})")
    
    print("\n" + "=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
