"""
Run SpaTrackerV2 on all chunks using wire tracking keypoints as prompt points.

For each chunk:
1. Load tracking keypoints from first frame of the chunk
2. Convert RGBD to SpaTrackerV2 format with depth filtering
3. Run SpaTrackerV2 with keypoints as query points
4. Save raw resolution visualization
"""
import numpy as np
import os
import subprocess
import sys

# Configuration
TRACKING_RESULTS_PATH = "/home/yehengz/deformable_seg/data/arm_traj2/wire_tracking_output/tracking_results.npy"
CHUNKS_DIR = "/home/yehengz/deformable_seg/data/bdlo_traj2"
SPATRACKER_DIR = "/home/yehengz/SpaTrackerV2"
FRAMES_PER_CHUNK = 45
NUM_CHUNKS = 5

# Camera intrinsics
FX = 606.1124267578125
FY = 605.8821411132812
CX = 641.7578125
CY = 365.6518859863281
DEPTH_SCALE = 1000.0
MAX_DEPTH = 1200.0  # mm


def get_chunk_keypoints_and_edges(tracking_results, chunk_idx, frames_per_chunk=45):
    """Get keypoints_2d and edges from the first frame of the specified chunk.
    
    Returns:
        keypoints_2d: (N, 2) array in (row, col) format
        edges: list of (i, j) edge tuples
    """
    first_frame_idx = chunk_idx * frames_per_chunk
    
    if first_frame_idx >= len(tracking_results):
        raise ValueError(f"Chunk {chunk_idx} starts at frame {first_frame_idx}, but only {len(tracking_results)} frames available")
    
    result = tracking_results[first_frame_idx]
    
    if not result['success']:
        # Find first successful frame in the chunk
        for i in range(first_frame_idx, min(first_frame_idx + frames_per_chunk, len(tracking_results))):
            if tracking_results[i]['success']:
                result = tracking_results[i]
                print(f"  Using frame {i} instead (first successful in chunk)")
                break
        else:
            raise ValueError(f"No successful tracking in chunk {chunk_idx}")
    
    keypoints_2d = result['keypoints_2d']  # Shape: (N, 2) - (row, col) format
    edges = result['edges']  # List of (i, j) tuples
    return keypoints_2d, edges


def create_queries_from_keypoints(keypoints_2d):
    """
    Convert keypoints_2d (row, col) to query format (t, x, y).
    
    SpaTrackerV2 expects queries as (t, x, y) where:
    - t: frame index (0 for first frame)
    - x: column (horizontal)
    - y: row (vertical)
    """
    N = len(keypoints_2d)
    queries = np.zeros((N, 3), dtype=np.float32)
    queries[:, 0] = 0  # t = 0 (first frame)
    queries[:, 1] = keypoints_2d[:, 1]  # x = col
    queries[:, 2] = keypoints_2d[:, 0]  # y = row
    return queries


def convert_rgbd(chunk_idx):
    """Convert RGBD data for a chunk."""
    input_path = os.path.join(CHUNKS_DIR, f"chunk{chunk_idx}", "rgbd.npz")
    output_path = os.path.join(CHUNKS_DIR, f"chunk{chunk_idx}", "spatracker_input.npz")
    
    print(f"  Converting RGBD: {input_path}")
    
    # Load data
    data = np.load(input_path, allow_pickle=True)
    color = data['color']
    depth = data['depth'].astype(np.float32)
    
    T, H, W, C = color.shape
    print(f"    Color shape: {color.shape}")
    
    # Convert color to (T, C, H, W) and normalize to [0, 1]
    video = color.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    
    # Apply depth threshold
    background_mask = depth > MAX_DEPTH
    n_filtered = np.sum(background_mask)
    print(f"    Filtering depth > {MAX_DEPTH}mm: {n_filtered}/{depth.size} pixels ({100*n_filtered/depth.size:.1f}%)")
    depth[background_mask] = 0
    
    # Convert to meters
    depths = depth / DEPTH_SCALE
    
    # Create intrinsics (same for all frames)
    K = np.array([
        [FX, 0, CX],
        [0, FY, CY],
        [0, 0, 1]
    ], dtype=np.float32)
    intrinsics = np.tile(K[None, :, :], (T, 1, 1))
    
    # Create extrinsics (identity for all frames)
    extrinsics = np.tile(np.eye(4, dtype=np.float32)[None, :, :], (T, 1, 1))
    
    # Save
    np.savez(output_path,
             video=video,
             depths=depths,
             intrinsics=intrinsics,
             extrinsics=extrinsics)
    
    print(f"    Saved to: {output_path}")
    return output_path


def run_spatracker_with_queries(chunk_idx, queries, edges):
    """Run SpaTrackerV2 with custom query points and edge definitions."""
    chunk_dir = os.path.join(CHUNKS_DIR, f"chunk{chunk_idx}")
    results_dir = os.path.join(chunk_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save queries to a file
    queries_path = os.path.join(chunk_dir, "queries.npy")
    np.save(queries_path, queries)
    print(f"  Saved {len(queries)} query points to: {queries_path}")
    
    # Save edges to a file
    edges_path = os.path.join(chunk_dir, "edges.npy")
    np.save(edges_path, edges, allow_pickle=True)
    print(f"  Saved {len(edges)} edges to: {edges_path}")
    
    # We need to modify the inference script to accept custom queries
    # For now, let's create a custom inference script
    inference_script = f'''
import sys
sys.path.insert(0, "{SPATRACKER_DIR}")

import numpy as np
import torch
import cv2
import os
from scipy.optimize import linear_sum_assignment
from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track

# Load data
data_dir = "{chunk_dir}"
npz_path = os.path.join(data_dir, "spatracker_input.npz")
raw_rgbd_path = os.path.join(data_dir, "rgbd.npz")
queries_path = os.path.join(data_dir, "queries.npy")
edges_path = os.path.join(data_dir, "edges.npy")
out_dir = os.path.join(data_dir, "results")

data = dict(np.load(npz_path, allow_pickle=True))
raw_data = dict(np.load(raw_rgbd_path, allow_pickle=True))
queries = np.load(queries_path)
edges = np.load(edges_path, allow_pickle=True).tolist()

video_tensor = torch.from_numpy(data["video"] * 255)
depth_tensor = data["depths"]
intrs = data["intrinsics"]
extrs = np.linalg.inv(data["extrinsics"])

# Raw color images for visualization (T, H, W, 3) uint8
raw_color = raw_data["color"]

print(f"Video shape: {{video_tensor.shape}}")
print(f"Depth shape: {{depth_tensor.shape}}")
print(f"Raw color shape: {{raw_color.shape}}")
print(f"Queries shape: {{queries.shape}}")
print(f"Edges: {{len(edges)}} edges")
print(f"Query points (t, x, y) where x=col, y=row:")
for i, q in enumerate(queries):
    print(f"  Point {{i}}: t={{q[0]:.0f}}, x={{q[1]:.1f}}, y={{q[2]:.1f}}")

# Load model
model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
model.spatrack.track_num = len(queries)
model.eval()
model.to("cuda")

# Run inference with custom queries
# NOTE: replace_ratio=0 to disable any point replacement/merging
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    (
        c2w_traj, intrs_out, point_map, conf_depth,
        track3d_pred, track2d_pred, vis_pred, conf_pred, video
    ) = model.forward(video_tensor, depth=depth_tensor,
                        intrs=intrs, extrs=extrs, 
                        queries=queries,
                        fps=1, full_point=False, iters_track=4,
                        query_no_BA=True, fixed_cam=False, stage=1, unc_metric=None,
                        support_frame=len(video_tensor)-1, replace_ratio=0.0)

# Save results
print(f"Track 2D shape: {{track2d_pred.shape}}")
print(f"Track 3D shape: {{track3d_pred.shape}}")

# Get track results - RAW output from SpaTrackerV2
# Format: track2d_np[t, i] = (x, y) = (col, row) for point i at time t
# NOTE: SpaTrackerV2 internally reorders points, so index i in output 
#       does NOT correspond to query index i
track2d_np = track2d_pred.cpu().numpy()[:, :, :2]  # (T, N, 2) - take only x,y
vis_np = vis_pred.cpu().numpy()  # (T, N)

# Debug: print track ranges
print(f"Track 2D x range: [{{track2d_np[..., 0].min():.1f}}, {{track2d_np[..., 0].max():.1f}}]")
print(f"Track 2D y range: [{{track2d_np[..., 1].min():.1f}}, {{track2d_np[..., 1].max():.1f}}]")
print(f"Visibility range: [{{vis_np.min():.2f}}, {{vis_np.max():.2f}}]")

# Check frame 0 correspondence
print(f"\\nFrame 0 query vs track positions:")
for i in range(len(queries)):
    qx, qy = queries[i, 1], queries[i, 2]
    tx, ty = track2d_np[0, i, 0], track2d_np[0, i, 1]
    dist = np.sqrt((qx-tx)**2 + (qy-ty)**2)
    status = 'OK' if dist < 5 else f'DRIFT={{dist:.0f}}'
    print(f"  Q{{i:2d}}: ({{qx:6.1f}},{{qy:5.1f}}) -> ({{tx:6.1f}},{{ty:5.1f}}) {{status}}")

# Hungarian algorithm mapping: find which track corresponds to which query
# SpaTrackerV2 internally reorders/merges points, so we need to map back
from scipy.optimize import linear_sum_assignment

query_pos = queries[:, 1:3]  # (N, 2) - x=col, y=row
track_pos_f0 = track2d_np[0, :, :2]  # (N, 2) - x=col, y=row at frame 0

# Build cost matrix: cost[q, t] = distance from query q to track t
n_queries = len(query_pos)
n_tracks = track_pos_f0.shape[0]
cost_matrix = np.zeros((n_queries, n_tracks))
for q in range(n_queries):
    for t in range(n_tracks):
        cost_matrix[q, t] = np.sqrt((query_pos[q, 0] - track_pos_f0[t, 0])**2 + 
                                     (query_pos[q, 1] - track_pos_f0[t, 1])**2)

# Solve assignment: query_idx[i] -> track_idx[i] is the optimal mapping
query_indices, track_indices = linear_sum_assignment(cost_matrix)

# Create mapping: for each query index, which track index to use
# query_to_track[q] = t means "use track t for query q"
query_to_track = dict(zip(query_indices, track_indices))

print(f"\\nHungarian mapping (query -> track):")
for qi in range(n_queries):
    ti = query_to_track.get(qi, qi)
    dist = cost_matrix[qi, ti]
    print(f"  query[{{qi:2d}}] -> track[{{ti:2d}}] (dist={{dist:.1f}})")

# Remap edges: original edge (i,j) uses queries i,j, need to use their corresponding tracks
remapped_edges = []
for (i, j) in edges:
    ti = query_to_track.get(i, i)
    tj = query_to_track.get(j, j)
    remapped_edges.append((ti, tj))
print(f"\\nRemapped {{len(edges)}} edges for track indices")

# Use raw color for visualization
T, H, W, C = raw_color.shape
N = track2d_np.shape[1]

print(f"\\nImage size: {{H}}x{{W}}, Num points: {{N}}, Num frames: {{T}}")
print(f"Using {{len(edges)}} edges from tracking results")

frames_dir = os.path.join(out_dir, "frames")
os.makedirs(frames_dir, exist_ok=True)

# Colors for visualization (rainbow)
colors = []
for i in range(N):
    hue = int(180 * i / N)
    color = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    colors.append(tuple(int(c) for c in color))

# Edge color (lime green like wire_tracking_main.py)
EDGE_COLOR = (50, 205, 50)  # BGR: lime green

# Trajectory tail length
TAIL_LENGTH = 30

for t in range(T):
    # Use raw color image (already uint8, RGB format)
    frame = raw_color[t].copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Draw trajectory tails first (so edges and points are on top)
    for i in range(N):
        # Get trajectory history for this point
        start_t = max(0, t - TAIL_LENGTH)
        traj = track2d_np[start_t:t+1, i, :2]  # (tail_len, 2) - x, y
        
        # Draw trajectory line with fading alpha
        for tt in range(len(traj) - 1):
            x1, y1 = traj[tt]
            x2, y2 = traj[tt + 1]
            
            # Check bounds
            if not (0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H):
                continue
            
            # Fade color based on position in tail
            alpha = (tt + 1) / len(traj)
            color_faded = tuple(int(c * alpha) for c in colors[i])
            
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_faded, 2)
    
    # Draw edge connections (wire structure) - draw before points so points are on top
    # Use REMAPPED edges to handle SpaTrackerV2's internal reordering
    # track2d format: (x, y) = (col, row), cv2 expects (col, row)
    for (ti, tj) in remapped_edges:
        if ti < N and tj < N:
            # track2d_np[t, ti, 0] = x = col, track2d_np[t, ti, 1] = y = row
            col1, row1 = float(track2d_np[t, ti, 0]), float(track2d_np[t, ti, 1])
            col2, row2 = float(track2d_np[t, tj, 0]), float(track2d_np[t, tj, 1])
            
            # Clamp to image bounds for drawing
            col1_draw = int(max(0, min(W-1, col1)))
            row1_draw = int(max(0, min(H-1, row1)))
            col2_draw = int(max(0, min(W-1, col2)))
            row2_draw = int(max(0, min(H-1, row2)))
            
            # Use consistent edge color (lime green)
            cv2.line(frame, (col1_draw, row1_draw), (col2_draw, row2_draw), EDGE_COLOR, 3)
    
    # Draw tracking points (ALL points unconditionally)
    n_visible = 0
    n_in_bounds = 0
    
    for i in range(N):
        # track2d format: (x, y) = (col, row)
        col, row = float(track2d_np[t, i, 0]), float(track2d_np[t, i, 1])
        visible = float(vis_np[t, i]) > 0.5
        
        in_bounds = 0 <= col < W and 0 <= row < H
        if in_bounds:
            n_in_bounds += 1
        if visible:
            n_visible += 1
        
        # Draw all points unconditionally - clamp to image for drawing
        col_draw = int(max(0, min(W-1, col)))
        row_draw = int(max(0, min(H-1, row)))
        
        # Color: full color always
        pt_color = colors[i]
        
        # Draw point (filled circle) - cv2 uses (col, row) = (x, y)
        cv2.circle(frame, (col_draw, row_draw), 8, pt_color, -1)
        
        # Draw point index with black outline for readability
        cv2.putText(frame, str(i), (col_draw + 10, row_draw + 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(frame, str(i), (col_draw + 10, row_draw + 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Add frame info
    info_text = f"Frame {{t}} | Points: {{N}} | Visible: {{n_visible}} | InBounds: {{n_in_bounds}}"
    cv2.putText(frame, info_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(frame, info_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
    
    # Save frame at raw resolution
    frame_path = os.path.join(frames_dir, f"frame_{{t:04d}}.png")
    cv2.imwrite(frame_path, frame)

print(f"Saved {{T}} frames at {{H}}x{{W}} resolution to {{frames_dir}}")

# Save NPZ results
data["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
data["track2d"] = track2d_np
data["track3d"] = track3d_pred.cpu().numpy()
data["visibility"] = vis_np
data["queries"] = queries
data["edges"] = np.array(edges, dtype=object)  # Save edges from tracking results

np.savez(os.path.join(out_dir, "result.npz"), **data, allow_pickle=True)
print(f"Results saved to {{out_dir}}/result.npz")
'''
    
    # Save and run the script
    script_path = os.path.join(chunk_dir, "run_spatracker.py")
    with open(script_path, 'w') as f:
        f.write(inference_script)
    
    print(f"  Running SpaTrackerV2...")
    result = subprocess.run(
        ["python", script_path],
        cwd=SPATRACKER_DIR,
        capture_output=False
    )
    
    return result.returncode == 0


def main():
    print("=" * 70)
    print("Running SpaTrackerV2 on all chunks with wire tracking keypoints")
    print("=" * 70)
    
    # Load tracking results
    print(f"\nLoading tracking results from: {TRACKING_RESULTS_PATH}")
    tracking_results = np.load(TRACKING_RESULTS_PATH, allow_pickle=True)
    print(f"Total frames: {len(tracking_results)}")
    
    # Process each chunk
    for chunk_idx in range(NUM_CHUNKS):
        print(f"\n{'='*70}")
        print(f"Processing chunk {chunk_idx}")
        print("=" * 70)
        
        # Get keypoints and edges from first frame of chunk
        print(f"\n1. Getting keypoints and edges from frame {chunk_idx * FRAMES_PER_CHUNK}...")
        try:
            keypoints_2d, edges = get_chunk_keypoints_and_edges(tracking_results, chunk_idx, FRAMES_PER_CHUNK)
            print(f"   Found {len(keypoints_2d)} keypoints, {len(edges)} edges")
        except Exception as e:
            print(f"   ERROR: {e}")
            continue
        
        # Convert to queries
        queries = create_queries_from_keypoints(keypoints_2d)
        print(f"   Query points: {queries.shape}")
        
        # Convert RGBD
        print(f"\n2. Converting RGBD data...")
        convert_rgbd(chunk_idx)
        
        # Run SpaTrackerV2
        print(f"\n3. Running SpaTrackerV2...")
        success = run_spatracker_with_queries(chunk_idx, queries, edges)
        
        if success:
            print(f"\n   ✓ Chunk {chunk_idx} completed successfully!")
        else:
            print(f"\n   ✗ Chunk {chunk_idx} failed!")
    
    print(f"\n{'='*70}")
    print("All chunks processed!")
    print("=" * 70)
    
    print(f"\n{'='*70}")
    print("All chunks processed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
