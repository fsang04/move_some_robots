"""
Script to read and visualize the shape matching pickle data.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os


# Load data
pkl_path = "/home/yehengz/deformable_seg/data/shape_matching/Take 2024-12-10 02.32.54 PM_panda4_midpoint_part1.pkl"
output_dir = "/home/yehengz/deformable_seg/data/shape_matching"

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

bdlo_data = np.array(data)
print(bdlo_data.shape)
bdlo_data = bdlo_data.squeeze()
print(bdlo_data.shape)
bdlo_data = bdlo_data.T.reshape(-1, 20, 3)
print(bdlo_data.shape)
list_of_bdlo_data = [bdlo_data[i] for i in range(bdlo_data.shape[0])]


# Wire connections
connections = [
    {'points': [1, 2, 3, 4, 5], 'color': 'red'},
    {'points': [5, 6, 7, 8, 9], 'color': 'red'},
    {'points': [9, 10, 11, 12, 13], 'color': 'red'},
    {'points': [5, 14, 15, 16, 17], 'color': 'blue'},
    {'points': [9, 18, 19, 20], 'color': 'blue'}
]


# Compute global axis limits from all frames (handle NaN/Inf)
all_points = bdlo_data.reshape(-1, 3)  # (n_frames * 20, 3)

# Filter out NaN and Inf values
valid_mask = np.isfinite(all_points).all(axis=1)
valid_points = all_points[valid_mask]

print(f"\nValid points: {valid_points.shape[0]} / {all_points.shape[0]}")

if valid_points.shape[0] > 0:
    x_min, x_max = valid_points[:, 0].min(), valid_points[:, 0].max()
    y_min, y_max = valid_points[:, 1].min(), valid_points[:, 1].max()
    z_min, z_max = valid_points[:, 2].min(), valid_points[:, 2].max()
else:
    # Default limits if no valid data
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    z_min, z_max = -1, 1

# Add some padding
padding = 0.05
x_pad = (x_max - x_min) * padding
y_pad = (y_max - y_min) * padding
z_pad = (z_max - z_min) * padding

AXIS_LIMITS = {
    'x': (x_min - x_pad, x_max + x_pad),
    'y': (y_min - y_pad, y_max + y_pad),
    'z': (z_min - z_pad, z_max + z_pad)
}

print(f"\nGlobal axis limits:")
print(f"  X: [{AXIS_LIMITS['x'][0]:.3f}, {AXIS_LIMITS['x'][1]:.3f}]")
print(f"  Y: [{AXIS_LIMITS['y'][0]:.3f}, {AXIS_LIMITS['y'][1]:.3f}]")
print(f"  Z: [{AXIS_LIMITS['z'][0]:.3f}, {AXIS_LIMITS['z'][1]:.3f}]")


def visualize_data(data_chunk, connections):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_chunk[:, 2], data_chunk[:, 1], data_chunk[:, 0], color='black', s=20)

    for conn in connections:
        points = np.array(conn['points']) - 1
        ax.plot(data_chunk[points, 2], data_chunk[points, 1], data_chunk[points, 0], color=conn['color'])

    # Fixed axis limits (note: plotting uses Z, Y, X order)
    ax.set_xlim(AXIS_LIMITS['z'])
    ax.set_ylim(AXIS_LIMITS['y'])
    ax.set_zlim(AXIS_LIMITS['x'])

    elev = 80
    azim = 90
    roll = 180
    ax.view_init(elev=elev, azim=azim, roll=roll)

    ax.set_xlabel('Z')
    ax.set_ylabel('Y')
    ax.set_zlabel('X')
    ax.legend()
    plt.show()


def save_frame_image(data_chunk, connections, save_path, frame_idx=None, total_frames=None):
    """Save a single frame as image."""
    # Skip if data contains NaN or Inf
    if not np.isfinite(data_chunk).all():
        print(f"  Skipping frame {frame_idx} - contains NaN/Inf")
        return False
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_chunk[:, 2], data_chunk[:, 1], data_chunk[:, 0], color='black', s=20)

    for conn in connections:
        points = np.array(conn['points']) - 1
        ax.plot(data_chunk[points, 2], data_chunk[points, 1], data_chunk[points, 0], color=conn['color'], linewidth=2)

    # Fixed axis limits (note: plotting uses Z, Y, X order)
    ax.set_xlim(AXIS_LIMITS['z'])
    ax.set_ylim(AXIS_LIMITS['y'])
    ax.set_zlim(AXIS_LIMITS['x'])

    elev = 80
    azim = 90
    roll = 180
    ax.view_init(elev=elev, azim=azim, roll=roll)

    ax.set_xlabel('Z')
    ax.set_ylabel('Y')
    ax.set_zlabel('X')
    
    # Add frame info to title
    if frame_idx is not None and total_frames is not None:
        ax.set_title(f'Frame {frame_idx}/{total_frames}')
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    return True


def create_video(list_of_data, connections, output_path, fps=30, step=5):
    """Create video by saving frames and combining with cv2."""
    frames_dir = os.path.join(output_dir, "temp_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    frame_paths = []
    n_frames = len(list_of_data)
    total_video_frames = len(range(0, n_frames, step))
    
    print(f"Saving {total_video_frames} frames...")
    
    for idx, i in enumerate(range(0, n_frames, step)):
        frame_path = os.path.join(frames_dir, f"frame_{i:05d}.png")
        success = save_frame_image(list_of_data[i], connections, frame_path, 
                                   frame_idx=i, total_frames=n_frames)
        if success:
            frame_paths.append(frame_path)
        
        if idx % 50 == 0:
            print(f"  Frame {idx}/{total_video_frames}")
    
    if len(frame_paths) == 0:
        print("No valid frames to create video!")
        return
    
    # Combine frames into video using cv2
    print(f"Combining {len(frame_paths)} frames into video...")
    
    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        video.write(frame)
    
    video.release()
    
    # Clean up temp frames
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir(frames_dir)
    
    print(f"Video saved to: {output_path}")


# Main
print(f"\nTotal frames: {len(list_of_bdlo_data)}")
print(f"Frame shape: {list_of_bdlo_data[0].shape}")

# Save first frame
first_frame_path = os.path.join(output_dir, "shape_matching_frame0.png")
save_frame_image(list_of_bdlo_data[0], connections, first_frame_path, 
                frame_idx=0, total_frames=len(list_of_bdlo_data))
print(f"Saved first frame to: {first_frame_path}")

# Create video
video_path = os.path.join(output_dir, "SM1210_panda4_midpoint_video.mp4")
create_video(list_of_bdlo_data, connections, video_path, fps=30, step=20)
