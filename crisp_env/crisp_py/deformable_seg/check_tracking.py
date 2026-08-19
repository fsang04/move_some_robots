import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import cv2
from pathlib import Path
import argparse


def depth_to_point_cloud(depth, intrinsics, rgb=None):
    """
    Lift depth map to 3D point cloud.
    
    Parameters
    ----------
    depth : np.ndarray, shape (H, W)
        Depth map.
    intrinsics : np.ndarray, shape (3, 3)
        Camera intrinsic matrix.
    rgb : np.ndarray, shape (H, W, 3), optional
        RGB image for coloring points.
    
    Returns
    -------
    points : np.ndarray, shape (N, 3)
        3D points.
    colors : np.ndarray, shape (N, 3), optional
        RGB colors normalized to [0, 1].
    """
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Compute 3D coordinates
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack and reshape
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    
    # Filter valid points (z > 0)
    valid_mask = points[:, 2] > 0
    points = points[valid_mask]
    
    colors = None
    if rgb is not None:
        colors = rgb.reshape(-1, 3)[valid_mask] / 255.0
    
    return points, colors


def normalize_depth_for_display(depth):
    """Normalize depth map for visualization."""
    valid_mask = depth > 0
    if not np.any(valid_mask):
        return np.zeros_like(depth, dtype=np.uint8)
    
    d_min = depth[valid_mask].min()
    d_max = depth[valid_mask].max()
    
    normalized = np.zeros_like(depth, dtype=np.float32)
    normalized[valid_mask] = (depth[valid_mask] - d_min) / (d_max - d_min + 1e-8)
    
    # Apply colormap
    colored = (plt.cm.viridis(normalized)[:, :, :3] * 255).astype(np.uint8)
    return colored


def create_video_from_frames(frames, output_path, fps=30):
    """Create video from list of frames."""
    if len(frames) == 0:
        print(f"No frames to write for {output_path}")
        return
    
    H, W = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    
    for frame in frames:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()
    print(f"Saved video: {output_path}")


def create_point_cloud_video(tracking_data, intrinsics, output_path, fps=30, downsample=8):
    """
    Create point cloud video using matplotlib.
    View: X right, -Y up, Z forward (positive).
    """
    print("Creating point cloud video (this may take a while)...")
    
    # Get sorted frame keys
    frame_keys = sorted(tracking_data.keys())
    
    # Set up figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    frames_rendered = []
    
    for i, frame_key in enumerate(frame_keys):
        data = tracking_data[frame_key]
        depth = data['transformed_depth']
        color_bgr = data['color']
        color = color_bgr[:, :, ::-1]  # BGR to RGB
        
        # Resize color to match transformed_depth if needed
        if color.shape[:2] != depth.shape:
            color_resized = cv2.resize(color, (depth.shape[1], depth.shape[0]))
        else:
            color_resized = color
        
        # Lift to 3D
        points, colors = depth_to_point_cloud(depth, intrinsics, color_resized)
        
        # Downsample for faster rendering
        if downsample > 1:
            points = points[::downsample]
            colors = colors[::downsample] if colors is not None else None
        
        # Clear and plot
        ax.clear()
        
        # Plot with X, -Y, Z orientation (flip Y for display)
        ax.scatter(
            points[:, 0],
            -points[:, 1],  # Flip Y
            points[:, 2],
            c=colors,
            s=0.5,
            marker='.',
        )
        
        # Set labels and view
        ax.set_xlabel('X')
        ax.set_ylabel('-Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {frame_key}')
        
        # Set consistent axis limits based on data
        ax.set_xlim([-500, 500])
        ax.set_ylim([-500, 500])
        ax.set_zlim([0, 2000])
        
        # Set view angle
        ax.view_init(elev=-90, azim=-90)
        
        # Render to image
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]  # RGBA -> RGB
        frames_rendered.append(img.copy())
        
        if (i + 1) % 10 == 0:
            print(f"  Rendered {i + 1}/{len(frame_keys)} frames")
    
    plt.close(fig)
    
    # Write video
    create_video_from_frames(frames_rendered, output_path, fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check tracking data and create videos")
    parser.add_argument('--n_frames', type=int, default=None, 
                        help='Number of frames to process (default: all frames)')
    args = parser.parse_args()
    
    print("Loading tracking data...")
    # tracking_data = np.load("./data/full/tracking_BDLO_data.npy", allow_pickle=True).item()
    # tracking_data = np.load("./data/arm_traj3/arm_with_wires_traj3_contact.npy", allow_pickle=True).item()
    tracking_data = np.load("/home/yehengz/deformable_seg/data/full/tracking_fabric2_data.npy", allow_pickle=True).item()


    # Camera intrinsics
    intrinsics = np.array([
        [606.1124267578125, 0, 641.7578125],
        [0, 605.8821411132812, 365.6518859863281],
        [0, 0, 1]
    ])
    
    # Output directory
    output_dir = Path("./data/arm_traj5_cloth")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get sorted frame keys
    frame_keys = sorted(tracking_data.keys())
    
    # Limit frames if specified
    if args.n_frames is not None:
        frame_keys = frame_keys[:args.n_frames]
    
    print(f"Processing {len(frame_keys)} frames")
    
    # Print sample frame info
    sample_data = tracking_data[frame_keys[0]]
    print("Frame data keys and shapes:")
    for k, v in sample_data.items():
        print(f"  {k}: {v.shape}")
    
    # ============================================================
    # 1. Create color video (data is BGR, convert to RGB for display)
    # ============================================================
    print("\nCreating color video...")
    color_frames = []
    for frame_key in frame_keys:
        color_bgr = tracking_data[frame_key]['color']
        color_rgb = color_bgr[:, :, ::-1]  # BGR to RGB
        color_frames.append(color_rgb)
    # Save each color frame as an image
    wire_tracking_frames_dir = Path("/home/yehengz/deformable_seg/data/arm_traj4_fabric/rgb_imgs")
    wire_tracking_frames_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(color_frames):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out_path = wire_tracking_frames_dir / f"frame_{idx:04d}.png"
        cv2.imwrite(str(out_path), frame_bgr)
    print(f"Saved {len(color_frames)} frames to {wire_tracking_frames_dir}")
    create_video_from_frames(color_frames, output_dir / "color_video_with_wires.mp4", fps=30)
    
    
    # ============================================================
    # 3. Create transformed_depth video (720, 1280)
    # ============================================================
    print("\nCreating transformed_depth video...")
    transformed_depth_frames = []
    for frame_key in frame_keys:
        depth = tracking_data[frame_key]['transformed_depth']
        depth_colored = normalize_depth_for_display(depth)
        transformed_depth_frames.append(depth_colored)
    create_video_from_frames(transformed_depth_frames, output_dir / "transformed_depth_video.mp4", fps=30)
    
    
    # # ============================================================
    # # 5. Create point cloud video
    # # ============================================================
    # print("\nCreating point cloud video...")
    # create_point_cloud_video(
    #     tracking_data, 
    #     intrinsics, 
    #     output_dir / "point_cloud_video.mp4", 
    #     fps=30,
    #     downsample=16  # Downsample for faster rendering
    # )
    
    print("\nAll videos created successfully!")
    print(f"Output directory: {output_dir.absolute()}")
