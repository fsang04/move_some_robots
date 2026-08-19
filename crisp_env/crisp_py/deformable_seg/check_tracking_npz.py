import numpy as np
import matplotlib.pyplot as plt
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


def create_point_cloud_video(color_data, depth_data, intrinsics, output_path, fps=30, downsample=8):
    """
    Create point cloud video using matplotlib.
    View: X right, -Y up, Z forward (positive).
    """
    print("Creating point cloud video (this may take a while)...")
    
    n_frames = len(color_data)
    
    # Set up figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    frames_rendered = []
    
    for i in range(n_frames):
        depth = depth_data[i]
        color = color_data[i]  # Already RGB
        
        # Resize color to match depth if needed
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
        ax.set_title(f'Frame {i}')
        
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
            print(f"  Rendered {i + 1}/{n_frames} frames")
    
    plt.close(fig)
    
    # Write video
    create_video_from_frames(frames_rendered, output_path, fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check tracking data from NPZ and create videos")
    parser.add_argument('--n_frames', type=int, default=None, 
                        help='Number of frames to process (default: all frames)')
    parser.add_argument('--input', type=str, 
                        default="/home/yehengz/deformable_seg/data/arm_traj5_cloth/rgbd.npz",
                        help='Input NPZ file path')
    parser.add_argument('--output_dir', type=str,
                        default="/home/yehengz/deformable_seg/data/arm_traj5_cloth",
                        help='Output directory')
    args = parser.parse_args()
    
    print("Loading tracking data from NPZ...")
    data = np.load(args.input)
    
    # NPZ has keys: 'color' (N x H x W x 3, BGR format) and 'depth' (N x H x W)
    color_data_bgr = data['color']  # N x H x W x 3 (BGR)
    depth_data = data['depth']  # N x H x W
    
    n_frames_total = len(color_data_bgr)
    print(f"Loaded {n_frames_total} frames")
    print(f"  Color shape: {color_data_bgr.shape} (BGR format)")
    print(f"  Depth shape: {depth_data.shape}")
    
    # Camera intrinsics
    intrinsics = np.array([
        [606.1124267578125, 0, 641.7578125],
        [0, 605.8821411132812, 365.6518859863281],
        [0, 0, 1]
    ])
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Limit frames if specified
    if args.n_frames is not None:
        n_frames = min(args.n_frames, n_frames_total)
    else:
        n_frames = n_frames_total
    
    print(f"Processing {n_frames} frames")
    
    # ============================================================
    # 1. Create color video (data is BGR, convert to RGB for display)
    # ============================================================
    print("\nCreating color video...")
    color_frames = []
    for i in range(n_frames):
        color_bgr = color_data_bgr[i]
        color_rgb = color_bgr[:, :, ::-1]  # BGR to RGB
        color_frames.append(color_rgb)
    
    # Save each color frame as an image
    rgb_imgs_dir = output_dir / "rgb_imgs"
    rgb_imgs_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(color_frames):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out_path = rgb_imgs_dir / f"frame_{idx:04d}.png"
        cv2.imwrite(str(out_path), frame_bgr)
    print(f"Saved {len(color_frames)} frames to {rgb_imgs_dir}")
    create_video_from_frames(color_frames, output_dir / "color_video.mp4", fps=30)
    
    # ============================================================
    # 2. Create depth video
    # ============================================================
    print("\nCreating depth video...")
    depth_frames = []
    for i in range(n_frames):
        depth = depth_data[i]
        depth_colored = normalize_depth_for_display(depth)
        depth_frames.append(depth_colored)
    create_video_from_frames(depth_frames, output_dir / "depth_video.mp4", fps=30)
    
    # # ============================================================
    # # 3. Create point cloud video (optional, slow)
    # # ============================================================
    # print("\nCreating point cloud video...")
    # # Convert BGR to RGB for point cloud visualization
    # color_data_rgb = color_data_bgr[:n_frames, :, :, ::-1]
    # create_point_cloud_video(
    #     color_data_rgb,
    #     depth_data[:n_frames],
    #     intrinsics, 
    #     output_dir / "point_cloud_video.mp4", 
    #     fps=30,
    #     downsample=16  # Downsample for faster rendering
    # )
    
    print("\nAll videos created successfully!")
    print(f"Output directory: {output_dir.absolute()}")
