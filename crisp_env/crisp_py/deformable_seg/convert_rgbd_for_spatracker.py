"""
Convert RGBD data to SpaTrackerV2 format.

SpaTrackerV2 expects an NPZ file with:
- video: (T, C, H, W) float32, values in [0, 1]
- depths: (T, H, W) float32, in meters
- intrinsics: (T, 3, 3) float32
- extrinsics: (T, 4, 4) float32
"""
import numpy as np
import argparse
import os

def convert_rgbd(input_path: str, output_path: str, 
                 fx: float, fy: float, cx: float, cy: float,
                 depth_scale: float = 1000.0,
                 max_depth: float = None):
    """
    Convert RGBD data to SpaTrackerV2 format.
    
    Args:
        input_path: Path to input npz with 'color' and 'depth'
        output_path: Path to output npz
        fx, fy: Focal lengths
        cx, cy: Principal point
        depth_scale: Depth scale factor (depth_value / depth_scale = meters)
        max_depth: Maximum depth threshold in mm (values above are set to 0)
    """
    print(f"Loading {input_path}...")
    data = np.load(input_path, allow_pickle=True)
    
    # Load color: (T, H, W, 3) uint8 -> (T, 3, H, W) float32 [0, 1]
    color = data['color']
    T, H, W, C = color.shape
    print(f"Color shape: {color.shape}")
    
    # Convert to (T, C, H, W) and normalize to [0, 1]
    video = color.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    print(f"Video shape: {video.shape}, range: [{video.min()}, {video.max()}]")
    
    # Load depth: (T, H, W) uint16 -> float32 in meters
    depth = data['depth'].astype(np.float32)
    
    # Apply max depth threshold (filter out background)
    if max_depth is not None:
        background_mask = depth > max_depth
        n_filtered = np.sum(background_mask)
        total_pixels = depth.size
        print(f"Filtering depth > {max_depth}mm: {n_filtered}/{total_pixels} pixels ({100*n_filtered/total_pixels:.1f}%)")
        depth[background_mask] = 0  # Set background to 0 (invalid)
    
    depths = depth / depth_scale
    print(f"Depths shape: {depths.shape}, range: [{depths.min()}, {depths.max()}] meters")
    
    # Create intrinsics: (T, 3, 3)
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)
    intrinsics = np.tile(K[None, :, :], (T, 1, 1))
    print(f"Intrinsics shape: {intrinsics.shape}")
    
    # Create extrinsics: identity (T, 4, 4)
    extrinsics = np.tile(np.eye(4, dtype=np.float32)[None, :, :], (T, 1, 1))
    print(f"Extrinsics shape: {extrinsics.shape}")
    
    # Save
    print(f"Saving to {output_path}...")
    np.savez(output_path,
             video=video,
             depths=depths,
             intrinsics=intrinsics,
             extrinsics=extrinsics)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RGBD to SpaTrackerV2 format")
    parser.add_argument("--input", type=str, required=True, help="Input npz path")
    parser.add_argument("--output", type=str, required=True, help="Output npz path")
    parser.add_argument("--fx", type=float, required=True, help="Focal length x")
    parser.add_argument("--fy", type=float, required=True, help="Focal length y")
    parser.add_argument("--cx", type=float, required=True, help="Principal point x")
    parser.add_argument("--cy", type=float, required=True, help="Principal point y")
    parser.add_argument("--depth_scale", type=float, default=1000.0, 
                        help="Depth scale (depth_value / scale = meters)")
    parser.add_argument("--max_depth", type=float, default=None,
                        help="Maximum depth threshold in mm (values above are set to 0)")
    
    args = parser.parse_args()
    
    convert_rgbd(
        input_path=args.input,
        output_path=args.output,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        depth_scale=args.depth_scale,
        max_depth=args.max_depth
    )
