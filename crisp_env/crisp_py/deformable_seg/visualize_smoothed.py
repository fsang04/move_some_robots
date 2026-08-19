#!/usr/bin/env python3
"""
Render a video comparing original vs smoothed 3D keypoint trajectories.

Usage:
    python visualize_smoothed.py --input_dir bdlo1_faster_free_ee_evaluation_results/chunk_0/clip_0
    python visualize_smoothed.py --input_dir bdlo1_faster_free_ee_evaluation_results/chunk_0/clip_0 --fps 30
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Render video of original vs smoothed keypoints")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing 3d_keypoints.npz and smoothed_3d_keypoints.npz")
    parser.add_argument("--smoothed_name", type=str, default="smoothed_3d_keypoints.npz")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path (default: <input_dir>/smoothed_comparison.mp4)")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    orig = np.load(input_dir / "3d_keypoints.npz")
    smooth = np.load(input_dir / args.smoothed_name)

    full_orig = orig["full"]       # T x K x 3
    full_smooth = smooth["full"]
    edges = orig["edge_connection"]  # E x 2
    T, K, _ = full_orig.shape

    output_path = args.output or str(input_dir / "smoothed_comparison.mp4")

    # Compute global axis limits from both datasets
    all_pts = np.concatenate([full_orig, full_smooth], axis=0)
    mins = np.nanmin(all_pts, axis=(0, 1))
    maxs = np.nanmax(all_pts, axis=(0, 1))
    center = (mins + maxs) / 2
    half_range = (maxs - mins).max() / 2 * 1.1

    fig = plt.figure(figsize=(14, 6))
    ax_orig = fig.add_subplot(1, 2, 1, projection="3d")
    ax_smooth = fig.add_subplot(1, 2, 2, projection="3d")
    fig.suptitle(f"{input_dir.parent.name}/{input_dir.name}", fontsize=13)

    def set_axes(ax, title):
        ax.set_xlim(center[0] - half_range, center[0] + half_range)
        ax.set_ylim(center[1] - half_range, center[1] + half_range)
        ax.set_zlim(center[2] - half_range, center[2] + half_range)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)

    def draw_frame(ax, pts, color):
        artists = []
        # Draw edges
        for i, j in edges:
            seg = pts[[i, j]]
            if not np.any(np.isnan(seg)):
                line, = ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], c=color, linewidth=1.5)
                artists.append(line)
        # Draw keypoints
        valid = ~np.any(np.isnan(pts), axis=1)
        sc = ax.scatter(pts[valid, 0], pts[valid, 1], pts[valid, 2],
                        c=color, s=20, depthshade=True)
        artists.append(sc)
        return artists

    frame_text = fig.text(0.5, 0.01, "", ha="center", fontsize=11)

    def update(t):
        ax_orig.cla()
        ax_smooth.cla()
        set_axes(ax_orig, "Original")
        set_axes(ax_smooth, "Smoothed")
        draw_frame(ax_orig, full_orig[t], "tab:blue")
        draw_frame(ax_smooth, full_smooth[t], "tab:orange")
        frame_text.set_text(f"Frame {t}/{T - 1}")

    print(f"Rendering {T} frames to {output_path} ...")
    anim = FuncAnimation(fig, update, frames=T, interval=1000 // args.fps)
    writer = FFMpegWriter(fps=args.fps, bitrate=2000)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
