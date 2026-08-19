#!/usr/bin/env python3
"""
export_zed_intrinsics.py

Write a ZED intrinsics .npz in the exact format the existing hand-eye pipeline
expects, so the Azure solver runs on ZED data with ZERO code changes.

resolve_base2cam_from_depth.py does:

    fi = np.load(args.intrinsics)
    K, dist = fi["K"].astype(np.float64), fi["dist"].astype(np.float64)

and consumes them only through cv2.undistortPoints() and cv2.projectPoints().
Because VIEW.LEFT is RECTIFIED, dist is all zeros, and both of those calls reduce
to plain pinhole math -- which is exactly correct for the ZED. Nothing in the
solver needs to know it is not looking at an Azure.

CRITICAL: K must match the resolution of the IMAGES IT WILL BE USED ON. Intrinsics
are per-resolution. apriltag_image.py rescales a stored K to the frame resolution,
but only when the aspect ratio matches -- HD2K is exactly 16:9, so 2208x1242,
1280x720 and 640x360 interconvert exactly. Mixing a 1280x720 K with HD2K frames
without that rescale would be wrong by the 1.725x scale factor.

The calibration path (capture_poses_and_images_for_calibration_*.py -> zed_calib_rgbd.py)
records HD2K frames, so zed_intrinsics_2208x1242.npz is the natural export.

Sources:
  --from-capture <run_dir>  read intrinsics.json from an existing capture (no camera needed)
  (default)                 query the camera live

Usage:
    # from an existing capture -- no camera, no need to close the Depth Viewer
    python export_zed_intrinsics.py --from-capture real_captures/green_flat \\
        --width 1280 --height 720 --out zed_intrinsics_1280x720.npz

    # live
    python export_zed_intrinsics.py --resolution HD2K --width 1280 --height 720

Then:
    python resolve_base2cam_from_depth.py --intrinsics <this file> ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def k_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """3x3 pinhole camera matrix."""
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def scale_intrinsics(
    native: dict,
    target_w: int,
    target_h: int,
) -> tuple[dict, float, float]:
    """Scale a native intrinsics dict to a target resolution.

    Args:
        native:   dict with width, height, fx, fy, cx, cy
        target_w: target width
        target_h: target height

    Returns:
        (scaled_dict, scale_x, scale_y)

    Raises:
        ValueError if the aspect ratio changes by more than 0.1%, which would make
        a single pinhole model inconsistent with a uniformly-resized image.
    """
    sx = target_w / float(native["width"])
    sy = target_h / float(native["height"])
    if abs(sx - sy) / max(sx, sy) > 1e-3:
        raise ValueError(
            f"non-uniform scale: {native['width']}x{native['height']} -> {target_w}x{target_h} "
            f"gives sx={sx:.6f} sy={sy:.6f}. Pick a target with the same aspect ratio "
            "(HD2K 2208x1242 is exactly 16:9, so 1280x720 and 640x360 both work)."
        )
    scaled = {
        "width": int(target_w),
        "height": int(target_h),
        "fx": native["fx"] * sx,
        "fy": native["fy"] * sy,
        "cx": native["cx"] * sx,
        "cy": native["cy"] * sy,
    }
    return scaled, sx, sy


def intrinsics_from_capture(run_dir: Path) -> dict:
    """Read the native intrinsics block from a capture's intrinsics.json.

    Args:
        run_dir: a capture directory containing intrinsics.json.

    Returns:
        dict with width, height, fx, fy, cx, cy (plus serial_number / camera_model
        when present).
    """
    path = run_dir / "intrinsics.json"
    if not path.is_file():
        raise FileNotFoundError(f"no intrinsics.json in {run_dir}")
    data = json.loads(path.read_text())
    native = dict(data["native"])
    for key in ("serial_number", "camera_model", "sdk_version", "baseline_m"):
        if key in data:
            native[key] = data[key]
    return native


def intrinsics_live(resolution: str) -> dict:
    """Open the camera briefly and read its rectified left-cam intrinsics.

    Uses DEPTH_MODE.NONE so open() takes ~2 s instead of loading an AI model --
    intrinsics depend on resolution, not depth mode.
    """
    import zed_camera as zc

    cfg = zc.ZedInitConfig(resolution=resolution, depth_mode="NONE")
    zed = None
    try:
        zed, resolved = zc.open_zed(cfg)
        intr = zc.get_intrinsics(zed)
        native = dict(intr["native"])
        native["serial_number"] = resolved["serial_number"]
        native["camera_model"] = resolved["camera_model"]
        native["sdk_version"] = resolved["sdk_version"]
        if "baseline_m" in intr:
            native["baseline_m"] = intr["baseline_m"]
        return native
    finally:
        import zed_camera as zc2
        zc2.close_zed(zed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ZED rectified intrinsics as a K/dist npz for the hand-eye pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--from-capture", default=None,
                        help="capture dir containing intrinsics.json (skips opening the camera)")
    parser.add_argument("--resolution", default="HD2K",
                        help="live query resolution (ignored with --from-capture)")
    parser.add_argument("--width", type=int, default=1280,
                        help="target width -- MUST match the images you will click")
    parser.add_argument("--height", type=int, default=720, help="target height")
    parser.add_argument("--out", default=None,
                        help="output npz (default: zed_intrinsics_<W>x<H>.npz next to this script)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.from_capture:
        native = intrinsics_from_capture(Path(args.from_capture).expanduser().resolve())
        source = f"intrinsics.json in {args.from_capture}"
    else:
        native = intrinsics_live(args.resolution)
        source = f"live camera at {args.resolution}"

    scaled, sx, sy = scale_intrinsics(native, args.width, args.height)

    out = (
        Path(args.out).expanduser().resolve() if args.out
        else Path(__file__).resolve().parent / f"zed_intrinsics_{args.width}x{args.height}.npz"
    )

    K = k_matrix(scaled["fx"], scaled["fy"], scaled["cx"], scaled["cy"])
    # 8 zeros: OpenCV's rational model order [k1,k2,p1,p2,k3,k4,k5,k6], matching the
    # Azure npz shape so the solver's dist handling is identical. VIEW.LEFT is
    # rectified, so every coefficient is genuinely zero -- this is not an approximation.
    dist = np.zeros(8, dtype=np.float64)

    np.savez(
        out,
        K=K,
        dist=dist,
        width=np.int64(scaled["width"]),
        height=np.int64(scaled["height"]),
        native_width=np.int64(native["width"]),
        native_height=np.int64(native["height"]),
        scale=np.float64(sx),
        serial_number=np.int64(native.get("serial_number", -1)),
        rectified=np.bool_(True),
    )

    print(f"[intr] source     : {source}")
    print(f"[intr] native     : {native['width']}x{native['height']}  "
          f"fx={native['fx']:.6f} fy={native['fy']:.6f} "
          f"cx={native['cx']:.6f} cy={native['cy']:.6f}")
    print(f"[intr] scale      : {sx:.6f} (uniform)")
    print(f"[intr] exported   : {scaled['width']}x{scaled['height']}  "
          f"fx={scaled['fx']:.6f} fy={scaled['fy']:.6f} "
          f"cx={scaled['cx']:.6f} cy={scaled['cy']:.6f}")
    print(f"[intr] dist       : zeros(8)  (VIEW.LEFT is rectified)")
    print(f"[saved] {out}")
    print()
    print("AprilTag path -- RESOLUTION-INDEPENDENT, nothing else to do:")
    print("  calculate_base_to_cam_nonlinear_opt.py --camera zed ... --use-depth-translation")
    print("  apriltag_image._camera_params_for() rescales this file to whatever resolution")
    print("  each frame actually is (16:9 only), so capturing at HD2K 2208x1242 with this")
    print(f"  {scaled['width']}x{scaled['height']} export is correct. Look for this line at solve time:")
    print(f"    [apriltag_image] rescaling ZED intrinsics {scaled['width']}x{scaled['height']} -> <WxH>")
    print()
    print("Click path -- K is used AS-IS, so resolution MUST match exactly:")
    print(f"  python resolve_base2cam_from_depth.py --intrinsics {out} \\")
    print("      --corr <clicks.npz> --delay-frames 0 --out <transform.npz>")
    print(f"  The clicked images must be exactly {scaled['width']}x{scaled['height']}.")
    print()
    print("Intrinsics are NOT used during capture -- only at solve time.")


if __name__ == "__main__":
    main()
