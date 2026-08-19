#!/usr/bin/env python3
"""
zed_rgbd_snapshot.py

Export one RGB-D snapshot from BOTH the left and right ZED views.

WHY A SCRIPT
    No ZED GUI tool can do this. ZED_Explorer records raw stereo only (no depth),
    and ZED_Depth_Viewer saves left-view depth only. The SDK computes depth in the
    LEFT rectified frame by default; the right-view depth map (MEASURE.DEPTH_RIGHT)
    exists only when the camera is opened with enable_right_side_measure=True,
    which is what open_zed(right_side_measure=True) does.

OUTPUTS (in --out, default ./zed_snapshot_<timestamp>/)
    left_color.png       (H,W,3) uint8 BGR, rectified left view
    left_depth_mm.png    (H,W)   uint16 millimetres, 0 = invalid, pixel-aligned
                                 with left_color (same convention as the Azure /
                                 zed_calib_rgbd pipeline)
    left_depth_vis.png   colourised depth for eyeballing
    right_color.png, right_depth_mm.png, right_depth_vis.png
                         same three, for the RIGHT rectified view (right depth is
                         aligned to right_color, NOT to the left image)
    snapshot.npz         all four arrays + the open_zed() info dict

RUN (humble is the only pixi env with pyzed)
    cd ~/move_some_robots/crisp_env/crisp_py
    pixi run -e humble python \
        hand_to_eye_calibration/roahm-deformable-objects/zed_rgbd_snapshot.py

    Defaults follow zed_calib_rgbd (HD2K, NEURAL_PLUS, manual exposure 10 tuned
    for the dark cloth rig). For a normally lit scene pass --exposure auto.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zed_calib_rgbd as zc


def _depth_to_mm(depths: list[np.ndarray]) -> np.ndarray:
    """Median-combine float32 metre maps (NaN=invalid) into uint16 mm, 0=invalid."""
    if len(depths) == 1:
        depth_m = depths[0]
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN pixels
            depth_m = np.nanmedian(np.stack(depths, 0), axis=0)
    depth_m = np.where(np.isfinite(depth_m) & (depth_m > 0.0), depth_m, 0.0)
    depth_mm = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
    depth_mm[depth_m <= 0.0] = 0
    return depth_mm


def grab_rgbd_stereo(zed, runtime, median_frames: int):
    """Grab one RGB-D frame for each of the left and right rectified views.

    Returns:
        dict with keys left_color, left_depth_mm, right_color, right_depth_mm.
    """
    import pyzed.sl as sl

    n = max(1, int(median_frames))
    # Mats stay alive for the whole call; deep_copy=True is mandatory (see the
    # dangling-pointer note in zed_calib_rgbd.grab_rgbd).
    img_mat, depth_mat = sl.Mat(), sl.Mat()

    views = {
        "left":  (sl.VIEW.LEFT,  sl.MEASURE.DEPTH),
        "right": (sl.VIEW.RIGHT, sl.MEASURE.DEPTH_RIGHT),
    }
    colors = {k: [] for k in views}
    depths = {k: [] for k in views}

    for _ in range(n):
        for _attempt in range(10):
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                break
        else:
            raise RuntimeError("ZED grab() failed 10x")

        for side, (view, measure) in views.items():
            if zed.retrieve_image(img_mat, view) != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"retrieve_image({view}) failed")
            if zed.retrieve_measure(depth_mat, measure) != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(
                    f"retrieve_measure({measure}) failed -- was the camera opened "
                    "with right_side_measure=True?"
                )
            bgra = img_mat.get_data(deep_copy=True)
            colors[side].append(cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR))
            raw = np.asarray(depth_mat.get_data(deep_copy=True), dtype=np.float32)
            depths[side].append(np.where(np.isfinite(raw) & (raw > 0.0), raw, np.nan))

    out = {}
    for side in views:
        color = (colors[side][0] if n == 1 else
                 np.median(np.stack(colors[side], 0), axis=0).astype(np.uint8))
        depth_mm = _depth_to_mm(depths[side])
        valid = depth_mm > 0
        if valid.any():
            print(f"[snapshot] {side}: valid_depth={100.0 * valid.mean():.1f}%  "
                  f"range={depth_mm[valid].min() / 1000.0:.3f}-"
                  f"{depth_mm[valid].max() / 1000.0:.3f} m")
        else:
            print(f"[snapshot] WARN: {side} view has no valid depth")
        out[f"{side}_color"] = color
        out[f"{side}_depth_mm"] = depth_mm
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="output directory (default ./zed_snapshot_<timestamp>)")
    p.add_argument("--resolution", default=zc.DEFAULT_RESOLUTION,
                   help=f"sl.RESOLUTION name (default {zc.DEFAULT_RESOLUTION})")
    p.add_argument("--frames", type=int, default=zc.DEFAULT_MEDIAN_FRAMES,
                   help="per-pixel median over N grabs (default "
                        f"{zc.DEFAULT_MEDIAN_FRAMES}; scene must be static)")
    p.add_argument("--exposure", default=str(zc.DEFAULT_EXPOSURE),
                   help=f"manual exposure 0-100, or 'auto' (default {zc.DEFAULT_EXPOSURE}, "
                        "tuned for the dark cloth rig)")
    p.add_argument("--warmup", type=int, default=zc.DEFAULT_WARMUP)
    args = p.parse_args()

    exposure = None if args.exposure.lower() == "auto" else int(args.exposure)
    out_dir = args.out or Path(f"zed_snapshot_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    zed = None
    try:
        zed, runtime, info = zc.open_zed(
            resolution=args.resolution,
            exposure=exposure,
            warmup_frames=args.warmup,
            right_side_measure=True,
        )
        frames = grab_rgbd_stereo(zed, runtime, median_frames=args.frames)
    finally:
        zc.close_zed(zed)

    for side in ("left", "right"):
        cv2.imwrite(str(out_dir / f"{side}_color.png"), frames[f"{side}_color"])
        cv2.imwrite(str(out_dir / f"{side}_depth_mm.png"), frames[f"{side}_depth_mm"])
        zc.save_depth_vis(out_dir / f"{side}_depth_vis.png", frames[f"{side}_depth_mm"])
    np.savez_compressed(out_dir / "snapshot.npz", info=np.array(info, dtype=object),
                        **frames)

    print(f"[snapshot] wrote {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
