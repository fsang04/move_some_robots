#!/usr/bin/env python3
"""
fs_depth_batch.py

Recompute the depth of a captured calibration sequence with FoundationStereo,
from the stereo pairs recorded by capture_poses_and_images_for_calibration*.py
(the ZED path saves color = VIEW.LEFT and color_right = VIEW.RIGHT since the
2026-08-18 edit; older captures have no color_right and cannot be processed).

Writes {side}_calibration_rgbd_fs.npz next to the original rgbd npz, in the
SAME format (color (N,H,W,3) BGR, depth (N,H,W) uint16 mm 0=invalid,
disparity_offset_px), so the solver consumes it via --rgbd-file:

    # 1. this script (foundation_stereo conda env, NOT the humble pixi env):
    ~/miniforge3/envs/foundation_stereo/bin/python fs_depth_batch.py \
        --calib-seq-name zed_calib_007 --side right

    # 2. solve twice, once per depth source (humble env, as usual):
    python calculate_base_to_cam_nonlinear_opt.py --camera zed --side right \
        --calib-seq-name zed_calib_007 --use-depth-translation
    python calculate_base_to_cam_nonlinear_opt.py --camera zed --side right \
        --calib-seq-name zed_calib_007 --use-depth-translation \
        --rgbd-file right_calibration_rgbd_fs.npz

FRAME / BIAS NOTES
    The FS point cloud is in the SAME frame as ZED depth: the left rectified
    camera frame (X right, Y down, Z forward). Both consume the identical
    SDK-rectified pair and unproject with the same rectified K, so nothing to
    align. Verified 2026-08-17 on zed_snapshot_20260817_163340: median
    disparity difference FS vs SDK = +0.06 px.

    For the same reason FS depth inherits this camera's constant disparity
    offset (the lens-yaw fault in zed_depth_correction.json) -- the offset
    lives in the rectified IMAGES, not in the SDK matcher. disparity_offset_px
    is therefore written as 0.0 (= raw, uncorrected), and the solver applies
    its usual correction to both depth sources alike.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent                    # zsy-testmycode
DATAPATH = _HERE.parent                                    # roahm-deformable-objects
_ZED_CAPTURE = _HERE.parents[2] / "zed_capture"            # crisp_py/zed_capture
DEFAULT_FS_ROOT = Path.home() / "move_some_robots" / "FoundationStereo"


def _load_intrinsics(width: int, height: int, fx_arg, baseline_arg):
    """(fx, baseline_m) for the captured resolution, from the shared ZED files.

    fx comes from zed_capture/zed_intrinsics_{W}x{H}.npz (RECTIFIED left camera,
    the same values the solver uses to unproject). There is deliberately NO
    rescaling fallback: the ZED rectifies separately per resolution, so fx at
    another resolution is NOT proportional to width (see the _rescaling note in
    zed_depth_correction.json). A capture at a new resolution needs its own
    intrinsics npz, or explicit --fx.
    """
    if fx_arg is not None and baseline_arg is not None:
        return float(fx_arg), float(baseline_arg)

    fx = fx_arg
    if fx is None:
        npz_path = _ZED_CAPTURE / f"zed_intrinsics_{width}x{height}.npz"
        if not npz_path.exists():
            raise SystemExit(
                f"No {npz_path.name} in {_ZED_CAPTURE} for this capture's "
                f"{width}x{height} resolution, and no --fx given. Do not scale "
                "another resolution's fx -- ZED rectified fx is per-resolution.")
        d = np.load(npz_path)
        K = d["K"]
        assert int(d["width"]) == width and int(d["height"]) == height
        fx = float(K[0, 0])
        print(f"[fs] fx = {fx:.6f} from {npz_path.name}")

    baseline = baseline_arg
    if baseline is None:
        json_path = _ZED_CAPTURE / "zed_depth_correction.json"
        with open(json_path) as f:
            baseline = float(json.load(f)["baseline_m"])
        print(f"[fs] baseline = {baseline:.6f} m from {json_path.name}")

    return float(fx), float(baseline)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--calib-seq-name", required=True)
    p.add_argument("--side", choices=["left", "right"], default="right",
                   help="which ARM's sequence (not which camera of the stereo head)")
    p.add_argument("--fs-root", type=Path, default=DEFAULT_FS_ROOT,
                   help="FoundationStereo repo root")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="checkpoint (default <fs-root>/pretrained_models/23-51-11/model_best_bp2.pth)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="run stereo matching at this scale (<=1). Depth is "
                        "upsampled back to native resolution either way, so the "
                        "output npz always matches the color frames.")
    p.add_argument("--hiera", type=int, default=1,
                   help="hierarchical inference; keep 1 for full-res HD2K")
    p.add_argument("--valid-iters", type=int, default=32)
    p.add_argument("--fx", type=float, default=None,
                   help="override rectified fx in px at CAPTURE resolution")
    p.add_argument("--baseline", type=float, default=None, help="override baseline in m")
    p.add_argument("--no-vis", action="store_true",
                   help="skip writing depth_fs_*.png preview images")
    args = p.parse_args()

    seq_dir = DATAPATH / "captured_calibration_data" / args.calib_seq_name
    rgbd_path = seq_dir / f"{args.side}_calibration_rgbd.npz"
    if not rgbd_path.exists():
        raise SystemExit(f"not found: {rgbd_path}")
    data = np.load(rgbd_path)
    if "color_right" not in data.files:
        raise SystemExit(
            f"{rgbd_path.name} has no color_right (keys: {data.files}). The right "
            "camera view is recorded by capture_poses_and_images_for_calibration_"
            "right.py since 2026-08-18; this sequence predates that -- re-capture.")
    colors = data["color"]          # (N,H,W,3) BGR, left rectified view
    rights = data["color_right"]    # (N,H,W,3) BGR, right rectified view
    n, H, W = colors.shape[0], colors.shape[1], colors.shape[2]
    print(f"[fs] {rgbd_path.name}: {n} pairs at {W}x{H}")

    fx, baseline = _load_intrinsics(W, H, args.fx, args.baseline)

    sys.path.insert(0, str(args.fs_root))
    import torch
    from omegaconf import OmegaConf
    from core.utils.utils import InputPadder
    from core.foundation_stereo import FoundationStereo

    torch.autograd.set_grad_enabled(False)
    ckpt_path = args.ckpt or args.fs_root / "pretrained_models/23-51-11/model_best_bp2.pth"
    cfg = OmegaConf.load(f"{ckpt_path.parent}/cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["valid_iters"] = args.valid_iters
    model = FoundationStereo(OmegaConf.create(cfg))
    ckpt = torch.load(str(ckpt_path), weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    print(f"[fs] model loaded from {ckpt_path}")

    scale = float(args.scale)
    assert scale <= 1.0
    depth_stack = np.zeros((n, H, W), dtype=np.uint16)
    t_start = time.time()
    for i in range(n):
        # left/right in RGB, as the model was trained
        l = cv2.cvtColor(colors[i], cv2.COLOR_BGR2RGB)
        r = cv2.cvtColor(rights[i], cv2.COLOR_BGR2RGB)
        if scale < 1.0:
            l = cv2.resize(l, fx=scale, fy=scale, dsize=None)
            r = cv2.resize(r, fx=scale, fy=scale, dsize=None)
        h, w = l.shape[:2]
        t0 = torch.as_tensor(l).cuda().float()[None].permute(0, 3, 1, 2)
        t1 = torch.as_tensor(r).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(t0.shape, divis_by=32, force_square=False)
        t0, t1 = padder.pad(t0, t1)
        with torch.amp.autocast("cuda", enabled=True):
            if args.hiera:
                disp = model.run_hierachical(t0, t1, iters=args.valid_iters,
                                             test_mode=True, small_ratio=0.5)
            else:
                disp = model.forward(t0, t1, iters=args.valid_iters, test_mode=True)
        disp = padder.unpad(disp.float()).cpu().numpy().reshape(h, w)

        # match falls off the right image -> unreliable, mark invalid
        xx = np.arange(w)[None, :].repeat(h, axis=0)
        bad = ~np.isfinite(disp) | (disp <= 0) | ((xx - disp) < 0)

        # disparity was measured on a (possibly) resized image, where fx scales
        # exactly by the resize factor (same image, unlike per-resolution SDK
        # rectification), so depth = (fx*scale)*B / disp at that scale.
        with np.errstate(divide="ignore", invalid="ignore"):
            depth_m = (fx * scale) * baseline / disp
        depth_m[bad] = 0.0
        if scale < 1.0:
            depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)

        d_mm = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
        d_mm[depth_m <= 0.0] = 0
        depth_stack[i] = d_mm
        valid = d_mm > 0
        print(f"[fs] {i + 1}/{n}: valid={100.0 * valid.mean():.1f}% "
              f"median={np.median(d_mm[valid]) / 1000.0 if valid.any() else 0:.3f} m "
              f"({(time.time() - t_start) / (i + 1):.1f} s/frame avg)")

    out_path = seq_dir / f"{args.side}_calibration_rgbd_fs.npz"
    # Uncompressed like the original, and same keys + provenance. color is the
    # LEFT view, identical to the source npz: FS depth is registered to it.
    np.savez(out_path, color=colors, depth=depth_stack,
             disparity_offset_px=np.float64(0.0),
             fs_model=str(ckpt_path), fs_scale=np.float64(scale),
             fs_hiera=np.int64(args.hiera), fs_valid_iters=np.int64(args.valid_iters),
             fx_px=np.float64(fx), baseline_m=np.float64(baseline))
    print(f"[fs] saved {out_path} depth {depth_stack.shape} uint16 mm (0 = invalid)")

    if not args.no_vis:
        sys.path.insert(0, str(DATAPATH))
        import zed_calib_rgbd
        frames_dir = seq_dir / "frames"
        for i in range(n):
            zed_calib_rgbd.save_depth_vis(
                frames_dir / f"depth_fs_{args.side}_image_{i}.png", depth_stack[i])
        print(f"[fs] saved {n} depth previews to {frames_dir}")


if __name__ == "__main__":
    main()
