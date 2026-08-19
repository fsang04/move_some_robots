#!/usr/bin/env python3
"""One rectified stereo pair -> FoundationStereo depth, written as a .npy.

The live-pick counterpart of fs_depth_batch.py: dual_green_pick_zed_joint.py
--fs (humble env) grabs VIEW.LEFT + VIEW.RIGHT from the open ZED and invokes
this script in the foundation_stereo conda env (torch and rclpy share no env),
then reads the result back:

    ~/miniforge3/envs/foundation_stereo/bin/python fs_depth_single.py \
        --left left.png --right right.png --out depth_mm.npy

Output: (H, W) uint16 millimetres, 0 = invalid, registered to the LEFT
rectified view -- the same frame as ZED SDK depth.

The depth written here is RAW (uncorrected), matching the *_fs.npz convention
(disparity_offset_px = 0.0): the CALLER applies the live disparity-offset
correction, exactly as capture_rgbd_native() does for SDK depth. The offset
lives in the rectified images both matchers consume, so it applies to
FoundationStereo depth as much as to SDK depth.

fx/baseline default to zed_capture/zed_intrinsics_{W}x{H}.npz for the input
images' resolution (via fs_depth_batch._load_intrinsics -- no rescaling
fallback, see there). The INFERENCE CORE below is kept in sync with
fs_depth_batch.py; change one, change both.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from fs_depth_batch import DEFAULT_FS_ROOT, _load_intrinsics


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--left", required=True, help="left rectified view (png/bmp)")
    p.add_argument("--right", required=True, help="right rectified view")
    p.add_argument("--out", required=True,
                   help="output .npy: (H,W) uint16 mm RAW depth, 0 = invalid")
    p.add_argument("--fx", type=float, default=None,
                   help="rectified left fx in px (default: intrinsics npz for "
                        "the images' resolution)")
    p.add_argument("--baseline", type=float, default=None, help="baseline in m")
    p.add_argument("--fs-root", type=Path, default=DEFAULT_FS_ROOT,
                   help="FoundationStereo checkout")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--scale", type=float, default=1.0,
                   help="input downscale; keep 1.0 for pick accuracy")
    p.add_argument("--hiera", type=int, default=1)
    p.add_argument("--valid-iters", type=int, default=32)
    args = p.parse_args()

    left = cv2.imread(args.left, cv2.IMREAD_COLOR)
    right = cv2.imread(args.right, cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise SystemExit(f"cannot read {args.left} / {args.right}")
    if left.shape != right.shape:
        raise SystemExit(f"shape mismatch: left {left.shape} right {right.shape}")
    H, W = left.shape[:2]
    fx, baseline = _load_intrinsics(W, H, args.fx, args.baseline)
    print(f"[fs-single] {W}x{H}  fx={fx:.3f} px  baseline={baseline:.6f} m")

    # ---- inference core: keep in sync with fs_depth_batch.py ----
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
    t0 = time.time()
    model = FoundationStereo(OmegaConf.create(cfg))
    ckpt = torch.load(str(ckpt_path), weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    print(f"[fs-single] model loaded from {ckpt_path} ({time.time() - t0:.1f} s)")

    scale = float(args.scale)
    assert scale <= 1.0
    l = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    r = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
    if scale < 1.0:
        l = cv2.resize(l, fx=scale, fy=scale, dsize=None)
        r = cv2.resize(r, fx=scale, fy=scale, dsize=None)
    h, w = l.shape[:2]
    t0 = torch.as_tensor(l).cuda().float()[None].permute(0, 3, 1, 2)
    t1 = torch.as_tensor(r).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(t0.shape, divis_by=32, force_square=False)
    t0, t1 = padder.pad(t0, t1)
    t_inf = time.time()
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

    with np.errstate(divide="ignore", invalid="ignore"):
        depth_m = (fx * scale) * baseline / disp
    depth_m[bad] = 0.0
    if scale < 1.0:
        depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)

    d_mm = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
    d_mm[depth_m <= 0.0] = 0
    # ---- end inference core ----

    np.save(args.out, d_mm)
    valid = d_mm > 0
    print(f"[fs-single] inference {time.time() - t_inf:.1f} s  "
          f"valid={100.0 * valid.mean():.1f}%  "
          f"median={np.median(d_mm[valid]) / 1000.0 if valid.any() else 0:.3f} m")
    print(f"[fs-single] saved {args.out} (RAW uint16 mm, 0 = invalid)")


if __name__ == "__main__":
    main()
