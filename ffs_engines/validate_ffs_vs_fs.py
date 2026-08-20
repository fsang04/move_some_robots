"""Validate Fast-FoundationStereo TRT engines against offline FoundationStereo.

Runs each engine (via realtime/ffs_trt.py, the SAME wrapper the live tracker
uses) on the recorded HD2K stereo pairs of a calibration sequence, and compares
the RAW FFS depth against the RAW offline FS depth stored in
{side}_calibration_rgbd_fs.npz (hiera, 32 iters -- the quality reference that
the a/d correction was fitted on). Neither side carries the a/d correction, so
the comparison is matcher-vs-matcher on identical rectified images.

Metrics per engine, pooled over frames, in the 500-2500 mm band:
  - median / mean-abs / p95-abs depth difference (mm)
  - median implied disparity difference (px at the capture resolution)
  - inference ms per pair (after warmup)

Optionally writes {side}_calibration_rgbd_ffs.npz in the fs_depth_batch.py
format (RAW, disparity_offset_px=0.0), so the hand-eye solver can run the
arm-floor test on FFS depth:
    python calculate_base_to_cam_nonlinear_opt.py --camera zed --side right \
        --calib-seq-name <seq> --use-depth-translation \
        --rgbd-file right_calibration_rgbd_ffs.npz --no-enhance

Run with the humble pixi python (this is the deployment test of the wrapper):
    .pixi/envs/humble/bin/python validate_ffs_vs_fs.py --engine <path.engine>
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

TRACKING = Path.home() / ('move_some_robots/crisp_env/crisp_py/'
                          'trackDeform3D-core-tracking')
DATA = Path.home() / ('move_some_robots/crisp_env/crisp_py/'
                      'hand_to_eye_calibration/roahm-deformable-objects/'
                      'captured_calibration_data')
sys.path.insert(0, str(TRACKING))

from realtime.ffs_trt import FfsTrtMatcher  # noqa: E402


def run(engine_path, seq, side, n_frames, write_npz, z_band):
    seq_dir = DATA / seq
    raw = np.load(seq_dir / f'{side}_calibration_rgbd.npz')
    fs = np.load(seq_dir / f'{side}_calibration_rgbd_fs.npz')
    assert float(fs['disparity_offset_px']) == 0.0, 'FS npz must be RAW'
    lefts, rights = raw['color'], raw['color_right']
    fs_depth = fs['depth']
    fx = float(fs['fx_px'])
    baseline_m = float(fs['baseline_m'])
    n = len(lefts) if n_frames is None else min(n_frames, len(lefts))
    print(f'[val] {seq}/{side}: {n} pairs {lefts.shape[2]}x{lefts.shape[1]}, '
          f'fx={fx:.3f}, B={baseline_m * 1000:.2f} mm')

    matcher = FfsTrtMatcher(str(engine_path))
    fx_b_mm = fx * baseline_m * 1000.0

    dz_all, dd_all, times = [], [], []
    ffs_stack = (np.zeros((n,) + fs_depth.shape[1:], dtype=np.uint16)
                 if write_npz else None)
    for i in range(n):
        t0 = time.monotonic()
        d_ffs = matcher.depth_mm(lefts[i], rights[i], fx=fx,
                                 baseline_m=baseline_m)
        times.append((time.monotonic() - t0) * 1000.0)
        if ffs_stack is not None:
            ffs_stack[i] = np.clip(d_ffs, 0.0, 65535.0).astype(np.uint16)
        d_fs = fs_depth[i].astype(np.float32)
        ok = (d_ffs > 0) & (d_fs > 0) & (d_fs >= z_band[0]) & (d_fs <= z_band[1])
        if not ok.any():
            continue
        dz = d_ffs[ok] - d_fs[ok]
        dd = fx_b_mm / d_ffs[ok] - fx_b_mm / d_fs[ok]   # implied disparity, px
        dz_all.append(dz)
        dd_all.append(dd)
        if i % 12 == 0:
            print(f'  {i + 1}/{n}: overlap {100 * ok.mean():.1f}%  '
                  f'median dz {np.median(dz):+.1f} mm  '
                  f'median ddisp {np.median(dd):+.3f} px  '
                  f'{times[-1]:.0f} ms')

    dz = np.concatenate(dz_all)
    dd = np.concatenate(dd_all)
    t = np.array(times[2:])   # first pairs carry warmup
    print(f'[val] engine {Path(engine_path).name}')
    print(f'      depth diff vs FS : median {np.median(dz):+.2f} mm, '
          f'mean|.| {np.abs(dz).mean():.2f} mm, p95|.| '
          f'{np.percentile(np.abs(dz), 95):.2f} mm')
    print(f'      implied disparity: median {np.median(dd):+.4f} px '
          f'(at {lefts.shape[2]} px width)')
    print(f'      inference        : {t.mean():.1f} ms/pair '
          f'(min {t.min():.1f}, max {t.max():.1f})')

    if ffs_stack is not None:
        out = seq_dir / f'{side}_calibration_rgbd_ffs.npz'
        np.savez(out, color=lefts[:n], depth=ffs_stack,
                 disparity_offset_px=np.float64(0.0),
                 disparity_scale=np.float64(1.0),
                 ffs_engine=np.str_(str(engine_path)),
                 fx_px=np.float64(fx), baseline_m=np.float64(baseline_m))
        print(f'[val] wrote {out} (RAW: the solver applies a/d itself)')
    return float(np.median(dd)), float(t.mean())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--engine', required=True)
    p.add_argument('--seq', default='zed_calib_fs_002')
    p.add_argument('--side', default='right', choices=['left', 'right'])
    p.add_argument('--n', type=int, default=None)
    p.add_argument('--write-npz', action='store_true')
    p.add_argument('--z-band', type=float, nargs=2, default=(500.0, 2500.0))
    a = p.parse_args()
    run(a.engine, a.seq, a.side, a.n, a.write_npz, tuple(a.z_band))
