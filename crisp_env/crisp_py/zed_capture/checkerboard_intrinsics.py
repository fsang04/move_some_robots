#!/usr/bin/env python3
"""Solve ZED intrinsics from checkerboard captures -- and grade the factory model.

Three fits, so the output answers three separate questions:

  model A  K = factory, dist = 0 (only poses fit)   "how wrong is what the
                                                     pipeline assumes today?"
  model B  K free, dist = 0                          "is it just fx/cx/cy, or
                                                     is there real distortion?"
  model C  K free, k1 k2 p1 p2 (k3 fixed 0)          the proposed replacement

If model C's rms is far below B's, the rectified stream carries residual
distortion -- the optics hypothesis for the range-scale error. If B ~ C but
both beat A, the factory fx/cx/cy are off but the stream is distortion-free.
If A ~ B ~ C at sub-pixel rms, the camera model is fine and the scale error
lives elsewhere (tag, gripper2tag).

Writes zed_intrinsics_checkerboard_{W}x{H}.npz (same key schema as
zed_intrinsics_2208x1242.npz, plus provenance). Never overwrites the factory
export.

    pixi run -e humble python zed_capture/checkerboard_intrinsics.py \
        zed_capture/checkerboard_HD2K --square-mm 25.0   # your MEASURED value

--selftest runs the whole solve path on synthetic corners with a known ground
truth and asserts recovery -- run it once before trusting real output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
GRID = (9, 6)                                       # inner corners of the 10x7 boards
SB_FLAGS = (cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            | cv2.CALIB_CB_NORMALIZE_IMAGE)

FACTORY = _HERE / 'zed_intrinsics_2208x1242.npz'


def object_grid(square_mm):
    ow, oh = GRID
    obj = np.zeros((ow * oh, 3), np.float32)
    obj[:, :2] = np.mgrid[0:ow, 0:oh].T.reshape(-1, 2) * (square_mm / 1000.0)
    return obj                                       # meters


def detect_all(image_dir):
    paths = sorted(Path(image_dir).glob('frame_*.png'))
    if not paths:
        raise SystemExit(f'no frame_*.png in {image_dir}')
    pts, used, shape = [], [], None
    for p in paths:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        shape = gray.shape
        ok, corners = cv2.findChessboardCornersSB(gray, GRID, flags=SB_FLAGS)
        if not ok:
            print(f'  {p.name}: no detection -- skipped')
            continue
        pts.append(corners.reshape(-1, 1, 2).astype(np.float32))
        used.append(p.name)
    print(f'detected {len(used)}/{len(paths)} images')
    if len(used) < 10:
        raise SystemExit('fewer than 10 usable views -- capture more')
    return pts, used, shape


def per_view_rms(obj, img_pts, rvecs, tvecs, K, dist):
    out = []
    for ip, rv, tv in zip(img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        out.append(float(np.sqrt(((proj - ip) ** 2).sum(axis=2).mean())))
    return np.asarray(out)


def fit(obj, img_pts, shape, flags, K0=None, d0=None, label=''):
    objs = [obj] * len(img_pts)
    K0 = K0.copy() if K0 is not None else None
    d0 = d0.copy() if d0 is not None else np.zeros(5)
    (rms, K, dist, rvecs, tvecs,
     sd_int, _sd_ext, _pve) = cv2.calibrateCameraExtended(
        objs, img_pts, shape[::-1], K0, d0, flags=flags)
    pv = per_view_rms(obj, img_pts, rvecs, tvecs, K, dist)
    print(f'\n[{label}] rms {rms:.3f} px   per-view median {np.median(pv):.3f} '
          f'max {pv.max():.3f}')
    print(f'  fx {K[0,0]:9.3f} +/- {sd_int[0][0]:.3f}    fy {K[1,1]:9.3f} +/- {sd_int[1][0]:.3f}')
    print(f'  cx {K[0,2]:9.3f} +/- {sd_int[2][0]:.3f}    cy {K[1,2]:9.3f} +/- {sd_int[3][0]:.3f}')
    print(f'  dist {np.round(dist.ravel()[:5], 6)}')
    return rms, K, dist, pv


ZERO_DIST = (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
             | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
             | cv2.CALIB_FIX_TANGENT_DIST)


def run(image_dir, square_mm, save=True):
    obj = object_grid(square_mm)
    img_pts, used, shape = detect_all(image_dir)
    H, W = shape

    Kf = None
    if FACTORY.exists() and (W, H) == (2208, 1242):
        Kf = np.asarray(np.load(FACTORY)['K'], dtype=np.float64)

    results = {}
    if Kf is not None:
        results['A: factory K, dist=0'] = fit(
            obj, img_pts, shape,
            ZERO_DIST | cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH
            | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO,
            K0=Kf, label='A: factory K, dist=0')
    results['B: free K, dist=0'] = fit(obj, img_pts, shape, ZERO_DIST,
                                       label='B: free K, dist=0')
    results['C: free K, k1 k2 p1 p2'] = fit(obj, img_pts, shape, cv2.CALIB_FIX_K3,
                                            label='C: free K, k1 k2 p1 p2')

    rmsC, KC, distC, _ = results['C: free K, k1 k2 p1 p2']
    if Kf is not None:
        print(f'\nfactory vs model C:  d_fx {KC[0,0]-Kf[0,0]:+.3f} px '
              f'({(KC[0,0]/Kf[0,0]-1)*100:+.3f} %)   d_cx {KC[0,2]-Kf[0,2]:+.3f} px   '
              f'd_cy {KC[1,2]-Kf[1,2]:+.3f} px')

    if save:
        out = _HERE / f'zed_intrinsics_checkerboard_{W}x{H}.npz'
        np.savez(out, K=KC, dist=distC.ravel(), width=W, height=H,
                 rms_px=rmsC, n_views=len(used), square_mm=square_mm,
                 model='pinhole k1 k2 p1 p2', source='checkerboard_intrinsics.py',
                 stream='VIEW.LEFT rectified, self-calib disabled')
        print(f'\nsaved -> {out}')
    return results


def selftest():
    """Synthesize corners from a known camera, recover it, assert agreement."""
    rng = np.random.default_rng(0)
    W, H = 2208, 1242
    K_true = np.array([[1400.0, 0, 1105.0], [0, 1402.5, 665.0], [0, 0, 1]])
    d_true = np.array([-0.02, 0.01, 0.0005, -0.0003, 0.0])
    obj = object_grid(25.0)
    img_pts = []
    n = 0
    while n < 45:
        rv = np.deg2rad(rng.uniform([-35, -35, -180], [35, 35, 180]))
        tv = np.array([rng.uniform(-0.35, 0.35), rng.uniform(-0.22, 0.22),
                       rng.uniform(0.5, 1.6)])
        tv[:2] += np.array([-0.1, -0.06])            # centre the board-ish
        proj, _ = cv2.projectPoints(obj, rv, tv, K_true, d_true)
        uv = proj.reshape(-1, 2)
        if uv.min() < 5 or uv[:, 0].max() > W - 5 or uv[:, 1].max() > H - 5:
            continue
        uv = uv + rng.normal(0, 0.05, uv.shape)      # 0.05 px corner noise
        img_pts.append(uv.reshape(-1, 1, 2).astype(np.float32))
        n += 1
    objs = [obj] * n
    rms, K, dist, *_ = cv2.calibrateCameraExtended(
        objs, img_pts, (W, H), None, np.zeros(5), flags=cv2.CALIB_FIX_K3)
    # Individual parameters trade off against each other (cx vs p2, fx vs k1),
    # so compare the MODELS AS MAPPINGS: project one grid of 3D rays through
    # both cameras and measure the worst pixel disagreement over the frame.
    gx, gy = np.meshgrid(np.linspace(-0.55, 0.55, 23), np.linspace(-0.35, 0.35, 15))
    rays = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], axis=1).astype(np.float32)
    p_true, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K_true, d_true)
    p_rec, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K, dist)
    uv_t = p_true.reshape(-1, 2)
    inside = ((uv_t[:, 0] >= 0) & (uv_t[:, 0] < W) & (uv_t[:, 1] >= 0) & (uv_t[:, 1] < H))
    d = (p_rec.reshape(-1, 2) - uv_t)[inside]
    # The CONSTANT part of the field is a principal-point shift, which any
    # downstream pose/hand-eye solve absorbs exactly (cx trades 1:1 with camera
    # orientation). What cannot be absorbed -- and what we are hunting in the
    # real camera -- is the non-constant part: scale and distortion mismatch.
    shift = d.mean(axis=0)
    resid = np.linalg.norm(d - shift, axis=1)
    print(f'selftest: rms {rms:.4f} px   constant shift {np.linalg.norm(shift):.3f} px '
          f'(absorbed downstream)   non-constant mapping error '
          f'median {np.median(resid):.4f}, max {resid.max():.4f} px '
          f'({inside.sum()} grid points)')
    assert rms < 0.15 and resid.max() < 0.5 and np.linalg.norm(shift) < 3.0, \
        'SELFTEST FAILED'
    print('selftest PASSED: non-absorbable model error < 0.5 px everywhere')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('image_dir', nargs='?')
    p.add_argument('--square-mm', type=float, default=None,
                   help='MEASURED square size (average a run of squares)')
    p.add_argument('--no-save', action='store_true')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    if not a.image_dir or a.square_mm is None:
        p.error('image_dir and --square-mm are required (or use --selftest)')
    run(a.image_dir, a.square_mm, save=not a.no_save)
