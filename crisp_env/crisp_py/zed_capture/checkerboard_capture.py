#!/usr/bin/env python3
"""Capture checkerboard views from the ZED for intrinsics calibration.

Opens the camera through the SAME path as every tag capture -- zed_calib_rgbd's
open_zed(), so the stream is byte-identical geometry: rectified VIEW.LEFT,
factory conf from the local settings dir, self-calibration disabled. Only the
photometric settings differ (shorter exposure allowed; motion blur is the enemy
of corner accuracy). Depth runs in PERFORMANCE mode purely because open_zed
enables depth; no depth is stored.

    cd $CP   # crisp_py
    pixi run -e humble python zed_capture/checkerboard_capture.py --out zed_capture/checkerboard_HD2K
    # and a second session for the live tracker's stream:
    pixi run -e humble python zed_capture/checkerboard_capture.py --resolution HD720 --out zed_capture/checkerboard_HD720

Keys:  SPACE save (only accepted if corners detect at full resolution)
       a     toggle auto-save (saves when detected + board moved since last save)
       q/ESC finish

Aim for 40+ accepted views: board large in frame (0.5-0.9 m) for most, several
at the 1.1-1.6 m working range, tilted up to ~40 deg in both axes, and corners
of the FRAME covered -- watch the 4x3 coverage grid in the preview. Distortion
is constrained only where corners have been seen.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_CALIB_DIR = _HERE.parent / 'hand_to_eye_calibration' / 'roahm-deformable-objects'
sys.path.insert(0, str(_CALIB_DIR))

# inner-corner grid of the printed 10x7-square boards
GRID = (9, 6)
SB_FLAGS_FULL = (cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
                 | cv2.CALIB_CB_NORMALIZE_IMAGE)


def detect(gray, fast=False):
    flags = 0 if fast else SB_FLAGS_FULL
    ok, corners = cv2.findChessboardCornersSB(gray, GRID, flags=flags)
    return (corners.reshape(-1, 2) if ok else None)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=str(_HERE / 'checkerboard_HD2K'))
    p.add_argument('--resolution', default='HD2K', choices=['HD2K', 'HD1080', 'HD720'])
    p.add_argument('--exposure', default='12',
                   help="percent, or 'auto'; shorter than the tag captures' 30 on "
                        'purpose -- a handheld board needs it')
    p.add_argument('--min-move-px', type=float, default=80.0,
                   help='auto mode: min corner-centroid motion between saves')
    args = p.parse_args()

    try:
        import zed_calib_rgbd
    except ImportError as exc:
        raise SystemExit(f'cannot import zed_calib_rgbd: {exc}')
    try:
        import pyzed.sl as sl
    except ImportError:
        raise SystemExit('pyzed not importable. Re-unzip the wheel (README_ZED.md '
                         'section 1) and export LD_LIBRARY_PATH=$HOME/zed_test/inspect/lib')

    exposure = None if args.exposure == 'auto' else int(args.exposure)
    zed, runtime, info = zed_calib_rgbd.open_zed(
        resolution=args.resolution, depth_mode='PERFORMANCE', exposure=exposure)
    print(f'[zed] open: {info}')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    img_mat = sl.Mat()
    saved, last_centroid, last_save_t = [], None, 0.0
    auto = False
    cov = np.zeros((3, 4), dtype=int)              # coverage grid, rows x cols

    print('SPACE=save   a=auto-save toggle   q=quit')
    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue
            if zed.retrieve_image(img_mat, sl.VIEW.LEFT) != sl.ERROR_CODE.SUCCESS:
                continue
            bgr = np.ascontiguousarray(img_mat.get_data()[:, :, :3])
            H, W = bgr.shape[:2]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            # fast half-res detection for live feedback only
            half = cv2.resize(gray, (W // 2, H // 2))
            c_fast = detect(half, fast=True)

            disp = cv2.resize(bgr, (W // 2, H // 2)).copy()
            if c_fast is not None:
                cv2.drawChessboardCorners(disp, GRID,
                                          c_fast.reshape(-1, 1, 2).astype(np.float32), True)
            for r in range(3):
                for c in range(4):
                    col = (0, 200, 0) if cov[r, c] else (0, 0, 180)
                    cv2.putText(disp, str(cov[r, c]),
                                (int((c + 0.45) * disp.shape[1] / 4),
                                 int((r + 0.55) * disp.shape[0] / 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
            cv2.putText(disp, f'saved {len(saved)}   auto={"ON" if auto else "off"}',
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.imshow('checkerboard capture', disp)
            k = cv2.waitKey(1) & 0xFF

            want_save = (k == ord(' '))
            if auto and c_fast is not None and time.time() - last_save_t > 1.0:
                cen = c_fast.mean(axis=0) * 2.0
                if last_centroid is None or np.linalg.norm(cen - last_centroid) > args.min_move_px:
                    want_save = True

            if want_save:
                corners = detect(gray, fast=False)          # full-res, exhaustive
                if corners is None:
                    print('  no full-res detection -- not saved')
                else:
                    name = f'frame_{len(saved):03d}.png'
                    cv2.imwrite(str(out / name), bgr)
                    saved.append(name)
                    last_centroid = corners.mean(axis=0)
                    last_save_t = time.time()
                    r = min(2, int(last_centroid[1] / H * 3))
                    c = min(3, int(last_centroid[0] / W * 4))
                    cov[r, c] += 1
                    print(f'  saved {name}  ({len(saved)} total)  coverage:\n{cov}')
            elif k == ord('a'):
                auto = not auto
            elif k in (ord('q'), 27):
                break
    finally:
        cv2.destroyAllWindows()
        zed_calib_rgbd.close_zed(zed)
        meta = {'resolution': args.resolution, 'width': None, 'height': None,
                'exposure': args.exposure, 'n_images': len(saved),
                'grid_inner_corners': GRID, 'stream': 'VIEW.LEFT rectified',
                'self_calib_disabled': True}
        try:
            meta['width'], meta['height'] = W, H
        except NameError:
            pass
        (out / 'meta.json').write_text(json.dumps(meta, indent=2))
        print(f'\n{len(saved)} images + meta.json in {out}')
        if len(saved) < 40:
            print('fewer than 40 views -- consider another pass; edge/corner '
                  'coverage is what constrains distortion')


if __name__ == '__main__':
    main()
