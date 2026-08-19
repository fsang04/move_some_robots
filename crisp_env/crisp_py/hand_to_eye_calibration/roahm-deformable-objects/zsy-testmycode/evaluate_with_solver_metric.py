#!/usr/bin/env python3
"""Score ANY extrinsic with the nonlinear solver's OWN residual metric.

The point: put the solvePnP-centres extrinsic on the same scoreboard as the
deployed solves. The metric is imported from calculate_base_to_cam_nonlinear_opt
rather than re-implemented, so the numbers are the solver's numbers:

    T_pred = X @ T_base_tag
    xi     = se3_log( inv(T_cam_tag_measured) @ T_pred )
    trans_err_mm = ||xi[3:6]|| * 1000        rot_err_deg = ||xi[0:3]|| in deg
    outlier rule: trans > 30 mm or rot > 8 deg    (TRANS_THRESH_MM / ROT_THRESH_DEG)

Every candidate X is scored against TWO versions of the measured tag pose, built
in one detection pass (frame_infos keeps both):

  ruler A 'depth'    -- rotation from AprilTag, translation from CORRECTED ZED
                        depth at the tag centre. This is the metric the published
                        residuals (left 8.14 mm, right 15.63 mm) were computed
                        with. The depth-translation solve was FITTED to exactly
                        this, so it wins ruler A by construction; what ruler A
                        measures for the PnP X is "how far is PnP from the depth
                        sensor", i.e. the size of the depth disagreement, not an
                        error of PnP per se.
  ruler B 'apriltag' -- rotation AND translation from AprilTag alone. Its range
                        comes from the 93 mm tag size + fx, so it is independent
                        of the ZED depth fault and its correction. Neither X was
                        fitted to per-frame apriltag translations (the deployed
                        depth solve used depth; PnP used centres only), so ruler
                        B is the closest thing to a neutral referee this dataset
                        contains.

Self-check: scoring the deployed depth-translation X on ruler A must reproduce
the published residual report. If it does, the harness is faithful and the PnP
rows can be read with the same trust.

    python evaluate_with_solver_metric.py --calib-seq-name zed_calib_003 --side right
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import calculate_base_to_cam_nonlinear_opt as solver
from calculate_base_to_cam_nonlinear_opt import (
    DATAPATH,
    _build_calibration_pairs,
    _load_depth_stack,
    _pose_error_rows,
    _print_report,
    set_depth_corrector,
    set_depth_intrinsics,
)
import zed_depth_config
from apriltag_image import _camera_params_for

REPO = Path(__file__).resolve().parents[3]              # .../crisp_py
INTRINSICS = REPO / 'zed_capture' / 'zed_intrinsics_2208x1242.npz'


def solve_pnp_x(base: Path, side: str) -> np.ndarray:
    """Deterministic PnP fit from the dumped centres (no RANSAC: both sides were
    100% inliers at 3 px, and determinism matters more here than robustness)."""
    from solve_pnp_centres import load_side
    intr = np.load(INTRINSICS)
    K = np.asarray(intr['K'], dtype=np.float64)
    dist = np.zeros(8)
    obj, uv, _ = load_side(base, side)
    ok, rvec, tvec = cv2.solvePnP(obj, uv, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError(f'{side}: solvePnP failed')
    rvec, tvec = cv2.solvePnPRefineLM(obj, uv, K, dist, rvec, tvec)
    X = np.eye(4)
    X[:3, :3] = cv2.Rodrigues(rvec)[0]
    X[:3, 3] = tvec.ravel()
    return X


def plain_cam_decomposition(X, T_cam_tag_list, T_base_tag_list):
    """Interpretation aid, NOT the solver metric: t_pred - t_meas in camera
    coordinates, split into lateral (xy) and along-ray (z) parts."""
    d = np.array([(X @ B)[:3, 3] - M[:3, 3]
                  for M, B in zip(T_cam_tag_list, T_base_tag_list)])
    lat = np.linalg.norm(d[:, :2], axis=1) * 1000.0
    dz = d[:, 2] * 1000.0
    return lat, dz


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--calib-seq-name', default='zed_calib_003')
    p.add_argument('--side', default='right', choices=['left', 'right'])
    p.add_argument('--max-images', type=int, default=200)
    args = p.parse_args()

    side = args.side
    base = Path(DATAPATH) / 'captured_calibration_data' / args.calib_seq_name
    frames_dir = base / 'frames'
    pose_file = base / f'{side}_calibration_poses.npz'

    # --- depth setup, byte-for-byte the solver main()'s zed branch ---
    depth_probe = _load_depth_stack(base, side)
    if depth_probe is None:
        raise SystemExit(f'no depth stack in {side}_calibration_rgbd.npz')
    dh, dw = int(depth_probe.shape[1]), int(depth_probe.shape[2])
    fx, fy, cx, cy = _camera_params_for('zed', dw, dh)
    set_depth_intrinsics(fx, fy, cx, cy)
    rgbd_path = base / f'{side}_calibration_rgbd.npz'
    set_depth_corrector(zed_depth_config.corrector_for(
        'zed', fx, unit='mm',
        offset_px_override=None,
        already_applied_px=zed_depth_config.dataset_applied_offset_px(rgbd_path)))

    # --- one detection pass; frame_infos carries the apriltag pose too ---
    (T_cam_tag_depth, T_base_tag_list, valid_indices,
     _sources, frame_infos) = _build_calibration_pairs(
        max_images=args.max_images,
        image_dir=frames_dir,
        pose_file=pose_file,
        calib_base_dir=base,
        side=side,
        camera='zed',
        use_depth_translation=True,
        enhance=False,                       # matches the published --no-enhance runs
        exclude_image_indices=set(),
    )
    T_cam_tag_april = [info['T_cam_tag_apriltag'] for info in frame_infos]

    # --- candidates ---
    candidates = {}
    for tag in ('depth_translation', 'apriltag_translation'):
        f = base / f'base2cam_transform_{side}_nonlinear_opt_{tag}.npz'
        if f.exists():
            candidates[f'deployed {tag}'] = np.asarray(
                np.load(f)['X_cam_base'], dtype=np.float64)
    candidates['solvePnP centres'] = solve_pnp_x(base, side)

    rulers = {'depth-lifted tag pose (published metric)': T_cam_tag_depth,
              'apriltag-only tag pose (depth-free ruler)': T_cam_tag_april}

    summary = []
    for rname, T_meas in rulers.items():
        for cname, X in candidates.items():
            rows = _pose_error_rows(X, T_meas, T_base_tag_list, valid_indices)
            outliers, _ = _print_report(
                f'[{side}] X = {cname}   ruler = {rname}', rows)
            trans = np.array([r['trans_err_mm'] for r in rows])
            rot = np.array([r['rot_err_deg'] for r in rows])
            lat, dz = plain_cam_decomposition(X, T_meas, T_base_tag_list)
            print(f'  camera-frame decomposition: lateral mean {lat.mean():.1f} mm, '
                  f'along-ray mean {dz.mean():+.1f} mm '
                  f'(std {dz.std():.1f})')
            summary.append((rname, cname, trans.mean(), np.median(trans),
                            trans.max(), rot.mean(), len(outliers), len(rows),
                            lat.mean(), dz.mean()))

    print('\n' + '=' * 118)
    print(f'SUMMARY  --  {side} arm, {args.calib_seq_name}, solver metric '
          f'(trans mm / rot deg), outlier rule >30 mm or >8 deg')
    print('=' * 118)
    hdr = (f'{"ruler":44s} {"candidate X":30s} {"mean":>7s} {"med":>7s} '
           f'{"max":>7s} {"rot":>6s} {"outl":>6s} {"lat":>6s} {"ray":>7s}')
    print(hdr)
    print('-' * 118)
    for rname, cname, m, md, mx, rm, no, n, lat, dz in summary:
        print(f'{rname:44s} {cname:30s} {m:7.2f} {md:7.2f} {mx:7.2f} '
              f'{rm:6.3f} {no:>3d}/{n:<3d} {lat:6.1f} {dz:+7.1f}')


if __name__ == '__main__':
    main()
