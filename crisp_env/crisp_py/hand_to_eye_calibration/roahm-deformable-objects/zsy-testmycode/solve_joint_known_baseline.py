#!/usr/bin/env python3
"""Joint two-arm hand-eye solve with a KNOWN base-to-base transform.

Tests three different base-to-base transforms:
A: idealized (assume y and z alignment to be 0, use solved rotation)
B: solved (use solved T_LR)
C: fully ideal (assume y and z alignment to be 0, assume rotation to be 180)

Idea (see also the free per-arm solver this wraps): if T_LR -- the right base
expressed in the left base frame -- is known, the two extrinsics stop being
independent:

    X_left = T_cam<-left_base          (the single 6-DOF unknown)
    X_right = X_left @ T_LR

Every right-arm tag pair is transferred into the LEFT base frame
(T_base_tag^Lframe = T_LR @ T_base_tag^R), the two arms' pairs are pooled,
and calculate_base_to_cam_nonlinear_opt's own init + optimizer run ONCE on
the combined set. Twice the data, half the unknowns, and a ~1.8 m point
cloud instead of ~0.5 m -- which is what pins the camera ROTATION.

The catch is GIGO: T_LR bias is forced into the solution. So this script
solves under several T_LR hypotheses and prints per-arm residual breakdowns;
the hypothesis under which BOTH arms fit well without one side systematically
sagging is the one the data believes.

    pixi run -e humble python solve_joint_known_baseline.py --calib-seq-name zed_calib_003
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from calculate_base_to_cam_nonlinear_opt import (
    DATAPATH,
    _build_calibration_pairs,
    _initial_guess_from_per_frame,
    _load_depth_stack,
    _optimize_X,
    _pose_error_rows,
    set_depth_corrector,
    set_depth_intrinsics,
)
import zed_depth_config
from apriltag_image import _camera_params_for


def build_side(base, side):
    dp = _load_depth_stack(base, side)
    dh, dw = int(dp.shape[1]), int(dp.shape[2])
    fx, fy, cx, cy = _camera_params_for('zed', dw, dh)
    set_depth_intrinsics(fx, fy, cx, cy)
    set_depth_corrector(zed_depth_config.corrector_for(
        'zed', fx, unit='mm', offset_px_override=None,
        already_applied_px=zed_depth_config.dataset_applied_offset_px(
            base / f'{side}_calibration_rgbd.npz')))
    Tc, Tb, vi, _, _ = _build_calibration_pairs(
        max_images=200, image_dir=base / 'frames',
        pose_file=base / f'{side}_calibration_poses.npz',
        calib_base_dir=base, side=side, camera='zed',
        use_depth_translation=True, enhance=False, exclude_image_indices=set())
    return Tc, Tb, vi


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def stats(rows, mask=None):
    t = np.array([r['trans_err_mm'] for r in rows])
    r = np.array([r['rot_err_deg'] for r in rows])
    if mask is not None:
        t, r = t[mask], r[mask]
    return (f'trans {t.mean():5.2f} / med {np.median(t):5.2f} / max {t.max():5.2f} mm   '
            f'rot {r.mean():5.3f} / max {r.max():5.3f} deg   (n={len(t)})')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--calib-seq-name', default='zed_calib_003')
    p.add_argument('--baseline-x', type=float, default=1.2748,
                   help='idealized T_LR x (m); y=z=0 by assumption')
    p.add_argument('--rot-weight', type=float, default=573.0)
    args = p.parse_args()

    base = Path(DATAPATH) / 'captured_calibration_data' / args.calib_seq_name
    RES = Path(__file__).resolve().parent / 'results' / args.calib_seq_name

    # reference: the free per-arm solves (post tag fix)
    XL = np.load(RES / 'base2cam_transform_left_nonlinear_opt_depth_translation.npz')['X_cam_base']
    XR = np.load(RES / 'base2cam_transform_right_nonlinear_opt_depth_translation.npz')['X_cam_base']
    T_LR_solved = np.linalg.inv(XL) @ XR

    Tc_L, Tb_L, vi_L = build_side(base, 'left')
    Tc_R, Tb_R, vi_R = build_side(base, 'right')
    nL, nR = len(Tb_L), len(Tb_R)

    # T_LR hypotheses
    Rz180 = np.diag([-1.0, -1.0, 1.0, 1.0])
    A = T_LR_solved.copy(); A[:3, 3] = [args.baseline_x, 0.0, 0.0]
    C = Rz180.copy();       C[:3, 3] = [args.baseline_x, 0.0, 0.0]
    variants = {
        'A idealized t=[x,0,0], solved rotation': A,
        'B fully as-solved (inv(XL)@XR)        ': T_LR_solved,
        'C fully idealized: t=[x,0,0], yaw=180 ': C,
    }

    lines = [f'combined pairs: {nL} left + {nR} right = {nL + nR}',
             f'T_LR solved: t={np.round(T_LR_solved[:3,3],4)} m, '
             f'yaw={np.degrees(np.arctan2(T_LR_solved[1,0], T_LR_solved[0,0])):+.2f} deg', '']

    for name, T_LR in variants.items():
        # transfer right pairs into the LEFT base frame
        Tb_R_in_L = [T_LR @ T for T in Tb_R]
        Tc_all = list(Tc_L) + list(Tc_R)
        Tb_all = list(Tb_L) + Tb_R_in_L
        vi_all = list(vi_L) + [1000 + i for i in vi_R]   # keep sides tellable

        X0, _ = _initial_guess_from_per_frame(Tc_all, Tb_all)
        X, _ = _optimize_X(X0, Tc_all, Tb_all, rot_weight=args.rot_weight,
                           trans_weight=1000.0)
        rows = _pose_error_rows(X, Tc_all, Tb_all, vi_all)
        is_left = np.array([r['image_index'] < 1000 for r in rows])

        X_right_implied = X @ T_LR
        lines += [
            f'--- {name} ---',
            f'  ALL   : {stats(rows)}',
            f'  left  : {stats(rows, is_left)}',
            f'  right : {stats(rows, ~is_left)}',
            f'  X_left  vs free solve: {np.linalg.norm(X[:3,3]-XL[:3,3])*1000:5.1f} mm, '
            f'{rot_angle_deg(X[:3,:3], XL[:3,:3]):.3f} deg',
            f'  X_right vs free solve: {np.linalg.norm(X_right_implied[:3,3]-XR[:3,3])*1000:5.1f} mm, '
            f'{rot_angle_deg(X_right_implied[:3,:3], XR[:3,:3]):.3f} deg',
            '']
        if name.startswith('A'):
            lines += ['  X_cam<-left_base (variant A):']
            lines += ['    [' + '  '.join(f'{v: .6f}' for v in row) + ']' for row in X]
            lines += ['']

    print('\n===JOINT===')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
