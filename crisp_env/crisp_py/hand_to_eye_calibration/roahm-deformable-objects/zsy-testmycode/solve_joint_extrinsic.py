#!/usr/bin/env python3
"""Joint two-arm hand-eye solve with a known base-to-base transform.

1. Solves for a transform from the right to left base T_LR using the same 
logic inv(T_left_base2cam) @ T_right_base2cam in make_transform_ee_cam_world.py:
    X       = T_cam<-left_base            (the single 6-DOF unknown)
    X_right = X @ T_LR                    (derived, exactly consistent)

2. Transforms all right-arm tag pairs into the left base frame with T_LR
3. Runs optimization using calculate_base_to_cam_nonlinear_opt.py logic, but instead
over one base frame, and outputs a singular extrinsic matrix instead of two. 

Options to pass in T_LR (--t-lr):
    solved  (default)  inv(X_L_free) @ X_R_free from the free per-arm solves in
                       results/<seq>/.
    ideal              t = [--baseline-x, 0, 0] with the solved rotation.
                       Uses the solved x-translation and rotation, but assumes perfect
                       y and z alignment ground truth.
    file               a 4x4 'T_LR' (or 'arr_0') from --t-lr-npz, e.g. from a
                       future touch-test measurement.

Translation mode: depth by default (tag rotation + depth-at-centre translation,
like the per-arm solver WITH --use-depth-translation); pass
--no-depth-translation to use the AprilTag pose translation instead -- no depth
stack, no disparity offset. Outputs are suffixed by the mode so the two can
never be confused.

Depth source: ZED SDK depth by default; --fs switches to FoundationStereo depth
(the {side}_calibration_rgbd_fs.npz stacks written by fs_depth_batch.py).
--t-lr solved then reads the per-arm *_nonlinear_opt_depth_translation_fs.npz
free solves, and every output is suffixed _fs, so an SDK-depth and an FS-depth
joint extrinsic can never be confused either:

    pixi run -e humble python solve_joint_extrinsic.py \
        --calib-seq-name zed_calib_fs_001 --fs --disparity-offset-px 5.9

Outputs (results/<seq>/, plus the calib dir with --save-to-calib-dir),
<mode> = depth_translation | depth_translation_fs | apriltag_translation:
    joint_extrinsic_<mode>.npz
        X_cam_left_base, X_cam_right_base, T_LR, t_lr_mode, translation_mode,
        disparity_offset_px (depth mode only), per-side residual stats
        -- the one-file form dual_green_pick_zed_joint.py consumes.
    base2cam_transform_{left,right}_joint_<mode>.npz
        solver-schema files (X_cam_base + positional arr_0 + provenance), so
        make_transform_ee_cam_world.py and every existing consumer work
        unchanged by just pointing at them.
    summary_joint_<mode>.txt

    pixi run -e humble python solve_joint_extrinsic.py --calib-seq-name zed_calib_003
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


def build_side(base, side, max_images, offset_px=None, scale=None,
               use_depth_translation=True, rgbd_file=None):
    if use_depth_translation:
        # Depth intrinsics + disparity correction are only needed when depth
        # becomes the translation; the apriltag mode never touches the stack.
        # The disparity correction applies to FoundationStereo depth (rgbd_file
        # *_fs.npz) exactly as to SDK depth: it is baked into the rectified
        # images both matchers consume.
        dp = _load_depth_stack(base, side, rgbd_file)
        dh, dw = int(dp.shape[1]), int(dp.shape[2])
        fx, fy, cx, cy = _camera_params_for('zed', dw, dh)
        set_depth_intrinsics(fx, fy, cx, cy)
        rgbd_path = base / (rgbd_file or f'{side}_calibration_rgbd.npz')
        set_depth_corrector(zed_depth_config.corrector_for(
            'zed', fx, unit='mm', offset_px_override=offset_px,
            scale_override=scale,
            already_applied_px=zed_depth_config.dataset_applied_offset_px(rgbd_path),
            already_applied_scale=zed_depth_config.dataset_applied_scale(rgbd_path)))
    Tc, Tb, vi, _, _ = _build_calibration_pairs(
        max_images=max_images, image_dir=base / 'frames',
        pose_file=base / f'{side}_calibration_poses.npz',
        calib_base_dir=base, side=side, camera='zed',
        use_depth_translation=use_depth_translation, enhance=False,
        exclude_image_indices=set(), rgbd_file=rgbd_file)
    return Tc, Tb, vi


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def stat_block(rows, mask=None):
    t = np.array([r['trans_err_mm'] for r in rows])
    r = np.array([r['rot_err_deg'] for r in rows])
    if mask is not None:
        t, r = t[mask], r[mask]
    return dict(n=len(t), trans_mean=float(t.mean()), trans_med=float(np.median(t)),
                trans_max=float(t.max()), rot_mean=float(r.mean()),
                rot_max=float(r.max()))


def fmt(s):
    return (f"trans {s['trans_mean']:5.2f} / med {s['trans_med']:5.2f} / "
            f"max {s['trans_max']:5.2f} mm   rot {s['rot_mean']:5.3f} / "
            f"max {s['rot_max']:5.3f} deg   (n={s['n']})")


def resolve_t_lr(mode, res_dir, baseline_x, t_lr_npz, mode_name='depth_translation',
                 applied_offset_px=None, applied_scale=None):
    """-> (T_LR 4x4, description string). T_LR maps right-base coords to
    left-base coords (the right base's pose expressed in the left base frame).

    The free per-arm solves are read for the SAME translation mode as this joint
    run: a depth-mode T_LR carries the depth biases of the solves it came from,
    so mixing modes would fold one mode's bias into the other's fit. For the
    same reason, applied_offset_px (the disparity offset THIS run corrects depth
    with) is compared against the offset stamped in each free solve; a mismatch
    means T_LR carries a different depth bias than the pairs being fit.
    """
    XL = XR = None
    free_l = res_dir / f'base2cam_transform_left_nonlinear_opt_{mode_name}.npz'
    free_r = res_dir / f'base2cam_transform_right_nonlinear_opt_{mode_name}.npz'
    if free_l.exists() and free_r.exists():
        loaded = [np.load(f) for f in (free_l, free_r)]
        XL, XR = (d['X_cam_base'] for d in loaded)
        if applied_offset_px is not None:
            for f, d in zip((free_l, free_r), loaded):
                if 'disparity_offset_px' not in d:
                    continue
                stamped = float(d['disparity_offset_px'])
                stamped_a = (float(d['disparity_scale'])
                             if 'disparity_scale' in d else 1.0)
                want_a = 1.0 if applied_scale is None else float(applied_scale)
                if (abs(stamped - applied_offset_px) > 1e-6
                        or abs(stamped_a - want_a) > 1e-9):
                    print(f'[WARN] {f.name} was solved at a={stamped_a:.4f}, '
                          f'd={stamped:+.2f} px but this joint run corrects depth '
                          f'at a={want_a:.4f}, d={applied_offset_px:+.2f} px '
                          '-- its bias differs from the pairs being fit.')

    if mode == 'file':
        if not t_lr_npz:
            raise SystemExit('--t-lr file needs --t-lr-npz')
        d = np.load(t_lr_npz)
        T = np.asarray(d['T_LR'] if 'T_LR' in d else d['arr_0'], dtype=np.float64)
        return T, f'file: {t_lr_npz}', XL, XR

    if XL is None:
        if mode_name == 'apriltag_translation':
            flag = 'WITHOUT --use-depth-translation'
        elif mode_name.endswith('_fs'):
            flag = ('--use-depth-translation --rgbd-file '
                    '<side>_calibration_rgbd_fs.npz')
        else:
            flag = '--use-depth-translation'
        raise SystemExit(f'--t-lr {mode} needs {free_l.name} and {free_r.name} in '
                         f'{res_dir} (run calculate_base_to_cam_nonlinear_opt.py '
                         f'for both sides {flag} first)')
    T_solved = np.linalg.inv(XL) @ XR
    if mode == 'solved':
        return T_solved, 'solved: inv(X_L_free) @ X_R_free', XL, XR
    # ideal: keep the solved rotation, idealize the translation
    T = T_solved.copy()
    T[:3, 3] = [baseline_x, 0.0, 0.0]
    return T, f'ideal: t=[{baseline_x}, 0, 0], solved rotation', XL, XR


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--calib-seq-name', default='zed_calib_003')
    p.add_argument('--t-lr', default='solved', choices=['solved', 'ideal', 'file'])
    p.add_argument('--t-lr-npz', default=None,
                   help="--t-lr file: npz holding a 4x4 under 'T_LR' (or 'arr_0')")
    p.add_argument('--baseline-x', type=float, default=1.2748)
    p.add_argument('--rot-weight', type=float, default=573.0)
    p.add_argument('--trans-weight', type=float, default=1000.0)
    p.add_argument('--d', '--disparity-offset-px', dest='disparity_offset_px',
                   type=float, default=None,
                   help='ZED disparity offset d, in pixels, for BOTH arms (model: '
                        'disp_true = a*disp + d). Default: '
                        'zed_capture/zed_depth_correction.json '
                        f'(currently {zed_depth_config.offset_px():.2f} px). Must match '
                        'the correction used for the free per-arm solves that --t-lr '
                        'solved reads, or T_LR and the pairs disagree. Ignored with '
                        '--no-depth-translation.')
    p.add_argument('--a', '--disparity-scale', dest='disparity_scale',
                   type=float, default=None,
                   help='ZED disparity scale a, dimensionless, for BOTH arms. '
                        'Default: zed_capture/zed_depth_correction.json '
                        f'(currently {zed_depth_config.scale():.4f}).')
    p.add_argument('--no-depth-translation', action='store_true',
                   help='use the AprilTag pose translation instead of depth for '
                        'T_cam_tag (the joint counterpart of the per-arm solver run '
                        'WITHOUT --use-depth-translation). No depth stack and no '
                        'disparity offset are involved. --t-lr solved then reads the '
                        'per-arm *_apriltag_translation solves, and every output is '
                        'named *_apriltag_translation.* so it can never be mistaken '
                        'for a depth-mode extrinsic.')
    p.add_argument('--fs', action='store_true',
                   help='use FoundationStereo depth instead of ZED SDK depth: the '
                        'pairs are built from {side}_calibration_rgbd_fs.npz (written '
                        'by fs_depth_batch.py), --t-lr solved reads the per-arm '
                        'base2cam_transform_{side}_nonlinear_opt_depth_translation_fs.npz '
                        'free solves, and every output is suffixed _fs. The disparity '
                        'offset correction applies unchanged (it lives in the '
                        'rectified images, not the matcher). Incompatible with '
                        '--no-depth-translation.')
    p.add_argument('--max-images', type=int, default=200)
    p.add_argument('--save-to-calib-dir', action='store_true',
                   help='also write the outputs into captured_calibration_data/<seq>')
    args = p.parse_args()

    base = Path(DATAPATH) / 'captured_calibration_data' / args.calib_seq_name
    res_dir = Path(__file__).resolve().parent / 'results' / args.calib_seq_name
    res_dir.mkdir(parents=True, exist_ok=True)

    use_depth = not args.no_depth_translation
    if args.fs and not use_depth:
        raise SystemExit('--fs selects the FoundationStereo DEPTH source; it cannot '
                         'be combined with --no-depth-translation.')
    mode_name = 'depth_translation' if use_depth else 'apriltag_translation'
    if args.fs:
        mode_name += '_fs'
    rgbd_files = {s: (f'{s}_calibration_rgbd_fs.npz' if args.fs else None)
                  for s in ('left', 'right')}
    print(f'[JOINT] translation mode: {mode_name}')
    if not use_depth and args.disparity_offset_px is not None:
        print('[WARN] --disparity-offset-px is ignored with --no-depth-translation '
              '(this solve never touches depth).')

    applied_off = applied_scale = None
    if use_depth:
        applied_off = (args.disparity_offset_px
                       if args.disparity_offset_px is not None
                       else zed_depth_config.offset_px())
        applied_scale = (args.disparity_scale
                         if args.disparity_scale is not None
                         else zed_depth_config.scale())
    T_LR, t_lr_desc, XL_free, XR_free = resolve_t_lr(
        args.t_lr, res_dir, args.baseline_x, args.t_lr_npz, mode_name=mode_name,
        applied_offset_px=applied_off, applied_scale=applied_scale)
    print(f'[JOINT] T_LR ({t_lr_desc}):')
    print(np.round(T_LR, 6))

    Tc_L, Tb_L, vi_L = build_side(base, 'left', args.max_images,
                                  offset_px=args.disparity_offset_px,
                                  scale=args.disparity_scale,
                                  use_depth_translation=use_depth,
                                  rgbd_file=rgbd_files['left'])
    Tc_R, Tb_R, vi_R = build_side(base, 'right', args.max_images,
                                  offset_px=args.disparity_offset_px,
                                  scale=args.disparity_scale,
                                  use_depth_translation=use_depth,
                                  rgbd_file=rgbd_files['right'])
    print(f'[JOINT] pairs: {len(Tb_L)} left + {len(Tb_R)} right = {len(Tb_L) + len(Tb_R)}')

    # transfer the right arm's robot-side poses into the LEFT base frame
    Tb_R_in_L = [T_LR @ T for T in Tb_R]
    Tc_all = list(Tc_L) + list(Tc_R)
    Tb_all = list(Tb_L) + Tb_R_in_L
    vi_all = list(vi_L) + [1000 + i for i in vi_R]      # sides stay tellable

    X0, _ = _initial_guess_from_per_frame(Tc_all, Tb_all)
    X, _ = _optimize_X(X0, Tc_all, Tb_all,
                       rot_weight=args.rot_weight, trans_weight=args.trans_weight)
    X_right = X @ T_LR

    rows = _pose_error_rows(X, Tc_all, Tb_all, vi_all)
    is_left = np.array([r['image_index'] < 1000 for r in rows])
    s_all, s_l, s_r = stat_block(rows), stat_block(rows, is_left), stat_block(rows, ~is_left)

    print('\n[JOINT] residuals (solver metric):')
    print(f'  ALL   : {fmt(s_all)}')
    print(f'  left  : {fmt(s_l)}')
    print(f'  right : {fmt(s_r)}')
    if XL_free is not None:
        print(f'  X_left  vs free solve: {np.linalg.norm(X[:3,3]-XL_free[:3,3])*1000:5.1f} mm, '
              f'{rot_angle_deg(X[:3,:3], XL_free[:3,:3]):.3f} deg')
        print(f'  X_right vs free solve: {np.linalg.norm(X_right[:3,3]-XR_free[:3,3])*1000:5.1f} mm, '
              f'{rot_angle_deg(X_right[:3,:3], XR_free[:3,:3]):.3f} deg')
        # A joint solve that fits far worse than the free ones means T_LR is
        # fighting the data (this is exactly how the idealized-baseline
        # hypothesis was rejected). Warn -- deploying it would bake the bias in.
        free_mean = 0.5 * (4.68 + 7.72)  # nominal; recomputed properly below
        rows_free_l = _pose_error_rows(XL_free, Tc_L, Tb_L, vi_L)
        rows_free_r = _pose_error_rows(XR_free, Tc_R, Tb_R, vi_R)
        free_mean = 0.5 * (stat_block(rows_free_l)['trans_mean']
                           + stat_block(rows_free_r)['trans_mean'])
        if s_all['trans_mean'] > 1.5 * free_mean:
            print(f'\n[WARNING] joint residual ({s_all["trans_mean"]:.2f} mm) is >1.5x the '
                  f'free solves ({free_mean:.2f} mm): this T_LR conflicts with the data. '
                  'Deploying it bakes that conflict into both arms.')

    print('\n[JOINT] X_cam_left_base:')
    print(X)
    print('[JOINT] X_cam_right_base = X @ T_LR:')
    print(X_right)

    # Stamp the offset the pairs were ACTUALLY corrected with, so the consumers
    # of these npz files pair their live depth with the right value. In apriltag
    # mode no depth exists, so the key is OMITTED entirely: the pick script
    # treats an absent key as "cannot verify" (a warning), whereas a fabricated
    # 0.0 would make it hard-refuse every live offset.
    provenance = {'translation_mode': np.str_(mode_name)}
    if use_depth:
        provenance['disparity_offset_px'] = np.float64(applied_off)
        provenance['disparity_scale'] = np.float64(applied_scale)
    out_dirs = [res_dir] + ([base] if args.save_to_calib_dir else [])
    for out in out_dirs:
        # the one-file form for joint consumers
        np.savez(out / f'joint_extrinsic_{mode_name}.npz',
                 X_cam_left_base=X, X_cam_right_base=X_right, T_LR=T_LR,
                 t_lr_mode=np.str_(args.t_lr), t_lr_desc=np.str_(t_lr_desc),
                 trans_mean_mm_all=np.float64(s_all['trans_mean']),
                 trans_mean_mm_left=np.float64(s_l['trans_mean']),
                 trans_mean_mm_right=np.float64(s_r['trans_mean']),
                 rot_mean_deg_all=np.float64(s_all['rot_mean']),
                 **provenance)
        # solver-schema per-side files (arr_0 positional for the old readers),
        # named *_joint_* so they can never be mistaken for the free solves
        for side, Xs in (('left', X), ('right', X_right)):
            np.savez(out / f'base2cam_transform_{side}_joint_{mode_name}.npz',
                     Xs, X_cam_base=Xs, X_base_cam=np.linalg.inv(Xs),
                     T_LR=T_LR, **provenance)
        print(f'[SAVED] {out}/joint_extrinsic_{mode_name}.npz (+ per-side files)')

    with open(res_dir / f'summary_joint_{mode_name}.txt', 'w') as f:
        f.write(f'Joint two-arm solve, seq {args.calib_seq_name}\n')
        f.write(f'translation_mode: {mode_name}\n')
        f.write(f'T_LR ({t_lr_desc}):\n{T_LR}\n\n')
        f.write(f'residuals ALL  : {fmt(s_all)}\n')
        f.write(f'residuals left : {fmt(s_l)}\n')
        f.write(f'residuals right: {fmt(s_r)}\n\n')
        f.write(f'X_cam_left_base:\n{X}\n\nX_cam_right_base = X @ T_LR:\n{X_right}\n')
        if use_depth:
            f.write(f'\ndisparity_offset_px: '
                    f'{float(provenance["disparity_offset_px"])}\n')
            f.write(f'disparity_scale: '
                    f'{float(provenance["disparity_scale"])}\n')
    print(f'[SAVED] {res_dir}/summary_joint_{mode_name}.txt')


if __name__ == '__main__':
    main()
