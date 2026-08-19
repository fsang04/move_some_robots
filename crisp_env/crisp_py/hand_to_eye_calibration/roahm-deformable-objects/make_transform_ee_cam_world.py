#!/usr/bin/env python3
"""Merge the two per-arm hand-eye results into one transform_ee_cam_world.npz.

calculate_base_to_cam_nonlinear_opt.py solves ONE arm at a time and writes
base2cam_transform_<side>_nonlinear_opt_<mode>.npz with the key X_cam_base
(= T_cam<-base, i.e. base2cam, meters). The tracking pipeline instead wants
both arms in a single file under the names utils/transforms.py:8 established:

    T_left_base2cam, T_right_base2cam, K   (+ the inverses, for the old drivers)

That file is what dlo_tracking_live.py --segmenter armdiff --calib takes.

Usage:
    # ZED (the current rig):
    python make_transform_ee_cam_world.py --camera zed --k-width 1280 \
        --left  captured_calibration_data/zed_calib_003/base2cam_transform_left_nonlinear_opt_depth_translation.npz \
        --right captured_calibration_data/zed_calib_003/base2cam_transform_right_nonlinear_opt_depth_translation.npz \
        --out   captured_calibration_data/zed_calib_003/transform_ee_cam_world.npz

    # Azure Kinect:
    python make_transform_ee_cam_world.py --camera azure \
        --left ... --right ... --out ...

K is per-camera AND per-resolution, so it must match the frames the live source
delivers: azure -> azure_intrinsics.py (1280x720); zed -> the exported
zed_capture/zed_intrinsics_2208x1242.npz, rescaled with --k-width (1280 for the
HD720 that ZedSource opens). --k-from-device reads it from the camera instead.
The EXTRINSICS are resolution-independent -- only K is not.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def load_base2cam(npz_path):
    """base2cam_transform_*.npz -> (4,4) T_cam<-base in meters."""
    data = np.load(npz_path)
    if 'X_cam_base' in data:
        T = data['X_cam_base']
    elif 'arr_0' in data:          # older writers only stored it positionally
        T = data['arr_0']
    else:
        raise KeyError(f"{npz_path} has neither 'X_cam_base' nor 'arr_0' "
                       f"(keys: {list(data.files)})")
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{npz_path}: expected a 4x4 transform, got {T.shape}")
    return T


def resolve_K(args):
    """The color intrinsics to store alongside the extrinsics.

    K is per-camera AND per-resolution; the extrinsics are neither. armdiff
    itself takes K from the live device, so this entry exists so the file is
    self-describing (and for the offline/replay drivers, which do read it).
    """
    if args.k_from_device:
        if args.camera == 'zed':
            import pyzed.sl as sl
            init = sl.InitParameters()
            # Must match the resolution the live source runs at -- the ZED
            # rectifies separately per resolution, so K is not transferable
            # between them (see --k-width's warning).
            init.camera_resolution = getattr(sl.RESOLUTION, args.zed_resolution)
            # Same root-free factory-calibration directory ZedSource uses;
            # without it the SDK tries to mkdir /usr/local/zed and open() fails.
            sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'zed_capture'))
            try:
                import zed_depth_config
                settings = zed_depth_config.settings_dir()
                if settings.is_dir():
                    init.optional_settings_path = str(settings)
                    print(f'[K] factory calibration from {settings}')
            except ImportError:
                pass
            zed = sl.Camera()
            status = zed.open(init)
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(
                    f'ZED open failed: {status}. Is the tracker or the Depth '
                    f'Viewer still holding the camera? A USB ZED allows one '
                    f'process at a time.')
            cam = (zed.get_camera_information()
                   .camera_configuration.calibration_parameters.left_cam)
            K = np.array([[cam.fx, 0.0, cam.cx],
                          [0.0, cam.fy, cam.cy],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
            zed.close()
            return K
        from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS
        from pyk4a.calibration import CalibrationType
        k4a = PyK4A(Config(color_resolution=ColorResolution.RES_720P,
                           depth_mode=DepthMode.NFOV_UNBINNED,
                           camera_fps=FPS.FPS_30,
                           synchronized_images_only=True))
        k4a.start()
        K = np.asarray(k4a.calibration.get_camera_matrix(CalibrationType.COLOR),
                       dtype=np.float64)
        k4a.stop()
        return K

    if args.camera == 'zed':
        k_npz = Path(args.k_npz) if args.k_npz else (
            Path(__file__).resolve().parents[2] / 'zed_capture' /
            'zed_intrinsics_2208x1242.npz')
        data = np.load(k_npz)
        K = np.asarray(data['K'], dtype=np.float64)
        if args.k_width is not None:
            s = args.k_width / float(data['width'])
            K = K * s
            K[2, 2] = 1.0
            print(f'[K] rescaled {int(data["width"])} -> {args.k_width} px wide (x{s:.6f})')
            print('[K] WARNING: the ZED rectifies separately per resolution, so its '
                  'RECTIFIED intrinsics are NOT proportional to width. SN22456 is '
                  'fx 1414.58 at 2208 px but 693.82 at 1280 px, where this scaling '
                  'predicts 820.04 -- an 18% error. Prefer --k-from-device, which '
                  'asks the camera for the K at the resolution you will run.')
        return K

    from azure_intrinsics import azure_intrinsics
    return np.asarray(azure_intrinsics, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--left', required=True, help='left arm base2cam_transform_*.npz')
    parser.add_argument('--right', required=True, help='right arm base2cam_transform_*.npz')
    parser.add_argument('--out', required=True, help='output transform_ee_cam_world.npz')
    parser.add_argument('--camera', choices=['azure', 'zed'], default='azure',
                        help='which camera this calibration belongs to; picks the '
                             'default source of K')
    parser.add_argument('--k-from-device', action='store_true',
                        help='read K from the connected camera instead of the '
                             'stored intrinsics (must match the calibration)')
    parser.add_argument('--k-npz', default=None,
                        help="zed: intrinsics npz with a 'K' key (default: "
                             "zed_capture/zed_intrinsics_2208x1242.npz)")
    parser.add_argument('--zed-resolution', default='HD720',
                        help='zed, with --k-from-device: resolution to read K at. '
                             'MUST match what the live source runs '
                             '(dlo_tracking_live.py --zed_resolution, default HD720)')
    parser.add_argument('--k-width', type=int, default=None,
                        help='zed: rescale K to this frame width (e.g. 1280 for '
                             'HD720 when the npz holds HD2K). ZED intrinsics are '
                             'per-resolution; the extrinsics are not.')
    args = parser.parse_args()

    T_left = load_base2cam(args.left)
    T_right = load_base2cam(args.right)
    K = resolve_K(args)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             T_left_base2cam=T_left,
             T_right_base2cam=T_right,
             T_cam2left=np.linalg.inv(T_left),
             T_cam2right=np.linalg.inv(T_right),
             T_left2right=np.linalg.inv(T_right) @ T_left,
             K=K.astype(np.float32))

    print('T_left_base2cam:\n', T_left)
    print('T_right_base2cam:\n', T_right)
    print('K:\n', K)
    print(f'\nWrote {out}')
    print('Use it with:  dlo_tracking_live.py --segmenter armdiff --calib '
          f'{out}')


if __name__ == '__main__':
    main()
