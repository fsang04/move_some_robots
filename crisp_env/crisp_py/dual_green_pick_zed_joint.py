#!/usr/bin/env python3

"""Dual-arm green pick (ZED) wrapper that uses a singular extrinsic matrix
(all expressed w.r.t. right base) instead of the original left_base2cam and
right_base2cam extrinsic matrices. 

This version loads ONE solved extrinsic in one frame, plus the known base-to-base transform, 
and derives both camera->base maps from them:

    X       = T_cam<-left_base      (solved jointly from BOTH arms' tag data)
    X_right = X @ T_LR              (exact, by construction)

    T_camera_to_left_base  = inv(X)
    T_camera_to_right_base = inv(T_LR) @ inv(X)

What that buys:
  * both maps come from one 6-DOF solve over ~119 frames instead of two
    ~58-frame solves -- lower noise, much better rotation conditioning;
  * the two arms are CONSISTENT with T_LR exactly: a shared target converts to
    the two base frames with zero internal contradiction, so a dual-arm grasp
    cannot be pulled apart by two calibrations disagreeing.

PRODUCE THE INPUT FIRST (writes joint_extrinsic_depth_translation.npz):
    cd hand_to_eye_calibration/roahm-deformable-objects/zsy-testmycode
    pixi run -e humble python solve_joint_extrinsic.py --calib-seq-name zed_calib_003

RUN IT
    cd /home/yizhouch/move_some_robots/crisp_env/crisp_py
    pixi run -e humble python dual_green_pick_zed_joint.py --dry-run
    pixi run -e humble python dual_green_pick_zed_joint.py

FOUNDATIONSTEREO VARIANT: --fs makes the whole run FS-based, both factors of
the grasp targets (targets = live depth x extrinsic):
  * extrinsic: loads joint_extrinsic_depth_translation_fs.npz from
    results/zed_calib_fs_001 (solved from the FS free per-arm solves);
  * live depth: instead of SDK MEASURE.DEPTH, grabs the left+right rectified
    views and runs FoundationStereo on the pair (fs_depth_single.py in the
    foundation_stereo conda env, ~10 s once per pick on the A6000). The result
    is registered to VIEW.LEFT like SDK depth, and the live disparity-offset
    correction applies to it unchanged.
Produce the extrinsic first:
    pixi run -e humble python solve_joint_extrinsic.py \
        --calib-seq-name zed_calib_fs_001 --fs --disparity-offset-px 5.9
then:
    pixi run -e humble python dual_green_pick_zed_joint.py --fs \
        --disparity-offset-px 5.9 --dry-run
The stored-vs-live disparity-offset check applies unchanged; pass
--disparity-offset-px to match the solve. --from-capture replays saved SDK
depth, so --fs then only selects the extrinsic. To run the FS extrinsic with
SDK live depth instead (legitimate: both depth sources share the rectified
frame and the offset), pass --joint-npz <the fs npz> WITHOUT --fs.
"""

import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

import dual_green_pick_zed as base

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_JOINT_NPZ = (REPO_ROOT / "hand_to_eye_calibration/roahm-deformable-objects"
                     / "zsy-testmycode/results/zed_calib_003"
                     / "joint_extrinsic_depth_translation.npz")
DEFAULT_FS_JOINT_NPZ = (REPO_ROOT / "hand_to_eye_calibration/roahm-deformable-objects"
                        / "zsy-testmycode/results/zed_calib_fs_001"
                        / "joint_extrinsic_depth_translation_fs.npz")
FS_ENV_PYTHON = Path.home() / "miniforge3/envs/foundation_stereo/bin/python"
FS_SINGLE_SCRIPT = (REPO_ROOT / "hand_to_eye_calibration/roahm-deformable-objects"
                    / "zsy-testmycode/fs_depth_single.py")


def make_fs_get_rgbd(offset_px):
    """Replacement for base.get_rgbd(): live FoundationStereo depth.

    Grabs ONE frame's left + right rectified views, runs FoundationStereo on
    the pair via fs_depth_single.py in the foundation_stereo conda env (torch
    and rclpy share no env, so it must be a subprocess; ~10 s on the A6000,
    model load included), and returns (color_bgr, depth_mm) in exactly the
    base script's format: uint16 mm, 0 = invalid, registered to VIEW.LEFT --
    the same frame as SDK depth, so nothing downstream changes.

    A single pair replaces the SDK path's n-frame median: the median fights
    SDK stereo speckle, which FS's global matching does not exhibit (it is
    dense even on the black backdrop where the SDK marks invalid).

    fs_depth_single.py writes RAW depth; the live disparity-offset correction
    is applied HERE, exactly as capture_rgbd_native() does for SDK depth. The
    offset lives in the rectified images both matchers consume, so skipping it
    would shift every grasp target along the viewing ray.
    """
    def _get_rgbd_fs(zed, runtime):
        sl = base.sl
        for _ in range(3):
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                break
        else:
            raise RuntimeError("zed.grab() failed 3x while capturing the FS pair")
        img_l, img_r = sl.Mat(), sl.Mat()
        if zed.retrieve_image(img_l, sl.VIEW.LEFT) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("retrieve_image(VIEW.LEFT) failed")
        if zed.retrieve_image(img_r, sl.VIEW.RIGHT) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("retrieve_image(VIEW.RIGHT) failed")
        color_bgr = cv2.cvtColor(img_l.get_data(deep_copy=True), cv2.COLOR_BGRA2BGR)
        right_bgr = cv2.cvtColor(img_r.get_data(deep_copy=True), cv2.COLOR_BGRA2BGR)

        # fx/baseline from the open camera, exactly like capture_rgbd_native()'s
        # correction path -- not from a file that could describe another mode.
        conf = zed.get_camera_information().camera_configuration
        fx = float(conf.calibration_parameters.left_cam.fx)
        baseline_m = float(conf.calibration_parameters.get_camera_baseline())

        dbg = base.DEBUG_DIR
        left_png, right_png = dbg / "fs_left.png", dbg / "fs_right.png"
        out_npy = dbg / "fs_depth_mm_raw.npy"
        cv2.imwrite(str(left_png), color_bgr)
        cv2.imwrite(str(right_png), right_bgr)
        cmd = [str(FS_ENV_PYTHON), str(FS_SINGLE_SCRIPT),
               "--left", str(left_png), "--right", str(right_png),
               "--out", str(out_npy),
               "--fx", f"{fx:.6f}", "--baseline", f"{baseline_m:.6f}"]
        print(f"[INFO] FoundationStereo live depth ({FS_SINGLE_SCRIPT.name}, "
              "foundation_stereo env)...")
        t0 = time.time()
        subprocess.run(cmd, check=True, timeout=600)
        print(f"[INFO] FoundationStereo pair done in {time.time() - t0:.1f} s")

        depth_m = np.load(out_npy).astype(np.float32) / 1000.0
        if offset_px:
            valid = depth_m > 0.0
            mean_before = float(depth_m[valid].mean()) if valid.any() else 0.0
            depth_m = base.zc.correct_depth_disparity_offset(
                depth_m, fx, baseline_m, offset_px)
            valid = depth_m > 0.0
            mean_after = float(depth_m[valid].mean()) if valid.any() else 0.0
            print(f"[depth-fix] disparity offset {offset_px:+.2f} px applied to FS "
                  f"depth (fx={fx:.4f}, B={baseline_m:.6f} m): mean valid depth "
                  f"{mean_before:.4f} -> {mean_after:.4f} m")
        depth_mm = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
        depth_mm[depth_m <= 0.0] = 0

        valid = depth_mm > 0
        print(f"[capture-fs] {color_bgr.shape[1]}x{color_bgr.shape[0]}  "
              f"valid_depth={100.0 * valid.mean():.1f}%")
        cv2.imwrite(str(dbg / "raw_color.png"), color_bgr)  # parity with get_rgbd()
        return color_bgr, depth_mm

    return _get_rgbd_fs


def load_joint_transforms(path):
    """joint_extrinsic_*.npz -> {'left': T_cam2base, 'right': T_cam2base}, info.

    Refuses a depth-correction mismatch outright: the joint extrinsic was solved
    from depth corrected by the stored offset, and this script feeds live depth
    corrected by zed_camera's current value. If they differ, every grasp target
    silently shifts along the viewing ray -- the same failure mode the base
    script guards against for captures.
    """
    path = Path(path)
    if not path.is_file():
        seq = path.parent.name
        fs_flag = " --fs" if path.stem.endswith("_fs") else ""
        raise RuntimeError(
            f"{path} is missing. Solve it first:\n"
            "  cd hand_to_eye_calibration/roahm-deformable-objects/zsy-testmycode\n"
            f"  pixi run -e humble python solve_joint_extrinsic.py "
            f"--calib-seq-name {seq}{fs_flag}")

    d = np.load(path)
    X_left = np.asarray(d["X_cam_left_base"], dtype=np.float64)
    T_LR = np.asarray(d["T_LR"], dtype=np.float64)
    # Recompose rather than trusting a stored X_cam_right_base, so the two maps
    # are consistent with T_LR even if the file was hand-edited.
    X_right = X_left @ T_LR

    stored_off = float(d["disparity_offset_px"]) if "disparity_offset_px" in d else None
    live_off = float(base.zc.DEPTH_DISPARITY_OFFSET_PX or 0.0)
    if stored_off is None:
        print("[WARN] joint npz records no disparity_offset_px; cannot verify the "
              f"depth correction matches the live {live_off:+.2f} px.")
    elif abs(stored_off - live_off) > 1e-6:
        raise RuntimeError(
            f"depth correction mismatch: the joint extrinsic was solved for "
            f"{stored_off:+.2f} px but the live capture applies {live_off:+.2f} px. "
            "Re-run solve_joint_extrinsic.py or align zed_depth_correction.json.")
    else:
        print(f"[INFO] depth correction matches: {live_off:+.2f} px "
              "(solve and live capture)")

    mode = str(d["t_lr_mode"]) if "t_lr_mode" in d else "?"
    print(f"[INFO] joint extrinsic: {path}")
    print(f"[INFO] T_LR mode: {mode}   right base in left frame: "
          f"{np.round(T_LR[:3, 3], 4)} m")
    if "trans_mean_mm_all" in d:
        print(f"[INFO] joint solve residual: {float(d['trans_mean_mm_all']):.2f} mm "
              f"mean over both arms")

    return ({"left": np.linalg.inv(X_left), "right": np.linalg.inv(X_right)},
            {"T_LR": T_LR, "path": str(path)})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint-npz", default=None,
                    help="joint_extrinsic_*.npz from solve_joint_extrinsic.py "
                         f"(default: {DEFAULT_JOINT_NPZ}, or the FS one with --fs)")
    ap.add_argument("--fs", action="store_true",
                    help="all-FoundationStereo run: load the FS joint extrinsic "
                         f"({DEFAULT_FS_JOINT_NPZ.name} in results/zed_calib_fs_001, "
                         "from solve_joint_extrinsic.py --fs) AND replace the live "
                         "SDK depth with FoundationStereo run on the left+right "
                         "rectified pair (~10 s, foundation_stereo env). An explicit "
                         "--joint-npz overrides the extrinsic half; --from-capture "
                         "disables the live-depth half.")
    ap.add_argument("--from-capture", metavar="RUN_DIR", default=None,
                    help="same as dual_green_pick_zed.py: use a saved capture with "
                         "a SAM mask instead of opening the camera")
    ap.add_argument("--disparity-offset-px", type=float, default=None,
                    help="override the live depth disparity correction (default: the "
                         "value in zed_capture/zed_depth_correction.json, currently "
                         f"{float(base.zc.DEPTH_DISPARITY_OFFSET_PX or 0.0):+.2f} px). "
                         "Must equal the offset stamped in --joint-npz -- use this to "
                         "run a joint extrinsic solved at a refit offset without "
                         "editing the json.")
    ap.add_argument("--dry-run", action="store_true",
                    help="detect and print the grasp targets, then exit; nothing moves")
    ap.add_argument("--test-run", action="store_true",
                    help="like a normal run (arms MOVE, video is recorded), but the "
                         "descend stops 10 cm above the grasp points and the grippers "
                         "never close. Ignored if --dry-run is given.")
    args = ap.parse_args()

    if args.joint_npz is None:
        args.joint_npz = str(DEFAULT_FS_JOINT_NPZ if args.fs else DEFAULT_JOINT_NPZ)
    elif args.fs:
        print(f"[WARN] --fs keeps FS live depth, but the extrinsic comes from the "
              f"explicitly given --joint-npz ({args.joint_npz})")

    if args.disparity_offset_px is not None:
        # capture_rgbd_native() bound its default offset at definition time, so
        # updating the module constant alone would satisfy every consistency
        # check below while the frames kept the OLD correction. Override both
        # the checked constant and the value actually applied to depth.
        json_off = float(base.zc.DEPTH_DISPARITY_OFFSET_PX or 0.0)
        base.zc.DEPTH_DISPARITY_OFFSET_PX = args.disparity_offset_px
        _capture = base.zc.capture_rgbd_native

        def _capture_with_override(*a, **kw):
            kw.setdefault("disparity_offset_px", args.disparity_offset_px)
            return _capture(*a, **kw)

        base.zc.capture_rgbd_native = _capture_with_override
        print(f"[INFO] live disparity offset OVERRIDDEN to "
              f"{args.disparity_offset_px:+.2f} px (json has {json_off:+.2f} px)")

    # --fs also swaps the LIVE depth source: instead of SDK depth, grab the
    # left+right rectified pair and let FoundationStereo match it. Installed
    # after the offset override above so it corrects with the final value.
    if args.fs and args.from_capture is None:
        if not FS_ENV_PYTHON.is_file():
            raise SystemExit(f"{FS_ENV_PYTHON} is missing -- the foundation_stereo "
                             "conda env is required for --fs live depth.")
        if not FS_SINGLE_SCRIPT.is_file():
            raise SystemExit(f"{FS_SINGLE_SCRIPT} is missing.")
        live_off = float(base.zc.DEPTH_DISPARITY_OFFSET_PX or 0.0)
        base.get_rgbd = make_fs_get_rgbd(live_off)
        print("[INFO] live depth source: FoundationStereo (left+right views, "
              f"corrected at {live_off:+.2f} px)")
    elif args.fs:
        print("[WARN] --from-capture replays saved SDK depth; --fs only selects "
              "the extrinsic for this run (no live FS depth).")

    transforms, info = load_joint_transforms(args.joint_npz)

    # File the grasp videos with the calibration actually in use. The joint npz
    # lives in results/<seq>/ (or captured_calibration_data/<seq>/), so its
    # parent directory names the sequence; without this, the base script's
    # import-time constants would keep saving every run under zed_calib_003.
    seq = Path(args.joint_npz).resolve().parent.name
    if (base.REPO_ROOT / "hand_to_eye_calibration/roahm-deformable-objects"
            / "captured_calibration_data" / seq).is_dir():
        base.set_calib_sequence(seq)
    else:
        print(f"[WARN] cannot infer a calibration sequence from {args.joint_npz} "
              f"('{seq}' is not in captured_calibration_data); grasp videos will "
              f"go to the default {base.PICK_RUN_DIR}")

    # main() in the base script calls load_camera_to_base_transform(path, name)
    # twice; serve the derived maps instead of loading two independent files.
    # name is 'left' / 'right'.
    def _serve_joint_transform(path, name):
        T = transforms[name]
        print(f"\n[INFO] {name} T_camera_to_base (derived from the joint extrinsic, "
              f"{info['path']}):")
        print(T)
        return T

    base.load_camera_to_base_transform = _serve_joint_transform
    base.LEFT_TRANSFORM_PATH = info["path"]      # cosmetic: what main() prints
    base.RIGHT_TRANSFORM_PATH = info["path"]

    base.main(dry_run=args.dry_run, from_capture=args.from_capture,
              test_run=args.test_run)


if __name__ == "__main__":
    main()
