#!/usr/bin/env python3
"""
zed_calib_rgbd.py

Shared ZED RGB-D capture for the calibration scripts.

WHY THIS EXISTS
    The ZED branch of capture_poses_and_images_for_calibration*.py recorded COLOR ONLY
    (`frame_list.append({"color": frame, "depth": None})`, and the final savez stored
    only `color`). That made --use-depth-translation impossible for the ZED, which is
    why calculate_base_to_cam_nonlinear_opt.py used to hard-block it. This module
    supplies the missing depth, in the uint16-millimetre / 0-means-invalid form the
    rest of the pipeline already expects from the Azure.

    It also fixes three smaller problems in the old ZED branch:
      * RESOLUTION.AUTO resolved to HD720; an explicit resolution is required so the
        saved frames match a known intrinsics set.
      * The saved PNG was 4-channel BGRA (`image.get_data()` straight to imwrite).
      * Self-calibration was left on, so intrinsics could shift between the
        calibration capture and later use.

CONVENTIONS (match the Azure path, so downstream code is unchanged)
    color : (H, W, 3) uint8  BGR
    depth : (H, W)    uint16 millimetres, 0 = invalid, registered to the color image

    ZED depth needs no alignment step -- MEASURE.DEPTH is already computed in the
    LEFT rectified camera frame, so it is pixel-aligned with VIEW.LEFT. Its native
    invalid values are NaN / +inf (TOO_FAR) / -inf (TOO_CLOSE), all of which are
    normalised to 0 here.

INTRINSICS
    Use apriltag_image._camera_params_for("zed", width, height), which loads the
    RECTIFIED left-camera values from a zed_intrinsics*.npz and rescales to the frame
    size. Do NOT use /usr/local/zed/settings/SN*.conf -- those are raw/unrectified and
    differ from the rectified set by ~46 px in cx and ~1.2% in fx.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:                                  # pragma: no cover
    raise ImportError("zed_calib_rgbd requires OpenCV") from exc


DEFAULT_RESOLUTION = "HD2K"          # 2208x1242, exactly 16:9, best depth on a ZED1
DEFAULT_DEPTH_MODE = "NEURAL_PLUS"
DEFAULT_DEPTH_MIN_M = 0.20
DEFAULT_DEPTH_MAX_M = 20.0
DEFAULT_CONFIDENCE = 47
DEFAULT_WARMUP = 30
DEFAULT_MEDIAN_FRAMES = 5

# Temporal depth stabilization, 0-100 (0 = raw depth, 1 = the SDK default).
# 10 is the value in the ZED Depth Viewer "best setup" the rig was tuned with, and
# this module previously left the parameter unset, i.e. at the SDK's 1. It is a
# TEMPORAL filter (it damps frame-to-frame oscillation), so it changes depth NOISE,
# not the stereo geometry -- the 16.03 px disparity offset in
# zed_capture/zed_depth_correction.json is a constant lens-yaw bias and is not
# recomputed from this. Verify with that file's depth/AprilTag ratio check after a
# capture if you change it further.
DEFAULT_DEPTH_STABILIZATION = 10

# Manual exposure, 0-100 (%). None = leave auto-exposure on.
#
# WHY THIS MATTERS FOR APRILTAG DETECTION
#   On the cloth rig the scene is a black backdrop with white robot arms. Auto
#   exposure meters for the dark background and blows out everything bright: the
#   tag's white border saturates to 255 and blooms into the neighbouring cells,
#   and the black cells lift to a mottled grey. AprilTag needs a clean black/white
#   step at the tag border, so detection becomes marginal -- measured 28/72 frames
#   on the right arm and 41/72 on the left. The Azure branch of
#   capture_poses_and_images_for_calibration*.py has always set manual exposure
#   (6000 us) for exactly this reason; the ZED path did not, which is the gap.
#
#   Lower value = darker = less bloom. Tune with the check in the module docstring:
#   you want <5% of the tag crop saturated and the black cells clearly below ~80.
DEFAULT_EXPOSURE = 10
DEFAULT_GAIN = None                  # 0-100; None = auto


def _settings_path():
    """The factory-calibration directory (SN<serial>.conf), or None for the default.

    Not the intrinsics -- see INTRINSICS in the module docstring; these conf values
    are raw/unrectified and must not be read directly. This is the file the SDK
    itself rectifies with, and open() fails outright with
    CALIBRATION_FILE_NOT_AVAILABLE when it cannot find one. Its default is
    /usr/local/zed/settings, which on this machine is root-owned and unreadable, so
    the SDK falls back to downloading -- `mkdir -p /usr/local/zed/settings` then
    fails with Permission denied and takes open() down with it.

    Shares zed_capture/settings with the live tracker (realtime/frame_source.py), so
    both rectify with the same conf.
    """
    try:
        zed_capture = Path(__file__).resolve().parents[2] / "zed_capture"
        if str(zed_capture) not in sys.path:
            sys.path.insert(0, str(zed_capture))
        import zed_depth_config
        path = zed_depth_config.settings_dir()
    except Exception:
        return None
    return path if path.is_dir() else None


def open_zed(
    resolution: str = DEFAULT_RESOLUTION,
    depth_mode: str = DEFAULT_DEPTH_MODE,
    depth_min_m: float = DEFAULT_DEPTH_MIN_M,
    depth_max_m: float = DEFAULT_DEPTH_MAX_M,
    confidence: int = DEFAULT_CONFIDENCE,
    depth_stabilization: int = DEFAULT_DEPTH_STABILIZATION,
    warmup_frames: int = DEFAULT_WARMUP,
    fps: int = 15,
    exposure: int | None = DEFAULT_EXPOSURE,
    gain: int | None = DEFAULT_GAIN,
    self_calib: bool = False,
    right_side_measure: bool = False,
):
    """Open the ZED with depth enabled and settings suited to calibration.

    Returns:
        (zed, runtime, info) where info records the negotiated width/height/fps,
        serial number, firmware and SDK version.

    Raises:
        RuntimeError with an actionable message if open() fails -- most often because
        ZED_Depth_Viewer still holds the camera (USB ZEDs are single-process).
    """
    import pyzed.sl as sl

    init = sl.InitParameters()
    init.camera_resolution = getattr(sl.RESOLUTION, resolution)
    init.camera_fps = int(fps)
    init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode)
    init.coordinate_units = sl.UNIT.METER          # so MEASURE.DEPTH is in metres
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE   # X right, Y down, Z forward
    init.depth_minimum_distance = float(depth_min_m)
    init.depth_maximum_distance = float(depth_max_m)
    init.depth_stabilization = int(depth_stabilization)
    # Self-calibration default OFF: it re-estimates the stereo extrinsics at every
    # open(), and the rectified intrinsics are derived from those. Letting it run can
    # shift fx/cx/cy between this capture and whatever uses the resulting calibration,
    # and if it SUCCEEDS the stored disparity offset no longer applies (see the
    # WARNING in zed_capture/zed_depth_correction.json). self_calib=True enables it
    # and turns on SDK verbose logging so its outcome is actually visible.
    init.camera_disable_self_calib = not self_calib
    init.sdk_verbose = 1 if self_calib else 0
    # MEASURE.DEPTH_RIGHT (depth registered to the RIGHT rectified view) is only
    # computed when this is set at open(); retrieving it otherwise fails.
    init.enable_right_side_measure = bool(right_side_measure)

    settings = _settings_path()
    if settings is not None:
        init.optional_settings_path = str(settings)
        print(f"[zed] factory calibration from {settings}")
    else:
        print("[zed] no local settings dir; the SDK will use /usr/local/zed/settings "
              "and may fail with CALIBRATION_FILE_NOT_AVAILABLE without root")

    if depth_mode.upper().startswith("NEURAL"):
        print(f"[zed] depth_mode={depth_mode}: the SDK may optimize its AI model on "
              "first use for this resolution. That can take minutes and prints "
              "nothing -- it is not a hang.")

    # Retry: this ZED's USB link throws EPROTO (-71) on the isochronous video stream
    # (visible in dmesg as "uvcvideo ... Non-zero status (-71)"), which makes open()
    # fail with CAMERA_NOT_DETECTED even though get_device_list() reports AVAILABLE.
    # It is transient and clears on a retry, so don't abort a robot capture over it.
    zed = sl.Camera()
    status = None
    for attempt in range(1, 4):
        status = zed.open(init)
        if status == sl.ERROR_CODE.SUCCESS:
            if attempt > 1:
                print(f"[zed] open() succeeded on attempt {attempt}")
            break
        name = getattr(status, "name", str(status))
        print(f"[zed] open() attempt {attempt}/3 failed: {name}")
        if attempt < 3:
            time.sleep(3.0)
    if status != sl.ERROR_CODE.SUCCESS:
        name = getattr(status, "name", str(status))
        raise RuntimeError(
            f"ZED open() failed after 3 attempts: {name}\n"
            "  1. A USB ZED allows only ONE process at a time -- close ZED_Depth_Viewer / "
            "ZED_Explorer (pgrep -af ZED_Depth_Viewer).\n"
            "  2. Check for a held device:  fuser -v /dev/video0\n"
            "  3. Check the USB link:  dmesg | grep -i uvcvideo | tail\n"
            "     Repeated 'Non-zero status (-71)' means a marginal cable/port -- move the\n"
            "     ZED to a rear USB 3 port with no hub, or replug it."
        )

    runtime = sl.RuntimeParameters()
    runtime.enable_depth = True
    runtime.enable_fill_mode = False
    runtime.confidence_threshold = int(confidence)
    runtime.texture_confidence_threshold = 100
    runtime.remove_saturated_areas = True

    conf = zed.get_camera_information().camera_configuration
    info = {
        "width": int(conf.resolution.width),
        "height": int(conf.resolution.height),
        "fps": float(conf.fps),
        "serial_number": int(zed.get_camera_information().serial_number),
        "firmware_version": int(conf.firmware_version),
        "sdk_version": sl.Camera.get_sdk_version(),
        "resolution_requested": resolution,
        "depth_mode": depth_mode,
        "depth_stabilization": int(depth_stabilization),
        "confidence": int(confidence),
        "self_calib_disabled": not self_calib,
        "right_side_measure": bool(right_side_measure),
    }
    print(f"[zed] opened SN{info['serial_number']} fw{info['firmware_version']} "
          f"sdk{info['sdk_version']} -> {info['width']}x{info['height']} @ "
          f"{info['fps']:g} fps, depth={depth_mode}, "
          f"stabilization={int(depth_stabilization)}, "
          f"self-calib {'ENABLED' if self_calib else 'DISABLED'}")
    if self_calib:
        # The SDK reports the self-calibration outcome only in its own verbose log
        # (the [ZED] lines above; SDK 4 has no status API for it). Print the rectified
        # intrinsics as a second, quantitative witness: the factory HD2K rectification
        # gives fx 1414.575 (zed_depth_correction.json reference_fx_px); if
        # self-calibration succeeded it re-rectifies the pair and these shift.
        lc = (zed.get_camera_information().camera_configuration
              .calibration_parameters.left_cam)
        print(f"[zed] self-calib check: look for '[ZED]' self-calibration lines above. "
              f"Rectified left fx/fy/cx/cy = {lc.fx:.3f}/{lc.fy:.3f}/{lc.cx:.3f}/"
              f"{lc.cy:.3f} (factory HD2K reference fx 1414.575 -- a shift means the "
              f"pair was re-rectified, and the stored disparity offset no longer "
              f"applies to this session).")

    # Manual exposure/gain BEFORE warmup, so the warmup frames settle at the value
    # we actually want. Setting either one implicitly disables AEC_AGC.
    if exposure is not None:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, int(exposure))
        info["exposure"] = int(exposure)
        print(f"[zed] manual EXPOSURE = {exposure} (auto-exposure disabled)")
    if gain is not None:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, int(gain))
        info["gain"] = int(gain)
        print(f"[zed] manual GAIN = {gain}")
    if exposure is None and gain is None:
        print("[zed] WARN: AUTO exposure. On a dark scene with bright subjects this blows "
              "out the AprilTag border and makes detection marginal -- set "
              "DEFAULT_EXPOSURE in zed_calib_rgbd.py before a calibration capture.")

    for _ in range(max(0, int(warmup_frames))):
        zed.grab(runtime)
    print(f"[zed] warmup: {warmup_frames} frames")

    return zed, runtime, info


def grab_rgbd(zed, runtime, median_frames: int = DEFAULT_MEDIAN_FRAMES,
              with_right_color: bool = False):
    """Grab one RGB-D frame.

    Args:
        median_frames: per-pixel median over N grabs. The arm is stationary at each
            calibration pose, so this is essentially free accuracy and it removes
            stereo speckle without the edge-bleeding of a spatial filter.
        with_right_color: also return the RIGHT rectified color image, so the
            stereo pair can be re-matched offline (e.g. by FoundationStereo).
            VIEW.RIGHT is the same rectified frame the SDK's own matcher consumed,
            so an offline matcher sees exactly the geometry MEASURE.DEPTH saw.

    Returns:
        (color_bgr, depth_mm) by default, or
        (color_bgr, depth_mm, right_bgr) when with_right_color=True.
        color_bgr/right_bgr: (H,W,3) uint8; depth_mm: (H,W) uint16 mm, 0 = invalid.

    Raises:
        RuntimeError if grab/retrieve fails.
    """
    import pyzed.sl as sl

    n = max(1, int(median_frames))
    # Keep the Mats alive for the whole call: get_data() returns an array with
    # OWNDATA=False AND base=None, so a freed Mat leaves a dangling pointer that
    # reads garbage silently. deep_copy=True is mandatory, not hygiene.
    img_mat, depth_mat = sl.Mat(), sl.Mat()

    colors, depths, rights = [], [], []
    for _ in range(n):
        for _attempt in range(10):
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                break
        else:
            raise RuntimeError("ZED grab() failed 10x")

        if zed.retrieve_image(img_mat, sl.VIEW.LEFT) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("retrieve_image(VIEW.LEFT) failed")
        if zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("retrieve_measure(MEASURE.DEPTH) failed")

        bgra = img_mat.get_data(deep_copy=True)                  # (H,W,4) BGRA
        colors.append(cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR))    # 3-channel, for imwrite

        raw = np.asarray(depth_mat.get_data(deep_copy=True), dtype=np.float32)
        # NaN (occluded/invalid), +inf (TOO_FAR), -inf (TOO_CLOSE) -> NaN for nanmedian
        depths.append(np.where(np.isfinite(raw) & (raw > 0.0), raw, np.nan))

        if with_right_color:
            if zed.retrieve_image(img_mat, sl.VIEW.RIGHT) != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError("retrieve_image(VIEW.RIGHT) failed")
            bgra_r = img_mat.get_data(deep_copy=True)
            rights.append(cv2.cvtColor(bgra_r, cv2.COLOR_BGRA2BGR))

    if n == 1:
        color_bgr, depth_m = colors[0], depths[0]
        right_bgr = rights[0] if with_right_color else None
    else:
        import warnings
        color_bgr = np.median(np.stack(colors, 0), axis=0).astype(np.uint8)
        if with_right_color:
            right_bgr = np.median(np.stack(rights, 0), axis=0).astype(np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN pixels
            depth_m = np.nanmedian(np.stack(depths, 0), axis=0)

    depth_m = np.where(np.isfinite(depth_m), depth_m, 0.0).astype(np.float32)
    depth_m[depth_m < 0.0] = 0.0

    depth_mm = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
    depth_mm[depth_m <= 0.0] = 0                       # Azure convention: 0 = invalid

    valid = depth_mm > 0
    if valid.any():
        print(f"[zed] grab: valid_depth={100.0 * valid.mean():.1f}%  "
              f"range={depth_m[valid].min():.3f}-{depth_m[valid].max():.3f} m")
    else:
        print("[zed] WARN: no valid depth in this frame")

    if with_right_color:
        return color_bgr, depth_mm, right_bgr
    return color_bgr, depth_mm


def save_depth_vis(path, depth_mm) -> None:
    """Write a viewable colour image of a uint16 millimetre depth map.

    The depth VALUES are already in {side}_calibration_rgbd.npz. This writes a
    picture you can open, so you can see the depth quality at each pose without
    a script. Invalid pixels (0) stay black.

    Args:
        path:     output .png path.
        depth_mm: (H, W) uint16 millimetres, 0 = invalid.
    """
    d = np.asarray(depth_mm, dtype=np.float32)
    valid = d > 0
    if not valid.any():
        cv2.imwrite(str(path), np.zeros((*d.shape, 3), np.uint8))
        return
    # 1st/99th percentile: a few multipath pixels must not flatten the whole scene.
    lo, hi = np.percentile(d[valid], (1.0, 99.0))
    norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~valid] = 0
    cv2.imwrite(str(path), vis)


def close_zed(zed) -> None:
    """Close the camera, swallowing teardown errors."""
    if zed is None:
        return
    try:
        zed.close()
    except Exception:
        pass
