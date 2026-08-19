#!/usr/bin/env python3
"""
zed_camera.py

Reusable ZED stereo-camera RGB-D helpers, mirroring the structure of
hand_to_eye_calibration/roahm-deformable-objects/zsy-testmycode/capture_azure_sam_mask.py
so the two capture paths are interchangeable downstream.

This module has NO import-time side effects (beyond importing pyzed), so other
scripts can do:

    import sys; sys.path.insert(0, "zed_capture")
    import zed_camera as zc
    zed, info = zc.open_zed(zc.ZedInitConfig())

Config surface mirrors the ZED Depth Viewer GUI (Settings > Processing > Depth).
The viewer's own persisted settings at

    ~/.config/StereoLabs/Depth Viewer.conf

are read with read_viewer_conf() but are NEVER written. Nothing in this module
opens that file for writing, and nothing touches /usr/local/zed/settings/.

Key semantic differences vs the Azure Kinect path
-------------------------------------------------
  * ZED MEASURE.DEPTH is float32 in InitParameters.coordinate_units (we force
    METER), not uint16 mm. No /1000 needed on retrieval.
  * ZED invalid depth uses THREE sentinels -- NaN (occluded/invalid),
    +inf (TOO_FAR), -inf (TOO_CLOSE). Azure uses 0. We normalise to the Azure
    convention (0.0 == invalid) on the way out so every existing consumer
    (run_dual_arm_cloth_deploy.py, run_dual_arm_real_eval.py, depth_to_vis)
    works unmodified.
  * ZED depth is already computed in the LEFT rectified camera frame, so it is
    registered to VIEW.LEFT. There is no depth_image_to_color_camera step.
  * Mat.get_data() returns an array with OWNDATA=False *and* base=None, i.e. it
    keeps no reference to the owning Mat. We always pass deep_copy=True.
  * ZED intrinsics are per-camera AND per-resolution, so they are read from the
    SDK at runtime and dumped to intrinsics.json rather than hardcoded.

Verified against ZED SDK 4.2.5 with an original ZED (MODEL.ZED, SN 22456).
"""

from __future__ import annotations

import configparser
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl

try:
    import zed_depth_config
except ImportError:  # imported from outside zed_capture/
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import zed_depth_config


# ============================================================
# Constants
# ============================================================

# Training resolution used by cloth-reveal-learning. The required artifacts
# (rgb.png / depth_m.npy / mask.png) are emitted at this size because
# run_dual_arm_cloth_deploy.py back-projects with INTRINSICS_640.
TRAIN_W = 640
TRAIN_H = 360

# ZED Depth Viewer's persisted settings. READ-ONLY from this module.
VIEWER_CONF_PATH = Path.home() / ".config" / "StereoLabs" / "Depth Viewer.conf"

# preprocess_observation depth clip range (must match project_config.py).
# Informational only -- we do not clip the saved depth.
DEPTH_CLIP_MIN_M = 0.20
DEPTH_CLIP_MAX_M = 2.50

# Grabs discarded before the real capture. With depth_stabilization != 0 the
# depth map is temporally filtered, so these frames also let stabilization
# converge -- not just AEC/AWB.
_WARMUP_FRAMES = 30

# Per-pixel median over N grabs. The single biggest quality lever on a static
# scene: it removes stereo speckle without the edge-bleeding of a spatial blur.
_MEDIAN_FRAMES = 5

_GRAB_RETRIES = 10

# ---------------------------------------------------------------------------
# Depth disparity-offset correction
# ---------------------------------------------------------------------------
# This camera (SN22456) reports depth that is too FAR. The cause is a constant
# disparity offset, not a scale error and not a baseline error.
#
# MEASURED TWICE, on two independent calibration captures:
#     zed_calib_001 : 16.26 +/- 0.51 px   (33 tag frames, 1.08-1.59 m)
#     zed_calib_002 : 16.03 +/- 0.37 px   (51 tag frames, 1.09-1.55 m)
#
# HOW WE KNOW IT IS AN OFFSET AND NOT A SCALE
#   depth = fx * B / disparity. A wrong fx would also make AprilTag PnP wrong,
#   and PnP agrees with robot forward kinematics to 2%. A wrong baseline would
#   scale depth by a CONSTANT factor. Instead the depth/PnP ratio RISES with
#   distance (1.114 at 1.09 m -> 1.169 at 1.53 m), which is the signature of a
#   constant disparity offset. One offset value fits every frame at every range.
#
# In angles this is a 0.649 deg error in the relative yaw (convergence) of the
# two lenses. SN22456.conf stores that as CV_2K = 0.0142736 rad = 0.818 deg.
# Stereolabs documents that self-calibration normally removes exactly this kind
# of drift at every open(). On this rig self-calibration always FAILS (the black
# backdrop has no texture), so the drift is never removed.
#
# !! THIS IS A WORKAROUND FOR A HARDWARE FAULT, NOT A FIX !!
#   Set it to 0.0 after you recalibrate the camera (ZED_Calibration, or a
#   successful self-calibration). If you leave it non-zero after a recalibration
#   it will push the depth wrong in the OTHER direction. The value is printed on
#   every capture and recorded in capture_config.json so it cannot apply silently.
#
# THE VALUE IS NOT DEFINED HERE. It lives in zed_capture/zed_depth_correction.json,
# because the calibration solvers under hand_to_eye_calibration/ must use the same
# number, and they cannot import this module (it needs pyzed). See zed_depth_config.
DEPTH_DISPARITY_OFFSET_PX = zed_depth_config.offset_px()


def correct_depth_disparity_offset(
    depth_m: np.ndarray,
    fx: float,
    baseline_m: float,
    offset_px: float | None = None,
) -> np.ndarray:
    """Undo a constant disparity offset in a metric depth map.

    The camera reports d_reported = d_true - offset_px, so it reports depth that
    is too far. We recover the true depth by adding the offset back in DISPARITY
    space, which is where the error actually lives:

        d_reported = fx * B / z_reported
        z_true     = fx * B / (d_reported + offset_px)

    A correction in disparity space is range-correct: it fixes near and far
    equally. A single multiplicative factor would not, which is why this is not
    implemented as a scale.

    The maths lives in zed_depth_config so that the calibration solvers use the
    identical implementation. This is a thin wrapper for the capture path.

    Args:
        depth_m:    (H, W) float32 metres. 0 or non-finite means invalid.
        fx:         RECTIFIED left-camera focal length in pixels, at the SAME
                    resolution as depth_m.
        baseline_m: stereo baseline in metres.
        offset_px:  disparity offset. 0.0 disables the correction. None means
                    "use the configured value".

    Returns:
        (H, W) float32 metres. Invalid pixels stay exactly as they were.
    """
    if offset_px is None:
        offset_px = zed_depth_config.offset_px()
    if not offset_px:
        return depth_m
    corrector = zed_depth_config.DepthCorrector(
        fx=fx, baseline_m=baseline_m, offset_px=offset_px, unit="m"
    )
    return corrector(depth_m)

# Viewer conf stores per-eye image height. Maps to sl.RESOLUTION names.
_VIEWER_HEIGHT_TO_RESOLUTION = {
    2180: "HD4K",
    1800: "QHDPLUS",
    1242: "HD2K",
    1200: "HD1200",
    1080: "HD1080",
    720: "HD720",
    600: "SVGA",
    376: "VGA",
}

# ZedImageConfig field -> sl.VIDEO_SETTINGS member name.
# Only settings that exist on a USB stereo ZED are listed. EXPOSURE_TIME,
# ANALOG_GAIN, DIGITAL_GAIN, EXPOSURE_COMPENSATION and DENOISING are ZED X /
# ZED X Mini only and are deliberately absent.
_IMAGE_SETTING_MAP = {
    "brightness": "BRIGHTNESS",
    "contrast": "CONTRAST",
    "hue": "HUE",
    "saturation": "SATURATION",
    "sharpness": "SHARPNESS",
    "gamma": "GAMMA",
    "gain": "GAIN",
    "exposure": "EXPOSURE",
    "aec_agc": "AEC_AGC",
    "wb_temperature": "WHITEBALANCE_TEMPERATURE",
    "wb_auto": "WHITEBALANCE_AUTO",
    "led_status": "LED_STATUS",
}

# Valid ranges from /usr/local/zed/include/sl/Camera.hpp (SDK 4.2.5).
_IMAGE_SETTING_RANGE = {
    "BRIGHTNESS": (0, 8),
    "CONTRAST": (0, 8),
    "HUE": (0, 11),
    "SATURATION": (0, 8),
    "SHARPNESS": (0, 8),
    "GAMMA": (1, 9),
    "GAIN": (0, 100),
    "EXPOSURE": (0, 100),
    "AEC_AGC": (0, 1),
    "WHITEBALANCE_TEMPERATURE": (2800, 6500),
    "WHITEBALANCE_AUTO": (0, 1),
    "LED_STATUS": (0, 1),
}


# ============================================================
# Config dataclasses -- one per ZED Depth Viewer panel
# ============================================================

@dataclass
class ZedInitConfig:
    """sl.InitParameters-level config.

    Mirrors the viewer's Settings > Processing > Depth > INITIALIZATION panel.
    Defaults reproduce the user's current viewer setup: HD2K@15, NEURAL_PLUS,
    0.2-20 m, 10% depth stabilization.
    """

    resolution: str = "HD2K"          # HD2K | HD1080 | HD720 | VGA | AUTO
    fps: int = 15                     # HD2K supports 15 only
    depth_mode: str = "NEURAL_PLUS"   # PERFORMANCE|QUALITY|ULTRA|NEURAL|NEURAL_PLUS
    depth_min_m: float = 0.20         # viewer "DEPTH MIN."
    depth_max_m: float = 20.0         # viewer "DEPTH MAX."
    depth_stabilization: int = 10     # viewer "DEPTH STABILIZATION" %, [0,100]
    flip: str = "OFF"                 # OFF | ON | AUTO
    serial_number: int | None = None  # None = first available camera
    # Self-calibration OFF by default -- see the module docstring. It re-estimates the
    # stereo extrinsics at every open(), and the rectified intrinsics are derived from
    # those, so letting it run can shift fx/cx/cy out from under a stored hand-eye
    # transform. Disabling it makes the intrinsics deterministic by intent.
    disable_self_calib: bool = True
    image_enhancement: bool = True    # ISP contrast enhancement (needs fw >= 1523)
    sdk_verbose: int = 0


@dataclass
class ZedRuntimeConfig:
    """sl.RuntimeParameters-level config.

    Mirrors the viewer's Settings > Processing > Depth > RUNTIME panel.
    """

    confidence: int = 47              # viewer "CONFIDENCE" %, [1,100], lower = stricter
    texture_confidence: int = 100     # viewer "TEXTURE CONFIDENCE" %, 100 = reject nothing
    fill_mode: bool = False           # viewer "ENABLE FILL MODE"
    remove_saturated: bool = True     # viewer "REMOVE SATURATED AREAS"


@dataclass
class ZedImageConfig:
    """sl.VIDEO_SETTINGS image controls. None means 'leave the camera alone'.

    Every field maps to a slider in the viewer's camera-control panel. Nothing
    is written to the camera unless you set it explicitly, so by default a
    capture inherits whatever the camera is already using.
    """

    brightness: int | None = None        # [0, 8]
    contrast: int | None = None          # [0, 8]
    hue: int | None = None               # [0, 11]
    saturation: int | None = None        # [0, 8]
    sharpness: int | None = None         # [0, 8]
    gamma: int | None = None             # [1, 9]
    gain: int | None = None              # [0, 100]  -- setting this clears AEC_AGC
    exposure: int | None = None          # [0, 100]  -- setting this clears AEC_AGC
    aec_agc: int | None = None           # 0 = manual, 1 = auto exposure/gain
    wb_temperature: int | None = None    # [2800, 6500] K, step 100 -- clears wb_auto
    wb_auto: int | None = None           # 0/1
    led_status: int | None = None        # 0/1 (needs fw >= 1523)


# ============================================================
# Enum resolution -- always by NAME, never by index
# ============================================================
#
# The viewer conf stores dropdown *indices* (e.g. quality=4). Those happen to
# match sl.DEPTH_MODE values in 4.2.5 (quality=4 <-> NEURAL, confirmed against
# the viewer's own "[Init] Depth mode: NEURAL" log line), but that coupling is
# not guaranteed across SDK versions. Everything user-facing therefore takes an
# enum NAME and is validated against the live enum.

def _enum_members(enum_cls) -> dict[str, object]:
    """Name -> member for a pyzed enum, tolerating enum.Enum and pybind styles."""
    members: dict[str, object] = {}
    try:
        for member in list(enum_cls):
            members[str(member.name)] = member
    except TypeError:
        pass
    if not members:
        for name in dir(enum_cls):
            if not name.isupper():
                continue
            try:
                val = getattr(enum_cls, name)
            except Exception:
                continue
            if isinstance(val, enum_cls):
                members[name] = val
    return members


def _enum_value(member) -> int | None:
    """Integer value of an enum member, or None."""
    if hasattr(member, "value"):
        try:
            return int(member.value)
        except Exception:
            pass
    try:
        return int(member)
    except Exception:
        return None


def enum_names(enum_cls) -> list[str]:
    """Sorted, user-facing member names for an enum (LAST excluded)."""
    return sorted(n for n in _enum_members(enum_cls) if n != "LAST")


# Aliases for how the viewer UI spells things.
_ENUM_ALIASES = {
    "NEUR+": "NEURAL_PLUS",
    "NEURAL+": "NEURAL_PLUS",
    "NEURALPLUS": "NEURAL_PLUS",
    "NEURAL PLUS": "NEURAL_PLUS",
    "OPEN": "ON",
    "CLOSE": "OFF",
}


def resolve_enum(enum_cls, name: str):
    """Look up an enum member by name against the live pyzed enum.

    Args:
        enum_cls: e.g. sl.DEPTH_MODE
        name:     e.g. "NEURAL_PLUS" (case-insensitive; "NEUR+" also accepted)

    Returns:
        The enum member.

    Raises:
        ValueError listing the valid names for the installed SDK.
    """
    key = str(name).strip().upper().replace("-", "_")
    key = _ENUM_ALIASES.get(key, key)
    members = _enum_members(enum_cls)
    if key not in members:
        raise ValueError(
            f"{getattr(enum_cls, '__name__', enum_cls)} has no member {name!r} "
            f"in SDK {sl.Camera.get_sdk_version()}. Valid: {enum_names(enum_cls)}"
        )
    return members[key]


def _depth_mode_name_from_value(value: int) -> str | None:
    """Map an integer depth-mode value (as stored in the viewer conf) to a name."""
    for name, member in _enum_members(sl.DEPTH_MODE).items():
        if name != "LAST" and _enum_value(member) == int(value):
            return name
    return None


# ============================================================
# ZED Depth Viewer conf -- READ ONLY
# ============================================================

def read_viewer_conf(path: Path = VIEWER_CONF_PATH) -> dict[str, int | str]:
    """Read the ZED Depth Viewer's persisted settings.

    Opened for reading only. This function -- and this module -- never writes to
    the viewer's config, so your viewer setup is left exactly as you left it.

    Args:
        path: conf file location.

    Returns:
        Flat dict of all keys across all sections, ints coerced where possible.
        Empty dict if the file is missing or unparseable.

    Note:
        The viewer only flushes this file on apply/exit, so it can lag what the
        GUI currently shows.
    """
    if not path.is_file():
        print(f"[zed] no viewer conf at {path} -- using script defaults")
        return {}

    parser = configparser.ConfigParser()
    parser.optionxform = str  # preserve key case
    try:
        with path.open("r") as handle:  # 'r' -- never 'w'
            parser.read_file(handle)
    except (OSError, configparser.Error) as exc:
        print(f"[zed] WARN: could not parse {path}: {exc}")
        return {}

    out: dict[str, int | str] = {}
    for section in parser.sections():
        for key, raw in parser.items(section):
            try:
                out[key] = int(raw)
            except ValueError:
                out[key] = raw
    return out


def viewer_conf_to_configs(
    conf: dict[str, int | str],
    init_cfg: ZedInitConfig | None = None,
    runtime_cfg: ZedRuntimeConfig | None = None,
) -> tuple[ZedInitConfig, ZedRuntimeConfig]:
    """Overlay viewer-conf values onto config dataclasses.

    Args:
        conf:        output of read_viewer_conf()
        init_cfg:    base config to copy from (defaults used if None)
        runtime_cfg: base config to copy from (defaults used if None)

    Returns:
        (ZedInitConfig, ZedRuntimeConfig) -- new objects; inputs untouched.

    Only keys the viewer actually persists are honoured: height, fps, quality,
    depth_min, depth_max. The viewer does NOT persist confidence /
    texture_confidence / fill_mode, so those keep their passed-in values.
    """
    init = ZedInitConfig(**asdict(init_cfg)) if init_cfg else ZedInitConfig()
    runtime = ZedRuntimeConfig(**asdict(runtime_cfg)) if runtime_cfg else ZedRuntimeConfig()
    if not conf:
        return init, runtime

    height = conf.get("height")
    if isinstance(height, int):
        if height in _VIEWER_HEIGHT_TO_RESOLUTION:
            init.resolution = _VIEWER_HEIGHT_TO_RESOLUTION[height]
        else:
            print(f"[zed] WARN: viewer height={height} unrecognised -- keeping {init.resolution}")

    fps = conf.get("fps")
    if isinstance(fps, int) and fps > 0:
        init.fps = fps

    quality = conf.get("quality")
    if isinstance(quality, int):
        name = _depth_mode_name_from_value(quality)
        if name:
            init.depth_mode = name
        else:
            print(f"[zed] WARN: viewer quality={quality} maps to no DEPTH_MODE -- keeping {init.depth_mode}")

    depth_min = conf.get("depth_min")
    if isinstance(depth_min, (int, float)):
        init.depth_min_m = float(depth_min) / 1000.0  # viewer stores mm
    depth_max = conf.get("depth_max")
    if isinstance(depth_max, (int, float)):
        init.depth_max_m = float(depth_max) / 1000.0

    if "hdr_mode" in conf:
        print(
            f"[zed] note: viewer hdr_mode={conf['hdr_mode']} ignored -- SDK "
            f"{sl.Camera.get_sdk_version()} exposes no HDR path for a stereo ZED "
            "(no InitParameters.enable_hdr, no VIDEO_SETTINGS.HDR*). Use manual "
            "exposure/gain for dynamic range instead."
        )

    print(
        f"[zed] seeded from viewer conf: resolution={init.resolution} fps={init.fps} "
        f"depth_mode={init.depth_mode} depth=[{init.depth_min_m:.2f}, {init.depth_max_m:.2f}] m"
    )
    return init, runtime


# ============================================================
# SDK parameter construction
# ============================================================

def build_init_parameters(cfg: ZedInitConfig) -> sl.InitParameters:
    """Translate ZedInitConfig into sl.InitParameters.

    Raises:
        ValueError on an unknown resolution / depth-mode / flip name.
    """
    init = sl.InitParameters()
    init.camera_resolution = resolve_enum(sl.RESOLUTION, cfg.resolution)
    init.camera_fps = int(cfg.fps)
    init.depth_mode = resolve_enum(sl.DEPTH_MODE, cfg.depth_mode)

    # METER so MEASURE.DEPTH comes back in metres, matching depth_m.npy.
    init.coordinate_units = sl.UNIT.METER
    # IMAGE (X right, Y down, Z forward) is what the existing back-projection
    # math assumes: x=(u-cx)*z/fx, y=(v-cy)*z/fy. Do not change this to a
    # Z-up system or every consumer's geometry silently flips.
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    init.depth_minimum_distance = float(cfg.depth_min_m)
    init.depth_maximum_distance = float(cfg.depth_max_m)
    init.depth_stabilization = int(cfg.depth_stabilization)
    init.camera_image_flip = resolve_enum(sl.FLIP_MODE, cfg.flip)
    init.camera_disable_self_calib = bool(cfg.disable_self_calib)
    init.enable_image_enhancement = bool(cfg.image_enhancement)
    init.sdk_verbose = int(cfg.sdk_verbose)

    if cfg.serial_number is not None:
        init.set_from_serial_number(int(cfg.serial_number))

    return init


def build_runtime_parameters(cfg: ZedRuntimeConfig) -> sl.RuntimeParameters:
    """Translate ZedRuntimeConfig into sl.RuntimeParameters.

    Note:
        sl.SENSING_MODE was removed in SDK 4.2; enable_fill_mode replaces
        SENSING_MODE::FILL. Fill mode is a *mode*, not an additive flag -- it
        overrides confidence_threshold, texture_confidence_threshold and
        remove_saturated_areas.
    """
    runtime = sl.RuntimeParameters()
    runtime.enable_depth = True
    runtime.enable_fill_mode = bool(cfg.fill_mode)
    runtime.confidence_threshold = int(cfg.confidence)
    runtime.texture_confidence_threshold = int(cfg.texture_confidence)
    runtime.remove_saturated_areas = bool(cfg.remove_saturated)
    runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA

    if cfg.fill_mode:
        print(
            "[zed] WARN: fill_mode=True overrides confidence, texture_confidence "
            "and remove_saturated_areas -- those settings will have no effect."
        )
    return runtime


# ============================================================
# Camera lifecycle
# ============================================================

def _open_error_message(status) -> str:
    """Human-actionable message for a failed sl.Camera.open()."""
    name = getattr(status, "name", str(status))
    lines = [f"ZED open() failed: {name}"]
    if name in (
        "CAMERA_NOT_DETECTED",
        "CAMERA_FAILED_TO_SETUP",
        "CAMERA_STREAM_FAILED_TO_START",
        "CAMERA_ALREADY_IN_USE",
    ):
        lines.append(
            "  A USB ZED allows only ONE process at a time. Close ZED_Depth_Viewer / "
            "ZED_Explorer / any other capture script, then retry."
        )
        lines.append("  Check with:  pgrep -af 'ZED_Depth_Viewer|ZED_Explorer'")
    if name in ("INVALID_RESOLUTION", "INVALID_CALIBRATION_FILE"):
        lines.append(
            "  This camera may not support that resolution/fps pair. Run with "
            "--list-config to print the resolutions the installed SDK exposes."
        )
    if name == "NO_GPU_COMPATIBLE":
        lines.append("  NEURAL / NEURAL_PLUS need a CUDA GPU. Try --depth-mode ULTRA.")
    return "\n".join(lines)


def validate_image_config(image_cfg: ZedImageConfig | None) -> None:
    """Range-check every non-None image setting.

    Called before open() so a bad value fails immediately rather than after the
    camera has been brought up (and, for NEURAL modes, after model optimization).

    Raises:
        ValueError naming the offending flag and its valid range.
    """
    if image_cfg is None:
        return
    for field_name, setting_name in _IMAGE_SETTING_MAP.items():
        value = getattr(image_cfg, field_name, None)
        if value is None:
            continue
        low, high = _IMAGE_SETTING_RANGE[setting_name]
        if not low <= int(value) <= high:
            raise ValueError(
                f"--{field_name.replace('_', '-')}={value} out of range [{low}, {high}]"
            )


def open_zed(
    init_cfg: ZedInitConfig,
    image_cfg: ZedImageConfig | None = None,
) -> tuple[sl.Camera, dict]:
    """Open the ZED and apply image controls.

    Args:
        init_cfg:  InitParameters-level config.
        image_cfg: optional VIDEO_SETTINGS overrides (None fields are left alone).

    Returns:
        (camera, resolved) where `resolved` records what the SDK actually
        negotiated -- model, serial, firmware, sdk version, real width/height/fps,
        the requested-vs-actual deltas, and the read-back image settings.

    Raises:
        RuntimeError with an actionable message if open() fails.
    """
    # Validate everything that can be validated before touching the hardware.
    init = build_init_parameters(init_cfg)
    validate_image_config(image_cfg)

    if init_cfg.depth_mode.upper().startswith("NEURAL"):
        print(
            f"[zed] depth_mode={init_cfg.depth_mode}: the SDK may optimize its AI model on "
            "first use for this resolution. That can take several minutes and prints "
            "nothing -- it is not a hang."
        )
    if int(init_cfg.depth_stabilization) != 0:
        print(
            f"[zed] depth_stabilization={init_cfg.depth_stabilization} is temporal and "
            "auto-enables positional tracking; warmup frames let it converge."
        )
    if float(init_cfg.depth_min_m) < 0.30:
        print(
            f"[zed] note: depth_min={init_cfg.depth_min_m:.2f} m is below the practical "
            "minimum of a USB stereo ZED. It widens the disparity search (slower) "
            "without recovering usable near-field depth."
        )

    if init_cfg.disable_self_calib:
        print(
            "[zed] self-calibration DISABLED -- rectified intrinsics come straight from "
            "the factory calibration and are reproducible run to run."
        )
    else:
        print(
            "[zed] WARN: self-calibration ENABLED. It may re-estimate the stereo extrinsics "
            "and shift the rectified fx/cx/cy, which invalidates any hand-eye transform "
            "solved against different intrinsics. Re-run calibration if the values move."
        )

    print(
        f"[zed] opening: resolution={init_cfg.resolution} fps={init_cfg.fps} "
        f"depth_mode={init_cfg.depth_mode} depth=[{init_cfg.depth_min_m:.2f}, "
        f"{init_cfg.depth_max_m:.2f}] m stabilization={init_cfg.depth_stabilization}"
    )

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        try:
            zed.close()
        except Exception:
            pass
        raise RuntimeError(_open_error_message(status))

    # From here on the camera is live: anything that raises must close it, or the
    # device stays locked against the next run (and against the Depth Viewer).
    try:
        return _finish_open(zed, init_cfg, image_cfg)
    except Exception:
        close_zed(zed)
        raise


def _finish_open(
    zed: sl.Camera,
    init_cfg: ZedInitConfig,
    image_cfg: ZedImageConfig | None,
) -> tuple[sl.Camera, dict]:
    """Post-open introspection and image-control application."""
    info = zed.get_camera_information()
    conf = info.camera_configuration
    actual_w = int(conf.resolution.width)
    actual_h = int(conf.resolution.height)
    actual_fps = float(conf.fps)

    resolved = {
        "sdk_version": sl.Camera.get_sdk_version(),
        "camera_model": str(getattr(info.camera_model, "name", info.camera_model)),
        "serial_number": int(info.serial_number),
        "firmware_version": int(conf.firmware_version),
        "input_type": str(getattr(info.input_type, "name", info.input_type)),
        "width": actual_w,
        "height": actual_h,
        "fps": actual_fps,
        "requested_resolution": init_cfg.resolution,
        "requested_fps": int(init_cfg.fps),
    }

    print(
        f"[zed] opened: {resolved['camera_model']} SN{resolved['serial_number']} "
        f"fw{resolved['firmware_version']} sdk{resolved['sdk_version']} -> "
        f"{actual_w}x{actual_h} @ {actual_fps:g} fps"
    )
    if int(init_cfg.fps) and abs(actual_fps - float(init_cfg.fps)) > 0.5:
        print(
            f"[zed] WARN: requested {init_cfg.fps} fps but the SDK negotiated "
            f"{actual_fps:g} fps for {init_cfg.resolution}."
        )

    resolved["image_settings"] = apply_image_settings(zed, image_cfg)
    return zed, resolved


def apply_image_settings(
    zed: sl.Camera,
    image_cfg: ZedImageConfig | None,
) -> dict[str, int]:
    """Apply the non-None VIDEO_SETTINGS, then read every one back.

    Args:
        zed:       an open camera.
        image_cfg: overrides; None fields are not touched.

    Returns:
        Read-back {SETTING_NAME: value} for the capture record. Settings the
        camera does not support are simply absent.

    Note:
        Setting GAIN or EXPOSURE implicitly clears AEC_AGC; setting
        WHITEBALANCE_TEMPERATURE implicitly clears WHITEBALANCE_AUTO.
    """
    if image_cfg is not None:
        for field_name, setting_name in _IMAGE_SETTING_MAP.items():
            value = getattr(image_cfg, field_name, None)
            if value is None:
                continue
            setting = getattr(sl.VIDEO_SETTINGS, setting_name, None)
            if setting is None:
                print(f"[zed] WARN: VIDEO_SETTINGS.{setting_name} absent in this SDK -- skipped")
                continue
            low, high = _IMAGE_SETTING_RANGE[setting_name]
            if not low <= int(value) <= high:
                raise ValueError(
                    f"--{field_name.replace('_', '-')}={value} out of range [{low}, {high}]"
                )
            status = zed.set_camera_settings(setting, int(value))
            if status != sl.ERROR_CODE.SUCCESS:
                print(
                    f"[zed] WARN: set {setting_name}={value} -> "
                    f"{getattr(status, 'name', status)} (unsupported on this model?)"
                )
            else:
                print(f"[zed] set {setting_name} = {value}")

    applied: dict[str, int] = {}
    for setting_name in _IMAGE_SETTING_MAP.values():
        setting = getattr(sl.VIDEO_SETTINGS, setting_name, None)
        if setting is None:
            continue
        try:
            result = zed.get_camera_settings(setting)
        except Exception:
            continue
        # SDK 4.1+ returns (ERROR_CODE, value); older returns a bare int.
        if isinstance(result, tuple):
            if result[0] != sl.ERROR_CODE.SUCCESS:
                continue
            value = result[1]
        else:
            value = result
        try:
            applied[setting_name] = int(value)
        except (TypeError, ValueError):
            continue
    return applied


def warmup(
    zed: sl.Camera,
    runtime: sl.RuntimeParameters,
    n_frames: int = _WARMUP_FRAMES,
) -> int:
    """Discard n_frames grabs so auto-exposure, white balance and depth
    stabilization converge before the real capture.

    Returns:
        Number of grabs that succeeded.
    """
    ok = 0
    for _ in range(max(0, int(n_frames))):
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            ok += 1
    print(f"[zed] warmup: {ok}/{int(n_frames)} frames grabbed")
    return ok


def _grab(zed: sl.Camera, runtime: sl.RuntimeParameters, retries: int = _GRAB_RETRIES) -> None:
    """grab() with retries. Raises RuntimeError if all attempts fail."""
    last = None
    for _ in range(max(1, retries)):
        status = zed.grab(runtime)
        if status == sl.ERROR_CODE.SUCCESS:
            return
        last = status
    raise RuntimeError(
        f"ZED grab() failed after {retries} attempts: {getattr(last, 'name', last)}"
    )


def capture_rgbd_native(
    zed: sl.Camera,
    runtime: sl.RuntimeParameters,
    n_median: int = _MEDIAN_FRAMES,
    disparity_offset_px: float = DEPTH_DISPARITY_OFFSET_PX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Capture one RGB-D frame at the camera's native resolution.

    Args:
        zed:      an open camera.
        runtime:  runtime parameters.
        n_median: grabs to combine with a per-pixel median. 1 = single shot.
                  >1 markedly reduces stereo speckle on a static scene, and
                  unlike a spatial filter it does not bleed across depth edges.
        disparity_offset_px: constant disparity offset to remove from the depth.
                  See DEPTH_DISPARITY_OFFSET_PX. Pass 0.0 to disable.

    Returns:
        color_bgr (H, W, 3) uint8   BGR, from VIEW.LEFT
        depth_m   (H, W)    float32 metres, 0.0 = invalid
        depth_mm  (H, W)    uint16  mm,     0   = invalid

    Depth is already registered to the left rectified image -- no alignment step.
    depth_mm is derived from the returned depth_m (not retrieved separately) so
    the two maps agree pixel-for-pixel after the median.
    """
    n_median = max(1, int(n_median))

    # Mats stay alive for the whole function: get_data() hands back an array
    # with base=None, so a freed Mat leaves a dangling pointer that reads
    # garbage silently. We also always deep_copy.
    img_mat = sl.Mat()
    depth_mat = sl.Mat()

    color_frames: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []

    for _ in range(n_median):
        _grab(zed, runtime)

        if zed.retrieve_image(img_mat, sl.VIEW.LEFT) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("retrieve_image(VIEW.LEFT) failed")
        if zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("retrieve_measure(MEASURE.DEPTH) failed")

        bgra = img_mat.get_data(deep_copy=True)  # (H, W, 4) uint8 BGRA
        color_frames.append(cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR))

        raw = np.asarray(depth_mat.get_data(deep_copy=True), dtype=np.float32)
        # Three sentinels: NaN (occluded/invalid), +inf (TOO_FAR), -inf
        # (TOO_CLOSE). None are exported to Python, so test numerically.
        # np.nan_to_num alone would turn +inf into 3.4e38, not 0.
        depth_frames.append(np.where(np.isfinite(raw) & (raw > 0.0), raw, np.nan))

    if n_median == 1:
        color_bgr = color_frames[0]
        depth_m = depth_frames[0]
    else:
        color_bgr = np.median(np.stack(color_frames, axis=0), axis=0).astype(np.uint8)
        with warnings.catch_warnings():
            # All-NaN pixels (never valid in any frame) are expected.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            depth_m = np.nanmedian(np.stack(depth_frames, axis=0), axis=0)

    # Back to the Azure convention: 0.0 == invalid.
    depth_m = np.where(np.isfinite(depth_m), depth_m, 0.0).astype(np.float32)
    depth_m[depth_m < 0.0] = 0.0

    # Remove the constant disparity offset. This runs BEFORE depth_mm is derived,
    # so both maps carry the correction and stay consistent with each other.
    if disparity_offset_px:
        conf = zed.get_camera_information().camera_configuration
        fx = float(conf.calibration_parameters.left_cam.fx)
        baseline_m = float(conf.calibration_parameters.get_camera_baseline())
        valid_before = depth_m > 0.0
        mean_before = float(depth_m[valid_before].mean()) if valid_before.any() else 0.0
        depth_m = correct_depth_disparity_offset(
            depth_m, fx, baseline_m, disparity_offset_px
        )
        valid_after = depth_m > 0.0
        mean_after = float(depth_m[valid_after].mean()) if valid_after.any() else 0.0
        print(f"[depth-fix] disparity offset {disparity_offset_px:+.2f} px applied "
              f"(fx={fx:.4f}, B={baseline_m:.6f} m): mean valid depth "
              f"{mean_before:.4f} -> {mean_after:.4f} m")

    depth_mm = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
    depth_mm[depth_m <= 0.0] = 0

    valid = depth_m > 0.0
    pct = 100.0 * float(valid.sum()) / float(depth_m.size)
    if valid.any():
        print(
            f"[capture] {color_bgr.shape[1]}x{color_bgr.shape[0]}  median_frames={n_median}  "
            f"valid_depth={pct:.1f}%  range={depth_m[valid].min():.3f}-{depth_m[valid].max():.3f} m"
        )
    else:
        print(f"[capture] WARN: no valid depth at all (median_frames={n_median})")

    return color_bgr, depth_m, depth_mm


def close_zed(zed: sl.Camera | None) -> None:
    """Close the camera, swallowing teardown errors."""
    if zed is None:
        return
    try:
        zed.close()
    except Exception:
        pass


# ============================================================
# Intrinsics
# ============================================================

def _k_matrix(intr: dict) -> list[list[float]]:
    return [
        [intr["fx"], 0.0, intr["cx"]],
        [0.0, intr["fy"], intr["cy"]],
        [0.0, 0.0, 1.0],
    ]


def get_intrinsics(zed: sl.Camera, target_wh: tuple[int, int] | None = None) -> dict:
    """Read left-camera intrinsics from the SDK at the live resolution.

    Args:
        zed:       an open camera.
        target_wh: optional (width, height) to also emit a scaled intrinsics set
                   for, e.g. (640, 360).

    Returns:
        dict with "native" (fx, fy, cx, cy, disto, K, width, height) and, when
        target_wh is given, "scaled" with the same fields plus scale_x/scale_y.

    Note:
        The SDK 4.0-era `camera_information.calibration_parameters` shortcut does
        not exist in 4.2 -- the path is
        get_camera_information().camera_configuration.calibration_parameters.
        These are RECTIFIED parameters, so disto is expected to be ~all-zero;
        calibration_parameters_raw holds the unrectified distortion.
    """
    info = zed.get_camera_information()
    conf = info.camera_configuration
    calib = conf.calibration_parameters
    left = calib.left_cam

    native_w = int(conf.resolution.width)
    native_h = int(conf.resolution.height)
    native = {
        "fx": float(left.fx),
        "fy": float(left.fy),
        "cx": float(left.cx),
        "cy": float(left.cy),
    }

    try:
        disto = [float(v) for v in np.asarray(left.disto).ravel().tolist()]
    except Exception:
        disto = []

    out: dict = {
        "sdk_version": sl.Camera.get_sdk_version(),
        "camera_model": str(getattr(info.camera_model, "name", info.camera_model)),
        "serial_number": int(info.serial_number),
        "firmware_version": int(conf.firmware_version),
        "source": "sl.Camera.get_camera_information().camera_configuration."
                  "calibration_parameters.left_cam (RECTIFIED, left eye)",
        "native": {
            "width": native_w,
            "height": native_h,
            **native,
            "K": _k_matrix(native),
            "disto": disto,
        },
    }

    try:
        # get_camera_baseline() returns a length in InitParameters.coordinate_units,
        # which build_init_parameters() forces to METER.
        out["baseline_m"] = float(calib.get_camera_baseline())
    except Exception:
        pass
    for key, attr in (("h_fov_deg", "h_fov"), ("v_fov_deg", "v_fov"), ("d_fov_deg", "d_fov")):
        try:
            out["native"][key] = float(getattr(left, attr))
        except Exception:
            pass

    if target_wh is not None:
        tw, th = int(target_wh[0]), int(target_wh[1])
        sx = tw / float(native_w)
        sy = th / float(native_h)
        scaled = {
            "fx": native["fx"] * sx,
            "fy": native["fy"] * sy,
            "cx": native["cx"] * sx,
            "cy": native["cy"] * sy,
        }
        out["scaled"] = {
            "width": tw,
            "height": th,
            "scale_x": sx,
            "scale_y": sy,
            **scaled,
            "K": _k_matrix(scaled),
            "disto": disto,
            "note": "Back-project depth_m.npy / depth_mm.npy with these. "
                    "Distortion coefficients are scale-invariant.",
        }
        if abs(sx - sy) > 1e-6:
            out["scaled"]["aspect_warning"] = (
                f"non-uniform scale (sx={sx:.6f}, sy={sy:.6f}): native "
                f"{native_w}x{native_h} and target {tw}x{th} differ in aspect ratio"
            )

    return out


# ============================================================
# Resizing
# ============================================================

def downscale_to_training_res(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    out_w: int = TRAIN_W,
    out_h: int = TRAIN_H,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downscale a native-resolution capture to the training resolution.

    Args:
        color_bgr: (H, W, 3) uint8 BGR
        depth_m:   (H, W)    float32 metres, 0.0 = invalid
        out_w:     target width  (default 640)
        out_h:     target height (default 360)

    Returns:
        color_bgr (out_h, out_w, 3) uint8   INTER_AREA
        depth_m   (out_h, out_w)    float32 metres, INTER_NEAREST
        depth_mm  (out_h, out_w)    uint16  mm

    INTER_NEAREST on depth preserves 0 == invalid and avoids interpolating
    across depth discontinuities, which would invent surfaces mid-air.
    """
    color = cv2.resize(color_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
    depth = cv2.resize(depth_m, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    depth = depth.astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0

    depth_mm = np.clip(depth * 1000.0, 0.0, 65535.0).astype(np.uint16)
    depth_mm[depth <= 0.0] = 0
    return color, depth, depth_mm


# ============================================================
# Depth visualization
# (verbatim from capture_azure_sam_mask.py -- takes MILLIMETRES, 0 = invalid)
# ============================================================

def depth_to_vis(depth_mm: np.ndarray) -> np.ndarray:
    """Turbo-colormap visualization of depth with labeled colorbar (invalid pixels = black).

    Args:
        depth_mm: (H, W) uint16 millimetres, 0 = invalid.

    Returns:
        (H, W + extra, 3) uint8 BGR -- WIDER than the input, colorbar appended.
    """
    valid = depth_mm > 0
    if not valid.any():
        return np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    d = depth_mm.astype(np.float32)
    # Robust range: use 1st/99th percentile so a handful of multipath/noise
    # pixels (which can read 10+ m) don't blow out the colormap and squash
    # the real scene into a tiny slice. Outliers just saturate at the ends.
    dmin = float(np.percentile(d[valid], 1.0))
    dmax = float(np.percentile(d[valid], 99.0))
    if dmax - dmin < 1e-6:  # degenerate (flat) scene -- fall back to true range
        dmin, dmax = float(d[valid].min()), float(d[valid].max())
    norm = np.clip((d - dmin) / (dmax - dmin + 1e-6), 0.0, 1.0)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~valid] = 0

    # --- colorbar ---
    H, W = vis.shape[:2]
    bar_w = max(30, W // 20)
    pad = max(6, bar_w // 5)
    label_w = 72  # pixels reserved for text labels
    total_extra = bar_w + pad * 2 + label_w

    canvas = np.zeros((H, W + total_extra, 3), dtype=np.uint8)
    canvas[:, :W] = vis

    # draw gradient strip
    bar_x = W + pad
    bar_h = H - 2 * pad
    for row in range(bar_h):
        t = 1.0 - row / max(bar_h - 1, 1)  # top=far (dmax), bottom=near (dmin)
        color_val = int(t * 255)
        color = cv2.applyColorMap(
            np.array([[color_val]], dtype=np.uint8), cv2.COLORMAP_TURBO
        )[0, 0].tolist()
        canvas[pad + row, bar_x : bar_x + bar_w] = color

    # draw tick labels (depth in metres)
    n_ticks = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, H / 1200)
    thickness = 1
    text_x = bar_x + bar_w + 4
    for i in range(n_ticks):
        t = i / (n_ticks - 1)  # 0=bottom(near), 1=top(far)
        depth_val_m = (dmin + t * (dmax - dmin)) / 1000.0
        y = pad + bar_h - int(t * bar_h)
        y = max(pad + 8, min(H - pad - 2, y))
        cv2.line(canvas, (bar_x - 3, y), (bar_x + bar_w + 2, y), (200, 200, 200), 1)
        cv2.putText(canvas, f"{depth_val_m:.2f}m", (text_x, y + 4),
                    font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)

    # title above the bar
    title = "depth"
    (tw, _), _ = cv2.getTextSize(title, font, font_scale, thickness)
    cv2.putText(canvas, title, (bar_x + (bar_w - tw) // 2, pad - 4),
                font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)

    return canvas


# ============================================================
# Introspection helper (no camera needed)
# ============================================================

def print_sdk_config_surface() -> None:
    """Print the config surface of the installed SDK. Opens no camera."""
    print("=" * 60)
    print(f"ZED SDK {sl.Camera.get_sdk_version()}")
    print("=" * 60)

    print("\nRESOLUTION (name -> WxH):")
    for name in enum_names(sl.RESOLUTION):
        if name == "AUTO":
            print(f"  {name:10s} resolved at open() -- HD720 on a USB stereo ZED")
            continue
        member = resolve_enum(sl.RESOLUTION, name)
        try:
            res = sl.get_resolution(member)
            width, height = int(res.width), int(res.height)
            ratio = f"{width / height:.4f}" if height else "?"
            print(f"  {name:10s} {width}x{height:<5d} aspect {ratio}")
        except Exception:
            print(f"  {name:10s} (size unavailable)")
    print("  Note: HD4K / QHDPLUS / HD1200 / SVGA are ZED X-family only.")

    print("\nDEPTH_MODE (name = value):")
    for name in enum_names(sl.DEPTH_MODE):
        print(f"  {name:14s} = {_enum_value(resolve_enum(sl.DEPTH_MODE, name))}")

    print("\nFLIP_MODE: " + ", ".join(enum_names(sl.FLIP_MODE)))

    print("\nImage controls (VIDEO_SETTINGS) exposed by this script:")
    for field_name, setting_name in _IMAGE_SETTING_MAP.items():
        present = hasattr(sl.VIDEO_SETTINGS, setting_name)
        low, high = _IMAGE_SETTING_RANGE[setting_name]
        flag = "" if present else "   [ABSENT in this SDK]"
        print(f"  --{field_name.replace('_', '-'):18s} {setting_name:26s} [{low}, {high}]{flag}")

    print("\nNot exposed (absent for a stereo ZED in SDK 4.2):")
    print("  HDR                 no InitParameters.enable_hdr, no VIDEO_SETTINGS.HDR*")
    print("  EXPOSURE_TIME / ANALOG_GAIN / DIGITAL_GAIN / DENOISING /")
    print("  EXPOSURE_COMPENSATION                        ZED X / ZED X Mini only")
    print("  IMU orientation     viewer point-cloud display only; the original")
    print("                      ZED (MODEL.ZED) has no IMU at all")
    print("=" * 60)


if __name__ == "__main__":
    print_sdk_config_surface()
