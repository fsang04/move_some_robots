"""Single source of truth for the ZED depth disparity-offset correction.

WHY THIS MODULE EXISTS
    Two separate trees consume the correction:
        zed_capture/            -- the perception capture path (applies it at capture)
        hand_to_eye_calibration/-- the calibration solvers (apply it at solve time)
    Both must use the SAME number. This module holds it, and the maths, in one
    place. It has no third-party dependency except numpy, and it does NOT import
    pyzed, so the calibration solvers can import it without a camera present.

WHERE THE NUMBER LIVES
    zed_capture/zed_depth_correction.json

    Found with the same 3-level search that apriltag_image.py uses for the
    intrinsics npz:
        $ZED_DEPTH_CORRECTION_JSON  ->  <repo>/zed_capture/  ->  this directory
    A one-shot override is available as $ZED_DEPTH_OFFSET_PX.

WHAT THE CORRECTION IS
    See the long comment in zed_camera.py. In short: this camera reports depth
    that is too far, because of a CONSTANT DISPARITY OFFSET (not a scale error,
    and not a baseline error). The fix belongs in disparity space:

        z_true = fx * B / (fx * B / z_reported + offset_px)

    A correction in disparity space is range-correct. A multiplicative factor
    would not be, which is why this is not implemented as a scale.

APPLYING IT TO A MEDIAN IS EXACT
    The solvers take the MEDIAN of an 11x11 depth patch and then correct that one
    scalar. This gives the same answer as correcting all 121 pixels and then
    taking the median, because the correction is strictly monotonic in z, and a
    monotonic map commutes with the median. Correcting one scalar per frame
    avoids doubling the memory of a (64, 1242, 2208) depth stack.

DOUBLE-CORRECTION IS THE MAIN HAZARD
    If the capture applies the offset AND the solver applies it, you get 32 px.
    Nothing crashes, and the calibration still converges -- to a wrong answer.
    The guard is a 'disparity_offset_px' key in the saved RGB-D npz:
        key absent           -> old dataset. Raw. Correct it.
        key present, == 0.0  -> raw. Correct it.
        key present, != 0.0  -> already corrected. Do nothing.
    Pass that value as already_applied_px to corrector_for().
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent

CONFIG_BASENAME = "zed_depth_correction.json"

# Cameras this correction applies to. It is a per-device measurement, so it must
# never be applied to an Azure Kinect (or to a second ZED) by accident.
SUPPORTED_CAMERAS = ("zed",)

_cache: dict = {}


# ---------------------------------------------------------------------------
# Locating and reading the config
# ---------------------------------------------------------------------------

def find_config() -> Path | None:
    """Locate zed_depth_correction.json. Returns a Path, or None."""
    env = os.environ.get("ZED_DEPTH_CORRECTION_JSON")
    if env and Path(env).is_file():
        return Path(env)
    for directory in (_REPO_ROOT / "zed_capture", _THIS_DIR):
        candidate = directory / CONFIG_BASENAME
        if candidate.is_file():
            return candidate
    return None


def load(force: bool = False) -> dict:
    """Read the config. Returns {} if there is no config file.

    A missing file is not an error: it means "no correction", which is the right
    default for a camera that was never measured, and for an Azure.
    """
    if not force and "cfg" in _cache:
        return _cache["cfg"]
    path = find_config()
    if path is None:
        _cache["cfg"] = {}
        _cache["path"] = None
        return {}
    with open(path, "r") as fh:
        cfg = json.load(fh)
    _cache["cfg"] = cfg
    _cache["path"] = path
    return cfg


def config_path() -> Path | None:
    """Path the values came from, or None."""
    load()
    return _cache.get("path")


def offset_px() -> float:
    """Configured disparity offset d in pixels. 0.0 means no shift.

    $ZED_DEPTH_OFFSET_PX overrides the file, for a one-shot experiment.
    """
    env = os.environ.get("ZED_DEPTH_OFFSET_PX")
    if env:
        return float(env)
    return float(load().get("disparity_offset_px", 0.0) or 0.0)


def scale() -> float:
    """Configured disparity scale a (dimensionless). 1.0 means no stretch.

    Correction model: disp_true = a * disp_reported + d. The scale absorbs
    percentage errors (baseline / rectified-focal mismatch) that a constant
    offset cannot; fitted vs AprilTag PnP ranging on zed_calib_fs_002 with the
    caliper-verified 95.0 mm tag (2026-08-19), both arms agreeing to 0.0007.

    $ZED_DEPTH_SCALE overrides the file, for a one-shot experiment.
    """
    env = os.environ.get("ZED_DEPTH_SCALE")
    if env:
        return float(env)
    return float(load().get("disparity_scale", 1.0) or 1.0)


def baseline_m() -> float:
    """Stereo baseline in metres.

    The capture path reads this from the live camera. The solvers have no camera,
    so for them this file is the only source.
    """
    value = load().get("baseline_m")
    if value is None:
        raise KeyError(
            f"'baseline_m' is missing from {config_path()}. The depth correction "
            "needs fx*B, and a solver has no camera to ask. Add it: for SN22456 it "
            "is 0.120001 (Baseline=120.001 in /usr/local/zed/settings/SN22456.conf)."
        )
    return float(value)


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------

def settings_dir() -> Path:
    """Directory holding the per-camera factory calibration (SN<serial>.conf).

    The ZED SDK defaults to /usr/local/zed/settings, which needs root to create
    -- and when it cannot write there, open() fails outright with
    CALIBRATION FILE NOT AVAILABLE. So a root-free copy inside the repo is
    preferred when present, and handed to the SDK as
    InitParameters.optional_settings_path. Order:

        1. $ZED_SETTINGS_PATH
        2. <repo>/zed_capture/settings          (root-free, checked into the rig)
        3. /usr/local/zed/settings              (the SDK default)

    Get the file for a serial from https://calib.stereolabs.com/?SN=<serial>
    (the parameter is case sensitive). Its md5 must match `conf_md5` in
    zed_depth_correction.json, or the disparity offset does not apply -- that is
    exactly what check_camera() verifies.
    """
    env = os.environ.get("ZED_SETTINGS_PATH")
    if env:
        return Path(env)
    local = _THIS_DIR / "settings"
    if local.is_dir():
        return local
    return Path("/usr/local/zed/settings")


def check_camera(verbose: bool = True) -> bool:
    """Verify the offset still belongs to the camera that is installed.

    The offset is valid only for one camera AND one factory calibration. If you
    run ZED_Calibration, or a self-calibration succeeds, or you swap the camera,
    the number is wrong -- and it then pushes the depth wrong in the OTHER
    direction. This compares the recorded serial and conf checksum with what is
    on disk now.

    Returns True if the config matches, or if there is nothing to check against.
    """
    import hashlib

    cfg = load()
    if not cfg:
        return True

    serial = cfg.get("camera_serial")
    recorded_md5 = cfg.get("conf_md5")
    if serial is None or not recorded_md5:
        return True

    conf = settings_dir() / f"SN{int(serial)}.conf"
    if not conf.is_file():
        if verbose:
            print(f"[depth-fix] WARNING: cannot verify the offset. {conf} is missing. "
                  f"Is SN{int(serial)} the camera that is connected?")
        return True

    actual_md5 = hashlib.md5(conf.read_bytes()).hexdigest()
    if actual_md5 == recorded_md5:
        return True

    if verbose:
        print("=" * 78)
        print(f"[depth-fix] STALE OFFSET. {conf.name} has changed since the offset "
              f"was measured.")
        print(f"            recorded md5 {recorded_md5}")
        print(f"            actual   md5 {actual_md5}")
        print(f"            The camera was recalibrated, so disparity_offset_px = "
              f"{offset_px()} px is no longer valid.")
        print(f"            RE-MEASURE it, or set it to 0.0 in {config_path()}.")
        print("=" * 78)
    return False


# ---------------------------------------------------------------------------
# The correction itself
# ---------------------------------------------------------------------------

class DepthCorrector:
    """Maps reported depth to corrected depth. Identity when disabled.

    Callable on a scalar or on an array. Invalid samples (<= 0, or non-finite)
    pass through untouched, because 0 means "no measurement" in every depth
    convention used here.

    Attributes:
        enabled: False means this is the identity. Call it anyway; it is cheap.
    """

    def __init__(self, fx: float, baseline_m: float, offset_px: float,
                 unit: str = "m", reason: str = "", scale: float = 1.0):
        if unit not in ("m", "mm"):
            raise ValueError(f"unit must be 'm' or 'mm', got {unit!r}")
        self.fx = float(fx)
        self.baseline_m = float(baseline_m)
        self.offset_px = float(offset_px)
        self.scale = float(scale)
        self.unit = unit
        self.reason = reason
        # fx*B carries the unit of the depth it will divide into.
        self._fx_b = self.fx * self.baseline_m * (1000.0 if unit == "mm" else 1.0)

    @property
    def enabled(self) -> bool:
        return bool(self.offset_px) or self.scale != 1.0

    def __call__(self, depth):
        """Correct a scalar or an array of depths, in self.unit.

        Model: disp_true = scale * disp_reported + offset_px, then back to depth.
        """
        if not self.enabled:
            return depth

        if np.isscalar(depth):
            z = float(depth)
            if not np.isfinite(z) or z <= 0.0:
                return depth
            return self._fx_b / (self.scale * (self._fx_b / z) + self.offset_px)

        out = np.asarray(depth, dtype=np.float32).copy()
        valid = np.isfinite(out) & (out > 0)
        if valid.any():
            disparity = self._fx_b / out[valid]
            out[valid] = (self._fx_b / (self.scale * disparity + self.offset_px)
                          ).astype(np.float32)
        return out

    def describe(self) -> str:
        if not self.enabled:
            return f"[depth-fix] correction OFF{(' -- ' + self.reason) if self.reason else ''}"
        # What the correction is worth, in mm, at two useful ranges.
        shifts = []
        for z_m in (1.0, 1.5):
            z = z_m * (1000.0 if self.unit == "mm" else 1.0)
            shifts.append(f"{(self(z) - z) * (1.0 if self.unit == 'mm' else 1000.0):+.1f} mm @ {z_m:.1f} m")
        return (f"[depth-fix] disparity a={self.scale:.4f}, d={self.offset_px:+.2f} px applied "
                f"(fx={self.fx:.2f} px, B={self.baseline_m * 1000:.1f} mm, unit={self.unit}): "
                + ", ".join(shifts))


def disabled_corrector(reason: str = "") -> DepthCorrector:
    """A no-op corrector. Safe module-level default."""
    return DepthCorrector(fx=1.0, baseline_m=1.0, offset_px=0.0, reason=reason)


def corrector_for(camera: str, fx: float, *, unit: str = "m",
                  offset_px_override: float | None = None,
                  scale_override: float | None = None,
                  already_applied_px: float = 0.0,
                  already_applied_scale: float = 1.0,
                  baseline_m_override: float | None = None,
                  verbose: bool = True) -> DepthCorrector:
    """Build the right corrector for one camera and one dataset.

    Args:
        camera:  'zed', 'azure', ... Anything not in SUPPORTED_CAMERAS gets the
                 identity, because the correction is a per-device measurement.
        fx:      RECTIFIED left-camera focal length in pixels, at the SAME
                 resolution as the depth it will correct.
        unit:    'm' for metric depth, 'mm' for the uint16 millimetre stacks that
                 the calibration captures store.
        offset_px_override: use this d instead of the config. 0.0 disables the
                 shift.
        scale_override: use this a instead of the config. 1.0 disables the
                 stretch.
        already_applied_px / already_applied_scale: what the dataset already had
                 applied, from the 'disparity_offset_px' / 'disparity_scale' keys
                 of its npz. Any non-raw pair means the data is already
                 corrected, so this returns the identity. THIS IS THE
                 DOUBLE-CORRECTION GUARD.
        baseline_m_override: use this instead of the config, e.g. a value read
                 from the live camera.

    Returns:
        A DepthCorrector. Never None, so the caller needs no branch.
    """
    if camera not in SUPPORTED_CAMERAS:
        corrector = disabled_corrector(f"camera is {camera!r}, not a ZED")
        if verbose:
            print(corrector.describe())
        return corrector

    if already_applied_px or (already_applied_scale and already_applied_scale != 1.0):
        corrector = disabled_corrector(
            f"the dataset already has a={already_applied_scale:.4f}, "
            f"d={already_applied_px:+.2f} px applied, so correcting again "
            "would double it")
        if verbose:
            print(corrector.describe())
        return corrector

    offset = offset_px() if offset_px_override is None else float(offset_px_override)
    scl = scale() if scale_override is None else float(scale_override)
    if not offset and scl == 1.0:
        corrector = disabled_corrector(
            "disparity correction is identity (a=1, d=0)"
            if offset_px_override is None and scale_override is None
            else "disabled on the command line")
        if verbose:
            print(corrector.describe())
        return corrector

    check_camera(verbose=verbose)
    baseline = baseline_m() if baseline_m_override is None else float(baseline_m_override)
    corrector = DepthCorrector(fx=fx, baseline_m=baseline, offset_px=offset,
                               unit=unit, scale=scl)
    if verbose:
        print(corrector.describe())
        if config_path() is not None and offset_px_override is None:
            print(f"[depth-fix] value from {config_path()}")
    return corrector


def dataset_applied_offset_px(npz_or_path) -> float:
    """Read the 'disparity_offset_px' provenance key from an RGB-D npz.

    Returns 0.0 when the key is absent, which is the correct reading for every
    dataset captured before the key existed (zed_calib_001..003): those hold RAW
    depth.
    """
    data = npz_or_path
    if isinstance(npz_or_path, (str, Path)):
        if not Path(npz_or_path).is_file():
            return 0.0
        data = np.load(npz_or_path)
    try:
        files = data.files
    except AttributeError:
        return 0.0
    if "disparity_offset_px" not in files:
        return 0.0
    return float(np.asarray(data["disparity_offset_px"]).reshape(-1)[0])


def dataset_applied_scale(npz_or_path) -> float:
    """Read the 'disparity_scale' provenance key from an RGB-D npz.

    Returns 1.0 when the key is absent -- correct for every dataset written
    before the scale term existed: those either hold RAW depth (offset key 0 or
    absent) or were corrected with the constant-offset model only (a = 1).
    """
    data = npz_or_path
    if isinstance(npz_or_path, (str, Path)):
        if not Path(npz_or_path).is_file():
            return 1.0
        data = np.load(npz_or_path)
    try:
        files = data.files
    except AttributeError:
        return 1.0
    if "disparity_scale" not in files:
        return 1.0
    return float(np.asarray(data["disparity_scale"]).reshape(-1)[0])
