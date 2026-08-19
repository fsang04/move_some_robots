#!/usr/bin/env python

"""AprilTag detection front-end shared by the calibration scripts.

Two changes from the original, both backward compatible for the Azure path:

1. DETECTOR BACKEND (fallback only -- normal runs are unaffected). The repo vendors a
   working apriltag.py in this directory with its C library under ../build/lib, so
   `import apriltag` succeeds whenever a script runs with this directory on sys.path,
   and that real detector is always preferred. If it is not importable -- a different
   cwd, or an env where the vendored .so will not load -- we fall back to
   apriltag_backend, a cv2.aruco-backed drop-in for the same DICT_APRILTAG_36h11 tags.
   This only adds robustness; it changes nothing when apriltag imports.

2. ZED INTRINSICS. This module used to hardcode
       ZED_CAMERA_PARAMS = (716.5634765625, 716.5634765625, 655.4454345703125, 395.7761535644531)
   Those numbers match NONE of the three cameras whose factory calibration is on this
   machine (/usr/local/zed/settings/SN{14451,22456,23516}.conf, LEFT_CAM_HD fx
   699.0-701.6), and they are +3.24% in fx and +19 px in cx away from SN22456's
   measured RECTIFIED HD720 values (fx 694.0772, cx 636.3453, cy 384.8484). AprilTag
   range scales with fx, so they biased every ZED tag pose by ~3 cm at 1 m -- and the
   error is silently resolution-specific. ZED intrinsics are per-camera AND
   per-resolution, so they are now LOADED, never hardcoded. If none can be found we
   raise instead of returning something plausible-but-wrong.

   Produce the npz with:
       pixi run -e humble python zed_capture/export_zed_intrinsics.py \
           --from-capture <a capture dir> --width 1280 --height 720

   Search order for the ZED npz (first hit wins):
       $ZED_INTRINSICS_NPZ
       <repo>/zed_capture/zed_intrinsics_*.npz
       <this dir>/zed_intrinsics_*.npz
   Any npz with a 3x3 "K" works.

3. SPECKLE GUARD (see _denoise_for_contour_detect). The vendored C detector runs with
   quad_contours enabled, and its contour tracer counts borders in an int16. On a
   noisy full-res frame the count overflows, contour_detect bails, and the caller
   dereferences the NULL it returns -- a hard segfault that kills the whole
   calibration run, not just the frame. Images are now probed and denoised only when
   they are close to that ceiling; safe images are passed through untouched.
"""

import os
from pathlib import Path

import cv2
import numpy as np

from azure_intrinsics import azure_intrinsics

try:                                    # prefer the real package when present
    import apriltag as _apriltag_pkg
    _BACKEND = "apriltag"
except Exception:                       # ImportError, or a broken build
    import apriltag_backend as _apriltag_pkg
    _BACKEND = "apriltag_backend(cv2.aruco)"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_DIR = Path(__file__).resolve().parent

_ZED_PARAMS_CACHE = {}


def _find_zed_intrinsics_npz():
    """Locate a ZED intrinsics npz. Returns a Path or None."""
    env = os.environ.get("ZED_INTRINSICS_NPZ")
    if env and Path(env).is_file():
        return Path(env)
    for directory in (_REPO_ROOT / "zed_capture", _THIS_DIR):
        hits = sorted(directory.glob("zed_intrinsics*.npz"))
        if hits:
            return hits[0]
    return None


def zed_camera_params(width=None, height=None):
    """Return (fx, fy, cx, cy) for the ZED's RECTIFIED left camera.

    Args:
        width, height: if given, the resolution the intrinsics must correspond to.
            When the stored npz was exported at a different resolution, the values
            are rescaled (only valid if the aspect ratio matches).

    Raises:
        FileNotFoundError with instructions if no intrinsics npz can be found.
    """
    key = (width, height)
    if key in _ZED_PARAMS_CACHE:
        return _ZED_PARAMS_CACHE[key]

    path = _find_zed_intrinsics_npz()
    if path is None:
        raise FileNotFoundError(
            "No ZED intrinsics npz found. ZED intrinsics are per-camera and "
            "per-resolution, so this module refuses to guess.\n"
            "Create one with:\n"
            "  pixi run -e humble python zed_capture/export_zed_intrinsics.py \\\n"
            "      --from-capture <capture dir with intrinsics.json> "
            "--width 1280 --height 720\n"
            "or point $ZED_INTRINSICS_NPZ at an npz containing a 3x3 'K'."
        )

    data = np.load(path)
    if "K" not in data.files:
        raise KeyError(f"{path} has no 'K' (keys: {data.files})")
    K = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    src_w = int(data["width"]) if "width" in data.files else None
    src_h = int(data["height"]) if "height" in data.files else None

    if width and height and src_w and src_h and (src_w, src_h) != (width, height):
        sx, sy = width / float(src_w), height / float(src_h)
        if abs(sx - sy) / max(sx, sy) > 1e-3:
            raise ValueError(
                f"{path} is for {src_w}x{src_h}; rescaling to {width}x{height} would "
                f"be non-uniform (sx={sx:.6f}, sy={sy:.6f}). Export at the resolution "
                "you actually captured."
            )
        print(f"[apriltag_image] rescaling ZED intrinsics {src_w}x{src_h} -> "
              f"{width}x{height} (x{sx:.6f})")
        fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
        src_w, src_h = width, height

    params = (float(fx), float(fy), float(cx), float(cy))
    print(f"[apriltag_image] ZED intrinsics from {path.name} "
          f"({src_w}x{src_h}): fx={params[0]:.6f} fy={params[1]:.6f} "
          f"cx={params[2]:.6f} cy={params[3]:.6f}")
    _ZED_PARAMS_CACHE[key] = params
    return params


def _camera_params_for(camera, width=None, height=None):
    """Return (fx, fy, cx, cy) for the given camera.

    Args:
        camera: "azure" or "zed".
        width, height: optional resolution the intrinsics must match (ZED only).
    """
    if camera == "zed":
        return zed_camera_params(width, height)
    if camera == "azure":
        K = azure_intrinsics
        return (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    raise ValueError(f"Unknown camera: {camera}. Use 'zed' or 'azure'.")


################################################################################
# Speckle guard for the contour-based quad detector
#
# apriltag_quad_contour.c thresholds with box_threshold(im, 255, invert=1, sz=15,
# tau=5) -- identical to cv2.adaptiveThreshold(MEAN_C, BINARY_INV, 15, 5) -- and hands
# the result to contour_detect(), whose border counter is an int16_t capped at
# INT16_MAX. Sensor speckle in an HD2K ZED frame survives that threshold as tens of
# thousands of 1-2 px blobs; measured border counts on zed_calib_002/right run
# 29k-36k against a ceiling of 32767. When it trips, contour_detect() returns NULL and
# quads_from_contours() calls zarray_size(NULL) on it -> SIGSEGV, taking the run down.
#
# A 3x3 median filter removes the speckle without touching the tag: on frames that
# detect both ways it moves the recovered translation by <0.15 mm and the rotation by
# <0.3 deg, and it slightly RAISES the decision margin. Gaussian blur (the detector's
# own quad_sigma knob) is not used as the first rung -- it erodes the tag's black
# border, biasing range outward by up to 3.5 mm at sigma=1.5, and at sigma=2.0 it
# flipped the planar-pose ambiguity outright. quad_decimate=2 flipped it too.
#
# The probe costs ~40 ms and the median ~1 ms, against ~1 s for a detection. Images
# under the trigger are returned as-is, so clean sequences (Azure 1280x720, and every
# frame that already worked) go through bit-identical to before.

_BORDER_LIMIT = 32767           # INT16_MAX, the CCOUNT_MAX in contrib/contour.c
_BORDER_TRIGGER = 26000         # ~80% of the limit; denoise only above this
_THRESH_BLOCK = 15              # qcp.threshold_neighborhood_size
_THRESH_C = 5                   # qcp.threshold_value


def _border_count(gray):
    """Number of borders contour_detect() would trace on this image."""
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, _THRESH_BLOCK, _THRESH_C)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return len(contours)


# Escalating rungs, cheapest and least destructive first. Measured worst-case border
# counts on the zed_calib_002 right frames: median 3 -> 3234, median 5 -> 855.
_DENOISE_LADDER = [
    ("median3", lambda im: cv2.medianBlur(im, 3)),
    ("median5", lambda im: cv2.medianBlur(im, 5)),
    ("median5+gauss1.5", lambda im: cv2.GaussianBlur(cv2.medianBlur(im, 5), (5, 5), 1.5)),
]


def _denoise_for_contour_detect(img, label=""):
    """Return img, denoised only if its border count risks overflowing contour_detect.

    Args:
        img: BGR or grayscale image as read by cv2.imread.
        label: image name, for the log line.

    Returns:
        The original array when it is already safe, otherwise the first rung of
        _DENOISE_LADDER that brings the border count under _BORDER_TRIGGER.

    Raises:
        RuntimeError if no rung gets under the trigger. Raising is deliberate: the
        alternative is handing the C detector an image that segfaults the process,
        and the calibration scripts already treat a failed frame as one to skip.
    """
    # Only the vendored C detector has the int16 border counter. The cv2.aruco
    # fallback has no such limit, so leave its input alone.
    if _BACKEND != "apriltag":
        return img

    # Matches detect_tags(), which calls COLOR_RGB2GRAY on a BGR array. The channel
    # swap is inherited, and the border count has to be measured on the same
    # grayscale the detector will actually threshold.
    def to_gray(im):
        return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) if im.ndim == 3 else im

    count = _border_count(to_gray(img))
    if count < _BORDER_TRIGGER:
        return img

    last = count
    for name, fn in _DENOISE_LADDER:
        cleaned = fn(img)
        last = _border_count(to_gray(cleaned))
        if last < _BORDER_TRIGGER:
            print(f"[apriltag_image] {label}: {count} borders vs contour_detect's "
                  f"{_BORDER_LIMIT} limit; {name} -> {last}")
            return cleaned

    raise RuntimeError(
        f"{label}: {count} borders, still {last} after every denoise rung "
        f"({', '.join(n for n, _ in _DENOISE_LADDER)}), against contour_detect's "
        f"{_BORDER_LIMIT} limit. Passing this to the detector would segfault the run. "
        "The frame is likely badly overexposed or out of focus -- exclude it with "
        "--exclude-images, or recapture."
    )


################################################################################
input_image_path = './test_image.png'
output_image_path = './test_image_output.png'


def apriltag_image(input_images=[input_image_path],
                   output_images=False,
                   output_images_path=[output_image_path],
                   display_images=True,
                   detection_window_name='AprilTag',
                   tag_size=0.05,
                   tag_family=None,
                   camera='zed',
                   dist=None,
                  ):

    '''
    Detect AprilTags from static images.

    Args:   input_images [list(str)]: List of images to run detection algorithm on
            output_images [bool]: Boolean flag to save/not images annotated with detections
            display_images [bool]: Boolean flag to display/not images annotated with detections
            detection_window_name [str]: Title of displayed (output) tag detection window
            camera [str]: 'zed' or 'azure' to select which camera intrinsics to use
            dist [array|None]: distortion coefficients. None/zeros for rectified images
                               (the ZED's VIEW.LEFT is rectified, so zeros are correct).

    Returns: the flat groups-of-four list [det, pose, e0, e1, ...] that
             _find_expected_tag() in the calculate_* scripts expects.
    '''
    for i, image in enumerate(input_images):

        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"could not read image: {image}")

        # Resolution matters for ZED intrinsics, so resolve them per image.
        height, width = img.shape[:2]
        camera_params = _camera_params_for(camera, width, height)

        name = os.path.split(image)[1]
        print('Reading {}...\n'.format(name))

        # Must happen before detect_tags: the overflow it prevents is a segfault, not
        # an exception, so there is nothing to catch after the fact.
        img = _denoise_for_contour_detect(img, label=name)

        if _BACKEND == "apriltag":
            options = _apriltag_pkg.DetectorOptions()
            if tag_family:
                options.families = tag_family
            detector = _apriltag_pkg.Detector(
                options=options, searchpath=_apriltag_pkg._get_dll_path()
            )
            result, overlay = _apriltag_pkg.detect_tags(img,
                                                        detector,
                                                        camera_params=camera_params,
                                                        tag_size=tag_size,
                                                        vizualization=3,
                                                        verbose=3,
                                                        annotation=True
                                                       )
        else:
            result, overlay = _apriltag_pkg.detect_tags(img,
                                                        None,
                                                        camera_params=camera_params,
                                                        tag_size=tag_size,
                                                        vizualization=3,
                                                        verbose=3,
                                                        annotation=True,
                                                        tag_family=tag_family or "tag36h11",
                                                        dist=dist,
                                                       )

        if output_images:
            output_path = output_images_path[i]
            cv2.imwrite(output_path, overlay)

        if display_images:
            cv2.imshow(detection_window_name, overlay)
            # Wait for a keypress OR window close; avoid hanging if the user closes the window.
            while True:
                key = cv2.waitKey(20)
                # Any key press continues
                if key != -1:
                    break
                # If window was closed, stop waiting
                try:
                    visible = cv2.getWindowProperty(detection_window_name, cv2.WND_PROP_VISIBLE)
                    if visible < 1:
                        break
                except Exception:
                    break
            cv2.destroyWindow(detection_window_name)

        return result

################################################################################

if __name__ == '__main__':
    print(f"[apriltag_image] detector backend: {_BACKEND}")
    apriltag_image()
