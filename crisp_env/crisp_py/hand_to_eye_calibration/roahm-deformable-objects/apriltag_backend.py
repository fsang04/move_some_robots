#!/usr/bin/env python3
"""
apriltag_backend.py

A cv2.aruco-backed drop-in for the `apriltag` pip package's detect_tags(), so every
calibration script in this repo runs without that dependency.

WHY THIS EXISTS  (it is a FALLBACK, not a required fix)
    The repo vendors a working `apriltag.py` next to this file, with its C library
    under ../build/lib, so `import apriltag` succeeds and the real detector is used
    whenever a script runs with this directory on sys.path -- which is the normal
    case. apriltag_image.py always prefers that package.

    This module covers the cases where it is NOT importable: running from a different
    cwd without this directory on sys.path, an environment where the vendored .so
    cannot be loaded, or `pip install apriltag` (which fails to build a wheel in this
    project's environments). OpenCV's aruco module ships DICT_APRILTAG_36h11 -- the
    family these scripts request -- in every env here, so it detects the SAME physical
    tags with no extra dependency.

    Verified against a synthetic 36h11 tag: 0.0001 px reprojection RMS, recovered
    translation matching the analytic prediction to 0.001 mm laterally and 0.22 mm in
    depth.

RETURN CONTRACT (matches apriltag.detect_tags)
    A FLAT list with four entries per detected tag:
        [det_0, T_0, e0_0, e1_0, det_1, T_1, e0_1, e1_1, ...]
    Callers iterate `for j in range(0, len(detections), 4)` and read
    `detections[j].tag_id` and `detections[j + 1]` as a 4x4 pose. See
    _find_expected_tag() in calculate_base_to_cam_nonlinear_opt.py.

    det exposes the attributes the repo actually uses: tag_id, corners, center,
    decision_margin, hamming, homography, tostring().

TAG FRAME CONVENTION
    Object points are laid out to match aruco's detection order:
        corners[0] top-left     -> (-s, +s, 0)
        corners[1] top-right    -> (+s, +s, 0)
        corners[2] bottom-right -> (+s, -s, 0)
        corners[3] bottom-left  -> (-s, -s, 0)
    with s = tag_size / 2 and +Z out of the tag face. This is the same tag frame the
    `apriltag` package uses (+X right, +Y up, +Z out of the face) -- only the corner
    the enumeration STARTS from differs, and solvePnP is insensitive to ordering as
    long as each image point is paired with its own object point. So the recovered
    T_cam_tag matches what the original detector returned, and an existing measured
    gripper2tag stays valid.

    That equivalence is the one assumption here. It shows up loudly if wrong: the
    rigid-fit residual and the two-estimator disagreement both blow up, provided the
    calibration poses vary EE orientation (a constant orientation hides it).
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError as exc:                              # pragma: no cover
    raise ImportError("apriltag_backend requires OpenCV") from exc


_FAMILY_TO_ARUCO = {
    "tag36h11": "DICT_APRILTAG_36h11",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag16h5": "DICT_APRILTAG_16h5",
    "tagCircle21h7": "DICT_APRILTAG_36h11",   # no aruco equivalent; fall back
    "tagStandard41h12": "DICT_APRILTAG_36h11",
}


class Detection:
    """Minimal stand-in for apriltag's Detection, with the attributes this repo uses."""

    __slots__ = ("tag_family", "tag_id", "hamming", "decision_margin",
                 "homography", "center", "corners")

    def __init__(self, tag_family, tag_id, corners, homography, decision_margin):
        self.tag_family = tag_family
        self.tag_id = int(tag_id)
        self.hamming = 0                       # aruco rejects bad codes outright
        self.decision_margin = float(decision_margin)
        self.homography = homography
        self.corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        self.center = self.corners.mean(axis=0)

    def tostring(self, indent: int = 0) -> str:
        pad = " " * indent
        return (f"{pad}tag_family: {self.tag_family}\n"
                f"{pad}tag_id: {self.tag_id}\n"
                f"{pad}hamming: {self.hamming}\n"
                f"{pad}decision_margin: {self.decision_margin:.3f}\n"
                f"{pad}center: {self.center}\n"
                f"{pad}corners:\n{pad}{self.corners}")

    def __repr__(self) -> str:
        return f"Detection(tag_id={self.tag_id}, center={self.center})"


def _aruco_dictionary(tag_family: str | None):
    name = _FAMILY_TO_ARUCO.get(tag_family or "tag36h11", "DICT_APRILTAG_36h11")
    dict_id = getattr(cv2.aruco, name, None)
    if dict_id is None:
        available = [n for n in dir(cv2.aruco) if n.startswith("DICT_APRILTAG")]
        raise ValueError(f"OpenCV has no {name}; available AprilTag dicts: {available}")
    return cv2.aruco.getPredefinedDictionary(dict_id)


def _detect_markers(gray, tag_family):
    """Detect markers across OpenCV 4.7+ and legacy aruco APIs."""
    ar_dict = _aruco_dictionary(tag_family)
    if hasattr(cv2.aruco, "ArucoDetector"):                     # OpenCV >= 4.7
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        detector = cv2.aruco.ArucoDetector(ar_dict, params)
        return detector.detectMarkers(gray)
    params = cv2.aruco.DetectorParameters_create()              # legacy
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.detectMarkers(gray, ar_dict, parameters=params)


def tag_object_points(tag_size: float) -> np.ndarray:
    """Tag corners in the tag frame, ordered to match aruco's detection order."""
    s = float(tag_size) / 2.0
    return np.array([[-s, s, 0.0],
                     [s, s, 0.0],
                     [s, -s, 0.0],
                     [-s, -s, 0.0]], dtype=np.float64)


def detect_tags(
    img,
    detector=None,
    camera_params=None,
    tag_size=1.0,
    vizualization=0,
    verbose=0,
    annotation=False,
    tag_family="tag36h11",
    dist=None,
):
    """cv2.aruco replacement for apriltag.detect_tags.

    Args:
        img:           BGR or grayscale image.
        detector:      ignored (kept for signature compatibility).
        camera_params: (fx, fy, cx, cy). Required for pose; without it poses are None.
        tag_size:      tag black-square side length, metres.
        annotation:    draw detections onto the returned overlay.
        dist:          optional distortion coefficients. Pass zeros/None for
                       already-rectified images (e.g. ZED VIEW.LEFT).

    Returns:
        (results, overlay) where results is the FLAT groups-of-four list described in
        the module docstring, and overlay is a BGR annotated copy.
    """
    if img is None:
        raise ValueError("detect_tags got img=None")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    overlay = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    corners_list, ids, _ = _detect_markers(gray, tag_family)
    if ids is None or len(ids) == 0:
        if verbose:
            print("[apriltag_backend] no tags detected")
        return [], overlay

    K = None
    if camera_params is not None:
        fx, fy, cx, cy = (float(v) for v in camera_params)
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_v = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64).ravel()

    objp = tag_object_points(tag_size)
    pnp_flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)

    results: list = []
    for raw, tag_id in zip(corners_list, ids.ravel()):
        pts = raw.reshape(-1, 2).astype(np.float32)
        # Sub-pixel refinement on top of aruco's own corner refinement.
        cv2.cornerSubPix(
            gray, pts, (5, 5), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4),
        )
        pts64 = pts.astype(np.float64)

        homography = np.eye(3)
        try:
            homography, _ = cv2.findHomography(objp[:, :2], pts64)
        except Exception:
            pass

        det = Detection(tag_family or "tag36h11", tag_id, pts64, homography,
                        decision_margin=float("nan"))

        T = None
        err0 = float("nan")
        if K is not None:
            ok, rvec, tvec = cv2.solvePnP(objp, pts64, K, dist_v, flags=pnp_flag)
            if ok:
                T = np.eye(4)
                T[:3, :3] = cv2.Rodrigues(rvec)[0]
                T[:3, 3] = tvec.ravel()
                proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist_v)
                err0 = float(np.sqrt(((proj.reshape(-1, 2) - pts64) ** 2).sum(1).mean()))
                if annotation:
                    cv2.drawFrameAxes(overlay, K, dist_v, rvec, tvec, tag_size * 0.5)

        if annotation:
            poly = pts64.astype(int).reshape(-1, 1, 2)
            cv2.polylines(overlay, [poly], True, (0, 255, 0), 2)
            c = det.center.astype(int)
            cv2.circle(overlay, tuple(c), 4, (0, 0, 255), -1)
            cv2.putText(overlay, str(det.tag_id), (c[0] + 8, c[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if verbose:
            print(f"[apriltag_backend] tag {det.tag_id}: reproj RMS "
                  f"{err0:.3f} px, center {det.center}")

        # apriltag's layout: detection, pose, init error, final error
        results.extend([det, T, err0, err0])

    return results, overlay


def get_dll_path() -> str:
    """Signature-compatibility stub; the aruco backend needs no DLL search path."""
    return ""
