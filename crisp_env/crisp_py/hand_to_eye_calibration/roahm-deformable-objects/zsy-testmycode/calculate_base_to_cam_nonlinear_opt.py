#!/usr/bin/env python3

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The ZED depth disparity-offset correction is shared with the capture path. Its
# value lives in zed_capture/zed_depth_correction.json so there is one source of
# truth. zed_depth_config needs only numpy, so importing it here is safe.
_ZED_CAPTURE_DIR = REPO_ROOT.parents[1] / "zed_capture"
if str(_ZED_CAPTURE_DIR) not in sys.path:
    sys.path.insert(0, str(_ZED_CAPTURE_DIR))
try:
    import zed_depth_config
except ImportError as exc:  # loud on purpose: a silent skip costs 220 mm of accuracy
    raise ImportError(
        f"Cannot import zed_depth_config from {_ZED_CAPTURE_DIR}. The ZED depth "
        "correction is required for --camera zed --use-depth-translation; without "
        "it the depth reads ~15% too far and the calibration is wrong by ~220 mm. "
        "Set $ZED_DEPTH_CORRECTION_JSON and put zed_depth_config.py beside it, or "
        "restore zed_capture/."
    ) from exc

import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from apriltag_image import apriltag_image


# ============================================================
# Azure color camera intrinsics
# Resolution: 1280 x 720
# ============================================================

AZURE_FX = 606.112427
AZURE_FY = 605.882141
AZURE_CX = 641.757812
AZURE_CY = 365.651886

DATAPATH = (
    "/home/yizhouch/move_some_robots/crisp_env/crisp_py/"
    "hand_to_eye_calibration/roahm-deformable-objects"
)

EXPECTED_TAG_ID = 3
TAG_SIZE_M = 0.0950   # MEASURED with calipers 2026-08-19 (outer black square edge, re-verified).
                     # The mount was long assumed 0.093; the print is 3.2% larger,
                     # which scaled every tag-derived range short by the same factor
                     # and biased the solved extrinsics ~50 mm along the viewing ray.
                     # zed_depth_correction.json's disparity offset was re-derived
                     # under this size the same day -- change them TOGETHER or the
                     # depth and tag rulers disagree again.

# The OUTLIER rule of the residual report. Also what the worst-case ranking
# normalizes by, so "worst" and "OUTLIER" always mean the same thing.
TRANS_THRESH_MM = 30.0
ROT_THRESH_DEG = 8.0


# ============================================================
# SE(3) utilities
# ============================================================

def _skew(w):
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ])


def _se3_log(T, eps=1e-9):
    R = T[0:3, 0:3]
    p = T[0:3, 3]

    w = Rotation.from_matrix(R).as_rotvec()
    theta = np.linalg.norm(w)
    W = _skew(w)

    if theta < eps:
        V_inv = np.eye(3) - 0.5 * W + (1.0 / 12.0) * (W @ W)
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / (theta * theta)
        V_inv = (
            np.eye(3)
            - 0.5 * W
            + (1.0 / (theta * theta)) * (1.0 - A / (2.0 * B)) * (W @ W)
        )

    v = V_inv @ p
    return np.hstack([w, v])


def _se3_exp(xi, eps=1e-9):
    w = xi[0:3]
    v = xi[3:6]

    theta = np.linalg.norm(w)
    W = _skew(w)

    if theta < eps:
        R = np.eye(3) + W + 0.5 * (W @ W)
        V = np.eye(3) + 0.5 * W + (1.0 / 6.0) * (W @ W)
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / (theta * theta)
        C = (1.0 - A) / (theta * theta)

        R = np.eye(3) + A * W + B * (W @ W)
        V = np.eye(3) + B * W + C * (W @ W)

    p = V @ v

    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = p
    return T


def _invert_T(T):
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    out = np.eye(4)
    out[0:3, 0:3] = R.T
    out[0:3, 3] = -R.T @ t
    return out


def _mean_se3(transforms, max_iters=100, tol=1e-9):
    if len(transforms) == 0:
        raise ValueError("Cannot compute SE3 mean of empty transform list.")

    translations = np.array([T[0:3, 3] for T in transforms])
    median_t = np.median(translations, axis=0)
    dists = np.linalg.norm(translations - median_t[None, :], axis=1)
    init_idx = int(np.argmin(dists))

    T_mean = transforms[init_idx].copy()
    print(f"[INFO] SE3 mean init index: {init_idx}, closest to median translation")

    for it in range(max_iters):
        xi_sum = np.zeros(6)

        for T_i in transforms:
            xi_sum += _se3_log(np.linalg.inv(T_mean) @ T_i)

        xi_avg = xi_sum / len(transforms)

        if np.linalg.norm(xi_avg) < tol:
            print(f"[INFO] SE3 mean converged at iter {it}")
            break

        T_mean = T_mean @ _se3_exp(xi_avg)

    return T_mean


# ============================================================
# Image enhancement utilities
# ============================================================

def _gamma_bgr(img_bgr, gamma):
    img_float = img_bgr.astype(np.float32) / 255.0
    out = np.power(img_float, gamma)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out


def _clahe_bgr(img_bgr, clip=5.0, tile=(8, 8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def _contrast_brightness(img_bgr, alpha=2.0, beta=60):
    return cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)


def _make_enhanced_versions(img_path, out_dir, image_index):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    versions = [("original", img_path)]

    enhanced_specs = []
    enhanced_specs.append(("L1_clahe3", _clahe_bgr(img, clip=3.0, tile=(8, 8))))
    enhanced_specs.append(("L1_gamma0p60", _gamma_bgr(img, gamma=0.60)))
    enhanced_specs.append(("L1_alpha1p8_beta45", _contrast_brightness(img, alpha=1.8, beta=45)))

    e_clahe5 = _clahe_bgr(img, clip=5.0, tile=(8, 8))
    enhanced_specs.append(("L2_clahe5", e_clahe5))
    enhanced_specs.append(("L2_clahe5_gamma0p50", _gamma_bgr(e_clahe5, gamma=0.50)))
    enhanced_specs.append(("L2_alpha2p5_beta80", _contrast_brightness(img, alpha=2.5, beta=80)))

    e_clahe8 = _clahe_bgr(img, clip=8.0, tile=(4, 4))
    enhanced_specs.append(("L3_clahe8_tile4", e_clahe8))
    enhanced_specs.append(("L3_clahe8_gamma0p35", _gamma_bgr(e_clahe8, gamma=0.35)))
    enhanced_specs.append(("L3_alpha3p5_beta120", _contrast_brightness(img, alpha=3.5, beta=120)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    enhanced_specs.append(("L4_gray_equalized", cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)))

    gray_clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4)).apply(gray)
    enhanced_specs.append(("L4_gray_clahe8", cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)))

    for name, enh in enhanced_specs:
        out_path = out_dir / f"image_{image_index:03d}_{name}.png"
        cv2.imwrite(str(out_path), enh)
        versions.append((name, out_path))

    return versions


# ============================================================
# AprilTag detection
# ============================================================

def _find_expected_tag(detections, expected_tag_id=EXPECTED_TAG_ID):
    if detections is None or len(detections) == 0:
        return None, None

    found_detection = None
    found_transform = None

    for j in range(0, len(detections), 4):
        det = detections[j]

        if det.tag_id == expected_tag_id:
            if found_detection is not None:
                raise RuntimeError(
                    f"Detected multiple tag_id={expected_tag_id} tags in the same image."
                )

            found_detection = det
            found_transform = np.asarray(detections[j + 1]).copy()

    return found_detection, found_transform


def _detect_with_enhancement_until_success(img_path, enhanced_dir, image_index, camera,
                                           enhance=True):
    """Try the original image, then progressively enhanced variants until one detects.

    enhance=False tries ONLY the original. The cascade costs 12 detections per failing
    frame at full resolution, and on overexposed frames the aggressive CLAHE variants
    amplify sensor noise until the detector aborts with "too many borders in
    contour_detect", so it is often pure cost. Failing frames are skipped either way --
    detection failure never aborts the run.
    """
    if enhance:
        versions = _make_enhanced_versions(img_path, enhanced_dir, image_index)
    else:
        versions = [("original", img_path)]

    for source_name, path in versions:
        detections = apriltag_image(
            [str(path)],
            output_images=False,
            display_images=False,
            tag_size=TAG_SIZE_M,
            tag_family="tag36h11",
            camera=camera,
        )

        det, T = _find_expected_tag(detections, expected_tag_id=EXPECTED_TAG_ID)

        if det is not None and T is not None:
            return det, T, source_name, path

    return None, None, None, None


# ============================================================
# Depth utilities
# ============================================================

def _rgbd_npz_path(calib_base_dir, side, rgbd_file=None):
    """The rgbd npz this run reads its depth from.

    Default is the capture's own {side}_calibration_rgbd.npz; --rgbd-file selects
    an alternative in the same directory and format (e.g. the FoundationStereo
    re-matched right_calibration_rgbd_fs.npz from fs_depth_batch.py), so the same
    sequence can be solved once per depth source and the residuals compared.
    """
    return calib_base_dir / (rgbd_file or f"{side}_calibration_rgbd.npz")


def _load_depth_stack(calib_base_dir, side, rgbd_file=None):
    rgbd_path = _rgbd_npz_path(calib_base_dir, side, rgbd_file)

    if not rgbd_path.exists():
        print(f"[WARN] RGB-D file not found: {rgbd_path}")
        return None

    data = np.load(rgbd_path)

    if "depth" not in data.files:
        print(f"[WARN] No depth key in {rgbd_path}. Keys are: {data.files}")
        return None

    depth = data["depth"]
    print(f"[INFO] Loaded depth stack: {rgbd_path}, shape={depth.shape}")
    return depth


# ZED depth disparity-offset correction. Identity until main() installs a real one,
# so an Azure run and a direct import both behave exactly as before.
_DEPTH_CORRECTOR = zed_depth_config.disabled_corrector("not configured yet")


def set_depth_corrector(corrector):
    """Install the depth corrector that _depth_at_pixel applies.

    Call this AFTER set_depth_intrinsics, because the correction needs fx.
    """
    global _DEPTH_CORRECTOR
    _DEPTH_CORRECTOR = corrector


def _depth_at_pixel(depth_img, u, v, patch_radius=5):
    h, w = depth_img.shape[:2]

    u = int(round(u))
    v = int(round(v))

    u0 = max(0, u - patch_radius)
    u1 = min(w, u + patch_radius + 1)
    v0 = max(0, v - patch_radius)
    v1 = min(h, v + patch_radius + 1)

    patch = depth_img[v0:v1, u0:u1]
    valid = patch[patch > 0]

    if valid.size == 0:
        return None

    # Correcting the median is identical to correcting every pixel and then taking
    # the median: the disparity-offset map is strictly monotonic in z, and a
    # monotonic map commutes with the median. This way we touch one scalar per
    # frame instead of a (N, 1242, 2208) array.
    return _DEPTH_CORRECTOR(float(np.median(valid)))


# Intrinsics used for DEPTH back-projection. Defaults to the Azure constants above so
# existing --camera azure runs are bit-identical; set_depth_intrinsics() overrides them
# for other cameras (see the --camera zed branch in main()).
_DEPTH_INTR = {"fx": AZURE_FX, "fy": AZURE_FY, "cx": AZURE_CX, "cy": AZURE_CY}


def set_depth_intrinsics(fx, fy, cx, cy):
    """Override the intrinsics used to unproject depth pixels into camera-frame 3D.

    These MUST correspond to the resolution of the recorded depth/color frames, and
    for a ZED they must be the RECTIFIED left-camera values (VIEW.LEFT is rectified,
    so distortion is zero and no undistortion step is needed).
    """
    _DEPTH_INTR.update(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))
    print(f"[INFO] depth back-projection intrinsics: fx={fx:.6f} fy={fy:.6f} "
          f"cx={cx:.6f} cy={cy:.6f}")


def _pixel_depth_to_camera_point(u, v, depth_mm):
    Z = depth_mm / 1000.0
    X = (float(u) - _DEPTH_INTR["cx"]) * Z / _DEPTH_INTR["fx"]
    Y = (float(v) - _DEPTH_INTR["cy"]) * Z / _DEPTH_INTR["fy"]
    return np.array([X, Y, Z], dtype=float)


# ============================================================
# Robot pose loading
# ============================================================

def _load_robot_poses(npz_path, max_images, exclude_image_indices=None):
    if exclude_image_indices is None:
        exclude_image_indices = set()

    if not Path(npz_path).exists():
        raise RuntimeError(f"Pose file not found: {npz_path}")

    data = np.load(npz_path)
    positions = []
    rotations = []

    for i in range(max_images):
        if i in exclude_image_indices:
            positions.append(None)
            rotations.append(None)
            continue

        key = "arr_" + str(i)

        if key not in data:
            print(f"[WARN] Pose key {key} not found in {npz_path}")
            positions.append(None)
            rotations.append(None)
            continue

        arr = data[key]
        positions.append(arr[0:3])
        rotations.append(Rotation.from_quat(arr[3:7]).as_matrix())

    return positions, rotations


# ============================================================
# Fixed gripper -> AprilTag transform
# ============================================================

gripper2tag = np.array([
    [0, 0, -1, 0.02],#ORIGINALLY -0.02!!!
    [0, -1, 0, 0],
    [-1, 0, 0, 0.0892],# adjusted 8/19 (originally 0.0905 -> 0.1055)
    [0, 0, 0, 1],
])


# ============================================================
# Build calibration pairs
# ===========================================================

def _build_calibration_pairs(
    max_images,
    image_dir,
    pose_file,
    calib_base_dir,
    side,
    camera,
    use_depth_translation=False,
    depth_patch_radius=5,
    exclude_image_indices=None,
    enhanced_dir=None,
    enhance=True,
    rgbd_file=None,
):
    if exclude_image_indices is None:
        exclude_image_indices = set()

    if enhanced_dir is None:
        enhanced_dir = Path(image_dir) / "nonlinear_enhanced_debug"

    depth_stack = None
    if use_depth_translation:
        depth_stack = _load_depth_stack(calib_base_dir, side, rgbd_file)
        if depth_stack is None:
            raise RuntimeError("--use-depth-translation was requested, but no depth stack is available.")

    positions, rotations = _load_robot_poses(
        pose_file,
        max_images=max_images,
        exclude_image_indices=exclude_image_indices,
    )

    T_cam_tag_list = []
    T_base_tag_list = []
    valid_indices = []
    source_names = []
    # Per-frame detection detail, kept so the worst-case dump can redraw a frame
    # without re-running the detector. Keyed by image_index in _save_worst_cases.
    frame_infos = []

    count_no_detection = 0
    count_no_depth = 0

    for image_index in range(max_images):
        if image_index in exclude_image_indices:
            continue

        img_path = Path(image_dir) / f"calibration_{side}_image_{image_index}.png"

        if not img_path.exists():
            print(f"[WARN] image {image_index}: missing {img_path}")
            count_no_detection += 1
            continue

        if positions[image_index] is None or rotations[image_index] is None:
            print(f"[WARN] image {image_index}: missing robot pose")
            continue

        det, T_cam_tag_apriltag, source_name, used_img_path = _detect_with_enhancement_until_success(
            img_path=img_path,
            enhanced_dir=enhanced_dir,
            image_index=image_index,
            camera=camera,
            enhance=enhance,
        )

        if det is None or T_cam_tag_apriltag is None:
            print(f"[WARN] image {image_index}: no AprilTag detection even after enhancement")
            count_no_detection += 1
            continue

        T_cam_tag = T_cam_tag_apriltag.copy()
        depth_mm = None

        if use_depth_translation:
            if image_index >= depth_stack.shape[0]:
                print(f"[WARN] image {image_index}: depth stack too short")
                count_no_depth += 1
                continue

            center = np.asarray(det.center).astype(float)
            u, v = float(center[0]), float(center[1])

            depth_mm = _depth_at_pixel(
                depth_stack[image_index],
                u,
                v,
                patch_radius=depth_patch_radius,
            )

            if depth_mm is None:
                print(f"[WARN] image {image_index}: no valid depth at tag center")
                count_no_depth += 1
                continue

            R_cam_tag = T_cam_tag_apriltag[0:3, 0:3]
            t_apriltag = T_cam_tag_apriltag[0:3, 3]
            t_depth = _pixel_depth_to_camera_point(u, v, depth_mm)

            T_cam_tag = np.eye(4)
            T_cam_tag[0:3, 0:3] = R_cam_tag
            T_cam_tag[0:3, 3] = t_depth

            diff_mm = np.linalg.norm(t_depth - t_apriltag) * 1000.0
            print(
                f"[DEPTH] image {image_index}: source={source_name}, "
                f"depth={depth_mm:.1f} mm, |t_depth - t_apriltag|={diff_mm:.1f} mm"
            )
        else:
            print(f"[TAG] image {image_index}: source={source_name}")

        T_base_gripper = np.eye(4)
        T_base_gripper[0:3, 0:3] = rotations[image_index]
        T_base_gripper[0:3, 3] = positions[image_index]

        T_base_tag = T_base_gripper @ gripper2tag

        T_cam_tag_list.append(T_cam_tag)
        T_base_tag_list.append(T_base_tag)
        valid_indices.append(image_index)
        source_names.append(source_name)
        frame_infos.append({
            "image_index": image_index,
            "img_path": img_path,
            "used_img_path": used_img_path,
            "source_name": source_name,
            "center_uv": np.asarray(det.center, dtype=float).copy(),
            "corners": np.asarray(det.corners, dtype=float).copy(),
            "depth_mm": depth_mm,
            "T_cam_tag_apriltag": T_cam_tag_apriltag.copy(),
        })

    print("\n[INFO] Build pairs summary")
    print("  valid pairs      :", len(valid_indices))
    print("  no detection     :", count_no_detection)
    print("  no valid depth   :", count_no_depth)
    print("  valid image ids  :", valid_indices)

    if len(valid_indices) < 4:
        raise RuntimeError(f"Too few valid calibration pairs: {len(valid_indices)}")

    return T_cam_tag_list, T_base_tag_list, valid_indices, source_names, frame_infos


# ============================================================
# Reports
# ============================================================

def _pose_error_rows(X_cam_base, T_cam_tag_list, T_base_tag_list, valid_indices):
    rows = []

    for idx, T_cam_tag, T_base_tag in zip(valid_indices, T_cam_tag_list, T_base_tag_list):
        T_pred = X_cam_base @ T_base_tag
        T_err = np.linalg.inv(T_cam_tag) @ T_pred
        xi = _se3_log(T_err)

        rot_err_deg = np.rad2deg(np.linalg.norm(xi[0:3]))
        trans_err_mm = np.linalg.norm(xi[3:6]) * 1000.0

        rows.append({
            "image_index": idx,
            "rot_err_deg": rot_err_deg,
            "trans_err_mm": trans_err_mm,
            "xi": xi,
        })

    return rows


def _error_stats_lines(rows, outliers):
    """The all-frames / inliers / outliers error block, as a list of lines.

    Built once and used twice: printed to the console and written into the summary
    txt, so the two can never disagree.
    """
    trans = np.array([r["trans_err_mm"] for r in rows])
    rot = np.array([r["rot_err_deg"] for r in rows])

    lines = [
        f"Translation error mean/std/max: "
        f"{np.mean(trans):.2f} / {np.std(trans):.2f} / {np.max(trans):.2f} mm"
        f"   [ALL {len(rows)} frames]",
        f"Rotation error mean/std/max: "
        f"{np.mean(rot):.3f} / {np.std(rot):.3f} / {np.max(rot):.3f} deg"
        f"   [ALL {len(rows)} frames]",
    ]

    # The two lines above include the outliers, so they answer "how good is the whole
    # set". The lines below split the two groups, which answers "how good are the
    # frames the rule accepted" and "how bad are the ones it rejected".
    # NOTE: no frame is removed from the fit. The OUTLIER label is diagnostic only.
    # Use --exclude-images to actually drop frames.
    bad_idx = set(outliers)
    keep = np.array([r["image_index"] not in bad_idx for r in rows], dtype=bool)
    if keep.any() and not keep.all():
        for label, mask in (("inliers ", keep), ("outliers", ~keep)):
            lines.append(
                f"  {label} ({int(mask.sum()):>3} frames): "
                f"trans {np.mean(trans[mask]):>7.2f} / {np.std(trans[mask]):>6.2f} / "
                f"{np.max(trans[mask]):>7.2f} mm    "
                f"rot {np.mean(rot[mask]):>6.3f} / {np.std(rot[mask]):>5.3f} / "
                f"{np.max(rot[mask]):>6.3f} deg"
            )

    return lines


def _print_report(title, rows, trans_thresh_mm=TRANS_THRESH_MM,
                  rot_thresh_deg=ROT_THRESH_DEG):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print("image_index | trans_err_mm | rot_err_deg | status")
    print("-" * 90)

    outliers = []

    for row in rows:
        idx = row["image_index"]
        te = row["trans_err_mm"]
        re = row["rot_err_deg"]
        is_outlier = te > trans_thresh_mm or re > rot_thresh_deg
        status = "OUTLIER" if is_outlier else "ok"

        if is_outlier:
            outliers.append(idx)

        print(f"{idx:>11} | {te:>12.2f} | {re:>11.3f} | {status}")

    stats_lines = _error_stats_lines(rows, outliers)

    print("-" * 90)
    for line in stats_lines:
        print(line)

    if outliers:
        print("[WARNING] outliers:", outliers)
    else:
        print("[INFO] no outliers by this rule")

    return outliers, stats_lines


# ============================================================
# Error distribution plot
# ============================================================

def _plot_error_distribution(rows_init, rows_opt, outliers, out_path, show=True,
                             title_suffix="",
                             trans_thresh_mm=TRANS_THRESH_MM,
                             rot_thresh_deg=ROT_THRESH_DEG):
    """Histogram + per-frame view of the residuals. Returns the saved path, or None.

    Always writes the PNG; only pops a window when a display is actually there,
    so this is safe over SSH. matplotlib is imported here, not at module scope,
    because the solver must keep running on a box that has no matplotlib.
    """
    try:
        import matplotlib
    except ImportError:
        print("[WARN] --plot-errors needs matplotlib, which is not installed. "
              "Skipping the plot.")
        return None

    headless = not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.array([r["image_index"] for r in rows_opt])
    trans_opt = np.array([r["trans_err_mm"] for r in rows_opt])
    rot_opt = np.array([r["rot_err_deg"] for r in rows_opt])
    trans_init = np.array([r["trans_err_mm"] for r in rows_init])
    rot_init = np.array([r["rot_err_deg"] for r in rows_init])

    bad = np.array([i in set(outliers) for i in idx], dtype=bool)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Calibration residual distribution{title_suffix}")

    for ax, init, opt, unit, thresh, name in (
        (axes[0, 0], trans_init, trans_opt, "mm", trans_thresh_mm, "Translation"),
        (axes[0, 1], rot_init, rot_opt, "deg", rot_thresh_deg, "Rotation"),
    ):
        bins = np.histogram_bin_edges(np.concatenate([init, opt]), bins=20)
        ax.hist(init, bins=bins, color="0.75", label="SE3-mean init")
        ax.hist(opt, bins=bins, color="tab:blue", alpha=0.75, label="optimized")
        ax.axvline(np.mean(opt), color="tab:green", ls="-",
                   label=f"opt mean {np.mean(opt):.2f}")
        ax.axvline(np.median(opt), color="tab:green", ls=":",
                   label=f"opt median {np.median(opt):.2f}")
        ax.axvline(thresh, color="tab:red", ls="--", label=f"outlier {thresh:g}")
        ax.set_xlabel(f"{name} error [{unit}]")
        ax.set_ylabel("frames")
        ax.set_title(f"{name} error histogram")
        ax.legend(fontsize=7)

    for ax, opt, unit, thresh, name in (
        (axes[1, 0], trans_opt, "mm", trans_thresh_mm, "Translation"),
        (axes[1, 1], rot_opt, "deg", rot_thresh_deg, "Rotation"),
    ):
        ax.bar(idx[~bad], opt[~bad], color="tab:blue", label="ok")
        if bad.any():
            ax.bar(idx[bad], opt[bad], color="tab:red", label="OUTLIER")
            for i, value in zip(idx[bad], opt[bad]):
                ax.annotate(str(i), (i, value), fontsize=7,
                            ha="center", va="bottom")
        ax.axhline(thresh, color="tab:red", ls="--", lw=1)
        ax.set_xlabel("image index")
        ax.set_ylabel(f"{name.lower()} error [{unit}]")
        ax.set_title(f"{name} error per frame (optimized)")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"[PLOT] error distribution saved to: {out_path}")

    if show:
        if headless:
            print("[PLOT] no DISPLAY, so the window was skipped. Open the PNG above.")
        else:
            plt.show()
    plt.close(fig)

    return out_path


# ============================================================
# Worst-case dump (image + point cloud)
# ============================================================

def _write_ply(path, xyz, rgb):
    """Binary little-endian PLY with per-point colour. xyz in metres, rgb uint8."""
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)

    verts = np.empty(
        len(xyz),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment frame: camera optical frame of the calibrated camera\n"
        "comment markers: measured tag pose = RGB axes, predicted tag pose = "
        "magenta/yellow/cyan axes, error vector = white\n"
        f"element vertex {len(verts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )

    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(verts.tobytes())


def _line_points(p0, p1, n=60):
    """n points along the segment p0->p1, for drawing markers in a point cloud."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    return np.asarray(p0, dtype=float)[None, :] * (1 - t) + np.asarray(p1, dtype=float)[None, :] * t


def _pose_axis_points(T, length=0.05, n=60):
    """Three axis segments of a pose, as (points, colours) with x/y/z first."""
    origin = T[0:3, 3]
    pts = [_line_points(origin, origin + T[0:3, k] * length, n) for k in range(3)]
    return pts


def _cloud_from_depth(depth_img_mm, color_bgr, stride=2, z_range_m=(0.05, 8.0)):
    """Back-project one depth frame into a coloured camera-frame cloud.

    Uses the same intrinsics and the same depth corrector as the calibration, so
    the cloud is in the exact geometry the transform was solved in.
    """
    depth = depth_img_mm[::stride, ::stride].astype(np.float32)
    depth = np.asarray(_DEPTH_CORRECTOR(depth), dtype=np.float32)

    h, w = depth.shape
    us = np.arange(0, depth_img_mm.shape[1], stride)[:w]
    vs = np.arange(0, depth_img_mm.shape[0], stride)[:h]
    uu, vv = np.meshgrid(us.astype(np.float32), vs.astype(np.float32))

    z = depth / 1000.0
    valid = np.isfinite(z) & (z > z_range_m[0]) & (z < z_range_m[1])

    z = z[valid]
    x = (uu[valid] - _DEPTH_INTR["cx"]) * z / _DEPTH_INTR["fx"]
    y = (vv[valid] - _DEPTH_INTR["cy"]) * z / _DEPTH_INTR["fy"]
    xyz = np.stack([x, y, z], axis=1)

    if color_bgr is not None:
        rgb = color_bgr[::stride, ::stride][:h, :w][valid][:, ::-1]  # BGR -> RGB
    else:
        rgb = np.full((len(xyz), 3), 200, dtype=np.uint8)

    return xyz, np.ascontiguousarray(rgb, dtype=np.uint8)


def _project(p_cam, intr):
    """Pinhole projection of a camera-frame point. Returns (u, v) or None if behind."""
    if p_cam[2] <= 1e-6:
        return None
    fx, fy, cx, cy = intr
    return (float(p_cam[0] * fx / p_cam[2] + cx), float(p_cam[1] * fy / p_cam[2] + cy))


# Axis colours, BGR. The two poses get DIFFERENT colour families on purpose: with
# one family for both, the picture is unreadable, and a marker glyph in an axis
# colour gets mistaken for an axis (a 40 px red '+' at the predicted origin once
# read as a 90 deg rotation error on a frame whose true error was 15.9 deg).
_AXIS_COLOURS_MEASURED = ((0, 0, 255), (0, 255, 0), (255, 0, 0))       # r / g / b
_AXIS_COLOURS_PREDICTED = ((255, 0, 255), (0, 255, 255), (255, 255, 0))  # m / y / c
_AXIS_LEN_M = 0.05


def _axis_angle_report(T_cam_tag, T_pred, intr):
    """Per-axis angle between the two tag poses, in 3D and as drawn on screen.

    They differ, sometimes wildly: an axis pointing near the optical axis projects
    to a few pixels, so a small 3D tilt can flip its on-screen direction by ~180
    deg. Reporting both stops the picture from being read as an error it is not.
    """
    deg_3d, deg_screen = [], []

    for k in range(3):
        cos3d = float(np.clip(T_cam_tag[0:3, k] @ T_pred[0:3, k], -1.0, 1.0))
        deg_3d.append(np.rad2deg(np.arccos(cos3d)))

        d = []
        for T in (T_cam_tag, T_pred):
            o = _project(T[0:3, 3], intr)
            tip = _project(T[0:3, 3] + T[0:3, k] * _AXIS_LEN_M, intr)
            d.append(None if (o is None or tip is None)
                     else np.array([tip[0] - o[0], tip[1] - o[1]]))

        if d[0] is None or d[1] is None or min(np.linalg.norm(d[0]), np.linalg.norm(d[1])) < 1e-6:
            deg_screen.append(float("nan"))
        else:
            cos2d = float(np.clip(d[0] @ d[1] / (np.linalg.norm(d[0]) * np.linalg.norm(d[1])),
                                  -1.0, 1.0))
            deg_screen.append(np.rad2deg(np.arccos(cos2d)))

    return deg_3d, deg_screen


def _draw_worst_case_image(img, info, row, T_cam_tag, T_pred, intr, rank):
    """Frame annotated with the detected tag and where the calibration says it is."""
    img = img.copy()
    corners = info["corners"].astype(np.int32)  # cv2 wants CV_32S, not int64
    # White, not green: green is the measured y-axis colour.
    cv2.polylines(img, [corners.reshape(-1, 1, 2)], True, (255, 255, 255), 2)

    uv_meas = _project(T_cam_tag[0:3, 3], intr)
    uv_pred = _project(T_pred[0:3, 3], intr)

    # Origins are neutral glyphs, never an axis colour: filled dot = measured,
    # hollow ring = predicted. The colour coding belongs to the axes alone.
    if uv_meas is not None:
        cv2.circle(img, (int(uv_meas[0]), int(uv_meas[1])), 5, (255, 255, 255), -1)
    if uv_pred is not None:
        cv2.circle(img, (int(uv_pred[0]), int(uv_pred[1])), 11, (255, 255, 255), 2)
    if uv_meas is not None and uv_pred is not None:
        cv2.line(img, (int(uv_meas[0]), int(uv_meas[1])),
                 (int(uv_pred[0]), int(uv_pred[1])), (255, 255, 255), 1)

    for T, colours, thickness in ((T_cam_tag, _AXIS_COLOURS_MEASURED, 4),
                                  (T_pred, _AXIS_COLOURS_PREDICTED, 3)):
        origin = _project(T[0:3, 3], intr)
        if origin is None:
            continue
        for k, colour in enumerate(colours):
            tip = _project(T[0:3, 3] + T[0:3, k] * _AXIS_LEN_M, intr)
            if tip is None:
                continue
            cv2.line(img, (int(origin[0]), int(origin[1])),
                     (int(tip[0]), int(tip[1])), colour, thickness)

    deg_3d, deg_screen = _axis_angle_report(T_cam_tag, T_pred, intr)

    # The tag is ~100 px wide in a 2208 px frame, so the overlay is unreadable at
    # full view. Paste a zoomed crop of the tag neighbourhood into the top right.
    points = [corners.astype(float)]
    for uv in (uv_meas, uv_pred):
        if uv is not None:
            points.append(np.array([uv], dtype=float))
    points = np.vstack(points)

    h_img, w_img = img.shape[:2]
    pad = max(60.0, 0.8 * float(np.ptp(points, axis=0).max()))
    x0 = int(max(0, points[:, 0].min() - pad))
    x1 = int(min(w_img, points[:, 0].max() + pad))
    y0 = int(max(0, points[:, 1].min() - pad))
    y1 = int(min(h_img, points[:, 1].max() + pad))

    if x1 - x0 > 10 and y1 - y0 > 10:
        crop = img[y0:y1, x0:x1]
        scale = min(3.0, (w_img / 3.0) / crop.shape[1], (h_img / 2.0) / crop.shape[0])
        if scale > 1.0:
            zoom = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_NEAREST)
            zh, zw = zoom.shape[:2]
            ox, oy = w_img - zw - 20, 20
            img[oy:oy + zh, ox:ox + zw] = zoom
            cv2.rectangle(img, (ox - 2, oy - 2), (ox + zw + 2, oy + zh + 2),
                          (255, 255, 255), 3)
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 2)
            cv2.putText(img, f"zoom x{scale:.1f}", (ox + 10, oy + zh - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    depth_txt = "n/a" if info["depth_mm"] is None else f"{info['depth_mm']:.1f} mm"
    lines = [
        f"worst #{rank}  image {info['image_index']}  source={info['source_name']}",
        f"trans err {row['trans_err_mm']:.2f} mm   rot err {row['rot_err_deg']:.3f} deg",
        f"tag center depth: {depth_txt}",
        "measured (AprilTag): x/y/z = red/green/blue, dot = origin",
        "predicted (calib)  : x/y/z = magenta/yellow/cyan, ring = origin",
        f"per-axis angle, 3D      : x {deg_3d[0]:5.1f}  y {deg_3d[1]:5.1f}  z {deg_3d[2]:5.1f} deg",
        f"per-axis angle, as drawn: x {deg_screen[0]:5.1f}  y {deg_screen[1]:5.1f}  z {deg_screen[2]:5.1f} deg",
        "an axis pointing near the optical axis is a few px long, so its",
        "drawn direction is not the error -- trust the 3D row.",
    ]
    for i, text in enumerate(lines):
        y = 40 + i * 34
        cv2.putText(img, text, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, text, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)

    return img


def _save_worst_cases(
    out_dir,
    rows,
    frame_infos,
    T_cam_tag_list,
    T_base_tag_list,
    valid_indices,
    X_opt,
    color_intr,
    depth_stack=None,
    n_worst=1,
    metric="combined",
    cloud_stride=2,
    tag="",
):
    """Dump the n_worst frames: annotated image, coloured point cloud, and a txt.

    "Worst" ranks by threshold-normalized error (max of trans/TRANS_THRESH_MM and
    rot/ROT_THRESH_DEG) so a large rotation error is not hidden behind millimetres,
    unless --worst-metric picks one of the two directly.
    """
    if n_worst <= 0 or not rows:
        return []

    info_by_index = {info["image_index"]: info for info in frame_infos}
    pair_by_index = {
        idx: (T_cam, T_base)
        for idx, T_cam, T_base in zip(valid_indices, T_cam_tag_list, T_base_tag_list)
    }

    def score(row):
        if metric == "trans":
            return row["trans_err_mm"]
        if metric == "rot":
            return row["rot_err_deg"]
        return max(row["trans_err_mm"] / TRANS_THRESH_MM,
                   row["rot_err_deg"] / ROT_THRESH_DEG)

    ranked = sorted(rows, key=score, reverse=True)[:n_worst]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    print(f"\n[WORST] saving {len(ranked)} worst frame(s) by metric={metric} to {out_dir}")

    # Drop this side+mode's files from an earlier run first. A run with a smaller
    # --save-worst-case leaves the previous run's higher ranks behind, and those
    # stale files claim to be "worst #2/#3" of a solve that no longer exists.
    # Scoped to this tag, so the other side's dump is never touched.
    if tag:
        for stale in sorted(out_dir.glob(f"worst*_{tag}_image*")):
            stale.unlink()
            print(f"  removed stale {stale.name}")

    for rank, row in enumerate(ranked, start=1):
        idx = row["image_index"]
        info = info_by_index.get(idx)
        T_cam_tag, T_base_tag = pair_by_index[idx]
        T_pred = X_opt @ T_base_tag

        # tag carries side + translation mode: left and right runs of one sequence
        # share this directory, and without it the second run silently overwrites
        # the first.
        stem = f"worst{rank}_{tag}_image{idx:03d}" if tag else f"worst{rank}_image{idx:03d}"

        # The frame png is the same array the capture wrote into the rgbd npz, so
        # it colours the cloud without loading a ~600 MB colour stack into RAM.
        img_bgr = None
        if info is not None:
            img_bgr = cv2.imread(str(info["img_path"]))
            if img_bgr is None:
                print(f"  [WARN] could not read {info['img_path']}")

        if img_bgr is not None:
            annotated = _draw_worst_case_image(img_bgr, info, row, T_cam_tag, T_pred,
                                               color_intr, rank)
            img_path = out_dir / f"{stem}_annotated.png"
            cv2.imwrite(str(img_path), annotated)
            written.append(img_path)
            print(f"  {img_path.name}")

        ply_path = None
        if depth_stack is not None and idx < depth_stack.shape[0]:
            color_bgr = img_bgr
            if color_bgr is not None and color_bgr.shape[0:2] != depth_stack.shape[1:3]:
                print(f"  [WARN] colour {color_bgr.shape[0:2]} does not match "
                      f"depth {depth_stack.shape[1:3]}; cloud will be grey")
                color_bgr = None

            xyz, rgb = _cloud_from_depth(depth_stack[idx], color_bgr,
                                         stride=cloud_stride)

            marker_xyz = []
            marker_rgb = []
            # Measured tag pose: RGB axes. Predicted: magenta/yellow/cyan.
            for T, colours in (
                (T_cam_tag, ((255, 0, 0), (0, 255, 0), (0, 0, 255))),
                (T_pred, ((255, 0, 255), (255, 255, 0), (0, 255, 255))),
            ):
                for pts, colour in zip(_pose_axis_points(T), colours):
                    marker_xyz.append(pts)
                    marker_rgb.append(np.tile(colour, (len(pts), 1)))
            err_pts = _line_points(T_cam_tag[0:3, 3], T_pred[0:3, 3], 120)
            marker_xyz.append(err_pts)
            marker_rgb.append(np.tile((255, 255, 255), (len(err_pts), 1)))

            xyz = np.vstack([xyz] + marker_xyz)
            rgb = np.vstack([rgb] + [np.asarray(c, dtype=np.uint8) for c in marker_rgb])

            ply_path = out_dir / f"{stem}_cloud.ply"
            _write_ply(ply_path, xyz, rgb)
            written.append(ply_path)
            print(f"  {ply_path.name}  ({len(xyz)} points, stride={cloud_stride})")
        else:
            print(f"  [WARN] no depth frame for image {idx}; point cloud skipped")

        txt_path = out_dir / f"{stem}_info.txt"
        with open(txt_path, "w") as f:
            f.write(f"worst rank        : {rank} of {len(rows)} frames (metric={metric})\n")
            f.write(f"image index       : {idx}\n")
            if info is not None:
                f.write(f"image             : {info['img_path']}\n")
                f.write(f"detector source   : {info['source_name']}\n")
                f.write(f"tag center (u,v)  : {info['center_uv']}\n")
                f.write(f"tag center depth  : {info['depth_mm']}\n")
            f.write(f"trans error [mm]  : {row['trans_err_mm']:.3f}\n")
            f.write(f"rot error [deg]   : {row['rot_err_deg']:.4f}\n")
            f.write(f"se3 error (rvec|t): {row['xi']}\n")
            f.write("\nT_cam_tag (measured):\n")
            f.write(str(T_cam_tag))
            f.write("\n\nT_cam_tag (predicted = X_cam_base @ T_base_tag):\n")
            f.write(str(T_pred))
            f.write("\n\nT_base_tag (robot FK @ gripper2tag):\n")
            f.write(str(T_base_tag))
            if info is not None:
                f.write("\n\nT_cam_tag (raw AprilTag pose, before any depth override):\n")
                f.write(str(info["T_cam_tag_apriltag"]))
            f.write("\n")
        written.append(txt_path)
        print(f"  {txt_path.name}")

    return written


# ============================================================
# Nonlinear optimization
# ============================================================

def _residual_for_optimizer(
    xi_update,
    X_init,
    T_cam_tag_list,
    T_base_tag_list,
    rot_weight=1.0,
    trans_weight=10000.0,
):
    """
    We optimize:
        X = X_init @ exp(xi_update)

    For each frame:
        predicted_T_cam_tag_i = X @ T_base_tag_i
        measured_T_cam_tag_i  = T_cam_tag_i

    residual_i = log(inv(measured) @ predicted)

    Rotation residual is radians.
    Translation residual is meters, multiplied by trans_weight.
    """
    X = X_init @ _se3_exp(xi_update)

    residuals = []

    for T_cam_tag, T_base_tag in zip(T_cam_tag_list, T_base_tag_list):
        T_pred = X @ T_base_tag
        T_err = np.linalg.inv(T_cam_tag) @ T_pred
        xi_err = _se3_log(T_err)

        r = np.hstack([
            rot_weight * xi_err[0:3],
            trans_weight * xi_err[3:6],
        ])
        residuals.append(r)

    return np.concatenate(residuals)


def _initial_guess_from_per_frame(T_cam_tag_list, T_base_tag_list):
    """
    Per frame:
        X_i = T_cam_tag_i @ inv(T_base_tag_i)
    Then SE(3) mean over X_i.
    """
    X_list = []

    for T_cam_tag, T_base_tag in zip(T_cam_tag_list, T_base_tag_list):
        X_i = T_cam_tag @ np.linalg.inv(T_base_tag)
        X_list.append(X_i)

    X_init = _mean_se3(X_list)
    return X_init, X_list


def _optimize_X(
    X_init,
    T_cam_tag_list,
    T_base_tag_list,
    robust_loss="huber",
    f_scale=10.0,
    max_nfev=200,
    rot_weight=1.0,
    trans_weight=1000.0,
):
    x0 = np.zeros(6)

    print("\n[INFO] Starting nonlinear least_squares optimization...")
    print("[INFO] Variable: X_cam_base = X_init @ exp(xi)")
    print("[INFO] robust_loss:", robust_loss)
    print("[INFO] f_scale:", f_scale)
    print("[INFO] rot_weight:", rot_weight)
    print("[INFO] trans_weight:", trans_weight)

    result = least_squares(
        _residual_for_optimizer,
        x0,
        args=(X_init, T_cam_tag_list, T_base_tag_list, rot_weight, trans_weight),
        loss=robust_loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        verbose=2,
    )

    X_opt = X_init @ _se3_exp(result.x)

    print("\n[INFO] Optimization result")
    print("  success:", result.success)
    print("  message:", result.message)
    print("  nfev:", result.nfev)
    print("  cost:", result.cost)
    print("  xi_update:", result.x)

    return X_opt, result


# ============================================================
# Main
# ============================================================

def _parse_exclude_images(s):
    if s is None or s.strip() == "":
        return set()
    return set(int(x.strip()) for x in s.split(",") if x.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Nonlinear optimization calibration for camera-to-base transform."
    )

    parser.add_argument("--camera", type=str, choices=["azure", "zed"], default="azure")
    parser.add_argument("--side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--calib-seq-name", type=str, required=True)
    parser.add_argument("--max-images", type=int, default=70)
    parser.add_argument("--exclude-images", type=str, default="")

    parser.add_argument(
        "--use-depth-translation",
        action="store_true",
        help="Use AprilTag rotation + Azure depth translation for T_cam_tag.",
    )
    parser.add_argument("--depth-patch-radius", type=int, default=5)
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Only try the original image for tag detection; skip the 12-variant "
             "enhancement cascade. Much faster when many frames fail (the cascade "
             "costs 12 full-res detections per failure and its aggressive CLAHE can "
             "trigger 'too many borders in contour_detect'). Undetected frames are "
             "skipped either way.",
    )

    parser.add_argument(
        "--save-to-calib-dir",
        action="store_true",
        help="Also save npz into captured_calibration_data/<seq>. Default only saves in zsy-testmycode/results.",
    )

    parser.add_argument(
        "--plot-errors",
        action="store_true",
        help="Plot the residual distribution (histograms of translation/rotation "
             "error, initial vs optimized, plus per-frame bars with the outliers "
             "flagged). Always saved as a PNG next to the summary; a window only "
             "opens when there is a display.",
    )
    parser.add_argument(
        "--no-show-plot",
        action="store_true",
        help="With --plot-errors, write the PNG but never open a window.",
    )

    parser.add_argument(
        "--save-worst-case", type=int, default=1, metavar="N",
        help="Save the N worst frames (annotated image + coloured point cloud + "
             "a txt with both tag poses) into <results>/<seq>/worst_case. 0 disables.",
    )
    parser.add_argument(
        "--worst-metric", type=str, default="combined",
        choices=["combined", "trans", "rot"],
        help="How to rank 'worst'. combined = max(trans/%.0fmm, rot/%.0fdeg), the "
             "same normalization the OUTLIER rule uses."
             % (TRANS_THRESH_MM, ROT_THRESH_DEG),
    )
    parser.add_argument(
        "--worst-cloud-stride", type=int, default=2,
        help="Pixel stride when back-projecting the worst frame's depth into a "
             "point cloud. 1 is full resolution (large files).",
    )

    parser.add_argument("--robust-loss", type=str, default="huber",
                        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"])
    parser.add_argument("--f-scale", type=float, default=10.0)
    parser.add_argument("--max-nfev", type=int, default=200)
    parser.add_argument("--rot-weight", type=float, default=1.0)
    parser.add_argument("--trans-weight", type=float, default=1000.0)
    parser.add_argument(
        "--rgbd-file", type=str, default=None,
        help="Alternative rgbd npz (filename inside the sequence directory, same "
             "format as {side}_calibration_rgbd.npz) to take the depth stack from. "
             "E.g. right_calibration_rgbd_fs.npz written by fs_depth_batch.py "
             "(FoundationStereo depth). Tag detection still runs on the capture's "
             "PNG frames; only the depth source changes. The ZED disparity-offset "
             "correction applies to FS depth exactly as to SDK depth (both are "
             "matched from the same rectified pair), so it stays on.")
    parser.add_argument(
        "--result-tag", type=str, default=None,
        help="Suffix added to every output name (npz/summary/plot/worst_case), so "
             "runs of the same sequence with different depth sources do not "
             "overwrite each other. Default: none for the standard rgbd file, or "
             "derived from --rgbd-file (e.g. right_calibration_rgbd_fs.npz -> 'fs').")
    parser.add_argument(
        "--d", "--disparity-offset-px", dest="disparity_offset_px",
        type=float, default=None,
        help="ZED disparity offset d, in pixels (model: disp_true = a*disp + d). "
             "Default: zed_capture/zed_depth_correction.json (currently "
             f"{zed_depth_config.offset_px():.2f} px). Pass 0 to disable the shift. "
             "Ignored unless --camera zed --use-depth-translation. Never applied "
             "twice: a dataset whose npz records its own correction is left alone.")
    parser.add_argument(
        "--a", "--disparity-scale", dest="disparity_scale",
        type=float, default=None,
        help="ZED disparity scale a, dimensionless (model: disp_true = a*disp + d). "
             "Default: zed_capture/zed_depth_correction.json (currently "
             f"{zed_depth_config.scale():.4f}). Pass 1 to disable the stretch.")

    args = parser.parse_args()

    side = args.side
    calib_base_dir = Path(DATAPATH) / "captured_calibration_data" / args.calib_seq_name
    frames_dir = calib_base_dir / "frames"
    pose_file = calib_base_dir / f"{side}_calibration_poses.npz"

    # An alternative depth source must not silently overwrite the standard run's
    # results, so it always gets a tag (derived from the filename if not given:
    # right_calibration_rgbd_fs.npz -> "fs").
    result_tag = args.result_tag
    if args.rgbd_file and not result_tag:
        stem = Path(args.rgbd_file).stem
        marker = f"{side}_calibration_rgbd"
        result_tag = (stem.replace(marker, "").strip("_") or stem) if marker in stem else stem
        print(f"[INFO] --result-tag not given; derived '{result_tag}' from --rgbd-file")

    # Depth back-projection needs intrinsics matching the recorded frames. Azure keeps
    # the module constants; other cameras must supply their own, at the right
    # resolution -- previously --use-depth-translation was hard-blocked for non-Azure.
    if args.camera != "azure" and args.use_depth_translation:
        depth_stack_probe = _load_depth_stack(calib_base_dir, side, args.rgbd_file)
        if depth_stack_probe is None:
            sys.exit(
                f"Error: --use-depth-translation needs a 'depth' array in "
                f"{_rgbd_npz_path(calib_base_dir, side, args.rgbd_file)}, but none was found.\n"
                "       The ZED branch of capture_poses_and_images_for_calibration.py "
                "records color only.\n"
                "       Re-capture with a ZED depth-enabled capture, or drop "
                "--use-depth-translation."
            )
        dh, dw = int(depth_stack_probe.shape[1]), int(depth_stack_probe.shape[2])
        from apriltag_image import _camera_params_for
        fx, fy, cx, cy = _camera_params_for(args.camera, dw, dh)
        print(f"[INFO] camera={args.camera}: depth stack is {dw}x{dh}")
        set_depth_intrinsics(fx, fy, cx, cy)

        # A ZED reports depth that is too far, by a constant disparity offset. Undo
        # it before the depth becomes a translation. already_applied_px is the guard
        # against double-correcting a capture that already did it. This applies to
        # FoundationStereo depth (--rgbd-file *_fs.npz) exactly as to SDK depth: the
        # offset is baked into the rectified images both matchers consume.
        rgbd_path = _rgbd_npz_path(calib_base_dir, side, args.rgbd_file)
        set_depth_corrector(zed_depth_config.corrector_for(
            args.camera, fx, unit="mm",
            offset_px_override=args.disparity_offset_px,
            scale_override=args.disparity_scale,
            already_applied_px=zed_depth_config.dataset_applied_offset_px(rgbd_path),
            already_applied_scale=zed_depth_config.dataset_applied_scale(rgbd_path),
        ))

    exclude_images = _parse_exclude_images(args.exclude_images)

    enhanced_dir = (
        Path(DATAPATH)
        / "zsy-testmycode"
        / "debug"
        / args.calib_seq_name
        / f"{side}_enhanced_images"
    )

    output_dir = (
        Path(DATAPATH)
        / "zsy-testmycode"
        / "results"
        / args.calib_seq_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print("[INFO] Nonlinear calibration")
    print("=" * 100)
    print("[INFO] side:", side)
    print("[INFO] seq:", args.calib_seq_name)
    print("[INFO] frames_dir:", frames_dir)
    print("[INFO] pose_file:", pose_file)
    print("[INFO] max_images:", args.max_images)
    print("[INFO] exclude_images:", sorted(exclude_images))
    print("[INFO] use_depth_translation:", args.use_depth_translation)
    print("[INFO] rgbd_file:", _rgbd_npz_path(calib_base_dir, side, args.rgbd_file).name,
          f"(result_tag={result_tag})" if result_tag else "")
    print("[INFO] output_dir:", output_dir)
    print("=" * 100)

    T_cam_tag_list, T_base_tag_list, valid_indices, source_names, frame_infos = _build_calibration_pairs(
        max_images=args.max_images,
        image_dir=frames_dir,
        pose_file=pose_file,
        calib_base_dir=calib_base_dir,
        side=side,
        camera=args.camera,
        use_depth_translation=args.use_depth_translation,
        depth_patch_radius=args.depth_patch_radius,
        enhance=not args.no_enhance,
        exclude_image_indices=exclude_images,
        enhanced_dir=enhanced_dir,
        rgbd_file=args.rgbd_file,
    )

    # Initial guess from the old logic:
    # X_i = T_cam_tag_i @ inv(T_base_tag_i), then SE3 mean.
    X_init, X_list = _initial_guess_from_per_frame(T_cam_tag_list, T_base_tag_list)

    print("\n[INFO] Initial X_cam_base from per-frame SE3 mean:")
    print(X_init)
    print("[INFO] Initial inverse X_base_cam:")
    print(np.linalg.inv(X_init))

    rows_init = _pose_error_rows(X_init, T_cam_tag_list, T_base_tag_list, valid_indices)
    _print_report("[Initial SE3-mean residual report]", rows_init)

    X_opt, result = _optimize_X(
        X_init,
        T_cam_tag_list,
        T_base_tag_list,
        robust_loss=args.robust_loss,
        f_scale=args.f_scale,
        max_nfev=args.max_nfev,
        rot_weight=args.rot_weight,
        trans_weight=args.trans_weight,
    )

    print("\n[INFO] Optimized X_cam_base:")
    print(X_opt)
    print("[INFO] Optimized inverse X_base_cam:")
    print(np.linalg.inv(X_opt))

    rows_opt = _pose_error_rows(X_opt, T_cam_tag_list, T_base_tag_list, valid_indices)
    outliers, stats_lines_opt = _print_report("[Optimized nonlinear residual report]", rows_opt)

    # Save in test folder by default. The result tag keeps depth-source variants
    # (e.g. FoundationStereo via --rgbd-file) in separate files.
    mode_name = "depth_translation" if args.use_depth_translation else "apriltag_translation"
    if result_tag:
        mode_name = f"{mode_name}_{result_tag}"
    out_name = f"base2cam_transform_{side}_nonlinear_opt_{mode_name}.npz"
    out_path = output_dir / out_name

    # One payload for both writes below. They used to be two literal argument lists,
    # which drifted apart the moment a key was added to only one of them.
    payload = dict(
        X_cam_base=X_opt,
        X_base_cam=np.linalg.inv(X_opt),
        X_init=X_init,
        valid_indices=np.array(valid_indices, dtype=int),
        outliers=np.array(outliers, dtype=int),
        optimization_x=result.x,
        optimization_cost=np.array([result.cost], dtype=float),
        # Provenance: which depth this transform was solved from. A depth_translation
        # transform is only valid for depth corrected by the SAME offset, so the
        # perception pipeline must apply this exact value.
        disparity_offset_px=np.float64(
            _DEPTH_CORRECTOR.offset_px if _DEPTH_CORRECTOR.enabled else 0.0),
        disparity_scale=np.float64(
            _DEPTH_CORRECTOR.scale if _DEPTH_CORRECTOR.enabled else 1.0),
    )

    # X_opt is also written positionally as 'arr_0', because older readers take it.
    np.savez(out_path, X_opt, **payload)

    print("\n[SAVED]")
    print("Saved nonlinear optimized matrix to:")
    print(" ", out_path)

    # Save a text summary next to it.
    summary_path = output_dir / f"summary_{side}_nonlinear_opt_{mode_name}.txt"
    with open(summary_path, "w") as f:
        f.write("Nonlinear calibration result\n")
        f.write(f"side: {side}\n")
        f.write(f"seq: {args.calib_seq_name}\n")
        f.write(f"use_depth_translation: {args.use_depth_translation}\n")
        f.write(f"rgbd_file: {_rgbd_npz_path(calib_base_dir, side, args.rgbd_file).name}\n")
        f.write(f"valid_indices: {valid_indices}\n")
        f.write(f"outliers: {outliers}\n")
        f.write(f"npz: {out_path}\n\n")
        # Same block the console prints for the optimized transform, so the summary
        # answers "how good is this calibration" without re-running the script.
        f.write("Optimized residuals:\n")
        for line in stats_lines_opt:
            f.write(line + "\n")
        f.write("\n")
        f.write("X_cam_base:\n")
        f.write(str(X_opt))
        f.write("\n\nX_base_cam = inv(X_cam_base):\n")
        f.write(str(np.linalg.inv(X_opt)))
        f.write("\n")

    print("Saved summary to:")
    print(" ", summary_path)

    if args.save_to_calib_dir:
        calib_out_path = calib_base_dir / out_name
        np.savez(calib_out_path, X_opt, **payload)
        print("\n[SAVED ALSO TO CALIB DIR]")
        print(" ", calib_out_path)

    if args.save_worst_case > 0:
        # The cloud must be built with the SAME intrinsics and the SAME depth
        # correction as the calibration. Both are already installed for
        # --use-depth-translation; without it they are still at their Azure
        # defaults, so re-resolve them here against the actual depth resolution.
        depth_stack = _load_depth_stack(calib_base_dir, side, args.rgbd_file)
        if depth_stack is not None:
            dh, dw = int(depth_stack.shape[1]), int(depth_stack.shape[2])
            from apriltag_image import _camera_params_for
            fx, fy, cx, cy = _camera_params_for(args.camera, dw, dh)
            set_depth_intrinsics(fx, fy, cx, cy)
            if args.camera == "zed" and not _DEPTH_CORRECTOR.enabled:
                rgbd_path = _rgbd_npz_path(calib_base_dir, side, args.rgbd_file)
                set_depth_corrector(zed_depth_config.corrector_for(
                    args.camera, fx, unit="mm",
                    offset_px_override=args.disparity_offset_px,
                    scale_override=args.disparity_scale,
                    already_applied_px=zed_depth_config.dataset_applied_offset_px(rgbd_path),
                    already_applied_scale=zed_depth_config.dataset_applied_scale(rgbd_path),
                ))
                print("[WORST] depth correction installed for the point cloud only "
                      "(the fit did not use depth).")

        color_intr = (AZURE_FX, AZURE_FY, AZURE_CX, AZURE_CY)
        if frame_infos:
            probe = cv2.imread(str(frame_infos[0]["img_path"]))
            if probe is not None:
                from apriltag_image import _camera_params_for
                color_intr = _camera_params_for(args.camera, probe.shape[1], probe.shape[0])

        _save_worst_cases(
            out_dir=output_dir / "worst_case",
            rows=rows_opt,
            frame_infos=frame_infos,
            T_cam_tag_list=T_cam_tag_list,
            T_base_tag_list=T_base_tag_list,
            valid_indices=valid_indices,
            X_opt=X_opt,
            color_intr=color_intr,
            depth_stack=depth_stack,
            n_worst=args.save_worst_case,
            metric=args.worst_metric,
            cloud_stride=max(1, args.worst_cloud_stride),
            tag=f"{side}_{mode_name}",
        )

    print("\n[IMPORTANT]")
    print("This saved matrix follows the same convention as the old calculate_base_to_cam output.")
    print("Your dual_green_pick.py currently loads T_saved and uses inv(T_saved) as camera_to_base.")
    print("So if you want to test this matrix in dual_green_pick.py, point TRANSFORM_PATH to the npz above.")

    # Dead last: plt.show() blocks until the window is closed, so everything this
    # run writes and prints is already done by the time the window appears.
    if args.plot_errors:
        _plot_error_distribution(
            rows_init,
            rows_opt,
            outliers,
            output_dir / f"error_distribution_{side}_nonlinear_opt_{mode_name}.png",
            show=not args.no_show_plot,
            title_suffix=f"  --  {args.calib_seq_name} / {side} / {mode_name}",
        )


if __name__ == "__main__":
    main()
