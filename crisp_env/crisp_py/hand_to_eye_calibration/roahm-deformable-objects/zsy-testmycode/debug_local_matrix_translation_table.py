#!/usr/bin/env python3

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# Make parent calibration folder importable, because this script lives in zsy-testmycode/
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

# Shared ZED depth disparity-offset correction. See zed_capture/zed_depth_config.py.
_ZED_CAPTURE_DIR = ROOT_DIR.parents[1] / "zed_capture"
if str(_ZED_CAPTURE_DIR) not in sys.path:
    sys.path.insert(0, str(_ZED_CAPTURE_DIR))
try:
    import zed_depth_config
except ImportError as exc:
    raise ImportError(
        f"Cannot import zed_depth_config from {_ZED_CAPTURE_DIR}."
    ) from exc

from apriltag_image import apriltag_image


# ============================================================
# Depth back-projection intrinsics, only used when --use-depth-translation
# Default is the Azure colour camera at 1280x720. set_depth_intrinsics()
# replaces them for any other camera, at the resolution of the recorded frames.
# ============================================================

AZURE_FX = 606.112427
AZURE_FY = 605.882141
AZURE_CX = 641.757812
AZURE_CY = 365.651886

_DEPTH_INTR = {"fx": AZURE_FX, "fy": AZURE_FY, "cx": AZURE_CX, "cy": AZURE_CY}


def set_depth_intrinsics(fx, fy, cx, cy):
    """Override the intrinsics used to unproject depth pixels into camera-frame 3D.

    They MUST match the resolution of the recorded depth frames. For a ZED they must
    be the RECTIFIED left-camera values, because VIEW.LEFT is already rectified.
    """
    _DEPTH_INTR.update(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))
    print(f"[INFO] depth back-projection intrinsics: fx={fx:.6f} fy={fy:.6f} "
          f"cx={cx:.6f} cy={cy:.6f}")


# ZED depth disparity-offset correction. Identity until main() installs a real one.
_DEPTH_CORRECTOR = zed_depth_config.disabled_corrector("not configured yet")


def set_depth_corrector(corrector):
    """Install the depth corrector that _depth_at_pixel applies."""
    global _DEPTH_CORRECTOR
    _DEPTH_CORRECTOR = corrector

EXPECTED_TAG_ID = 3
TAG_SIZE_M = 0.0950   # measured with calipers 2026-08-19 (re-verified); was wrongly assumed 0.093


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


def _mean_se3(transforms, max_iters=100, tol=1e-9):
    if len(transforms) == 0:
        raise ValueError("Cannot compute SE3 mean of empty transform list.")

    translations = np.array([T[0:3, 3] for T in transforms])
    median_t = np.median(translations, axis=0)
    dists = np.linalg.norm(translations - median_t[None, :], axis=1)
    init_idx = int(np.argmin(dists))
    T_mean = transforms[init_idx].copy()

    print(f"[INFO] SE3 mean init index among valid frames: {init_idx}")

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
    enhanced_specs.append(("L1_gamma0p70", _gamma_bgr(img, gamma=0.70)))
    enhanced_specs.append(("L1_alpha1p5_beta35", _contrast_brightness(img, alpha=1.5, beta=35)))

    e_clahe5 = _clahe_bgr(img, clip=5.0, tile=(8, 8))
    enhanced_specs.append(("L2_clahe5", e_clahe5))
    enhanced_specs.append(("L2_gamma0p50", _gamma_bgr(img, gamma=0.50)))
    enhanced_specs.append(("L2_alpha2p2_beta75", _contrast_brightness(img, alpha=2.2, beta=75)))

    e_clahe8 = _clahe_bgr(img, clip=8.0, tile=(4, 4))
    enhanced_specs.append(("L3_clahe8_tile4", e_clahe8))
    enhanced_specs.append(("L3_gamma0p35", _gamma_bgr(img, gamma=0.35)))

    for name, enh in enhanced_specs:
        out_path = out_dir / f"debug_enhanced_{image_index:03d}_{name}.png"
        cv2.imwrite(str(out_path), enh)
        versions.append((name, out_path))

    return versions


# ============================================================
# AprilTag detection
# ============================================================

def _find_expected_tag(detections):
    if detections is None or len(detections) == 0:
        return None, None

    found_det = None
    found_T = None

    for j in range(0, len(detections), 4):
        det = detections[j]
        if det.tag_id == EXPECTED_TAG_ID:
            if found_det is not None:
                raise RuntimeError(f"Detected multiple tag_id={EXPECTED_TAG_ID} tags.")
            found_det = det
            found_T = np.asarray(detections[j + 1]).copy()

    return found_det, found_T


def _detect_with_enhancement(img_path, enhanced_dir, image_index, camera):
    versions = _make_enhanced_versions(img_path, enhanced_dir, image_index)

    for source_name, path in versions:
        detections = apriltag_image(
            [str(path)],
            output_images=False,
            display_images=False,
            tag_size=TAG_SIZE_M,
            tag_family="tag36h11",
            camera=camera,
        )

        det, T = _find_expected_tag(detections)
        if det is not None and T is not None:
            return det, T, source_name

    return None, None, None


# ============================================================
# Depth utilities
# ============================================================

def _load_depth_stack(calib_base_dir, side):
    rgbd_path = calib_base_dir / f"{side}_calibration_rgbd.npz"
    if not rgbd_path.exists():
        print(f"[WARN] RGB-D file not found: {rgbd_path}")
        return None

    data = np.load(rgbd_path)
    if "depth" not in data.files:
        print(f"[WARN] No depth key in {rgbd_path}. Keys are {data.files}")
        return None

    depth = data["depth"]
    print(f"[INFO] Loaded depth stack: {rgbd_path}, shape={depth.shape}")
    return depth


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

    # Correcting the median equals correcting every pixel then taking the median:
    # the disparity-offset map is monotonic in z, so it commutes with the median.
    return _DEPTH_CORRECTOR(float(np.median(valid)))


def _pixel_depth_to_camera_point(u, v, depth_mm):
    Z = depth_mm / 1000.0
    X = (float(u) - _DEPTH_INTR["cx"]) * Z / _DEPTH_INTR["fx"]
    Y = (float(v) - _DEPTH_INTR["cy"]) * Z / _DEPTH_INTR["fy"]
    return np.array([X, Y, Z], dtype=float)


# ============================================================
# Robot pose loading
# ============================================================

gripper2tag = np.array([
    [0, 0, -1, -0.02],
    [0, -1, 0, 0],
    [-1, 0, 0, 0.0905],
    [0, 0, 0, 1],
])


def _load_robot_pose(pose_file, image_index):
    data = np.load(pose_file)
    key = "arr_" + str(image_index)
    if key not in data:
        return None

    arr = data[key]

    T_base_gripper = np.eye(4)
    T_base_gripper[0:3, 0:3] = Rotation.from_quat(arr[3:7]).as_matrix()
    T_base_gripper[0:3, 3] = arr[0:3]

    return T_base_gripper


# ============================================================
# Main diagnostic logic
# ============================================================

def _parse_int_set(s):
    if s is None or s.strip() == "":
        return set()
    return set(int(x.strip()) for x in s.split(",") if x.strip())


def _parse_target_cam(s):
    if s is None or s.strip() == "":
        return None
    parts = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError("--target-cam must be 'x,y,z'")
    return np.array([parts[0], parts[1], parts[2], 1.0], dtype=float)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Debug per-frame local camera/base matrix translation vs final matrix. "
            "No npz is saved. Outputs CSV table only."
        )
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
        "--final-matrix-path",
        type=str,
        default="",
        help=(
            "Optional existing base2cam npz. If empty, compute SE3 mean from valid per-frame matrices."
        ),
    )

    parser.add_argument(
        "--target-cam",
        type=str,
        default="",
        help=(
            "Optional camera-frame target point x,y,z. "
            "If provided, table also compares p_base_i vs p_base_final for this target."
        ),
    )

    parser.add_argument(
        "--sort-by",
        type=str,
        default="abs_ee_xy",
        choices=["image", "abs_ee_xy", "diff_cam2base_y", "diff_target_y"],
    )
    parser.add_argument(
        "--disparity-offset-px", type=float, default=None,
        help="ZED depth disparity-offset correction, in pixels. Default: the value in "
             f"zed_capture/zed_depth_correction.json (currently "
             f"{zed_depth_config.offset_px():.2f} px). Pass 0 for raw depth. Ignored "
             "for --camera azure.")

    args = parser.parse_args()

    calib_base_dir = ROOT_DIR / "captured_calibration_data" / args.calib_seq_name
    frames_dir = calib_base_dir / "frames"
    pose_file = calib_base_dir / f"{args.side}_calibration_poses.npz"

    if not frames_dir.exists():
        raise RuntimeError(f"frames dir not found: {frames_dir}")
    if not pose_file.exists():
        raise RuntimeError(f"pose file not found: {pose_file}")

    exclude_images = _parse_int_set(args.exclude_images)
    target_cam = _parse_target_cam(args.target_cam)

    output_dir = ROOT_DIR / "zsy-testmycode" / "debug_tables" / args.calib_seq_name
    enhanced_dir = ROOT_DIR / "zsy-testmycode" / "debug" / args.calib_seq_name / f"{args.side}_enhanced_for_local_table"
    output_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    depth_stack = None
    if args.use_depth_translation:
        depth_stack = _load_depth_stack(calib_base_dir, args.side)
        if depth_stack is None:
            raise RuntimeError("No depth stack available.")

        # Non-Azure cameras need their own intrinsics, at the recorded resolution,
        # plus the ZED disparity-offset correction. This used to be hard-blocked.
        if args.camera != "azure":
            _dh, _dw = int(depth_stack.shape[1]), int(depth_stack.shape[2])
            from apriltag_image import _camera_params_for
            _fx, _fy, _cx, _cy = _camera_params_for(args.camera, _dw, _dh)
            print(f"[INFO] camera={args.camera}: depth stack is {_dw}x{_dh}")
            set_depth_intrinsics(_fx, _fy, _cx, _cy)
            set_depth_corrector(zed_depth_config.corrector_for(
                args.camera, _fx, unit="mm",
                offset_px_override=args.disparity_offset_px,
                already_applied_px=zed_depth_config.dataset_applied_offset_px(
                    calib_base_dir / f"{args.side}_calibration_rgbd.npz"),
            ))

    rows = []
    T_i_list = []
    valid_indices = []

    print("\n" + "=" * 100)
    print("[INFO] Building per-frame matrices")
    print("=" * 100)

    for image_index in range(args.max_images):
        if image_index in exclude_images:
            continue

        img_path = frames_dir / f"calibration_{args.side}_image_{image_index}.png"
        if not img_path.exists():
            print(f"[WARN] image {image_index}: missing image")
            continue

        T_base_gripper = _load_robot_pose(pose_file, image_index)
        if T_base_gripper is None:
            print(f"[WARN] image {image_index}: missing robot pose")
            continue

        det, T_cam_tag_apriltag, source = _detect_with_enhancement(
            img_path=img_path,
            enhanced_dir=enhanced_dir,
            image_index=image_index,
            camera=args.camera,
        )

        if det is None:
            print(f"[WARN] image {image_index}: no tag detection")
            continue

        T_cam_tag = T_cam_tag_apriltag.copy()
        depth_mm = np.nan

        if args.use_depth_translation:
            if image_index >= depth_stack.shape[0]:
                print(f"[WARN] image {image_index}: depth stack too short")
                continue

            center = np.asarray(det.center).astype(float)
            u, v = float(center[0]), float(center[1])
            depth_mm_val = _depth_at_pixel(
                depth_stack[image_index],
                u,
                v,
                patch_radius=args.depth_patch_radius,
            )

            if depth_mm_val is None:
                print(f"[WARN] image {image_index}: no valid center depth")
                continue

            depth_mm = depth_mm_val

            R_cam_tag = T_cam_tag_apriltag[0:3, 0:3]
            t_depth = _pixel_depth_to_camera_point(u, v, depth_mm_val)

            T_cam_tag = np.eye(4)
            T_cam_tag[0:3, 0:3] = R_cam_tag
            T_cam_tag[0:3, 3] = t_depth

        T_base_tag = T_base_gripper @ gripper2tag

        # This is the same per-frame matrix convention as old calculate_base_to_cam:
        # T_i = T_cam_tag @ T_tag_base = T_cam_base
        T_i_cam_base = T_cam_tag @ np.linalg.inv(T_base_tag)

        T_i_list.append(T_i_cam_base)
        valid_indices.append(image_index)

        T_i_base_cam = np.linalg.inv(T_i_cam_base)

        ee = T_base_gripper[0:3, 3]
        tag_base = T_base_tag[0:3, 3]

        rows.append({
            "image_index": image_index,
            "source": source,
            "depth_mm": depth_mm,

            # arm/base-frame end-effector coordinate
            "ee_x": ee[0],
            "ee_y": ee[1],
            "ee_z": ee[2],
            "abs_ee_xy": abs(ee[0]) + abs(ee[1]),

            # tag coordinate in arm/base frame
            "tag_base_x": tag_base[0],
            "tag_base_y": tag_base[1],
            "tag_base_z": tag_base[2],

            # per-frame matrix translation only
            "cam2base_i_tx": T_i_base_cam[0, 3],
            "cam2base_i_ty": T_i_base_cam[1, 3],
            "cam2base_i_tz": T_i_base_cam[2, 3],

            "base2cam_i_tx": T_i_cam_base[0, 3],
            "base2cam_i_ty": T_i_cam_base[1, 3],
            "base2cam_i_tz": T_i_cam_base[2, 3],

            # Filled after final matrix is known
            "cam2base_final_tx": np.nan,
            "cam2base_final_ty": np.nan,
            "cam2base_final_tz": np.nan,
            "diff_cam2base_tx_mm": np.nan,
            "diff_cam2base_ty_mm": np.nan,
            "diff_cam2base_tz_mm": np.nan,

            "target_base_i_x": np.nan,
            "target_base_i_y": np.nan,
            "target_base_i_z": np.nan,
            "target_base_final_x": np.nan,
            "target_base_final_y": np.nan,
            "target_base_final_z": np.nan,
            "diff_target_x_mm": np.nan,
            "diff_target_y_mm": np.nan,
            "diff_target_z_mm": np.nan,
        })

        print(f"[OK] image {image_index}: source={source}")

    if len(T_i_list) < 3:
        raise RuntimeError(f"Too few valid per-frame matrices: {len(T_i_list)}")

    print("\n[INFO] valid images:", valid_indices)

    if args.final_matrix_path.strip():
        final_path = Path(args.final_matrix_path)
        data = np.load(final_path)
        T_final_cam_base = data["arr_0"]
        print("[INFO] Loaded final base2cam/cam_base matrix from:", final_path)
    else:
        T_final_cam_base = _mean_se3(T_i_list)
        print("[INFO] Computed final matrix as SE3 mean from valid per-frame matrices.")

    T_final_base_cam = np.linalg.inv(T_final_cam_base)

    print("\n[INFO] Final T_cam_base translation:", T_final_cam_base[0:3, 3])
    print("[INFO] Final T_base_cam translation:", T_final_base_cam[0:3, 3])

    for r, T_i_cam_base in zip(rows, T_i_list):
        T_i_base_cam = np.linalg.inv(T_i_cam_base)

        r["cam2base_final_tx"] = T_final_base_cam[0, 3]
        r["cam2base_final_ty"] = T_final_base_cam[1, 3]
        r["cam2base_final_tz"] = T_final_base_cam[2, 3]

        diff_t = T_i_base_cam[0:3, 3] - T_final_base_cam[0:3, 3]
        r["diff_cam2base_tx_mm"] = diff_t[0] * 1000.0
        r["diff_cam2base_ty_mm"] = diff_t[1] * 1000.0
        r["diff_cam2base_tz_mm"] = diff_t[2] * 1000.0

        if target_cam is not None:
            p_base_i = T_i_base_cam @ target_cam
            p_base_final = T_final_base_cam @ target_cam
            diff_p = p_base_i[0:3] - p_base_final[0:3]

            r["target_base_i_x"] = p_base_i[0]
            r["target_base_i_y"] = p_base_i[1]
            r["target_base_i_z"] = p_base_i[2]

            r["target_base_final_x"] = p_base_final[0]
            r["target_base_final_y"] = p_base_final[1]
            r["target_base_final_z"] = p_base_final[2]

            r["diff_target_x_mm"] = diff_p[0] * 1000.0
            r["diff_target_y_mm"] = diff_p[1] * 1000.0
            r["diff_target_z_mm"] = diff_p[2] * 1000.0

    if args.sort_by == "image":
        rows_sorted = sorted(rows, key=lambda x: x["image_index"])
    elif args.sort_by == "abs_ee_xy":
        rows_sorted = sorted(rows, key=lambda x: x["abs_ee_xy"], reverse=True)
    elif args.sort_by == "diff_cam2base_y":
        rows_sorted = sorted(rows, key=lambda x: abs(x["diff_cam2base_ty_mm"]), reverse=True)
    elif args.sort_by == "diff_target_y":
        rows_sorted = sorted(rows, key=lambda x: abs(x["diff_target_y_mm"]), reverse=True)
    else:
        rows_sorted = rows

    mode = "depth_translation" if args.use_depth_translation else "apriltag_translation"
    out_csv = output_dir / f"local_matrix_translation_table_{args.side}_{mode}.csv"

    fieldnames = [
        "image_index", "source", "depth_mm",

        "ee_x", "ee_y", "ee_z", "abs_ee_xy",
        "tag_base_x", "tag_base_y", "tag_base_z",

        "cam2base_i_tx", "cam2base_i_ty", "cam2base_i_tz",
        "cam2base_final_tx", "cam2base_final_ty", "cam2base_final_tz",
        "diff_cam2base_tx_mm", "diff_cam2base_ty_mm", "diff_cam2base_tz_mm",

        "base2cam_i_tx", "base2cam_i_ty", "base2cam_i_tz",

        "target_base_i_x", "target_base_i_y", "target_base_i_z",
        "target_base_final_x", "target_base_final_y", "target_base_final_z",
        "diff_target_x_mm", "diff_target_y_mm", "diff_target_z_mm",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow(r)

    print("\n" + "=" * 120)
    print("[TABLE PREVIEW]")
    print("Sorted by:", args.sort_by)
    print("=" * 120)
    print(
        "img | ee_x   ee_y   ee_z  | absxy | "
        "cam2base_i_t(mm) - final_t(mm) = diff(mm)"
    )
    print("-" * 120)

    for r in rows_sorted[:30]:
        print(
            f"{r['image_index']:>3d} | "
            f"{r['ee_x']:+.3f} {r['ee_y']:+.3f} {r['ee_z']:+.3f} | "
            f"{r['abs_ee_xy']:.3f} | "
            f"dx={r['diff_cam2base_tx_mm']:+7.1f}, "
            f"dy={r['diff_cam2base_ty_mm']:+7.1f}, "
            f"dz={r['diff_cam2base_tz_mm']:+7.1f}"
        )

    if target_cam is not None:
        print("\n" + "=" * 120)
        print("[TARGET POINT PREVIEW]")
        print("target_cam =", target_cam[:3])
        print("=" * 120)
        print("img | target diff base mm: dx dy dz")
        print("-" * 120)
        for r in rows_sorted[:30]:
            print(
                f"{r['image_index']:>3d} | "
                f"dx={r['diff_target_x_mm']:+7.1f}, "
                f"dy={r['diff_target_y_mm']:+7.1f}, "
                f"dz={r['diff_target_z_mm']:+7.1f}"
            )

    print("\n[SAVED CSV]")
    print(out_csv)
    print("\nOpen with:")
    print(f"  libreoffice --calc {out_csv}")
    print("or:")
    print(f"  python - <<'PY2'\nimport pandas as pd\np='{out_csv}'\ndf=pd.read_csv(p)\nprint(df.head(30).to_string())\nPY2")


if __name__ == "__main__":
    main()
