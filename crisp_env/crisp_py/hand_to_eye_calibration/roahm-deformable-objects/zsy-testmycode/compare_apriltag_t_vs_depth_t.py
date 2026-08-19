#!/usr/bin/env python3

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Shared ZED depth disparity-offset correction. See zed_capture/zed_depth_config.py.
_ZED_CAPTURE_DIR = REPO_ROOT.parents[1] / "zed_capture"
if str(_ZED_CAPTURE_DIR) not in sys.path:
    sys.path.insert(0, str(_ZED_CAPTURE_DIR))
try:
    import zed_depth_config
except ImportError as exc:
    raise ImportError(
        f"Cannot import zed_depth_config from {_ZED_CAPTURE_DIR}. This script "
        "compares AprilTag translation against DEPTH translation, so an "
        "uncorrected ZED depth would make the comparison meaningless."
    ) from exc

import argparse
from pathlib import Path
import sys
import cv2
import numpy as np

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
TAG_SIZE_M = 0.0950   # measured with calipers 2026-08-19 (re-verified); was wrongly assumed 0.093


# ============================================================
# Image enhancement utilities
# ============================================================

def _gamma_bgr(img_bgr, gamma):
    """
    gamma < 1 makes image brighter.
    """
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
    """
    out = alpha * img + beta
    """
    return cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)


def _make_enhanced_versions(img_path, out_dir, image_index):
    """
    Return list of (name, path).
    The first one is original image.
    Later ones are increasingly enhanced.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    versions = []

    # Original
    versions.append(("original", img_path))

    # Strong but not insane enhancement chain
    enhanced_specs = []

    # Level 1
    enhanced_specs.append(("L1_clahe3", _clahe_bgr(img, clip=3.0, tile=(8, 8))))
    enhanced_specs.append(("L1_gamma0p60", _gamma_bgr(img, gamma=0.60)))
    enhanced_specs.append(("L1_alpha1p8_beta45", _contrast_brightness(img, alpha=1.8, beta=45)))

    # Level 2
    e_clahe5 = _clahe_bgr(img, clip=5.0, tile=(8, 8))
    enhanced_specs.append(("L2_clahe5", e_clahe5))
    enhanced_specs.append(("L2_clahe5_gamma0p50", _gamma_bgr(e_clahe5, gamma=0.50)))
    enhanced_specs.append(("L2_alpha2p5_beta80", _contrast_brightness(img, alpha=2.5, beta=80)))

    # Level 3
    e_clahe8 = _clahe_bgr(img, clip=8.0, tile=(4, 4))
    enhanced_specs.append(("L3_clahe8_tile4", e_clahe8))
    enhanced_specs.append(("L3_clahe8_gamma0p35", _gamma_bgr(e_clahe8, gamma=0.35)))
    enhanced_specs.append(("L3_alpha3p5_beta120", _contrast_brightness(img, alpha=3.5, beta=120)))

    # Grayscale-style enhancement converted back to BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    gray_eq_bgr = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
    enhanced_specs.append(("L4_gray_equalized", gray_eq_bgr))

    gray_clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4)).apply(gray)
    gray_clahe_bgr = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)
    enhanced_specs.append(("L4_gray_clahe8", gray_clahe_bgr))

    for name, enh in enhanced_specs:
        out_path = out_dir / f"image_{image_index:03d}_{name}.png"
        cv2.imwrite(str(out_path), enh)
        versions.append((name, out_path))

    return versions


# ============================================================
# AprilTag detection utilities
# ============================================================

def _find_expected_tag(detections, expected_tag_id=EXPECTED_TAG_ID):
    """
    Existing repo's apriltag_image returns a list where the old code iterates:
        for j in range(0, len(detections), 4):
            detections[j]     -> detection object
            detections[j + 1] -> 4x4 transform

    We keep that same assumption.
    """
    if detections is None or len(detections) == 0:
        return None, None

    found_detection = None
    found_transform = None

    for j in range(0, len(detections), 4):
        det = detections[j]
        if det.tag_id == expected_tag_id:
            if found_detection is not None:
                raise RuntimeError(f"Detected multiple tag_id={expected_tag_id} tags.")
            found_detection = det
            found_transform = np.asarray(detections[j + 1]).copy()

    return found_detection, found_transform


def _detect_with_enhancement_until_success(img_path, enhanced_dir, image_index, camera):
    """
    Try original first.
    If fail, generate enhanced images and try them one by one.
    Return:
        detection, transform, source_name, used_image_path
    """
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

        det, T = _find_expected_tag(detections, expected_tag_id=EXPECTED_TAG_ID)

        if det is not None and T is not None:
            return det, T, source_name, path

    return None, None, None, None


# ============================================================
# Depth utilities
# ============================================================

def _load_depth_stack(calib_base_dir, side):
    rgbd_path = calib_base_dir / f"{side}_calibration_rgbd.npz"

    if not rgbd_path.exists():
        raise RuntimeError(f"RGB-D file not found: {rgbd_path}")

    data = np.load(rgbd_path)

    if "depth" not in data.files:
        raise RuntimeError(f"No 'depth' key in {rgbd_path}. Keys are: {data.files}")

    depth = data["depth"]
    print(f"[INFO] Loaded depth stack: {rgbd_path}")
    print(f"[INFO] depth shape: {depth.shape}")
    return depth


# ZED depth disparity-offset correction. Identity until main() installs a real one.
_DEPTH_CORRECTOR = zed_depth_config.disabled_corrector("not configured yet")


def set_depth_corrector(corrector):
    """Install the depth corrector that _depth_at_pixel applies. Needs fx, so call
    this after set_depth_intrinsics."""
    global _DEPTH_CORRECTOR
    _DEPTH_CORRECTOR = corrector


def _depth_at_pixel(depth_img, u, v, patch_radius=5):
    """
    Return robust median depth around pixel (u, v), in millimeters.
    """
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


# Intrinsics used for depth unprojection / reprojection. Defaults to the Azure
# constants above so --camera azure is bit-identical; set_depth_intrinsics() overrides
# them for other cameras. Previously these were hardcoded, so a ZED comparison would
# have silently used Azure numbers.
_DEPTH_INTR = {"fx": AZURE_FX, "fy": AZURE_FY, "cx": AZURE_CX, "cy": AZURE_CY}


def set_depth_intrinsics(fx, fy, cx, cy):
    """Override the intrinsics used for depth <-> pixel conversion.

    Must correspond to the resolution of the recorded frames, and for a ZED must be the
    RECTIFIED left-camera values (VIEW.LEFT is rectified; distortion is zero).
    """
    _DEPTH_INTR.update(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))
    print(f"[INFO] depth intrinsics: fx={fx:.6f} fy={fy:.6f} cx={cx:.6f} cy={cy:.6f}")


def _pixel_depth_to_camera_point(u, v, depth_mm):
    """
    Convert a color pixel + registered depth to a camera-frame 3D point.
    """
    Z = depth_mm / 1000.0
    X = (float(u) - _DEPTH_INTR["cx"]) * Z / _DEPTH_INTR["fx"]
    Y = (float(v) - _DEPTH_INTR["cy"]) * Z / _DEPTH_INTR["fy"]

    return np.array([X, Y, Z], dtype=float)


def _project_camera_point_to_pixel(p_cam):
    """
    Project camera-frame 3D point back to color pixel.
    """
    X, Y, Z = p_cam[:3]
    if Z <= 0:
        return None

    u = int(round(_DEPTH_INTR["fx"] * X / Z + _DEPTH_INTR["cx"]))
    v = int(round(_DEPTH_INTR["fy"] * Y / Z + _DEPTH_INTR["cy"]))
    return u, v


# ============================================================
# Annotation
# ============================================================

def _draw_text_block(img, lines, x=20, y=30):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    line_h = 22

    # black background rectangle
    max_len = max(len(s) for s in lines)
    rect_w = min(img.shape[1] - x - 10, int(max_len * 10))
    rect_h = line_h * len(lines) + 12
    cv2.rectangle(img, (x - 8, y - 22), (x + rect_w, y - 22 + rect_h), (0, 0, 0), -1)

    for i, s in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(img, s, (x, yy), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _annotate_result(
    original_img_path,
    used_img_path,
    out_path,
    image_index,
    source_name,
    center_uv,
    t_apriltag,
    t_depth,
    depth_mm,
):
    """
    Save side-by-side image:
      left: original
      right: image used for detection
    Draw:
      yellow dot = AprilTag detection center pixel
      cyan dot = projection of t_apriltag
      magenta dot = projection of t_depth
    """
    original = cv2.imread(str(original_img_path))
    used = cv2.imread(str(used_img_path))

    if original is None:
        raise RuntimeError(f"Could not read original image: {original_img_path}")
    if used is None:
        raise RuntimeError(f"Could not read used image: {used_img_path}")

    h = min(original.shape[0], used.shape[0])
    w = min(original.shape[1], used.shape[1])

    original = original[:h, :w].copy()
    used = used[:h, :w].copy()

    # draw on used image
    u, v = center_uv
    cv2.circle(used, (int(round(u)), int(round(v))), 8, (0, 255, 255), -1)  # yellow

    px_apriltag = _project_camera_point_to_pixel(t_apriltag)
    px_depth = _project_camera_point_to_pixel(t_depth)

    if px_apriltag is not None:
        cv2.circle(used, px_apriltag, 8, (255, 255, 0), 2)  # cyan outline

    if px_depth is not None:
        cv2.circle(used, px_depth, 12, (255, 0, 255), 2)  # magenta outline

    diff = t_depth - t_apriltag
    diff_norm_mm = np.linalg.norm(diff) * 1000.0

    lines = [
        f"image {image_index}, detect={source_name}",
        "yellow filled: detected tag center pixel",
        "cyan circle: project(t_apriltag)",
        "magenta circle: project(t_depth)",
        f"depth(center) = {depth_mm:.1f} mm",
        f"t_apriltag = [{t_apriltag[0]:+.4f}, {t_apriltag[1]:+.4f}, {t_apriltag[2]:+.4f}] m",
        f"t_depth    = [{t_depth[0]:+.4f}, {t_depth[1]:+.4f}, {t_depth[2]:+.4f}] m",
        f"diff depth-apriltag = [{diff[0]*1000:+.1f}, {diff[1]*1000:+.1f}, {diff[2]*1000:+.1f}] mm",
        f"|diff| = {diff_norm_mm:.1f} mm",
    ]

    _draw_text_block(used, lines, x=20, y=32)

    cv2.putText(original, "original image", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(used, "used image for detection", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    combo = np.hstack([original, used])
    cv2.imwrite(str(out_path), combo)


# ============================================================
# Main
# ============================================================

def _parse_indices(s):
    if s is None or s.strip() == "":
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare AprilTag translation t_apriltag vs Azure depth translation t_depth "
            "on a few calibration images."
        )
    )

    parser.add_argument("--camera", type=str, choices=["azure", "zed"], default="azure")
    parser.add_argument("--side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--calib-seq-name", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument(
        "--image-indices",
        type=str,
        default="",
        help="Optional comma-separated image indices, e.g. '0,3,7,20,25'. If empty, use first num-images.",
    )
    parser.add_argument("--depth-patch-radius", type=int, default=5)
    parser.add_argument(
        "--d", "--disparity-offset-px", dest="disparity_offset_px",
        type=float, default=None,
        help="ZED disparity offset d, in pixels (model: disp_true = a*disp + d). "
             f"Default: zed_capture/zed_depth_correction.json (currently "
             f"{zed_depth_config.offset_px():.2f} px). Pass 0 to see the raw depth "
             "error. Ignored for --camera azure.")
    parser.add_argument(
        "--a", "--disparity-scale", dest="disparity_scale",
        type=float, default=None,
        help="ZED disparity scale a, dimensionless. Default: "
             f"zed_capture/zed_depth_correction.json (currently "
             f"{zed_depth_config.scale():.4f}). Pass 1 to disable the stretch.")

    args = parser.parse_args()

    side = args.side
    calib_base_dir = Path(DATAPATH) / "captured_calibration_data" / args.calib_seq_name
    frames_dir = calib_base_dir / "frames"

    # Non-Azure cameras need their own depth intrinsics, at the recorded resolution.
    if args.camera != "azure":
        _probe = _load_depth_stack(calib_base_dir, side)
        if _probe is None:
            sys.exit(
                f"Error: no 'depth' array in "
                f"{calib_base_dir / (side + '_calibration_rgbd.npz')}. Re-capture with a "
                "depth-enabled camera path."
            )
        _dh, _dw = int(_probe.shape[1]), int(_probe.shape[2])
        from apriltag_image import _camera_params_for
        _fx, _fy, _cx, _cy = _camera_params_for(args.camera, _dw, _dh)
        set_depth_intrinsics(_fx, _fy, _cx, _cy)

        # Undo the ZED disparity error, or this comparison measures the fault
        # instead of the difference between the two translation sources.
        _rgbd = calib_base_dir / f"{side}_calibration_rgbd.npz"
        set_depth_corrector(zed_depth_config.corrector_for(
            args.camera, _fx, unit="mm",
            offset_px_override=args.disparity_offset_px,
            scale_override=args.disparity_scale,
            already_applied_px=zed_depth_config.dataset_applied_offset_px(_rgbd),
            already_applied_scale=zed_depth_config.dataset_applied_scale(_rgbd),
        ))

    if not frames_dir.exists():
        raise RuntimeError(f"Frames directory not found: {frames_dir}")

    depth_stack = _load_depth_stack(calib_base_dir, side)

    out_dir = calib_base_dir / "t_compare_debug"
    enhance_dir = out_dir / "enhanced_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    enhance_dir.mkdir(parents=True, exist_ok=True)

    indices = _parse_indices(args.image_indices)
    if indices is None:
        indices = list(range(args.num_images))

    print("\n" + "=" * 100)
    print("[INFO] Comparing t_apriltag vs t_depth")
    print("=" * 100)
    print("[INFO] calib sequence:", args.calib_seq_name)
    print("[INFO] side:", side)
    print("[INFO] image indices:", indices)
    print("[INFO] output dir:", out_dir)
    print("=" * 100)

    for image_index in indices:
        img_path = frames_dir / f"calibration_{side}_image_{image_index}.png"

        if not img_path.exists():
            print(f"\n[WARN] image {image_index}: missing image file: {img_path}")
            continue

        if image_index >= depth_stack.shape[0]:
            print(f"\n[WARN] image {image_index}: no depth image, depth_stack has shape {depth_stack.shape}")
            continue

        det, T_cam_tag_apriltag, source_name, used_img_path = _detect_with_enhancement_until_success(
            img_path=img_path,
            enhanced_dir=enhance_dir,
            image_index=image_index,
            camera=args.camera,
        )

        if det is None or T_cam_tag_apriltag is None:
            print(f"\n[FAIL] image {image_index}: could not detect tag_id={EXPECTED_TAG_ID}, even after enhancement")
            continue

        center = np.asarray(det.center).astype(float)
        u, v = float(center[0]), float(center[1])

        depth_img = depth_stack[image_index]
        depth_mm = _depth_at_pixel(depth_img, u, v, patch_radius=args.depth_patch_radius)

        if depth_mm is None:
            print(f"\n[FAIL] image {image_index}: detected tag but no valid depth around center ({u:.1f}, {v:.1f})")
            continue

        t_apriltag = T_cam_tag_apriltag[0:3, 3].astype(float)
        t_depth = _pixel_depth_to_camera_point(u, v, depth_mm)

        diff = t_depth - t_apriltag
        diff_norm_mm = np.linalg.norm(diff) * 1000.0

        print("\n" + "-" * 100)
        print(f"[IMAGE {image_index}] detected using: {source_name}")
        print(f"center pixel: u={u:.2f}, v={v:.2f}")
        print(f"depth at center patch median: {depth_mm:.1f} mm")
        print(f"t_apriltag [m]: {t_apriltag}")
        print(f"t_depth    [m]: {t_depth}")
        print(f"diff = t_depth - t_apriltag [mm]: {diff * 1000.0}")
        print(f"|diff| [mm]: {diff_norm_mm:.2f}")

        out_path = out_dir / f"compare_t_image_{image_index:03d}.png"
        _annotate_result(
            original_img_path=img_path,
            used_img_path=used_img_path,
            out_path=out_path,
            image_index=image_index,
            source_name=source_name,
            center_uv=(u, v),
            t_apriltag=t_apriltag,
            t_depth=t_depth,
            depth_mm=depth_mm,
        )

        print(f"[INFO] saved annotation: {out_path}")

    print("\n[DONE]")
    print(f"Open results with:")
    print(f"  xdg-open {out_dir}")


if __name__ == "__main__":
    main()
