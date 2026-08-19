#!/usr/bin/env python3

"""Dual-arm pick of a green object, using the ZED stereo camera.

ZED port of dual_green_pick.py. The Azure original is untouched.

WHAT CHANGED, AND WHY IT IS SHORT
    The Azure coupling was narrow: init_azure(), get_rgbd(), and one grab inside
    the video recorder. `intrinsics` was already a plain (fx, fy, cx, cy) tuple, so
    the green detection, the grasp planning and both arms' motion are unchanged.

    The camera convention also matches exactly: the Azure colour camera and the ZED
    under COORDINATE_SYSTEM.IMAGE are both X-right, Y-down, Z-forward. So the
    back-projection maths carries over with no sign changes.

    Two Azure steps disappear:
      * transformation.depth_image_to_color_camera() -- the ZED's MEASURE.DEPTH is
        already registered to VIEW.LEFT.
      * the k4a manual-exposure dance -- ZedImageConfig covers it.

THE DEPTH CORRECTION IS THE POINT OF THIS SCRIPT
    This camera reports depth ~15% too FAR. capture_rgbd_native() removes it, and
    the transform below was solved from depth corrected by the SAME value. If the
    two ever disagree, the grasp target is wrong by ~200 mm. init_zed() prints the
    active offset so it cannot happen silently.

RUN IT
    cd /home/roahmlab/move_some_robots/crisp_env/crisp_py
    # no robot, no ROS: prints the grasp targets and writes the debug images
    pixi run -e humble python dual_green_pick_zed.py --dry-run
    # the real thing
    pixi run -e humble python dual_green_pick_zed.py
"""

import argparse
import sys
import time
from pathlib import Path
import threading

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from std_msgs.msg import Float64MultiArray

from crisp_py.robot import Robot

# The ZED capture layer lives in zed_capture/. It also owns the depth disparity
# correction, so importing it is what keeps this script consistent with the
# calibration that produced the transforms below.
sys.path.insert(0, str(Path(__file__).resolve().parent / "zed_capture"))
import pyzed.sl as sl          # noqa: E402
import zed_camera as zc        # noqa: E402


# ============================================================
# User config: paths
# ============================================================

# Everything is derived from this file's location, so the script survives a move
# to another machine. The Azure original hardcoded absolute paths.
REPO_ROOT = Path(__file__).resolve().parent
CALIB_DIR = (REPO_ROOT / "hand_to_eye_calibration/roahm-deformable-objects"
             / "captured_calibration_data" / "zed_calib_003")

# USE depth_translation, NOT apriltag_translation. Each transform absorbs the bias
# of the data it was solved from, so it is only valid for the same kind of data.
# This script feeds DEPTH into the transform, so the transform must come from
# depth. See "Which file to deploy" in zed_capture/README.md.
LEFT_TRANSFORM_PATH = str(
    CALIB_DIR / "base2cam_transform_left_nonlinear_opt_depth_translation.npz")
RIGHT_TRANSFORM_PATH = str(
    CALIB_DIR / "base2cam_transform_right_nonlinear_opt_depth_translation.npz")

CONTROL_CONFIG = str(REPO_ROOT / "config/control/default_cartesian_impedance.yaml")

DEBUG_DIR = Path("/tmp/dual_green_pick_zed_debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# User config: video recording with projected grasp dots
# ============================================================

RECORD_VIDEO = True

# Keep the grasp videos with the calibration they were produced from, so a recording
# can always be traced to the transform that aimed it. A subdirectory, NOT the calib
# dir itself, so pick debris never mixes with the calibration dataset.
PICK_RUN_DIR = CALIB_DIR / "pick_runs"
PICK_RUN_DIR.mkdir(parents=True, exist_ok=True)

# Timestamped, because the point of keeping them here is to compare runs. The still
# images below still go to DEBUG_DIR and are overwritten each run.
_RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")
VIDEO_PATH = PICK_RUN_DIR / f"pick_{_RUN_STAMP}.mp4"


def set_calib_sequence(seq_name):
    """Re-point the transforms AND every kept output at another calibration sequence.

    The module-level paths above are derived from zed_calib_003 at import time,
    so a run against another sequence would still file its grasp videos under
    003 -- breaking the whole point of PICK_RUN_DIR (a recording must trace to
    the calibration that aimed it). Call this (or pass --calib-seq-name) before
    main().
    """
    global CALIB_DIR, LEFT_TRANSFORM_PATH, RIGHT_TRANSFORM_PATH
    global PICK_RUN_DIR, VIDEO_PATH
    CALIB_DIR = (REPO_ROOT / "hand_to_eye_calibration/roahm-deformable-objects"
                 / "captured_calibration_data" / seq_name)
    if not CALIB_DIR.is_dir():
        raise RuntimeError(f"no such calibration sequence: {CALIB_DIR}")
    LEFT_TRANSFORM_PATH = str(
        CALIB_DIR / "base2cam_transform_left_nonlinear_opt_depth_translation.npz")
    RIGHT_TRANSFORM_PATH = str(
        CALIB_DIR / "base2cam_transform_right_nonlinear_opt_depth_translation.npz")
    PICK_RUN_DIR = CALIB_DIR / "pick_runs"
    PICK_RUN_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_PATH = PICK_RUN_DIR / f"pick_{_RUN_STAMP}.mp4"
    print(f"[INFO] calibration sequence: {seq_name} -- videos go to {PICK_RUN_DIR}")
VIDEO_FPS = 15.0          # the ZED caps at 15 fps in HD2K
VIDEO_LEAD_IN_SEC = 2.0   # record this long before the arms start moving
VIDEO_TAIL_SEC = 2.0      # record this long after the lift finishes
VIDEO_DOT_RADIUS = 20     # scaled 1.725x: HD2K is 2208 wide vs the Azure's 1280
VIDEO_SCALE = 0.5         # shrink the mp4; 2208x1242 frames make a huge file
VIDEO_DRAW_CENTER = False  # keep video clean: only blue/green grasp points by default

# Overwrite the same mp4 path every run instead of appending/keeping stale files.
VIDEO_OVERWRITE = True


# ============================================================
# User config: ZED camera
# ============================================================

# These MATCH what the calibration capture used (zed_calib_rgbd.py). Do not change
# one without the other: confidence and texture decide WHICH depth pixels survive,
# so calibrating on one population and deploying on another introduces a bias.
ZED_RESOLUTION = "HD2K"          # 2208x1242, exactly 16:9, best depth on a ZED1
ZED_FPS = 15
ZED_DEPTH_MODE = "NEURAL_PLUS"
ZED_DEPTH_MIN_M = 0.20
ZED_DEPTH_MAX_M = 20.0
ZED_CONFIDENCE = 47
# Temporal depth fusion, [0,100]; the SDK default (and zed_camera's) is 10, the
# calibration capture used 10. Stabilization fuses depth over TIME, so grabs
# taken right after something moved carry stale surfaces -- keep the warmup
# frames in mind if this is raised further.
ZED_DEPTH_STABILIZATION = 16

# Per-pixel median over N grabs. The scene is static while we detect, so this is
# nearly free and it removes stereo speckle without bleeding across depth edges.
ZED_MEDIAN_FRAMES = 5
ZED_WARMUP_FRAMES = 30

# Auto exposure. aec_agc=1 is REQUIRED to actually restore auto: ZED settings
# persist across opens, so exposure=None only means "leave whatever was last set".
# A manual value chosen under different lighting is what ruined zed_calib_002.
ZED_AEC_AGC = 0
ZED_EXPOSURE = 10
ZED_GAIN = None


# ============================================================
# User config: grasp geometry
# ============================================================

# Camera frame convention:
# X_cam roughly points image-right, Y_cam image-down, Z_cam forward/depth.
#
# Red dot    = physical-highest center / centroid target
# Blue dot   = left grasp point from the center
# Green dot  = right grasp point from the center
#
# These offsets are applied in CAMERA frame, so the debug image points move left/right.
OFFSET_CAM_LEFT = np.array([-0.10, 0.00, 0.00, 0.00])
OFFSET_CAM_RIGHT = np.array([+0.10, 0.00, 0.00, 0.00])

# The detected point is the physical top/highest surface.
# Real grasp target can be slightly LOWER than that top point.
# This is applied in ROBOT BASE frame z after camera->base transform.
# 0.01 means actual grasp target is 1 cm lower than the detected high surface.
LEFT_GRASP_POINT_OFFSET_BASE = np.array([0.00, 0.00, -0.004])
RIGHT_GRASP_POINT_OFFSET_BASE = np.array([0.00, 0.00, -0.004])


# ============================================================
# User config: motion
# ============================================================

SPEED_APPROACH = 0.035
SPEED_DESCEND = 0.018
SPEED_LIFT = 0.018

PRE_GRASP_Z_OFFSET = 0.12
GRASP_Z_OFFSET = -0.005
LIFT_Z_OFFSET = 0.18

# --test-run: full approach, but the descend stops this far ABOVE the grasp
# point and the grippers never close. 2 cm below PRE_GRASP_Z_OFFSET, so the
# test still exercises a visible descend step.
TEST_RUN_HOVER_M = 0.10

# --test-run: wait this long after "reached" before comparing the measured EE
# pose against the commanded hover point, so the impedance controller has
# settled and the number is steady-state error, not the tail of the motion.
TEST_RUN_SETTLE_S = 1.5

# --test-run settle-and-correct: the impedance controller has no integrator,
# so it settles a constant offset (residual force / stiffness) away from the
# target. Re-commanding the mirrored point cancels that offset regardless of
# its cause. Two passes remove most of it; more passes gain nothing.
TEST_RUN_CORRECT_ITERS = 2
# Stop correcting once the error is below this; the controller cannot do
# better than its own noise floor.
TEST_RUN_CORRECT_DONE_M = 0.001
# Refuse to correct an error larger than this: a big error means something
# other than settle offset is wrong (collision, wrong target, bad pose feed).
TEST_RUN_CORRECT_MAX_M = 0.03

CONFIRM_BEFORE_MOVE = True

# Conservative workspace limits.
# Do NOT loosen these just because a point fails. If it fails, the perception target is wrong.
LEFT_WORKSPACE_MIN = np.array([0.00, -0.90, -0.03])
LEFT_WORKSPACE_MAX = np.array([0.80,  0.40, 0.80])

RIGHT_WORKSPACE_MIN = np.array([0.00, -0.40, -0.03])
RIGHT_WORKSPACE_MAX = np.array([0.90,  0.90, 0.80])


# ============================================================
# User config: green detection
# ============================================================

LOWER_GREEN = np.array([30, 25, 20])
UPPER_GREEN = np.array([100, 255, 255])
MIN_GREEN_AREA = 60       # scaled ~3x: pixel AREA grows as 1.725^2

# ROI in image ratio. This avoids table reflections / robot bases.
ROI_X_MIN_RATIO = 0.20
ROI_X_MAX_RATIO = 0.80
ROI_Y_MIN_RATIO = 0.30
ROI_Y_MAX_RATIO = 0.92

# Physical highest surface selection.
# We compute each green pixel's point in robot base frame, then select the top-z band.
TOP_HEIGHT_PERCENTILE = 50.0
TOP_HEIGHT_BAND_M = 0.020
TOP_HEIGHT_MIN_PIXELS = 75        # scaled ~3x (area)
TOP_HEIGHT_FALLBACK_TOP_N = 240   # scaled ~3x (area)

# Remove obviously impossible object heights in base z.
# Table is around 0.016 m in your previous tests; box/cloth should not be at 0.6 m.
OBJECT_TOP_Z_MIN = -0.02
OBJECT_TOP_Z_MAX = 0.35

# If depth image is noisy, subsampling keeps computation lighter.
PIXEL_SAMPLE_STRIDE = 3   # HD2K has ~3x the pixels; keeps the sample count similar

# Grasp-point selection constraint:
# left/right desired points are computed from the center, but the final selected
# pixels are snapped back onto REAL green pixels. This prevents grasp points from
# floating outside the object after offset.
GRASP_SURFACE_BAND_M = 0.060       # candidate grasp pixels must be within this height band below the top
GRASP_SNAP_MAX_PIXEL_DIST = 380.0  # warn if desired point is farther than this from nearest green candidate (scaled 1.725x)
HEIGHT_SCORE_WEIGHT_PX = 138.0     # prefer physically higher candidate pixels while snapping (scaled 1.725x)


# ============================================================
# Transform
# ============================================================

def load_camera_to_base_transform(path: str, name: str) -> np.ndarray:
    data = np.load(path)
    T_saved = data["arr_0"]

    # calculate_base_to_cam.py saves base->cam; grasping needs camera->base.
    T_camera_to_base = np.linalg.inv(T_saved)

    print(f"\n[INFO] {name} transform file:", path)
    print(f"[INFO] {name} T_saved:")
    print(T_saved)
    print(f"[INFO] {name} T_camera_to_base = inv(T_saved):")
    print(T_camera_to_base)

    return T_camera_to_base


# ============================================================
# ZED stereo camera
# ============================================================

def init_zed():
    """Open the ZED with the settings the calibration capture used.

    Returns:
        (zed, runtime, intrinsics) where intrinsics is (fx, fy, cx, cy) for the
        RECTIFIED left camera at the live resolution. Same 4-tuple shape the Azure
        path returned, so nothing downstream changes.

    Raises:
        RuntimeError with an actionable message if open() fails -- most often
        because ZED_Depth_Viewer still holds the camera (USB ZEDs are one-process).
    """
    print("[INFO] Opening ZED...")
    init_cfg = zc.ZedInitConfig(
        resolution=ZED_RESOLUTION,
        fps=ZED_FPS,
        depth_mode=ZED_DEPTH_MODE,
        depth_min_m=ZED_DEPTH_MIN_M,
        depth_max_m=ZED_DEPTH_MAX_M,
        depth_stabilization=ZED_DEPTH_STABILIZATION,
    )
    image_cfg = zc.ZedImageConfig(
        exposure=ZED_EXPOSURE, gain=ZED_GAIN, aec_agc=ZED_AEC_AGC)

    zed, resolved = zc.open_zed(init_cfg, image_cfg)
    runtime = zc.build_runtime_parameters(zc.ZedRuntimeConfig(confidence=ZED_CONFIDENCE))
    zc.warmup(zed, runtime, ZED_WARMUP_FRAMES)

    intr = zc.get_intrinsics(zed)["native"]
    intrinsics = (intr["fx"], intr["fy"], intr["cx"], intr["cy"])

    print(f"[INFO] ZED {resolved['width']}x{resolved['height']} @ {resolved['fps']:g} fps, "
          f"depth={ZED_DEPTH_MODE}, serial={resolved.get('serial_number')}")
    print("[INFO] ZED intrinsics fx, fy, cx, cy:", intrinsics)

    # The transform was solved from depth corrected by this exact offset. If these
    # disagree the grasp target is wrong by ~200 mm, so make it loud either way.
    if zc.DEPTH_DISPARITY_OFFSET_PX:
        print(f"[INFO] depth disparity correction ACTIVE: "
              f"{zc.DEPTH_DISPARITY_OFFSET_PX:+.2f} px")
    else:
        print("[WARN] depth disparity correction is OFF. This camera reads ~15% too")
        print("[WARN] FAR, and the loaded transform assumes the correction is on.")
        print("[WARN] Check zed_capture/zed_depth_correction.json before trusting this.")

    return zed, runtime, intrinsics


def load_capture(run_dir):
    """Load a capture written by zed_capture/capture_zed_sam_mask.py.

    Needs, at NATIVE resolution:
        rgb_full.png        BGR
        depth_mm_full.npy   uint16 mm, 0 = invalid
        mask_full.png       uint8, 255 = target   (from --sam-checkpoint)

    Returns:
        (color_bgr, depth_mm, mask, intrinsics)

    Raises:
        RuntimeError naming the missing file, or if the capture's depth was not
        corrected with the same disparity offset this deployment expects. That
        check is the whole point: the transform is only valid for depth corrected
        by the value it was solved with.
    """
    run_dir = Path(run_dir)
    need = {"rgb_full.png": None, "depth_mm_full.npy": None, "mask_full.png": None}
    for name in need:
        if not (run_dir / name).is_file():
            raise RuntimeError(
                f"{run_dir / name} is missing. Create the capture with:\n"
                f"  pixi run -e sam2 python zed_capture/capture_zed_sam_mask.py \\\n"
                f"      --sam-checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \\\n"
                f"      --output {run_dir}\n"
                "mask_full.png needs --sam-checkpoint; without it no mask is written.")

    color_bgr = cv2.imread(str(run_dir / "rgb_full.png"))
    depth_mm = np.load(str(run_dir / "depth_mm_full.npy"))
    mask = cv2.imread(str(run_dir / "mask_full.png"), cv2.IMREAD_GRAYSCALE)

    # The capture ALREADY applied the disparity correction, so this script must not
    # apply it again -- 32 px would read too near instead of too far, and nothing
    # would crash. Verify the value instead of assuming it.
    cfg_path = run_dir / "capture_config.json"
    applied = None
    if cfg_path.is_file():
        import json
        cfg = json.load(open(cfg_path))
        applied = cfg.get("capture", {}).get("disparity_offset_px")
    expected = zc.DEPTH_DISPARITY_OFFSET_PX
    if applied is None:
        print(f"[WARN] {cfg_path} does not record disparity_offset_px. This script "
              f"expects depth already corrected by {expected:+.2f} px and cannot verify it.")
    elif abs(float(applied) - float(expected)) > 1e-6:
        raise RuntimeError(
            f"depth correction mismatch: the capture applied {float(applied):+.2f} px, "
            f"but the deployed transform was solved for {expected:+.2f} px. "
            "Re-capture, or align zed_capture/zed_depth_correction.json.")
    else:
        print(f"[INFO] capture depth already corrected by {float(applied):+.2f} px -- matches")

    intr_path = run_dir / "intrinsics.json"
    if not intr_path.is_file():
        raise RuntimeError(f"{intr_path} is missing; intrinsics are needed to unproject.")
    import json
    native = json.load(open(intr_path))["native"]
    intrinsics = (native["fx"], native["fy"], native["cx"], native["cy"])

    print(f"[INFO] loaded capture {run_dir}")
    print(f"       color {color_bgr.shape}  depth {depth_mm.shape}  "
          f"mask {int((mask>0).sum())} px")
    print(f"       intrinsics fx,fy,cx,cy = {intrinsics}")
    return color_bgr, depth_mm, mask, intrinsics


def get_rgbd(zed, runtime):
    """Grab one RGB-D frame. Returns (color_bgr, depth_mm), like the Azure path.

    depth_mm is uint16 millimetres with 0 = invalid -- the same convention the Azure
    path produced, which is why the green-detection code needs no change. The depth
    is already registered to VIEW.LEFT, so there is no alignment step.

    The disparity-offset correction is applied inside capture_rgbd_native().
    """
    color_bgr, _depth_m, depth_mm = zc.capture_rgbd_native(
        zed, runtime, n_median=ZED_MEDIAN_FRAMES)
    cv2.imwrite(str(DEBUG_DIR / "raw_color.png"), color_bgr)
    return color_bgr, depth_mm


def grab_color_only(zed, runtime):
    """One colour frame for the video recorder, with no depth retrieval.

    capture_rgbd_native() would also pull MEASURE.DEPTH, which at HD2K NEURAL_PLUS
    is the expensive part and is not needed for a video overlay.

    Returns the frame, or None if the grab failed (the recorder just skips it).
    """
    if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        return None
    mat = sl.Mat()
    if zed.retrieve_image(mat, sl.VIEW.LEFT) != sl.ERROR_CODE.SUCCESS:
        return None
    # .copy() via cvtColor: get_data() hands back an array whose buffer belongs to
    # the Mat, so it must not outlive it.
    return cv2.cvtColor(mat.get_data(), cv2.COLOR_BGRA2BGR).copy()


# ============================================================
# Camera geometry / workspace
# ============================================================

def pixel_depth_to_camera_point(u, v, depth_mm, intrinsics):
    fx, fy, cx, cy = intrinsics

    Z = depth_mm / 1000.0
    X = (float(u) - cx) * Z / fx
    Y = (float(v) - cy) * Z / fy

    return np.array([X, Y, Z, 1.0], dtype=float)


def pixels_depth_to_camera_points(us, vs, depth_mm, intrinsics):
    fx, fy, cx, cy = intrinsics

    Z = depth_mm.astype(np.float64) / 1000.0
    X = (us.astype(np.float64) - cx) * Z / fx
    Y = (vs.astype(np.float64) - cy) * Z / fy

    ones = np.ones_like(Z)
    return np.stack([X, Y, Z, ones], axis=1)


def camera_point_to_pixel(p_cam, intrinsics):
    fx, fy, cx, cy = intrinsics
    X, Y, Z = p_cam[:3]

    if Z <= 0:
        return None

    u = int(round(fx * X / Z + cx))
    v = int(round(fy * Y / Z + cy))

    return u, v


def _draw_grasp_dots_on_frame(frame_bgr, p_center_cam, p_left_cam, p_right_cam,
                              intrinsics, p_left_hover_cam=None,
                              p_right_hover_cam=None):
    """Overlay fixed camera-frame grasp points onto a live color frame.

    Blue  = left grasp point
    Green = right grasp point
    Red   = center point, optional via VIDEO_DRAW_CENTER
    Rings (test run only) = EE hover targets, TEST_RUN_HOVER_M above the grasps.
    Filled dot vs ring keeps "where the cloth is" and "where the EE will stop"
    tellable at a glance in the same arm colour.
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    def to_px(p_cam):
        px = camera_point_to_pixel(p_cam, intrinsics)
        if px is None:
            return None
        u, v = px
        return (u, v) if (0 <= u < w and 0 <= v < h) else None

    def draw_point(p_cam, color_bgr, radius):
        px = to_px(p_cam)
        if px is None:
            return
        # Black outline makes the dot visible on green cloth / robot / table.
        cv2.circle(out, px, radius + 4, (0, 0, 0), -1)
        cv2.circle(out, px, radius, color_bgr, -1)

    def draw_target_ring(p_cam, color_bgr, radius):
        px = to_px(p_cam)
        if px is None:
            return
        u, v = px
        cv2.circle(out, (u, v), radius, (0, 0, 0), 8)
        cv2.circle(out, (u, v), radius, color_bgr, 4)
        for du, dv in ((1, 0), (0, 1)):
            a = (u - du * radius, v - dv * radius)
            b = (u + du * radius, v + dv * radius)
            cv2.line(out, a, b, (0, 0, 0), 4)
            cv2.line(out, a, b, color_bgr, 2)

    if VIDEO_DRAW_CENTER:
        draw_point(p_center_cam, (0, 0, 255), VIDEO_DOT_RADIUS)

    draw_point(p_left_cam, (255, 0, 0), VIDEO_DOT_RADIUS)
    draw_point(p_right_cam, (0, 255, 0), VIDEO_DOT_RADIUS)

    if p_left_hover_cam is not None:
        draw_target_ring(p_left_hover_cam, (255, 0, 0), VIDEO_DOT_RADIUS + 10)
    if p_right_hover_cam is not None:
        draw_target_ring(p_right_hover_cam, (0, 255, 0), VIDEO_DOT_RADIUS + 10)

    return out


class GraspOverlayVideoRecorder:
    """Continuously record ZED colour video with the fixed grasp dots overlaid.

    Uses the SAME already-open camera. A USB ZED allows only ONE process, so never
    start a second capture (or ZED_Depth_Viewer) while this runs.

    The grabbing happens on a background thread while the main thread only commands
    the robots, so no two threads touch the camera at once.
    """

    def __init__(self, zed, runtime, intrinsics, p_center_cam, p_left_cam,
                 p_right_cam, out_path, p_left_hover_cam=None,
                 p_right_hover_cam=None):
        self.zed = zed
        self.runtime = runtime
        self.intrinsics = intrinsics
        self.p_center_cam = p_center_cam.copy()
        self.p_left_cam = p_left_cam.copy()
        self.p_right_cam = p_right_cam.copy()
        self.p_left_hover_cam = (None if p_left_hover_cam is None
                                 else np.asarray(p_left_hover_cam, dtype=float).copy())
        self.p_right_hover_cam = (None if p_right_hover_cam is None
                                  else np.asarray(p_right_hover_cam, dtype=float).copy())
        self.out_path = Path(out_path)
        self.stop_event = threading.Event()
        self.thread = None
        self.writer = None
        self.frames_written = 0

    def start(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        if VIDEO_OVERWRITE and self.out_path.exists():
            try:
                self.out_path.unlink()
                print(f"[VIDEO] Removed old video so this run overwrites it: {self.out_path}")
            except Exception as e:
                print(f"[VIDEO-WARN] Could not remove old video {self.out_path}: {e}")
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"[VIDEO] Recording started: {self.out_path}")

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        print(f"[VIDEO] Recording stopped. frames={self.frames_written}, file={self.out_path}")

    def _open_writer_if_needed(self, frame_bgr):
        if self.writer is not None:
            return
        h, w = frame_bgr.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.out_path), fourcc, VIDEO_FPS, (w, h))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {self.out_path}")

    def _run(self):
        period = 1.0 / max(float(VIDEO_FPS), 1.0)
        last_t = 0.0

        while not self.stop_event.is_set():
            now = time.time()
            if now - last_t < period:
                time.sleep(0.002)
                continue
            last_t = now

            try:
                frame_bgr = grab_color_only(self.zed, self.runtime)
                if frame_bgr is None:
                    continue

                frame_bgr = _draw_grasp_dots_on_frame(
                    frame_bgr=frame_bgr,
                    p_center_cam=self.p_center_cam,
                    p_left_cam=self.p_left_cam,
                    p_right_cam=self.p_right_cam,
                    intrinsics=self.intrinsics,
                    p_left_hover_cam=self.p_left_hover_cam,
                    p_right_hover_cam=self.p_right_hover_cam,
                )

                if VIDEO_SCALE != 1.0:
                    frame_bgr = cv2.resize(
                        frame_bgr, None, fx=VIDEO_SCALE, fy=VIDEO_SCALE,
                        interpolation=cv2.INTER_AREA)

                self._open_writer_if_needed(frame_bgr)
                self.writer.write(frame_bgr)
                self.frames_written += 1

            except Exception as e:
                print(f"[VIDEO-WARN] recording frame failed: {e}")
                time.sleep(0.05)


def check_workspace(p, side):
    if side == "left":
        lo, hi = LEFT_WORKSPACE_MIN, LEFT_WORKSPACE_MAX
    else:
        lo, hi = RIGHT_WORKSPACE_MIN, RIGHT_WORKSPACE_MAX

    if np.any(p < lo) or np.any(p > hi):
        raise RuntimeError(
            f"\n{side} target outside safety workspace. Robot will NOT move.\n"
            f"p = {p}\n"
            f"min = {lo}\n"
            f"max = {hi}\n"
        )


def workspace_mask(points_base, side):
    if side == "left":
        lo, hi = LEFT_WORKSPACE_MIN, LEFT_WORKSPACE_MAX
    else:
        lo, hi = RIGHT_WORKSPACE_MIN, RIGHT_WORKSPACE_MAX

    return np.all((points_base >= lo) & (points_base <= hi), axis=1)


# ============================================================
# Detection
# ============================================================

def build_green_mask(color_bgr):
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    h_img, w_img = mask.shape[:2]

    roi = np.zeros_like(mask)
    x_min = int(ROI_X_MIN_RATIO * w_img)
    x_max = int(ROI_X_MAX_RATIO * w_img)
    y_min = int(ROI_Y_MIN_RATIO * h_img)
    y_max = int(ROI_Y_MAX_RATIO * h_img)
    roi[y_min:y_max, x_min:x_max] = 255
    mask = cv2.bitwise_and(mask, roi)

    cv2.imwrite(str(DEBUG_DIR / "green_mask_debug.png"), mask)
    return mask, (x_min, y_min, x_max, y_max)


def find_best_green_contour(mask, color_bgr, roi_box):
    x_min, y_min, x_max, y_max = roi_box

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug = color_bgr.copy()
    cv2.rectangle(debug, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

    if len(contours) == 0:
        cv2.imwrite(str(DEBUG_DIR / "green_detection_failed.png"), debug)
        raise RuntimeError("No green object detected inside ROI.")

    candidates = []
    roi_cx = 0.5 * (x_min + x_max)
    roi_cy = 0.5 * (y_min + y_max)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_GREEN_AREA:
            continue

        M = cv2.moments(contour)
        if abs(M["m00"]) < 1e-6:
            continue

        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])
        dist_to_roi_center = np.hypot(u - roi_cx, v - roi_cy)

        score = area - 0.05 * dist_to_roi_center
        candidates.append({
            "contour": contour,
            "area": area,
            "u": u,
            "v": v,
            "score": score,
        })

    if len(candidates) == 0:
        cv2.imwrite(str(DEBUG_DIR / "green_too_small.png"), debug)
        raise RuntimeError("Green object detected, but all contours are too small.")

    best = max(candidates, key=lambda x: x["score"])
    return best["contour"], best["area"]


def _choose_grasp_candidate_on_green(
    side,
    desired_cam,
    candidate_indices,
    pts_cam,
    us,
    vs,
    top_score_z_all,
    intrinsics,
):
    """
    The requested left/right offset gives a DESIRED location, but the final point
    must lie on the green object. So we project desired_cam into the image, then
    pick the nearest candidate green pixel in the allowed physical-height band.
    """
    if len(candidate_indices) == 0:
        raise RuntimeError(f"No green candidate pixels available for {side} grasp.")

    desired_px = camera_point_to_pixel(desired_cam, intrinsics)
    if desired_px is None:
        raise RuntimeError(f"Desired {side} grasp point projects behind camera.")

    du = us[candidate_indices].astype(np.float64) - float(desired_px[0])
    dv = vs[candidate_indices].astype(np.float64) - float(desired_px[1])
    pixel_dist = np.sqrt(du * du + dv * dv)

    # Among nearby green pixels, prefer higher physical-z pixels.
    cand_z = top_score_z_all[candidate_indices]
    z_span = max(float(np.max(cand_z) - np.min(cand_z)), 1e-6)
    z_penalty_px = HEIGHT_SCORE_WEIGHT_PX * (float(np.max(cand_z)) - cand_z) / z_span

    score = pixel_dist + z_penalty_px
    local_best = int(np.argmin(score))
    best_idx = int(candidate_indices[local_best])

    nearest_dist = float(pixel_dist[local_best])
    selected_cam = pts_cam[best_idx].copy()
    selected_cam[3] = 1.0
    selected_px = (int(us[best_idx]), int(vs[best_idx]))

    if nearest_dist > GRASP_SNAP_MAX_PIXEL_DIST:
        print(
            f"[WARN] {side} desired grasp is far from green object: "
            f"desired_px={desired_px}, selected_px={selected_px}, "
            f"pixel_dist={nearest_dist:.1f}. Check OFFSET_CAM_{side.upper()} or mask."
        )

    return selected_cam, desired_px, selected_px, nearest_dist


def detect_green_physical_highest_center_and_grasps(
    color_bgr,
    depth,
    intrinsics,
    T_camera_to_left_base,
    T_camera_to_right_base,
    external_mask=None,
):
    """
    Detect green object's physical-highest center, then compute two grasp points.

    Important difference from the previous version:
    - Offset points are NOT blindly accepted.
    - Offset only defines desired left/right locations.
    - Final left/right grasp pixels are snapped to valid green pixels on the object.
    - Therefore the debug blue/green dots must lie on the green object.
    """
    if external_mask is not None:
        # A SAM mask replaces the green-colour test entirely.
        #
        # WHY: measured on this rig, the cloth and the black backdrop are
        # colorimetrically IDENTICAL -- at auto exposure S median 25 vs 21, at
        # exposure 20 S median 64 vs 64. No hue/sat/value threshold separates them.
        # Worse, at auto exposure the cloth interior blows out to S=2, and a white
        # pixel has no colour at all, so 96.3% of it cannot pass a green test. The
        # colour mask therefore picked up the backdrop at ~2.0 m and put the grasp
        # target in mid-air. Depth separates the two perfectly; the only thing
        # missing was a trustworthy mask, which is what SAM provides.
        mask = (np.asarray(external_mask) > 0).astype(np.uint8) * 255
        if mask.shape[:2] != color_bgr.shape[:2]:
            raise RuntimeError(
                f"mask shape {mask.shape[:2]} != image shape {color_bgr.shape[:2]}. "
                "Use mask_full.png (native resolution), not mask.png.")
        # No ROI crop: the user pointed at the object, so trusting them beats
        # second-guessing with an image-fraction box.
        roi_box = (0, 0, mask.shape[1], mask.shape[0])
        contour, area = find_best_green_contour(mask, color_bgr, roi_box)
        print(f"[INFO] using the supplied SAM mask: {int((mask>0).sum())} px")
    else:
        mask, roi_box = build_green_mask(color_bgr)
        contour, area = find_best_green_contour(mask, color_bgr, roi_box)
    x_min, y_min, x_max, y_max = roi_box

    contour_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)

    valid = (contour_mask > 0) & (depth > 0)

    if PIXEL_SAMPLE_STRIDE > 1:
        sample_mask = np.zeros_like(valid, dtype=bool)
        sample_mask[::PIXEL_SAMPLE_STRIDE, ::PIXEL_SAMPLE_STRIDE] = True
        valid = valid & sample_mask

    vs, us = np.where(valid)
    if len(us) == 0:
        cv2.imwrite(str(DEBUG_DIR / "green_no_valid_depth.png"), color_bgr)
        raise RuntimeError("Green object detected, but no valid depth inside contour.")

    depth_vals = depth[vs, us].astype(np.float64)
    pts_cam = pixels_depth_to_camera_points(us, vs, depth_vals, intrinsics)

    # Convert all green candidate pixels to both bases.
    pts_left_base_h = (T_camera_to_left_base @ pts_cam.T).T
    pts_right_base_h = (T_camera_to_right_base @ pts_cam.T).T
    pts_left_base = pts_left_base_h[:, :3] / pts_left_base_h[:, 3:4]
    pts_right_base = pts_right_base_h[:, :3] / pts_right_base_h[:, 3:4]

    left_z_ok = (pts_left_base[:, 2] >= OBJECT_TOP_Z_MIN) & (pts_left_base[:, 2] <= OBJECT_TOP_Z_MAX)
    right_z_ok = (pts_right_base[:, 2] >= OBJECT_TOP_Z_MIN) & (pts_right_base[:, 2] <= OBJECT_TOP_Z_MAX)

    # Physical top score: average the two calibrated base-z estimates.
    top_score_z_all = 0.5 * (pts_left_base[:, 2] + pts_right_base[:, 2])

    # Basic candidate set: green pixels with plausible physical height.
    keep = left_z_ok & right_z_ok
    if np.count_nonzero(keep) == 0:
        debug = color_bgr.copy()
        cv2.drawContours(debug, [contour], -1, (0, 0, 255), 2)
        cv2.imwrite(str(DEBUG_DIR / "green_physical_highest_no_valid_candidates.png"), debug)
        raise RuntimeError(
            "No valid green pixels after physical height filtering. "
            "Check transform or OBJECT_TOP_Z_MIN/MAX."
        )

    kept_indices = np.where(keep)[0]
    kept_z = top_score_z_all[kept_indices]

    # Highest physical region for CENTER computation.
    z_thresh = np.percentile(kept_z, TOP_HEIGHT_PERCENTILE)
    max_z = float(np.max(kept_z))
    top_band = kept_z >= max(z_thresh, max_z - TOP_HEIGHT_BAND_M)

    if np.count_nonzero(top_band) < TOP_HEIGHT_MIN_PIXELS:
        order = np.argsort(kept_z)[::-1]
        n = min(TOP_HEIGHT_FALLBACK_TOP_N, len(order))
        top_kept_indices = kept_indices[order[:n]]
    else:
        top_kept_indices = kept_indices[top_band]

    # One center point from physical-highest region.
    top_pts_cam = pts_cam[top_kept_indices]
    p_center_cam = np.median(top_pts_cam, axis=0)
    p_center_cam[3] = 1.0

    top_us = us[top_kept_indices]
    top_vs = vs[top_kept_indices]
    center_u = int(round(np.mean(top_us)))
    center_v = int(round(np.mean(top_vs)))

    # Candidate set for grasp points: still on green object, still physically high,
    # but slightly larger band than top center so left/right points can exist on cloth.
    grasp_surface = keep & (top_score_z_all >= max_z - GRASP_SURFACE_BAND_M)
    grasp_candidate_indices = np.where(grasp_surface)[0]

    if len(grasp_candidate_indices) < TOP_HEIGHT_MIN_PIXELS:
        print(
            f"[WARN] Too few grasp candidates in top surface band: {len(grasp_candidate_indices)}. "
            "Falling back to all plausible green pixels."
        )
        grasp_candidate_indices = kept_indices

    # Desired positions from center offsets.
    desired_left_cam = p_center_cam + OFFSET_CAM_LEFT
    desired_right_cam = p_center_cam + OFFSET_CAM_RIGHT

    # Final positions are snapped to actual green pixels.
    p_left_cam, desired_left_px, selected_left_px, left_snap_dist = _choose_grasp_candidate_on_green(
        side="left",
        desired_cam=desired_left_cam,
        candidate_indices=grasp_candidate_indices,
        pts_cam=pts_cam,
        us=us,
        vs=vs,
        top_score_z_all=top_score_z_all,
        intrinsics=intrinsics,
    )
    p_right_cam, desired_right_px, selected_right_px, right_snap_dist = _choose_grasp_candidate_on_green(
        side="right",
        desired_cam=desired_right_cam,
        candidate_indices=grasp_candidate_indices,
        pts_cam=pts_cam,
        us=us,
        vs=vs,
        top_score_z_all=top_score_z_all,
        intrinsics=intrinsics,
    )

    # Convert selected grasp pixels to bases.
    p_center_left_base_h = T_camera_to_left_base @ p_center_cam
    p_center_right_base_h = T_camera_to_right_base @ p_center_cam
    p_center_left_base = p_center_left_base_h[:3] / p_center_left_base_h[3]
    p_center_right_base = p_center_right_base_h[:3] / p_center_right_base_h[3]

    p_left_base_raw_h = T_camera_to_left_base @ p_left_cam
    p_right_base_raw_h = T_camera_to_right_base @ p_right_cam
    p_left_base_raw = p_left_base_raw_h[:3] / p_left_base_raw_h[3]
    p_right_base_raw = p_right_base_raw_h[:3] / p_right_base_raw_h[3]

    # Actual grasp is slightly lower than detected surface point.
    # Apply final per-arm grasp offsets in each robot base frame.
    # This is added directly to the initially computed grasp coordinate.
    # Example: [0, 0, -0.010] means grasp 1 cm lower in that arm's base z direction.
    p_left_base = p_left_base_raw + LEFT_GRASP_POINT_OFFSET_BASE
    p_right_base = p_right_base_raw + RIGHT_GRASP_POINT_OFFSET_BASE

    # Clean debug image: only show the three final points.
    # Red  = physical-highest center.
    # Blue = final left grasp point, snapped onto green object pixels.
    # Green = final right grasp point, snapped onto green object pixels.
    top_mask = np.zeros(mask.shape, dtype=np.uint8)
    top_mask[top_vs, top_us] = 255

    debug = color_bgr.copy()

    center_px = camera_point_to_pixel(p_center_cam, intrinsics)
    left_px = camera_point_to_pixel(p_left_cam, intrinsics)
    right_px = camera_point_to_pixel(p_right_cam, intrinsics)

    if center_px is not None:
        cv2.circle(debug, center_px, 10, (0, 0, 255), -1)
    else:
        cv2.circle(debug, (center_u, center_v), 10, (0, 0, 255), -1)

    if left_px is not None:
        cv2.circle(debug, left_px, 10, (255, 0, 0), -1)

    if right_px is not None:
        cv2.circle(debug, right_px, 10, (0, 255, 0), -1)

    # Save the same clean three-dot view under this name, so your old xdg-open command still works.
    cv2.imwrite(str(DEBUG_DIR / "green_physical_highest_detection_debug.png"), debug)

    print(f"[INFO] Green contour area: {area:.1f}")
    print(f"[INFO] Candidate pixels total: {len(us)}")
    print(f"[INFO] Candidate pixels kept after height filter: {np.count_nonzero(keep)}")
    print(f"[INFO] Top-region pixels used for center: {len(top_kept_indices)}")
    print(f"[INFO] Grasp candidate pixels on green: {len(grasp_candidate_indices)}")
    print(f"[INFO] Top-score z max: {max_z:.4f} m, percentile threshold: {z_thresh:.4f} m")
    print(f"[INFO] Top center pixel approx: u={center_u}, v={center_v}")
    print(f"[INFO] Left desired px {desired_left_px} -> selected green px {selected_left_px}, snap_dist={left_snap_dist:.1f}px")
    print(f"[INFO] Right desired px {desired_right_px} -> selected green px {selected_right_px}, snap_dist={right_snap_dist:.1f}px")
    print("[INFO] LEFT_GRASP_POINT_OFFSET_BASE  =", LEFT_GRASP_POINT_OFFSET_BASE)
    print("[INFO] RIGHT_GRASP_POINT_OFFSET_BASE =", RIGHT_GRASP_POINT_OFFSET_BASE)

    print("\n[INFO] Center base estimate:")
    print("  center in left base :", p_center_left_base)
    print("  center in right base:", p_center_right_base)

    print("\n[INFO] Raw base grasp points before downward offset:")
    print("  p_left_base_raw :", p_left_base_raw)
    print("  p_right_base_raw:", p_right_base_raw)

    print("\n[INFO] Actual base grasp points after downward offset:")
    print("  p_left_base :", p_left_base)
    print("  p_right_base:", p_right_base)

    return {
        "contour": contour,
        "top_mask": top_mask,
        "p_center_cam": p_center_cam,
        "p_left_cam": p_left_cam,
        "p_right_cam": p_right_cam,
        "p_left_base": p_left_base,
        "p_right_base": p_right_base,
        "p_left_base_raw": p_left_base_raw,
        "p_right_base_raw": p_right_base_raw,
    }

def save_dual_grasp_debug_image(color_bgr, contour, p_center_cam, p_left_cam,
                                p_right_cam, intrinsics,
                                p_left_hover_cam=None, p_right_hover_cam=None):
    # Clean debug image: only the points you care about. Filled dots = grasp
    # points; rings (test run) = EE hover targets above them.
    debug = color_bgr.copy()

    center_px = camera_point_to_pixel(p_center_cam, intrinsics)
    left_px = camera_point_to_pixel(p_left_cam, intrinsics)
    right_px = camera_point_to_pixel(p_right_cam, intrinsics)

    if center_px is not None:
        cv2.circle(debug, center_px, 10, (0, 0, 255), -1)

    if left_px is not None:
        cv2.circle(debug, left_px, 10, (255, 0, 0), -1)

    if right_px is not None:
        cv2.circle(debug, right_px, 10, (0, 255, 0), -1)

    def ring(p_cam, color_bgr_):
        px = camera_point_to_pixel(p_cam, intrinsics)
        if px is None:
            return
        u, v = px
        cv2.circle(debug, (u, v), 16, (0, 0, 0), 5)
        cv2.circle(debug, (u, v), 16, color_bgr_, 2)
        for du, dv in ((1, 0), (0, 1)):
            cv2.line(debug, (u - du * 16, v - dv * 16),
                     (u + du * 16, v + dv * 16), (0, 0, 0), 3)
            cv2.line(debug, (u - du * 16, v - dv * 16),
                     (u + du * 16, v + dv * 16), color_bgr_, 1)

    if p_left_hover_cam is not None:
        ring(p_left_hover_cam, (255, 0, 0))
    if p_right_hover_cam is not None:
        ring(p_right_hover_cam, (0, 255, 0))

    out_path = DEBUG_DIR / "dual_grasp_debug.png"
    cv2.imwrite(str(out_path), debug)

    print("[INFO] Saved clean debug image:", out_path)
    print("[INFO] Color meaning: red=center, blue=left grasp, green=right grasp")
    if p_left_hover_cam is not None or p_right_hover_cam is not None:
        print("[INFO]                rings = EE test-run hover targets "
              f"({TEST_RUN_HOVER_M*100:.0f} cm above the grasp points)")


# ============================================================
# Robot / gripper
# ============================================================

# One publisher per (robot, topic), reused. The original created a NEW publisher on
# every call and published immediately -- see the wait below for why that loses
# messages -- and never destroyed them.
_GRIPPER_PUBS: dict = {}


def publish_gripper(robot, side, value, wait_for_sub_s: float = 3.0):
    """Command one gripper. 1.0 = open, 0.0 = closed.

    WHY THE WAIT: a brand-new ROS 2 publisher is not connected to anything yet.
    Discovery takes ~100-500 ms, and a message published before the controller has
    matched is DROPPED with no error. The original code published 6 times over
    0.6 s straight after create_publisher, which is a race it can lose -- and when
    it loses, the gripper simply does not move and nothing says why.

    So: reuse the publisher, and wait until something is actually subscribed. If
    nobody ever subscribes, say so loudly and name the likely cause, because a
    silent no-op here looks exactly like a mechanical gripper fault.
    """
    topic = f"/{side}/gripper/gripper_position_controller/commands"

    key = (id(robot), topic)
    if key not in _GRIPPER_PUBS:
        _GRIPPER_PUBS[key] = robot.node.create_publisher(Float64MultiArray, topic, 10)
    pub = _GRIPPER_PUBS[key]

    t0 = time.time()
    while pub.get_subscription_count() == 0 and (time.time() - t0) < wait_for_sub_s:
        time.sleep(0.05)

    n_sub = pub.get_subscription_count()
    if n_sub == 0:
        print(f"[GRIPPER-WARN] nothing is subscribed to {topic} after "
              f"{wait_for_sub_s:.1f}s, so this command will be LOST and the {side} "
              f"gripper will NOT move.")
        print(f"[GRIPPER-WARN] most likely the {side} gripper_position_controller is "
              f"not active. Check with:")
        print(f"[GRIPPER-WARN]   ros2 control list_controllers -c /{side}/controller_manager")
        print(f"[GRIPPER-WARN]   ros2 topic info {topic} --verbose")
    else:
        print(f"[INFO] {side} gripper: {n_sub} subscriber(s), "
              f"waited {1000*(time.time()-t0):.0f} ms, commanding {value:.1f}")

    msg = Float64MultiArray()
    msg.data = [float(value)]

    for _ in range(6):
        pub.publish(msg)
        time.sleep(0.1)


def gripper_open(robot, side):
    print(f"[INFO] Opening {side} gripper...")
    publish_gripper(robot, side, 1.0)
    time.sleep(1.0)


def gripper_close(robot, side):
    print(f"[INFO] Closing {side} gripper...")
    publish_gripper(robot, side, 0.0)
    time.sleep(1.5)


def set_grasp_orientation(pose, side):
    """
    Fixed downward-ish orientation.
    Left and right are opposite, so yaw signs are opposite.
    Tune initial_rotation if gripper opening direction is wrong.
    """
    original_rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])

    if side == "left":
        initial_rotation = 0.0
    elif side == "right":
        initial_rotation = 0.0
    else:
        raise ValueError(side)

    turn_rotation = np.array([
        [np.cos(initial_rotation), -np.sin(initial_rotation), 0.0],
        [np.sin(initial_rotation),  np.cos(initial_rotation), 0.0],
        [0.0, 0.0, 1.0],
    ])

    pose.orientation = Rotation.from_matrix(turn_rotation @ original_rotation)
    return pose


def move_slow(robot, side, position, speed, name):
    position = np.array(position, dtype=float)
    check_workspace(position, side)

    pose = robot.end_effector_pose.copy()
    pose = set_grasp_orientation(pose, side)
    pose.position = position

    print(f"[INFO] Moving {side} to {name}: {position}, speed={speed} m/s")
    robot.move_to(pose=pose, speed=speed)
    print(f"[INFO] {side} reached {name}")
    time.sleep(0.5)


def settle_and_correct(robot, side, target, speed, name):
    # Cancel the impedance controller's constant settle offset by commanding
    # the mirrored point: commanded -= (measured - target). The commanded
    # point accumulates across passes -- re-deriving it from `target` each
    # pass would throw away the previous correction and oscillate.
    # Assumes the arm has already settled at `target` for TEST_RUN_SETTLE_S.
    target = np.asarray(target, dtype=float)
    commanded = target.copy()
    lo, hi = ((LEFT_WORKSPACE_MIN, LEFT_WORKSPACE_MAX) if side == "left"
              else (RIGHT_WORKSPACE_MIN, RIGHT_WORKSPACE_MAX))
    for i in range(TEST_RUN_CORRECT_ITERS):
        error = robot.end_effector_pose.position - target
        err_norm = np.linalg.norm(error)
        if err_norm < TEST_RUN_CORRECT_DONE_M:
            print(f"[CORRECT] {side}: error {1000.0 * err_norm:.2f} mm "
                  "already below threshold, done.")
            return
        if err_norm > TEST_RUN_CORRECT_MAX_M:
            print(f"[CORRECT] {side}: error {1000.0 * err_norm:.1f} mm exceeds "
                  f"{1000.0 * TEST_RUN_CORRECT_MAX_M:.0f} mm limit; NOT "
                  "correcting. Check for a collision or a wrong target.")
            return
        print(f"[CORRECT] {side}: pass {i + 1}, error {1000.0 * err_norm:.2f} mm")
        # Clip so the safety check in move_slow cannot abort the run when the
        # corrected point lands just outside the workspace box.
        commanded = np.clip(commanded - error, lo, hi)
        move_slow(robot, side, commanded, speed, f"{name}_corr{i + 1}")
        time.sleep(TEST_RUN_SETTLE_S)


def init_robot(side):
    print(f"[INFO] Connecting to {side} robot...")
    robot = Robot(namespace=f"/{side}", name=f"dual_green_pick_{side}_client")
    robot.wait_until_ready(timeout=15.0)

    print(f"[INFO] Switching {side} to cartesian_impedance_controller...")
    robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
    robot.cartesian_controller_parameters_client.load_param_config(
        file_path=CONTROL_CONFIG
    )

    return robot


def run_parallel(fn1, fn2):
    t1 = threading.Thread(target=fn1)
    t2 = threading.Thread(target=fn2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


# ============================================================
# Main
# ============================================================

def main(dry_run=False, from_capture=None, test_run=False):
    print("[INFO] Starting dual-arm green object pick (ZED).")
    print("[INFO] Mode: physical-highest top center -> left/right grasp points -> z-down offset.")
    if dry_run:
        print("[INFO] DRY RUN: no robot is contacted and nothing moves.")
    elif test_run:
        print(f"[INFO] TEST RUN: arms stop {TEST_RUN_HOVER_M*100:.0f} cm above the "
              "grasp points; grippers never close, nothing is lifted.")

    T_camera_to_left_base = load_camera_to_base_transform(LEFT_TRANSFORM_PATH, "left")
    T_camera_to_right_base = load_camera_to_base_transform(RIGHT_TRANSFORM_PATH, "right")

    # Two ways in. --from-capture uses a SAM mask made in the sam2 env, because
    # SAM needs torch and the robot needs rclpy, and no single pixi env has both.
    sam_mask = None
    zed = runtime = None
    if from_capture is not None:
        color_bgr, depth, sam_mask, intrinsics = load_capture(from_capture)
    else:
        zed, runtime, intrinsics = init_zed()

    recorder = None
    left_robot = right_robot = None
    try:
        if from_capture is None:
            color_bgr, depth = get_rgbd(zed, runtime)


        result = detect_green_physical_highest_center_and_grasps(
            color_bgr=color_bgr,
            depth=depth,
            intrinsics=intrinsics,
            T_camera_to_left_base=T_camera_to_left_base,
            T_camera_to_right_base=T_camera_to_right_base,
            external_mask=sam_mask,
        )

        p_center_cam = result["p_center_cam"]
        p_left_cam = result["p_left_cam"]
        p_right_cam = result["p_right_cam"]
        p_left_base = result["p_left_base"]
        p_right_base = result["p_right_base"]
        contour = result["contour"]

        print("\n[INFO] Camera-frame points:")
        print("  p_center_cam:", p_center_cam[:3])
        print("  p_left_cam  :", p_left_cam[:3])
        print("  p_right_cam :", p_right_cam[:3])

        print("\n[INFO] Base-frame grasp points:")
        print("  p_left_base :", p_left_base)
        print("  p_right_base:", p_right_base)

        check_workspace(p_left_base, "left")
        check_workspace(p_right_base, "right")

        left_pre = p_left_base.copy()
        left_pre[2] += PRE_GRASP_Z_OFFSET

        right_pre = p_right_base.copy()
        right_pre[2] += PRE_GRASP_Z_OFFSET

        # p_left_base / p_right_base already include the per-arm 3D grasp offsets.
        left_grasp = p_left_base.copy()
        left_grasp[2] += GRASP_Z_OFFSET

        right_grasp = p_right_base.copy()
        right_grasp[2] += GRASP_Z_OFFSET

        left_lift = p_left_base.copy()
        left_lift[2] += LIFT_Z_OFFSET

        right_lift = p_right_base.copy()
        right_lift[2] += LIFT_Z_OFFSET

        # --test-run: hover above the ACTUAL grasp z (left_grasp includes
        # GRASP_Z_OFFSET), so the hover is exactly TEST_RUN_HOVER_M above where
        # the gripper would have closed.
        left_hover = left_grasp.copy()
        left_hover[2] += TEST_RUN_HOVER_M
        right_hover = right_grasp.copy()
        right_hover[2] += TEST_RUN_HOVER_M

        print("\n[INFO] Planned waypoints:")
        print("  left_pre    :", left_pre)
        print("  left_grasp  :", left_grasp)
        print("  left_lift   :", left_lift)
        print("  right_pre   :", right_pre)
        print("  right_grasp :", right_grasp)
        print("  right_lift  :", right_lift)
        if test_run:
            print("  left_hover  :", left_hover, " (test-run stop)")
            print("  right_hover :", right_hover, " (test-run stop)")

        # The hover targets are defined in BASE frame; map them back through
        # the inverse extrinsics so the image/video overlays can draw where
        # the EEs will actually stop. None outside a test run -> not drawn.
        p_left_hover_cam = p_right_hover_cam = None
        if test_run:
            p_left_hover_cam = (np.linalg.inv(T_camera_to_left_base)
                                @ np.append(left_hover[:3], 1.0))
            p_right_hover_cam = (np.linalg.inv(T_camera_to_right_base)
                                 @ np.append(right_hover[:3], 1.0))

        save_dual_grasp_debug_image(
            color_bgr=color_bgr,
            contour=contour,
            p_center_cam=p_center_cam,
            p_left_cam=p_left_cam,
            p_right_cam=p_right_cam,
            intrinsics=intrinsics,
            p_left_hover_cam=p_left_hover_cam,
            p_right_hover_cam=p_right_hover_cam,
        )

        print("\n[INFO] Debug images:")
        print(" ", DEBUG_DIR / "raw_color.png")
        print(" ", DEBUG_DIR / "green_mask_debug.png")
        print(" ", DEBUG_DIR / "green_physical_highest_detection_debug.png")
        print(" ", DEBUG_DIR / "dual_grasp_debug.png")


        if dry_run:
            print("\n[INFO] DRY RUN complete. Nothing moved.")
            print("[INFO] Compare the grasp points above against the object by eye,")
            print("[INFO] and check the dots in dual_grasp_debug.png sit on it.")
            return

        if CONFIRM_BEFORE_MOVE:
            what = ("hover above the grasp points (TEST RUN)" if test_run
                    else "pick")
            ans = input(f"\nType START to move BOTH robots slowly ({what}): ").strip()
            if ans != "START":
                print("[INFO] User did not type START. Exit without moving.")
                return

        # --from-capture detects from a SAVED frame, so the camera was never opened.
        # Nothing else holds it though, so open it now purely to record the motion.
        # The dots are projected with the capture's intrinsics, which are the same
        # camera at the same resolution, so they stay valid.
        if RECORD_VIDEO and zed is None:
            print("[VIDEO] --from-capture: opening the ZED for RECORDING only "
                  "(detection already used the saved frame).")
            try:
                zed, runtime, _ = init_zed()
            except Exception as exc:
                print(f"[VIDEO-WARN] could not open the ZED to record: {exc}")
                print("[VIDEO-WARN] continuing WITHOUT video. The motion still runs.")

        # A hover test is not a pick attempt; name its video so the pick_runs
        # folder stays honest about which recordings actually grasped.
        video_out = (VIDEO_PATH.with_name(VIDEO_PATH.name.replace("pick_", "test_", 1))
                     if test_run else VIDEO_PATH)

        if RECORD_VIDEO and zed is not None:
            recorder = GraspOverlayVideoRecorder(
                zed=zed,
                runtime=runtime,
                intrinsics=intrinsics,
                p_center_cam=p_center_cam,
                p_left_cam=p_left_cam,
                p_right_cam=p_right_cam,
                out_path=video_out,
                p_left_hover_cam=p_left_hover_cam,
                p_right_hover_cam=p_right_hover_cam,
            )
            recorder.start()
            print(f"[VIDEO] Lead-in recording for {VIDEO_LEAD_IN_SEC:.1f} sec before robot motion...")
            time.sleep(VIDEO_LEAD_IN_SEC)

        left_robot = init_robot("left")
        right_robot = init_robot("right")

        gripper_open(left_robot, "left")
        gripper_open(right_robot, "right")

        run_parallel(
            lambda: move_slow(left_robot, "left", left_pre, SPEED_APPROACH, "pre_grasp"),
            lambda: move_slow(right_robot, "right", right_pre, SPEED_APPROACH, "pre_grasp"),
        )

        if test_run:
            # Descend to the hover points and STOP: no gripper close, no lift.
            run_parallel(
                lambda: move_slow(left_robot, "left", left_hover, SPEED_DESCEND, "test_hover"),
                lambda: move_slow(right_robot, "right", right_hover, SPEED_DESCEND, "test_hover"),
            )

            # Measured-vs-commanded at the hover. move_to() streams interpolated
            # TARGETS and never checks the measured pose, and a cartesian
            # impedance controller settles wherever stiffness balances gravity
            # and friction -- so "reached" above does not mean the EE is
            # physically there. This split decides where the gripper-vs-ring
            # gap in the video comes from: ~0 mm here means the controller is
            # fine and the gap is calibration bias; several mm here is
            # controller settle error (stiffness / settle-and-correct), which
            # recalibrating cannot fix.
            time.sleep(TEST_RUN_SETTLE_S)
            print("\n[TRACK] EE tracking error at the hover points "
                  f"(measured - commanded, base frame, after {TEST_RUN_SETTLE_S:.1f}s settle):")
            for side, robot, hover in (("left", left_robot, left_hover),
                                       ("right", right_robot, right_hover)):
                err_mm = 1000.0 * (robot.end_effector_pose.position - hover)
                print(f"[TRACK]   {side:5s}: [{err_mm[0]:+7.2f}, {err_mm[1]:+7.2f}, "
                      f"{err_mm[2]:+7.2f}] mm   |err| = {np.linalg.norm(err_mm):6.2f} mm")
            print("[TRACK] ~0 mm -> the image gap is calibration bias; "
                  "several mm -> controller settle error, not calibration.")

            # Settle-and-correct: cancel the constant settle offset the
            # numbers above just measured, then report what is left. The
            # residual is what recalibration/stiffness tuning would still
            # have to explain.
            run_parallel(
                lambda: settle_and_correct(left_robot, "left", left_hover,
                                           SPEED_DESCEND, "test_hover"),
                lambda: settle_and_correct(right_robot, "right", right_hover,
                                           SPEED_DESCEND, "test_hover"),
            )
            print(f"\n[TRACK] residual error after settle-and-correct "
                  f"(max {TEST_RUN_CORRECT_ITERS} passes):")
            for side, robot, hover in (("left", left_robot, left_hover),
                                       ("right", right_robot, right_hover)):
                err_mm = 1000.0 * (robot.end_effector_pose.position - hover)
                print(f"[TRACK]   {side:5s}: [{err_mm[0]:+7.2f}, {err_mm[1]:+7.2f}, "
                      f"{err_mm[2]:+7.2f}] mm   |err| = {np.linalg.norm(err_mm):6.2f} mm")
        else:
            run_parallel(
                lambda: move_slow(left_robot, "left", left_grasp, SPEED_DESCEND, "grasp"),
                lambda: move_slow(right_robot, "right", right_grasp, SPEED_DESCEND, "grasp"),
            )

            run_parallel(
                lambda: gripper_close(left_robot, "left"),
                lambda: gripper_close(right_robot, "right"),
            )

            run_parallel(
                lambda: move_slow(left_robot, "left", left_lift, SPEED_LIFT, "lift"),
                lambda: move_slow(right_robot, "right", right_lift, SPEED_LIFT, "lift"),
            )

        if RECORD_VIDEO:
            tail_after = "hover" if test_run else "lift"
            print(f"[VIDEO] Tail recording for {VIDEO_TAIL_SEC:.1f} sec after {tail_after}...")
            time.sleep(VIDEO_TAIL_SEC)

        if test_run:
            print("[INFO] TEST RUN finished: arms are holding "
                  f"{TEST_RUN_HOVER_M*100:.0f} cm above the grasp points. "
                  "Check the hover positions against the object before a real pick.")
        else:
            print("[INFO] Dual-arm pick attempt finished.")
        if recorder is not None and recorder.frames_written > 0:
            print(f"[VIDEO] saved: {video_out}")
        elif RECORD_VIDEO:
            print("[VIDEO] no video was written -- see the [VIDEO...] lines above.")

    finally:
        if recorder is not None:
            try:
                recorder.stop()
            except Exception as e:
                print(f"[VIDEO-WARN] failed to stop recorder cleanly: {e}")

        # A USB ZED allows only ONE process, so always hand it back. The Azure
        # original never closed its camera.
        if zed is not None:
            try:
                zc.close_zed(zed)
            except Exception as e:
                print(f"[WARN] failed to close the ZED cleanly: {e}")

        for robot in (left_robot, right_robot):
            if robot is None:
                continue
            try:
                robot.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-capture", metavar="RUN_DIR", default=None,
                    help="use a saved capture with a SAM mask instead of opening the "
                         "camera and thresholding on green. Make it first with:  "
                         "pixi run -e sam2 python zed_capture/capture_zed_sam_mask.py "
                         "--sam-checkpoint sam2/checkpoints/sam2.1_hiera_large.pt "
                         "--output RUN_DIR   . Needs rgb_full.png, depth_mm_full.npy "
                         "and mask_full.png in RUN_DIR. SAM needs torch (sam2 env) and "
                         "the robot needs rclpy (humble env), so this cannot be one step.")
    ap.add_argument("--dry-run", action="store_true",
                    help="detect and print the grasp targets, then exit. Never contacts "
                         "a robot and never moves. Use this to check the calibration.")
    ap.add_argument("--test-run", action="store_true",
                    help="like a normal run (arms MOVE, video is recorded), but the "
                         f"descend stops {TEST_RUN_HOVER_M*100:.0f} cm above the grasp "
                         "points and the grippers never close. The step between "
                         "--dry-run and a real pick. Ignored if --dry-run is given.")
    ap.add_argument("--calib-seq-name", default=None,
                    help="calibration sequence to use: transforms are read from and "
                         "grasp videos are saved into captured_calibration_data/<seq>. "
                         "Default: zed_calib_003 (the module constants).")
    _a = ap.parse_args()
    if _a.calib_seq_name:
        set_calib_sequence(_a.calib_seq_name)
    main(dry_run=_a.dry_run, from_capture=_a.from_capture, test_run=_a.test_run)