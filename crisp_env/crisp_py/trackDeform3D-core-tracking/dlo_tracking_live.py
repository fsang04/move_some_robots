"""Live DLO tracking driver (REALTIME_SAM2_OVERVIEW.md §2.3).

The offline driver dlo_tracking.py is UNCHANGED; this driver adds the live path.
It builds one WireTracker (no clips), loops over frames from a FrameSource, and
shows the tracked keypoints in an OpenCV window.

Three sources:
    --source replay   recorded chunk with the shipped masks + robot EE poses.
                      Needs only the base trackdeform3d environment. This is the
                      parity path against the offline driver.
    --source kinect   live Azure Kinect (pyk4a) + per-frame masks from a live
                      segmenter. No robot stream is needed.
    --source zed      live ZED 2 (pyzed). Colour and depth are already rectified
                      and mutually registered, so there is no undistortion step;
                      K is the rectified left-camera matrix, at --zed_resolution.

Three live segmenters (--segmenter, live sources only):
    sam2     SAM2 streaming masks (default; torch/transformers, GPU).
    pcdiff   the deformable_seg point-cloud-difference method: an empty-scene
             depth reference is captured at start-up, and each frame's mask is
             what stands in front of it in 3D (realtime/pcdiff_segmenter.py).
             No GPU / torch. The arms become foreground once they move away
             from their reference pose -- keep them outside the --z_range gate
             or rely on the largest-component filter.
    armdiff  pcdiff plus the ARMS SUBTRACTED: per frame the two Franka arms
             are depth-rendered from the live crisp_py joint stream (FK +
             franka_description meshes, realtime/arm_reference.py) and every
             pixel at/behind the rendered arm is removed, so the arms may
             manipulate the object inside the workspace gate. Background
             reference: --bg_mode temporal (the frame --lag frames ago; no
             ritual, finds what moves) or static (empty-scene median).
             Needs the hand-eye calibration (--calib) made under the
             newCameraMatrix=K undistortion convention. No GPU / torch.

Four init modes (--init; they answer "where are the cable ends on frame 0,
and what prompts SAM2"):
    fk       the EE pair straight from the joint stream (armdiff default, and
             armdiff only). The grippers HOLD the cable ends, so FK plus the
             hand-eye calibration already give both ends in camera mm -- no
             candidate mask, no skeleton, no guess from the image. The driver
             waits until the armdiff mask is big enough AND reaches both
             projected grippers (--min_init_mask_px, --max_ee_mask_px), then
             initializes. It also keeps the pair live for every later frame.
    auto     no manual input at all (kinect default). A promptless candidate
             mask (ridge filter in the workspace depth gate, or background
             subtraction with --bootstrap bgsub) prompts SAM2; the endpoints
             of the accepted SAM2 skeleton, back-projected through the depth,
             become the EE pair (realtime/bootstrap.py). Frames are retried
             until one passes the acceptance test -- lay the cable open and
             uncrossed at session start. With pcdiff/armdiff the ridge
             candidates are switched OFF (bootstrap trusted_only): a ridge
             component would otherwise replace the segmenter's own mask
             whenever that mask fails the acceptance test.
    click    you click the two cable ends (+ optional extra points) on the
             first frame; the clicks prompt SAM2 and give the EE pair.
    replay   the recorded robot EE poses and shipped masks (replay default;
             the parity path).

After frame 0 no EE input is needed: the single-DLO tracking loop takes its
leaf anchors from the detected skeleton tips (wire_tracker.py:1032-1044). In
auto mode the driver also refreshes the stored EE pair from the current mask
whenever tracking skips, so a warm restart re-initializes with CURRENT ends;
in fk mode it refreshes the pair from the joint stream on EVERY frame, which
costs one FK chain per arm.

Usage:
    python dlo_tracking_live.py --source replay --chunk 1
    python dlo_tracking_live.py --source replay --chunk 1 --init auto  # test auto w/o robot data
    python dlo_tracking_live.py --source replay --chunk 1 --init auto --sam2_on_replay
                                # full kinect rehearsal on recorded data (SAM2, no shipped masks)
    python dlo_tracking_live.py --source kinect                        # fully automatic start
    python dlo_tracking_live.py --source kinect --init click
    python dlo_tracking_live.py --source kinect --segmenter pcdiff  # no SAM2/GPU
    python dlo_tracking_live.py --source kinect --segmenter armdiff --calib <rig>.npz
                                # arms in the workspace, subtracted via live joints
    python dlo_tracking_live.py --source kinect --segmenter armdiff --joints fixed
                                # rendering rehearsal without the robots
    python dlo_tracking_live.py --source zed --segmenter armdiff --calib <rig>.npz
                                # the ZED rig (realtime/README_ZED.md); --init fk
    python dlo_tracking_live.py --source zed --segmenter armdiff --calib <rig>.npz \
                                --bg_mode static
                                # + finds a cable that does NOT move (see --bg_mode)
    python dlo_tracking_live.py --source zed --segmenter armdiff --calib <rig>.npz \
                                --debug_every 10
                                # + a 2x2 panel every 10 frames: colour, depth,
                                #   rendered arm depth, mask + arm outline + EE
                                #   (add --debug_dir <folder> to write PNGs)
    python dlo_tracking_live.py --source kinect --record output/dlo_live/session_1

--record writes the session in the exact chunk format (rgbd.npz, masks/masks.npz,
left/right_arm_poses.npz, calibration/transform_ee_cam_world.npz), so the
unchanged offline dlo_tracking.py can evaluate a live session afterwards.
Keys in the display window: q or ESC = stop.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from tracker.wire_tracker import WireTracker
from realtime.frame_source import KinectSource, ReplaySource, ZedSource
from realtime.bootstrap import (auto_init, path_end_to_3d, pixel_to_3d_mm,
                                skeleton_path, BackgroundSubtractor)


# ============================================================================
# helpers
# ============================================================================

def build_tracker(K, n_keypoints, ee_poses_3d, max_depth=2000.0):
    """Same parameters as the offline driver (dlo_tracking.py:49-79)."""
    intrinsics = np.array([
        [K[0, 0], 0, K[0, 2]],
        [0, K[1, 1], K[1, 2]],
        [0, 0, 1]
    ])
    tracker = WireTracker(
        intrinsics=intrinsics,
        n_keypoints=n_keypoints,
        target_branch_nodes=0,
        target_leaf_nodes=2,
        max_depth=max_depth,
        top_k_components=1,
        n_outer_iterations=20,
        n_edge_iterations=15,
        edge_weight=0.5,
        edge_tolerance=0.02,
        repulsion_iterations=200,
        repulsion_lr=10.0,
        repulsion_k_neighbors=3,
        enable_node_matching=True,
        enable_geometry_constraint=True,
        enable_ee_injection=True,
        ee_poses_3d=ee_poses_3d,
    )
    return tracker, intrinsics


def project_mm(pts_3d, K):
    """(N,3) camera-frame mm -> (N,2) pixel (x, y)."""
    pts_3d = np.atleast_2d(pts_3d)
    z = np.maximum(pts_3d[:, 2], 1e-6)
    x = pts_3d[:, 0] * K[0, 0] / z + K[0, 2]
    y = pts_3d[:, 1] * K[1, 1] / z + K[1, 2]
    return np.stack([x, y], axis=1)


def pixel_to_3d_mm(depth_u16, xy, K, win=7):
    """Back-project one clicked pixel to camera-frame mm.

    Uses the median of the valid depths in a win x win window, because the
    exact clicked pixel can have no depth. Returns None when the whole window
    has no depth.
    """
    H, W = depth_u16.shape
    x, y = int(round(xy[0])), int(round(xy[1]))
    r = win // 2
    patch = depth_u16[max(0, y - r):min(H, y + r + 1),
                      max(0, x - r):min(W, x + r + 1)].astype(np.float64)
    valid = patch[patch > 0]
    if len(valid) == 0:
        return None
    z = float(np.median(valid))
    return np.array([(x - K[0, 2]) * z / K[0, 0],
                     (y - K[1, 2]) * z / K[1, 1],
                     z])


def click_points(window, bgr, n_min=2):
    """Collect clicked (x, y) points on the first frame.

    Click the TWO CABLE ENDS first, then optional extra points on the cable.
    ENTER/SPACE = confirm, z = undo last click, ESC = abort (returns None).
    """
    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        vis = bgr.copy()
        cv2.putText(vis, f'click the 2 cable ends (+ extra points), '
                         f'{len(pts)} clicked  [ENTER=ok  z=undo  ESC=abort]',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for i, (x, y) in enumerate(pts):
            color = (255, 0, 255) if i < 2 else (0, 255, 0)
            cv2.circle(vis, (int(x), int(y)), 6, color, 2)
        cv2.imshow(window, vis)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32) and len(pts) >= n_min:   # ENTER / SPACE
            return np.array(pts)
        if key == ord('z') and pts:
            pts.pop()
        if key == 27:                               # ESC
            return None


def draw_overlay(bgr, result, intrinsics, ee_px, fps_now):
    vis = bgr.copy()
    if result.get('success'):
        kp = result['keypoints']
        px = project_mm(kp, intrinsics)
        for i, j in (result.get('edges') or []):
            cv2.line(vis, (int(px[i, 0]), int(px[i, 1])),
                     (int(px[j, 0]), int(px[j, 1])), (0, 255, 0), 2)
        for x, y in px:
            cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)
        label = f"{result.get('mode', 'track')}  {fps_now:.1f} fps"
    else:
        label = f"SKIP ({result.get('reason', '?')})  {fps_now:.1f} fps"
    if ee_px is not None:
        for x, y in ee_px:
            cv2.circle(vis, (int(x), int(y)), 8, (255, 0, 255), 2)
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2)
    return vis


def colorize_depth(depth_mm, z_range):
    """(H,W) depth in mm -> BGR turbo image, black where there is no depth.

    Only the workspace band is stretched over the colour ramp: outside
    z_range everything saturates, so the cable stands out against the table
    instead of against the whole 0-10 m sensor range.

    Call it on an ALREADY DOWNSCALED depth image -- the per-pixel float work
    dominates, and shrinking first is what keeps the debug panel at a few ms
    (a full-res colormap of a 1280x720 frame costs ~7 ms on its own).
    """
    lo, hi = float(z_range[0]), float(z_range[1])
    d = depth_mm.astype(np.float32, copy=False)
    norm = np.clip((d - lo) * (1.0 / max(hi - lo, 1e-6)), 0.0, 1.0)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.bitwise_and(vis, vis, mask=(d > 0).astype(np.uint8))


def debug_panel(bgr, depth, mask, arm_depth, ee_px, z_range, scale=0.5,
                label=''):
    """2x2 verification panel for the armdiff pipeline.

      colour             | depth (turbo, workspace band)
      rendered arm depth | mask (green) + arm silhouette (red) + EE (magenta)

    The bottom row is the whole armdiff question in one picture: whether the
    rendered arm lands ON the real arm (bottom-left against the arm in the
    depth image above it), and whether what survives the subtraction is the
    cable and only the cable (bottom-right).

    Everything is downscaled by `scale` BEFORE any per-pixel work.
    """
    def shrink(img, interp=cv2.INTER_NEAREST):
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)

    def label_tile(t, text):
        cv2.putText(t, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        return t

    color_s = shrink(bgr, cv2.INTER_AREA)
    color_t = label_tile(color_s.copy(), f'color {label}')
    depth_t = label_tile(colorize_depth(shrink(depth), z_range), 'depth')

    if arm_depth is not None:
        arm_s = shrink(arm_depth)
        arm_valid = (arm_s > 0).astype(np.uint8)
        arm_t = label_tile(colorize_depth(arm_s, z_range),
                           f'rendered arm {int(arm_valid.sum() / scale ** 2)} px')
    else:
        arm_valid = None
        arm_t = label_tile(np.zeros_like(color_s), 'rendered arm (none)')

    m = shrink((mask > 0).astype(np.uint8))
    green = np.zeros_like(color_s)
    green[:, :, 1] = 255
    over = color_s.copy()
    cv2.copyTo(cv2.addWeighted(color_s, 0.35, green, 0.65, 0.0), m, over)
    if arm_valid is not None:
        cnts, _ = cv2.findContours(arm_valid, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cnts, -1, (0, 0, 255), 1)
    if ee_px is not None:
        for x, y in np.atleast_2d(np.asarray(ee_px, dtype=float) * scale):
            cv2.circle(over, (int(x), int(y)), 6, (255, 0, 255), 2)
    mask_t = label_tile(over, f'mask {int(m.sum() / scale ** 2)} px')

    return np.vstack([np.hstack([color_t, depth_t]),
                      np.hstack([arm_t, mask_t])])


class ChunkRecorder:
    """Records a live session in the offline chunk format, so the unchanged
    dlo_tracking.py can run on it afterwards.

    A Kinect session has no robot: the EE pair is the (static) clicked ends.
    The poses are written in the CAMERA frame in meters with identity rotation,
    and the calibration file holds identity base2cam transforms -- then
    get_ee_positions_cam reproduces the camera-frame mm positions exactly.
    """

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.color, self.depth, self.masks, self.left, self.right = [], [], [], [], []

    def add(self, frame, mask, ee_pair_mm):
        self.color.append(frame.color)
        self.depth.append(frame.depth)
        self.masks.append(mask.astype(np.uint8))
        for lst, pos_mm in ((self.left, ee_pair_mm[0]), (self.right, ee_pair_mm[1])):
            lst.append(np.array([pos_mm[0] / 1000.0, pos_mm[1] / 1000.0,
                                 pos_mm[2] / 1000.0, 1.0, 0.0, 0.0, 0.0]))

    def save(self, K, depth_offset_px=0.0, disparity_scale=1.0,
             depth_source='zed'):
        if not self.color:
            return
        (self.out_dir / 'masks').mkdir(parents=True, exist_ok=True)
        (self.out_dir / 'calibration').mkdir(parents=True, exist_ok=True)
        # disparity_offset_px / disparity_scale are provenance, not data: they
        # record the (a, d) correction ALREADY APPLIED to this depth, so a
        # solver re-run on this session does not apply it a second time (the
        # guards zed_depth_config.dataset_applied_offset_px / _scale read
        # both). depth_source records which matcher made the depth (zed SDK /
        # ffs = Fast-FoundationStereo / kinect).
        np.savez(self.out_dir / 'rgbd.npz',
                 color=np.stack(self.color), depth=np.stack(self.depth),
                 disparity_offset_px=np.float64(depth_offset_px),
                 disparity_scale=np.float64(disparity_scale),
                 depth_source=np.str_(depth_source))
        np.savez_compressed(self.out_dir / 'masks' / 'masks.npz',
                            masks=np.stack(self.masks))
        np.savez(self.out_dir / 'left_arm_poses.npz', *self.left)
        np.savez(self.out_dir / 'right_arm_poses.npz', *self.right)
        np.savez(self.out_dir / 'calibration' / 'transform_ee_cam_world.npz',
                 T_left_base2cam=np.eye(4), T_right_base2cam=np.eye(4),
                 K=np.asarray(K, dtype=np.float32))
        print(f"  Recorded {len(self.color)} frames in chunk format -> {self.out_dir}")
        print(f"  (point dlo_tracking.py's data_base at the parent folder to evaluate it)")


# ============================================================================
# main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', choices=['replay', 'kinect', 'zed'], default='replay')
    # replay options
    parser.add_argument('--chunk', type=int, default=1, help='replay: chunk index')
    parser.add_argument('--data_base', default='input_data/dlo',
                        help='replay: folder with chunk_*/ and calibration/')
    parser.add_argument('--fps', type=float, default=None,
                        help='replay: pace playback to this rate (default: as fast as possible)')
    parser.add_argument('--sam2_on_replay', action='store_true',
                        help='replay: IGNORE the shipped masks and run the full kinect '
                             'pipeline (bootstrap candidates -> SAM2 prompting -> '
                             'streaming SAM2 masks) on the recorded frames. The '
                             'pre-camera rehearsal of the live path; needs --init auto '
                             'or click, torch/transformers, and a GPU.')
    parser.add_argument('--max_frames', type=int, default=None)
    # kinect options
    parser.add_argument('--no_undistort', action='store_true',
                        help='kinect: skip undistortion (newCameraMatrix=K convention). '
                             'Has no zed equivalent -- the SDK delivers rectified frames.')
    # zed options
    parser.add_argument('--zed_resolution', default='HD720',
                        help='zed: SDK resolution (HD720 / HD1080 / HD2K). K is '
                             'per-resolution, so the --calib K must match')
    parser.add_argument('--zed_fps', type=int, default=30, help='zed: camera fps')
    parser.add_argument('--zed_depth_mode', default='NEURAL_PLUS',
                        help='zed: NEURAL / NEURAL_PLUS / ULTRA. Default matches '
                             'the calibration capture (zed_calib_rgbd.py): the '
                             'disparity correction was fit under NEURAL_PLUS. '
                             'NEURAL is faster but off-model.')
    parser.add_argument('--zed_exposure', type=int, default=10,
                        help='zed: manual exposure 0-100, matching the '
                             'calibration capture (auto-exposure saturates the '
                             'white arms on the black backdrop). -1 = auto')
    parser.add_argument('--zed_stabilization', type=int, default=10,
                        help='zed: temporal depth stabilization 0-100, matching '
                             'the calibration capture. Damps frame-to-frame '
                             'depth flicker (helps --bg_mode temporal)')
    parser.add_argument('--zed_confidence', type=int, default=47,
                        help='zed: depth confidence threshold, matching the '
                             'calibration capture -- uncertain pixels become '
                             'holes instead of noise. -1 = SDK default')
    parser.add_argument('--zed_depth_offset_px', type=float, default=None,
                        help='zed: disparity-offset correction, in px AT THE FRAME '
                             'WIDTH IN USE. Default: the configured offset, '
                             'rescaled from zed_depth_correction.json. Pass 0 to '
                             'disable -- but the hand-eye extrinsics were solved '
                             'from CORRECTED depth, so armdiff will then eat the '
                             'object. Set it to 0 only after a ZED recalibration')
    parser.add_argument('--depth_source', choices=['zed', 'ffs'], default='zed',
                        help='zed: which stereo matcher produces the depth. '
                             'zed = the SDK matcher (--zed_depth_mode). '
                             'ffs = Fast-FoundationStereo TensorRT '
                             '(realtime/ffs_trt.py): the SDK matcher is OFF, '
                             'the rectified pair goes through --ffs_engine, and '
                             'the SAME a/d correction is applied -- the (a,d) '
                             'fault lives in the rectified images, not in the '
                             'matcher. --zed_depth_mode/--zed_confidence/'
                             '--zed_stabilization are ignored in ffs mode')
    parser.add_argument('--ffs_engine',
                        default=str(Path.home() / 'move_some_robots' /
                                    'ffs_engines' /
                                    'ffs_23-36-37_it8_736x1280.engine'),
                        help='--depth_source ffs: the TensorRT engine, built by '
                             'ffs_engines/export_and_build.sh. An engine is '
                             'tied to ONE GPU model and ONE TensorRT version; '
                             'on a mismatch, rebuild it')
    parser.add_argument('--segmenter', choices=['sam2', 'pcdiff', 'armdiff'], default='sam2',
                        help='kinect: per-frame mask source. sam2 = SAM2 streaming '
                             '(GPU); pcdiff = 3D point-cloud difference against an '
                             'empty-scene reference (deformable_seg port, no GPU); '
                             'armdiff = pcdiff plus the arms RENDERED from the live '
                             'joint stream and subtracted, so the arms may work '
                             'inside the workspace gate (no GPU, needs crisp_py)')
    parser.add_argument('--sam2_model', default='facebook/sam2.1-hiera-tiny')
    parser.add_argument('--close_ksize', type=int, default=5,
                        help='kinect: morphological close on the live mask')
    parser.add_argument('--pcdiff_threshold', type=float, default=30.0,
                        help='pcdiff/armdiff: 3D distance to the reference in mm '
                             'above which a pixel is foreground')
    # armdiff options (rendered-arm subtraction; realtime/armdiff_segmenter.py)
    parser.add_argument('--calib', default=None,
                        help='armdiff: transform_ee_cam_world.npz with '
                             'T_left/right_base2cam for THIS rig '
                             '(default: <data_base>/calibration/...)')
    parser.add_argument('--joints', choices=['crisp', 'fixed'], default='crisp',
                        help='armdiff: joint stream. crisp = live crisp_py '
                             'Robot subscribers; fixed = constant ready pose '
                             '(rendering rehearsal without robots)')
    parser.add_argument('--left_ns', default='left',
                        help='armdiff: ROS namespace of the left arm')
    parser.add_argument('--right_ns', default='right',
                        help='armdiff: ROS namespace of the right arm')
    parser.add_argument('--bg_mode', choices=['temporal', 'static'], default='temporal',
                        help='armdiff: background reference. temporal = the frame '
                             '--lag frames ago (no ritual, finds what moves); '
                             'static = empty-scene median like pcdiff (finds the '
                             'object even at rest; needs the capture ritual)')
    parser.add_argument('--lag', type=int, default=5,
                        help='armdiff temporal: how many frames back the '
                             'reference capture lies')
    parser.add_argument('--arm_tol', type=float, default=40.0,
                        help='armdiff: mm from the rendered arm surface still '
                             'counted as arm (covers calibration + joint lag)')
    parser.add_argument('--arm_dilate', type=int, default=9,
                        help='armdiff: px to grow the rendered arm silhouette')
    parser.add_argument('--arm_points', type=int, default=20000,
                        help='armdiff: surface samples per arm for the renderer')
    parser.add_argument('--arm_model', default='fr3',
                        help='armdiff: franka_description arm variant '
                             '(fr3 / fer / fp3)')
    parser.add_argument('--grasp_offset', type=float, default=None,
                        help='armdiff fk init: METERS along the hand +z axis, '
                             'from the hand frame to the point the gripper '
                             'holds (default: arm_reference.GRASP_Z = 0.1034, '
                             'the franka hand TCP). Measure it for a different '
                             'tool or a different grasp')
    # init options
    parser.add_argument('--init', choices=['auto', 'click', 'replay', 'fk'],
                        default=None,
                        help='first-frame source of the EE pair + SAM2 prompt '
                             '(default: replay->replay, armdiff->fk, '
                             'other live->auto)')
    parser.add_argument('--min_init_mask_px', type=int, default=500,
                        help='fk init: mask pixels the segmenter must deliver '
                             'before the tracker initializes')
    parser.add_argument('--max_ee_mask_px', type=float, default=60.0,
                        help='fk init: how far the nearest mask pixel may lie '
                             'from a projected gripper. Above this the mask is '
                             'not the cable in the grippers, and the driver '
                             'waits. Raise it with --arm_dilate')
    parser.add_argument('--init_timeout', type=float, default=60.0,
                        help='fk init: seconds to wait for a mask that reaches '
                             'both grippers (one attempt costs ~10 ms, so this '
                             'is a time budget, not --max_init_attempts)')
    parser.add_argument('--bootstrap', choices=['ridge', 'bgsub'], default='ridge',
                        help='auto init: promptless candidate-mask source')
    parser.add_argument('--bg_frames', type=int, default=30,
                        help='auto init, bgsub: empty-scene depth frames to record')
    parser.add_argument('--z_range', type=float, nargs=2, default=(500.0, 2000.0),
                        metavar=('ZMIN', 'ZMAX'),
                        help='auto init: workspace depth gate in mm')
    parser.add_argument('--max_init_attempts', type=int, default=150,
                        help='auto init: frames to try before giving up')
    # common
    parser.add_argument('--n_keypoints', type=int, default=15)
    parser.add_argument('--max_depth', type=float, default=2000.0)
    parser.add_argument('--no_display', action='store_true')
    parser.add_argument('--debug_every', type=int, default=0,
                        help='every N frames, show a 2x2 panel (colour, depth, '
                             'rendered arm depth, mask+arm outline+EE) in a '
                             'second window -- the armdiff verification view. '
                             '0 = off')
    parser.add_argument('--debug_scale', type=float, default=0.5,
                        help='--debug_every: per-tile scale of the panel')
    parser.add_argument('--debug_dir', default=None,
                        help='--debug_every: also write each panel as a PNG here, '
                             'into a <YYYYMMDD_HHMMSS>/ subfolder per run -- the '
                             'same stamp as output/dlo_live/ '
                             '(works with --no_display, e.g. over ssh). '
                             'fk init: additionally writes init_<t>s.jpg every '
                             '2 s while waiting, and init_accepted.jpg for the '
                             'frame the init took -- no --debug_every needed')
    parser.add_argument('--record', default=None,
                        help='record the session in chunk format to this folder')
    parser.add_argument('--output', default=None,
                        help='folder for 3d_keypoints.npz (default: output/dlo_live/<time>)')
    args = parser.parse_args()

    init_mode = args.init or ('replay' if args.source == 'replay' else
                              'fk' if args.segmenter == 'armdiff' else 'auto')
    if init_mode == 'replay' and args.source != 'replay':
        parser.error('--init replay needs --source replay (recorded robot poses)')
    if init_mode == 'click' and args.no_display:
        parser.error('--init click needs the display for the first-frame clicks')
    if init_mode == 'fk' and args.segmenter != 'armdiff':
        parser.error('--init fk needs --segmenter armdiff: the EE pair comes '
                     'from the rendered-arm joint stream, which only armdiff has')
    if args.segmenter in ('pcdiff', 'armdiff') and args.source not in ('kinect', 'zed'):
        parser.error(f'--segmenter {args.segmenter} needs a live source '
                     f'(--source kinect or zed; replay ships its own masks)')
    if args.no_undistort and args.source == 'zed':
        parser.error('--no_undistort has no zed equivalent: the SDK computes depth '
                     'on the RECTIFIED left image, so unrectified colour has no '
                     'matching depth')
    if args.depth_source == 'ffs' and args.source != 'zed':
        parser.error('--depth_source ffs needs --source zed: it consumes the '
                     'rectified ZED stereo pair')
    if args.sam2_on_replay and args.source != 'replay':
        parser.error('--sam2_on_replay only makes sense with --source replay')
    if args.sam2_on_replay and init_mode == 'replay':
        parser.error('--sam2_on_replay needs --init auto or --init click: SAM2 must be '
                     'prompted on the first frame, and the replay init never prompts it')

    script_dir = Path(__file__).resolve().parent
    session_stamp = time.strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.output) if args.output else \
        script_dir / 'output' / 'dlo_live' / session_stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    window = 'dlo_tracking_live'
    debug_window = 'dlo_debug (color | depth / arm | mask)'
    # one timestamped subfolder per run, same stamp as out_dir, so panels from
    # different sessions never mix and correlate 1:1 with their keypoints
    debug_dir = Path(args.debug_dir) / session_stamp if args.debug_dir else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        # every parameter of the run, so a panel folder is self-describing:
        # the flags in question (exposure, stabilization, confidence, depth
        # mode, bg_mode, ...) all live in args, so dump args wholesale
        with open(debug_dir / 'params.txt', 'w') as f:
            f.write(f"session {session_stamp}\n")
            f.write(f"command: {' '.join(sys.argv)}\n\n")
            for k, v in sorted(vars(args).items()):
                f.write(f"{k} = {v}\n")
        print(f"  debug panels -> {debug_dir} (params.txt written)")

    # ---------------- source ----------------
    if args.source == 'replay':
        data_base = script_dir / args.data_base
        source = ReplaySource(data_base / f'chunk_{args.chunk}',
                              data_base / 'calibration',
                              fps=args.fps, max_frames=args.max_frames)
    elif args.source == 'zed':
        source = ZedSource(resolution=args.zed_resolution, fps=args.zed_fps,
                           depth_mode=args.zed_depth_mode,
                           depth_offset_px=args.zed_depth_offset_px,
                           exposure=(None if args.zed_exposure < 0
                                     else args.zed_exposure),
                           depth_stabilization=args.zed_stabilization,
                           confidence=(None if args.zed_confidence < 0
                                       else args.zed_confidence),
                           depth_source=args.depth_source,
                           ffs_engine=args.ffs_engine)
    else:
        source = KinectSource(undistort=not args.no_undistort)

    segmenter = None
    recorder = ChunkRecorder(args.record) if args.record else None
    keypoints_history, edges_arr = [], None
    ee_names = None                      # armdiff: the arm order of the EE pair

    if args.segmenter == 'armdiff' and args.bg_mode == 'temporal':
        print("  NOTE: --bg_mode temporal reports only what MOVED in the last "
              f"{args.lag} frames.\n"
              "        A cable at rest gives an EMPTY mask, so the init waits "
              "for ever.\n"
              "        Move the cable during the start-up, or use --bg_mode static.")

    def get_frame():
        """source.get(), minus the shipped masks when rehearsing the live path."""
        f = source.get()
        if f is not None and args.sam2_on_replay:
            f.mask = None
        return f

    with source:
        frame = get_frame()
        if frame is None:
            print('ERROR: the source gave no frame')
            return
        K = frame.K
        print(f"First frame: {frame.color.shape[1]}x{frame.color.shape[0]}  "
              f"K = fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}")

        # ---------------- first frame: EE pair + mask ----------------
        # a live segmenter is necessary whenever the source gives no masks
        # (kinect, or replay rehearsing the live path via --sam2_on_replay)
        if args.sam2_on_replay:
            from realtime.sam2_segmenter import Sam2Segmenter
            print(f"  Loading SAM2 ({args.sam2_model})...")
            segmenter = Sam2Segmenter(model=args.sam2_model, close_ksize=args.close_ksize)
        elif args.source in ('kinect', 'zed'):
            if args.segmenter == 'pcdiff':
                from realtime.pcdiff_segmenter import PointCloudDiffSegmenter
                segmenter = PointCloudDiffSegmenter(
                    K, n_background=args.bg_frames,
                    threshold_mm=args.pcdiff_threshold,
                    z_range=tuple(args.z_range), close_ksize=args.close_ksize)
                print(f"  Recording {args.bg_frames} empty-scene frames for the "
                      f"point-cloud reference -- keep the scene EMPTY...")
                while not segmenter.add_background(frame.depth):
                    frame = get_frame()
                    if frame is None:
                        print('ERROR: source ended during reference capture')
                        return
                print("  Reference stored. PLACE THE CABLE now (open, not crossed).")
                frame = get_frame()
                if frame is None:
                    print('ERROR: source ended after reference capture')
                    return
            elif args.segmenter == 'armdiff':
                from realtime.armdiff_segmenter import (ArmDiffSegmenter, ArmDiffPipeline,
                                                        ee_mask_distance_px)
                from realtime.arm_reference import (FrankaArmModel, ArmDepthRenderer,
                                                    load_base2cam, GRASP_Z)
                from realtime.joint_source import CrispJointSource, ConstantJointSource

                calib = Path(args.calib) if args.calib else \
                    script_dir / args.data_base / 'calibration' / 'transform_ee_cam_world.npz'
                print(f"  armdiff: hand-eye calibration from {calib}")
                base2cam = load_base2cam(calib)
                model = FrankaArmModel(arm=args.arm_model, n_points=args.arm_points)
                # the EE pair keeps this order: left first, like ChunkRecorder
                ee_names = (args.left_ns.strip('/'), args.right_ns.strip('/'))
                renderer = ArmDepthRenderer(
                    {ee_names[0]: (model, base2cam['left']),
                     ee_names[1]: (model, base2cam['right'])},
                    K, frame.depth.shape)
                if args.joints == 'crisp':
                    print(f"  armdiff: waiting for /{args.left_ns} and "
                          f"/{args.right_ns} joint streams (crisp_py)...")
                    joints = CrispJointSource((args.left_ns, args.right_ns)).start()
                else:
                    print("  armdiff: FIXED ready-pose joints (rendering rehearsal)")
                    joints = ConstantJointSource(
                        {args.left_ns.strip('/'): ConstantJointSource.READY,
                         args.right_ns.strip('/'): ConstantJointSource.READY})
                segmenter = ArmDiffPipeline(
                    ArmDiffSegmenter(K, mode=args.bg_mode, lag=args.lag,
                                     n_background=args.bg_frames,
                                     threshold_mm=args.pcdiff_threshold,
                                     arm_tol_mm=args.arm_tol,
                                     arm_dilate_px=args.arm_dilate,
                                     z_range=tuple(args.z_range),
                                     close_ksize=args.close_ksize),
                    renderer, joints,
                    grasp_z=(GRASP_Z if args.grasp_offset is None
                             else args.grasp_offset))
                if args.bg_mode == 'static':
                    print(f"  Recording {args.bg_frames} empty-scene frames for the "
                          f"background reference -- keep the scene EMPTY...")
                    while not segmenter.add_background(frame.depth):
                        frame = get_frame()
                        if frame is None:
                            print('ERROR: source ended during reference capture')
                            return
                    print("  Reference stored. PLACE THE CABLE now (open, not crossed).")
                    frame = get_frame()
                    if frame is None:
                        print('ERROR: source ended after reference capture')
                        return
            else:
                from realtime.sam2_segmenter import Sam2Segmenter
                print(f"  Loading SAM2 ({args.sam2_model})...")
                segmenter = Sam2Segmenter(model=args.sam2_model, close_ksize=args.close_ksize)

        if init_mode == 'replay':
            ee_poses_3d = source.ee_poses_3d                 # (T,2,3) mm, robot stream

        elif init_mode == 'click':
            clicks = click_points(window, frame.color)
            if clicks is None:
                print('Aborted.')
                return
            ends = [pixel_to_3d_mm(frame.depth, c, K) for c in clicks[:2]]
            if any(e is None for e in ends):
                print('ERROR: no depth at a clicked end; click again on the cable')
                return
            ee_poses_3d = np.array([ends])                   # (1,2,3): init frame only
            print(f"  Clicked ends (camera mm): {ends[0].round(1)}  {ends[1].round(1)}")
            if segmenter is not None:
                if args.segmenter in ('pcdiff', 'armdiff'):
                    frame.mask = segmenter.segment(frame.depth)   # clicks give the EE pair only
                else:
                    frame.mask = segmenter.segment(frame.color, prompt_points_xy=clicks)

        elif init_mode == 'fk':
            # armdiff only: the grippers HOLD the cable ends, and the joint
            # stream plus the hand-eye calibration give them directly. Nothing
            # here reads the colour image, so no candidate mask can hijack the
            # start (realtime/bootstrap.py is not used at all).
            #
            # The frame is accepted when the segmenter's own mask is large
            # enough AND reaches both projected grippers. Without the second
            # test the tracker would initialize on any surviving blob, and the
            # EE pair would snap to a skeleton that belongs to something else.
            # The wait is a TIME budget, not a frame count: one attempt costs
            # only a render plus array ops (~10 ms), so --max_init_attempts
            # would expire in seconds -- long before anyone can place a cable.
            print(f"  fk init: EE pair from the joint stream (grasp offset "
                  f"{segmenter.grasp_z * 1000:.1f} mm along hand +z, "
                  f"arms {ee_names[0]}/{ee_names[1]})")
            print(f"  PLACE THE CABLE in both grippers now (open, not crossed). "
                  f"Waiting up to {args.init_timeout:.0f} s...")
            ee_mm, dist, n_px, ok = None, np.full(2, np.inf), 0, False
            t_wait, t_said = time.monotonic(), 0.0
            while time.monotonic() - t_wait < args.init_timeout:
                frame.mask = segmenter.segment(frame.depth)
                ee_mm = segmenter.ee_poses_mm(ee_names)
                n_px = int(frame.mask.sum())
                ee_px_now = project_mm(ee_mm, K)
                dist = ee_mask_distance_px(frame.mask, ee_px_now)
                if n_px >= args.min_init_mask_px and dist.max() <= args.max_ee_mask_px:
                    ok = True
                    break
                waited = time.monotonic() - t_wait
                if waited - t_said >= 2.0:
                    t_said = waited
                    print(f"  fk init: mask {n_px} px (need "
                          f"{args.min_init_mask_px}), gripper-to-mask "
                          f"{dist[0]:.0f}/{dist[1]:.0f} px (need "
                          f"<= {args.max_ee_mask_px:.0f})  [{waited:.0f} s]")
                    if debug_dir is not None:
                        panel = debug_panel(frame.color, frame.depth, frame.mask,
                                            getattr(segmenter, 'last_arm_depth', None),
                                            ee_px_now, tuple(args.z_range),
                                            scale=args.debug_scale,
                                            label=f'init {waited:.0f}s {n_px}px')
                        cv2.imwrite(str(debug_dir / f'init_{waited:04.0f}s.jpg'),
                                    panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
                frame = get_frame()
                if frame is None:
                    print('ERROR: source ended before the fk init accepted a frame')
                    return
            if not ok:
                print(f"ERROR: fk init gave up after {args.init_timeout:.0f} s: "
                      f"the armdiff mask never reached both grippers "
                      f"(mask {n_px} px, gripper-to-mask "
                      f"{dist[0]:.0f}/{dist[1]:.0f} px).\n"
                      f"  In --bg_mode temporal a cable at REST is invisible -- "
                      f"move the cable, or use --bg_mode static.\n"
                      f"  Other causes: the cable is outside --z_range, "
                      f"--arm_tol/--arm_dilate eat it at the gripper (raise "
                      f"--max_ee_mask_px), or the hand-eye calibration is off.")
                return
            ee_poses_3d = ee_mm[None]                        # (1,2,3): live pair
            print(f"  fk init OK (mask {n_px} px, gripper-to-mask "
                  f"{dist[0]:.0f}/{dist[1]:.0f} px)")
            print(f"  FK ends (camera mm): {ee_mm[0].round(1)}  {ee_mm[1].round(1)}")
            if debug_dir is not None:
                panel = debug_panel(frame.color, frame.depth, frame.mask,
                                    getattr(segmenter, 'last_arm_depth', None),
                                    project_mm(ee_mm, K), tuple(args.z_range),
                                    scale=args.debug_scale,
                                    label=f'init ACCEPTED {n_px}px')
                cv2.imwrite(str(debug_dir / 'init_accepted.jpg'), panel,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                print(f"  init panels -> {debug_dir}/init_*.jpg")

        else:                                                # auto: no manual input
            bg = None
            # pcdiff/armdiff make their own reference; their mask becomes the
            # trusted frame.mask candidate, so --bootstrap is moot.
            if args.bootstrap == 'bgsub' and args.segmenter == 'sam2':
                bg = BackgroundSubtractor(n_background=args.bg_frames)
                print(f"  Recording {args.bg_frames} empty-scene frames -- keep the scene EMPTY...")
                while not bg.add_background(frame.depth):
                    frame = get_frame()
                    if frame is None:
                        print('ERROR: source ended during background capture')
                        return
                print("  Background stored. PLACE THE CABLE now (open, not crossed).")
                frame = get_frame()

            # pcdiff/armdiff have no prompt/refine interface -- auto_init must
            # not treat them as SAM2 (the candidate mask itself is the result)
            sam2 = segmenter if args.segmenter == 'sam2' else None
            # ... and their mask must not be REPLACED by a ridge candidate when
            # it fails the acceptance test (an empty mask always fails): rank
            # order alone does not protect it, trusted_only does.
            trusted_only = args.segmenter in ('pcdiff', 'armdiff')
            boot = None
            for attempt in range(args.max_init_attempts):
                if args.segmenter in ('pcdiff', 'armdiff') and segmenter is not None:
                    frame.mask = segmenter.segment(frame.depth)
                boot = auto_init(frame, segmenter=sam2, bg_subtractor=bg,
                                 z_range=tuple(args.z_range),
                                 trusted_only=trusted_only)
                if boot is not None:
                    break
                if attempt % 10 == 0:
                    print(f"  auto init: no accepted cable yet (attempt {attempt + 1})...")
                frame = get_frame()
                if frame is None:
                    print('ERROR: source ended before the auto init accepted a frame')
                    return
            if boot is None:
                print(f"ERROR: auto init failed after {args.max_init_attempts} frames. "
                      f"Adjust --z_range, try --bootstrap bgsub, or use --init click.")
                if trusted_only:
                    print(f"  --segmenter {args.segmenter}: only its OWN mask can "
                          f"start the session. Check that the mask is not empty; "
                          f"--init fk needs no mask acceptance test at all.")
                return
            ee_poses_3d = boot['ee_pair'][None]              # (1,2,3): init frame only
            frame.mask = boot['mask']
            print(f"  Auto init OK (source={boot['source']}, path {len(boot['path_rc'])} px)")
            print(f"  Auto ends (camera mm): {boot['ee_pair'][0].round(1)}  "
                  f"{boot['ee_pair'][1].round(1)}")

        tracker, intrinsics = build_tracker(K, args.n_keypoints, ee_poses_3d,
                                            max_depth=args.max_depth)
        ee_px = project_mm(ee_poses_3d[0], intrinsics)

        # ---------------- loop ----------------
        fps_ema, t_prev = 0.0, time.monotonic()
        n_done = 0
        ee_pair_now = ee_poses_3d[0]
        try:
            while frame is not None:
                mask = frame.mask
                if mask is None:
                    mask = (segmenter.segment(frame.depth)
                            if args.segmenter in ('pcdiff', 'armdiff')
                            else segmenter.segment(frame.color))

                depth = frame.depth.astype(np.float32)
                exclude_mask = (1 - (mask > 0)).astype(np.uint8)   # same inversion as dlo_tracking.py:95
                result = tracker.process_frame(depth=depth, arm_depth=None, rgb=None,
                                               precomputed_arm_mask=exclude_mask)

                if result.get('success'):
                    keypoints_history.append(result['keypoints'].copy())
                    if edges_arr is None and result.get('edges') is not None:
                        edges_arr = np.array(result['edges'])
                else:
                    keypoints_history.append(np.full((args.n_keypoints, 3), np.nan))
                    # Auto mode: refresh the stored EE pair from the current
                    # mask on every skipped frame. A warm restart re-runs the
                    # init, and the init reads ee_poses_3d[0] -- this keeps
                    # that entry CURRENT instead of frozen at frame 0.
                    if init_mode == 'auto':
                        acc = skeleton_path(mask)
                        if acc is not None:
                            fresh = [path_end_to_3d(acc['path_rc'], frame.depth, K, from_start=True),
                                     path_end_to_3d(acc['path_rc'], frame.depth, K, from_start=False)]
                            if all(e is not None for e in fresh):
                                tracker.ee_poses_3d = np.array([fresh])
                                ee_px = project_mm(tracker.ee_poses_3d[0], intrinsics)

                # fk mode: the joint stream gives the pair on every frame for
                # the cost of two FK chains, so keep it live. Single-DLO
                # TRACKING never reads ee_poses_3d (wire_tracker.py:1032-1044) --
                # this feeds a warm restart, the overlay and the recording.
                if init_mode == 'fk':
                    ee_pair_now = segmenter.ee_poses_mm(ee_names)
                    tracker.ee_poses_3d = ee_pair_now[None]
                    ee_px = project_mm(ee_pair_now, intrinsics)
                else:
                    ee_pair_now = ee_poses_3d[min(frame.idx, len(ee_poses_3d) - 1)]

                if recorder is not None:
                    recorder.add(frame, mask, ee_pair_now)

                now = time.monotonic()
                inst = 1.0 / max(now - t_prev, 1e-6)
                fps_ema = inst if n_done == 0 else 0.9 * fps_ema + 0.1 * inst
                t_prev = now
                n_done += 1
                if args.max_frames and n_done >= args.max_frames:
                    break

                # ---- verification view (every --debug_every frames) ----
                # Costs one colormap + one resize per tile, so it is kept off
                # the per-frame path: at N frames apart its amortized cost is
                # 1/N of a few ms.
                if args.debug_every and (n_done - 1) % args.debug_every == 0:
                    arm_depth_now = getattr(segmenter, 'last_arm_depth', None)
                    panel = debug_panel(frame.color, frame.depth, mask,
                                        arm_depth_now, ee_px,
                                        tuple(args.z_range),
                                        scale=args.debug_scale,
                                        label=f'#{frame.idx}')
                    if not args.no_display:
                        cv2.imshow(debug_window, panel)
                    if debug_dir is not None:
                        # jpg, not png: same picture, ~3 ms instead of ~16 ms
                        cv2.imwrite(str(debug_dir / f'panel_{frame.idx:05d}.jpg'),
                                    panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if args.segmenter == 'armdiff':
                        d_ee = ee_mask_distance_px(mask, ee_px)
                        print(f"  [dbg #{frame.idx}] mask {int((mask > 0).sum())} px  "
                              f"arm {int((arm_depth_now > 0).sum()) if arm_depth_now is not None else 0} px  "
                              f"EE->mask {np.round(d_ee, 1)} px  {fps_ema:.1f} fps")

                if not args.no_display:
                    cv2.imshow(window, draw_overlay(frame.color, result,
                                                    intrinsics, ee_px, fps_ema))
                    if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                        break

                frame = get_frame()
        except KeyboardInterrupt:
            print('\nStopped by user.')
        finally:
            if not args.no_display:
                cv2.destroyAllWindows()

    # ---------------- save ----------------
    raw_3d = np.array(keypoints_history)
    np.savez(out_dir / '3d_keypoints.npz',
             full=raw_3d,
             edge_connection=edges_arr if edges_arr is not None else np.array([]))
    n_ok = int(np.isfinite(raw_3d[:, 0, 0]).sum()) if len(raw_3d) else 0
    print(f"\nProcessed {n_done} frames ({n_ok} tracked)  mean {fps_ema:.1f} fps")
    print(f"Keypoints -> {out_dir / '3d_keypoints.npz'}")
    if recorder is not None:
        recorder.save(
            K,
            depth_offset_px=getattr(source, 'depth_offset_px', None) or 0.0,
            disparity_scale=getattr(source, 'depth_scale_a', 1.0) or 1.0,
            depth_source=getattr(source, 'depth_source', None) or args.source)


if __name__ == '__main__':
    main()
