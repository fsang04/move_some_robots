"""Self-check for the rendered-arm segmentation path (no camera, no robot).

    python realtime/test_arm_reference.py         # from trackDeform3D root

Checks, in order:
  1. FK correctness: rpy convention against scipy, and the flange position at
     the FR3 'ready' pose against the published value (0.307, 0, 0.590) m.
  2. Rendering: a synthetic camera sees the arm, depth values are sane,
     and the per-frame cost (FK + transform + splat) is measured.
  3. Segmentation end to end on synthetic depth: a moving 'cable' in front of
     a flat table with a moving arm -- the mask must keep the cable at its
     NEW location and drop the arm, the ghost, and the background. Both
     temporal and static modes.

numpy + cv2 + scipy only (the base tracking environment).
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime.arm_reference import (FrankaArmModel, ArmDepthRenderer,
                                    rpy_matrix, transform)
from realtime.armdiff_segmenter import ArmDiffSegmenter
from realtime.joint_source import ConstantJointSource

READY = ConstantJointSource.READY

K = np.array([[600.0, 0.0, 640.0],
              [0.0, 600.0, 360.0],
              [0.0, 0.0, 1.0]])
SHAPE = (720, 1280)

# camera 1.6 m in front of the robot, looking horizontally at the base area:
# camera x = base -y, camera y = base -z, camera z = base +x  (det = +1)
T_BASE2CAM = np.eye(4)
T_BASE2CAM[:3, :3] = np.array([[0.0, -1.0, 0.0],
                               [0.0, 0.0, -1.0],
                               [1.0, 0.0, 0.0]])
T_BASE2CAM[:3, 3] = [0.0, 0.45, 1.6]     # base sits 1.6 m ahead, 0.45 m down


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ''))
    if not ok:
        sys.exit(1)


def test_fk(model):
    print("1. forward kinematics")
    from scipy.spatial.transform import Rotation as R
    rpy = (0.3, -1.1, 2.0)
    check('URDF rpy convention matches scipy extrinsic xyz',
          np.allclose(rpy_matrix(*rpy), R.from_euler('xyz', rpy).as_matrix(), atol=1e-12))

    frames = model.body_transforms(READY)
    flange = frames[7] @ transform((0, 0, 0.107))     # link7 -> link8/flange
    pos = flange[:3, 3]
    ref = np.array([0.3069, 0.0, 0.5903])             # published FR3 ready flange
    check('flange position at the ready pose', np.allclose(pos, ref, atol=0.015),
          f'got {pos.round(4)}, want ~{ref}')
    z_axis = flange[:3, :3] @ [0, 0, 1]
    check('flange points straight down at the ready pose',
          np.allclose(z_axis, [0, 0, -1], atol=1e-6), f'z axis {z_axis.round(4)}')

    # fingers: symmetric about the hand's xz plane, gap = 2 * finger_q
    hand, lf, rf = frames[8], frames[9], frames[10]
    gap = np.linalg.norm(lf[:3, 3] - rf[:3, 3])
    check('finger gap = 2 x finger travel', np.isclose(gap, 0.08, atol=1e-9),
          f'gap {gap:.4f} m at finger_q=0.04')
    mid = 0.5 * (lf[:3, 3] + rf[:3, 3])
    expect_mid = (hand @ transform((0, 0, 0.0584)))[:3, 3]
    check('finger midpoint sits on the hand axis',
          np.allclose(mid, expect_mid, atol=1e-9))


def test_render(model):
    print("2. depth rendering")
    renderer = ArmDepthRenderer({'left': (model, T_BASE2CAM)}, K, SHAPE)
    q = {'left': READY}
    arm = renderer.render(q)
    n_px = int((arm > 0).sum())
    check('arm is visible', n_px > 5000, f'{n_px} px')
    vals = arm[arm > 0]
    check('depth range is sane (camera is ~0.7-2.5 m from the arm)',
          700 < vals.min() < 2500 and 700 < vals.max() < 2600,
          f'{vals.min():.0f}..{vals.max():.0f} mm')

    # a second arm doubles the work; time the realistic two-arm case
    T2 = T_BASE2CAM @ transform((0.0, 0.9, 0.0))      # second base 0.9 m along +x cam
    renderer2 = ArmDepthRenderer({'left': (model, T_BASE2CAM),
                                  'right': (model, T2)}, K, SHAPE)
    q2 = {'left': READY, 'right': READY}
    renderer2.render(q2)                              # warm-up
    t0 = time.perf_counter()
    n = 30
    for i in range(n):
        qq = {k: v + 0.02 * np.sin(i * 0.3) for k, v in q2.items()}
        renderer2.render(qq)
    ms = (time.perf_counter() - t0) / n * 1000.0
    check('two-arm render cost', ms < 25.0, f'{ms:.1f} ms/frame')
    return renderer


def composite(bg, arm_depth, cable_rect=None, cable_z=1300.0):
    """Synthetic measured depth: table + arm + optional cable bar."""
    d = bg.copy()
    a = arm_depth > 0
    d[a] = np.minimum(d[a], arm_depth[a])
    if cable_rect is not None:
        r0, r1, c0, c1 = cable_rect
        d[r0:r1, c0:c1] = cable_z
    return d.astype(np.uint16)


def test_segmenter(model):
    print("3. synthetic segmentation")
    renderer = ArmDepthRenderer({'left': (model, T_BASE2CAM)}, K, SHAPE)
    bg = np.full(SHAPE, 1500.0, np.float32)           # flat table at 1.5 m

    qA = {'left': READY}
    qB = {'left': READY + np.array([0.15, 0.1, 0, 0.1, 0, -0.1, 0])}
    armA, armB = renderer.render(qA), renderer.render(qB)

    cableA = (150, 165, 200, 640)                     # 15 px tall bar, moves down
    cableB = (250, 265, 200, 640)
    frameA = composite(bg, armA, cableA)
    frameB = composite(bg, armB, cableB)

    # -------- temporal mode: reference = previous frame --------
    seg = ArmDiffSegmenter(K, mode='temporal', lag=1, threshold_mm=30.0,
                           arm_tol_mm=40.0, arm_dilate_px=9,
                           z_range=(500.0, 2000.0), keep_largest=False)
    m0 = seg.segment(frameA, armA)
    check('temporal: warm-up frame gives an empty mask', m0.sum() == 0)
    mask = seg.segment(frameB, armB)

    new = np.zeros(SHAPE, bool); new[252:263, 210:630] = True     # cable interior
    old = np.zeros(SHAPE, bool); old[152:163, 210:630] = True     # ghost interior
    # pixels the arm claimed (grown by arm_dilate_px) in EITHER frame have no
    # trustworthy reference / are arm-removed -- the conservative drop there
    # is by design, so the coverage check applies outside them
    import cv2
    k = np.ones((19, 19), np.uint8)
    arm_zone = (cv2.dilate((armA > 0).astype(np.uint8), k) > 0) \
             | (cv2.dilate((armB > 0).astype(np.uint8), k) > 0)
    clear = new & ~arm_zone
    cable_hit = mask[clear].mean()
    ghost_hit = mask[old & ~arm_zone].mean() if (old & ~arm_zone).any() else 0.0
    arm_hit = mask[armB > 0].mean()
    check('temporal: moved cable is kept (outside the arm zone)',
          cable_hit > 0.98, f'{cable_hit:.2%} of {int(clear.sum())} px covered')
    check('temporal: ghost at the old location is dropped', ghost_hit < 0.02,
          f'{ghost_hit:.2%} leaked')
    check('temporal: rendered arm is dropped', arm_hit < 0.02, f'{arm_hit:.2%} leaked')
    # background = everything except the cable rects (plus a close-ksize
    # margin for the morphological close) and the arm zone
    pad = np.zeros(SHAPE, bool)
    pad[145:170, 195:645] = True                     # old rect + margin
    pad[245:270, 195:645] = True                     # new rect + margin
    bg_only = ~pad & ~arm_zone
    check('temporal: background is dropped', mask[bg_only].mean() < 0.001,
          f'{mask[bg_only].mean():.4%} leaked')

    # -------- static mode: empty-scene reference, arm enters later --------
    seg = ArmDiffSegmenter(K, mode='static', n_background=5, threshold_mm=30.0,
                           arm_tol_mm=40.0, arm_dilate_px=9,
                           z_range=(500.0, 2000.0), keep_largest=False)
    for _ in range(5):
        noisy = bg + np.random.default_rng(0).normal(0, 2, SHAPE).astype(np.float32)
        seg.add_background(noisy.astype(np.uint16))
    mask = seg.segment(frameB, armB)
    check('static: cable is kept (outside the arm zone)',
          mask[new & ~arm_zone].mean() > 0.98)
    # where the cable bar crosses IN FRONT of the arm the pixels must stay
    # (that is the point of the depth test) -- the leak check is arm-only px
    arm_only = (armB > 0) & ~pad
    check('static: arm working INSIDE the gate is dropped',
          mask[arm_only].mean() < 0.02, f'{mask[arm_only].mean():.2%} leaked')
    # only arm pixels in FRONT of the table can leak (the rest the table hides)
    visible_arm = (armB > 0) & (armB < 1450.0) & ~pad
    check('static: without arm_depth the arm would leak (sanity of the fix)',
          seg.segment(frameB, None)[visible_arm].mean() > 0.5,
          f'{seg.segment(frameB, None)[visible_arm].mean():.2%} of the visible arm')

    # timing: full segment() cost on top of the render
    seg = ArmDiffSegmenter(K, mode='temporal', lag=1)
    seg.segment(frameA, armA)
    t0 = time.perf_counter()
    for _ in range(30):
        seg.segment(frameB, armB)
    ms = (time.perf_counter() - t0) / 30 * 1000.0
    check('segment() cost', ms < 20.0, f'{ms:.1f} ms/frame')


def test_fk_ee_pair(model):
    """The --init fk path: the EE pair comes from the joints, not from an image.

    Scene: two arms at the ready pose and a synthetic cable drawn between their
    two grasp points, in front of a table. Checks that ee_poses_mm agrees with
    the FK, that the mask reaches both grippers, that the tracker initializes a
    real chain from that pair, and that a pair which does NOT belong to the
    mask is REJECTED instead of collapsing onto one pixel.
    """
    print("4. fk EE pair -> tracker init")
    import cv2
    from realtime.arm_reference import GRASP_Z
    from realtime.armdiff_segmenter import ArmDiffPipeline, ee_mask_distance_px
    from tracker.wire_tracker import WireTracker

    TABLE, Z_RANGE = 2200.0, (500.0, 2500.0)
    T2 = T_BASE2CAM @ transform((0.0, 0.9, 0.0))       # second base, 0.9 m away
    renderer = ArmDepthRenderer({'left': (model, T_BASE2CAM),
                                 'right': (model, T2)}, K, SHAPE)
    q = {'left': READY, 'right': READY}
    pipe = ArmDiffPipeline(
        ArmDiffSegmenter(K, mode='static', n_background=3, threshold_mm=30.0,
                         arm_tol_mm=40.0, arm_dilate_px=9, z_range=Z_RANGE,
                         keep_largest=True),
        renderer, ConstantJointSource({'left': READY, 'right': READY}))

    # ---- the EE pair itself ----
    ee_mm = pipe.ee_poses_mm(('left', 'right'))
    pts = renderer.grasp_points_cam_mm(q)
    check('ee_poses_mm follows the requested arm order',
          np.allclose(ee_mm[0], pts['left']) and np.allclose(ee_mm[1], pts['right']))
    check('ee_poses_mm order is not hard-coded',
          np.allclose(pipe.ee_poses_mm(('right', 'left')), ee_mm[::-1]))
    hand_cam = (T_BASE2CAM[:3, :3] @ model.body_transforms(READY)[8][:3, 3]
                + T_BASE2CAM[:3, 3]) * 1000.0
    check('grasp point sits GRASP_Z beyond the hand frame',
          np.isclose(np.linalg.norm(ee_mm[0] - hand_cam), GRASP_Z * 1000.0, atol=1e-6),
          f'{np.linalg.norm(ee_mm[0] - hand_cam):.1f} mm')
    ee_px = np.stack([ee_mm[:, 0] * K[0, 0] / ee_mm[:, 2] + K[0, 2],
                      ee_mm[:, 1] * K[1, 1] / ee_mm[:, 2] + K[1, 2]], axis=1)
    check('both grippers project into the image',
          ((ee_px[:, 0] > 0) & (ee_px[:, 0] < SHAPE[1])
           & (ee_px[:, 1] > 0) & (ee_px[:, 1] < SHAPE[0])).all(),
          f'{ee_px.round(0).tolist()}')

    # ---- synthetic scene: a cable between the two grasp points ----
    arm = renderer.render(q)
    depth = np.full(SHAPE, TABLE, np.float32)
    on_arm = arm > 0
    depth[on_arm] = np.minimum(depth[on_arm], arm[on_arm])
    cable = np.zeros(SHAPE, np.uint8)
    cv2.line(cable, tuple(ee_px[0].round().astype(int)),
             tuple(ee_px[1].round().astype(int)), 1, 12)
    depth[cable > 0] = float(ee_mm[:, 2].mean())
    depth = depth.astype(np.uint16)

    for _ in range(3):
        pipe.add_background(np.full(SHAPE, TABLE, np.uint16))
    mask = pipe.segment(depth)
    check('the cable survives the arm subtraction', int(mask.sum()) > 500,
          f'{int(mask.sum())} px')
    dist = ee_mask_distance_px(mask, ee_px)
    check('the mask reaches both grippers', dist.max() <= 60.0,
          f'{dist[0]:.0f}/{dist[1]:.0f} px from the grasp pixels')

    # ---- the tracker initializes a real chain from that pair ----
    n_kp = 15
    intr = np.array([[K[0, 0], 0, K[0, 2]], [0, K[1, 1], K[1, 2]], [0, 0, 1]])
    tracker = WireTracker(intrinsics=intr, n_keypoints=n_kp,
                          target_branch_nodes=0, target_leaf_nodes=2,
                          max_depth=Z_RANGE[1], top_k_components=1,
                          repulsion_iterations=20, repulsion_lr=10.0,
                          repulsion_k_neighbors=3, ee_poses_3d=ee_mm[None])
    res = tracker.process_frame(depth=depth.astype(np.float32), arm_depth=None,
                                precomputed_arm_mask=(1 - (mask > 0)).astype(np.uint8))
    check('init succeeds on the fk pair', bool(res.get('success')),
          f"reason={res.get('reason')}")
    kp = res['keypoints']
    spread = float(np.linalg.norm(kp.max(axis=0) - kp.min(axis=0)))
    check('keypoints are spread along the cable, not collapsed', spread > 500.0,
          f'{spread:.0f} mm bounding-box diagonal')
    edge_len = np.array([np.linalg.norm(kp[i] - kp[j]) for i, j in res['edges']])
    check('every chain edge has a real length', edge_len.min() > 1.0,
          f'shortest edge {edge_len.min():.1f} mm')

    # ---- the guard: an EE pair that does NOT belong to the mask ----
    # A bar off to one side, well below both grippers, with MORE than
    # min_skeleton_pixels of skeleton -- so the run reaches the EE projection.
    # Every pixel above the bar has the same nearest skeleton pixel (its top
    # end), so both EEs snap there, and the BFS between them is one pixel long.
    # That is the live failure: init used to report success with all 15
    # keypoints stacked on that one pixel.
    far = np.zeros(SHAPE, np.uint8)
    far[450:650, 20:36] = 1
    bad = WireTracker(intrinsics=intr, n_keypoints=n_kp, target_branch_nodes=0,
                      target_leaf_nodes=2, max_depth=Z_RANGE[1],
                      top_k_components=1, repulsion_iterations=20,
                      ee_poses_3d=ee_mm[None])
    d2 = np.full(SHAPE, TABLE, np.float32)
    d2[far > 0] = float(ee_mm[:, 2].mean())
    res2 = bad.process_frame(depth=d2, arm_depth=None,
                             precomputed_arm_mask=(1 - far).astype(np.uint8))
    check('a pair that misses the mask is REJECTED, not collapsed',
          not res2.get('success')
          and res2.get('reason') in ('ee_pair_on_one_skeleton_pixel',
                                     'ee_path_too_short'),
          f"reason={res2.get('reason')}")
    check('a rejected init leaves the tracker uninitialized',
          not bad.is_initialized)
    check('the driver would have refused this frame anyway',
          ee_mask_distance_px(far, ee_px).min() > 60.0,
          f'{ee_mask_distance_px(far, ee_px).round(0).tolist()} px to the mask')


if __name__ == '__main__':
    t0 = time.perf_counter()
    model = FrankaArmModel()
    print(f"model load + sampling: {(time.perf_counter() - t0) * 1000:.0f} ms "
          f"({sum(len(s) for s in model.samples)} points)")
    test_fk(model)
    test_render(model)
    test_segmenter(model)
    test_fk_ee_pair(model)
    print("all checks passed")
