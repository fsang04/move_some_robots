"""Arm-aware point-cloud-difference segmenter for the live tracking port.

The live port of the offline deformable_seg segmentation
(seg_with_arms_utils.py: background_subtraction -> apply_depth_threshold ->
largest component) for the case pcdiff_segmenter.py explicitly does NOT
solve (README.md §4): the arms manipulating the object INSIDE the workspace
gate. Offline, the arm reference is a synchronized arm-only replay of the
same trajectory; live, arm_reference.py RENDERS that reference per frame
from the streamed joint configurations.

Per frame, three subtractions on the depth image (both clouds share the
same pixel rays, so every 3D point distance is |dz| * ||ray|| exactly, the
pcdiff_segmenter.py identity):

  1. background: a pixel is a candidate only if its point moved CLOSER than
     threshold_mm against the reference depth. Two reference modes:
       'temporal' -- the frame `lag` frames ago (ring buffer; the two-capture
                     scheme: no start-up ritual, the arms+object may already
                     be in view; only what MOVES within `lag` frames is
                     found). The closer-only test kills the ghosts a moving
                     object leaves at its OLD location.
       'static'   -- the median of n_background empty-scene frames (the
                     pcdiff_segmenter capture ritual; finds the object even
                     when it rests, and the arms are handled by step 2
                     instead of poisoning the reference).
  2. arms: pixels at or BEHIND the rendered arm depth (within arm_tol_mm,
     after growing the rendered silhouette by arm_dilate_px) are removed.
     Points clearly IN FRONT of the arm survive -- a cable hanging in front
     of a wrist stays in the mask.
  3. cleanup: workspace z gate, valid depth, morphological close + largest
     component (the same clean_mask every live segmenter uses).

numpy + cv2 only. The renderer adds ~3-6 ms/frame; the diffs are array ops.
"""
from collections import deque

import cv2
import numpy as np

from realtime.sam2_segmenter import clean_mask
from realtime.arm_reference import ArmDepthRenderer, GRASP_Z  # noqa: F401  (re-export)


def ee_mask_distance_px(mask, ee_px):
    """(N,) px distance from each EE pixel to the nearest mask pixel.

    The agreement test between the two halves of an armdiff session: the mask
    says where the object is, the joint stream says where the grippers are, and
    the grippers HOLD the object. A mask that does not reach them is not the
    object they hold, and an initialization on it puts the whole keypoint chain
    somewhere else. The distance transform of the INVERTED mask gives the field
    in one call; np.inf marks an EE pixel outside the image, or an empty mask.

    Expect a few tens of px even when everything is right: segment() removes
    the pixels at and behind the rendered arm (grown by arm_dilate_px), so the
    mask stops short of the fingertips by about that margin.
    """
    ee_px = np.atleast_2d(ee_px)
    if not mask.any():
        return np.full(len(ee_px), np.inf)
    dt = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
    H, W = mask.shape
    out = []
    for x, y in ee_px:
        xi, yi = int(round(x)), int(round(y))
        inside = 0 <= xi < W and 0 <= yi < H
        out.append(float(dt[yi, xi]) if inside else np.inf)
    return np.array(out)


class ArmDiffSegmenter:
    """One cleaned object mask per frame from depth + rendered arm depth.

    Usage (temporal mode -- no ritual, arms may be moving from frame 0):
        seg = ArmDiffSegmenter(K, mode='temporal', lag=5)
        mask = seg.segment(frame.depth, arm_depth)   # empty until lag frames

    Usage (static mode -- pcdiff-style empty-scene reference):
        seg = ArmDiffSegmenter(K, mode='static')
        while not seg.add_background(frame.depth): frame = source.get()
        mask = seg.segment(frame.depth, arm_depth)

    threshold_mm  -- background diff: how far a point must move to be
                     foreground (>= sensor noise, <= object height).
    arm_tol_mm    -- arm diff: how close to the rendered arm surface a point
                     may lie and still be called arm. Covers collision-mesh
                     inaccuracy + hand-eye calibration error + the joint
                     stream lagging the pixels; raise it before blaming FK.
    arm_dilate_px -- 2D growth of the rendered silhouette (min-filter, so
                     the grown ring keeps a valid arm depth and the in-front
                     test still works there).
    """

    def __init__(self, K, mode: str = 'temporal', lag: int = 5,
                 n_background: int = 30, threshold_mm: float = 30.0,
                 arm_tol_mm: float = 40.0, arm_dilate_px: int = 9,
                 z_range=(500.0, 2000.0), close_ksize: int = 5,
                 keep_largest: bool = True):
        if mode not in ('temporal', 'static'):
            raise ValueError(f"mode must be 'temporal' or 'static', got {mode!r}")
        self.K = np.asarray(K, dtype=np.float64)
        self.mode = mode
        self.lag = int(lag)
        self.n_background = int(n_background)
        self.threshold_mm = float(threshold_mm)
        self.arm_tol_mm = float(arm_tol_mm)
        self.arm_dilate_px = int(arm_dilate_px)
        self.z_range = z_range
        self.close_ksize = close_ksize
        self.keep_largest = keep_largest

        self._ring = deque(maxlen=max(1, self.lag))   # (depth f32, arm_valid)
        self._bg_frames = []
        self._bg_z = None
        self._bg_valid = None
        self._ray = None
        if self.arm_dilate_px > 0:
            k = 2 * self.arm_dilate_px + 1
            self._dilate_kernel = np.ones((k, k), np.uint8)
        else:
            self._dilate_kernel = None

    # ------------------------------------------------------------------ setup

    @property
    def ready(self) -> bool:
        """True when segment() can produce a non-empty mask."""
        if self.mode == 'static':
            return self._bg_z is not None
        return len(self._ring) == self._ring.maxlen

    def add_background(self, depth) -> bool:
        """Static mode: feed one EMPTY-scene depth frame (uint16 mm).
        Returns True when the reference is complete."""
        if self.mode != 'static':
            raise RuntimeError("add_background() is for mode='static'; "
                               "temporal mode needs no ritual")
        self._bg_frames.append(depth.astype(np.float32))
        if len(self._bg_frames) < self.n_background:
            return False
        stack = np.stack(self._bg_frames)
        stack[stack == 0] = np.nan               # holes must not drag the median
        self._bg_z = np.nan_to_num(np.nanmedian(stack, axis=0)).astype(np.float32)
        self._bg_valid = self._bg_z > 0
        self._bg_frames = []
        return True

    def _rays(self, shape):
        if self._ray is None or self._ray.shape != shape:
            H, W = shape
            u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                               np.arange(H, dtype=np.float32))
            xf = (u - self.K[0, 2]) / self.K[0, 0]
            yf = (v - self.K[1, 2]) / self.K[1, 1]
            self._ray = np.sqrt(xf * xf + yf * yf + 1.0).astype(np.float32)
        return self._ray

    # ---------------------------------------------------------------- per frame

    def segment(self, depth, arm_depth=None) -> np.ndarray:
        """(H,W) uint16 mm depth [+ (H,W) float32 mm rendered arm depth]
        -> cleaned (H,W) uint8 mask, values 0 and 1.

        arm_depth=None skips the arm subtraction (temporal mode then behaves
        like pure two-capture differencing: arms AND object stay foreground).
        """
        d = depth.astype(np.float32)
        ray = self._rays(d.shape)

        arm_removed, arm_valid = self._arm_test(d, arm_depth)

        # ---- 1. background reference ----
        if self.mode == 'static':
            if self._bg_z is None:
                raise RuntimeError("mode='static': feed empty-scene frames "
                                   "via add_background() first")
            ref_z, ref_valid = self._bg_z, self._bg_valid
        else:
            if len(self._ring) == self._ring.maxlen:
                ref_z, ref_arm = self._ring[0]        # the frame `lag` ago
                ref_valid = (ref_z > 0) & ~ref_arm    # was-arm pixels: no ref
            else:
                ref_z, ref_valid = None, None
            self._ring.append((d, arm_valid))
            if ref_z is None:
                return np.zeros(d.shape, np.uint8)    # warming up

        # closer-only 3D distance: keeps what arrived, drops the ghost of
        # what left (their diff has the opposite sign)
        moved = ray * (ref_z - d) > self.threshold_mm

        # ---- 2 + 3. arm subtraction, gates, cleanup ----
        fg = (moved & ref_valid & ~arm_removed
              & (d > self.z_range[0]) & (d < self.z_range[1]))
        return clean_mask(fg.astype(np.uint8), self.close_ksize,
                          keep_largest=self.keep_largest)

    def _arm_test(self, d, arm_depth):
        """-> (arm_removed, arm_valid): pixels the arm claims (at/behind its
        rendered surface, within tolerance) and pixels it covers at all."""
        if arm_depth is None:
            zeros = np.zeros(d.shape, bool)
            return zeros, zeros
        a = arm_depth
        if self._dilate_kernel is not None:
            grow = a.copy()
            grow[a == 0] = np.float32(1e9)            # min-filter: grown ring
            grow = cv2.erode(grow, self._dilate_kernel)  # keeps a valid depth
            a = np.where(grow < 1e8, grow, 0.0).astype(np.float32)
        arm_valid = a > 0
        arm_removed = arm_valid & (d >= a - self.arm_tol_mm)
        return arm_removed, arm_valid


class ArmDiffPipeline:
    """Renderer + joint source + segmenter behind the one-argument
    segment(depth) interface the live driver already speaks (the
    PointCloudDiffSegmenter duck type: ready / add_background / segment).

    ee_poses_mm() adds what no other live segmenter can give: the gripper
    positions. The joint stream and the hand-eye calibration are already here,
    so the driver reads the EE pair from FK instead of a guess from the image
    (--init fk in dlo_tracking_live.py).

    last_arm_depth keeps the most recent rendered arm depth for display and
    debugging (driver overlay, threshold tuning). last_snapshot keeps the joint
    state that made it, so ee_poses_mm() returns the poses of the SAME frame as
    the mask.
    """

    def __init__(self, segmenter: ArmDiffSegmenter, renderer, joint_source,
                 grasp_z: float = GRASP_Z):
        self.segmenter = segmenter
        self.renderer = renderer
        self.joint_source = joint_source
        self.grasp_z = float(grasp_z)
        self.last_arm_depth = None
        self.last_snapshot = None

    @property
    def ready(self) -> bool:
        return self.segmenter.ready

    def add_background(self, depth) -> bool:
        return self.segmenter.add_background(depth)

    def segment(self, depth) -> np.ndarray:
        snap = self.joint_source.latest()
        self.last_snapshot = snap
        self.last_arm_depth = self.renderer.render(snap.q, snap.finger_q)
        return self.segmenter.segment(depth, self.last_arm_depth)

    def ee_poses_mm(self, names=None, fresh: bool = False) -> np.ndarray:
        """(N, 3) gripper positions in CAMERA-frame mm, one row per arm.
        """
        snap = self.joint_source.latest() \
            if (fresh or self.last_snapshot is None) else self.last_snapshot
        pts = self.renderer.grasp_points_cam_mm(snap.q, self.grasp_z)
        order = list(pts) if names is None else [str(n).strip('/') for n in names]
        return np.array([pts[n] for n in order], dtype=np.float64)
