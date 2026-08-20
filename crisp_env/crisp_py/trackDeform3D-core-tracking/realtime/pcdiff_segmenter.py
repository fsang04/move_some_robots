"""Point-cloud-difference segmenter for the live tracking port.

Integrates deformable_seg per-frame segmentation math
(deformable_seg/seg_with_arms_utils.py: depth_to_point_cloud_full ->
background_subtraction -> apply_depth_threshold -> largest component) with the
live Frame stream. The offline script diffs against a SYNCHRONIZED arm-only
replay of the same robot trajectory; live there is no second synchronized
stream, so the reference is a STATIC empty-scene depth median -- the same
capture pattern as bootstrap.BackgroundSubtractor, but the comparison is the
full 3D point distance (the deformable_seg formulation), not only z.

The 3D distance costs nothing extra here: both clouds are back-projections of
the SAME pixel grid, so the difference vector lies along the pixel's viewing
ray and ||p - p_bg|| == ||(xf, yf, 1)|| * |z - z_bg| exactly. The per-pixel
ray norms are precomputed once; segment() is then a few array ops per frame.

Consequences of the static reference:
  * anything that was in the reference scene and later MOVES (a robot arm)
    becomes foreground -- keep the arms outside the z_range workspace gate,
    or accept that the largest-component filter must beat them;
  * pixels with no valid reference depth are treated as background
    (conservative; same convention as bootstrap.BackgroundSubtractor).

No torch / GPU: numpy + cv2 only.
"""
import numpy as np

from realtime.sam2_segmenter import clean_mask


class PointCloudDiffSegmenter:
    """Empty-scene 3D background subtraction: one cleaned mask per frame.

    Usage:
        seg = PointCloudDiffSegmenter(K)
        while not seg.add_background(frame.depth):   # scene EMPTY
            frame = source.get()
        # place the object, then per frame:
        mask = seg.segment(frame.depth)

    threshold_mm plays the role of seg_with_arms.py's
    arm_subtraction_threshold (80 mm there, against a noisier moving-arm
    reference; a static median reference supports a tighter default).
    """

    def __init__(self, K, n_background: int = 30, threshold_mm: float = 30.0,
                 z_range=(500.0, 2000.0), close_ksize: int = 5):
        self.K = np.asarray(K, dtype=np.float64)
        self.n_background = n_background
        self.threshold_mm = float(threshold_mm)
        self.z_range = z_range
        self.close_ksize = close_ksize
        self._frames = []
        self._bg_z = None       # (H,W) float32 mm, median empty-scene depth
        self._bg_valid = None   # (H,W) bool
        self._ray = None        # (H,W) float32, ||(xf, yf, 1)|| per pixel

    @property
    def ready(self) -> bool:
        return self._bg_z is not None

    def add_background(self, depth) -> bool:
        """Feed one empty-scene depth frame (uint16 mm). True when enough."""
        self._frames.append(depth.astype(np.float32))
        if len(self._frames) < self.n_background:
            return False
        stack = np.stack(self._frames)
        stack[stack == 0] = np.nan       # holes must not drag the median down
        self._bg_z = np.nan_to_num(np.nanmedian(stack, axis=0)).astype(np.float32)
        self._bg_valid = self._bg_z > 0
        self._frames = []

        H, W = self._bg_z.shape
        u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                           np.arange(H, dtype=np.float32))
        xf = (u - self.K[0, 2]) / self.K[0, 0]
        yf = (v - self.K[1, 2]) / self.K[1, 1]
        self._ray = np.sqrt(xf * xf + yf * yf + 1.0).astype(np.float32)
        return True

    def segment(self, depth) -> np.ndarray:
        """(H,W) uint16 mm -> cleaned (H,W) uint8 mask, values 0 and 1."""
        if not self.ready:
            raise RuntimeError('feed empty-scene frames via add_background() first')
        d = depth.astype(np.float32)
        dist = self._ray * np.abs(d - self._bg_z)     # exact 3D point distance
        fg = ((dist > self.threshold_mm) & self._bg_valid
              & (d > self.z_range[0]) & (d < self.z_range[1]))
        return clean_mask(fg.astype(np.uint8), self.close_ksize, keep_largest=True)
