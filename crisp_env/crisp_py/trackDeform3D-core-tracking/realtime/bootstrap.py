"""Automatic first-frame start-up for the live DLO driver.

This module finds the cable in the first frame with no manual input
(no clicks, no boxes, no prompts). The steps (auto_init) are:

    1. Make one or more candidate masks (see the sources below).
    2. Skeletonize each mask and test it: does it look like one open cable?
    3. Take interior points of the skeleton path. Use them as the SAM2 prompt.
    4. Let SAM2 grow the mask to the full cable. Repeat while the path grows.
    5. Take the two farthest endpoints of the final mask. Back-project them
       through the depth image. These are the two gripper (EE) positions.

Candidate-mask sources (the best accepted path wins):

    1. The mask stored in the frame itself (replay data only).
    2. Depth background subtraction. This needs empty-scene frames first.
    3. Frangi ridge filter, limited to the workspace depth range. Idea:
       "the cable is the thin, long thing". Needs no color and no setup.

A candidate mask does not need to cover the whole cable. It only needs to be
correct where it is: its skeleton points must lie ON the cable. SAM2 then
completes the rest. (Test on chunk_1: a ridge mask that covered half the
cable gave a full-length path, with ends 16 mm from the true grippers.)

A mask passes the test only when its skeleton is one open curve: enough
pixels, exactly two real endpoints, and one path between them that covers
most of the skeleton. If no candidate passes, the caller tries the next
frame.
"""
from collections import deque

import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import skeletonize

from realtime.sam2_segmenter import clean_mask


# ============================================================================
# acceptance test + endpoint extraction (option A)
# ============================================================================

def skeleton_path(mask, min_skeleton_px: int = 100, min_coverage: float = 0.6,
                  max_mean_width_px: float = None):
    """Accept `mask` as one open cable, or return None.

    Returns dict(ends_rc (2,2), path_rc (N,2), skel, mean_width_px) on
    acceptance:
        ends_rc -- the two skeleton endpoints with the largest path distance
                   between them (rejects endpoints of small mask spurs)
        path_rc -- the ordered skeleton pixels between them

    Acceptance: >= min_skeleton_px skeleton pixels, >= 2 endpoint pixels
    (one-neighbor pixels), and the end-to-end path covers >= min_coverage of
    the whole skeleton (a cable skeleton IS mostly one path; a blob or a
    self-crossing cable is not).

    max_mean_width_px adds the THINNESS test: mask area / path length must
    stay below it. This is what separates a cable (~12 px wide on this rig)
    from another elongated object -- the live test showed SAM2 happily grows
    a prompt on a robot-arm highlight into the whole arm (~100 px wide), and
    only the width tells them apart. None = no width test.

    The mask is reduced to its largest connected component first -- raw masks
    (including the shipped ground truth) carry small noise blobs, and their
    skeletons would break the coverage test.
    """
    cleaned = clean_mask(mask, close_ksize=3, keep_largest=True)
    skel = skeletonize(cleaned > 0)
    coords = np.argwhere(skel)
    if len(coords) < min_skeleton_px:
        return None

    H, W = skel.shape
    idx = np.full((H, W), -1, dtype=np.int64)
    idx[coords[:, 0], coords[:, 1]] = np.arange(len(coords))
    neighbors = [[] for _ in range(len(coords))]
    for k, (r, c) in enumerate(coords):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and idx[rr, cc] >= 0:
                    neighbors[k].append(idx[rr, cc])

    endpoints = [k for k in range(len(coords)) if len(neighbors[k]) == 1]
    if len(endpoints) < 2:
        return None

    def bfs(start):
        dist = np.full(len(coords), -1, dtype=np.int64)
        parent = np.full(len(coords), -1, dtype=np.int64)
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in neighbors[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
        return dist, parent

    # double BFS: farthest endpoint pair along the skeleton (graph diameter)
    d0, _ = bfs(endpoints[0])
    reach = [k for k in endpoints if d0[k] >= 0]
    a = reach[int(np.argmax(d0[reach]))]
    d1, parent = bfs(a)
    reach = [k for k in endpoints if d1[k] >= 0]
    b = reach[int(np.argmax(d1[reach]))]
    if a == b:
        return None

    path = []
    cur = b
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    if len(path) < min_coverage * len(coords):
        return None

    mean_width = float(cleaned.sum()) / len(path)
    if max_mean_width_px is not None and mean_width > max_mean_width_px:
        return None

    return {
        'ends_rc': coords[[a, b]],
        'path_rc': coords[path],
        'skel': skel,
        'mean_width_px': mean_width,
    }


def sample_prompts_xy(path_rc, n_points: int = 8, lo: float = 0.08, hi: float = 0.92):
    """n_points (x, y) pixels spread along the INTERIOR of the skeleton path.

    The interior (lo..hi of the arc) is used because the path ends are the
    least reliable part of a partial candidate; interior points of an accepted
    path lie on the cable by construction and cover it lengthwise (the
    `gt_points` prompt mode that test_sam2_mask_dlo.py measured as SAM2's
    upper bound)."""
    pick = (np.linspace(lo, hi, n_points) * (len(path_rc) - 1)).astype(int)
    return path_rc[pick][:, ::-1].astype(np.float64)          # (row,col) -> (x,y)


def gripper_neg_prompts(mask, path_rc, width_factor: float = 2.0,
                        min_run_px: int = 4, max_walk: int = 80,
                        taper_px: int = 12, max_per_end: int = 2):
    """SAM2 negative prompts placed where the mask extends onto a gripper.

    This extension is detectable by width: along the skeleton path
    the cable is uniformly thin, the gripper is several times wider. Local
    width = 2 x distance transform of the mask at the path pixel; the cable's
    own width is the median over the path interior. A wide stretch that
    BEGINS within taper_px of a path end is bleed. The end pixel itself is
    not required to be wide: a skeleton tip inside a wide blob tapers thin
    right at the boundary (measured on chunk_1, the last pixel reads ~cable
    width while the 40 px behind it read 3x that).

    Returns (N, 2) float64 (x, y) negative points on the widest pixels of
    each end.
    """
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    w = 2.0 * dt[path_rc[:, 0], path_rc[:, 1]]
    n = len(w)
    if n < 20:
        return np.empty((0, 2))
    cable_w = float(np.median(w[int(0.2 * n):int(0.8 * n)]))
    thr = width_factor * cable_w
    negs = []
    for order in (np.arange(n), np.arange(n)[::-1]):     # walk in from each end
        head = order[:max_walk]
        wide_pos = np.flatnonzero(w[head] > thr)
        if len(wide_pos) >= min_run_px and wide_pos[0] <= taper_px:
            run = head[wide_pos]
            widest = run[np.argsort(w[run])[::-1][:max_per_end]]
            negs += [(float(path_rc[k, 1]), float(path_rc[k, 0])) for k in widest]
    return np.asarray(negs, dtype=np.float64).reshape(-1, 2)


def pixel_to_3d_mm(depth_u16, xy, K, win: int = 7):
    """Back-project one pixel to camera-frame mm via the median depth of a
    win x win window (the exact pixel can have no depth). None when the whole
    window has no depth."""
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


def path_end_to_3d(path_rc, depth, K, from_start: bool, max_walk: int = 40):
    """Back-project one end of the skeleton path, walking inward past depth
    holes (mask endpoints often sit on pixels with no depth return)."""
    steps = path_rc[:max_walk] if from_start else path_rc[::-1][:max_walk]
    for r, c in steps:
        p = pixel_to_3d_mm(depth, (c, r), K, win=7)
        if p is not None:
            return p
    return None


# ============================================================================
# promptless candidate-mask sources
# ============================================================================

def ridge_candidates(bgr, depth=None, z_range=(500.0, 2000.0),
                     sigmas=(1, 2, 3, 4, 5, 6), percentile: float = 98.0,
                     close_ksize: int = 9, top_k: int = 3,
                     min_component_px: int = 300):
    """Thin-structure (Frangi) candidates, color-free.

    CLAHE local-contrast enhancement runs first: measured on the live rig, a
    thin gray wire in a dim tent is invisible to Frangi on the raw gray image
    and clearly found after CLAHE (sigma 1 covers a 2-3 px wire; the wide
    close bridges the fragmented response of a low-contrast cable).

    Returns up to top_k component masks PER POLARITY (dark ridges and bright
    ridges), largest first. Multiple components per polarity matter: the live
    test showed the biggest bright thin structure can be a robot-arm specular
    highlight, with the actual cable as the second component. The caller
    tests every candidate and lets the downstream checks (acceptance + width)
    decide.

    The workspace depth gate (z_range, mm) removes thin structures outside
    the workspace -- window frames, tent edges, far clutter. The gate is
    DILATED before use, because the depth at the border pixels of a thin
    object is unreliable.

    One-time cost only (~1.5 s at 720p), so speed does not matter here.
    """
    gray8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray8 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16)).apply(gray8)
    gray = gray8.astype(np.float64) / 255.0
    gate = None
    if depth is not None:
        gate = ((depth > z_range[0]) & (depth < z_range[1])).astype(np.uint8)
        gate = cv2.dilate(gate, np.ones((9, 9), np.uint8))
    out = []
    for black_ridges in (True, False):
        r = frangi(gray, sigmas=sigmas, black_ridges=black_ridges)
        m = (r > np.percentile(r, percentile)).astype(np.uint8)
        if gate is not None:
            m &= gate
        if close_ksize > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                                 np.ones((close_ksize, close_ksize), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        order = 1 + np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]
        for comp in order[:top_k]:
            if stats[comp, cv2.CC_STAT_AREA] < min_component_px:
                break
            out.append((lab == comp).astype(np.uint8))
    return out


class BackgroundSubtractor:
    """Depth background subtraction: "the cable is the thing that was not
    there before". Feed empty-scene depth frames first, then call candidate().
    Weak for a cable lying flat on the table (mm-level depth change)."""

    def __init__(self, n_background: int = 30, delta_mm: float = 15.0,
                 close_ksize: int = 5):
        self.n_background = n_background
        self.delta_mm = delta_mm
        self.close_ksize = close_ksize
        self._frames = []
        self._bg = None

    @property
    def ready(self) -> bool:
        return self._bg is not None

    def add_background(self, depth) -> bool:
        """Feed one empty-scene depth frame. Returns True when enough."""
        self._frames.append(depth.astype(np.float32))
        if len(self._frames) >= self.n_background:
            stack = np.stack(self._frames)
            stack[stack == 0] = np.nan
            self._bg = np.nan_to_num(np.nanmedian(stack, axis=0))
            self._frames = []
            return True
        return False

    def candidate(self, depth):
        d = depth.astype(np.float32)
        m = ((self._bg - d > self.delta_mm) & (d > 0) & (self._bg > 0)).astype(np.uint8)
        return clean_mask(m, close_ksize=self.close_ksize, keep_largest=True)


# ============================================================================
# one-call bootstrap for the driver
# ============================================================================

def try_bootstrap(frame, bg_subtractor=None, min_skeleton_px: int = 100,
                  z_range=(500.0, 2000.0), trusted_only: bool = False):
    """Try every available candidate source on one frame.

    Returns a LIST of accepted candidates, each dict(mask, ends_rc, path_rc,
    source). Empty list when nothing passes (caller: next frame).

    Order: source trust first, path length second. The frame's own mask
    (replay) IS the object and always comes first; background subtraction is
    object-specific evidence and comes second; ridge components are generic
    "thin things" and come last, ranked by path length among themselves.

    trusted_only drops the ridge group. Rank order alone does not protect a
    trusted mask: the ranking only decides which ACCEPTED candidate wins, so a
    ridge component still takes over whenever the trusted mask fails the
    acceptance test (an empty pcdiff/armdiff mask always does). Those
    segmenters own the object, so pass trusted_only=True for them and let the
    caller retry the next frame instead. It also saves the ~1.5 s Frangi
    filter per attempt.
    """
    groups = [[], [], []]                            # frame_mask, bgsub, ridge
    if frame.mask is not None:                                   # replay path
        groups[0].append(('frame_mask', (frame.mask > 0).astype(np.uint8)))
    if bg_subtractor is not None and bg_subtractor.ready:
        groups[1].append(('bgsub', bg_subtractor.candidate(frame.depth)))
    if not trusted_only:
        groups[2] = [(f'ridge{i}', m) for i, m in
                     enumerate(ridge_candidates(frame.color, frame.depth, z_range))]

    accepted_list = []
    for group in groups:
        accepted_group = []
        for source, cand in group:
            accepted = skeleton_path(cand, min_skeleton_px=min_skeleton_px)
            if accepted is None:
                continue
            accepted_group.append({'mask': cand, 'ends_rc': accepted['ends_rc'],
                                   'path_rc': accepted['path_rc'], 'source': source})
        accepted_group.sort(key=lambda b: len(b['path_rc']), reverse=True)
        accepted_list += accepted_group
    return accepted_list


def auto_init(frame, segmenter=None, bg_subtractor=None,
              min_skeleton_px: int = 100, z_range=(500.0, 2000.0),
              n_prompts: int = 10, max_refine_rounds: int = 3,
              max_mean_width_px: float = 30.0, trusted_only: bool = False):
    """Fully automatic session start on one frame: mask + EE pair, no clicks.

    Every accepted candidate is tried in rank order. With a segmenter: the
    candidate's interior skeleton points prompt SAM2, and SAM2's mask is
    refined (re-prompt from its own skeleton) while the accepted path keeps
    growing. The FINAL mask must pass the acceptance test INCLUDING the
    thinness test (max_mean_width_px) -- this is what rejects a candidate
    that turned out to be a robot arm or another wide object, and moves on
    to the next candidate. The segmenter session stays prompted with the
    winning object, so the driver continues streaming with it.

    Without a segmenter (replay): the candidate mask itself is the result,
    with the same width test.

    Returns dict(mask, ee_pair (2,3) mm, path_rc, source) or None -- on None
    the caller simply tries the next frame.

    trusted_only forbids the ridge candidates (see try_bootstrap): use it when
    the frame mask comes from a segmenter that owns the object.
    """
    for boot in try_bootstrap(frame, bg_subtractor, min_skeleton_px, z_range,
                              trusted_only=trusted_only):
        mask, path_rc, source = boot['mask'], boot['path_rc'], boot['source']

        if segmenter is not None:
            prev_len, ok = 0, True
            for _ in range(max_refine_rounds):
                if len(path_rc) <= prev_len * 1.10:  # no real growth -> stop
                    break
                prev_len = len(path_rc)
                segmenter.reset()
                m = segmenter.segment(frame.color,
                                      prompt_points_xy=sample_prompts_xy(path_rc, n_prompts))
                accepted = skeleton_path(m, min_skeleton_px=min_skeleton_px,
                                         max_mean_width_px=max_mean_width_px)
                if accepted is None:
                    ok = False                       # wide/broken: next candidate
                    break
                mask, path_rc = m, accepted['path_rc']
            if not ok:
                continue

            # De-bleed: the final mask often continues past the grasp onto the
            # gripper (and SAM2's memory would keep that for every streamed
            # frame). Detect the too-wide end runs and re-prompt ONCE with
            # negative points on them, so the session remembers where the
            # cable stops. On failure the positive-only session is restored.
            negs = gripper_neg_prompts(mask, path_rc)
            if len(negs):
                pos = sample_prompts_xy(path_rc, n_prompts)
                segmenter.reset()
                m = segmenter.segment(frame.color, prompt_points_xy=pos,
                                      neg_points_xy=negs)
                accepted = skeleton_path(m, min_skeleton_px=min_skeleton_px,
                                         max_mean_width_px=max_mean_width_px)
                if accepted is not None:
                    mask, path_rc = m, accepted['path_rc']
                else:                                # trim broke the cable:
                    segmenter.reset()                # restore the session
                    segmenter.segment(frame.color, prompt_points_xy=pos)
        else:
            accepted = skeleton_path(mask, min_skeleton_px=min_skeleton_px,
                                     max_mean_width_px=max_mean_width_px)
            if accepted is None:
                continue
            path_rc = accepted['path_rc']

        ends = [path_end_to_3d(path_rc, frame.depth, frame.K, from_start=True),
                path_end_to_3d(path_rc, frame.depth, frame.K, from_start=False)]
        if any(e is None for e in ends):
            continue
        return {'mask': mask, 'ee_pair': np.array(ends),
                'path_rc': path_rc, 'source': source}
    return None
