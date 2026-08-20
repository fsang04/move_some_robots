"""Feasibility test: can SAM2 produce masks the DLO tracker can actually initialize on?

Runs SAM2 (HuggingFace `Sam2VideoModel`) over input_data/dlo/chunk_1 and scores every
frame against the shipped ground-truth masks. The decisive metric is NOT IoU -- it is
whether a BFS path still exists between the two gripper pixels on the thinned skeleton,
because that is literally what `_initialize_single_dlo` requires (wire_init.py:118-140).
A SAM2 mask with 0.9 IoU but one missing pixel fails the tracker.

Uses SAM2's STREAMING api (`init_video_session()` with no video, then one frame at a
time). Two reasons: it does not hold 606 preprocessed frames in VRAM, and it is the same
code path a live camera would use -- so the timing here is meaningful.

All foreground/skeleton computation calls the REAL tracker methods, so the numbers are
exactly what WireTracker would see.

Usage:
    python test_sam2_mask_dlo.py                                 # depth_snap, all frames
    python test_sam2_mask_dlo.py --prompt ee                     # raw projection (fails)
    python test_sam2_mask_dlo.py --prompt ee_snap                 # cheats with GT; SAM2 upper-mid
    python test_sam2_mask_dlo.py --prompt gt_points --n-points 8  # upper bound on SAM2
    python test_sam2_mask_dlo.py --snap-win 20 --snap-tol 40      # depth_snap tuning
    python test_sam2_mask_dlo.py --model facebook/sam2.1-hiera-small

Measured so far on dlo/chunk_1 (bf16, RTX A6000, 31.4 ms/frame steady state):
    --prompt ee        IoU 0.006   skeleton cleared the 100 px floor on   0.0% of frames
    --prompt ee_snap   IoU 0.692   skeleton cleared the floor on        100.0% of frames
The whole difference is one prompt point being 13 px off, so depth_snap exists to close
that gap without ground truth.

Requires: torch, transformers (with sam2_video), numpy, opencv-python, scikit-image.
"""

import argparse
import ast
import json
import struct
import time
import zipfile
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from tracker.wire_tracker import WireTracker
from utils.transforms import load_transforms, get_ee_positions_cam

# use CUDA synchronization TF32 for fp32 matmuls, and bf16 autocast at the forward call
# speedup test results: fp32 106.3 ms -> bf16+TF32 31.5 ms per frame.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================================
# npz readers that do not blow up memory
# ============================================================================

def parse_npy_header(fh):
    """Read an .npy header from a stream positioned at its start.

    Returns (shape, fortran_order, dtype). Pure Python on purpose: numpy 2.x removed
    the private `np.lib.format._read_array_header` that deform_with_hands/kinect.py:51
    still calls, so that helper raises AttributeError on numpy >= 2.
    """
    magic = fh.read(6)
    if magic != b'\x93NUMPY':
        raise ValueError(f'not a .npy stream (magic={magic!r})')
    major = fh.read(1)[0]
    fh.read(1)                                  # minor
    hlen_size = 2 if major == 1 else 4
    hlen = int.from_bytes(fh.read(hlen_size), 'little')
    hdr = ast.literal_eval(fh.read(hlen).decode('latin1').strip())
    return hdr['shape'], hdr['fortran_order'], np.dtype(hdr['descr'])


def memmap_npz(path, name):
    """Memory-map one STORED (uncompressed) member of an .npz.

    rgbd.npz members are Stored, so this avoids a 1.6 GB read.
    """
    with open(path, 'rb') as fh:
        info = zipfile.ZipFile(fh).getinfo(name + '.npy')
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"member '{name}' is compressed ({info.compress_type}); "
                             'cannot memmap -- use DeflatedFrameReader instead')
        fh.seek(info.header_offset)
        # ZIP local file header is 30 bytes; [26:28] name len, [28:30] extra len
        n, m = struct.unpack('<HH', fh.read(30)[26:30])
        fh.seek(info.header_offset + 30 + n + m)
        shape, fortran, dtype = parse_npy_header(fh)
        offset = fh.tell()
    if fortran:
        raise ValueError('fortran-order arrays are not supported')
    return np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=shape)


class DeflatedFrameReader:
    """Sequential frame reader for a DEFLATED (H,W) uint8 member of an .npz.

    masks.npz is compressed, so it cannot be memmapped and a full load is 558 MB.
    Frames are read in order, one at a time. Call next() with strictly increasing idx.
    """

    def __init__(self, path, member):
        self.z = zipfile.ZipFile(path)
        self.fh = self.z.open(member + '.npy')
        self.shape, fortran, dtype = parse_npy_header(self.fh)
        if fortran:
            raise ValueError('fortran-order arrays are not supported')
        if dtype.itemsize != 1:
            raise ValueError(f'expected a 1-byte dtype (uint8/bool), got {dtype}')
        self.dtype = dtype
        self.H, self.W = self.shape[1], self.shape[2]
        self.frame_bytes = self.H * self.W
        self.pos = 0

    def next(self, idx):
        assert idx >= self.pos, f'reader is sequential: asked {idx}, at {self.pos}'
        while self.pos < idx:                      # skip forward
            self.fh.read(self.frame_bytes)
            self.pos += 1
        buf = self.fh.read(self.frame_bytes)
        self.pos += 1
        return np.frombuffer(buf, dtype=self.dtype).reshape(self.H, self.W)

    def close(self):
        self.fh.close()
        self.z.close()


def load_poses(path, n_take=None):
    """left_arm_poses.npz / right_arm_poses.npz -> (T, 7). Mirrors data_loading.py:40."""
    z = np.load(path)
    n = len(z.files)
    n = min(n, n_take) if n_take else n
    return np.array([z[f'arr_{i}'] for i in range(n)])


# ============================================================================
# geometry helpers -- reuse the tracker's own graph so the test is faithful
# ============================================================================

def project_to_pixels(pts_3d, K):
    """(N,3) camera-frame mm -> (N,2) as (row, col). Same formula as
    WireTracker._project_3d_to_2d (wire_tracker.py:325-344)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = np.maximum(pts_3d[:, 2], 1e-6)
    col = pts_3d[:, 0] * fx / z + cx
    row = pts_3d[:, 1] * fy / z + cy
    return np.stack([row, col], axis=1)


def bfs_path_exists(adjacency, start_idx, end_idx):
    """Exactly the BFS the DLO init runs (wire_init.py:122-140)."""
    if start_idx is None or end_idx is None:
        return False
    visited = np.zeros(len(adjacency), dtype=bool)
    q = deque([start_idx])
    visited[start_idx] = True
    while q:
        cur = q.popleft()
        if cur == end_idx:
            return True
        for nb, _ in adjacency[cur]:
            if not visited[nb]:
                visited[nb] = True
                q.append(nb)
    return bool(visited[end_idx])


def nearest_coord_idx(coords, target_rowcol):
    """Snap a pixel to the nearest skeleton node. Mirrors wire_init.py:107-109."""
    if len(coords) == 0:
        return None
    d = np.linalg.norm(coords - np.asarray(target_rowcol, dtype=np.float64), axis=1)
    return int(np.argmin(d))


def clean_mask(m, close_ksize, keep_largest):
    """The cleanup a live segmenter MUST do itself: the tracker only filters
    components on the init frame (wire_tracker.py:274-276)."""
    out = (m > 0).astype(np.uint8)
    if close_ksize > 0:
        k = np.ones((close_ksize, close_ksize), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    if keep_largest:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        if n > 1:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            out = (lab == biggest).astype(np.uint8)
    return out


def score_frame(tracker, mask, depth, ee_rowcol, max_depth, max_snap_px=30.0):
    """Foreground -> skeleton -> BFS, using the tracker's real methods.

    Returns the four acceptance numbers plus the intermediate counts.
    """
    fg = ((mask > 0) & (depth > 0)).astype(np.uint8)
    fg = tracker._apply_depth_threshold(fg, depth)          # wire_tracker.py:188
    n_fg = int(fg.sum())

    n_cc = 0
    if n_fg:
        n_cc = cv2.connectedComponentsWithStats(fg, connectivity=8)[0] - 1

    # what an INIT frame sees (top-k with k=1) vs what a TRACKING frame sees (no filter)
    fg_init = tracker._get_top_k_components(fg, k=1) if n_fg else fg
    skel_init = tracker._skeletonize(fg_init)
    skel_track = tracker._skeletonize(fg)
    n_skel_init, n_skel_track = int(skel_init.sum()), int(skel_track.sum())

    # `ee_path` must mean "the tracker would initialize CORRECTLY", so it is gated twice.
    #
    #  1. the skeleton must clear min_skeleton_pixels. process_frame checks this FIRST
    #     (wire_tracker.py:1204) and returns 'insufficient_skeleton', so without this gate
    #     the BFS runs on frames the tracker would never reach.
    #  2. both snaps must be short. nearest_coord_idx has no distance limit -- neither does
    #     the real wire_init.py:108 -- so on a tiny stray blob BOTH gripper pixels snap
    #     into that same blob and a path trivially exists. That is how an earlier run
    #     reported 72.4% success while the skeleton cleared the floor on 0.0% of frames:
    #     439/606 of those "successes" had skeletons of 2-44 px.
    #
    # Gate 2 makes this test STRICTER than the shipped tracker. That is deliberate: the
    # tracker would "succeed" on a wrong blob, and a silent wrong init is a failure here.
    reachable = False
    snap0 = snap1 = float('nan')
    if n_skel_init >= tracker.min_skeleton_pixels:
        coords, _, adjacency = tracker._build_skeleton_graph(skel_init)  # wire_init.py:450
        s = nearest_coord_idx(coords, ee_rowcol[0])
        e = nearest_coord_idx(coords, ee_rowcol[1])
        if s is not None and e is not None:
            snap0 = float(np.linalg.norm(coords[s] - np.asarray(ee_rowcol[0], float)))
            snap1 = float(np.linalg.norm(coords[e] - np.asarray(ee_rowcol[1], float)))
            if s != e and max(snap0, snap1) <= max_snap_px:
                reachable = bfs_path_exists(adjacency, s, e)

    return dict(n_fg=n_fg, n_cc=n_cc, n_skel_init=n_skel_init,
                n_skel_track=n_skel_track, ee_path=reachable,
                snap0=snap0, snap1=snap1)


def iou(a, b):
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


# ============================================================================
# prompt construction
# ============================================================================

def depth_snap(px_rc, ee_z, depth, max_depth, win=20, tol=40.0):
    """Move a projected gripper pixel onto the nearest pixel whose DEPTH looks right.

    Live-available: uses only the depth image and the gripper's own 3D position, so it
    needs no mask and no ground truth. That matters because SAM2 needs the point in
    order to produce the mask, while wire_init.py:106-109 snaps to a mask it already has.

    NOT a directed search. Every pixel in the (2*win+1)^2 window is tested against three
    conditions and the NEAREST survivor wins, so the direction is an output.

    Defaults from a measured sweep on dlo/chunk_1 (40 frames x 2 grippers), distance to
    the nearest true cable pixel:
        raw projection      mean 9.15 px   max 16.68   within 2 px  21%
        win=20 tol=40       mean 2.02 px   max  5.00   within 2 px  66%   <- these defaults
        win=45 tol=120      mean 2.34 px   max  5.39   within 2 px  55%
        win=10 tol=20       mean 1.39 px   max  3.16   but NO CANDIDATE on 45/80 tries
    tol must exceed the systematic depth offset: the cable sits ~20 mm FARTHER from the
    camera than the reported gripper, so tol=20 rejects the cable itself.

    Caveat measured on frame 0: the depth test is not very selective -- 1685 of 8281
    window pixels passed and only 28% of those were cable. This lands the point NEAR the
    cable; it does not identify the cable.

    Returns the snapped (row, col), or None when no pixel in the window qualifies.
    """
    H, W = depth.shape
    r0, c0 = int(round(px_rc[0])), int(round(px_rc[1]))
    r1, r2 = max(0, r0 - win), min(H, r0 + win + 1)
    c1, c2 = max(0, c0 - win), min(W, c0 + win + 1)
    if r1 >= r2 or c1 >= c2:
        return None
    sub = depth[r1:r2, c1:c2].astype(np.float32)
    ok = (sub > 0) & (sub < max_depth) & (np.abs(sub - ee_z) < tol)
    if not ok.any():
        return None
    rr, cc = np.nonzero(ok)
    j = int(np.argmin((rr + r1 - px_rc[0]) ** 2 + (cc + c1 - px_rc[1]) ** 2))
    return np.array([rr[j] + r1, cc[j] + c1], dtype=np.float64)


def build_prompt(mode, ee_rowcol, ee_z, gt_mask, depth, max_depth,
                 n_points=8, win=20, tol=40.0):
    """Return (points_xy, labels) for SAM2's first frame.

    ee         -- raw projected gripper pixels. Measured 9.15 px off the cable, and one
                  gripper landed on a finger with NO depth return -> IoU 0.006.
    depth_snap -- ee, then moved onto the nearest correct-depth pixel. THE LIVE CANDIDATE:
                  needs no ground truth. Measured 2.02 px off.
    ee_snap    -- gripper pixels snapped to the nearest TRUE cable pixel. Not reproducible
                  live; isolates "can SAM2 track a thin cable" from "is the prompt good".
    gt_points  -- n_points spread along the true cable. Upper bound on SAM2's ability.
    """
    if mode == 'ee':
        pts_rc = np.asarray(ee_rowcol, dtype=np.float64)

    elif mode == 'depth_snap':
        pts_rc = []
        for k, p in enumerate(ee_rowcol):
            s = depth_snap(p, ee_z[k], depth, max_depth, win, tol)
            if s is None:
                print(f'    WARNING: depth_snap found no candidate for gripper {k} '
                      f'(win={win}, tol={tol}); falling back to the raw projection')
                s = np.asarray(p, dtype=np.float64)
            else:
                print(f'    gripper {k}: ({p[0]:.1f},{p[1]:.1f}) -> ({s[0]:.0f},{s[1]:.0f})  '
                      f'moved {np.linalg.norm(s - p):.1f} px')
            pts_rc.append(s)
        pts_rc = np.stack(pts_rc)

    elif mode in ('ee_snap', 'gt_points'):
        gt_rc = np.argwhere(gt_mask > 0)
        if len(gt_rc) == 0:
            raise RuntimeError('ground-truth mask is empty on the prompt frame')
        if mode == 'ee_snap':
            pts_rc = np.stack([gt_rc[np.argmin(np.linalg.norm(gt_rc - p, axis=1))]
                               for p in ee_rowcol])
        else:
            # spread along the cable by arc order proxy: sort by row+col, take even strides
            order = np.argsort(gt_rc[:, 0] + gt_rc[:, 1])
            pick = np.linspace(0, len(order) - 1, n_points).astype(int)
            pts_rc = gt_rc[order[pick]]
    else:
        raise ValueError(mode)

    pts_xy = np.asarray(pts_rc, dtype=np.float64)[:, ::-1]      # (row,col) -> (x,y)
    labels = np.ones(len(pts_xy), dtype=np.int32)
    return pts_xy, labels


# ============================================================================
# main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--chunk-dir', default='input_data/dlo/chunk_1')
    ap.add_argument('--calib-dir', default='input_data/dlo/calibration')
    ap.add_argument('--mask-file', default='masks/masks.npz')
    ap.add_argument('--mask-key', default='masks')
    ap.add_argument('--model', default='facebook/sam2.1-hiera-tiny',
                    help='also try facebook/sam2.1-hiera-small / -base-plus / -large')
    ap.add_argument('--prompt', default='depth_snap',
                    choices=['ee', 'depth_snap', 'ee_snap', 'gt_points'],
                    help='depth_snap is the only mode reproducible on a live camera')
    ap.add_argument('--n-points', type=int, default=8, help='for --prompt gt_points')
    ap.add_argument('--snap-win', type=int, default=20,
                    help='depth_snap search half-width, px (swept: 20 is best)')
    ap.add_argument('--snap-tol', type=float, default=40.0,
                    help='depth_snap depth tolerance, mm. Must exceed the ~20 mm '
                         'systematic gripper-to-cable offset (swept: 40 is best)')
    ap.add_argument('--max-snap-px', type=float, default=30.0,
                    help='ee_path fails if a gripper snaps further than this onto the '
                         'skeleton; without it a tiny stray blob scores as success')
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--max-depth', type=float, default=2000.0,
                    help='must match the driver (dlo_tracking.py:63), NOT the class default')
    ap.add_argument('--close-ksize', type=int, default=5,
                    help='morphological close to bridge SAM2 gaps; 0 to disable')
    ap.add_argument('--no-keep-largest', action='store_true')
    ap.add_argument('--mask-threshold', type=float, default=0.0)
    ap.add_argument('--out', default='output/sam2_mask_test_dlo.json')
    ap.add_argument('--save-overlays', type=int, default=0,
                    help='write this many overlay PNGs (leading frames only)')
    ap.add_argument('--save-video', default=None,
                    help='write an annotated mp4 of EVERY frame, e.g. '
                         'output/sam2_depthsnap.mp4 -- the way to see drift and failures')
    ap.add_argument('--video-fps', type=int, default=30)
    args = ap.parse_args()

    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    root = Path(__file__).resolve().parent
    chunk = root / args.chunk_dir
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}  model={args.model}  prompt={args.prompt}')

    # ---------------- data ----------------
    color = memmap_npz(chunk / 'rgbd.npz', 'color')      # (T,H,W,3) uint8 BGR
    depth_mm = memmap_npz(chunk / 'rgbd.npz', 'depth')   # (T,H,W) uint16 mm
    gt = DeflatedFrameReader(chunk / args.mask_file, args.mask_key)

    T = min(color.shape[0], depth_mm.shape[0], gt.shape[0])
    if args.max_frames:
        T = min(T, args.max_frames)
    H, W = color.shape[1], color.shape[2]
    print(f'frames={T}  {H}x{W}  color={color.dtype} depth={depth_mm.dtype}')

    tf = load_transforms(root / args.calib_dir)
    K = tf['K']
    intrinsics = np.array([[K[0, 0], 0, K[0, 2]], [0, K[1, 1], K[1, 2]], [0, 0, 1]])

    left = load_poses(chunk / 'left_arm_poses.npz', T)
    right = load_poses(chunk / 'right_arm_poses.npz', T)
    n_pose = min(len(left), len(right))
    if n_pose < T:
        print(f'  note: only {n_pose} poses for {T} frames; truncating to {n_pose}')
        T = n_pose

    ee_3d = np.stack([get_ee_positions_cam(left[i], right[i],
                                          tf['T_left_base2cam'], tf['T_right_base2cam'])
                      for i in range(T)])                                  # (T,2,3) mm
    ee_px = np.stack([project_to_pixels(ee_3d[i], intrinsics) for i in range(T)])  # (T,2,2)

    # Tracker instance used ONLY for its segmentation/graph helpers, so the numbers
    # below are exactly what WireTracker would compute. Params match dlo_tracking.py.
    tracker = WireTracker(intrinsics=intrinsics, n_keypoints=15,
                          target_branch_nodes=0, target_leaf_nodes=2,
                          max_depth=args.max_depth, top_k_components=1)

    # ---------------- model ----------------
    model = Sam2VideoModel.from_pretrained(args.model, device_map=device).eval()
    processor = Sam2VideoProcessor.from_pretrained(args.model)
    session = processor.init_video_session(inference_device=device)   # streaming: no video

    rows = []
    ov_dir = root / 'output' / 'sam2_mask_test_overlays'
    if args.save_overlays:
        ov_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if args.save_video:
        vid_path = root / args.save_video
        vid_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                 args.video_fps, (W, H))
        if not writer.isOpened():
            raise RuntimeError(f'could not open {vid_path} for writing')
        print(f'  writing video -> {vid_path}')

    for i in range(T):
        bgr = np.asarray(color[i])
        dep = np.asarray(depth_mm[i]).astype(np.float32)
        gt_i = gt.next(i)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)      # SAM2 wants RGB; npz stores BGR

        t0 = time.time()
        inputs = processor(images=rgb, device=device, return_tensors='pt').to(model.device)

        if i == 0:
            pts_xy, labels = build_prompt(args.prompt, ee_px[0], ee_3d[0, :, 2], gt_i,
                                          dep, args.max_depth, args.n_points,
                                          args.snap_win, args.snap_tol)
            print(f'  prompt ({args.prompt}): {len(pts_xy)} point(s) '
                  f'{[tuple(np.round(p, 1)) for p in pts_xy]}')
            processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=0,
                obj_ids=1,
                input_points=[[pts_xy.tolist()]],       # [batch][obj][point][xy]
                input_labels=[[labels.tolist()]],
                original_size=inputs.original_sizes[0], # required when streaming
            )

        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(inference_session=session, frame=inputs.pixel_values[0])
        logits = processor.post_process_masks(
            [out.pred_masks], original_sizes=inputs.original_sizes, binarize=False)[0]
        sam_ms = (time.time() - t0) * 1000.0

        raw = (logits[0, 0].float().cpu().numpy() > args.mask_threshold).astype(np.uint8)
        if raw.shape != (H, W):                          # belt and braces
            raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_NEAREST)
        pred = clean_mask(raw, args.close_ksize, not args.no_keep_largest)

        s_pred = score_frame(tracker, pred, dep, ee_px[i], args.max_depth, args.max_snap_px)
        s_gt = score_frame(tracker, gt_i, dep, ee_px[i], args.max_depth, args.max_snap_px)

        rows.append(dict(frame=i, sam_ms=sam_ms,
                         iou=iou(pred, gt_i),
                         n_px_pred=int(pred.sum()), n_px_gt=int((gt_i > 0).sum()),
                         n_cc_raw=int(cv2.connectedComponentsWithStats(raw, 8)[0] - 1),
                         **{f'pred_{k}': v for k, v in s_pred.items()},
                         **{f'gt_{k}': v for k, v in s_gt.items()}))

        if i < args.save_overlays or writer is not None:
            row = rows[-1]
            vis = bgr.copy()
            vis[gt_i > 0] = (0, 255, 0)                  # ground truth green (missed)
            vis[pred > 0] = (0, 0, 255)                  # SAM2 red (extra)
            vis[(pred > 0) & (gt_i > 0)] = (0, 255, 255) # overlap yellow (correct)
            for r, c in ee_px[i]:
                cv2.circle(vis, (int(c), int(r)), 7, (255, 0, 255), 2)
            ok = row['pred_ee_path']
            cv2.rectangle(vis, (0, 0), (W, 78), (0, 0, 0), -1)
            cv2.putText(vis, f"frame {i:4d}/{T}   IoU {row['iou']:.3f}   "
                             f"skel {row['pred_n_skel_init']:4d}   "
                             f"snap {max(row['pred_snap0'], row['pred_snap1']):5.1f}px",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(vis, 'INIT OK' if ok else 'INIT WOULD FAIL',
                        (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if ok else (0, 0, 255), 2)
            cv2.putText(vis, 'yellow=correct  green=missed  red=extra  magenta=gripper',
                        (W - 640, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            if i < args.save_overlays:
                cv2.imwrite(str(ov_dir / f'{i:05d}.png'), vis)
            if writer is not None:
                writer.write(vis)

        if i % 25 == 0 or i == T - 1:
            r = rows[-1]
            print(f"  [{i:4d}/{T}] iou={r['iou']:.3f} px={r['n_px_pred']:6d} "
                  f"cc_raw={r['n_cc_raw']:3d} skel_init={r['pred_n_skel_init']:5d} "
                  f"ee_path={r['pred_ee_path']}  {r['sam_ms']:.0f}ms")

    gt.close()
    if writer is not None:
        writer.release()

    # ---------------- verdict ----------------
    def frac(key):
        return 100.0 * sum(bool(r[key]) for r in rows) / len(rows)

    ious = np.array([r['iou'] for r in rows])
    skel = np.array([r['pred_n_skel_init'] for r in rows])
    ccs = np.array([r['n_cc_raw'] for r in rows])
    ms = np.array([r['sam_ms'] for r in rows])
    floor = tracker.min_skeleton_pixels

    print('\n' + '=' * 78)
    print(f'SAM2 MASK FEASIBILITY -- {args.model} -- prompt={args.prompt} -- {len(rows)} frames')
    print('=' * 78)
    print(f'  IoU vs ground truth      mean {ious.mean():.3f}  min {ious.min():.3f}')
    print(f'  raw components (want 1)  mean {ccs.mean():.2f}  max {ccs.max()}  '
          f'| ==1 on {100.0 * (ccs == 1).mean():.1f}% of frames')
    print(f'  skeleton px (floor {floor})  mean {skel.mean():.0f}  min {skel.min()}  '
          f'| above floor on {100.0 * (skel >= floor).mean():.1f}% of frames')
    print(f'  SAM2 time per frame      mean {ms.mean():.1f} ms  max {ms.max():.1f} ms  '
          f'-> {1000.0 / ms.mean():.1f} fps')
    sn = np.array([max(r['pred_snap0'], r['pred_snap1']) for r in rows], float)
    sn_ok = sn[np.isfinite(sn)]
    print(f'  gripper->skeleton snap   mean {sn_ok.mean() if len(sn_ok) else float("nan"):.1f} px  '
          f'max {sn_ok.max() if len(sn_ok) else float("nan"):.1f}  '
          f'(limit {args.max_snap_px:.0f}; n/a on {int((~np.isfinite(sn)).sum())} frames)')
    print()
    print(f'  >>> WOULD INITIALIZE CORRECTLY (the acceptance test): {frac("pred_ee_path"):.1f}% of frames')
    print(f'      same test on ground-truth masks:                 {frac("gt_ee_path"):.1f}% of frames')
    print(f'      (requires: skeleton >= {floor} px AND both gripper snaps <= '
          f'{args.max_snap_px:.0f} px AND a BFS path between them)')
    print()
    ok_init = rows[0]['pred_ee_path']
    print(f'  Frame 0 would initialize: {"YES" if ok_init else "NO"}')
    above = 100.0 * (skel >= floor).mean()
    if above < 95.0:
        print(f'  VERDICT: FAIL on the skeleton floor -- only {above:.1f}% of frames reach '
              f'{floor} px.')
        print('           The tracker returns \'insufficient_skeleton\' and never even')
        print('           attempts an init. Check the prompt first (--save-overlays 5),')
        print('           then --close-ksize / --model.')
    elif frac('pred_ee_path') < 95.0:
        print('  VERDICT: skeleton size is fine but the topology is not -- gaps or a bad')
        print('           snap break the gripper-to-gripper path. Try --close-ksize 7.')
    else:
        print('  VERDICT: PASS. SAM2 masks would initialize and track. Now check the fps')
        print('           line against 33.3 ms (SAM2 pipelined on its own thread); the')
        print('           22 ms inline budget is NOT the relevant bar.')

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(config=vars(args), rows=rows), indent=1))
    print(f'\n  per-frame report -> {out_path}')
    if args.save_overlays:
        print(f'  overlays        -> {ov_dir}')


if __name__ == '__main__':
    main()
