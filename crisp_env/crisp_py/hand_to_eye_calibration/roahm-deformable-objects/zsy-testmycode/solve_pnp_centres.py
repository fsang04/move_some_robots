#!/usr/bin/env python3
"""Hand-eye extrinsic from tag CENTRES alone, by cv2.solvePnP. One side at a time.

The idea: the tag centre traced through the capture trajectory is a cloud of 3D
points whose base-frame coordinates the robot already knows (FK @ gripper2tag),
each seen at one 2D pixel. That is a textbook camera-resectioning problem, and
solvePnP answers it directly:

    objectPoints  tag centre in the ROBOT BASE frame  = (T_base_gripper @ gripper2tag)[:3,3]
    imagePoints   detected centre (u,v)               = dump_tag_detections.py -> centres.npz
    cameraMatrix  K at 2208x1242                      = zed_capture/zed_intrinsics_2208x1242.npz
    distCoeffs    zeros                               = VIEW.LEFT is already rectified
    -> (rvec, tvec)  =  T_cam_base, i.e. exactly the X_cam_base the nonlinear solver reports

What this formulation BUYS, versus calculate_base_to_cam_nonlinear_opt.py:

  * Depth is never touched. The whole ZED disparity-offset workaround
    (zed_depth_correction.json, the +186 mm at 1.5 m, the double-correction trap)
    is irrelevant to a solve that only reads pixels.
  * AprilTag's per-frame ROTATION is never used, and a small planar tag at 1.4 m
    is precisely where that estimate is weakest.
  * Only the TRANSLATION column of the hand-measured gripper2tag matters. Its
    rotation block -- unverified, and visibly hand-edited in the source -- drops
    out of the problem entirely, because a point has no orientation.

What it COSTS: with no rotation observations, the solution leans entirely on the
3D spread of the centres and the angular spread of the viewing rays. Those are
reported below; a tight ray cone is the failure mode to watch for.

Reprojection error is NOT a fair way to rank this against the depth solve -- PnP
minimises it by construction. The honest referee is the base-to-base check at
the end, which uses no tag and no depth: solve both arms independently, then ask
what each pair implies about where one robot base sits relative to the other.

    python solve_pnp_centres.py --calib-seq-name zed_calib_003
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from calculate_base_to_cam_nonlinear_opt import _load_robot_poses, gripper2tag

REPO = Path(__file__).resolve().parents[3]          # .../crisp_py
INTRINSICS = REPO / 'zed_capture' / 'zed_intrinsics_2208x1242.npz'


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def load_side(base: Path, side: str):
    """-> object points (N,3) in base frame, image points (N,2), image indices."""
    d = np.load(base / f'detections_{side}' / 'centres.npz', allow_pickle=True)
    idx, uv = np.asarray(d['image_index']), np.asarray(d['centre_uv'], dtype=np.float64)
    pos, rot = _load_robot_poses(base / f'{side}_calibration_poses.npz',
                                 max_images=int(idx.max()) + 1)
    obj = []
    for i in idx:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = rot[i], pos[i]
        obj.append((T @ gripper2tag)[:3, 3])         # tag CENTRE in base frame
    return np.asarray(obj, dtype=np.float64), uv, idx


def reproj_err(obj, uv, R, t, K, dist):
    proj, _ = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], t, K, dist)
    return np.linalg.norm(proj.reshape(-1, 2) - uv, axis=1)


def solve_side(base: Path, side: str, K, dist, ransac_px: float):
    obj, uv, idx = load_side(base, side)
    n = len(obj)
    print(f'\n{"="*72}\n{side.upper()}  --  {n} centres\n{"="*72}')

    # Geometry first: a resectioning result is only as trustworthy as the spread
    # of the points it was solved from.
    s = np.linalg.svd(obj - obj.mean(0), compute_uv=False)
    print(f'  object-point extent XYZ : {np.round(obj.ptp(0), 3)} m')
    print(f'  PCA singular values     : {np.round(s, 3)}   planarity s3/s1 = {s[2]/s[0]:.3f}')
    print(f'  image footprint         : u {uv[:,0].ptp():.0f} px, v {uv[:,1].ptp():.0f} px  of 2208x1242')

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj, uv, K, dist, flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=ransac_px, iterationsCount=20000, confidence=0.9999)
    if not ok:
        raise RuntimeError(f'{side}: solvePnPRansac failed')
    inl = np.sort(inliers.ravel()) if inliers is not None else np.arange(n)
    print(f'  RANSAC ({ransac_px:.1f} px)        : {len(inl)}/{n} inliers'
          + (f'   outlier image ids {list(idx[np.setdiff1d(np.arange(n), inl)])}'
             if len(inl) < n else ''))

    rvec, tvec = cv2.solvePnPRefineLM(obj[inl], uv[inl], K, dist, rvec, tvec)
    R = cv2.Rodrigues(rvec)[0]
    t = tvec.ravel()

    e_in = reproj_err(obj[inl], uv[inl], R, t, K, dist)
    e_all = reproj_err(obj, uv, R, t, K, dist)
    print(f'  reprojection, inliers   : rms {np.sqrt((e_in**2).mean()):.2f} px, '
          f'median {np.median(e_in):.2f}, max {e_in.max():.2f}')
    print(f'  reprojection, all points: rms {np.sqrt((e_all**2).mean()):.2f} px, '
          f'median {np.median(e_all):.2f}, max {e_all.max():.2f}')

    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, t
    ray = obj @ R.T + t
    u = ray / np.linalg.norm(ray, axis=1, keepdims=True)
    m = u.mean(0) / np.linalg.norm(u.mean(0))
    print(f'  camera-frame depth      : {ray[:,2].min():.3f} .. {ray[:,2].max():.3f} m '
          f'(span {ray[:,2].ptp()*1000:.0f} mm)')
    print(f'  viewing-ray cone        : max {np.degrees(np.arccos(np.clip(u@m,-1,1))).max():.1f} deg off the mean ray')

    # Same points, but scored with the DEPLOYED depth-translation extrinsic. PnP
    # wins this metric by construction -- it is reported to size the disagreement,
    # not to declare a winner.
    dep = base / f'base2cam_transform_{side}_nonlinear_opt_depth_translation.npz'
    X_dep = None
    if dep.exists():
        X_dep = np.asarray(np.load(dep)['X_cam_base'], dtype=np.float64)
        e_dep = reproj_err(obj, uv, X_dep[:3, :3], X_dep[:3, 3], K, dist)
        print(f'\n  deployed depth-translation extrinsic, on these same centres:')
        print(f'    reprojection          : rms {np.sqrt((e_dep**2).mean()):.2f} px, '
              f'median {np.median(e_dep):.2f}, max {e_dep.max():.2f}')
        print(f'    differs from PnP by   : {np.linalg.norm(X[:3,3]-X_dep[:3,3])*1000:.1f} mm, '
              f'{rot_angle_deg(X[:3,:3], X_dep[:3,:3]):.3f} deg')
    return X, X_dep


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--calib-seq-name', default='zed_calib_003')
    p.add_argument('--side', default='both', choices=['left', 'right', 'both'],
                   help="each side is always solved from its OWN frames only; "
                        "'both' just runs the two solves back to back and adds the "
                        "base-to-base cross-check, which needs a solution per arm")
    p.add_argument('--ransac-px', type=float, default=3.0)
    p.add_argument('--save', action='store_true',
                   help='write base2cam_transform_<side>_pnp_centres.npz beside the others')
    args = p.parse_args()

    base = Path(__file__).resolve().parents[1] / 'captured_calibration_data' / args.calib_seq_name
    intr = np.load(INTRINSICS)
    K, dist = np.asarray(intr['K'], dtype=np.float64), np.asarray(intr['dist'], dtype=np.float64)
    print(f'sequence   : {base}')
    print(f'intrinsics : {INTRINSICS.name}  fx={K[0,0]:.3f} cx={K[0,2]:.3f} cy={K[1,2]:.3f}')
    print(f'distCoeffs : {np.round(dist.ravel(), 6)}  (VIEW.LEFT is rectified)')

    sides = ('left', 'right') if args.side == 'both' else (args.side,)
    X, X_dep = {}, {}
    for side in sides:
        X[side], X_dep[side] = solve_side(base, side, K, dist, args.ransac_px)
        print(f'\n  X_cam_base ({side}), solvePnP on tag centres:')
        for row in X[side]:
            print('    [' + '  '.join(f'{v: .6f}' for v in row) + ']')
        if args.save:
            out = base / f'base2cam_transform_{side}_pnp_centres.npz'
            np.savez(out, X_cam_base=X[side], method='solvePnP_tag_centres',
                     disparity_offset_px=np.float64(0.0))
            print(f'  saved -> {out.name}')

    if args.side != 'both':
        return

    # The referee. Neither arm's solve knows the other exists, and this quantity
    # involves no tag and no depth -- so agreement is real evidence, not a
    # restatement of the objective either method minimised.
    print(f'\n{"="*72}\nBASE-TO-BASE CROSS-CHECK  (independent of both objectives)\n{"="*72}')
    for label, S in (('solvePnP centres', X), ('depth translation', X_dep)):
        if S['left'] is None or S['right'] is None:
            continue
        T = np.linalg.inv(S['left']) @ S['right']
        yaw = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
        print(f'  {label:18s}: right base at {np.round(T[:3,3],4)} m, '
              f'|t| = {np.linalg.norm(T[:3,3]):.4f} m, yaw {yaw:+.2f} deg')
    if X_dep['left'] is not None and X_dep['right'] is not None:
        A = np.linalg.inv(X['left']) @ X['right']
        B = np.linalg.inv(X_dep['left']) @ X_dep['right']
        print(f'\n  the two methods disagree about the base-to-base pose by '
              f'{np.linalg.norm(A[:3,3]-B[:3,3])*1000:.1f} mm, '
              f'{rot_angle_deg(A[:3,:3], B[:3,:3]):.3f} deg')


if __name__ == '__main__':
    main()
