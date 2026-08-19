# FoundationStereo depth for hand-to-eye calibration

Recompute a captured ZED calibration sequence's depth with FoundationStereo and
solve the extrinsics twice — once with ZED SDK depth, once with FS depth — so
the two solutions can be compared as a consistency metric.

## Workflow

```bash
# 1. Capture — humble env, records left + right views + ZED depth
cd ~/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects
python capture_poses_and_images_for_calibration_right.py --camera zed --seq-name zed_calib_007

# 2. FoundationStereo depth — foundation_stereo conda env (~7 s/frame on the A6000)
cd zsy-testmycode
~/miniforge3/envs/foundation_stereo/bin/python fs_depth_batch.py \
    --calib-seq-name zed_calib_007 --side right

# 3a. Solve with ZED SDK depth (baseline metric)
python calculate_base_to_cam_nonlinear_opt.py --camera zed --side right \
    --calib-seq-name zed_calib_007 --use-depth-translation

# 3b. Solve with FoundationStereo depth (second metric)
python calculate_base_to_cam_nonlinear_opt.py --camera zed --side right \
    --calib-seq-name zed_calib_007 --use-depth-translation \
    --rgbd-file right_calibration_rgbd_fs.npz
```

Steps 1, 3a, 3b run in the usual humble pixi env; only step 2 uses the
`foundation_stereo` conda env (called by absolute interpreter path, no
activation needed).

## What each step does

**Step 1 — capture.** Records robot poses plus the ZED stereo pair per frame
into `captured_calibration_data/<seq>/`. The FS path needs the raw right view:
the ZED capture saves `color` = VIEW.LEFT and `color_right` = VIEW.RIGHT since
the 2026-08-18 edit of the capture scripts. **Older captures have no
`color_right` and cannot be reprocessed with FS.**

**Step 2 — `fs_depth_batch.py`.** Runs FoundationStereo on each SDK-rectified
stereo pair and writes `{side}_calibration_rgbd_fs.npz` next to the original
`{side}_calibration_rgbd.npz`, in the same format (`color` (N,H,W,3) BGR,
`depth` (N,H,W) uint16 mm with 0 = invalid, `disparity_offset_px`). fx and
baseline are read from `crisp_py/zed_capture/zed_intrinsics_{W}x{H}.npz` for
the capture's exact resolution — there is deliberately no rescaling fallback
(ZED rectified fx is per-resolution); a new resolution needs its own
intrinsics npz or explicit `--fx`/`--baseline`.

**Step 3 — solver.** Identical solve twice; `--rgbd-file` (a filename relative
to the sequence directory) swaps the depth source. When `--rgbd-file` is given
without `--result-tag`, a tag is derived automatically
(`right_calibration_rgbd_fs.npz` → `fs`), so the ZED-depth and FS-depth solves
write separate result files — compare their extrinsics / residuals.

## Joint two-arm solve and dual pick with FS depth

Once the free per-arm FS solves exist (step 3b, run for **both** sides), the
joint solver and the dual pick each take an `--fs` flag:

```bash
# joint solve: FS depth stacks + FS free solves -> *_fs outputs
pixi run -e humble python solve_joint_extrinsic.py \
    --calib-seq-name zed_calib_fs_001 --fs --disparity-offset-px 5.9

# dual pick: consumes results/zed_calib_fs_001/joint_extrinsic_depth_translation_fs.npz
cd ~/move_some_robots/crisp_env/crisp_py
pixi run -e humble python dual_green_pick_zed_joint.py --fs --disparity-offset-px 5.9 --dry-run
```

`solve_joint_extrinsic.py --fs` builds the pairs from
`{side}_calibration_rgbd_fs.npz`, reads
`base2cam_transform_{side}_nonlinear_opt_depth_translation_fs.npz` for
`--t-lr solved`, and suffixes every output `_fs`
(`joint_extrinsic_depth_translation_fs.npz`,
`base2cam_transform_{side}_joint_depth_translation_fs.npz`,
`summary_joint_depth_translation_fs.txt`). It warns if a free solve was
stamped with a different disparity offset than the joint run applies.

`dual_green_pick_zed_joint.py --fs` makes the whole run FS-based — both
factors of the grasp targets (targets = live depth × extrinsic):

- **extrinsic**: loads the FS joint file above instead of the SDK-depth one;
- **live depth**: replaces SDK `MEASURE.DEPTH` with FoundationStereo run on
  the live left+right rectified pair. The pick script (humble env) grabs
  `VIEW.LEFT` + `VIEW.RIGHT`, writes them to
  `/tmp/dual_green_pick_zed_debug/fs_{left,right}.png`, and subprocesses
  `fs_depth_single.py` in the `foundation_stereo` conda env (~10 s per pick
  on the A6000, model load included; torch and rclpy share no env). The raw
  FS depth comes back via `fs_depth_mm_raw.npy` and the live disparity-offset
  correction is applied to it in the humble env, exactly as
  `capture_rgbd_native()` does for SDK depth.

The stored-vs-live disparity-offset guard applies unchanged, so pass
`--disparity-offset-px` matching the solve when the json value differs.
`--from-capture` replays saved SDK depth, so `--fs` then only selects the
extrinsic. To run the FS extrinsic with SDK live depth (legitimate: both
depth sources share the rectified frame and the offset), pass
`--joint-npz <the fs npz>` without `--fs`.

`fs_depth_single.py` is the one-pair counterpart of `fs_depth_batch.py`
(same inference core, kept in sync; same raw-depth convention). Validated
2026-08-18: bit-identical to the batch output on the same pair, and an
end-to-end `--fs --dry-run` put all three grasp dots on the object.

Reference numbers (zed_calib_fs_001, solved 2026-08-18 at 5.9 px): joint FS
residual **3.67 mm** mean over 143 pairs (left 4.11 / right 3.24), vs
**9.14 mm** for the SDK-depth joint solve on the same capture; joint-vs-free
agreement 1.8 mm (left) / 0.6 mm (right).

## Frame and bias notes

- FS depth is in the **same frame as ZED SDK depth**: the left rectified
  camera frame (X right, Y down, Z forward). Both consume the identical
  SDK-rectified pair and unproject with the same rectified K — nothing to
  align. Verified 2026-08-17 on `zed_snapshot_20260817_163340`: median
  disparity difference FS vs SDK = **+0.06 px**.
- FS depth **inherits this camera's constant disparity offset** (the lens-yaw
  fault tracked in `zed_depth_correction.json`) because the offset lives in
  the rectified images, not in the SDK matcher. `fs_depth_batch.py` therefore
  writes `disparity_offset_px = 0.0` (raw, uncorrected) and the solver applies
  its usual correction to both depth sources alike. The offset drifts between
  sessions (~12 px → ~6 px between 2026-08-11 and 08-14), so refit it per
  capture rather than trusting a stored value.

## Useful `fs_depth_batch.py` options

| Flag | Default | Purpose |
|---|---|---|
| `--side {left,right}` | `right` | which arm's rgbd npz to reprocess |
| `--fs-root` | `~/move_some_robots/FoundationStereo` | FoundationStereo checkout |
| `--ckpt` | auto | model checkpoint override |
| `--scale` | `1.0` | input downscale (keep 1.0 for calibration accuracy) |
| `--valid-iters` | `32` | refinement iterations |
| `--fx`, `--baseline` | from intrinsics npz | manual override for unlisted resolutions |
| `--no-vis` | off | skip debug visualizations |

FoundationStereo itself lives in `~/move_some_robots/FoundationStereo`
(see its `readme.md` for env setup; standalone single-pair runs use
`run_headless.py` there).
