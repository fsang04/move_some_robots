# Fast-FoundationStereo live depth (README_FFS)

Written in ASD-STE100 Simplified Technical English.
Date: 2026-08-20. Model: Fast-FoundationStereo (NVlabs, CVPR 2026),
checkpoint 23-36-37, 8 refinement iterations.

## Summary

The live ZED tracker can now compute depth with Fast-FoundationStereo (FFS)
in place of the ZED SDK matcher. FFS runs as a TensorRT engine inside the
humble pixi environment. It does not use torch. One flag selects the source:

    # SDK depth (the default; the old behavior, unchanged):
    pixi run -e humble python dlo_tracking_live.py --source zed --segmenter armdiff

    # Fast-FoundationStereo depth:
    pixi run -e humble python dlo_tracking_live.py --source zed --segmenter armdiff \
        --depth_source ffs

Why this matters: on the calibration data, FFS depth has the same accuracy as
offline FoundationStereo (3.17 mm mean arm residual, identical to FS), but it
is ~50x faster (0.14 s against 6-7 s for each frame). SDK depth on the same
data gives 9.66 mm.

## What was added

| File | What it is |
|---|---|
| `realtime/ffs_trt.py` | NEW. The TensorRT runner. numpy+cv2 only at import time; tensorrt and cuda-python load lazily inside the class. It converts one rectified BGR pair to a RAW depth map in mm. |
| `~/move_some_robots/ffs_engines/` | NEW folder. The engines, the export/build scripts, the validation script, the logs, and the pre-change file backups. |
| `~/move_some_robots/Fast-FoundationStereo/` | NEW clone of the NVlabs repository, with all checkpoints in `weights/weights/`. |
| conda env `ffs_export` | NEW, temporary. python 3.12 + torch 2.6 + TensorRT. Used ONE time, for the ONNX export and the engine builds. The live code never uses it. You can delete it. |

## What was changed

| File | Change |
|---|---|
| `realtime/frame_source.py` | `ZedSource` has two new parameters: `depth_source` ('zed' or 'ffs') and `ffs_engine` (path). In ffs mode: the SDK opens with `DEPTH_MODE.NONE`, the capture thread retrieves `VIEW.LEFT` + `VIEW.RIGHT`, the engine computes disparity, and depth = fx·baseline/disparity. The SAME `DepthCorrector` is then applied. `ReplaySource` now forwards the recorded provenance stamps (see below). |
| `dlo_tracking_live.py` | Two new flags: `--depth_source {zed,ffs}` (default `zed`) and `--ffs_engine` (default: the 736x1280 engine). `ChunkRecorder.save` now stamps `disparity_scale` and `depth_source` next to `disparity_offset_px` (before, only the offset was stamped). |
| humble pixi env | `tensorrt-cu12==10.16.1.11` and `cuda-python` installed with pip. See "Environment rules" below. |

With `--depth_source zed` (the default), the old code path is unchanged.

## The engines

All engines: checkpoint 23-36-37, 8 iterations, max_disp 192, fp16, TensorRT
10.16.1.11, built on the RTX A6000. Accuracy is measured against offline
FoundationStereo (hiera, 32 iters) on the 72 HD2K pairs of zed_calib_fs_002,
right side, both RAW. Speed is measured at live HD720 input, warm.

| Engine | Median disparity agreement with FS | Speed (HD720) | Rate | Nearest valid depth at HD720 |
|---|---|---|---|---|
| `ffs_23-36-37_it8_736x1280.engine` (DEFAULT) | -0.09 px | 135 ms | 7.4 Hz | 0.43 m |
| `ffs_23-36-37_it8_576x960.engine` | -0.22 px | 84 ms | 11.9 Hz | 0.33 m |
| `ffs_23-36-37_it8_384x640.engine` | -0.28 px | 38 ms | 26.4 Hz | 0.22 m |

Disparity agreement is at the 2208 px reference width. The p95 depth
difference against FS is ~60 mm for all three engines; this lives at object
edges and occlusions, not in the systematic geometry. The first pair after
start costs ~2x (warmup). To select a non-default engine:

    --depth_source ffs --ffs_engine ~/move_some_robots/ffs_engines/ffs_23-36-37_it8_576x960.engine

## Validation (2026-08-20)

The arm-floor test: solve the hand-eye calibration of zed_calib_fs_002 from
each depth source with `calculate_base_to_cam_nonlinear_opt.py
--use-depth-translation --no-enhance --rot-weight 573`. Lower is better.

| Depth source | Right arm (70 frames) | Left arm (65 frames) |
|---|---|---|
| ZED SDK NEURAL_PLUS | 9.66 / 3.70 / 18.06 mm | 7.25 / 4.11 / 17.83 mm |
| Offline FoundationStereo | 3.17 / 1.03 / 4.94 mm | 3.06 / 1.68 / 7.22 mm |
| FFS TRT 736x1280 | 3.17 / 1.00 / 5.36 mm | 3.07 / 1.43 / 6.35 mm |

Values are translation error mean / std / max. The FFS npz files
(`{side}_calibration_rgbd_ffs.npz`) and the solve outputs (suffix `_ffs`) are
next to the FS ones. To validate a new engine or a new sequence:

    .pixi/envs/humble/bin/python ~/move_some_robots/ffs_engines/validate_ffs_vs_fs.py \
        --engine <path.engine> --side right [--write-npz]

## The a/d correction rule

APPLY the disparity correction (a=1.012, d=5.7 from
zed_capture/zed_depth_correction.json) to FFS depth, exactly as to SDK depth.
The fault is a lens-yaw error in the RECTIFIED IMAGES. Every matcher that
reads those images inherits it. The code does this for you:

- Live: `ZedSource` applies its `DepthCorrector` to FFS depth in the capture
  thread. The d value is rescaled by the fx ratio for the resolution in use.
- Offline: `validate_ffs_vs_fs.py --write-npz` writes RAW depth with
  `disparity_offset_px=0.0`. The solver reads the stamp and applies the full
  correction itself. This is the same convention as `fs_depth_batch.py`.

Do not read a 0.0 stamp as "no correction needed". It means "raw, not yet
corrected". Set the correction to zero everywhere only after a ZED
recalibration.

## Provenance stamps

Recorded chunks (`--record`) now stamp three keys in rgbd.npz:
`disparity_offset_px`, `disparity_scale`, `depth_source`. The double-correction
guard (`zed_depth_config.corrector_for`) reads the first two. Before this
change, only the offset was stamped; the scale provenance gap is closed.
`ReplaySource` reads the stamps of the chunk it plays and forwards them, so a
re-record of a corrected chunk stays marked as corrected.

## Environment rules

1. The humble pixi env now contains `tensorrt-cu12==10.16.1.11` and
   `cuda-python`, installed with `python -m pip` (NOT through pixi.toml).
   A pixi environment REBUILD REMOVES THEM, exactly as it removes the manual
   pyzed wheel. After a rebuild, run:

       .pixi/envs/humble/bin/python -m pip install tensorrt-cu12==10.16.1.11 "cuda-python<12.7"

2. An engine is tied to ONE GPU model and ONE TensorRT version. The engines
   here are for the RTX A6000 + TensorRT 10.16.1.11. After a GPU change or a
   TensorRT change, rebuild:

       bash ~/move_some_robots/ffs_engines/export_and_build.sh

   Do NOT install TensorRT 11: it removed the fp16 builder flag, and the
   engines will not load under it. Keep 10.16.1.11 in the humble env and in
   any build env.

3. The one-time export env (`conda ffs_export`) is disposable:

       conda env remove -n ffs_export

## Limits and notes

- max_disp is 192 engine pixels. Objects nearer than the "nearest valid
  depth" column above get no valid disparity. The tracking workspace
  (0.5-2.0 m) is inside the valid range for all three engines.
- In ffs mode the flags `--zed_depth_mode`, `--zed_confidence`,
  `--zed_stabilization` do nothing. They shape the SDK matcher only.
- FFS depth is dense. The SDK confidence holes do not exist. The tracker's
  own gates (`--max_depth`, `--z_range`) still apply.
- The full-resolution engine caps the capture rate at ~7 Hz. If the tracker
  needs more rate than accuracy, pass the 576x960 or 384x640 engine.
- SAM2 (`--segmenter sam2`) still does not run in the humble env. The ffs
  mode shares the GPU with nothing else in the armdiff/pcdiff paths.
- The 135 ms includes ~25 ms of host-side preprocessing (resize, normalize).
  A future optimization can move it to the GPU.

## Rollback

The pre-change files are in
`~/move_some_robots/ffs_engines/backup_pre_ffs/`
(`frame_source.py`, `dlo_tracking_live.py`). Copy them back to undo the code
change. To remove the runtime from the humble env:

    .pixi/envs/humble/bin/python -m pip uninstall tensorrt-cu12 cuda-python
