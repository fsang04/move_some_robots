# ZED depth correction: from constant disparity offset to offset + scale

This documents the depth-correction model for ZED SN22456: what the original
constant disparity offset was, how it is applied, how its error propagated into
hand-eye calibration, and why it was upgraded (2026-08-19) to a two-parameter
offset + scale model.

Single source of truth: [`zed_depth_correction.json`](zed_depth_correction.json)
(read by [`zed_depth_config.py`](zed_depth_config.py), imported by both the
capture path and the calibration solvers). Current values: **a = 1.012,
d = 5.7 px**.

---

## 1. Why depth needs correcting at all

Stereo depth is computed from disparity:

```
z = fx · B / disp        fx = 1414.575 px (HD2K rectified), B = 0.120001 m
                         → fx·B ≈ 169.75 px·m
```

On this camera the reported disparity carries a systematic error. The cause of
the constant part is a small relative yaw between the two lenses that the
factory calibration no longer captures (5.7 px / 1414.6 px ≈ 0.23°). The SDK's
self-calibration would normally re-track this at every `open()`, but it always
fails on this rig (the black backdrop has no texture), so the error is never
removed and must be corrected in software. Per-lens intrinsics are **not** the
problem — factory K was verified against a checkerboard to 0.257 px rms.

## 2. The original model: constant disparity offset

```
disp_true = disp + d          ⇔          z_true = fx·B / (fx·B/z + d)
```

One number, `d` (px), applied to depth everywhere:

- **Capture**: `zed_camera.capture_rgbd_native()` corrects SDK depth at capture
  time via `correct_depth_disparity_offset()`.
- **Solvers**: `calculate_base_to_cam_nonlinear_opt.py` /
  `solve_joint_extrinsic.py` correct the depth they read from rgbd npz stacks
  (FoundationStereo npzs are stored **raw**, stamped `disparity_offset_px=0.0`,
  and corrected at consumption).
- **Picks**: live capture is corrected the same way; loaded extrinsics carry a
  provenance stamp and are hard-refused if solved under a different correction
  (double-correction guard).

### What one pixel of disparity is worth

Along-ray depth sensitivity is `Δz ≈ z²/(fx·B) · Δd`:

| range | 1 px of disparity |
|---|---|
| 1.00 m | ≈ 5.9 mm |
| 1.25 m | ≈ 9.2 mm |
| 1.50 m | ≈ 13.3 mm |

So a 1 px error in `d` is a ~9 mm depth error at working range — larger than
the entire hand-eye residual budget. The current full correction (a = 1.012,
d = 5.7) moves depth by **−43.6 mm @ 1.0 m, −64.0 mm @ 1.25 m,
−88.1 mm @ 1.5 m**.

### How `d` is measured

Fit against AprilTag PnP range as ground truth: for each calibration frame,
implied offset `d_i = fx·B·(1/z_tag − 1/z_raw)`. A good fit has std < 1 px and
no trend against range. (`compare_apriltag_t_vs_depth_t.py` computes the
per-frame terms; the tag size enters through `z_tag`, so `d` and `TAG_SIZE_M`
are coupled — the tag was re-measured 96.0 → 95.5 → **95.0 mm**, and every
0.5 mm of tag size moves the fitted offset by ~0.7 px at 1.25 m.)

## 3. How the constant model failed

Two symptoms accumulated over a week of captures (96.0 mm-tag era values):

**(1) The "constant" kept drifting between captures:**

| capture | date | fitted d (left / right) | trend vs range |
|---|---|---|---|
| zed_calib_003 | 08-11 | 12.45 / 11.55 px | — |
| zed_calib_004 | 08-14 | ~5.0 px | — |
| zed_calib_005 | 08-17 | 7.13±1.53 / 5.43±1.16 px | — |
| zed_calib_007 | 08-17 | 6.13±1.20 / 5.68±1.03 px | −1.34 / −0.91 px/m |
| zed_calib_fs_001 | 08-18 | 5.59±0.95 px | −0.77 px/m |

The 003→004 jump was a real physical change in the camera. But the residual
spread *after* that (5.4–7.1 px, plus the consistent negative px/m trends and
the left/right disagreement) was not drift at all — see below.

**(2) Range-correlated residual envelope in solves.** The per-frame translation
error of a hand-eye solve was largest exactly in the frames where the arm moved
toward/away from the camera — a periodic envelope tracking tag range. A
constant `d` that is correct at 1.25 m is wrong at 1.0 m and 1.5 m, and since
capture trajectories sweep range, the leftover error shows up as a
range-shaped residual instead of noise.

**The diagnosis:** the true error has a *multiplicative* component. If
`disp_true = a·disp + d` with a ≠ 1, then fitting a constant-only model
returns an effective constant `d_eff(z) = d + (a−1)·fx·B/z` — it depends on
the capture's range window. With a = 1.012, d = 5.7:

| range window centre | constant-equivalent offset |
|---|---|
| 1.0 m | 7.7 px |
| 1.25 m | 7.4 px |
| 1.5 m | 7.0 px |

This reproduces the "drift" (each capture sampled a different range window),
the negative px/m trends, and the left/right disagreement (the left arm's
capture volume sat nearer than the right's, so left always fitted a larger
constant — e.g. 7.0 vs 6.3 at the time the deployed average was 6.7).

## 4. The offset + scale model

```
disp_true = a · disp + d      ⇔      z_true = fx·B / (a · fx·B/z + d)
```

- `d` (px): the lens-yaw slide, same meaning as before.
- `a` (dimensionless): absorbs percentage errors in the `fx·B` product
  (baseline / rectified-focal mismatch) — a 1.2% stretch that no constant can
  represent.

### The fit that settled it

Fitted PnP-vs-**raw-FoundationStereo**-depth on `zed_calib_fs_002` with the
caliper-verified 95.0 mm tag. FS depth is 4–6× quieter than SDK depth at the
tag, which is what made `a` resolvable at all:

| arm | n | range | constant model | scale model | residual |
|---|---|---|---|---|---|
| left | 67 | 0.93–1.35 m | d=+7.53±0.43, trend −1.76 px/m | **a=1.0125, d=+5.72** | 0.36 px |
| right | 72 | 1.01–1.53 m | d=+7.21±0.36, trend −1.30 px/m | **a=1.0118, d=+5.70** | 0.28 px |
| pooled | 139 | — | d=+7.37±0.42 | a=1.0132, d=+5.57 | 0.32 px |

The two arms are fully independent captures and agree to **0.0007 on `a` and
0.02 px on `d`** — the strongest confirmation this data can give that the 1.2%
stretch is real. Under the constant model the same data wants 7.4 px *with a
trend*; under the scale model the trend is gone and the residual is pure
noise. Deployed values: **a = 1.012, d = 5.7** (json, 2026-08-19).

### Results under the new model (zed_calib_fs_002, FS depth)

- Free per-arm solves: left **3.06 mm**, right **3.17 mm** mean translation
  residual.
- Joint two-arm solve: **3.12 mm** mean (left 3.02 / right 3.22 — balanced),
  rot 0.64°.
- Joint vs free extrinsic agreement: **0.1 mm / 0.007°** per arm (was ~1.8 mm
  on fs_001 under the constant model).
- The range-correlated residual envelope is gone.

(For reference: FS depth also broke the old ~6.7 mm right-arm SDK-depth floor
— fs_001 right solved at 1.38 mm with FS vs 8.27 mm with SDK depth, same
detections, same offset. The floor was SDK matcher noise, not rig geometry.)

### Known residual ambiguity

`a` is degenerate with tag size in the PnP fit: a=1.012 @ 95.0 mm tag is
indistinguishable from a=1.000 @ 96.15 mm. The tag was caliper-measured at
95.0 mm, so a=1.012 is adopted; an FK-based scale check was inconclusive
(per-arm constant FK-reference errors alias into the slope at ~1.7% per
10 mm of `gripper2tag` error).

## 5. Rules for using the correction

- **CLI flags** are now `--d` (offset, px) and `--a` (scale) on the solvers,
  `compare_apriltag_t_vs_depth_t.py`, `capture_zed_sam_mask.py`, and both pick
  scripts. Old spellings `--disparity-offset-px` / `--disparity-scale` still
  work as aliases. Omit both to use the json defaults; `--d 0 --a 1` = raw
  depth.
- **Env overrides** for one-shot experiments: `$ZED_DEPTH_OFFSET_PX`,
  `$ZED_DEPTH_SCALE` (and `$ZED_DEPTH_CORRECTION_JSON` to point at a different
  file).
- **Resolution rescaling**: `d` scales with rectified **fx** (not width — the
  ZED rectifies per resolution: fx is 1414.575 @ 2208 px but 693.82 @ 1280 px,
  where width-scaling would predict 820.04, an 18% error). `a` is
  dimensionless and must **never** be rescaled.
- **Provenance stamps + guards**: every solve npz stamps the
  (`disparity_offset_px`, `disparity_scale`) pair it was solved under; pick
  scripts compare the full pair against the live correction and hard-refuse a
  mismatch. Npzs from the constant era (stamped scale-less, e.g. 5.9 or
  6.7 px) are refused on purpose — re-solve them. An absent scale key reads as
  1.0 (backward compatibility for raw FS npzs).
- **One correction end-to-end for joint extrinsics**: per-side offsets would
  inject their difference (~6 mm) straight into T_LR.
- **Refit from every new calibration capture** before trusting a solve — the
  camera has physically shifted once already (08-11 → 08-14) and
  self-calibration cannot catch it on this rig.

## 6. Relationship to ZED_Calibration

Running Stereolabs' recalibration tool would fold the same physics into the
`.conf` file: `d` is relative yaw (exactly what it estimates) and `a` is
plausibly baseline error. If it is ever run: rectification changes, **all
extrinsics and FS npzs are invalidated**, and (a, d) must be **re-fitted**
(expected near 1.0 / 0.0 — but verify, don't assume; the screen-based tool is
typically less accurate than this fit's 0.3 px residual). Never leave the old
(a, d) applied on top of a new calibration — it would push depth wrong in the
other direction. `zed_depth_config.check_camera()` compares `conf_md5` and
warns if the conf file changed.
