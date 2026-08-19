# ZED capture & hand-eye calibration

Everything for the ZED stereo camera on the dual-FR3 cloth rig: single-shot RGB-D
capture for the cloth pipeline, and hand-eye (base→camera) calibration.

Camera: **original ZED, SN 22456**, firmware 1523, ZED SDK **4.2.5**.

---

## Contents

| File | Purpose |
|---|---|
| `capture_zed_sam_mask.py` | Single-shot RGB-D capture + optional interactive SAM2 mask. ZED counterpart of `capture_azure_sam_mask.py`. **Part A only.** |
| `zed_camera.py` | Reusable ZED module for Part A (config dataclasses, viewer-conf reader, capture, intrinsics, depth colormap). No import-time side effects. **Part B does not use this** — see [Camera configuration](#camera-configuration--which-file-to-edit). |
| `export_zed_intrinsics.py` | Writes the `K`/`dist` npz the calibration pipeline needs. |
| `zed_intrinsics_2208x1242.npz` | Exported rectified intrinsics at HD2K. Auto-discovered and rescaled by `apriltag_image.py`. |
| `zed_depth_correction.json` | The depth disparity-offset value. One source of truth for Part A **and** Part B. See [Depth disparity offset](#depth-disparity-offset). |
| `zed_depth_config.py` | Loader + the correction maths. numpy only, no `pyzed`, so the solvers can import it. |
| `real_captures/` | Part A capture output. |

Removed as unused: `record_zed_calib_poses.py` and `solve_base2cam_apriltag.py` (an
earlier standalone calibration attempt, superseded by the Part B pipeline below and
referenced by nothing), and `command.txt` (folded into this README, and stale — it
predated the depth correction).

Files changed **outside** this folder, under `hand_to_eye_calibration/roahm-deformable-objects/`:

| File | Change |
|---|---|
| `zed_calib_rgbd.py` | **new** — shared ZED RGB-D capture for the calibration scripts. Records **raw** depth on purpose; the solvers correct it. |
| `capture_poses_and_images_for_calibration{,_left,_right}.py` | ZED branch now records **depth**; saves 3-channel BGR; depth saved for any camera; failed-grab index misalignment fixed; RGB-D npz now records `disparity_offset_px` so the depth correction can never be applied twice. |
| `apriltag_image.py` | ZED intrinsics **loaded and rescaled** instead of a wrong hardcoded constant; cv2.aruco fallback detector. |
| `apriltag_backend.py` | **new** — cv2.aruco fallback for the `apriltag` package. |
| `zsy-testmycode/calculate_base_to_cam_nonlinear_opt.py` | Azure-only guard on `--use-depth-translation` removed; depth intrinsics camera-aware (Azure unchanged); **applies the ZED depth disparity-offset correction**; `--disparity-offset-px`; result npz records the offset it was solved with; the two duplicated `savez` argument lists merged into one payload. |
| `zsy-testmycode/compare_apriltag_t_vs_depth_t.py` | Same fixes — it also hardcoded Azure intrinsics for depth unprojection, so a `--camera zed` verification would have been silently wrong. Azure-only guard removed; correction applied; `--disparity-offset-px`. |
| `zsy-testmycode/debug_local_matrix_translation_table.py` | Same fixes. It used to **refuse** `--use-depth-translation` for a ZED, and it back-projected with Azure intrinsics. Now camera-aware, correction applied, `--disparity-offset-px`. |

---

## Environments

Two pixi envs, because no single one has everything:

| Env | Has | Use for |
|---|---|---|
| `sam2` | pyzed, torch+CUDA, sam2, cv2 | RGB-D capture with SAM masking (Part A) |
| `humble` | pyzed, crisp_py, rclpy, cv2 | anything touching the robot (Part B) |

`pyzed` was installed into both by unzipping `pyzed-4.2-cp311-cp311-linux_x86_64.whl` into
site-packages (neither env has a usable `pip`; the wheel is just a `.so` + `__init__.py`).
This is outside pixi's management, so a future `pixi install` may drop it — re-unzip if
`import pyzed` starts failing.

**A USB ZED allows only ONE process at a time.** Close the Depth Viewer before any capture:

```bash
pgrep -af 'ZED_Depth_Viewer|ZED_Explorer'    # must be empty
```

---

# Part A — RGB-D capture

Run from the `crisp_py` root:

```bash
cd /home/roahmlab/move_some_robots/crisp_env/crisp_py

# capture + interactive SAM mask
pixi run -e sam2 python zed_capture/capture_zed_sam_mask.py \
    --sam-checkpoint sam2/checkpoints/sam2.1_hiera_large.pt

# capture only (no torch needed)
pixi run -e sam2 python zed_capture/capture_zed_sam_mask.py

# inspect without opening the camera
pixi run -e sam2 python zed_capture/capture_zed_sam_mask.py --list-config
pixi run -e sam2 python zed_capture/capture_zed_sam_mask.py --show-viewer-conf
```

Mask controls: **L-click** = target, **R-click** = background, **Enter** = accept, **r** = reset, **q** = quit.

### Defaults

Mirror the ZED Depth Viewer setup: `HD2K @ 15 fps · NEURAL_PLUS · depth 0.20–20.0 m ·
stabilization 10 · fill off · saturated-removal on · confidence 47 · texture 100`,
**self-calibration disabled**. Every knob is a CLI flag (`--help`, grouped by viewer panel).

### Output — `real_captures/run_<timestamp>/`

| File | dtype / shape | Notes |
|---|---|---|
| `rgb.png` | uint8 BGR 360×640 | written BGR so PIL `.convert("RGB")` reads back correctly |
| `depth_m.npy` | float32 (360,640) | metres, `0` = invalid |
| `depth_mm.npy` | uint16 (360,640) | mm, `0` = invalid |
| `mask.png` | uint8 (360,640) | 255 = target (only with `--sam-checkpoint`) |
| `overlay.png` | uint8 BGR | mask overlay |
| `rgb_full.png`, `depth_m_full.npy`, `depth_mm_full.npy` | native 2208×1242 | for 3-D back-projection |
| `depth_vis.png`, `depth_vis_full.png` | uint8 BGR | turbo + labelled colorbar |
| `intrinsics.json` | — | `native` + `scaled` blocks, K matrices, baseline |
| `capture_config.json` | — | every setting actually in effect |

`0 = invalid` everywhere: ZED's native NaN / +inf (TOO_FAR) / −inf (TOO_CLOSE) are
normalised so existing Azure consumers work unchanged.

### Back-projecting to a point cloud

Pair the intrinsics block with the matching depth file — `native` with `depth_m_full.npy`,
`scaled` with `depth_m.npy`. Mixing them scales the cloud by 3.45× and still looks plausible.

```python
import json, cv2, numpy as np
run  = "zed_capture/real_captures/run_YYYYmmdd_HHMMSS"
K    = json.load(open(f"{run}/intrinsics.json"))
depth, intr, img = np.load(f"{run}/depth_m_full.npy"), K["native"], "rgb_full.png"

h, w  = depth.shape
u, v  = np.meshgrid(np.arange(w), np.arange(h))
valid = depth > 0                        # skip this and you get a blob at the origin
z     = depth[valid]
pts   = np.stack([(u[valid]-intr["cx"])*z/intr["fx"],
                  (v[valid]-intr["cy"])*z/intr["fy"], z], 1)   # (N,3) m, camera frame
cols  = cv2.imread(f"{run}/{img}")[..., ::-1][valid]           # (N,3) RGB
```

`disto` is all zeros — the images are already rectified, so **do not** apply distortion.
Frame is `COORDINATE_SYSTEM.IMAGE`: X right, Y down, Z forward.

Alternatively let the SDK build the cloud: `retrieve_measure(mat, sl.MEASURE.XYZRGBA)`
returns metric XYZ directly (invalid = NaN/±inf there, not 0).

---

# Part B — Hand-eye calibration

Produces `T_base→cam` per arm using an AprilTag on the gripper with a **manually measured**
`gripper2tag`, refined with **depth**.

## Summary — current state

Latest run: **`zed_calib_003`**, both arms, solved 2026-08-12.

```
capture   capture_poses_and_images_for_calibration_{left,right}.py --camera zed
             -> zed_calib_rgbd.py opens the camera, records HD2K colour + RAW depth
solve     python calculate_base_to_cam_nonlinear_opt.py --camera zed --side left --calib-seq-name zed_calib_001 --use-depth-translation --no-enhance --rot-weight 573 --save-to-calib-dir
             -> applies the 16.03 px depth correction, then optimises
verify    compare_apriltag_t_vs_depth_t.py    -> expect |t_depth - t_apriltag| ~5 mm
```

**Deploy these two** (`p_cam = T @ p_base`):

```
captured_calibration_data/zed_calib_003/base2cam_transform_left_nonlinear_opt_depth_translation.npz
captured_calibration_data/zed_calib_003/base2cam_transform_right_nonlinear_opt_depth_translation.npz
```

| side | mode | residual | outliers |
|---|---|---|---|
| left | apriltag | 5.38 mm | 0 / 59 |
| **left** | **depth** | **8.14 mm** | **0 / 59** |
| right | apriltag | 9.22 mm | 4 / 56 |
| **right** | **depth** | **15.63 mm** | 7 / 56 |

Four things about this run that are easy to get wrong:

1. **Deploy `depth_translation`, not `apriltag_translation`,** even though its residual
   number is larger. The residual of each file is measured against the data it was solved
   from, so the numbers are not comparable. See [Which file to deploy](#which-file-to-deploy).
2. **The depth needs the 16.03 px correction, in the pipeline as well.** These transforms
   are only valid for corrected depth. Without it the depth reads ~15% too far.
3. **`--rot-weight 573`, not the default `1.0`.** At 1.5 m, 1° of rotation costs 26 mm of
   position, so the default under-weights rotation badly.
4. **The left arm is the better calibration.** Its poses stay nearer, so its tag views are
   closer and better lit. Prefer it when a single camera pose must serve both arms.

Independent check that uses no tag: the two arms are calibrated from separate datasets, so
the base-to-base geometry they imply is a free test. `apriltag` gives 3130.1 mm / 179.229°,
`depth` gives 3126.5 mm / 179.161° — agreement to **3.6 mm and 0.07°**.

Open item: both references carry a systematic error (the tag is −2% against robot FK, the
corrected depth −0.6% against the tag), so which is closer to truth is **not yet settled**.
A tape measure on a flat wall at 1.0 m and 1.8 m would settle it.

## Camera configuration — which file to edit

**Part A and Part B open the camera through different modules, with independent copies of
the settings.** Changing one does not change the other. This is the single most common way
to get an unexpected result.

| what you want to change | Part A (perception capture) | Part B (calibration capture) |
|---|---|---|
| module that opens the camera | `zed_capture/zed_camera.py` | `hand_to_eye_calibration/roahm-deformable-objects/zed_calib_rgbd.py` |
| how to change it | **a CLI flag** on `capture_zed_sam_mask.py` (every knob is one; `--help` groups them by viewer panel), or the dataclass defaults in `zed_camera.py` | **edit the `DEFAULT_*` constants** in `zed_calib_rgbd.py`. `capture_poses_and_images_for_calibration_*.py` calls `open_zed()` with no arguments, so **there is no flag** |
| resolution / fps / depth mode / range | `ZedInitConfig` | `DEFAULT_RESOLUTION`, `fps=15`, `DEFAULT_DEPTH_MODE`, `DEFAULT_DEPTH_MIN_M`, `DEFAULT_DEPTH_MAX_M` |
| confidence / texture / fill / saturated | `ZedRuntimeConfig` | `DEFAULT_CONFIDENCE`, and the literals in `open_zed()` |
| exposure / gain / white balance | `ZedImageConfig` | `DEFAULT_EXPOSURE`, `DEFAULT_GAIN` |
| frame averaging | not used | `DEFAULT_MEDIAN_FRAMES = 5`, `DEFAULT_WARMUP = 30` |

Shared by both, and the right place for anything camera-wide:

| what | file | note |
|---|---|---|
| **depth disparity offset** | `zed_capture/zed_depth_correction.json` | edit once, every consumer follows |
| **intrinsics** | `zed_capture/zed_intrinsics_2208x1242.npz` | regenerate with `export_zed_intrinsics.py`; `apriltag_image.py` finds and rescales it |
| tag id / size | `zsy-testmycode/calculate_base_to_cam_nonlinear_opt.py` | `EXPECTED_TAG_ID = 3`, `TAG_SIZE_M = 0.093` |
| `gripper2tag` | `calculate_base_to_cam.py:487` | hand-measured, `[−0.02, 0, 0.0905]` m |

The two config copies currently agree on everything that affects the result — HD2K, 15 fps,
NEURAL_PLUS, 0.20–20.0 m, confidence 47, texture 100, fill off, saturated-removal on,
self-calibration disabled. Two knobs differ, both harmless:

- `depth_stabilization` — Part A sets 10, Part B leaves the SDK default. Part B averages 5
  frames at a stationary pose instead, which does the same job better.
- `image_enhancement` — Part A sets it explicitly, Part B takes the default (also on).

> **If you re-tune depth quality in the ZED Depth Viewer, the new numbers must be copied
> into BOTH files** to change both paths. There is no shared config for these. Only the
> depth offset and the intrinsics are genuinely shared.

## 0. Physical setup

- Mount the 3D-printed AprilTag mount on the EE — **tag36h11, id 3, 93 mm**
  (hardcoded `EXPECTED_TAG_ID=3`, `TAG_SIZE_M=0.093`)
- **Close the gripper**
- Close ZED_Depth_Viewer (see above)
- Robots powered, brakes released, FCI enabled

## 1. Start the ROS container — Terminal A

```bash
cd /home/roahmlab/move_some_robots/crisp/crisp_controllers_demos
LEFT_ROBOT_IP=192.168.2.2 RIGHT_ROBOT_IP=192.168.2.3 \
  docker compose up launch_dual_franka
```

Leave running. The compose sets `ROS_DOMAIN_ID=100` and `RMW=fastdds`.

> Bash history has the IPs **both ways** (`.2/.3` and `.3/.2`). If left/right come up
> swapped, that's why.

## 2. Verify connectivity — Terminal B

```bash
cd /home/roahmlab/move_some_robots/crisp_env/crisp_py
export ROS_DOMAIN_ID=100          # must match the container
export ROS_LOCALHOST_ONLY=0

pixi run -e humble ros2 topic list | grep -E '^/(left|right)/'
```

Expect `/left/...` and `/right/...`. Nothing means a domain/RMW mismatch, not a robot fault.

Optional safe reset:

```bash
cd hand_to_eye_calibration/roahm-deformable-objects
pixi run -e humble python send_both_arms_home.py
```

## 3. Export ZED intrinsics — once per camera

```bash
cd /home/roahmlab/move_some_robots/crisp_env/crisp_py
pixi run -e humble python zed_capture/export_zed_intrinsics.py \
    --from-capture zed_capture/real_captures/run_20260810_171902 \
    --width 2208 --height 1242
```

**Already done** — `zed_intrinsics_2208x1242.npz` exists, matching the HD2K frames that
`zed_calib_rgbd.py` records. `apriltag_image.py` discovers it automatically (via
`$ZED_INTRINSICS_NPZ`, then `zed_capture/`, then its own directory) and rescales to the
frame resolution when needed. HD2K is exactly 16:9, so 2208×1242, 1280×720 and 640×360
interconvert exactly: `fx=1414.575439, cx=1101.675415, cy=669.754150`.

Redo only if you swap cameras. ZED intrinsics are per-camera **and** per-resolution.

## 4. Capture poses + images + depth

Run **from the calibration directory** so `import apriltag` and `import zed_calib_rgbd` resolve:

```bash
cd /home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects
export ROS_DOMAIN_ID=100

pixi run -e humble python capture_poses_and_images_for_calibration_left.py \
    --camera zed --seq-name zed_calib_004

pixi run -e humble python capture_poses_and_images_for_calibration_right.py \
    --camera zed --seq-name zed_calib_004
```

Each arm draws a figure-eight, pausing every 7th control step (~2 s settle) to grab.
Use the **same `--seq-name`** for both arms, with the camera untouched between them, so
`T_left2right` is meaningful.

Camera settings come from `zed_calib_rgbd.py`'s `DEFAULT_*` constants — these scripts pass
no camera arguments. See [Camera configuration](#camera-configuration--which-file-to-edit).

Watch `[zed] grab: valid_depth=XX%`. Below ~50% on the tag usually means the tag is in the
left-edge stereo dead band or closer than the working minimum.

The depth is saved **raw**, on purpose. The solver corrects it, so the value can be revised
without re-capturing. The npz records `disparity_offset_px = 0.0` to say so.

One viewable depth picture per pose is written to `frames/depth_{side}_image_{i}.png` —
**after** the sweep finishes, not during it, so an empty `frames/` mid-run is normal.

Output — `captured_calibration_data/zed_calib_004/`:

```
frames/calibration_{left,right}_image_{0..N}.png    BGR, 2208x1242
{left,right}_calibration_poses.npz                 [px,py,pz,qx,qy,qz,qw] per pose
{left,right}_calibration_rgbd.npz                  color + depth (uint16 mm, 0=invalid)
```

Depth coverage is good here: the `_left`/`_right` scripts sweep depth with
`y_amp` 0.20 / 0.26 m, so the single-depth degeneracy that limited the earlier Azure
calibration (4 cm tag-depth span) does not apply.

## 5. Solve — with depth

```bash
cd zsy-testmycode

for side in left right; do
  pixi run -e humble python calculate_base_to_cam_nonlinear_opt.py \
      --camera zed --side $side --calib-seq-name zed_calib_003 \
      --use-depth-translation --max-images 70 --no-enhance \
      --rot-weight 573 --save-to-calib-dir
done
```

`--use-depth-translation` = **AprilTag rotation + depth translation**. The tag nails
orientation, but PnP is structurally weak along the viewing ray; depth measures that axis
directly. This is the same fix that took the Azure calibration from ~30 mm radial error to
~4 mm.

The ZED depth disparity offset is applied automatically. Look for this line, and check the
value is the one you expect:

```
[depth-fix] disparity offset +16.03 px applied (fx=1414.58 px, B=120.0 mm, unit=mm): -86.3 mm @ 1.0 m, -186.1 mm @ 1.5 m
[depth-fix] value from .../zed_capture/zed_depth_correction.json
```

`--rot-weight 573` balances the two error terms: at 1.5 m, 1° of rotation costs 26 mm of
position, so rotation deserves far more weight than the default `1.0`. `--no-enhance` skips
the image-enhancement retry ladder — 11 s instead of ~20 min, and it changes nothing when
the tag is already detected in the original frame.

Run **without** `--use-depth-translation` as a baseline — comparing the two localises any
problem to depth quality vs `gripper2tag`:

```bash
pixi run -e humble python calculate_base_to_cam_nonlinear_opt.py \
    --camera zed --side left --calib-seq-name zed_calib_003 \
    --max-images 70 --no-enhance --rot-weight 573
```

To see what the depth offset is worth, solve once with the correction disabled:

```bash
... --use-depth-translation --disparity-offset-px 0
```

Drop bad frames and re-solve without re-capturing:

```bash
... --exclude-images 3,17,42
```

Other knobs: `--depth-patch-radius 5`, `--robust-loss huber`, `--f-scale 10.0`,
`--trans-weight 1000.0`, `--max-nfev 200`.

## 6. The final extrinsic

```
zsy-testmycode/results/zed_calib_003/base2cam_transform_{side}_nonlinear_opt_{mode}.npz
captured_calibration_data/zed_calib_003/base2cam_transform_{side}_nonlinear_opt_{mode}.npz
```

(the second only with `--save-to-calib-dir`)

Convention: **`p_cam = T @ p_base`**.

### Which file to deploy

**Use `depth_translation`, and pair it with corrected depth.** The two modes are not
interchangeable, because each absorbs the bias of the data it was solved from:

| pairing | residual | note |
|---|---|---|
| `apriltag` + AprilTag data | 9.22 mm | the calibration test, **not** what the pipeline runs |
| **`depth` + corrected depth** | **15.63 mm** | self-consistent. **Deploy this.** |
| `apriltag` + corrected depth | 16.93 mm | mixes two distance notions; leaves a **9 mm constant bias** |

The 9.22 mm of the `apriltag` file is not the accuracy you get from depth data. The
perception pipeline feeds **depth** into the transform, so the transform must have been
solved from depth.

The two transforms differ by an almost pure translation — 8.8 mm, with only 0.0145° of
rotation, flat across 0.8–1.9 m. That 9 mm is a **bias**, not noise: averaging a thousand
cloth pixels removes the ±10 mm per-pixel noise but leaves the 9 mm untouched.

Each result npz records the `disparity_offset_px` it was solved with. Read it back before
you deploy:

```python
float(np.load(path)["disparity_offset_px"])   # 16.03 for depth, 0.0 for apriltag
```

`zed_calib_003`, both arms, `--rot-weight 573`:

| side | mode | all frames | inliers | outliers |
|---|---|---|---|---|
| left | apriltag | 5.38 mm | — | 0 / 59 |
| **left** | **depth** | **8.14 mm** | — | **0 / 59** |
| right | apriltag | 9.22 mm | 9.07 mm | 4 / 56 |
| **right** | **depth** | **15.63 mm** | **13.95 mm** | 7 / 56 |

The left arm is markedly better on both. Its poses reach less far, so its tag views are
nearer and better lit.

**Cross-check that uses no tag:** the two arms are calibrated from completely separate
datasets, so the base-to-base geometry they imply is an independent test. `apriltag` gives
3130.1 mm and 179.229°; `depth` gives 3126.5 mm and 179.161°. The two methods agree to
**3.6 mm and 0.07°**.

## 7. Verify

```bash
pixi run -e humble python compare_apriltag_t_vs_depth_t.py --camera zed \
    --calib-seq-name zed_calib_003 --side left --num-images 30
```

`--num-images` defaults to only 5; use more so a trend is visible. `--image-indices 0,5,10,...`
picks specific frames instead.

This tool applies the depth correction too. Expect `|t_depth − t_apriltag|` around **5 mm**.
To see the uncorrected fault, add `--disparity-offset-px 0` — that gives ~173 mm:

```bash
... --disparity-offset-px 0
```

Check whether the tag-vs-depth gap **grows with range**. The tag's scale comes from its
printed size, independent of the stereo baseline — so a slope there is the only available
test for whether the factory rectification has gone stale (see *Self-calibration* below).
A residual slope after the correction means the offset needs re-measuring.

Verified round-trip after the intrinsics fix: pixel → 3D → pixel returns the original
pixel exactly for both cameras, and the Azure numbers are unchanged.

## 8. Shut down

`Ctrl-C` in Terminal A, then `docker compose down`.

---

## ZED reference (all measured on SN 22456)

### Intrinsics — use the RECTIFIED set

The script retrieves `VIEW.LEFT` + `MEASURE.DEPTH`, both **rectified**, so the rectified
intrinsics are the correct ones:

| | HD2K 2208×1242 | 1280×720 | 640×360 |
|---|---|---|---|
| fx = fy | 1414.575439 | 820.043733 | 410.021867 |
| cx | 1101.675415 | 638.652415 | 319.326207 |
| cy | 669.754150 | 388.263276 | 194.131638 |

`disto` = zeros. Baseline 0.120001 m.

**Do not use `/usr/local/zed/settings/SN22456.conf`.** That file is raw/unrectified
(`[LEFT_CAM_2K]` fx=1398.03, cx=1147.64, cy=629.238, k1=−0.170559). Rectification shifts
cx by −46 px, cy by +40 px and fx by +1.18%; substituting it costs ~39 mm lateral and
~34 mm vertical error at 1.2 m.

Rectification is *computed* at every `open()` from the whole factory calibration (both
eyes' intrinsics, distortion, and the stereo extrinsics), producing one shared virtual
camera applied to both eyes with `T = [0.120001, 0, 0]`.

**Verified**: the depth map is generated with exactly these numbers —
`z == fx_rect · B / disparity` holds to float32 precision (median |err| 0.0000 mm, p99
0.0002 mm over 2.2 M px), in both PERFORMANCE and NEURAL_PLUS. Back-projection therefore
carries no residual model error.

`cx, cy` are **not** the image centre (cy is 49 px off at HD2K). Never substitute `W/2, H/2`.

### Self-calibration — disabled on purpose

`camera_disable_self_calib = True` everywhere. The SDK otherwise re-estimates the stereo
extrinsics at every `open()`, and the rectified intrinsics derive from those, so a
successful self-calibration can shift fx/cx/cy and silently invalidate a stored hand-eye
transform.

On this rig self-calibration **fails every open** anyway (`Error code: 0x01` — the black
backdrop is untextured and the working distance is inside the 1 m exclusion), so intrinsics
are recomputed from unrefined factory data and never change. Five consecutive opens gave
byte-identical values.

Consequence: intrinsics are reproducible, **but a genuine mechanical drift since the
Nov-2025 factory calibration cannot be detected or corrected** and would appear as
point-cloud tilt with no error message. To test, aim at a bright textured scene with
nothing inside 1 m and run:

```bash
pixi run -e sam2 python zed_capture/capture_zed_sam_mask.py --enable-self-calib --sdk-verbose 1
```

If self-calibration then succeeds and reports different fx/cx/cy, the factory calibration
is stale and hand-eye must be re-run.

### Depth quality

- HD2K + NEURAL_PLUS: **1.15 mm** plane-fit RMS on a flat surface at ~1.2 m, ~78–80% valid
- HD720 + NEURAL: 1.96 mm RMS, 98% valid — the extra fill is low-quality background,
  so **higher valid-% is not better**. HD2K is the right default.
- HD2K is exactly 16:9, so 640×360 and 1280×720 are uniform downscales.

### Depth disparity offset

**This camera reports depth that is ~15% too FAR.** The cause is a constant disparity
offset, not a scale error and not a baseline error. Every consumer corrects it.

**The value lives in one place:** `zed_capture/zed_depth_correction.json`.

```json
{ "disparity_offset_px": 16.03, "baseline_m": 0.120001,
  "camera_serial": 22456, "conf_md5": "04912983355a1431b46ba7cd3a3d5c00" }
```

`zed_depth_config.py` finds it with the same 3-level search that `apriltag_image.py`
uses for the intrinsics npz:

```
$ZED_DEPTH_CORRECTION_JSON  ->  <repo>/zed_capture/  ->  the module's own directory
```

Edit the JSON and **every** consumer follows: the capture path, the solver, the
comparison tool, the debug table. `$ZED_DEPTH_OFFSET_PX` overrides it for one run.

**The maths** — in disparity space, where the error actually lives:

```
z_true = fx·B / (fx·B / z_reported + offset_px)
```

At `fx = 1414.58 px` and `B = 120.0 mm` this is **−86.3 mm at 1.0 m** and
**−186.1 mm at 1.5 m**. A single multiplicative factor cannot do this, which is why the
correction is not a scale.

**Where it is applied:**

| consumer | when | why |
|---|---|---|
| `capture_zed_sam_mask.py` | at capture | its output *is* the product, so it must ship corrected |
| `zed_calib_rgbd.py` | **never** | it stays a pure sensor reader |
| the three solvers | at solve time | raw data on disk survives a revision of the value, and existing datasets stay usable |

**It is never applied twice.** The RGB-D npz records `disparity_offset_px`. `0.0` or a
missing key means raw, so the solver corrects it. A non-zero value means the data is
already corrected, so the solver leaves it alone. Two corrections would give 32 px, and
nothing would crash — the calibration would just be silently wrong.

**Measured evidence** (right arm, `zed_calib_003`, 56 frames):

| | `|t_depth − t_apriltag|` | hand-eye residual | outliers |
|---|---|---|---|
| raw | 172.60 mm mean, 244.40 max | 35.22 mm | 32 / 56 |
| corrected | **5.42 mm** mean, 15.80 max | **15.63 mm** | **7 / 56** |

The corrected depth/tag ratio is flat across range — 1.0016 (1.0–1.2 m), 0.9970
(1.2–1.4 m), 0.9904 (1.4–1.6 m) — which is the proof that an offset, not a scale, was the
fault.

> **This is a workaround for a hardware fault, not a fix.** Set
> `disparity_offset_px` to `0.0` after you run `ZED_Calibration`, after a
> self-calibration succeeds, or if you swap the camera. If you leave it non-zero after a
> recalibration it pushes the depth wrong in the **other** direction.
> `zed_depth_config.check_camera()` compares `conf_md5` with the file on disk and warns
> you — but it cannot detect a successful self-calibration, because that does not touch
> the `.conf` file.

To re-measure: capture a tag dataset across 1.0–1.9 m, then fit the single `d` that
satisfies `fx·B/z_depth + d = fx·B/z_pnp`. A good fit has a standard deviation under 1 px
and no trend against range.

### Left-edge stereo dead band

A point at left-image column `u` appears in the right image at `u − disparity`; if
`u < disparity` it has no match. So a band on the **left edge** never gets depth, of width
`fx·B/z`:

| scene depth | dead band @ HD2K | @ 640×360 |
|---|---|---|
| 0.5 m | 339 px | 98 px |
| 1.2 m | 141 px | 41 px |
| 1.8 m | 102 px | 30 px |

Measured: at 1.669 m median depth, predicted 101.7 px and the first column above 50% valid
was x = 104. Keep the workspace (and the calibration tag) out of that band. There is no
equivalent constraint on the right edge.

### Not available on this camera

- **HDR** — SDK 4.2.5 has no `InitParameters.enable_hdr` and no `VIDEO_SETTINGS.HDR*` for a
  stereo ZED. The viewer's `hdr_mode` is a GUI-only key.
- **IMU** — the original ZED has none. The viewer's "IMU orientation" is a point-cloud
  display option and does nothing here.
- `EXPOSURE_TIME`, `ANALOG_GAIN`, `DIGITAL_GAIN`, `DENOISING`, `EXPOSURE_COMPENSATION` —
  ZED X family only.
- **VGA** — `grab()` reportedly fails at VGA on this unit; all other resolutions are fine.

Working image controls: brightness/contrast/saturation/sharpness 0–8, hue 0–11, gamma 1–9,
gain 0–100, exposure 0–100, `AEC_AGC` 0/1, white balance 2800–6500 (step 100),
`WHITEBALANCE_AUTO` 0/1, `LED_STATUS` 0/1.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ZED open() failed: CAMERA_NOT_DETECTED` | Depth Viewer or another process holds the camera. `pgrep -af ZED_Depth_Viewer` |
| Long silent pause on first NEURAL/NEURAL_PLUS open | SDK is optimising its AI model for that resolution. Minutes, not a hang. |
| `ros2 topic list` shows no `/left`, `/right` | `ROS_DOMAIN_ID` mismatch — must be 100 on both sides; container uses FastDDS |
| `No ZED intrinsics npz found` | Run step 3, or set `$ZED_INTRINSICS_NPZ` |
| `--use-depth-translation` errors about a missing `depth` array | The capture was made with the old color-only ZED branch. Re-capture. |
| SAM2 `RuntimeError` about the parent directory | cwd is the repo root, which shadows the `sam2` package. The script chdirs automatically; run it as documented. |
| Few tag detections | The solver retries with CLAHE/gamma variants; if it still fails the tag is out of frame, too oblique, or blown out. Check `debug/<seq>/` |
| Left/right arms swapped | Swap `LEFT_ROBOT_IP` / `RIGHT_ROBOT_IP` in step 1 |

---

## Caveats

- **Part B has been run** — `zed_calib_001`, `002` and `003`. Step 4 still moves both arms,
  so keep the e-stop in reach. The trajectory centres (`left [0.22, −0.32, 0.4]`,
  `right [0.40, 0.20, 0.30]`) are unchanged across all three runs and are known to stay in
  the ZED's field of view.
- Confirm the tag stays visible across the whole figure-eight. A tag that leaves the frame
  just silently reduces the usable image count.
- **Lighting matters more than exposure tuning.** `zed_calib_002` was captured with the room
  lights off and a manual exposure chosen from a lights-on sweep: mean image level 13/255 and
  only 6 of 72 frames detected the tag. Keep the room lights on and leave exposure on auto.
  Note that `DEFAULT_EXPOSURE = None` does **not** restore auto — ZED settings persist across
  opens, so `None` means "whatever manual value was last set". Set `AEC_AGC = 1` to truly
  restore auto.
- **The right arm is consistently worse than the left** (15.63 mm vs 8.14 mm on
  `zed_calib_003`, 7 outliers vs 0). Its poses reach further, so its tag views are more
  distant and more oblique. If you need one good arm, use the left.
- `gripper2tag` is the hand-measured matrix at `calculate_base_to_cam.py:487`
  (translation `[−0.02, 0, 0.0905]` m). Its *refined* variants were previously discredited —
  the measured one held up, and the distortion that made it look wrong on the Azure does not
  exist on the ZED (rectified `disto` = 0).
