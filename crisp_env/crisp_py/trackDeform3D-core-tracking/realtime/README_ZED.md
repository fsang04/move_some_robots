# Running the live pipeline with the ZED

Step-by-step for the **ZED** variant of the live DLO tracker: environment,
hand-eye calibration, and the live run. The method itself is unchanged — see
[README.md](README.md) for what the three segmenters do and
[../../zed_capture/README.md](../../zed_capture/README.md) for the ZED camera and
calibration internals. This file is only the running order.

All paths below are relative to `/home/yizhouch/move_some_robots/crisp_env/crisp_py`
(referred to as **`$CP`**). Every python command runs through `pixi run -e humble`
**from `$CP`** — `pixi` resolves its environment from `pixi.toml` in the current
directory, and the env activation sets `ROS_DOMAIN_ID=100` for you.

---

## Status: what runs today

| Step | State |
|---|---|
| 1. Environment | needs `pyzed` re-unzipped (see §1) |
| 2. Offline geometry self-check | **works** — all checks pass |
| 3. Robot bring-up | **works** |
| 4. Hand-eye calibration | **done** — `zed_calib_003`, both arms, solved 2026-08-12 |
| 5. Merge to `transform_ee_cam_world.npz` | **done** — file written, see §5 |
| 6. Live run | code ready; needs the ZED SDK installed (§1) |

### B1 — `--source zed` — DONE

`dlo_tracking_live.py` now takes `--source zed` and constructs `ZedSource`, with
`--zed_resolution` (default HD720), `--zed_fps`, `--zed_depth_mode`.
`--no_undistort` is rejected for the ZED, since the SDK computes depth on the
rectified left image and unrectified colour has no matching depth. The replay
parity path is unaffected (10/10 frames tracked after the change).

### B2 — depth disparity correction — DONE
This camera (SN 22456) reports depth ~15 % too far because of a constant
16.03 px disparity offset ([zed_depth_correction.json](../../zed_capture/zed_depth_correction.json)).
The `zed_calib_003` extrinsics were solved from *corrected* depth, so they are
only valid against corrected depth. Uncorrected, the depth reads **+86 mm too far
at 1.0 m and +186 mm too far at 1.5 m**. `ZedSource._capture_loop` applies no
correction.

Why that breaks `armdiff` specifically: the arm test is
`arm_removed = d >= a - arm_tol_mm` ([armdiff_segmenter.py:184](armdiff_segmenter.py#L184)),
where `a` is the RENDERED arm depth — FK plus `T_base2cam`, i.e. true metric mm —
and `d` is the MEASURED depth. Those are two different distance scales unless the
measurement is corrected, and the bias does not cancel the way it does in a
depth-vs-depth background diff. With `d` reading 186 mm too far and
`--arm_tol 40`, everything reads as "behind the arm": a cable hanging 100 mm in
FRONT of a wrist measures at `a + 86` and gets subtracted as arm. The `--z_range`
gate slips the same way — a true 1.65 m surface reports 1.85 m and falls outside
a `700 1800` gate.

**Fixed.** `ZedSource` now corrects the depth before it leaves the class, in
`_capture_loop` while the array is still float32 (correcting after the uint16
cast would quantise twice). Invalid samples (0) pass through untouched.

The offset is **rescaled to the frame width in use**: the stored 16.03 px is a
disparity at the full 2208 px rectified width, and `corrector_for` does *not*
rescale it. A disparity in px scales with image width exactly as `fx` does, and
`z = fxB/(fxB/z + offset)` is invariant only when both carry the same scale — so
HD720 uses `16.03 × 1280/2208 = 9.29 px` with `fx = 820.04`. Verified identical
either way: HD2K (16.03 px, fx 1414.58) and HD720 (9.29 px, fx 820.04) both give
**−86.28 mm at 1.0 m and −186.11 mm at 1.5 m**, matching the solver's published
figures. Using the stored 16.03 px at HD720 would over-correct by 1.7×.

`reference_width_px` is now a key in
[zed_depth_correction.json](../../zed_capture/zed_depth_correction.json) so the
rescaling has one source of truth rather than a constant buried in the source.

Escape hatch: `--zed_depth_offset_px 0` (or `ZedSource(depth_offset_px=0.0)`)
disables it, with a printed warning. A missing config is a **loud** error, not a
silent skip — the same stance the solvers take, for the same reason. Sessions
written by `--record` now carry `disparity_offset_px` in their `rgbd.npz`, so a
solver re-run on live-recorded data cannot double-correct it.

> This offset is a **workaround for a hardware fault**, not a fix. Set it to 0
> after a successful ZED self-calibration or a camera swap, or it pushes the
> depth wrong the other way.

---

## 1. Environment

```bash
cd /home/yizhouch/move_some_robots/crisp_env/crisp_py     # = $CP
pixi run -e humble python -c "import pyzed.sl; print('pyzed OK')"
```

There are TWO separate pieces, and the wheel alone is not enough:

**(a) The `pyzed` wheel** — python bindings only. Not pixi-managed, so any
`pixi install` drops it. Re-add with (the env is Python 3.11.13, matching `cp311`):

```bash
unzip -o pyzed-4.2-cp311-cp311-linux_x86_64.whl \
      -d .pixi/envs/humble/lib/python3.11/site-packages/
```

**(b) The ZED SDK itself** — `libsl_zed.so` plus udev rules. Missing `pyzed`
gives `ModuleNotFoundError: No module named 'pyzed'`; missing SDK gives
`ImportError: libsl_zed.so: cannot open shared object file`; missing udev rules
give a clean import but `open -> CAMERA NOT DETECTED` even though `lsusb` lists
the camera.

On **this** machine the SDK payload is unpacked at `~/zed_test/inspect/` but was
never installed. **The installer needs root, but the SDK itself does not** — the
libraries work in place:

```bash
export LD_LIBRARY_PATH=$HOME/zed_test/inspect/lib:$LD_LIBRARY_PATH
```

Verified: `pyzed` then imports and reports SDK **4.2.5**, matching the
`pyzed-4.2-cp311` wheel and the version the calibration was made under. If you
can get root, `cd ~/zed_test/inspect && sudo ./linux_install_release.sh` is
tidier — it puts the libs under `/usr/local/zed/lib` with ldconfig, and creates
`/usr/local/zed/settings/` where the SDK caches `SN22456.conf`. That settings dir
is what `zed_depth_config.check_camera()` md5-checks against
[zed_depth_correction.json](../../zed_capture/zed_depth_correction.json) to
confirm the 16.03 px offset still matches this camera's factory calibration;
without it you get a warning, not a failure.

**What genuinely needs root is USB write permission**, and only that. The V4L2
nodes are already yours — systemd-logind sets an ACL (`user:yizhouch:rw-` on
`/dev/video0`), so group membership is irrelevant. But `/dev/bus/usb/<bus>/<dev>`
is `root:root 0664`, and the SDK needs **write** there to issue control transfers;
without it `sl.Camera.get_device_list()` returns `[]` and `open()` fails with
`CAMERA FAILED TO SETUP` after repeated "force a device reboot" attempts.

Note the shipped `99-slabs.rules` covers product IDs `f681`/`f781`/`f881` — **not
`f582`, which is this camera** (the original ZED). So even a full `sudo` install
would not fix the USB permission here. The minimal admin ask is one line in
`/etc/udev/rules.d/99-zed-f582.rules`:

```
SUBSYSTEM=="usb", ATTRS{idVendor}=="2b03", ATTRS{idProduct}=="f582", MODE="0666"
```

Without root at all, the practical route is **Docker** — this account is in the
`docker` group, and the repo already runs its ROS stack in `privileged: true`
containers with `/dev:/dev` mounted (see `crisp_controllers_demos/docker-compose.yaml`);
Stereolabs publish matching `stereolabs/zed:4.2-*` images.

One more root-free adjustment: `DEPTH_MODE.NEURAL` writes optimised AI models
under `/usr/local/zed/resources`. Without that directory, use
`--zed_depth_mode ULTRA`. This does not disturb the calibration —
`zed_depth_correction.json` records that NEURAL / NEURAL_PLUS / ULTRA agree to
0.2 %, which is why the depth mode was ruled out as the cause of the offset.

Verify the whole stack before going further:

```bash
pixi run -e humble python -c "
import pyzed.sl as sl
z = sl.Camera(); print(z.open(sl.InitParameters()))"    # expect SUCCESS
```

Two more environment facts:

- **No torch, no transformers in the `humble` env.** `--segmenter sam2` cannot
  run here. Use `armdiff` (arms allowed inside the workspace) or `pcdiff`. Both
  are CPU-only and fast enough.
- **The calibration scripts have `/home/roahmlab/...` hardcoded.** Repoint them
  once:
  ```bash
  cd $CP/hand_to_eye_calibration/roahm-deformable-objects
  sed -i 's#/home/roahmlab/#/home/yizhouch/#g' \
    capture_poses_and_images_for_calibration_left.py \
    capture_poses_and_images_for_calibration_right.py \
    zsy-testmycode/calculate_base_to_cam_nonlinear_opt.py \
    zsy-testmycode/compare_apriltag_t_vs_depth_t.py
  ```
- **A USB ZED allows exactly one process at a time.** Close the Depth Viewer
  before any capture or live run:
  ```bash
  pgrep -af 'ZED_Depth_CPViewer|ZED_Explorer'    # must print nothing
  ```

## 2. Verify the geometry offline — no camera, no robot

```bash
cd $CP
pixi run -e humble python trackDeform3D-core-tracking/realtime/test_arm_reference.py
```

All checks pass here: flange at the FR3 ready pose `[0.3069, 0, 0.5903]` m,
two-arm render 4.6 ms/frame, `segment()` 8.5 ms/frame. Run this before touching
hardware — it separates FK/render bugs from calibration bugs.

Optional parity check of the tracker itself on shipped data:

```bash
cd $CP/trackDeform3D-core-tracking
pixi run -e humble python dlo_tracking_live.py --source replay --chunk 1
```

## 3. Bring up the robots

**Terminal A** — the ROS 2 stack, from the demos repo (where `docker-compose.yaml` lives):

```bash
cd /home/yizhouch/move_some_robots/crisp/crisp_controllers_demos
LEFT_ROBOT_IP=192.168.2.2 RIGHT_ROBOT_IP=192.168.2.3 \
  docker compose up launch_dual_franka
```

Leave it running. It launches `dual_franka.launch.py`, which pushes the `left`
and `right` namespaces — that is where `/left/joint_states` and
`/right/joint_states` come from, and those are exactly what
[joint_source.py](joint_source.py) subscribes to. The compose file sets
`ROS_DOMAIN_ID=100` and `RMW=fastdds`.

> Bash history has the IPs **both ways** (`.2/.3` and `.3/.2`). If left and
> right come up swapped, that is why.

**Terminal B** — confirm, from `$CP`:

```bash
cd $CP
pixi run -e humble ros2 topic list | grep -E '^/(left|right)/'
pixi run -e humble ros2 topic hz /left/joint_states
```

`ros2 topic hz` prints `WARNING: topic ... does not appear to be published yet`
once at start-up, before its subscription receives anything. That warning is
harmless — if `average rate:` lines follow, the topic is live. Expect **~1000 Hz**
(the FR3 control rate). Ctrl-C to stop.

Nothing at all means an `ROS_DOMAIN_ID` / RMW mismatch, not a robot fault. The
`humble` env leaves `RMW_IMPLEMENTATION` unset, i.e. FastDDS — which matches the
compose default. Do not set `RMW=zenoh` in the demos `.env` unless you also
export `RMW_IMPLEMENTATION=rmw_zenoh_cpp` on this side and start
`launch_zenoh_router`.

Optional safe reset:

```bash
cd $CP/hand_to_eye_calibration/roahm-deformable-objects
pixi run -e humble python send_both_arms_home.py
```

## 4. Hand-eye calibration — already done

`zed_calib_003` (both arms, 2026-08-12) is the current calibration. **Skip to §5
unless the camera or a robot base has moved.** Residuals: left 8.14 mm (0/59
outliers), right 15.63 mm (7/56). Independent check with no tag involved — the
base-to-base geometry the two independently-solved arms imply agrees to 3.6 mm
and 0.07°.

To redo it, the full procedure with all the ZED-specific reasoning is
[../../zed_capture/README.md](../../zed_capture/README.md) Part B. The short form:

**4a. Physical setup.** AprilTag **tag36h11, id 3, 93 mm** on the EE mount
(hardcoded as `EXPECTED_TAG_ID`/`TAG_SIZE_M`), gripper **closed**, Depth Viewer
closed, robots powered with FCI enabled. If your tag mount differs from the
hand-measured `gripper2tag` at
[calculate_base_to_cam_nonlinear_opt.py:403](../../hand_to_eye_calibration/roahm-deformable-objects/zsy-testmycode/calculate_base_to_cam_nonlinear_opt.py#L403),
edit that matrix or the result is silently wrong.

**4b. Capture**, from the calibration directory so `import apriltag` and
`import zed_calib_rgbd` resolve. Use the **same `--seq-name`** for both arms and
do not touch the camera in between:

```bash
cd $CP/hand_to_eye_calibration/roahm-deformable-objects
pixi run -e humble python capture_poses_and_images_for_calibration_left.py  --camera zed --seq-name zed_calib_004
pixi run -e humble python capture_poses_and_images_for_calibration_right.py --camera zed --seq-name zed_calib_004
```

Each arm draws a figure-eight, pausing every 7th control step to grab. Watch
`[zed] grab: valid_depth=XX%`; below ~50 % on the tag usually means the tag is
in the left-edge stereo dead band or too close. Camera settings come from the
`DEFAULT_*` constants in `zed_calib_rgbd.py` — these scripts pass no camera
arguments, so there is no flag for them. Depth is stored **raw** on purpose; the
solver corrects it, and the npz records `disparity_offset_px = 0.0` to say so.

**4c. Solve**, per arm:

```bash
cd zsy-testmycode
for side in left right; do
  pixi run -e humble python calculate_base_to_cam_nonlinear_opt.py \
      --camera zed --side $side --calib-seq-name zed_calib_004 \
      --use-depth-translation --max-images 70 --no-enhance \
      --rot-weight 573 --save-to-calib-dir
done
```

Three flags that are not optional:

- `--use-depth-translation` — AprilTag rotation *plus* depth translation. PnP is
  structurally weak along the viewing ray; depth measures that axis directly.
- `--rot-weight 573` — at 1.5 m, 1° of rotation costs 26 mm of position, so the
  default `1.0` under-weights rotation badly.
- `--no-enhance` — skips the image-enhancement retry ladder: 11 s instead of
  ~20 min, and it changes nothing when the tag is already detected.

Confirm the correction line appears, with the value you expect:

```
[depth-fix] disparity offset +16.03 px applied (fx=1414.58 px, B=120.0 mm, unit=mm): -86.3 mm @ 1.0 m, -186.1 mm @ 1.5 m
```

Drop bad frames and re-solve without re-capturing: `--exclude-images 3,17,42`.

**Deploy the `depth_translation` file, not `apriltag_translation`**, even though
its residual number is larger — each file's residual is measured against the data
it was solved from, so the two are not comparable. `apriltag` extrinsics paired
with corrected depth leave a 9 mm constant bias.

## 5. Merge the two arms into `transform_ee_cam_world.npz`

The solver writes one file per arm, keyed `X_cam_base`. The tracker wants both
arms in one file under the names [utils/transforms.py:8](../utils/transforms.py#L8)
established (`T_left_base2cam`, `T_right_base2cam`, `K`):

```bash
cd /home/yizhouch/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects
pixi run -e humble python make_transform_ee_cam_world.py --camera zed --k-width 1280 \
  --left  captured_calibration_data/zed_calib_003/base2cam_transform_left_nonlinear_opt_depth_translation.npz \
  --right captured_calibration_data/zed_calib_003/base2cam_transform_right_nonlinear_opt_depth_translation.npz \
  --out   captured_calibration_data/zed_calib_003/transform_ee_cam_world.npz
```

**Already run** for `zed_calib_003` — the output file exists.

`--k-width 1280` rescales the exported HD2K intrinsics
(`zed_capture/zed_intrinsics_2208x1242.npz`, `fx = 1414.575`) to the HD720 that
`ZedSource` opens, giving `fx = 820.044, cx = 638.652, cy = 388.263`. K is
per-camera **and** per-resolution; the extrinsics are neither. `armdiff` itself
reads K from the live device, so this entry is for the offline/replay drivers and
to keep the file self-describing — but a wrong K in it is a trap worth avoiding.

## 6. The live run

The driver side is ready; this needs only the ZED SDK from §1 to be installed.

```bash
cd /home/yizhouch/move_some_robots/crisp_env/crisp_py/trackDeform3D-core-tracking
pixi run -e humble python dlo_tracking_live.py \
    --source zed --segmenter armdiff --zed_depth_mode ULTRA \
    --calib ../hand_to_eye_calibration/roahm-deformable-objects/captured_calibration_data/zed_calib_003/transform_ee_cam_world.npz \
    --left_ns left --right_ns right \
    --z_range 700 1800 \
    --record output/dlo_live/session_1
```

`q` or `ESC` stops. Keypoints go to `output/dlo_live/<timestamp>/3d_keypoints.npz`;
`--record` additionally writes the session in chunk format so the unchanged
offline `dlo_tracking.py` can evaluate it afterwards.

Set `--z_range ZMIN ZMAX` (mm) to your **real** workspace depth.

### The start-up (`--init fk`, the armdiff default)

`armdiff` does **not** use `realtime/bootstrap.py`. It has something no other
segmenter has: the joint stream. The grippers *hold* the cable ends, so FK plus
the hand-eye calibration already give both ends in camera mm — no candidate
mask, no skeleton, no guess from the image. `--grasp_offset` (default 0.1034 m,
the franka hand TCP) is the distance from the hand frame to the held point.

The driver waits for two conditions before it initializes, then reports both:

| condition | option | default |
|---|---|---|
| the mask is big enough | `--min_init_mask_px` | 500 px |
| the mask reaches **both** projected grippers | `--max_ee_mask_px` | 60 px |
| how long to wait for them | `--init_timeout` | 60 s |

The second condition is the important one. Without it the tracker initializes
on whatever blob survives, the EE pair snaps to a skeleton belonging to
something else, and every keypoint lands on one pixel. Expect a few tens of px
even when all is well: the arm subtraction (`--arm_dilate`) removes the cable
right at the fingertips, so the mask stops short of them.

**`--bg_mode temporal` (the default) cannot see a cable at rest.** It reports
only what moved closer in the last `--lag` frames, so a cable lying still gives
an empty mask and the init waits out its timeout. Move the cable during the
start-up, or use `--bg_mode static`.

Useful variants:

```bash
# rehearse the rendering with no robots running (fixed ready pose):
... --segmenter armdiff --joints fixed

# object at rest, or arms parked outside the gate — needs the empty-scene ritual:
... --segmenter armdiff --bg_mode static --bg_frames 30
... --segmenter pcdiff

# click the two cable ends instead of the fk init:
... --init click

# the image-based start-up (ridge/bgsub candidates + acceptance test). For
# armdiff the ridge candidates are switched OFF, because a generic "thin thing"
# would otherwise replace the segmenter's own mask whenever that mask fails the
# acceptance test — an empty mask always fails:
... --init auto
```

## 7. Validate the calibration live, before trusting any output

Take the cable **out of the scene**, run with `--joints crisp`, and move the
arms. The mask should stay **empty**. Arm pixels surviving means either the
calibration is off or the joint snapshot lags the exposure: raise `--arm_tol`
(mm from the rendered surface, default 40) and `--arm_dilate` (px, default 9), or
slow the arms. A translation error of *x* mm leaves an *x* mm rim of arm pixels.

If the residual arm pixels form **streaks along the direction of motion**, that
is the joint lag (the snapshot is the latest received state, not the state at
exposure time), not the calibration.

## Gotchas, collected

- The default `--calib` points at `input_data/dlo/calibration/` — the **old
  Azure rig**. Always pass `--calib` explicitly.
- `--bg_mode temporal` (the default) only sees what **moves**: a cable segment
  at rest for more than `--lag` (5) frames drops out of the raw mask. If the arms
  are not keeping the cable in motion, use `--bg_mode static`.
- Fingers render fully open, so the gripper region over-subtracts rather than
  leaking arm pixels.
- ZED colour and depth are already rectified and mutually registered, so there is
  no undistortion step and `--no_undistort` has no ZED meaning (`ZedSource`
  rejects `undistort=False`). The Azure-only warning about matching the
  `newCameraMatrix = K` convention does not apply here.
- ZED depth is float32 mm with NaN/±inf for invalid pixels; `ZedSource` converts
  to uint16 mm with invalid → 0, matching the recorded chunk format.
