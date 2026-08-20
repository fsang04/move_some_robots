# Live Azure Kinect Port for the DLO Tracker

This document describes the camera wrapper (frame_source.py), the live mask
segmenters (sam2_segmenter.py, pcdiff_segmenter.py, armdiff_segmenter.py),
and the live driver. Either live camera feed or original sample input data
can be used.

**There are three interchangeable segmentation methods.** Every file below
belongs to exactly one of them, or to the shared infrastructure they all use:

| Method | `--segmenter` | Files | One line |
|---|---|---|---|
| **1. SAM2 segmentation** | `sam2` (default) | sam2_segmenter.py, bootstrap.py | Prompt SAM2 once on frame 0, then let its streaming memory carry the mask. GPU. |
| **2. Point-cloud difference** | `pcdiff` | pcdiff_segmenter.py | Diff the live 3D points against a static empty-scene reference. CPU only, no prompt. |
| **3. Arm point-cloud difference** | `armdiff` | armdiff_segmenter.py, arm_reference.py, joint_source.py, test_arm_reference.py | Same 3D difference, but the robot arms are rendered from the live joints and subtracted, so the arms may manipulate the object. CPU only, needs joints + calibration. |
| *(shared)* | — | frame_source.py, dlo_tracking_live.py | Camera/replay input and the driver that runs any of the three. |

---

## 1. The new files

### 1.0 Shared infrastructure (all three methods)

| File | Function |
|---|---|
| [frame_source.py](frame_source.py) | The camera wrapper. `Frame` is one data record: color (BGR uint8), depth (uint16 mm, registered to color), K, index, time. `KinectSource` reads the live Azure Kinect through `pyk4a`. `ReplaySource` plays back a recorded chunk in the same format. |
| [dlo_tracking_live.py](../dlo_tracking_live.py) | The live driver. It builds one `WireTracker` with the same parameters as the offline driver, loops over frames from a `FrameSource`, and shows the keypoints in a window. `--record` writes the session in the chunk format, so the unchanged offline driver can evaluate it later. |

### 1.1 Method 1 — SAM2 segmentation (`--segmenter sam2`)

| File | Function |
|---|---|
| [sam2_segmenter.py](sam2_segmenter.py) | `Sam2Segmenter` makes the object mask for each live frame. It uses the same SAM2 streaming code path that [test_sam2_mask_dlo.py](../test_sam2_mask_dlo.py) validated, and it cleans the mask (close + keep the largest component). |
| [bootstrap.py](bootstrap.py) | The automatic session start (no clicks, no prompts) that SAM2 needs, since SAM2 must be told what to segment. It finds the cable with a promptless candidate mask, verifies it with the acceptance test, prompts SAM2 from its skeleton, and reads the two cable ends from the final mask. (Methods 2 and 3 reuse only its acceptance test + end extraction — they need no prompt; see §2.5.) |

### 1.2 Method 2 — point-cloud difference (`--segmenter pcdiff`)

| File | Function |
|---|---|
| [pcdiff_segmenter.py](pcdiff_segmenter.py) | `PointCloudDiffSegmenter`, the alternative to SAM2. The deformable_seg point-cloud-difference method: capture an empty-scene depth reference once, then per frame mark every pixel whose 3D point moved more than a threshold from the reference. numpy + cv2 only — no torch, no GPU, ~3 ms/frame. |

### 1.3 Method 3 — arm point-cloud difference (`--segmenter armdiff`)

Method 2 plus a rendered model of the robot arms, so the arms can be in the
workspace. The three files split as: *what the arms look like*
(arm_reference.py), *where the arms are* (joint_source.py), *the per-frame
subtraction* (armdiff_segmenter.py).

| File | Function |
|---|---|
| [armdiff_segmenter.py](armdiff_segmenter.py) | `ArmDiffSegmenter`: background subtraction PLUS the arms subtracted via the rendered reference, so the arms may manipulate the object inside the workspace gate — the live port of the offline deformable_seg arm-replay method. numpy + cv2 only, ~10 ms/frame on top of the render. |
| [arm_reference.py](arm_reference.py) | The rendered-arm depth reference. `FrankaArmModel` builds FK + surface point samples straight from franka_description (kinematics.yaml + collision STLs — no xacro, no pyrender, no GL); `ArmDepthRenderer` splats both arms into a (H,W) depth image in ~5 ms/frame. |
| [joint_source.py](joint_source.py) | The live joint stream. `CrispJointSource` reads both arms' joints from the crisp_py JointState subscriptions; `ConstantJointSource` is the fixed-pose stand-in for rehearsal and tests. |
| [test_arm_reference.py](test_arm_reference.py) | Self-check for this method, no camera or robot needed: FK against the published FR3 ready pose, render sanity + timing, and synthetic end-to-end segmentation (moving cable + moving arm over a table). |

---

## 2. How to use it

Same driver for all three methods; `--segmenter` picks the method. Keys in the
display window: `q` or `ESC` = stop. In the click step: `ENTER`/`SPACE` =
confirm, `z` = undo the last click, `ESC` = abort.

### 2.1 Command lines, by method

#### Recorded data (no camera, no robot — the parity path)

```bash
# Shipped masks, needs only the base environment:
python dlo_tracking_live.py --source replay --chunk 1

# Method 1 on recorded data: SAM2 segmentation + autonomous prompting
# (no empty scene needed):
python dlo_tracking_live.py --source replay --chunk 1 --init auto --sam2_on_replay
```

#### Method 1 — SAM2 segmentation (live Kinect, default)

```bash
pip install pyk4a          # one new dependency; torch + transformers are already installed

# Fully automatic start (no clicks):
python dlo_tracking_live.py --source kinect [--record output/dlo_live/session_1]

# Manual click start:
python dlo_tracking_live.py --source kinect --init click
```

#### Method 2 — point-cloud difference (no SAM2, no GPU)

```bash
python dlo_tracking_live.py --source kinect --segmenter pcdiff
```

#### Method 3 — arm point-cloud difference (arms inside the workspace)

```bash
# Arms SUBTRACTED using the live joint stream (crisp_py):
python dlo_tracking_live.py --source kinect --segmenter armdiff --calib <rig_calib>.npz

# Rendering rehearsal without robots (fixed ready pose, no ROS):
python dlo_tracking_live.py --source kinect --segmenter armdiff --joints fixed

# Geometry self-check, no hardware at all:
python realtime/test_arm_reference.py
```

### 2.2 The three methods side by side (`--segmenter`, kinect only)

The per-frame mask source is independent of the init mode:

| | **1.** `sam2` (default) | **2.** `pcdiff` | **3.** `armdiff` |
|---|---|---|---|
| Method | SAM2 streaming memory, prompted once on frame 0 | 3D point-cloud difference against a static empty-scene reference ([pcdiff_segmenter.py](pcdiff_segmenter.py)) | background difference + the arms RENDERED from the live joints and subtracted ([armdiff_segmenter.py](armdiff_segmenter.py)) |
| Origin | [test_sam2_mask_dlo.py](../test_sam2_mask_dlo.py) validation | deformable_seg `seg_with_arms_utils.py` (back-project → point distance → depth gate → largest component) | the same offline method's arm-replay reference, rendered live from kinematics ([arm_reference.py](arm_reference.py)) |
| Needs | torch + transformers, CUDA GPU | numpy + cv2 only | numpy + cv2 + crisp_py joints + hand-eye calibration |
| Speed | ~30 ms/frame (GPU) | ~3 ms/frame (CPU) | ~15 ms/frame (CPU: ~5 render + ~10 diff) |
| Start-up ritual | none | `--bg_frames` empty-scene frames, then place the cable | none (`--bg_mode temporal`) or the pcdiff ritual (`--bg_mode static`) |
| Fails when | mask drifts off the object, thin cable breaks the memory | anything that was in the reference MOVES (a robot arm), or the object lies within `--pcdiff_threshold` mm of the background | the calibration/joint stream is off by more than `--arm_tol` + `--arm_dilate`, or (temporal mode) the object stops moving for longer than `--lag` frames |
| Details below | §2.4 | §2.5 | §2.6 |

### 2.3 The three init modes (`--init`) — shared by all methods

The init mode answers two questions for frame 0: *where are the cable ends*
(the EE pair) and *which object must SAM2 segment* (the prompt — method 1
only). After frame 0 neither input is needed: the single-DLO tracking loop
takes its leaf anchors from the detected skeleton tips
([wire_tracker.py:1032-1044](../tracker/wire_tracker.py#L1032-L1044)).

| Mode | EE pair from | SAM2 prompt from | Manual input |
|---|---|---|---|
| `auto` (kinect default) | endpoints of the accepted mask skeleton | interior points of the same skeleton | none |
| `click` | two clicked ends, back-projected | the clicks | 2+ clicks |
| `replay` (replay default) | recorded robot poses | not used (shipped masks) | none |

### 2.4 Method 1 — how `auto` bootstrapping works ([bootstrap.py](bootstrap.py))

1. **Candidate mask, no prompt.** Default: a Frangi ridge filter finds the
   "thin, long thing" inside the workspace depth gate (`--z_range ZMIN ZMAX`,
   mm). Alternative: `--bootstrap bgsub` records the empty scene first and
   takes the pixels that changed.
2. **Acceptance test.** The candidate's skeleton must be one open curve:
   enough pixels, two real endpoints, one path that covers most of the
   skeleton. A frame that fails is dropped and the next frame is tried —
   lay the cable open and uncrossed at session start.
3. **SAM2 completes the object.** Interior points of the accepted path prompt
   SAM2. The candidate only needs precision (points ON the cable), not
   completeness — SAM2 grows it to the full cable. Refine rounds re-prompt
   from SAM2's own skeleton while the path keeps growing.
4. **The EE pair.** The two farthest endpoints of the final skeleton,
   back-projected through the depth (walking inward past depth holes).
5. **Warm restarts stay automatic.** On every skipped frame the driver
   refreshes the stored EE pair from the current mask, so a restart
   re-initializes with CURRENT ends, not frame-0 ends.

### 2.5 Method 2 — how `pcdiff` works ([pcdiff_segmenter.py](pcdiff_segmenter.py))

The reference is the median of `--bg_frames` empty-scene depth frames (holes
excluded). Per frame, a pixel is foreground when its 3D point is more than
`--pcdiff_threshold` mm (default 30) from the reference point, inside the
`--z_range` gate. Because both point clouds share the same pixel rays, the 3D
distance equals `|Δdepth| · ‖ray‖` exactly — so the per-frame cost is a few
array operations. The offline deformable_seg script diffs against a
*synchronized arm-only replay* of the same robot trajectory; live there is no
second synchronized stream, so the static reference stands in for it (see
Cautions) — that gap is what method 3 closes.

With `--init auto`, the pcdiff mask itself is the frame-0 candidate: it goes
through the same acceptance + thinness tests and EE-endpoint extraction as
method 1, but no SAM2 prompting/refinement rounds are needed (`--bootstrap` is
ignored — the reference capture already happened).

### 2.6 Method 3 — how `armdiff` works

Per frame, three subtractions on the depth image (all point distances use the
pcdiff ray identity, so everything is array ops):

1. **Background.** `--bg_mode temporal` diffs against the frame `--lag` frames
   ago — the two-capture scheme: no start-up ritual, arms and object may
   already be in view; whatever MOVED within the lag window is foreground,
   and the closer-only test drops the ghost at the old location.
   `--bg_mode static` diffs against an empty-scene median (the pcdiff
   ritual) and also finds an object at rest.
2. **Arms.** `CrispJointSource` reads both arms' joints from the crisp_py
   JointState subscriptions; `FrankaArmModel` runs FK (origins from
   franka_description's kinematics.yaml, every joint revolute about local z);
   `ArmDepthRenderer` transforms ~20k pre-sampled surface points per arm by
   FK + `T_*_base2cam` and z-buffers them into the pixel grid. Every pixel
   at or BEHIND the rendered arm surface (within `--arm_tol` mm, silhouette
   grown by `--arm_dilate` px) is removed. Points clearly IN FRONT survive —
   a cable hanging before a wrist stays in the mask.
3. **Cleanup.** Workspace `--z_range` gate, morphological close, largest
   component — the same `clean_mask` as the other segmenters.

Trade-offs to know about:

- **Temporal mode only sees what moves.** A cable segment that lies still for
  more than `--lag` frames drops out of the raw mask (the largest-component
  filter then decides what survives). If the arms keep the cable in motion
  this is the zero-ritual mode; otherwise use `--bg_mode static`.
- **The joint snapshot is the latest received, not the exposure-time state.**
  At 30 fps the rendered arm can trail the pixels by roughly a frame;
  `--arm_tol` (depth) and `--arm_dilate` (2D) absorb that. If arm streaks
  survive along the motion direction, raise them or slow the arms.
- **Collision meshes, open fingers.** The renderer uses the collision STLs
  (the finger has none — its four xacro collision boxes are used) and renders
  fingers fully open by default: the largest silhouette, i.e. over-subtract
  near the gripper rather than leak arm pixels.

`--init auto` behaves as in method 2: the armdiff mask is the frame-0
candidate, no SAM2 prompting needed.

---

## 3. Verification (all on the shipped chunk 1)

1. **Parity**: the replay path through the live driver gives keypoints that are
   **bit-identical** to the offline `output/dlo/chunk_1/clip_0/3d_keypoints.npz`
   (max difference 0.0 mm over 20 frames), at 66+ fps.
2. **Kinect-style init**: a one-frame EE array (the shape clicks or auto init
   give) initializes and tracks 10/10 frames.
3. **Fully automatic start** (ridge → SAM2 → tracker, no shipped masks, no
   robot poses, no clicks): the auto EE ends land **22 mm / 11 mm** from the
   true robot gripper positions; the init path length is 1033.5 mm vs
   1034.7 mm for the offline robot-EE init; 30/30 frames tracked.
4. **Mask sources measured against the shipped ground truth**: a blue HSV
   threshold reaches IoU 0.58 and passes the tracker's acceptance test on
   96.7% of frames at 3.6 ms/frame (CPU); SAM2-tiny streams at ~30 ms/frame
   (GPU). The ridge filter alone finds only part of the cable — that is why
   `auto` uses it as the SAM2 prompt source, not as the mask.

---

## 4. Cautions

**All methods.** The live mask quality controls the tracker in live mode.

- In `auto` mode, set `--z_range` to your real workspace depth (mm). A wide
  gate lets the ridge filter latch onto other thin structures (window frames,
  tent edges). The acceptance test cannot tell "a cable" from "another thin
  object" — the depth gate must do that.
- `KinectSource` undistorts with `newCameraMatrix = K` by default (the
  convention from [deform_with_hands/kinect.py:39](../deform_with_hands/kinect.py#L39)).
  If you later add a real robot stream, make the hand-eye calibration with the
  same convention (open point §7.1 of
  [REALTIME_SAM2_OVERVIEW.md](../REALTIME_SAM2_OVERVIEW.md)).

**Method 1 (`sam2`).**

- Use a larger `--sam2_model` or a bigger `--close_ksize` when the skeleton
  path between the two ends breaks.

**Method 2 (`pcdiff`).**

- It uses a STATIC reference: anything present at reference capture that later
  moves (a robot arm entering the workspace) becomes foreground. Keep the arms
  outside the `--z_range` gate, or rely on the largest-component filter
  beating them — segmentation of the cable while an arm manipulates it inside
  the gate is NOT solved by the static reference. That case is what method 3
  is for: it renders the arms from kinematics (the faithful live port of the
  offline arm-replay method).
- `--pcdiff_threshold` (mm) must exceed the depth sensor noise but stay below
  the object's height above the background. 30 mm works for a cable on a
  table; a cable lying flatter than that needs SAM2.

**Method 3 (`armdiff`).**

- It stands or falls with the hand-eye calibration: the
  `T_left/right_base2cam` in `--calib` must be for THIS rig, in meters, and
  made under the same `newCameraMatrix = K` undistortion convention the
  KinectSource uses (see All methods above). A translation error of x mm
  leaves an x mm rim of arm pixels that `--arm_dilate` must cover.
  Validate the geometry offline first: `python realtime/test_arm_reference.py`
  (FK + rendering + synthetic segmentation, no hardware), then check the
  live overlay with `--joints crisp` while the cable is out of the scene —
  the mask should be EMPTY while the arms move.
