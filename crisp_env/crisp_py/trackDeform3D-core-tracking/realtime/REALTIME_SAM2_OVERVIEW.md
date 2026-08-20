# Live Camera + SAM2 Port — Overview for All Four Trackers

This document gives the plan for two changes:

1. Replace the recorded input data with a **frame-source wrapper**. The wrapper supplies
   frames from a live RGBD camera or from the recorded chunks.
2. Replace the precomputed mask files with **SAM2**. SAM2 makes the object mask for each
   frame in real time.

The document then lists the changes for each tracker: dlo, bdlo, cloth, and fabric.

Related document: [REALTIME_DLO_BDLO_CHANGES.md](REALTIME_DLO_BDLO_CHANGES.md) holds the
deep error analysis for the two wire types. This overview refers to it and does not repeat
it. Each `file:line` reference below was opened and checked against the working tree.

---

## 1. The pipeline today (offline)

All four drivers have the same shape. The driver loads full arrays from disk with
`load_chunk_data` at [utils/data_loading.py:15](utils/data_loading.py#L15). The driver cuts
the arrays into clips. The driver makes a new tracker for each clip. The driver gives the
full EE array of the clip to the tracker constructor. Then the driver loops over the frames
and calls `process_frame`.

| Type | Driver | Tracker class | Mask file | Mask direction into the tracker | EE reads inside the tracker |
|---|---|---|---|---|---|
| dlo | [dlo_tracking.py](dlo_tracking.py) | `WireTracker` | `masks/masks.npz` | Inverted: `1 - mask` becomes `precomputed_arm_mask` ([dlo_tracking.py:95](dlo_tracking.py#L95)) | Init only, frame 0 ([wire_init.py:93](initialization/wire_init.py#L93)) |
| bdlo | [bdlo_tracking.py](bdlo_tracking.py) | `WireTracker` | `masks/masks.npz` | Inverted, same as dlo ([bdlo_tracking.py:107](bdlo_tracking.py#L107)) | Init **and every frame** ([wire_tracker.py:1049-1051](tracker/wire_tracker.py#L1049-L1051)) |
| cloth | [cloth_tracking.py](cloth_tracking.py) | `ClothTrackerFull` | `fg_masks/masks.npz` | Direct foreground mask ([cloth_tracker.py:880-894](tracker/cloth_tracker.py#L880-L894)) | Init only ([cloth_init.py:789-792](initialization/cloth_init.py#L789-L792)) |
| fabric | [fabric_tracking.py](fabric_tracking.py) | `FabricTrackerFull` | `fg_mask.npz` | Direct foreground mask ([fabric_tracker.py:708](tracker/fabric_tracker.py#L708)) | Init only ([fabric_init.py:100-101](initialization/fabric_init.py#L100-L101), [:648](initialization/fabric_init.py#L648)) |

Three more facts set the shape of the port:

- The recorded color is **BGR** and the recorded depth is **uint16 millimeters**, registered
  to the color frame ([README.md:71](README.md#L71)). The live wrapper must give the same
  format.
- The wire tracker receives an `rgb` argument and never reads it
  ([wire_tracker.py:1169](tracker/wire_tracker.py#L1169)). The color image is necessary for
  SAM2 only, not for the trackers.
- The drivers do three offline-only steps after tracking: Gaussian smoothing of the full
  trajectory, video rendering, and metric evaluation. These steps read the full clip. A live
  loop cannot do them in the same way.

---

## 2. The target pipeline (live)

```
camera ──► FrameSource ──► color ──► Sam2Segmenter ──► mask ──┐
              │                                               ▼
              └───────────► depth, K ────────────────► live driver ──► tracker ──► output
robot ──► EE pair (camera frame) ─────────────────────────────┘
```

Three new components carry the port. The trackers keep their algorithms.

### 2.1 Component 1: the frame-source wrapper

Make one small interface and two implementations.

```python
class FrameSource:                  # new file, e.g. realtime/frame_source.py
    def start(self): ...
    def stop(self): ...
    def get_latest(self) -> Frame   # blocks until a new frame is available
# Frame: color (H,W,3) uint8 BGR · depth (H,W) float32 mm, registered to color
#        K (3,3) · timestamp · frame_idx
```

- `CameraSource` reads the device (Azure Kinect through `pyk4a`, or RealSense through
  `pyrealsense2`). It aligns the depth to the color frame. It converts the depth to
  millimeters. It reads `K` from the device. The undistortion convention already exists at
  [deform_with_hands/kinect.py:39](deform_with_hands/kinect.py#L39): remap with
  `newCameraMatrix = K`.
- `ReplaySource` reads the existing `rgbd.npz` chunks and plays them back. This keeps the
  offline data usable and gives the parity test (Section 5).
- The source keeps **only the newest frame**. When the consumer is slow, the source drops
  old frames. It does not queue them.

### 2.2 Component 2: the SAM2 segmenter

Make one class, e.g. `Sam2Segmenter`, around the SAM2 streaming (camera) predictor.

- **Start:** give a prompt on the first frame. Use a manual box or click for bring-up.
  A later option: project the two EE positions into the image with `K` and use them as
  positive point prompts, because the grippers touch the object.
- **Each frame:** call the predictor track step. SAM2 propagates the object with its
  internal memory. The output is one binary mask, `uint8`, values 0 and 1, at full frame
  resolution.
- **Clean the mask before the tracker sees it.** Apply a morphological close, fill the
  holes, and remove blobs. Keep the components that touch the dilated mask of the last
  frame. This step is necessary: the tracking path of the wire tracker applies **no**
  spatial cleanup ([REALTIME_DLO_BDLO_CHANGES.md §3.5](REALTIME_DLO_BDLO_CHANGES.md)), so a
  stray SAM2 blob becomes a false leaf and then a frozen anchor.
- **Monitor the mask.** Compare the mask area and the IoU against the last frame. When the
  values fall below a threshold, re-prompt SAM2. Seed the new prompt from the tracker: the
  last 3D keypoints, projected to 2D, are positive points on the object.
- **Mask direction stays a driver concern.** The segmenter always returns the object mask.
  The wire path inverts it (`1 - mask`); the cloth and fabric paths pass it directly.
- **Run SAM2 in its own thread** with a newest-mask slot, for the same drop rule as the
  frame source. The tracker then consumes a mask that can lag one frame. Record this lag;
  at 30 fps it is 33 ms and it is acceptable for slow manipulation.

### 2.3 Component 3: the live driver

Make one new driver, e.g. `realtime_tracking.py --type {dlo,bdlo,cloth,fabric}`. Keep the
four offline drivers unchanged for the benchmark.

The live driver:

1. Builds the tracker **once**. There are no clips. The warm-restart logic inside the
   trackers already covers long sessions
   ([wire_tracker.py:1220-1229](tracker/wire_tracker.py#L1220-L1229),
   [cloth_tracker.py:909-912](tracker/cloth_tracker.py#L909-L912)).
2. Loops: get the newest frame, get the newest mask, get the current EE pair, call
   `process_frame`, publish the keypoints.
3. Does **not** do the offline post-steps. `smooth_trajectories` is a non-causal Gaussian
   filter over the full clip — replace it with a causal filter (short EMA) only if the
   consumer needs smooth output. Skip the per-frame evaluation cloud
   ([cloth_tracking.py:118](cloth_tracking.py#L118)) and the video render.
4. Optionally records color, depth, mask, and keypoints to disk in the chunk format. Then
   the existing offline evaluation can run on a live session afterwards.
5. Reads the EE pair from the robot middleware and transforms it with the existing
   calibration (`get_ee_positions_cam`). When no robot stream exists, it passes `None`.

---

## 3. Changes for each tracker

### 3.0 One change is common to all four: the EE array

Today each driver bakes a full clip array into the constructor
(`'ee_poses_3d': clip_ee_poses` at [dlo_tracking.py:75](dlo_tracking.py#L75),
[bdlo_tracking.py:87](bdlo_tracking.py#L87), [cloth_tracking.py:96](cloth_tracking.py#L96),
[fabric_tracking.py:85](fabric_tracking.py#L85)). The trackers index this array with a frame
counter. In a live session the counter grows without limit. Four guarded read sites then
fail silently:

- [wire_tracker.py:1050](tracker/wire_tracker.py#L1050) — bdlo loses the EE guidance, and
  gripped leaves can exchange identity with free leaves.
- [cloth_init.py:789](initialization/cloth_init.py#L789) and
  [fabric_init.py:645](initialization/fabric_init.py#L645) — a warm restart late in the
  session makes **no** EE-to-corner mapping, with no message.
- [wire_init.py:93](initialization/wire_init.py#L93) reads index `[0]` always — a warm
  restart uses the gripper positions of the first frame.

**The change:** remove the array. Accept the current pair as a keyword argument —
`process_frame(..., ee_pair=None)` — and give the current pair to `initialize`. When the
caller gives no pair, set the stored pair to `None`. Never keep a stale pair.

### 3.1 WireTracker — dlo

| Item | Change |
|---|---|
| Mask input | None in the tracker. The live driver inverts the SAM2 wire mask and passes it as `precomputed_arm_mask`, exactly as today ([wire_tracker.py:265-269](tracker/wire_tracker.py#L265-L269)). |
| Color input | Drop the unused `rgb` argument from the live call path. |
| EE pair | Necessary at init and at each warm restart only (proof in [REALTIME_DLO_BDLO_CHANGES.md §2.2](REALTIME_DLO_BDLO_CHANGES.md)). Apply the Section 3.0 change at [wire_init.py:93](initialization/wire_init.py#L93). |
| Speed | 11.17 ms per frame, measured. The margin at 30 fps is +22.1 ms. SAM2 fits in this margin with a small model, but the separate thread stays the better design. |
| Free win | Delete the dead init repulsion at [wire_init.py:186-195](initialization/wire_init.py#L186-L195); each restart becomes 120 ms faster. |
| Robustness | Correct the two-leaf guard, the degenerate-init check, and the constructor defaults ([REALTIME_DLO_BDLO_CHANGES.md §2.4-2.6](REALTIME_DLO_BDLO_CHANGES.md)). |

### 3.2 WireTracker — bdlo

| Item | Change |
|---|---|
| Mask input | Same inversion as dlo. The **init** mask is strict: exactly 2 junctions and 4 tips on one connected skeleton. The segmenter cleanup (Section 2.2) is mandatory before init. Change the hard `assert` at [wire_initializer.py:649](initialization/wire_initializer.py#L649) to a soft skip, so a bad SAM2 frame does not stop the process. |
| Color input | Delete the `cvtColor` call at [bdlo_tracking.py:104](bdlo_tracking.py#L104); the tracker discards the argument. |
| EE pair | Necessary **every frame** ([wire_tracker.py:1047-1051](tracker/wire_tracker.py#L1047-L1051)). Apply the Section 3.0 change: `ee_pair` keyword each call, `None` when absent. |
| Speed | 37.65 ms per frame, measured. This is already −4.3 ms at 30 fps **before** SAM2. Do the speed work first: sparse node-detection graph, geometry-step rewrite, bounded init retry ([REALTIME_DLO_BDLO_CHANGES.md §3.7](REALTIME_DLO_BDLO_CHANGES.md)). |
| Compatibility | [deform_with_hands/s4_tracking.py](deform_with_hands/s4_tracking.py) subclasses and monkeypatches this tracker. Make all new arguments keyword-only with defaults, and update that file in the same commit. |

### 3.3 ClothTracker

| Item | Change |
|---|---|
| Mask input | Drop-in. `process_frame(depth, mask, frame_idx)` already takes a direct foreground mask ([cloth_tracker.py:880-894](tracker/cloth_tracker.py#L880-L894)). Give it the cleaned SAM2 mask. |
| Init requirement | The init detects 4 corners on the mask contour ([cloth_tracker.py:404](tracker/cloth_tracker.py#L404)) and denoises the contour. Ragged SAM2 edges and holes break this. The segmenter cleanup (close + fill + largest component) must run on every frame that can become an init or restart frame. |
| EE pair | Init only. `_establish_ee_to_corner_mapping` reads the array once ([cloth_init.py:95](initialization/cloth_init.py#L95), [:787-792](initialization/cloth_init.py#L787-L792)). Tracking uses only the stored corner indices ([cloth_tracker.py:934-943](tracker/cloth_tracker.py#L934-L943)). Apply the Section 3.0 change, which also repairs the silent restart loss. |
| Frame index | Stop passing a global `frame_idx` in the live loop. After the Section 3.0 change the index has no consumer; the internal counter is enough. |
| Configuration | `--segment_interior_nodes` is a required, per-garment topology ([run_all.sh](run_all.sh)). A live session must receive it before start. |
| Speed | **Unmeasured.** The cloth driver has no `StageTimer`, and the init uses `repulsion_iterations=500`. Add the timer, measure one chunk, then set the SAM2 budget. |

### 3.4 FabricTracker

| Item | Change |
|---|---|
| Mask input | Drop-in, same as cloth ([fabric_tracker.py:708](tracker/fabric_tracker.py#L708)). |
| Init requirement | The init finds the 4 rectangle corners and snaps the EE positions to them ([fabric_init.py:100-123](initialization/fabric_init.py#L100-L123)). The same segmenter cleanup applies. |
| EE pair | Init only ([fabric_init.py:100-101](initialization/fabric_init.py#L100-L101), mapping at [:630-648](initialization/fabric_init.py#L630-L648)). Tracking pins corners by index only ([fabric_tracker.py:812](tracker/fabric_tracker.py#L812)). Apply the Section 3.0 change. |
| Frame index | Same as cloth: stop passing a global index. |
| Speed | **Unmeasured.** Same action as cloth: add the timer, measure, then budget. |

---

## 4. Speed budget at 30 fps (33.3 ms per frame)

| Type | Tracker time | Margin for SAM2 in the same thread | Decision |
|---|---|---|---|
| dlo | 11.17 ms, measured | +22.1 ms | Possible in-loop; separate thread preferred |
| bdlo | 37.65 ms, measured | negative | Separate thread **and** tracker speed work required |
| cloth | unknown | unknown | Measure first |
| fabric | unknown | unknown | Measure first |

Measure SAM2 itself on the target GPU before you fix the design. Use the smallest
checkpoint that holds the object (`sam2.1_hiera_tiny` or `_small`). With the separate
thread, the mask lags the depth by one frame; the tracker still runs at camera rate.

---

## 5. Order of work

1. **Frame source + parity test.** Run `ReplaySource` with the shipped masks through the
   live driver. The dlo `chunk_1` keypoints must be bit-identical to the offline output.
2. **SAM2 quality gate, offline.** Run SAM2 on the recorded color frames. Compare each mask
   against the shipped mask (IoU, per type). This also closes the open question about the
   unknown mask generator ([REALTIME_DLO_BDLO_CHANGES.md §5.1](REALTIME_DLO_BDLO_CHANGES.md)).
3. **dlo live.** It has the time margin and needs the EE pair at init only.
4. **cloth and fabric live.** After the timing measurement of Section 4.
5. **bdlo live, last.** It needs the per-frame EE pair and the speed work.

---

## 6. New dependencies

[requirements.txt](requirements.txt) today has no deep-learning stack. The port adds:

- `torch` + `torchvision` with CUDA, and a GPU on the live machine
- the `sam2` package and one checkpoint file
- one camera SDK: `pyk4a` (Azure Kinect) or `pyrealsense2` (RealSense)

---

## 7. Open points

1. The undistortion state of the recorded frames is unknown
   ([REALTIME_DLO_BDLO_CHANGES.md §5.2](REALTIME_DLO_BDLO_CHANGES.md)). Undistort the live
   frames with `newCameraMatrix=K` and redo the hand-eye calibration under that convention.
2. The transport for the robot EE stream is not chosen (ROS topic, ZMQ, or none). The
   camera clock and the robot clock also need one synchronization rule.
3. The SAM2 frame rate on the target GPU is unknown until Step 2 of Section 5.
4. The cloth and fabric per-frame times are unknown until the timer is added.
