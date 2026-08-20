"""Frame sources for the live tracking port (REALTIME_SAM2_OVERVIEW.md §2.1).

One small interface, three implementations:

    KinectSource -- live Azure Kinect through pyk4a. The SDK registers the depth
                    into the color frame; K comes from the device calibration.
    ZedSource    -- live ZED 2 through pyzed. Depth is computed on the rectified
                    left image, so it is registered to the color frame already;
                    K is the rectified left-camera matrix.
    ReplaySource -- plays back the sample recorded input_data/<type>/chunk_* folder in the
                    exact same Frame format, so there is an option for live vs sample stream.

All sources produce the format the recorded chunks already use (README.md:71):
color (H,W,3) uint8 BGR, depth (H,W) uint16 millimeters registered to color.

ReplaySource additionally fills Frame.mask (shipped ground-truth mask) and
Frame.ee_pair (robot EE positions in camera-frame mm); the live sources leave
both None -- the live driver supplies them (SAM2 mask, clicked ends).

pyk4a / pyzed are imported lazily inside the respective start(), so ReplaySource
runs in the base trackdeform3d environment with no camera SDK installed.
"""
import ast
import struct
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from utils.transforms import load_transforms, get_ee_positions_cam


@dataclass
class Frame:
    idx: int                                # source frame counter (gaps = dropped frames)
    t: float                                # seconds, time.monotonic() at capture
    color: np.ndarray                       # (H,W,3) uint8 BGR
    depth: np.ndarray                       # (H,W) uint16 mm, registered to color
    K: np.ndarray                           # (3,3) color intrinsics
    mask: Optional[np.ndarray] = None       # (H,W) uint8 {0,1} object mask (replay only)
    ee_pair: Optional[np.ndarray] = None    # (2,3) float mm, camera frame (replay only)


class FrameSource:
    """start() -> get() until it returns None (end of stream) -> stop()."""

    def start(self) -> 'FrameSource':
        return self

    def stop(self):
        pass

    def get(self) -> Optional[Frame]:
        raise NotImplementedError

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


# ============================================================================
# Azure Kinect
# ============================================================================

class KinectSource(FrameSource):
    """Live Azure Kinect (720p color, NFOV unbinned depth, 30 fps).

    A capture thread keeps ONLY the newest frame: when the consumer is slow, old
    frames are dropped, never queued. get() blocks until a frame newer than the
    last returned one is available.

    undistort=True remaps color and depth with newCameraMatrix=K -- the same
    convention as deform_with_hands/kinect.py:39 (see REALTIME_SAM2_OVERVIEW.md
    §7.1). The hand-eye calibration must use the same convention.
    """

    def __init__(self, undistort: bool = True, timeout: float = 5.0):
        self.undistort = undistort
        self.timeout = timeout
        self.K = None
        self._k4a = None
        self._maps = None
        self._latest = None
        self._last_idx = -1
        self._running = False
        self._cond = threading.Condition()
        self._thread = None

    def start(self) -> 'KinectSource':
        import cv2
        from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS
        from pyk4a.calibration import CalibrationType

        self._k4a = PyK4A(Config(
            color_resolution=ColorResolution.RES_720P,
            depth_mode=DepthMode.NFOV_UNBINNED,
            camera_fps=FPS.FPS_30,
            synchronized_images_only=True,
        ))
        self._k4a.start()
        self.K = np.asarray(
            self._k4a.calibration.get_camera_matrix(CalibrationType.COLOR),
            dtype=np.float64)
        if self.undistort:
            dist = np.asarray(
                self._k4a.calibration.get_distortion_coefficients(CalibrationType.COLOR),
                dtype=np.float64)
            self._maps = cv2.initUndistortRectifyMap(
                self.K, dist, None, self.K, (1280, 720), cv2.CV_32FC1)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self):
        import cv2
        idx = 0
        while self._running:
            try:
                cap = self._k4a.get_capture()
            except Exception:
                continue
            if cap.color is None or cap.transformed_depth is None:
                continue
            color = np.ascontiguousarray(cap.color[:, :, :3])   # BGRA -> BGR
            depth = cap.transformed_depth                        # uint16 mm, color-aligned
            if self._maps is not None:
                color = cv2.remap(color, self._maps[0], self._maps[1], cv2.INTER_LINEAR)
                depth = cv2.remap(depth, self._maps[0], self._maps[1], cv2.INTER_NEAREST)
            frame = Frame(idx=idx, t=time.monotonic(), color=color, depth=depth, K=self.K)
            idx += 1
            with self._cond:
                self._latest = frame       # newest-frame slot: older frames drop here
                self._cond.notify_all()

    def get(self) -> Frame:
        with self._cond:
            ok = self._cond.wait_for(
                lambda: self._latest is not None and self._latest.idx > self._last_idx,
                timeout=self.timeout)
            if not ok:
                raise TimeoutError(f'no Kinect frame within {self.timeout}s')
            self._last_idx = self._latest.idx
            return self._latest

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._k4a is not None:
            self._k4a.stop()
            self._k4a = None


# ============================================================================
# ZED 2
# ============================================================================

class ZedSource(FrameSource):
    """Live ZED 2 through pyzed:

    Two differences from the Kinect:

    * The SDK computes depth on the RECTIFIED left image, so color and depth are
      already registered to each other and already undistorted -- there is no
      remap step and no distortion to remove. `undistort` is accepted only so
      this class is a drop-in for KinectSource; passing False is an error
      because unrectified color has no matching depth.
    * MEASURE.DEPTH is float32 mm with NaN/inf for invalid pixels; it is
      converted to uint16 mm with invalid -> 0, matching the recorded chunks.

    K is the rectified left-camera matrix, so the hand-eye calibration must have
    been taken with the same VIEW.LEFT images (get_zed_intrinsics.py reads the
    same `calibration_parameters.left_cam`).

    THE DEPTH IS CORRECTED before it leaves this class. This camera reports depth
    too far by a constant disparity offset (zed_capture/zed_depth_correction.json),
    and the hand-eye extrinsics were solved from CORRECTED depth -- so the arm
    renderer's true-metric depth and this measured depth are only comparable once
    the same correction is applied here. Uncorrected, `armdiff` reads every pixel
    as behind the arm and eats the cable. The stored offset is a disparity at the
    full rectified width, so it is rescaled to the frame width in use; see
    `depth_offset_px`. Pass depth_offset_px=0.0 to opt out deliberately.

    depth_source='ffs' replaces the SDK matcher with Fast-FoundationStereo (a
    TensorRT engine, see realtime/ffs_trt.py): the capture thread retrieves the
    rectified LEFT+RIGHT pair, the engine computes disparity, and depth comes
    from fx*baseline/disparity. Everything downstream is unchanged -- same
    Frame contract, and the SAME a/d correction is applied, because the (a, d)
    fault lives in the rectified images both matchers consume, not in the SDK
    matcher. depth_mode/min_depth_mm/confidence/depth_stabilization only shape
    the SDK matcher, so they are ignored in ffs mode (the SDK runs with
    DEPTH_MODE.NONE, images only).
    """

    def __init__(self, undistort: bool = True, timeout: float = 5.0,
                 resolution: str = 'HD720', fps: int = 30,
                 depth_mode: str = 'NEURAL', min_depth_mm: float = 300.0,
                 confidence: Optional[int] = None,
                 depth_offset_px: Optional[float] = None,
                 exposure: Optional[int] = None,
                 depth_stabilization: Optional[int] = None,
                 depth_source: str = 'zed',
                 ffs_engine: Optional[str] = None):
        if not undistort:
            raise ValueError('ZedSource always delivers rectified frames; '
                             'undistort=False has no ZED equivalent')
        if depth_source not in ('zed', 'ffs'):
            raise ValueError(f"depth_source must be 'zed' or 'ffs', "
                             f"got {depth_source!r}")
        if depth_source == 'ffs' and not ffs_engine:
            raise ValueError("depth_source='ffs' needs ffs_engine=<path to the "
                             ".engine built by ffs_engines/export_and_build.sh>")
        self.undistort = undistort
        self.timeout = timeout
        self.resolution = resolution
        self.fps = fps
        self.depth_mode = depth_mode
        self.min_depth_mm = min_depth_mm
        self.confidence = confidence
        self.exposure = exposure                     # 0-100 (%); None = auto
        self.depth_stabilization = depth_stabilization   # 0-100; None = SDK default
        self.depth_offset_px = depth_offset_px   # None = from the config file
        self.depth_scale_a = 1.0                 # the applied a, for provenance
        self.depth_source = depth_source
        self.ffs_engine = ffs_engine
        self._ffs = None
        self._baseline_m = None
        self._depth_corrector = None
        self.K = None
        self._zed = None
        self._runtime = None
        self._latest = None
        self._last_idx = -1
        self._running = False
        self._cond = threading.Condition()
        self._thread = None

    def start(self) -> 'ZedSource':
        import pyzed.sl as sl

        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, self.resolution)
        init.camera_fps = self.fps
        init.coordinate_units = sl.UNIT.MILLIMETER     # so MEASURE.DEPTH is mm
        if self.depth_source == 'ffs':
            # Images only: Fast-FoundationStereo computes depth from the
            # rectified pair, so the SDK matcher (and its min-depth /
            # stabilization knobs) has nothing to do.
            init.depth_mode = sl.DEPTH_MODE.NONE
        else:
            init.depth_mode = getattr(sl.DEPTH_MODE, self.depth_mode)
            init.depth_minimum_distance = self.min_depth_mm
            if self.depth_stabilization is not None:
                init.depth_stabilization = int(self.depth_stabilization)
        # Self-calibration re-estimates the stereo extrinsics at open(); when it
        # succeeds, the rectified fx/cx/cy shift and the stored disparity
        # correction (zed_capture/zed_depth_correction.json) plus the hand-eye
        # calibration no longer apply to what this session delivers. The
        # calibration capture (zed_calib_rgbd.py) disables it for the same
        # reason -- live frames must be rectified with the SAME geometry the
        # rig was calibrated under.
        init.camera_disable_self_calib = True

        # Where the SDK finds SN<serial>.conf, the factory stereo calibration it
        # rectifies with. Its default is /usr/local/zed/settings, which needs
        # root to create -- without it open() fails CALIBRATION FILE NOT
        # AVAILABLE, since the SDK's own download does `mkdir /usr/local/zed`.
        settings = self._settings_path()
        if settings is not None:
            init.optional_settings_path = str(settings)

        self._zed = sl.Camera()
        status = self._zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            self._zed = None
            raise RuntimeError(f'ZED open failed: {status}')

        self._runtime = sl.RuntimeParameters()
        if self.confidence is not None:
            self._runtime.confidence_threshold = self.confidence

        # Manual exposure, matching the calibration capture: auto-exposure on
        # this rig meters for the black backdrop and saturates the white arms,
        # which starves the stereo network of texture exactly where the
        # armdiff subtraction needs stable depth. Setting it implicitly
        # disables AEC_AGC (same as zed_calib_rgbd.py).
        if self.exposure is not None:
            self._zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE,
                                          int(self.exposure))
            print(f'[zed] manual EXPOSURE = {self.exposure} '
                  f'(auto-exposure disabled)')

        info = self._zed.get_camera_information()
        cam = info.camera_configuration.calibration_parameters.left_cam
        self.K = np.array([[cam.fx, 0.0, cam.cx],
                           [0.0, cam.fy, cam.cy],
                           [0.0, 0.0, 1.0]], dtype=np.float64)

        width = int(info.camera_configuration.resolution.width)
        try:    # the live baseline, when this pyzed exposes it; else the config's
            baseline_mm = abs(float(
                info.camera_configuration.calibration_parameters.stereo_transform
                .get_translation().get()[0]))
            baseline_m = baseline_mm / 1000.0 if 50.0 < baseline_mm < 500.0 else None
        except Exception:
            baseline_m = None
        self._depth_corrector = self._build_depth_corrector(
            fx=float(cam.fx), width=width, baseline_m=baseline_m)
        self._baseline_m = baseline_m
        if self.depth_source == 'ffs':
            self._ffs = self._build_ffs_matcher()

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _settings_path(self):
        """The factory-calibration directory, or None to leave the SDK default.

        Shares zed_depth_config.settings_dir() with the staleness check, so the
        conf the SDK rectifies with is the same one whose md5 validates the
        disparity offset.
        """
        try:
            zed_capture = Path(__file__).resolve().parents[2] / 'zed_capture'
            if str(zed_capture) not in sys.path:
                sys.path.insert(0, str(zed_capture))
            import zed_depth_config
            path = zed_depth_config.settings_dir()
        except Exception:
            return None
        if not path.is_dir():
            print(f'[zed] settings dir {path} does not exist; the SDK will try to '
                  f'create its default and may fail without root')
            return None
        print(f'[zed] factory calibration from {path}')
        return path

    def _build_depth_corrector(self, fx, width, baseline_m=None):
        """Adding 16px depth offset for Zed camera at this specific resolution.

        Zed camera is calibrated with a 16px depth offset to resolve depth sensor
        issues. This depth corrector applies the same offset to the live camera
        stream.

        Pass depth_offset_px=0.0 to ZedSource to opt out on purpose.
        """
        if self.depth_offset_px == 0.0:
            print('[zed] depth correction DISABLED by depth_offset_px=0.0 -- the '
                  'hand-eye extrinsics expect CORRECTED depth')
            return None

        zed_capture = Path(__file__).resolve().parents[2] / 'zed_capture'
        if str(zed_capture) not in sys.path:
            sys.path.insert(0, str(zed_capture))
        try:
            import zed_depth_config
        except ImportError as exc:
            raise ImportError(
                f'Cannot import zed_depth_config from {zed_capture}. The ZED depth '
                'disparity-offset correction is required: the hand-eye extrinsics '
                'were solved from CORRECTED depth, and raw ZED depth reads ~15% too '
                'far. Restore zed_capture/, set $ZED_DEPTH_CORRECTION_JSON, or pass '
                'ZedSource(depth_offset_px=0.0) to opt out deliberately.') from exc

        if self.depth_offset_px is not None:
            offset_at_width = float(self.depth_offset_px)
        else:
            # Scale by the fx ratio, NOT the width ratio: the ZED rectifies
            # separately per resolution, so rectified fx is not proportional to
            # width (1414.58 @ 2208 px, but 693.82 @ 1280 px -- width-scaling
            # would say 820.04, an 18% error that over-corrects by 14 mm at 1 m).
            cfg = zed_depth_config.load()
            ref_fx = float(cfg.get('reference_fx_px', 0.0))
            if not ref_fx:
                raise KeyError(
                    f"'reference_fx_px' is missing from "
                    f"{zed_depth_config.config_path()}. The disparity offset is "
                    f"quoted at one resolution and must be rescaled by fx to be "
                    f"used at another; without the reference fx that cannot be "
                    f"done. For SN22456 it is 1414.575439 (the rectified HD2K "
                    f"left-camera fx, as in zed_intrinsics_2208x1242.npz).")
            offset_at_width = zed_depth_config.offset_px() * (fx / ref_fx)
            print(f'[zed] disparity offset rescaled by fx {ref_fx:.2f} -> {fx:.2f} '
                  f'({width} px wide): {zed_depth_config.offset_px():.2f} -> '
                  f'{offset_at_width:.2f} px')

        corrector = zed_depth_config.corrector_for(
            'zed', fx, unit='mm', offset_px_override=offset_at_width,
            baseline_m_override=baseline_m)
        self.depth_offset_px = offset_at_width
        self.depth_scale_a = float(corrector.scale)   # provenance (ChunkRecorder)
        return corrector if corrector.enabled else None

    def _build_ffs_matcher(self):
        """The Fast-FoundationStereo TRT engine that replaces the SDK matcher.

        Needs the stereo baseline to turn disparity into depth: the live one
        from the SDK when available, else the measured baseline_m from
        zed_depth_correction.json (same source the corrector uses).
        """
        if self._baseline_m is None:
            zed_capture = Path(__file__).resolve().parents[2] / 'zed_capture'
            if str(zed_capture) not in sys.path:
                sys.path.insert(0, str(zed_capture))
            import zed_depth_config
            baseline = float(zed_depth_config.load().get('baseline_m', 0.0))
            if not baseline:
                raise RuntimeError(
                    'depth_source="ffs": no stereo baseline. Neither the SDK '
                    'stereo_transform nor baseline_m in '
                    'zed_capture/zed_depth_correction.json provided one.')
            self._baseline_m = baseline
        try:
            from realtime.ffs_trt import FfsTrtMatcher
        except ImportError:                       # run with realtime/ on sys.path
            from ffs_trt import FfsTrtMatcher
        matcher = FfsTrtMatcher(self.ffs_engine)
        print(f'[ffs] SDK matcher OFF; depth = Fast-FoundationStereo '
              f'(baseline {self._baseline_m * 1000:.2f} mm, fx {self.K[0, 0]:.2f} px, '
              f'a/d correction {"ON" if self._depth_corrector is not None else "OFF"})')
        return matcher

    def _capture_loop(self):
        import pyzed.sl as sl
        color_mat, depth_mat, right_mat = sl.Mat(), sl.Mat(), sl.Mat()
        idx = 0
        said_ffs = False
        while self._running:
            if self._zed.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
                continue
            self._zed.retrieve_image(color_mat, sl.VIEW.LEFT)        # BGRA uint8
            # get_data() views the Mat buffer that the next grab() overwrites,
            # so the arrays below must be copies.
            color = np.ascontiguousarray(color_mat.get_data()[:, :, :3])
            if self._ffs is not None:
                self._zed.retrieve_image(right_mat, sl.VIEW.RIGHT)   # rectified
                right = np.ascontiguousarray(right_mat.get_data()[:, :, :3])
                d = self._ffs.depth_mm(color, right, fx=float(self.K[0, 0]),
                                       baseline_m=self._baseline_m)
                if not said_ffs:
                    said_ffs = True
                    print(f'[ffs] first pair inferred in '
                          f'{self._ffs.last_infer_ms:.1f} ms '
                          f'({self._ffs.width}x{self._ffs.height} engine); '
                          f'later pairs are faster (warmup)')
            else:
                self._zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)  # float32 mm
                d = np.nan_to_num(depth_mat.get_data(),
                                  nan=0.0, posinf=0.0, neginf=0.0)
            if self._depth_corrector is not None:
                # while still float32: correcting after the uint16 cast would
                # quantise twice. Invalid (0) samples pass through untouched.
                d = self._depth_corrector(d)
            np.clip(d, 0.0, 65535.0, out=d)
            depth = d.astype(np.uint16)
            if depth.ndim == 3:                  # some pyzed builds return (H,W,1)
                depth = depth[:, :, 0]
            frame = Frame(idx=idx, t=time.monotonic(), color=color, depth=depth, K=self.K)
            idx += 1
            with self._cond:
                self._latest = frame       # newest-frame slot: older frames drop here
                self._cond.notify_all()

    def get(self) -> Frame:
        with self._cond:
            ok = self._cond.wait_for(
                lambda: self._latest is not None and self._latest.idx > self._last_idx,
                timeout=self.timeout)
            if not ok:
                raise TimeoutError(f'no ZED frame within {self.timeout}s')
            self._last_idx = self._latest.idx
            return self._latest

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._zed is not None:
            self._zed.close()
            self._zed = None


# ============================================================================
# Recorded-chunk replay
# ============================================================================

def _parse_npy_header(fh):
    """Read an .npy header from a stream positioned at its start.
    Same helper as test_sam2_mask_dlo.py:50 (numpy 2.x removed the private one)."""
    magic = fh.read(6)
    if magic != b'\x93NUMPY':
        raise ValueError(f'not a .npy stream (magic={magic!r})')
    major = fh.read(1)[0]
    fh.read(1)                                  # minor
    hlen_size = 2 if major == 1 else 4
    hlen = int.from_bytes(fh.read(hlen_size), 'little')
    hdr = ast.literal_eval(fh.read(hlen).decode('latin1').strip())
    return hdr['shape'], hdr['fortran_order'], np.dtype(hdr['descr'])


def _memmap_npz(path, name):
    """Memory-map one STORED member of an .npz (rgbd.npz is Stored; avoids a
    full read of the 1.4 GB file). Same helper as test_sam2_mask_dlo.py:68."""
    with open(path, 'rb') as fh:
        info = zipfile.ZipFile(fh).getinfo(name + '.npy')
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"member '{name}' is compressed; cannot memmap")
        fh.seek(info.header_offset)
        n, m = struct.unpack('<HH', fh.read(30)[26:30])
        fh.seek(info.header_offset + 30 + n + m)
        shape, fortran, dtype = _parse_npy_header(fh)
        offset = fh.tell()
    if fortran:
        raise ValueError('fortran-order arrays are not supported')
    return np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=shape)


class ReplaySource(FrameSource):
    """Plays back a recorded chunk as if it were a camera.

    Exposes `ee_poses_3d` (T,2,3) camera-frame mm after start(), because the
    tracker constructor wants the array up front (dlo_tracking.py:75).

    fps=None plays as fast as the consumer runs; fps=30 paces the playback to
    real time (frames are NOT dropped when the consumer is slow -- replay is
    the deterministic parity path).
    """

    def __init__(self, chunk_dir, calib_dir, mask_file: str = 'masks/masks.npz',
                 mask_key: str = 'masks', fps: Optional[float] = None,
                 max_frames: Optional[int] = None):
        self.chunk_dir = Path(chunk_dir)
        self.calib_dir = Path(calib_dir)
        self.mask_file = mask_file
        self.mask_key = mask_key
        self.fps = fps
        self.max_frames = max_frames
        self.K = None
        self.ee_poses_3d = None
        self._i = 0
        self._t0 = None

    def start(self) -> 'ReplaySource':
        self._color = _memmap_npz(self.chunk_dir / 'rgbd.npz', 'color')
        self._depth = _memmap_npz(self.chunk_dir / 'rgbd.npz', 'depth')
        # Forward the recorded provenance, so a --record re-record of this
        # chunk stamps what the depth ACTUALLY carries (the correction applied
        # at capture), not raw -- otherwise a solver run on the re-record
        # would apply the a/d correction a second time.
        with np.load(self.chunk_dir / 'rgbd.npz') as z:
            self.depth_offset_px = (float(z['disparity_offset_px'])
                                    if 'disparity_offset_px' in z.files else 0.0)
            self.depth_scale_a = (float(z['disparity_scale'])
                                  if 'disparity_scale' in z.files else 1.0)
            self.depth_source = (str(z['depth_source'])
                                 if 'depth_source' in z.files else 'replay')
        mask_path = self.chunk_dir / self.mask_file
        self._masks = np.load(mask_path)[self.mask_key] if mask_path.exists() else None

        tf = load_transforms(self.calib_dir)
        self.K = tf['K']

        left = np.load(self.chunk_dir / 'left_arm_poses.npz')
        right = np.load(self.chunk_dir / 'right_arm_poses.npz')
        n_pose = min(len(left.files), len(right.files))

        self._T = min(self._color.shape[0], self._depth.shape[0], n_pose)
        if self._masks is not None:
            self._T = min(self._T, self._masks.shape[0])
        if self.max_frames:
            self._T = min(self._T, self.max_frames)

        self.ee_poses_3d = np.stack([
            get_ee_positions_cam(left[f'arr_{i}'], right[f'arr_{i}'],
                                 tf['T_left_base2cam'], tf['T_right_base2cam'])
            for i in range(self._T)])
        return self

    def get(self) -> Optional[Frame]:
        if self._i >= self._T:
            return None
        if self.fps:
            if self._t0 is None:
                self._t0 = time.monotonic()
            due = self._t0 + self._i / self.fps
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        i = self._i
        self._i += 1
        return Frame(
            idx=i, t=time.monotonic(),
            color=np.asarray(self._color[i]),
            depth=np.asarray(self._depth[i]),
            K=self.K,
            mask=np.asarray(self._masks[i]) if self._masks is not None else None,
            ee_pair=self.ee_poses_3d[i],
        )
