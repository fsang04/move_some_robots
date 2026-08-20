"""Fast-FoundationStereo TensorRT runner for the live ZED path.

Runs a single-file TRT engine exported by
Fast-FoundationStereo/scripts/make_single_onnx.py (see
/home/yizhouch/move_some_robots/ffs_engines/export_and_build.sh). The engine
computes disparity from a rectified stereo pair; this module turns that into
the depth map ZedSource delivers.

Import cost: numpy + cv2 only. tensorrt and cuda-python are imported lazily
inside FfsTrtMatcher.__init__, so this module follows the realtime/ rule
(numpy+cv2-only at import time) and every FrameSource still imports in every
env. The humble pixi env needs `tensorrt-cu12` and `cuda-python` installed
via `python -m pip` -- like the pyzed wheel, a pixi env REBUILD REMOVES THEM
(README_ZED.md section 1); re-add with:
    .pixi/envs/humble/bin/python -m pip install tensorrt-cu12==<engine version> cuda-python

Preprocessing contract (must match run_demo_single_trt.py):
  - RGB channel order (the ZED delivers BGR; converted here)
  - direct-stretch resize to the engine's fixed (H, W), no padding
  - ImageNet normalization: (pixel/255 - mean) / std
  - NCHW float32; bindings 'left_image', 'right_image'; output 'disparity'

Geometry: the engine sees images stretched by sx = W_engine / W_input, so its
disparity is in ENGINE-resolution pixels and fx scales by the same sx:
    depth = (fx * sx) * baseline / disparity
Depth is resolution-invariant, so the (H_eng, W_eng) depth map is resized
back to the input size with nearest-neighbour (no cross-edge blending).

An engine is tied to ONE GPU model and ONE TensorRT version. On a version
mismatch deserialization fails; rebuild with ffs_engines/export_and_build.sh.

The a/d disparity correction (zed_capture/zed_depth_correction.json) is NOT
applied here on purpose: the caller (ZedSource._capture_loop) applies the same
DepthCorrector it applies to SDK depth. The (a, d) fault lives in the
RECTIFIED IMAGES (lens yaw the factory calibration misses), not in any
matcher, so FFS disparity carries it exactly like SDK disparity does.
"""
import time

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _import_cudart():
    """cuda-python moved its modules across versions; try both layouts."""
    try:
        from cuda.bindings import runtime as cudart      # cuda-python >= 12.6
    except ImportError:
        from cuda import cudart                          # cuda-python <= 12.5
    return cudart


class FfsTrtMatcher:
    """One TRT engine, buffers allocated once, one infer() per stereo pair.

    Not thread-safe: one instance belongs to one thread (ZedSource's capture
    thread). Building a second instance in the same process is fine.
    """

    def __init__(self, engine_path: str, verbose: bool = True):
        import tensorrt as trt
        cudart = self._cudart = _import_cudart()

        self.engine_path = str(engine_path)
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, 'rb') as f:
            blob = f.read()
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(
                f'TensorRT could not deserialize {self.engine_path} '
                f'(installed TensorRT {trt.__version__}). Engines only load '
                f'under the TensorRT version and GPU they were built on -- '
                f'rebuild with ffs_engines/export_and_build.sh.')
        self.context = self.engine.create_execution_context()

        # Fixed shapes from the engine: left_image (1,3,H,W) -> disparity.
        names = [self.engine.get_tensor_name(i)
                 for i in range(self.engine.num_io_tensors)]
        self._in_names = [n for n in names
                          if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self._out_names = [n for n in names
                           if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if sorted(self._in_names) != ['left_image', 'right_image'] \
                or self._out_names != ['disparity']:
            raise RuntimeError(
                f'unexpected engine bindings {names} in {self.engine_path}; '
                f'expected left_image/right_image -> disparity '
                f'(make_single_onnx.py export)')
        shape = tuple(self.engine.get_tensor_shape('left_image'))
        self.height, self.width = int(shape[2]), int(shape[3])

        # Device buffers, allocated once. IO is float32 on both sides -- the
        # export declares fp32 IO and the fp16 builder flag only changes the
        # INTERNAL precision, but assert it: the byte counts below depend on it.
        for name in names:
            if self.engine.get_tensor_dtype(name) != trt.DataType.FLOAT:
                raise RuntimeError(
                    f'engine tensor {name} is '
                    f'{self.engine.get_tensor_dtype(name)}, expected FLOAT '
                    f'(fp32 IO); this engine was not built by '
                    f'ffs_engines/export_and_build.sh')
        self._check(cudart.cudaSetDevice(0))
        self._buf = {}
        for name in names:
            n = int(np.prod(self.engine.get_tensor_shape(name))) * 4
            err, ptr = cudart.cudaMalloc(n)
            self._check(err)
            self._buf[name] = (ptr, n)
            self.context.set_tensor_address(name, ptr)
        err, self._stream = cudart.cudaStreamCreate()
        self._check(err)
        self._disp_host = np.empty((self.height, self.width), dtype=np.float32)
        self.last_infer_ms = None
        if verbose:
            print(f'[ffs] engine {self.engine_path}: {self.width}x{self.height}, '
                  f'TensorRT {trt.__version__}')

    def _check(self, err):
        # cudart calls return (cudaError_t, ...); cudaSuccess is 0.
        code = err[0] if isinstance(err, tuple) else err
        if int(code) != 0:
            raise RuntimeError(f'CUDA error {code} in ffs_trt')

    def _upload(self, name: str, img_bgr: np.ndarray):
        cudart = self._cudart
        img = img_bgr
        if img.shape[:2] != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # same as fs_depth_batch.py
        norm = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.ascontiguousarray(norm.transpose(2, 0, 1)[None])
        ptr, nbytes = self._buf[name]
        self._check(cudart.cudaMemcpyAsync(
            ptr, chw.ctypes.data, nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self._stream))

    def infer_disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        """Disparity (H_eng, W_eng) float32, in ENGINE-resolution pixels."""
        cudart = self._cudart
        t0 = time.monotonic()
        self._upload('left_image', left_bgr)
        self._upload('right_image', right_bgr)
        if not self.context.execute_async_v3(self._stream):
            raise RuntimeError('TensorRT execute_async_v3 failed')
        ptr, nbytes = self._buf['disparity']
        self._check(cudart.cudaMemcpyAsync(
            self._disp_host.ctypes.data, ptr, self._disp_host.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self._stream))
        self._check(cudart.cudaStreamSynchronize(self._stream))
        self.last_infer_ms = (time.monotonic() - t0) * 1000.0
        return self._disp_host

    def depth_mm(self, left_bgr: np.ndarray, right_bgr: np.ndarray,
                 fx: float, baseline_m: float) -> np.ndarray:
        """RAW depth (H_in, W_in) float32 mm from one rectified BGR pair.

        Raw means: no a/d disparity correction -- the caller applies it.
        Invalid pixels are 0: non-positive disparity, and the left band where
        the match falls outside the right image (x - disp < 0), where the
        model can only hallucinate (the SDK has no depth there either).
        """
        h_in, w_in = left_bgr.shape[:2]
        disp = self.infer_disparity(left_bgr, right_bgr)
        sx = self.width / float(w_in)
        fx_b_mm = (fx * sx) * (baseline_m * 1000.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = fx_b_mm / disp
        xx = np.arange(self.width, dtype=np.float32)[None, :]
        invalid = ~np.isfinite(depth) | (disp <= 0.0) | ((xx - disp) < 0.0)
        depth[invalid] = 0.0
        if (h_in, w_in) != (self.height, self.width):
            depth = cv2.resize(depth, (w_in, h_in),
                               interpolation=cv2.INTER_NEAREST)
        return depth

    def close(self):
        cudart = self._cudart
        for ptr, _ in self._buf.values():
            cudart.cudaFree(ptr)
        self._buf = {}
        cudart.cudaStreamDestroy(self._stream)
