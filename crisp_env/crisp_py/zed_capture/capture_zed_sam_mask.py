#!/usr/bin/env python3
"""
capture_zed_sam_mask.py

Capture one synchronized RGB+depth frame from a ZED stereo camera, optionally
run SAM2/SAM3 interactively to generate a mask, and save files compatible with
cloth-reveal-learning's preprocess_observation().

This is the ZED counterpart of
hand_to_eye_calibration/roahm-deformable-objects/zsy-testmycode/capture_azure_sam_mask.py
and produces the same artifact set, so both feed the same consumers.

Target format (matches cloth-reveal-learning/sac/env.py):
  - rgb.png           uint8 BGR   (640 x 360)   <- training resolution
  - depth_m.npy       float32 m   (640 x 360)   <- preprocess_observation expects metres
  - depth_mm.npy      uint16 mm   (640 x 360)   <- downsampled (INTER_NEAREST)
  - depth_mm_full.npy uint16 mm   (native)      <- full-res raw depth, for 3-D back-projection
  - depth_m_full.npy  float32 m   (native)      <- full-res raw depth (metres)
  - mask.png          uint8 binary (640 x 360)  <- 255=target, 0=bg   (only with --sam-checkpoint)
  - overlay.png       debug visualization                              (only with --sam-checkpoint)
  - depth_vis.png / depth_vis_full.png          turbo colormap + colorbar
  - intrinsics.json   SDK-reported left-cam intrinsics, native + scaled
  - capture_config.json  every setting actually in effect

In all depth artifacts 0 == invalid, matching the Azure convention. ZED's native
NaN / +inf (TOO_FAR) / -inf (TOO_CLOSE) sentinels are normalised away.

Unlike the Azure script, intrinsics are NOT hardcoded -- ZED intrinsics are
per-camera and per-resolution, so they are read from the SDK at the live
resolution and written to intrinsics.json. Back-project depth_m.npy with the
"scaled" block and depth_m_full.npy with the "native" block.

IMPORTANT: a USB ZED allows only ONE process at a time. Close ZED_Depth_Viewer
before running this. Your viewer's saved settings are never modified -- this
script only ever reads ~/.config/StereoLabs/Depth Viewer.conf.

Configuration
-------------
Defaults reproduce the current ZED Depth Viewer setup:

    HD2K @ 15 fps, NEURAL_PLUS, depth 0.20-20.0 m, depth stabilization 10,
    fill mode off, remove saturated areas on, confidence 47, texture 100

HD2K (2208 x 1242) is exactly 16:9, so the downscale to 640 x 360 is a uniform
3.45x with no anisotropy, and the "scaled" intrinsics are just the native set
times 0.289855 on both axes.

Every knob is a CLI flag (see --help, grouped by viewer panel). Pass
--from-viewer-conf to instead seed resolution / fps / depth mode / depth range
from the viewer's own saved file; explicit flags still win over it.

Self-calibration is DISABLED by default (InitParameters.camera_disable_self_calib
= True). Rationale: the SDK otherwise re-estimates the stereo extrinsics at every
open(), and because the rectified intrinsics are *derived* from those extrinsics,
a successful self-calibration can shift fx/cx/cy -- silently invalidating a
hand-eye transform that was solved against different values. Disabling it makes
the intrinsics deterministic and reproducible run to run.

To check whether the camera has mechanically drifted, pass --enable-self-calib
--sdk-verbose 1 while aiming at a bright, textured scene with nothing inside 1 m.
If self-calibration then succeeds and reports different fx/cx/cy than your stored
intrinsics.json, the factory calibration is stale and hand-eye must be re-run.

Note that intrinsics are per-resolution: switching --resolution changes fx, fy,
cx and cy. Always read them from the capture's intrinsics.json rather than
copying numbers between runs at different resolutions.

Usage:
    # inspect what the installed SDK exposes -- opens no camera
    python capture_zed_sam_mask.py --list-config

    # show what the viewer's saved conf would give you -- opens no camera
    python capture_zed_sam_mask.py --show-viewer-conf

    # capture only, using the built-in HD2K/NEURAL_PLUS preset
    python capture_zed_sam_mask.py

    # capture + interactive mask. The checkpoint may be an absolute path, a path
    # relative to your cwd or the repo root, or a bare filename that is looked up
    # in <repo>/sam2/checkpoints/ -- all four resolve.
    python capture_zed_sam_mask.py \\
        --sam-checkpoint sam2/checkpoints/sam2.1_hiera_large.pt
    python capture_zed_sam_mask.py --sam-checkpoint sam2.1_hiera_large.pt

    # override individual viewer settings (resolution stays HD2K @ 15 by default)
    python capture_zed_sam_mask.py --depth-mode NEURAL --depth-max-m 3.0 \\
        --confidence 60 --median-frames 9

    # check for mechanical drift -- the only reason to re-enable self-calibration
    python capture_zed_sam_mask.py --enable-self-calib --sdk-verbose 1

Environment (pyzed lives in the sam2 pixi env after a one-time install):
    cd /home/roahmlab/move_some_robots/crisp_env/crisp_py
    pixi run -e sam2 pip install --no-deps ./pyzed-4.2-cp311-cp311-linux_x86_64.whl
    pixi run -e sam2 python zed_capture/capture_zed_sam_mask.py --list-config

Interactive controls (OpenCV window, only with --sam-checkpoint):
    Left-click     add foreground point  (on the target)
    Right-click    add background point  (table / background)
    Enter / Space  accept current mask and save
    r              reset all points, start over
    q              quit without saving mask
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

import zed_camera as zc


# ============================================================
# SAM2 / SAM3 loader
# ============================================================

def _guard_sam_cwd() -> None:
    """Guard against sam2's parent-directory import trap.

    sam2/build_sam.py raises at import time if cwd is the parent of the sam2
    repo, because the repo dir shadows the installed package. cwd == crisp_py
    root triggers it, so chdir to this script's directory instead.
    """
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    if Path.cwd().resolve() == repo_root and (repo_root / "sam2" / "sam2").is_dir():
        print(f"[sam] cwd {repo_root} shadows the sam2 package -- chdir -> {here}")
        os.chdir(here)


def resolve_sam_checkpoint(checkpoint: str) -> str:
    """Resolve a checkpoint argument to an absolute path.

    MUST be called before _guard_sam_cwd(), because that chdir invalidates any
    relative path the user typed on the command line.

    Args:
        checkpoint: absolute path, or a path relative to the current directory,
                    or a bare filename to look up in <repo>/sam2/checkpoints/.

    Returns:
        Absolute path as a string.

    Raises:
        FileNotFoundError listing the checkpoints that do exist.
    """
    repo_root = Path(__file__).resolve().parent.parent
    ckpt_dir = repo_root / "sam2" / "checkpoints"

    candidates = [Path(checkpoint).expanduser()]
    if not candidates[0].is_absolute():
        candidates.append((Path.cwd() / candidates[0]).resolve())
        candidates.append(repo_root / candidates[0])
        candidates.append(ckpt_dir / candidates[0].name)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    available = sorted(p.name for p in ckpt_dir.glob("*.pt")) if ckpt_dir.is_dir() else []
    lines = [f"SAM checkpoint not found: {checkpoint!r}", "  tried:"]
    lines += [f"    {c}" for c in candidates]
    if available:
        lines.append(f"  available in {ckpt_dir}:")
        lines += [f"    {name}" for name in available]
        lines.append(f"  e.g. --sam-checkpoint {available[0]}")
    else:
        lines.append(f"  no *.pt found in {ckpt_dir}")
    raise FileNotFoundError("\n".join(lines))


def load_sam_predictor(checkpoint: str, config: str, device: str):
    """Load a SAM2 or SAM3 image predictor.

    Args:
        checkpoint: path to a .pt checkpoint. Resolved to an absolute path first,
                    since the cwd guard below would break a relative one.
        config:     hydra config name, e.g. "configs/sam2.1/sam2.1_hiera_l.yaml".
                    Must include the "configs/..." prefix -- sam2 registers its
                    config module at the package root.
        device:     torch device string.

    Returns:
        A predictor exposing set_image() and predict().
    """
    checkpoint = resolve_sam_checkpoint(checkpoint)
    _guard_sam_cwd()

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2(config, checkpoint, device=device)
        predictor = SAM2ImagePredictor(model)
        print(f"[sam] loaded SAM2  config={config}")
        return predictor
    except ImportError:
        pass

    try:
        from sam3.build_sam import build_sam3
        from sam3.sam3_image_predictor import SAM3ImagePredictor
        model = build_sam3(config, checkpoint, device=device)
        predictor = SAM3ImagePredictor(model)
        print(f"[sam] loaded SAM3  config={config}")
        return predictor
    except ImportError:
        pass

    raise ImportError(
        "Neither SAM2 nor SAM3 found.\n"
        "This repo vendors SAM2 in the 'sam2' pixi env:\n"
        "  pixi run -e sam2 python zed_capture/capture_zed_sam_mask.py ...\n"
        "Checkpoints: <repo>/sam2/checkpoints/sam2.1_hiera_large.pt"
    )


# ============================================================
# Interactive mask generation
# ============================================================

def run_interactive_mask(color_bgr: np.ndarray, predictor) -> np.ndarray | None:
    """Display the image; user clicks points to prompt SAM.

    SAM re-runs on every click and shows a live mask preview.

    Args:
        color_bgr: (H, W, 3) uint8 BGR.
        predictor: a SAM image predictor.

    Returns:
        (H, W) uint8 binary mask, 255 = target. None if the user pressed q.
    """
    image_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    fg_pts: list[tuple[int, int]] = []
    bg_pts: list[tuple[int, int]] = []
    current_mask: np.ndarray | None = None

    win_h, win_w = color_bgr.shape[:2]
    WIN = "SAM mask: L-click=target  R-click=bg  Enter=save  r=reset  q=quit"

    def rerun_sam() -> None:
        nonlocal current_mask
        if not fg_pts and not bg_pts:
            current_mask = None
            return
        coords = np.array(fg_pts + bg_pts, dtype=np.float32)
        labels = np.array([1] * len(fg_pts) + [0] * len(bg_pts), dtype=np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        current_mask = (masks[best] > 0).astype(np.uint8) * 255

    def render_display() -> np.ndarray:
        vis = color_bgr.copy()
        if current_mask is not None:
            overlay = np.zeros_like(vis)
            overlay[:, :, 1] = current_mask  # green tint
            vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        for (x, y) in fg_pts:
            cv2.circle(vis, (x, y), 7, (0, 255, 0), -1)
            cv2.circle(vis, (x, y), 7, (0, 0, 0), 2)
        for (x, y) in bg_pts:
            cv2.circle(vis, (x, y), 7, (0, 0, 255), -1)
            cv2.circle(vis, (x, y), 7, (0, 0, 0), 2)
        n_target = int((current_mask > 0).sum()) if current_mask is not None else 0
        info = f"fg={len(fg_pts)} bg={len(bg_pts)} target_px={n_target}"
        cv2.putText(vis, info, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(vis, info, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return vis

    def on_mouse(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            fg_pts.append((x, y))
            rerun_sam()
            cv2.imshow(WIN, render_display())
        elif event == cv2.EVENT_RBUTTONDOWN:
            bg_pts.append((x, y))
            rerun_sam()
            cv2.imshow(WIN, render_display())

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, win_w * 2, win_h * 2)
    cv2.setMouseCallback(WIN, on_mouse)
    cv2.imshow(WIN, render_display())
    print("[sam] click on the target (L=target, R=background). Enter=accept, r=reset, q=quit.")

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):  # Enter or Space
            if current_mask is None:
                print("[sam] no mask yet -- click on the target first.")
            else:
                break
        elif key == ord("r"):
            fg_pts.clear()
            bg_pts.clear()
            current_mask = None
            cv2.imshow(WIN, render_display())
            print("[sam] reset points.")
        elif key == ord("q"):
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return current_mask


# ============================================================
# Saving
# ============================================================

def save_capture(
    out_dir: Path,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    depth_mm: np.ndarray,
    color_full: np.ndarray,
    depth_m_full: np.ndarray,
    depth_mm_full: np.ndarray,
    intrinsics: dict,
    capture_config: dict,
    save_full_res: bool = True,
) -> dict[str, Path]:
    """Write every non-mask artifact.

    Args:
        out_dir:        run directory; created if absent.
        color_bgr:      (360, 640, 3) uint8 BGR
        depth_m:        (360, 640) float32 metres, 0 = invalid
        depth_mm:       (360, 640) uint16 mm
        color_full:     native-resolution BGR
        depth_m_full:   native-resolution float32 metres
        depth_mm_full:  native-resolution uint16 mm
        intrinsics:     get_intrinsics() output
        capture_config: everything that was in effect, for reproducibility
        save_full_res:  write the native-resolution arrays too

    Returns:
        {name: path} for everything written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def _record(name: str, path: Path, note: str) -> None:
        written[name] = path
        print(f"[saved] {path}  {note}")

    # --- training resolution -------------------------------------------
    path = out_dir / "rgb.png"
    cv2.imwrite(str(path), color_bgr)
    _record("rgb", path, f"shape={color_bgr.shape} dtype=uint8 BGR")

    path = out_dir / "depth_m.npy"
    np.save(str(path), depth_m)
    _record("depth_m", path, f"shape={depth_m.shape} dtype=float32 unit=metres 0=invalid")

    path = out_dir / "depth_mm.npy"
    np.save(str(path), depth_mm)
    _record("depth_mm", path, f"shape={depth_mm.shape} dtype=uint16 unit=mm 0=invalid")

    path = out_dir / "depth_vis.png"
    cv2.imwrite(str(path), zc.depth_to_vis(depth_mm))
    _record("depth_vis", path, "turbo + colorbar")

    # --- native resolution ---------------------------------------------
    if save_full_res:
        path = out_dir / "rgb_full.png"
        cv2.imwrite(str(path), color_full)
        _record("rgb_full", path, f"shape={color_full.shape} dtype=uint8 BGR (native)")

        path = out_dir / "depth_m_full.npy"
        np.save(str(path), depth_m_full)
        _record("depth_m_full", path,
                f"shape={depth_m_full.shape} dtype=float32 unit=metres (native)")

        path = out_dir / "depth_mm_full.npy"
        np.save(str(path), depth_mm_full)
        _record("depth_mm_full", path,
                f"shape={depth_mm_full.shape} dtype=uint16 unit=mm (native)")

        path = out_dir / "depth_vis_full.png"
        cv2.imwrite(str(path), zc.depth_to_vis(depth_mm_full))
        _record("depth_vis_full", path, "turbo + colorbar (native)")

    # --- metadata -------------------------------------------------------
    path = out_dir / "intrinsics.json"
    path.write_text(json.dumps(intrinsics, indent=2) + "\n")
    _record("intrinsics", path, "SDK left-cam intrinsics, native + scaled")

    path = out_dir / "capture_config.json"
    path.write_text(json.dumps(capture_config, indent=2) + "\n")
    _record("capture_config", path, "settings actually in effect")

    return written


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ZED stereo RGB-D capture + optional SAM mask "
                    "(cloth-reveal-learning format)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Camera / InitParameters flags default to None so --from-viewer-conf can
    # fill them in, with explicit flags always winning.
    cam = parser.add_argument_group(
        "camera  (viewer: input panel)",
        "Defaults: HD2K @ 15 fps. Run --list-config for what this SDK supports.",
    )
    cam.add_argument("--resolution", default=None,
                     help="HD2K | HD1080 | HD720 | VGA | AUTO  (default: HD2K)")
    cam.add_argument("--fps", type=int, default=None,
                     help="frames per second; HD2K supports 15 only (default: 15)")
    cam.add_argument("--serial", type=int, default=None,
                     help="open a specific camera by serial number")
    cam.add_argument("--flip", default=None, help="OFF | ON | AUTO (default: OFF)")
    cam.add_argument("--enable-self-calib", action="store_true",
                     help="re-enable ZED self-calibration (OFF by default). Only for "
                          "checking whether the camera has mechanically drifted -- aim at "
                          "a bright textured scene with nothing inside 1 m and add "
                          "--sdk-verbose 1. Enabling it can shift the rectified intrinsics "
                          "and invalidate a stored hand-eye transform.")
    cam.add_argument("--no-image-enhancement", action="store_true",
                     help="disable the ISP contrast enhancement")
    cam.add_argument("--sdk-verbose", type=int, default=0, help="pyzed verbosity")

    depth = parser.add_argument_group(
        "depth  (viewer: Settings > Processing > Depth)",
        "Defaults mirror the current viewer setup: NEURAL_PLUS, 0.20-20.0 m, "
        "stabilization 10, fill off, saturated removal on, confidence 47, texture 100.",
    )
    depth.add_argument("--depth-mode", default=None,
                       help="PERFORMANCE | QUALITY | ULTRA | NEURAL | NEURAL_PLUS "
                            "(default: NEURAL_PLUS)")
    depth.add_argument("--depth-min-m", type=float, default=None,
                       help="viewer DEPTH MIN., metres (default: 0.20)")
    depth.add_argument("--depth-max-m", type=float, default=None,
                       help="viewer DEPTH MAX., metres (default: 20.0)")
    depth.add_argument("--depth-stabilization", type=int, default=None,
                       help="viewer DEPTH STABILIZATION %%, 0-100; non-zero enables "
                            "positional tracking (default: 10)")
    depth.add_argument("--confidence", type=int, default=None,
                       help="viewer CONFIDENCE %%, 1-100, lower = stricter (default: 47)")
    depth.add_argument("--texture-confidence", type=int, default=None,
                       help="viewer TEXTURE CONFIDENCE %%, 1-100 (default: 100)")
    depth.add_argument("--fill-mode", action="store_true",
                       help="viewer ENABLE FILL MODE; overrides both confidence "
                            "thresholds and saturated-area removal")
    depth.add_argument("--keep-saturated", action="store_true",
                       help="disable viewer REMOVE SATURATED AREAS")

    img = parser.add_argument_group(
        "image controls  (viewer: camera-control sliders)",
        "Unset flags leave the camera at whatever it is already using.",
    )
    img.add_argument("--brightness", type=int, default=None, help="0-8")
    img.add_argument("--contrast", type=int, default=None, help="0-8")
    img.add_argument("--hue", type=int, default=None, help="0-11")
    img.add_argument("--saturation", type=int, default=None, help="0-8")
    img.add_argument("--sharpness", type=int, default=None, help="0-8")
    img.add_argument("--gamma", type=int, default=None, help="1-9")
    img.add_argument("--gain", type=int, default=None, help="0-100; also clears auto exposure")
    img.add_argument("--exposure", type=int, default=None, help="0-100; also clears auto exposure")
    img.add_argument("--aec-agc", type=int, default=None,
                     help="1 = auto exposure/gain, 0 = manual")
    img.add_argument("--wb-temperature", "--wb-temp", dest="wb_temperature",
                     type=int, default=None,
                     help="white balance 2800-6500 K; also clears auto white balance")
    img.add_argument("--wb-auto", type=int, default=None, help="0 or 1")
    img.add_argument("--led-status", "--led", dest="led_status", type=int, default=None,
                     help="front LED, 0 or 1")

    cap = parser.add_argument_group("capture")
    cap.add_argument("--median-frames", type=int, default=zc._MEDIAN_FRAMES,
                     help="per-pixel median over N grabs; 1 = single shot")
    cap.add_argument("--warmup-frames", type=int, default=zc._WARMUP_FRAMES,
                     help="grabs discarded so exposure and depth stabilization settle")
    cap.add_argument("--from-viewer-conf", action="store_true",
                     help="seed resolution/fps/depth-mode/depth-range from the ZED Depth "
                          "Viewer's saved conf (read-only; explicit flags still win)")
    cap.add_argument("--d", "--disparity-offset-px", dest="disparity_offset_px",
                     type=float, default=zc.DEPTH_DISPARITY_OFFSET_PX,
                     help="disparity offset d (model: disp_true = a*disp + d), which "
                          "makes depth read too far. Default: the value in "
                          "zed_capture/zed_depth_correction.json (currently "
                          f"{zc.DEPTH_DISPARITY_OFFSET_PX:.2f} px), which the calibration "
                          "solvers also read. Pass 0 to disable -- do that after you "
                          "recalibrate the camera, or the depth goes wrong the other way.")
    cap.add_argument("--a", "--disparity-scale", dest="disparity_scale",
                     type=float, default=zc.DEPTH_DISPARITY_SCALE,
                     help="disparity scale a, dimensionless. Default: the value in "
                          "zed_capture/zed_depth_correction.json (currently "
                          f"{zc.DEPTH_DISPARITY_SCALE:.4f}). Pass 1 to disable.")

    out = parser.add_argument_group("output")
    out.add_argument("--output", default=None,
                     help="run directory (default: <script_dir>/real_captures/run_<timestamp>)")
    out.add_argument("--out-width", type=int, default=zc.TRAIN_W, help="training width")
    out.add_argument("--out-height", type=int, default=zc.TRAIN_H, help="training height")
    out.add_argument("--no-full-res", action="store_true",
                     help="skip the native-resolution artifacts")

    sam = parser.add_argument_group("SAM mask  (optional)")
    sam.add_argument("--sam-checkpoint", default=None,
                     help="path to a SAM2/SAM3 .pt checkpoint. Omit to skip masking.")
    sam.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml",
                     help="SAM hydra config name; keep the configs/... prefix")
    sam.add_argument("--device", default="cuda", help="torch device for SAM")

    info = parser.add_argument_group("introspection  (open no camera, then exit)")
    info.add_argument("--list-config", action="store_true",
                      help="print the config surface of the installed SDK and exit")
    info.add_argument("--show-viewer-conf", action="store_true",
                      help="print the viewer's saved conf and what it maps to, then exit")

    return parser.parse_args()


def build_configs(
    args: argparse.Namespace,
) -> tuple[zc.ZedInitConfig, zc.ZedRuntimeConfig, zc.ZedImageConfig]:
    """Layer config sources: script defaults, then viewer conf, then CLI flags.

    Returns:
        (init_cfg, runtime_cfg, image_cfg)
    """
    init_cfg = zc.ZedInitConfig()
    runtime_cfg = zc.ZedRuntimeConfig()

    if args.from_viewer_conf:
        init_cfg, runtime_cfg = zc.viewer_conf_to_configs(
            zc.read_viewer_conf(), init_cfg, runtime_cfg
        )

    # Explicit flags win over both defaults and the viewer conf.
    for attr, flag in (
        ("resolution", "resolution"),
        ("fps", "fps"),
        ("depth_mode", "depth_mode"),
        ("depth_min_m", "depth_min_m"),
        ("depth_max_m", "depth_max_m"),
        ("depth_stabilization", "depth_stabilization"),
        ("flip", "flip"),
    ):
        value = getattr(args, flag)
        if value is not None:
            setattr(init_cfg, attr, value)

    init_cfg.serial_number = args.serial
    # Self-calibration is off unless explicitly asked for.
    init_cfg.disable_self_calib = not bool(args.enable_self_calib)
    init_cfg.image_enhancement = not args.no_image_enhancement
    init_cfg.sdk_verbose = int(args.sdk_verbose)

    if args.confidence is not None:
        runtime_cfg.confidence = args.confidence
    if args.texture_confidence is not None:
        runtime_cfg.texture_confidence = args.texture_confidence
    runtime_cfg.fill_mode = bool(args.fill_mode)
    runtime_cfg.remove_saturated = not args.keep_saturated

    image_cfg = zc.ZedImageConfig(
        brightness=args.brightness,
        contrast=args.contrast,
        hue=args.hue,
        saturation=args.saturation,
        sharpness=args.sharpness,
        gamma=args.gamma,
        gain=args.gain,
        exposure=args.exposure,
        aec_agc=args.aec_agc,
        wb_temperature=args.wb_temperature,
        wb_auto=args.wb_auto,
        led_status=args.led_status,
    )
    return init_cfg, runtime_cfg, image_cfg


def main() -> None:
    args = parse_args()

    # ---- Introspection paths: no camera opened ---------------------------
    if args.list_config:
        zc.print_sdk_config_surface()
        return

    if args.show_viewer_conf:
        conf = zc.read_viewer_conf()
        print(f"[zed] {zc.VIEWER_CONF_PATH} (read-only):")
        for key, value in sorted(conf.items()):
            print(f"  {key} = {value}")
        init_cfg, runtime_cfg = zc.viewer_conf_to_configs(conf)
        print("\nmaps to:")
        print(f"  init    = {init_cfg}")
        print(f"  runtime = {runtime_cfg}")
        print("\nThe viewer does not persist confidence / texture confidence / fill mode,")
        print("so those keep this script's defaults.")
        return

    init_cfg, runtime_cfg, image_cfg = build_configs(args)

    # Resolve the checkpoint now, while the cwd is still the user's: the sam2 cwd
    # guard chdirs later and would invalidate a relative path. Doing it here also
    # means a bad path fails before we spend a capture on it.
    if args.sam_checkpoint:
        args.sam_checkpoint = resolve_sam_checkpoint(args.sam_checkpoint)
        print(f"[sam] checkpoint: {args.sam_checkpoint}")

    # Absolute: the sam2 cwd guard chdirs before mask.png is written, so a
    # relative --output would otherwise land somewhere unexpected.
    out_dir = (
        Path(args.output).expanduser().resolve() if args.output
        else Path(__file__).resolve().parent / "real_captures" / time.strftime("run_%Y%m%d_%H%M%S")
    )

    # ---- Step 1: open the camera and capture -----------------------------
    zed = None
    try:
        zed, resolved = zc.open_zed(init_cfg, image_cfg)
        runtime = zc.build_runtime_parameters(runtime_cfg)

        zc.warmup(zed, runtime, args.warmup_frames)

        print(f"[capture] grabbing (median over {args.median_frames} frame(s))...")
        color_full, depth_m_full, depth_mm_full = zc.capture_rgbd_native(
            zed, runtime, n_median=args.median_frames,
            disparity_offset_px=args.disparity_offset_px,
            disparity_scale=args.disparity_scale,
        )

        intrinsics = zc.get_intrinsics(zed, target_wh=(args.out_width, args.out_height))
    finally:
        zc.close_zed(zed)
        print("[zed] camera closed.")

    # ---- Step 2: downscale to training resolution ------------------------
    color_bgr, depth_m, depth_mm = zc.downscale_to_training_res(
        color_full, depth_m_full, args.out_width, args.out_height
    )
    native_h, native_w = depth_m_full.shape[:2]
    print(f"[resize] {native_w}x{native_h} -> {args.out_width}x{args.out_height}")

    scaled = intrinsics.get("scaled", {})
    if "aspect_warning" in scaled:
        print(f"[resize] WARN: {scaled['aspect_warning']}")

    valid = depth_m > 0
    if valid.any():
        print(f"[depth]  valid range: {depth_m[valid].min():.3f} - {depth_m[valid].max():.3f} m "
              f"({100.0 * valid.mean():.1f}% of pixels)")
    else:
        print("[depth]  WARN: no valid depth after downscaling")

    # ---- Step 3: save ----------------------------------------------------
    capture_config = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "camera": resolved,
        "init": asdict(init_cfg),
        "runtime": asdict(runtime_cfg),
        "image_requested": {k: v for k, v in asdict(image_cfg).items() if v is not None},
        "capture": {
            "median_frames": int(args.median_frames),
            "warmup_frames": int(args.warmup_frames),
            "from_viewer_conf": bool(args.from_viewer_conf),
            "disparity_offset_px": float(args.disparity_offset_px),
            "disparity_scale": float(args.disparity_scale),
            "native_width": int(native_w),
            "native_height": int(native_h),
            "out_width": int(args.out_width),
            "out_height": int(args.out_height),
        },
        "conventions": {
            "depth_invalid": "0 (ZED NaN / +inf TOO_FAR / -inf TOO_CLOSE normalised to 0)",
            "depth_units": "metres in depth_m*.npy, millimetres in depth_mm*.npy",
            "rgb_order": "BGR as written by cv2.imwrite; PIL .convert('RGB') reads it back correctly",
            "coordinate_system": "sl.COORDINATE_SYSTEM.IMAGE (X right, Y down, Z forward)",
            "depth_registration": "already aligned to VIEW.LEFT; no RGB-depth alignment applied",
            "depth_disparity_correction":
                (f"disparity a={args.disparity_scale:.4f}, "
                 f"d={args.disparity_offset_px:+.2f} px removed: "
                 "z = fx*B/(a * fx*B/z_reported + d). Set --d 0 --a 1 "
                 "after recalibrating the camera.")
                if (args.disparity_offset_px or args.disparity_scale != 1.0)
                else "none (raw SDK depth)",
        },
    }

    save_capture(
        out_dir,
        color_bgr, depth_m, depth_mm,
        color_full, depth_m_full, depth_mm_full,
        intrinsics, capture_config,
        save_full_res=not args.no_full_res,
    )

    # ---- Step 4: optional SAM mask ---------------------------------------
    mask = None
    if args.sam_checkpoint:
        predictor = load_sam_predictor(args.sam_checkpoint, args.sam_config, args.device)
        mask = run_interactive_mask(color_bgr, predictor)

        if mask is None:
            print("[sam] user quit -- mask not saved.")
        else:
            mask_path = out_dir / "mask.png"
            cv2.imwrite(str(mask_path), mask)
            target_px = int((mask > 0).sum())
            print(f"[saved] {mask_path}  target_px={target_px} "
                  f"({100.0 * target_px / mask.size:.1f}%)")

            green_layer = np.zeros_like(color_bgr)
            green_layer[:, :, 1] = mask
            overlay = cv2.addWeighted(color_bgr, 0.55, green_layer, 0.45, 0)
            overlay_path = out_dir / "overlay.png"
            cv2.imwrite(str(overlay_path), overlay)
            print(f"[saved] {overlay_path}")
    else:
        print("[sam] skipped (no --sam-checkpoint). "
              "Downstream consumers need mask.png -- rerun with a checkpoint "
              "or write one yourself.")

    # ---- Summary ---------------------------------------------------------
    native = intrinsics["native"]
    print()
    print("=" * 60)
    print(f"Output: {out_dir}")
    print()
    print(f"Training-resolution outputs ({args.out_width} x {args.out_height}):")
    print("  rgb.png        -- uint8 BGR, for render['rgb']")
    print("  depth_m.npy    -- float32 metres, 0=invalid, for render['depth']")
    print("  depth_mm.npy   -- uint16 mm, downsampled (INTER_NEAREST)")
    if mask is not None:
        print("  mask.png       -- uint8 binary (255=target), for render['target_mask']")
        print("  overlay.png    -- visual check")
    else:
        print("  mask.png       -- NOT WRITTEN (no SAM checkpoint)")
    print()
    if not args.no_full_res:
        print(f"Native-resolution raw depth ({native_h} x {native_w}) -- for 3-D back-projection:")
        print("  rgb_full.png / depth_m_full.npy / depth_mm_full.npy / depth_vis_full.png")
        print()
    print(f"Intrinsics at {args.out_width}x{args.out_height} "
          f"(back-project depth_m.npy / depth_mm.npy):")
    for key in ("fx", "fy", "cx", "cy"):
        print(f"  {key} = {scaled[key]:.6f}")
    print(f"  scale from native: sx={scaled['scale_x']:.6f} sy={scaled['scale_y']:.6f}")
    print()
    print(f"Intrinsics at {native['width']}x{native['height']} "
          f"(back-project depth_*_full.npy):")
    for key in ("fx", "fy", "cx", "cy"):
        print(f"  {key} = {native[key]:.6f}")
    print()
    print("These are per-camera and per-resolution -- do NOT hardcode them.")
    print("Both sets are in intrinsics.json alongside the full K matrices.")
    print()
    print("To run inference with the trained actor:")
    print("  import sys; sys.path.insert(0, 'cloth-reveal-learning')")
    print("  import numpy as np, cv2")
    print("  from sac.env import preprocess_observation")
    print("  from project_config import CFG")
    print("  render = {")
    print(f"      'rgb':         cv2.imread('{out_dir / 'rgb.png'}'),")
    print(f"      'depth':       np.load('{out_dir / 'depth_m.npy'}'),")
    print(f"      'target_mask': cv2.imread('{out_dir / 'mask.png'}', 0) > 0,")
    print("  }")
    print("  obs = preprocess_observation(render, CFG, rng=None, augment=False)")
    print("  # obs.shape == (5, 64, 64)  float32")
    print("=" * 60)


if __name__ == "__main__":
    main()
