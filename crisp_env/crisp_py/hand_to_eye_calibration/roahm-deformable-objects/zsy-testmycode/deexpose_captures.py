#!/usr/bin/env python3
"""Build a de-exposed copy of an overexposed calibration capture.

WHY THIS EXISTS. zed_calib_004 was captured with the tag overexposed: the white
cells clip to 255 and bloom, and the glossy black cells carry a specular sheen
that lifts them to gray (~60-90). The solver's enhancement cascade
(_make_enhanced_versions) cannot help -- every one of its 11 variants BRIGHTENS
(gamma 0.60/0.50/0.35, alpha/beta boosts, CLAHE), because it was built for the
opposite failure. Measured on 004 with --no-enhance: 15/72 (left) and 28/72
(right) detections, vs 61 and 58 on 003.

WHAT IT DOES. For every calibration_<side>_image_<i>.png it tries the detector
on the ORIGINAL first; if that detects, the frame is copied through
byte-identical. Otherwise it walks a darkening ladder, mildest rung first, and
writes the FIRST rung that detects. Rungs are monotonic pointwise LUTs (gamma,
and black-point stretch + gamma), so they move no edges; only the last-resort
rung adds a 3x3 median. Frames where no rung detects are copied through
unchanged and listed -- on 004 those are poses where the tag is partly out of
frame (e.g. left 20/25), which no photometric fix can recover.

Rung hit-rates measured on 004's missed frames before writing this script:
bp60_g3 rescued 9/13 of the frames that resisted plain gamma; gamma alone
peaked at 2.5. scale-by-0.6 rescued nothing (adaptive threshold is invariant
to pure scaling).

OUTPUT. A sibling sequence directory
    captured_calibration_data/<seq>_deexposed/
        frames/calibration_<side>_image_<i>.png   processed (or copied) frames
        frames/depth_*.png                        symlinks to the originals
        *_calibration_{poses,rgbd}.npz            symlinks to the originals
        deexpose_manifest.json                    which rung each frame used
The original sequence is never touched. Downstream tools take the new name:

    pixi run -e humble python deexpose_captures.py --calib-seq-name zed_calib_004
    pixi run -e humble python dump_tag_detections.py \
        --calib-seq-name zed_calib_004_deexposed --side left --no-enhance
    pixi run -e humble python calculate_base_to_cam_nonlinear_opt.py \
        --camera zed --side left --calib-seq-name zed_calib_004_deexposed \
        --use-depth-translation --no-enhance --rot-weight 573 ...

Keep --no-enhance: every frame that can detect now detects on the stored
pixels, so the brightening cascade is pure cost here too.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

# Same import trick as dump_tag_detections.py: the solver module sets up
# sys.path for apriltag_image and guards main(), so this is side-effect free.
from calculate_base_to_cam_nonlinear_opt import EXPECTED_TAG_ID, TAG_SIZE_M
from apriltag_image import apriltag_image


def _gamma_lut(g):
    return (np.linspace(0.0, 1.0, 256) ** g * 255.0).astype(np.uint8)


def _blackpoint_lut(bp, g):
    x = np.clip((np.arange(256, dtype=np.float64) - bp) / (255.0 - bp), 0.0, 1.0)
    return (x ** g * 255.0).astype(np.uint8)


def _apply(img, lut, median=0):
    out = cv2.LUT(img, lut)
    if median:
        out = cv2.medianBlur(out, median)
    return out


# Mildest first. All rungs are monotonic LUTs (no spatial effect); the median
# rung is last because it is the only one that touches geometry at all.
LADDER = [
    ("gamma2p0", lambda im: _apply(im, _gamma_lut(2.0))),
    ("gamma2p5", lambda im: _apply(im, _gamma_lut(2.5))),
    ("bp60_g2", lambda im: _apply(im, _blackpoint_lut(60, 2.0))),
    ("bp60_g3", lambda im: _apply(im, _blackpoint_lut(60, 3.0))),
    ("gamma4p0", lambda im: _apply(im, _gamma_lut(4.0))),
    ("med3_bp60_g3", lambda im: cv2.medianBlur(_apply(im, _blackpoint_lut(60, 3.0)), 3)),
]


def _detects(png_path):
    """True iff the expected tag is found in the image at png_path."""
    with contextlib.redirect_stdout(io.StringIO()):
        dets = apriltag_image([str(png_path)], output_images=False,
                              display_images=False, tag_size=TAG_SIZE_M,
                              tag_family="tag36h11", camera="zed")
    for j in range(0, len(dets or []), 4):
        if dets[j].tag_id == EXPECTED_TAG_ID:
            return True
    return False


def _symlink(src: Path, dst: Path):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def process_side(src_frames: Path, dst_frames: Path, side: str, max_images: int,
                 tmp: Path):
    manifest = {}
    for i in range(max_images):
        name = f"calibration_{side}_image_{i}.png"
        src = src_frames / name
        if not src.exists():
            continue
        dst = dst_frames / name

        if _detects(src):
            shutil.copy2(src, dst)              # byte-identical pass-through
            manifest[i] = "original"
            print(f"  {side} {i:3d}: original")
            continue

        img = cv2.imread(str(src))
        chosen = None
        for rung_name, fn in LADDER:
            probe = tmp / f"probe_{side}_{i}.png"
            cv2.imwrite(str(probe), fn(img))
            if _detects(probe):
                shutil.move(str(probe), dst)
                chosen = rung_name
                break
        if chosen is None:
            shutil.copy2(src, dst)              # keep the frame; solver skips it
            manifest[i] = "UNRESCUED"
            print(f"  {side} {i:3d}: UNRESCUED (copied unchanged)")
        else:
            manifest[i] = chosen
            print(f"  {side} {i:3d}: {chosen}")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib-seq-name", required=True)
    p.add_argument("--sides", default="left,right")
    p.add_argument("--out-name", default=None,
                   help="default: <calib-seq-name>_deexposed")
    p.add_argument("--max-images", type=int, default=200)
    args = p.parse_args()

    base = Path(__file__).resolve().parents[1] / "captured_calibration_data"
    src = base / args.calib_seq_name
    if not (src / "frames").is_dir():
        raise SystemExit(f"no such sequence: {src / 'frames'}")
    dst = base / (args.out_name or f"{args.calib_seq_name}_deexposed")
    if dst.resolve() == src.resolve():
        raise SystemExit("refusing to overwrite the source sequence")
    (dst / "frames").mkdir(parents=True, exist_ok=True)

    tmp = dst / "frames" / ".probe"
    tmp.mkdir(exist_ok=True)

    # Pose and depth data are untouched by a photometric fix: link, don't copy.
    for f in sorted(src.glob("*.npz")):
        _symlink(f, dst / f.name)
    for f in sorted((src / "frames").glob("depth_*.png")):
        _symlink(f, dst / "frames" / f.name)

    sides = [s.strip() for s in args.sides.split(",") if s.strip()]
    manifest = {"source_sequence": str(src), "tag_id": EXPECTED_TAG_ID,
                "tag_size_m": TAG_SIZE_M, "ladder": [n for n, _ in LADDER],
                "sides": {}}
    for side in sides:
        print(f"--- {side}")
        manifest["sides"][side] = process_side(src / "frames", dst / "frames",
                                               side, args.max_images, tmp)
    shutil.rmtree(tmp, ignore_errors=True)

    (dst / "deexpose_manifest.json").write_text(json.dumps(manifest, indent=2))

    for side in sides:
        m = manifest["sides"][side]
        n = len(m)
        orig = sum(v == "original" for v in m.values())
        bad = sorted(k for k, v in m.items() if v == "UNRESCUED")
        print(f"\n{side}: {n} frames -- {orig} already detected, "
              f"{n - orig - len(bad)} rescued, {len(bad)} unrescued"
              + (f" {bad}" if bad else ""))
    print(f"\nwrote {dst}")
    print(f"solve with:  --calib-seq-name {dst.name}  (keep --no-enhance)")


if __name__ == "__main__":
    main()
