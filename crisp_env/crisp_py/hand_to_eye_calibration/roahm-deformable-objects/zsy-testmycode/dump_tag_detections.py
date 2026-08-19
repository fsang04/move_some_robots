#!/usr/bin/env python3
"""Dump AprilTag centre detections for visual inspection -- no solving.

Runs the SAME detector the calibration solver runs
(_detect_with_enhancement_until_success, so the same enhancement ladder, the
same tag family, the same EXPECTED_TAG_ID) over every frame of a captured
sequence, and writes out what it found. Nothing here optimises anything, so it
is safe to run before committing to a solve.

Two outputs per side:

  detections_<side>/frame_<i>.png   full frame, tag outlined, centre crosshaired,
                                    with a zoomed inset because the tag is ~100 px
                                    wide in a 2208 px image
  detections_<side>/montage.png     every frame's tag crop in one grid -- the fast
                                    way to eyeball 60 detections for a bad one

plus detections_<side>/centres.npz, holding image_index, centre_uv and corners.
That npz is exactly the 2D half of the centre-only PnP formulation, so this
doubles as the data-prep step for that experiment.

    python dump_tag_detections.py --calib-seq-name zed_calib_003 --side right --no-enhance

--no-enhance matches what you would pass to the solver: it tries only the
original image. Leave it off to see which frames need the enhancement ladder.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# calculate_base_to_cam_nonlinear_opt sets up its own sys.path for apriltag_image
# and zed_capture, and guards main() behind __name__, so importing it is cheap
# and side-effect free.
from calculate_base_to_cam_nonlinear_opt import (
    EXPECTED_TAG_ID,
    TAG_SIZE_M,
    _detect_with_enhancement_until_success,
)

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def annotate(img, det, image_index, source_name):
    """Full frame with the tag outlined, corners numbered, centre crosshaired."""
    img = img.copy()
    corners = np.asarray(det.corners, dtype=float)
    centre = np.asarray(det.center, dtype=float)

    cv2.polylines(img, [corners.astype(np.int32).reshape(-1, 1, 2)], True, WHITE, 3)
    for k, c in enumerate(corners):
        cv2.circle(img, tuple(c.astype(int)), 6, GREEN, -1)
        cv2.putText(img, str(k), tuple((c + 12).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2, cv2.LINE_AA)

    cu, cv_ = int(round(centre[0])), int(round(centre[1]))
    cv2.line(img, (cu - 25, cv_), (cu + 25, cv_), RED, 2)
    cv2.line(img, (cu, cv_ - 25), (cu, cv_ + 25), RED, 2)
    cv2.circle(img, (cu, cv_), 4, RED, -1)

    # Zoomed inset: at 2208 px wide the tag is a postage stamp, and the whole
    # point of this dump is judging a sub-pixel centre by eye.
    crop = tag_crop(img, corners, pad_scale=1.2)
    if crop is not None:
        h_img, w_img = img.shape[:2]
        scale = min(4.0, (w_img / 3.0) / crop.shape[1], (h_img / 2.0) / crop.shape[0])
        if scale > 1.0:
            zoom = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_NEAREST)
            zh, zw = zoom.shape[:2]
            ox, oy = w_img - zw - 20, 20
            img[oy:oy + zh, ox:ox + zw] = zoom
            cv2.rectangle(img, (ox - 2, oy - 2), (ox + zw + 2, oy + zh + 2), WHITE, 3)

    lines = [
        f'image {image_index}   source={source_name}   tag_id={det.tag_id}',
        f'centre (u,v) = ({centre[0]:.2f}, {centre[1]:.2f}) px',
        f'tag size {TAG_SIZE_M*1000:.0f} mm   family tag36h11',
    ]
    for i, text in enumerate(lines):
        y = 45 + i * 40
        cv2.putText(img, text, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6,
                    cv2.LINE_AA)
        cv2.putText(img, text, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2,
                    cv2.LINE_AA)
    return img


def tag_crop(img, corners, pad_scale=1.2):
    """Square-ish crop around the tag, or None if it lands off-frame."""
    h, w = img.shape[:2]
    pad = max(40.0, pad_scale * float(np.ptp(corners, axis=0).max()))
    x0, x1 = int(max(0, corners[:, 0].min() - pad)), int(min(w, corners[:, 0].max() + pad))
    y0, y1 = int(max(0, corners[:, 1].min() - pad)), int(min(h, corners[:, 1].max() + pad))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    return img[y0:y1, x0:x1]


def build_montage(tiles, labels, cols=8, tile=220):
    """One grid of every tag crop, so a bad frame stands out at a glance."""
    if not tiles:
        return None
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.zeros((rows * tile, cols * tile, 3), dtype=np.uint8)
    for i, (t, label) in enumerate(zip(tiles, labels)):
        t = cv2.resize(t, (tile, tile), interpolation=cv2.INTER_AREA)
        cv2.putText(t, label, (6, tile - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(t, label, (6, tile - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    WHITE, 1, cv2.LINE_AA)
        r, c = divmod(i, cols)
        sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = t
        cv2.rectangle(sheet, (c * tile, r * tile), ((c + 1) * tile - 1, (r + 1) * tile - 1),
                      (60, 60, 60), 1)
    return sheet


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--calib-seq-name', required=True)
    p.add_argument('--side', required=True, choices=['left', 'right'])
    p.add_argument('--camera', default='zed', choices=['zed', 'azure'])
    p.add_argument('--max-images', type=int, default=200)
    p.add_argument('--no-enhance', action='store_true',
                   help='try only the original image, as the solver does with --no-enhance')
    p.add_argument('--out', default=None, help='default: <seq>/detections_<side>')
    p.add_argument('--montage-cols', type=int, default=8)
    args = p.parse_args()

    base = (Path(__file__).resolve().parents[1] / 'captured_calibration_data'
            / args.calib_seq_name)
    frames = base / 'frames'
    if not frames.is_dir():
        raise SystemExit(f'no such sequence: {frames}')
    out = Path(args.out) if args.out else base / f'detections_{args.side}'
    out.mkdir(parents=True, exist_ok=True)
    enhanced_dir = frames / 'dump_enhanced_debug'

    print(f'sequence : {base}')
    print(f'side     : {args.side}   tag_id={EXPECTED_TAG_ID}  size={TAG_SIZE_M*1000:.0f} mm')
    print(f'enhance  : {not args.no_enhance}')
    print(f'output   : {out}\n')

    idx, centres, corners_all, sources = [], [], [], []
    tiles, labels, missing = [], [], []

    for i in range(args.max_images):
        img_path = frames / f'calibration_{args.side}_image_{i}.png'
        if not img_path.exists():
            continue
        det, T, source_name, used_path = _detect_with_enhancement_until_success(
            img_path=img_path, enhanced_dir=enhanced_dir, image_index=i,
            camera=args.camera, enhance=not args.no_enhance)

        if det is None:
            missing.append(i)
            print(f'  image {i:3d}: NO DETECTION')
            continue

        img = cv2.imread(str(used_path))
        centre = np.asarray(det.center, dtype=float)
        cs = np.asarray(det.corners, dtype=float)

        cv2.imwrite(str(out / f'frame_{i:03d}.png'),
                    annotate(img, det, i, source_name))
        crop = tag_crop(img, cs, pad_scale=0.9)
        if crop is not None:
            c2 = crop.copy()
            # mark the centre inside the crop too -- the montage is the thing
            # people actually scan, so it has to carry the centre as well
            cc = centre - np.array([max(0, cs[:, 0].min() - max(40.0, 0.9 * float(np.ptp(cs, axis=0).max()))),
                                    max(0, cs[:, 1].min() - max(40.0, 0.9 * float(np.ptp(cs, axis=0).max())))])
            cu, cvv = int(round(cc[0])), int(round(cc[1]))
            if 0 <= cu < c2.shape[1] and 0 <= cvv < c2.shape[0]:
                cv2.line(c2, (cu - 14, cvv), (cu + 14, cvv), RED, 2)
                cv2.line(c2, (cu, cvv - 14), (cu, cvv + 14), RED, 2)
            tiles.append(c2)
            labels.append(f'{i}')

        idx.append(i)
        centres.append(centre)
        corners_all.append(cs)
        sources.append(source_name)
        edge = float(np.mean([np.linalg.norm(cs[k] - cs[(k + 1) % 4]) for k in range(4)]))
        print(f'  image {i:3d}: centre=({centre[0]:8.2f},{centre[1]:8.2f}) px  '
              f'mean edge={edge:6.1f} px  source={source_name}')

    if not idx:
        raise SystemExit('no detections at all -- wrong side, sequence, or tag id?')

    sheet = build_montage(tiles, labels, cols=args.montage_cols)
    if sheet is not None:
        cv2.imwrite(str(out / 'montage.png'), sheet)

    centres = np.asarray(centres)
    np.savez(out / 'centres.npz', image_index=np.asarray(idx), centre_uv=centres,
             corners=np.asarray(corners_all),
             source_name=np.asarray(sources, dtype=object), allow_pickle=True)

    print(f'\n  detected {len(idx)} frames, missed {len(missing)}'
          + (f' {missing}' if missing else ''))
    print(f'  centre u range {centres[:,0].min():7.1f} .. {centres[:,0].max():7.1f} px')
    print(f'  centre v range {centres[:,1].min():7.1f} .. {centres[:,1].max():7.1f} px')
    print(f'\n  wrote {len(idx)} annotated frames + montage.png + centres.npz to\n    {out}')


if __name__ == '__main__':
    main()
