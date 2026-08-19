#!/usr/bin/env python3
"""Generate printable checkerboard PDFs for the ZED intrinsics calibration.

Vector PDF at exact physical size: print at 100% / "Actual size" (never
"fit to page"), then VERIFY with the printed scale bar before mounting --
printers silently rescale, and x/y can even scale differently.

Outputs into checkerboard_patterns/:
    checkerboard_A4_10x7_25mm.pdf   (9x6 inner corners, 25 mm squares)
    checkerboard_A3_10x7_35mm.pdf   (9x6 inner corners, 35 mm squares)

The 10x7 layout is asymmetric (even x odd) on purpose: it makes the board's
orientation unambiguous to the detector, so no view can be folded onto
another by a 180-degree flip.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

MM = 1.0 / 25.4                      # mm -> inch

OUT = Path(__file__).resolve().parent / 'checkerboard_patterns'


def make_board(page_w_mm, page_h_mm, cols, rows, square_mm, out_pdf):
    fig = plt.figure(figsize=(page_w_mm * MM, page_h_mm * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w_mm)
    ax.set_ylim(0, page_h_mm)
    ax.set_aspect('equal')
    ax.axis('off')

    bw, bh = cols * square_mm, rows * square_mm
    x0, y0 = (page_w_mm - bw) / 2.0, (page_h_mm - bh) / 2.0
    if x0 < 8 or y0 < 8:
        raise SystemExit(f'{out_pdf.name}: board {bw}x{bh} mm leaves less than an '
                         f'8 mm quiet border on the page -- shrink the squares')

    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                ax.add_patch(Rectangle((x0 + c * square_mm, y0 + r * square_mm),
                                       square_mm, square_mm,
                                       facecolor='black', edgecolor='none'))

    # 200 mm scale bar in the bottom margin: verify the print BEFORE mounting.
    bar_y = y0 / 2.0
    bar_x0 = (page_w_mm - 200.0) / 2.0
    ax.plot([bar_x0, bar_x0 + 200.0], [bar_y, bar_y], 'k-', lw=0.8)
    for x in (bar_x0, bar_x0 + 200.0):
        ax.plot([x, x], [bar_y - 1.5, bar_y + 1.5], 'k-', lw=0.8)
    ax.text(page_w_mm / 2.0, bar_y + 2.5,
            'scale bar: exactly 200.0 mm tick-to-tick -- measure after printing',
            ha='center', va='bottom', fontsize=6)

    ax.text(x0, y0 + bh + 3,
            f'{cols}x{rows} squares ({cols-1}x{rows-1} inner corners), '
            f'{square_mm:.1f} mm nominal -- MEASURE a run of squares and pass '
            f'the true value as --square-mm',
            ha='left', va='bottom', fontsize=6)

    OUT.mkdir(exist_ok=True)
    fig.savefig(out_pdf, format='pdf')
    plt.close(fig)
    print(f'  wrote {out_pdf}  (board {bw:.0f} x {bh:.0f} mm on '
          f'{page_w_mm:.0f} x {page_h_mm:.0f} mm page)')


if __name__ == '__main__':
    # A4 landscape 297 x 210
    make_board(297.0, 210.0, 10, 7, 25.0, OUT / 'checkerboard_A4_10x7_25mm.pdf')
    # A3 landscape 420 x 297
    make_board(420.0, 297.0, 10, 7, 35.0, OUT / 'checkerboard_A3_10x7_35mm.pdf')
