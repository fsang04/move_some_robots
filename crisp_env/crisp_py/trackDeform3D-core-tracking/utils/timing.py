"""Per-stage timing accumulation for the tracking pipelines.

The trackers already measure each pipeline stage and return the numbers in
``result['timing']`` (see ``WireTracker.track`` / ``WireTracker.process_frame``);
this module collects those per-frame dicts and renders them as a report.

Stage keys for the wire trackers (DLO / BDLO), in pipeline order:

    segmentation    depth -> foreground mask -> skeleton -> point cloud
    node_detection  branch / leaf detection on the skeleton graph
    cpd             CPD registration from the previous keypoints
    hungarian       anchor re-identification (0.0 when ablated off)
    ee_injection    EE pose injection (0.0 for DLO: leaves keep tracked pos)
    geometry        Gauss-Seidel edge-length + projection optimization
    total           whole per-frame ``track()`` call

Note on ``total``: the tracker stops that timer at the end of ``track()``, which
is *before* ``process_frame`` injects ``segmentation`` -- so ``total`` covers
stages 2-6 only. The report therefore uses ``frame_total = total + segmentation``
as the 100% denominator, and shows ``total`` minus stages 2-6 as "(unmeasured)".
"""

import numpy as np

# Pipeline order for display; keys absent from the data are skipped.
STAGE_ORDER = [
    'segmentation',
    'node_detection',
    'cpd',
    'hungarian',
    'ee_injection',
    'geometry',
]


class StageTimer:
    """Accumulates per-frame ``result['timing']`` dicts.

    Initialization frames (``mode`` of ``init``/``restart``) report a completely
    different stage set than tracking frames -- ``build_graph``, ``repulsion``,
    ``find_path`` and their own ``total`` -- so they are kept in a separate pool.
    Pooling them would corrupt the tracking percentages, since one init frame's
    ``total`` is not comparable to a tracking frame's.
    """

    def __init__(self):
        self.stages = {}       # tracking stage -> list of per-frame seconds
        self.init_stages = {}  # init stage -> list of seconds (usually 1 frame)
        self.n_frames = 0      # tracking frames that contributed timing
        self.n_init = 0        # init / restart frames
        self.n_skipped = 0     # frames with no timing (failed / skipped)

    def add(self, result):
        """Record one frame. Accepts the full tracker result dict."""
        timing = result.get('timing') if isinstance(result, dict) else None
        if not timing:
            self.n_skipped += 1
            return
        # initialize() sets no 'mode', so detect the phase from the stage keys.
        # Only 'cpd' and 'geometry' are exclusive to track(); 'node_detection' is
        # emitted by the BDLO initializer too (wire_init.py) and cannot be used here.
        is_init = (result.get('mode') == 'restart'
                   or not any(k in timing for k in ('cpd', 'geometry')))
        bucket = self.init_stages if is_init else self.stages
        for key, value in timing.items():
            bucket.setdefault(key, []).append(float(value))
        if is_init:
            self.n_init += 1
        else:
            self.n_frames += 1

    def _ordered_keys(self):
        """Stages in pipeline order, then any unrecognized keys, minus 'total'."""
        known = [k for k in STAGE_ORDER if k in self.stages]
        extra = sorted(k for k in self.stages if k not in STAGE_ORDER and k != 'total')
        return known + extra

    def report(self, title='Per-Stage Timing'):
        """Render the accumulated timings as a text table."""
        if self.n_frames == 0:
            return f"{title}\n{'-' * 100}\nNo timing data collected.\n"

        keys = self._ordered_keys()
        # track()'s 'total' excludes segmentation (added later by process_frame),
        # so build a true per-frame total to use as the denominator.
        track_totals = np.asarray(self.stages.get('total', []), dtype=float)
        seg = np.asarray(self.stages.get('segmentation', []), dtype=float)
        if track_totals.size and seg.size == track_totals.size:
            totals = track_totals + seg
        elif track_totals.size:
            totals = track_totals
        else:
            totals = np.array([], dtype=float)
        grand = float(totals.sum()) if totals.size else sum(
            float(np.sum(self.stages[k])) for k in keys
        )

        lines = [
            title,
            '-' * 100,
            f"Frames timed: {self.n_frames}"
            + (f"   (skipped, no timing: {self.n_skipped})" if self.n_skipped else ''),
            '',
            f"{'Stage':<16}{'Mean (ms)':>12}{'Std (ms)':>11}"
            f"{'Max (ms)':>11}{'Total (s)':>12}{'% of total':>12}",
            '-' * 100,
        ]

        measured = 0.0
        for key in keys:
            vals = np.asarray(self.stages[key], dtype=float)
            stage_total = float(vals.sum())
            measured += stage_total
            pct = 100.0 * stage_total / grand if grand > 0 else 0.0
            lines.append(
                f"{key:<16}{vals.mean() * 1e3:>12.3f}{vals.std() * 1e3:>11.3f}"
                f"{vals.max() * 1e3:>11.3f}{stage_total:>12.3f}{pct:>11.1f}%"
            )

        lines.append('-' * 100)
        if totals.size:
            tv = totals
            remainder = grand - measured
            pct_rem = 100.0 * remainder / grand if grand > 0 else 0.0
            lines.append(
                f"{'(unmeasured)':<16}{'':>12}{'':>11}{'':>11}"
                f"{remainder:>12.3f}{pct_rem:>11.1f}%"
            )
            lines.append(
                f"{'FRAME TOTAL':<16}{tv.mean() * 1e3:>12.3f}{tv.std() * 1e3:>11.3f}"
                f"{tv.max() * 1e3:>11.3f}{grand:>12.3f}{100.0:>11.1f}%"
            )
            fps = 1.0 / tv.mean() if tv.mean() > 0 else float('inf')
            lines.append('')
            lines.append(f"Mean per-frame: {tv.mean() * 1e3:.2f} ms  ->  {fps:.1f} fps (tracking only)")

        # Initialization is a one-off frame-0 cost, reported separately.
        if self.n_init:
            init_total = self.init_stages.get('total', [])
            init_grand = float(np.sum(init_total)) if init_total else 0.0
            lines.append('')
            lines.append(f"Initialization ({self.n_init} frame{'s' if self.n_init > 1 else ''})")
            lines.append('-' * 100)
            lines.append(f"{'Stage':<20}{'Total (ms)':>14}{'% of init':>12}")
            lines.append('-' * 100)
            for key in sorted(self.init_stages, key=lambda k: -float(np.sum(self.init_stages[k]))):
                if key == 'total':
                    continue
                st = float(np.sum(self.init_stages[key]))
                pct = 100.0 * st / init_grand if init_grand > 0 else 0.0
                lines.append(f"{key:<20}{st * 1e3:>14.3f}{pct:>11.1f}%")
            lines.append('-' * 100)
            lines.append(f"{'TOTAL':<20}{init_grand * 1e3:>14.3f}{100.0:>11.1f}%")
        lines.append('')
        return '\n'.join(lines)

    def write(self, path, title='Per-Stage Timing'):
        """Write the report to ``path`` and return the text."""
        text = self.report(title)
        with open(path, 'w') as fh:
            fh.write(text)
        return text

    def save_npz(self, path):
        """Save raw per-frame series so runs can be compared or re-plotted.

        Tracking stages keep their plain names; init stages are prefixed with
        ``init_`` since the two pools have different lengths and overlapping keys.
        """
        if self.n_frames == 0 and self.n_init == 0:
            return
        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in self.stages.items()}
        arrays.update({f'init_{k}': np.asarray(v, dtype=np.float64)
                       for k, v in self.init_stages.items()})
        np.savez(path, **arrays)
