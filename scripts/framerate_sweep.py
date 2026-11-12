"""How does phase detection degrade as the frame rate drops?

Usage:
  python scripts/framerate_sweep.py outputs/IMG_6004.npz data/labels/IMG_6004.json [--window-s 0.06]

Takes a high-frame-rate keypoint file, keeps every k-th frame (k = 1, 2, 4, 8 ...) at every
possible offset, runs smoothing + phase detection with the frame-rate guard disabled, and
scores top and impact against the hand labels (the two validated phases).

Limits, stated plainly:
* Subsampling 240 fps keypoints is *optimistic* for real low-rate video: real 30 fps footage
  has longer exposures, so more motion blur at the fast part of the swing, and pose
  estimation is worse on blurred frames. Keypoints here were made on sharp 240 fps frames.
* One swing, one label set. This shows how the *rules* behave with fewer samples, not how
  accurate they are in general.
* The label is a frame in the original clip; its time is label / original fps. Error is in
  milliseconds of the detected phase time against that.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from swing.labels import PhaseLabels
from swing.phases import PhaseParams, detect_phases
from swing.pose import PoseSequence
from swing.smoothing import smooth_sequence


def run(seq, labels, k, offset, window_s, smooth=True):
    sub = replace(seq, keypoints=seq.keypoints[offset::k], fps=seq.fps / k)
    s = smooth_sequence(sub, window_s=window_s) if smooth else sub
    ph = detect_phases(s, PhaseParams(min_fps=0))
    out = {}
    for p in ("top", "impact"):
        f = ph.frames[p]
        out[p] = None if f is None else 1000.0 * (f / sub.fps - (labels.labels[p] - offset) / seq.fps
                                                  - 0.0)
    return out, ph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("labels")
    ap.add_argument("--window-s", type=float, default=0.06)
    ap.add_argument("--factors", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--no-smooth", action="store_true", help="skip smoothing")
    a = ap.parse_args()
    seq = PoseSequence.load(a.npz)
    labels = PhaseLabels.load(a.labels)
    print(f"{a.npz}: {len(seq)} frames @ {seq.fps:.1f} fps; labels top={labels.labels['top']} "
          f"impact={labels.labels['impact']}; smoothing {'off' if a.no_smooth else f'window {a.window_s} s (min 5 frames)'}")
    print(f"{'fps':>6} {'1 frame':>8} | {'top err ms: min / median / max':>32} | "
          f"{'impact err ms: min / median / max':>34} | not found")
    for k in a.factors:
        errs = {"top": [], "impact": []}
        missing = 0
        for off in range(k):
            e, ph = run(seq, labels, k, off, a.window_s, smooth=not a.no_smooth)
            for p in errs:
                if e[p] is None:
                    missing += 1
                else:
                    errs[p].append(e[p])
        fps = seq.fps / k

        def fmt(v):
            return "-" if not v else f"{min(v):+7.0f} / {np.median(v):+7.0f} / {max(v):+7.0f}"

        print(f"{fps:6.1f} {1000 / fps:6.1f}ms | {fmt(errs['top']):>32} | {fmt(errs['impact']):>34} | {missing}")


if __name__ == "__main__":
    main()
