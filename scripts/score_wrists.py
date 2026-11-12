"""Score backends' raw wrist positions against hand-clicked wrists.

Usage:
  python scripts/score_wrists.py data/annotations/IMG_6004_wrists.json \
      mediapipe=outputs/IMG_6004.npz balanced=outputs/IMG_6004_rtmpose_bal.npz \
      perf=outputs/IMG_6004_rtmpose_perf.npz --torso 344 [--per-frame]

Distances are between a backend's raw wrist and your click, in torso lengths (and px).
Clicks you marked hidden are your best guess, not ground truth, so they are reported
separately. One swing, one annotator, a few dozen frames: a sanity check, not an
accuracy benchmark.
"""
import argparse

import numpy as np

from swing.annotations import WristAnnotations, summarise, wrist_errors
from swing.pose import PoseSequence

ap = argparse.ArgumentParser()
ap.add_argument("annotations")
ap.add_argument("npz", nargs="+", help="name=path.npz")
ap.add_argument("--torso", type=float, required=True, help="torso length in px (px / torso = reported unit)")
ap.add_argument("--per-frame", action="store_true")
a = ap.parse_args()

ann = WristAnnotations.load(a.annotations)
nvis = sum(p.visible for d in ann.points.values() for p in d.values())
nall = sum(len(d) for d in ann.points.values())
print(f"{len(ann.points)} frames, {nall} clicks ({nvis} visible, {nall - nvis} hidden guesses); "
      f"torso {a.torso:g} px\n")

runs = {}
for spec in a.npz:
    name, path = spec.split("=", 1)
    runs[name] = wrist_errors(ann, PoseSequence.load(path))

hdr = f"{'backend':10} {'subset':8} {'n':>4} {'miss':>4} {'median':>7} {'mean':>7} {'p90':>7} {'max':>7}   (torso lengths)"
print(hdr)
for name, errs in runs.items():
    for label, sel in (("visible", True), ("hidden", False)):
        for wrist in ("both", "left_wrist", "right_wrist"):
            sub = [e for e in errs if e.visible == sel and wrist in ("both", e.wrist)]
            if not sub:
                continue
            s = summarise(sub, a.torso)
            nums = " ".join(f"{s[k]:7.3f}" for k in ("median", "mean", "p90", "max")) if "median" in s else ""
            print(f"{name:10} {label:8} {s['n']:>4} {s['missing']:>4} {nums}   {wrist}")
if a.per_frame:
    frames = sorted({(e.frame, e.wrist, e.visible) for es in runs.values() for e in es})
    print("\nframe wrist        vis   " + "  ".join(f"{n:>9}" for n in runs) + "   (px)")
    for f, w, v in frames:
        row = [next(e.px for e in runs[n] if e.frame == f and e.wrist == w) for n in runs]
        print(f"{f:5} {w:12} {'yes' if v else 'HID'}   " + "  ".join(f"{x:9.1f}" for x in row))
