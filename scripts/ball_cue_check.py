"""Run the ball-leaves-tee cue on clips and compare with hand labels.

Usage: python scripts/ball_cue_check.py <video> <raw keypoints .npz> [--fps 240] [--labels file.json]
"""

import argparse
from dataclasses import replace

from swing.impact_cue import ball_cue
from swing.labels import PhaseLabels
from swing.phases import detect_phases
from swing.pose import PoseSequence
from swing.smoothing import smooth_sequence

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("npz")
ap.add_argument("--fps", type=float, help="capture rate to assume (for slowed clips)")
ap.add_argument("--labels")
a = ap.parse_args()

seq = PoseSequence.load(a.npz)
if a.fps:
    seq = replace(seq, fps=a.fps)
s = smooth_sequence(seq)
ph = detect_phases(s)
cue = ball_cue(a.video, s, ph.frames)
lab = PhaseLabels.load(a.labels).labels.get("impact") if a.labels else None
print(f"hand-based impact: {ph.frames['impact']}   ball cue: {cue.frame}   your label: {lab}")
if lab is not None:
    for name, f in (("hand-based", ph.frames["impact"]), ("ball cue", cue.frame)):
        if f is not None:
            print(f"  {name:10s} error {f - lab:+d} frames = {1000 * (f - lab) / s.fps:+.0f} ms at {s.fps:.0f} fps")
print(f"candidates {cue.candidates}, score {cue.score}, center {cue.center}, radius {cue.radius}")
if cue.note:
    print("note:", cue.note)
