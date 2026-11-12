"""Draw the detected club shaft on frames around the shaft-parallel estimate, to check it by eye.

Usage: python scripts/shaft_check.py <video> <raw keypoints .npz> [--fps 240] [--out sheet.jpg]

Frames shown: the ends of the bracket and a few between them, with the detected shaft line
(yellow) where one was found and its elevation in degrees. The estimate is where that elevation
crosses zero; confirm the yellow line is on the club.
"""

import argparse
from dataclasses import replace

import cv2
import numpy as np

from swing.io import iter_frames
from swing.phases import detect_phases
from swing.pose import J, PoseSequence
from swing.shaft import shaft_cue, shaft_elevation
from swing.smoothing import smooth_sequence

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("npz")
ap.add_argument("--fps", type=float); ap.add_argument("--out", default="outputs/shaft_check.jpg")
a = ap.parse_args()

seq = PoseSequence.load(a.npz)
if a.fps:
    seq = replace(seq, fps=a.fps)
s = smooth_sequence(seq)
ph = detect_phases(s).frames
cue = shaft_cue(a.video, s, ph)
print(f"takeaway {ph['takeaway']}, top {ph['top']} -> shaft-parallel {cue.frame}, bracket {cue.bracket}, {cue.note}")
if cue.frame is None:
    raise SystemExit(1)
lo, hi = cue.bracket
want = sorted({lo, cue.frame - 4, cue.frame, cue.frame + 4, hi, max(lo - 8, ph["takeaway"]), min(hi + 8, ph["top"])})
k = s.keypoints
torso = float(np.nanmedian(np.linalg.norm(k[:, [J["left_shoulder"], J["right_shoulder"]], :2].mean(1)
                                          - k[:, [J["left_hip"], J["right_hip"]], :2].mean(1), axis=1)))
tiles = []
for i, f in iter_frames(a.video, 0, max(want) + 1):
    if i not in want:
        continue
    hands = np.nanmean(k[i, [J["left_wrist"], J["right_wrist"]], :2], axis=0)
    elbow = np.nanmean(k[i, [J["left_elbow"], J["right_elbow"]], :2], axis=0)
    r = shaft_elevation(f, hands, elbow if np.isfinite(elbow).all() else None, torso)
    if r:
        cv2.line(f, tuple(int(v) for v in r[2][0]), tuple(int(v) for v in r[2][1]), (0, 255, 255), 5)
    cv2.circle(f, tuple(int(v) for v in hands), 10, (0, 0, 255), -1)
    rad = int(1.4 * torso); cx, cy = int(hands[0]), int(hands[1])
    c = cv2.resize(f[max(0, cy - rad):cy + rad, max(0, cx - rad):cx + rad], (260, 260))
    tag = f"{i}  " + ("no line" if r is None else f"{r[0]:+.0f} deg") + ("  <- estimate" if i == cue.frame else "")
    cv2.rectangle(c, (0, 0), (260, 28), (0, 0, 0), -1)
    cv2.putText(c, tag, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    tiles.append(c)
cv2.imwrite(a.out, np.hstack(tiles))
print("->", a.out)
