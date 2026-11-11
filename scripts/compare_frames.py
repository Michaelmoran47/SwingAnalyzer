"""Draw wrist/hip markers from two backends on chosen frames, side by side.

Usage:
  python scripts/compare_frames.py data/raw/IMG_6004.MOV \
      mediapipe=outputs/IMG_6004.npz rtmpose=outputs/IMG_6004_rtmpose.npz \
      --frames 985 1000 1010 1020 1030 1045 --out outputs/blur_check.jpg

First backend: hollow red circles; second: filled green. Large = wrists, small =
hips. Prints each backend's confidence for the wrists. This is a *visual* check:
look at whether the markers sit on the golfer's hands. Confidence alone does not
tell you (a model can be confident and wrong).
"""

import argparse

import cv2
import numpy as np

from swing.io import iter_frames
from swing.pose import J, PoseSequence

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("npz", nargs=2, help="name=path.npz (exactly two)")
ap.add_argument("--frames", type=int, nargs="+", required=True)
ap.add_argument("--out", default="outputs/compare_frames.jpg")
ap.add_argument("--crop-y", type=int, nargs=2, default=[300, 1500])
a = ap.parse_args()

def _full(path):
    """Keypoints indexed by video frame; frames outside the file's range are NaN."""
    q = PoseSequence.load(path)
    n = q.start_frame + len(q)
    k = np.full((n, 17, 3), np.nan, np.float32)
    k[q.start_frame:] = q.keypoints
    return k


seqs = [(n, _full(p)) for n, p in (x.split("=", 1) for x in a.npz)]
cols = [(0, 0, 255), (0, 220, 0)]
want, tiles = set(a.frames), {}
for i, f in iter_frames(a.video, min(a.frames), max(a.frames) + 1):
    if i not in want:
        continue
    for (name, k), col, thick in zip(seqs, cols, (3, -1)):
        for jn, r in (("left_wrist", 13), ("right_wrist", 13), ("left_hip", 10), ("right_hip", 10)):
            x, y, _ = k[i, J[jn]] if i < len(k) else (np.nan,) * 3
            if np.isfinite(x):
                cv2.circle(f, (int(x), int(y)), r, col, thick)
    cv2.putText(f, str(i), (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 6)
    y0, y1 = a.crop_y
    tiles[i] = cv2.resize(f[y0:y1, :], (360, int(360 * (y1 - y0) / f.shape[1])))
cv2.imwrite(a.out, np.hstack([tiles[i] for i in a.frames if i in tiles]))
for i in a.frames:
    print(i, " | ".join(f"{n} wrist conf L/R {k[i, J['left_wrist'], 2]:.2f} {k[i, J['right_wrist'], 2]:.2f}"
                        for n, k in seqs))
print("->", a.out)
