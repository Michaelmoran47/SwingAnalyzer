"""Distance between two keypoint files over a frame range, in torso lengths.

Usage: python scripts/compare_keypoint_files.py a.npz b.npz --start 985 --end 1050 [--torso 344]

Frames are video frame indices; either file may cover only part of the clip.
This is a difference between two estimates, not an error against the truth.
"""
import argparse

import numpy as np

from swing.pose import J, PoseSequence

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--start", type=int, required=True); ap.add_argument("--end", type=int, required=True)
ap.add_argument("--torso", type=float, required=True, help="torso length in px (see `swing phases`/metrics)")
ap.add_argument("--far", type=float, default=0.5, help="count frames farther apart than this many torsos")
x = ap.parse_args()


def window(path):
    q = PoseSequence.load(path)
    lo, hi = x.start - q.start_frame, x.end - q.start_frame
    if lo < 0 or hi > len(q):
        raise SystemExit(f"{path} covers frames {q.start_frame}-{q.start_frame + len(q) - 1}, not {x.start}-{x.end - 1}")
    return q.keypoints[lo:hi]


A, B = window(x.a), window(x.b)
n = x.end - x.start
for j in ("left_wrist", "right_wrist", "left_hip", "right_hip"):
    d = np.linalg.norm(A[:, J[j], :2] - B[:, J[j], :2], axis=1) / x.torso
    print(f"{j:12s} median {np.nanmedian(d):.3f}  max {np.nanmax(d):.3f} torso; "
          f"> {x.far} torso apart in {int((d > x.far).sum())}/{n} frames")
