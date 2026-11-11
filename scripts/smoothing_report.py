"""Quantify what smoothing does to a saved keypoint file.

Usage: python scripts/smoothing_report.py outputs/IMG_6004.npz [--still-s 1.0]

Metrics (all in image pixels):
* static noise: std of each joint's position about its own mean over the first
  ``--still-s`` seconds. This only measures noise if the golfer really is still
  then -- check the clip. It is an upper bound on pure jitter (real postural
  sway is included).
* jitter: RMS of the second difference (px/frame^2) over the whole clip.
* peak speed: maximum wrist speed (px/s). Raw peaks are inflated by noise;
  heavy smoothing deflates real peaks. This shows the trade-off.
"""

from __future__ import annotations

import argparse

import numpy as np

from swing.pose import J, PoseSequence
from swing.smoothing import fill_gaps, smooth_sequence

SCAN = [None, 0.03, 0.06, 0.10, 0.15]  # None = raw


def static_noise(k, n):
    seg = k[:n, :, :2]
    return float(np.nanmean(np.nanstd(seg, axis=0)))


def jitter(k):
    d2 = np.diff(k[:, :, :2], n=2, axis=0)
    return float(np.sqrt(np.nanmean(d2**2)))


def peak_speed(k, joint, fps):
    v = np.linalg.norm(np.diff(k[:, J[joint], :2], axis=0), axis=1) * fps
    return float(np.nanmax(v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--still-s", type=float, default=1.0)
    ap.add_argument("--min-conf", type=float, default=0.3)
    a = ap.parse_args()

    seq = PoseSequence.load(a.npz)
    n_still = int(a.still_s * seq.fps)
    raw_conf = np.nan_to_num(seq.keypoints[:, :, 2])
    masked = (raw_conf < a.min_conf) | ~np.isfinite(seq.keypoints[:, :, 0])
    _, filled = fill_gaps(seq.keypoints, seq.fps, a.min_conf)
    print(f"{a.npz}: {len(seq)} frames @ {seq.fps:.2f} fps, backend={seq.backend}")
    print(f"joint-frames below conf {a.min_conf}: {masked.mean():.1%}; "
          f"interpolated: {filled.mean():.1%}; "
          f"still left as NaN: {(masked & ~filled).mean():.1%}")
    print(f"static window: first {n_still} frames ({a.still_s:g} s)\n")
    print(f"{'window':>8} {'static_px':>10} {'jitter':>8} {'L_wrist_pk':>11} {'R_wrist_pk':>11}")
    for w in SCAN:
        k = seq.keypoints if w is None else smooth_sequence(
            seq, a.min_conf, window_s=w).keypoints
        label = "raw" if w is None else f"{w * 1000:.0f} ms"
        print(f"{label:>8} {static_noise(k, n_still):10.2f} {jitter(k):8.3f} "
              f"{peak_speed(k, 'left_wrist', seq.fps):11.0f} "
              f"{peak_speed(k, 'right_wrist', seq.fps):11.0f}")


if __name__ == "__main__":
    main()
