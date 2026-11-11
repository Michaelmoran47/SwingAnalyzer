"""Plot raw vs smoothed wrist y-position over time.

Usage: python scripts/plot_trajectories.py raw.npz smooth.npz out.png [--t0 2.8 --t1 4.8]
Grey bands: frames where the joint is missing after smoothing.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swing.pose import J, PoseSequence

ap = argparse.ArgumentParser()
ap.add_argument("raw"); ap.add_argument("smooth"); ap.add_argument("out")
ap.add_argument("--t0", type=float, default=None); ap.add_argument("--t1", type=float, default=None)
a = ap.parse_args()
raw, sm = PoseSequence.load(a.raw), PoseSequence.load(a.smooth)
t = np.arange(len(raw)) / raw.fps
lo, hi = (a.t0 or t[0]), (a.t1 or t[-1])
fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
for col, name in enumerate(("left_wrist", "right_wrist")):
    for row, ax_i in enumerate((0, 1)):
        ax = axes[row, col]
        ax.plot(t, raw.keypoints[:, J[name], ax_i], color="#bbbbbb", lw=1, label="raw (MediaPipe)")
        ax.plot(t, sm.keypoints[:, J[name], ax_i], color="#d95f02", lw=1.5, label="smoothed")
        miss = ~np.isfinite(sm.keypoints[:, J[name], 0])
        ax.fill_between(t, *ax.get_ylim(), where=miss, color="#000", alpha=0.08, step="mid")
        ax.set_title(f"{name} {'x' if ax_i == 0 else 'y'} (px)"); ax.set_xlim(lo, hi)
axes[1, 0].set_xlabel("time (s)"); axes[1, 1].set_xlabel("time (s)")
axes[0, 0].legend(loc="upper left")
fig.tight_layout(); fig.savefig(a.out, dpi=90)
