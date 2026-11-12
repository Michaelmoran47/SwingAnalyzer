"""Start of the swing from whole-body motion.

The takeaway is where the swing starts: the first perceptible movement after the golfer is
set. Wrist speed alone, with a threshold relative to the *peak downswing speed*, ignores the
small early motion in the shoulders, hips and arms that a person sees as the start, and put
the takeaway 37-96 frames late on three hand-labelled swings.

Method:

* Pool eight joints (shoulders, hips, elbows, wrists): the mean of their per-frame speeds.
  Averaging reduces the keypoint noise.
* Noise level ``N`` = the median of the quietest 25 % of frames before the top, a measure of
  how still this golfer is on this clip (so it adapts to resolution and pose quality).
* A frame is *quiet* if the pooled speed is at most ``kappa * N`` (default 3).
* The takeaway is the first frame after the **last quiet stretch** (``quiet_s`` seconds, at
  least 10 frames, all joints tracked) that ends before the hand-based takeaway.

A waggle or forward press before the swing counts as movement; the method cannot tell it from
a takeaway. If no quiet stretch is found, nothing is returned and the caller keeps the
hand-based frame.
"""

from __future__ import annotations

import numpy as np

from swing.pose.base import J

POOL = ("left_shoulder", "right_shoulder", "left_hip", "right_hip",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist")
MIN_JOINTS = 4  # a frame needs at least this many pooled joints, or it is "unknown"
QUIET_FRACTION = 0.25  # share of pre-top frames used to estimate the noise level


def pooled_speed(keypoints: np.ndarray) -> np.ndarray:
    """Mean per-frame speed (px/frame, central difference) of the pooled joints; NaN if too few."""
    idx = [J[n] for n in POOL]
    xy = keypoints[:, idx, :2]
    sp = np.full((len(keypoints), len(idx)), np.nan)
    if len(keypoints) >= 3:
        sp[1:-1] = np.linalg.norm(xy[2:] - xy[:-2], axis=2) / 2.0
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames -> NaN
        e = np.nanmean(sp, axis=1)
    e[np.isfinite(sp).sum(axis=1) < MIN_JOINTS] = np.nan
    return e


def body_motion_onset(
    keypoints: np.ndarray, fps: float, t0: int, top: int, kappa: float = 3.0, quiet_s: float = 0.10
) -> int | None:
    """Frame the swing starts, or ``None`` if no quiet stretch precedes ``t0``.

    ``t0`` is the hand-based takeaway (a frame inside the early backswing, known to be at or
    after the true start) and ``top`` the top of the backswing; ``keypoints`` are smoothed.
    """
    if not 0 < t0 < top <= len(keypoints):
        return None
    e = pooled_speed(keypoints)
    pre = e[:top]
    finite = pre[np.isfinite(pre)]
    if len(finite) < 20:
        return None
    n_quiet = max(5, int(QUIET_FRACTION * len(finite)))
    noise = float(np.median(np.sort(finite)[:n_quiet]))
    if not noise > 0:
        return None
    quiet = np.isfinite(e) & (e <= kappa * noise)
    length = max(10, int(round(quiet_s * fps)))
    for end in range(min(t0, len(e) - 1), length - 1, -1):
        window = slice(end - length + 1, end + 1)
        if quiet[window].all():  # every frame quiet and tracked
            return end + 1
    return None
