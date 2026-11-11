"""Keypoint cleaning: gap filling, then zero-phase Savitzky-Golay smoothing.

Why these choices:

* Analysis is offline, so we can use a zero-phase (non-causal) filter. A causal
  filter such as a One Euro filter would lag the motion and shift phase timing
  (top, impact) later in time.
* Savitzky-Golay fits a local polynomial, so it keeps peaks (fast downswing)
  better than a plain moving average of the same width.
* Windows are specified in seconds, so the same settings work at 120 and 240 fps.

Low-confidence joints are treated as missing, not as measurements.
Interpolated joints are flagged in ``PoseSequence.filled`` and keep their
original (low) confidence so later stages can down-weight them.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.signal import savgol_filter


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``(start, end)`` index ranges where ``mask`` is True."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def fill_gaps(
    keypoints: np.ndarray, fps: float, min_conf: float = 0.3, max_gap_s: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Mask unreliable joints and linearly interpolate short gaps.

    Returns ``(keypoints, filled)``. Positions that are NaN or have confidence
    below ``min_conf`` are masked; gaps no longer than ``max_gap_s`` that have a
    good sample on both sides are interpolated. Longer gaps stay NaN. Confidence
    values are never modified (NaN confidence becomes 0).
    """
    out = np.array(keypoints, dtype=np.float32, copy=True)
    out[:, :, 2] = np.nan_to_num(out[:, :, 2], nan=0.0)
    T, K, _ = out.shape
    max_gap = int(round(max_gap_s * fps))
    filled = np.zeros((T, K), dtype=bool)
    t = np.arange(T)
    for k in range(K):
        bad = ~np.isfinite(out[:, k, :2]).all(axis=1) | (out[:, k, 2] < min_conf)
        out[bad, k, :2] = np.nan
        for s, e in _runs(bad):
            if s == 0 or e == T or e - s > max_gap:
                continue
            for ax in (0, 1):
                out[s:e, k, ax] = np.interp(t[s:e], [s - 1, e], [out[s - 1, k, ax], out[e, k, ax]])
            filled[s:e, k] = True
    return out, filled


def savgol_smooth(
    keypoints: np.ndarray, fps: float, window_s: float = 0.06, polyorder: int = 3
) -> np.ndarray:
    """Zero-phase Savitzky-Golay smoothing of x/y, per joint, over finite runs.

    Runs shorter than the window are left untouched. Confidence is unchanged.
    """
    win = max(int(round(window_s * fps)) | 1, polyorder + 2 | 1)  # odd, > polyorder
    out = np.array(keypoints, dtype=np.float32, copy=True)
    finite = np.isfinite(out[:, :, :2]).all(axis=2)
    for k in range(out.shape[1]):
        for s, e in _runs(finite[:, k]):
            if e - s < win:
                continue
            for ax in (0, 1):
                out[s:e, k, ax] = savgol_filter(out[s:e, k, ax], win, polyorder)
    return out


def smooth_sequence(seq, min_conf: float = 0.3, max_gap_s: float = 0.1,
                    window_s: float = 0.06, polyorder: int = 3):
    """Fill short gaps, then smooth. Returns a new ``PoseSequence``."""
    filled_kps, filled = fill_gaps(seq.keypoints, seq.fps, min_conf, max_gap_s)
    smoothed = savgol_smooth(filled_kps, seq.fps, window_s, polyorder)
    return replace(seq, keypoints=smoothed, filled=filled)
