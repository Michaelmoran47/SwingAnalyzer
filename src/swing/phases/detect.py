"""Rule-based swing phase detection from the hand (wrist) trajectory.

Rules follow the definitions in docs/labelling.md and were fixed *before*
being scored against labels. They assume one swing per clip by a right-handed
golfer and image y pointing down (hands "highest" = smallest y).

Signal: ``hands`` = mean of the available wrists, short gaps interpolated.
All thresholds are relative (fractions of peak hand speed or of torso length)
so they do not depend on resolution, distance or frame rate.

1. peak: frame of maximum hand speed (the downswing).
2. impact: lowest hand position (max y) within ``impact_window_s`` after the peak.
3. top: highest hand position (min y) before the peak.
4. address: last still frame before the backswing, where "still" means speed
   below ``still_frac`` of peak, scanning back from the top (skipping the slow
   region around the top itself).
5. takeaway: first frame after address where the hands have moved more than
   ``takeaway_disp`` torso lengths from their address position.
6. finish: first frame after impact where hand speed stays below ``finish_frac``
   of peak for ``finish_hold_s``.

A phase that cannot be found is ``None`` and a reason is added to ``warnings``;
nothing is guessed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from swing.pose.base import J, PoseSequence


@dataclass(frozen=True)
class PhaseParams:
    max_gap_s: float = 0.3  # longest hand-signal gap we interpolate
    still_frac: float = 0.02  # speed (of peak) below which hands count as still
    takeaway_disp: float = 0.05  # torso lengths moved from address position
    impact_window_s: float = 0.20
    finish_frac: float = 0.10
    finish_hold_s: float = 0.15


@dataclass
class PhaseResult:
    frames: dict[str, int | None]
    fps: float
    warnings: list[str] = field(default_factory=list)

    def time_s(self, phase: str) -> float | None:
        f = self.frames.get(phase)
        return None if f is None else f / self.fps


def hand_signal(seq: PoseSequence, max_gap_s: float = 0.3) -> np.ndarray:
    """(T, 2) mean of finite wrists per frame, gaps up to ``max_gap_s`` interpolated."""
    w = np.stack([seq.keypoints[:, J["left_wrist"], :2], seq.keypoints[:, J["right_wrist"], :2]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames -> NaN
        hands = np.nanmean(w, axis=0)
    return _interp_short_gaps(hands, int(round(max_gap_s * seq.fps)))


def torso_length(seq: PoseSequence) -> float:
    k = seq.keypoints
    sh = np.nanmean(k[:, [J["left_shoulder"], J["right_shoulder"]], :2], axis=1)
    hip = np.nanmean(k[:, [J["left_hip"], J["right_hip"]], :2], axis=1)
    d = np.linalg.norm(sh - hip, axis=1)
    return float(np.nanmedian(d))


def _interp_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    out = x.copy()
    T = len(x)
    bad = ~np.isfinite(x).all(axis=1)
    i = 0
    while i < T:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j < T and bad[j]:
            j += 1
        if i > 0 and j < T and (j - i) <= max_gap:
            for ax in (0, 1):
                out[i:j, ax] = np.interp(np.arange(i, j), [i - 1, j], [x[i - 1, ax], x[j, ax]])
        i = j
    return out


def detect_phases_from_hands(
    hands: np.ndarray, fps: float, torso: float, params: PhaseParams = PhaseParams()
) -> PhaseResult:
    """Core detector on a ``(T, 2)`` hand trajectory (px); NaN = unknown."""
    p = params
    T = len(hands)
    frames: dict[str, int | None] = {k: None for k in
                                    ("address", "takeaway", "top", "impact", "finish")}
    warn: list[str] = []
    res = PhaseResult(frames, fps, warn)

    finite = np.isfinite(hands).all(axis=1)
    if finite.sum() < 10:
        warn.append("too few frames with a hand position")
        return res

    # speed in px/s (central difference); NaN wherever either neighbour is unknown
    speed = np.full(T, np.nan)
    v = np.linalg.norm(hands[2:] - hands[:-2], axis=1) * fps / 2.0
    speed[1:-1] = v

    peak = int(np.nanargmax(speed))
    vmax = float(speed[peak])
    if not vmax > 0:
        warn.append("no hand motion found")
        return res

    # impact: lowest hands shortly after the peak speed
    hi = min(T, peak + int(round(p.impact_window_s * fps)) + 1)
    seg = hands[peak:hi, 1]
    if np.isfinite(seg).any():
        frames["impact"] = peak + int(np.nanargmax(seg))
    else:
        warn.append("no hand position after peak speed; impact not found")

    # top: highest hands (min y) before the peak speed
    seg = hands[:peak + 1, 1]
    if not np.isfinite(seg).any():
        warn.append("no hand position before peak speed; top not found")
        return res
    top = int(np.nanargmin(seg))
    frames["top"] = top

    # address: scan back from the top, first out of the (slow) top region, then
    # through the backswing, to the last still frame before the motion began
    still = np.nan_to_num(speed, nan=np.inf) < p.still_frac * vmax
    j = top
    while j > 0 and still[j]:
        j -= 1
    while j > 0 and not still[j]:
        j -= 1
    if not still[j]:
        warn.append("no still period before the backswing; address not found")
    else:
        frames["address"] = j
        # takeaway: first frame the hands have left their address position
        disp = np.linalg.norm(hands - hands[j], axis=1)
        moved = np.flatnonzero(np.nan_to_num(disp[j:], nan=0.0) > p.takeaway_disp * torso)
        if len(moved):
            frames["takeaway"] = j + int(moved[0])
        else:
            warn.append("hands never moved beyond the takeaway threshold")

    # finish: first frame after impact where speed stays low for finish_hold_s
    start = frames["impact"] if frames["impact"] is not None else peak
    hold = int(round(p.finish_hold_s * fps))
    low = np.nan_to_num(speed, nan=np.inf) < p.finish_frac * vmax
    for t in range(start, T - hold):
        if low[t:t + hold].all():
            frames["finish"] = t
            break
    else:
        warn.append("hands never settled after impact; finish not found "
                    "(missing keypoints at the end of the clip?)")
    return res


def detect_phases(seq: PoseSequence, params: PhaseParams = PhaseParams()) -> PhaseResult:
    """Detect phases from a (smoothed) ``PoseSequence``."""
    hands = hand_signal(seq, params.max_gap_s)
    torso = torso_length(seq)
    res = detect_phases_from_hands(hands, seq.fps, torso, params)
    # frames are indices into the sequence; shift to video frame indices
    res.frames = {k: (None if v is None else v + seq.start_frame) for k, v in res.frames.items()}
    return res
