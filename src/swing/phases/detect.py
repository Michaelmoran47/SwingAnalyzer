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
   below ``still_frac`` of peak. Scanning back from the top, first skip the slow
   region around the top until speed has risen above ``rise_frac`` of peak
   (speed flickers near the top, so stillness is only trusted after that).
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
    rise_frac: float = 0.10  # speed (of peak) the backswing must reach before stillness counts
    still_frac: float = 0.02  # speed (of peak) below which hands count as still
    takeaway_disp: float = 0.05  # torso lengths moved from address position
    impact_window_s: float = 0.20
    finish_frac: float = 0.10
    finish_hold_s: float = 0.15
    # Lowest frame rate checked (scripts/framerate_sweep.py: 240 fps keypoints subsampled to
    # 15 fps still gave top and impact within one frame interval of the hand labels). Below
    # it detection is not attempted.
    min_fps: float = 15.0
    # Believable durations for a real-time swing. Deliberately wide: they exist to catch a
    # clip that was slowed down (or a detection that went wrong), not to judge a swing.
    backswing_range_s: tuple[float, float] = (0.2, 3.0)  # takeaway to top
    downswing_range_s: tuple[float, float] = (0.08, 1.5)  # top to impact


@dataclass
class PhaseResult:
    frames: dict[str, int | None]
    fps: float
    warnings: list[str] = field(default_factory=list)
    plausible: bool = True  # False when the detected timings do not look like a real-time swing

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

    if fps < p.min_fps:
        warn.append(
            f"frame rate {fps:.1f} fps is below {p.min_fps:.0f}, so phases were not detected. "
            "If the clip was recorded in slo-mo and re-timed to play slowly, give the capture "
            "frame rate (CLI --fps; in the demo, the 'Capture frame rate' control)")
        res.plausible = False
        return res

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
    v = np.nan_to_num(speed, nan=np.inf)
    still = v < p.still_frac * vmax
    # Leave the top region only once speed has clearly risen (rise_frac of peak).
    # Speed flickers around the still level near the top, so "first non-still
    # frame" is not a reliable exit.
    j = top
    while j > 0 and v[j] < p.rise_frac * vmax:
        j -= 1
    if v[j] < p.rise_frac * vmax:
        warn.append("backswing speed never rose clearly above the still level; address not found")
        return res
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
    issues = plausibility_issues(frames, fps, p)
    if issues:
        res.plausible = False
        warn.append(
            "these timings do not look like a real-time swing (" + "; ".join(issues) + "). If the clip "
            "was recorded in slow motion and re-timed to play slowly, give the capture frame rate "
            "you recorded at (CLI --fps; in the demo, the 'Capture frame rate' control)")
    return res


def plausibility_issues(frames: dict[str, int | None], fps: float,
                        params: PhaseParams = PhaseParams()) -> list[str]:
    """Reasons the detected phases cannot be a real-time swing; empty if they look fine.

    Catches phases out of order or sharing a frame, and backswing/downswing durations
    outside ``params.*_range_s``. A clip slowed down several times fails these.
    """
    order = ["address", "takeaway", "top", "impact", "finish"]
    present = [(n, frames[n]) for n in order if frames.get(n) is not None]
    issues = []
    if any(b <= a for (_, a), (_, b) in zip(present, present[1:])):
        issues.append("phases are out of order or share a frame: "
                      + ", ".join(f"{n} {f}" for n, f in present))
    t, top, imp = frames.get("takeaway"), frames.get("top"), frames.get("impact")
    if t is not None and top is not None and top > t:
        b = (top - t) / fps
        lo, hi = params.backswing_range_s
        if not lo <= b <= hi:
            issues.append(f"backswing {b:.2f} s is outside {lo}-{hi} s")
    if top is not None and imp is not None and imp > top:
        d = (imp - top) / fps
        lo, hi = params.downswing_range_s
        if not lo <= d <= hi:
            issues.append(f"downswing {d:.2f} s is outside {lo}-{hi} s")
    return issues


def detect_phases(seq: PoseSequence, params: PhaseParams = PhaseParams()) -> PhaseResult:
    """Detect phases from a (smoothed) ``PoseSequence``."""
    hands = hand_signal(seq, params.max_gap_s)
    torso = torso_length(seq)
    res = detect_phases_from_hands(hands, seq.fps, torso, params)
    # frames are indices into the sequence; shift to video frame indices
    res.frames = {k: (None if v is None else v + seq.start_frame) for k, v in res.frames.items()}
    return res
