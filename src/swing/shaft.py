"""Shaft-parallel in the backswing, from the picture.

The pose model has no club keypoint, so the shaft has to be found in the video. In the image
the shaft is "parallel to the ground" when the club head is level with the hands, which holds in
any view if the camera is level: so the checkpoint is the frame where the shaft's *elevation*
(the angle of the line from the hands to the club head above the horizontal) crosses zero on
the way up. At address the head is below the hands (about -45 to -60 degrees on the clips tried);
at the top it is above them (about +55 to +60).

Method:

1. For frames between the takeaway and the top, find the strongest straight edge that passes
   through the hands and extends away from them (not back along the forearm): Canny + a
   probabilistic Hough transform in a window around the hands. That line is the shaft; its end
   farther from the hands is the head, which gives the elevation.
2. Fit a monotone (non-decreasing) curve through the readings, because the shaft only rotates
   upward in a backswing; this removes the occasional line that is really a forearm or shirt edge.
3. The checkpoint is where the fitted curve crosses zero, linearly interpolated, **reported with
   the bracket**: the frames of the last reading below horizontal and the first above it.

Why a bracket, not a frame: the shaft crosses horizontal while the club is moving fastest, so it
is motion-blurred there and often not detected. On one clip the crossing fell in a gap of 48
frames with no reading. The estimate is the interpolation inside that gap.

What it measures: the shaft's angle in the *image plane*, not in 3D. It is right when the camera
is level and the shaft is not pointing along the camera's line of sight (a perfectly
down-the-line shaft at that moment would look short and the angle would be unreliable; the clips
tried were not that exact). It is unvalidated: there are no hand labels for it yet.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import cv2
import numpy as np

from swing.io import iter_frames
from swing.pose.base import J, PoseSequence

MIN_LENGTH_TORSO = 0.5  # a shaft line is at least this long (torso lengths): the false lines seen were 0.36-0.49
MAX_PERP_TORSO = 0.12  # and passes this close to the hands
MAX_BEHIND_TORSO = 0.2  # it may extend this far behind the hands (the grip end), no more
BEHIND_FROM, BEHIND_TO, BEHIND_FRACTION = 0.2, 0.5, 0.6  # edge pixels this far behind the hands: not a shaft
MIN_SIDE_READINGS = 3  # readings needed clearly below and clearly above horizontal
CLEAR_DEG = 10.0  # "clearly" below/above
WIDE_BRACKET = 0.6  # a bracket wider than this share of the backswing is flagged low confidence


@dataclass
class ShaftCue:
    frame: int | None  # estimated frame the shaft is level; None if it could not be found
    bracket: tuple[int, int] | None = None  # last reading below and first above horizontal
    readings: int = 0
    low_confidence: bool = False
    note: str = ""


# --------------------------------------------------------------------------- the picture

def _continues_behind(near_edges: np.ndarray, hands: np.ndarray, u: np.ndarray, torso: float) -> bool:
    """True if edge pixels run along the line behind the hands (0.2-0.5 torso lengths back).

    A shaft ends at the grip; a background edge or a table line does not.
    """
    hits = total = 0
    for sdist in np.linspace(BEHIND_FROM * torso, BEHIND_TO * torso, 12):
        x, y = hands - sdist * u
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < near_edges.shape[0] and 0 <= xi < near_edges.shape[1]:
            total += 1
            hits += bool(near_edges[yi, xi])
    return total >= 6 and hits / total >= BEHIND_FRACTION


def shaft_elevation(frame_bgr: np.ndarray, hands, elbow, torso: float):
    """``(elevation_deg, length_px, (hands_xy, head_end_xy))`` of the shaft line through ``hands``, or ``None``.

    ``elbow`` (may be ``None``) is used to reject lines that run back along the forearm.
    Elevation is positive when the end farther from the hands is above them.
    """
    r = int(1.1 * torso)
    x0, y0 = int(hands[0]) - r, int(hands[1]) - r
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(frame_bgr.shape[1], x0 + 2 * r), min(frame_bgr.shape[0], y0 + 2 * r)
    if xb - xa < 16 or yb - ya < 16:
        return None
    gray = cv2.GaussianBlur(cv2.cvtColor(frame_bgr[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = cv2.Canny(gray, 40, 120)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=30,
                           minLineLength=int(MIN_LENGTH_TORSO * torso), maxLineGap=int(0.05 * torso))
    if segs is None:
        return None
    h = np.array([hands[0] - xa, hands[1] - ya], float)
    near_edges = cv2.dilate(edges, np.ones((5, 5), np.uint8)) > 0
    out = None
    if elbow is not None and np.isfinite(elbow).all():
        out = np.asarray(hands, float) - np.asarray(elbow, float)  # forearm direction, elbow -> hands
    best = None
    for x1, y1, x2, y2 in segs[:, 0]:
        p1, p2 = np.array([x1, y1], float), np.array([x2, y2], float)
        d = p2 - p1
        length = float(np.linalg.norm(d))
        if length < 1:
            continue
        if abs(np.cross(d / length, h - p1)) > MAX_PERP_TORSO * torso:
            continue  # does not pass through the hands
        u = d / length
        t1, t2 = float(np.dot(p1 - h, u)), float(np.dot(p2 - h, u))
        if min(t1, t2) < -MAX_BEHIND_TORSO * torso and max(t1, t2) > MAX_BEHIND_TORSO * torso:
            continue  # a background edge running right through the hands: a shaft ends at the grip
        far = p2 if np.linalg.norm(p2 - h) > np.linalg.norm(p1 - h) else p1
        v = far - h
        if _continues_behind(near_edges, h, v / max(np.linalg.norm(v), 1e-9), torso):
            continue  # the edge carries on behind the hands: a background line, not a shaft
        if np.linalg.norm(v) < MIN_LENGTH_TORSO * torso:
            continue  # does not extend away from the hands
        if out is not None and float(np.dot(v, out)) < 0:
            continue  # runs back along the forearm
        if best is None or length > best[1]:
            best = (float(np.degrees(np.arctan2(-v[1], abs(v[0])))), length,
                    ((float(h[0] + xa), float(h[1] + ya)), (float(far[0] + xa), float(far[1] + ya))))
    return best


# --------------------------------------------------------------------------- pure logic

def isotonic_increasing(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Non-decreasing fit by pooling adjacent violators, each pool taking its weighted *median*.

    The median (an L1 fit) keeps one stray reading from dragging the curve; a mean-based fit let a
    single false line move the crossing by 20 frames on one clip.
    """
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)

    def wmedian(vals, wts):
        order = np.argsort(vals)
        v, ww = np.asarray(vals)[order], np.asarray(wts)[order]
        c = np.cumsum(ww)
        return float(v[np.searchsorted(c, 0.5 * c[-1])])

    blocks = []  # each: [values, weights, fit]
    for yi, wi in zip(y, w):
        blocks.append([[yi], [wi], yi])
        while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
            v2, w2, _ = blocks.pop()
            v1, w1, _ = blocks.pop()
            vals, wts = v1 + v2, w1 + w2
            blocks.append([vals, wts, wmedian(vals, wts)])
    return np.concatenate([np.full(len(v), f) for v, _, f in blocks])


def crossing_from_readings(frames: np.ndarray, elevations: np.ndarray, weights: np.ndarray,
                           backswing_frames: int) -> ShaftCue:
    """Where the shaft's elevation crosses zero going up, from noisy readings.

    ``frames`` ascending. Needs ``MIN_SIDE_READINGS`` readings clearly below and clearly above
    horizontal; otherwise there is no answer. The estimate is the zero of a monotone fit,
    interpolated between neighbouring readings; the bracket is the last raw reading below zero and
    the first above it around that point.
    """
    frames, elevations = np.asarray(frames), np.asarray(elevations, float)
    n = len(frames)
    below = int((elevations < -CLEAR_DEG).sum())
    above = int((elevations > CLEAR_DEG).sum())
    if below < MIN_SIDE_READINGS or above < MIN_SIDE_READINGS:
        return ShaftCue(None, readings=n, note=(
            f"too few shaft readings on both sides of horizontal ({below} clearly below, {above} clearly "
            f"above, from {n} readings; the shaft was probably hidden or blurred)"))
    fit = isotonic_increasing(elevations, weights)
    up = np.nonzero(fit >= 0)[0]
    if len(up) == 0 or up[0] == 0:
        return ShaftCue(None, readings=n, note="the shaft never rose through horizontal in the readings")
    j = int(up[0])
    lo_f, hi_f = fit[j - 1], fit[j]
    t = frames[j - 1] + (0.0 - lo_f) / (hi_f - lo_f) * (frames[j] - frames[j - 1]) if hi_f > lo_f else frames[j]
    last_neg = int(frames[np.nonzero(elevations[:j + 1] < 0)[0][-1]]) if (elevations[:j + 1] < 0).any() else int(frames[j - 1])
    first_pos = int(frames[j + np.nonzero(elevations[j:] > 0)[0][0]]) if (elevations[j:] > 0).any() else int(frames[j])
    bracket = (min(last_neg, first_pos), max(last_neg, first_pos))
    wide = (bracket[1] - bracket[0]) > WIDE_BRACKET * max(1, backswing_frames)
    note = f"between frames {bracket[0]} and {bracket[1]}" + ("; the bracket is wide, treat as rough" if wide else "")
    return ShaftCue(int(round(t)), bracket, n, wide, note)


# --------------------------------------------------------------------------- orchestration

def shaft_cue(video, seq: PoseSequence, frames: dict[str, int | None], max_readings: int = 90) -> ShaftCue:
    """Estimate the shaft-parallel frame for one clip.

    ``seq`` is the smoothed keypoints (hands, elbows, torso); ``frames`` needs ``takeaway`` and
    ``top``. Returns ``ShaftCue(None, ...)`` with a note when the picture does not support an answer.
    """
    tk, top = frames.get("takeaway"), frames.get("top")
    if tk is None or top is None or not tk < top:
        return ShaftCue(None, note="needs the takeaway and the top (in order) to know where to look")
    k = seq.keypoints
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        torso = float(np.nanmedian(np.linalg.norm(
            k[:, [J["left_shoulder"], J["right_shoulder"]], :2].mean(axis=1)
            - k[:, [J["left_hip"], J["right_hip"]], :2].mean(axis=1), axis=1)))
    if not np.isfinite(torso) or torso <= 0:
        return ShaftCue(None, note="torso not tracked")
    step = max(1, int(round((top - tk) / max_readings)))
    want = set(range(tk, top + 1, step))
    fs, es, ws = [], [], []
    for i, f in iter_frames(video, 0, top + 1):
        if i not in want:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            hands = np.nanmean(k[i, [J["left_wrist"], J["right_wrist"]], :2], axis=0)
            elbow = np.nanmean(k[i, [J["left_elbow"], J["right_elbow"]], :2], axis=0)
        if not np.isfinite(hands).all():
            continue
        r = shaft_elevation(f, hands, elbow if np.isfinite(elbow).all() else None, torso)
        if r is not None:
            fs.append(i)
            es.append(r[0])
            ws.append(r[1])
    if not fs:
        return ShaftCue(None, note="no shaft line found through the hands in the backswing")
    return crossing_from_readings(np.array(fs), np.array(es), np.array(ws), top - tk)


def refine_shaft(phases, cue: ShaftCue):
    """Return ``(phases, note)`` with ``shaft_parallel`` filled in when it fits between takeaway and top."""
    from dataclasses import replace

    if cue.frame is None:
        return phases, f"shaft-parallel not found: {cue.note}" if cue.note else "shaft-parallel not found"
    f = dict(phases.frames)
    tk, top = f.get("takeaway"), f.get("top")
    if tk is None or top is None or not tk < cue.frame < top:
        return phases, f"shaft-parallel estimate (frame {cue.frame}) is not between the takeaway and the top: not used"
    f["shaft_parallel"] = cue.frame
    sources = dict(phases.sources)
    sources["shaft_parallel"] = "the club shaft's angle in the picture (estimate)"
    return replace(phases, frames=f, sources=sources), f"shaft-parallel {cue.note}"
