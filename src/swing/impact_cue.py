"""Impact from the picture: the first frame the ball visibly leaves the tee.

Wrist tracking cannot see the club, so "lowest hands" can be several frames away from contact
(on a slow-motion clip the hands stall for ~6 frames while the club swings through). The
ball itself is a better cue: it sits still until impact and is gone afterwards.

Method (parameters set from geometry, not tuned on clips):

1. Search region: near the feet, extending toward the ball (the side the hands are on at
   address) out to ~3 torso lengths.
2. Candidates: small bright round blobs in a background image built from the *late backswing*
   (club far from the ball, ball still on the tee). At address the club head touches the ball
   and merges with it into one odd-shaped blob, so the address frames cannot be used. A ball is
   ~0.06-0.09 torso lengths across; candidates are sized 0.04-0.14.
3. Which blob is the ball: the one that looks like "ball there" before the swing and like
   "bare grass" (the same patch with the ball painted out) at the end. White shoes, alignment
   sticks and the tee do not turn into bare grass, and a leg crossing the spot matches neither
   state, so they fail. Brightness alone decides nothing.
4. Impact = the first frame, from the top of the swing on, where that patch stops looking like
   "ball there" and never does again. That is the club arriving at the ball, a frame or so before
   the ball is clearly gone.

If no blob passes (no visible ball, ball hidden, net, black mat, heavy blur), the result is
``None`` and the caller keeps the hand-based frame. Nothing is guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import warnings

import cv2
import numpy as np

from swing.io import iter_frames
from swing.pose.base import J, PoseSequence

BALL_DIAMETER_TORSO = (0.04, 0.14)  # ball diameter as a fraction of torso length, with margin
MAX_PEAKS = 5000  # compressed grass and shoes give hundreds of bright peaks; a ball can rank low
MIN_PEAK = 50.0  # gray levels a blob's peak must stand above its surroundings
MIN_DELTA = 25.0  # RMS gray levels the ball must differ from bare grass (dirt flecks do not)
STATE_FRACTION = 0.4  # a frame is "present"/"gone" if within this fraction of the full change
PRE_PRESENT = 0.9  # share of pre-swing frames that must look like the ball is there
POST_GONE = 0.7  # share of the final frames that must look like bare grass


@dataclass
class BallCue:
    frame: int | None  # first frame the ball patch changes for good; None if no ball found
    center: tuple[float, float] | None = None  # full-frame pixel coordinates
    radius: float | None = None
    score: float | None = None  # post-swing change minus pre-swing noise
    candidates: int = 0
    note: str = ""


# --------------------------------------------------------------------------- pure image logic

def find_candidates(gray: np.ndarray, torso: float, max_n: int = 10) -> list[tuple[float, float, float, float]]:
    """Small bright round blobs in a gray image as ``(cx, cy, radius, contrast)``, best first.

    ``gray`` is the (background) image of the search region; ``torso`` is in pixels. A blob's
    extent is taken at half of its own peak response above its surroundings (a fixed global
    threshold lets a bright ball merge with neighbouring bright grass into one huge region).
    """
    dmin, dmax = (f * torso for f in BALL_DIAMETER_TORSO)
    k = max(3, int(round(2.4 * dmax)) | 1)
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    kn = max(3, int(round(dmin)) | 1)
    peaks = (th == cv2.dilate(th, np.ones((kn, kn), np.uint8))) & (th > MIN_PEAK)
    ys, xs = np.nonzero(peaks)
    order = np.argsort(-th[ys, xs])[:MAX_PEAKS]
    amin, amax = np.pi / 4 * dmin**2 * 0.6, np.pi / 4 * dmax**2 * 1.4
    half_win = int(1.6 * dmax) + 2
    claimed = np.zeros(th.shape, bool)
    out = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if claimed[y, x]:
            continue
        y0, y1 = max(0, y - half_win), min(th.shape[0], y + half_win + 1)
        x0, x1 = max(0, x - half_win), min(th.shape[1], x + half_win + 1)
        sub = (th[y0:y1, x0:x1] > 0.5 * th[y, x]).astype(np.uint8)
        n, lab = cv2.connectedComponents(sub, connectivity=8)
        comp = lab == lab[y - y0, x - x0]
        claimed[y0:y1, x0:x1] |= comp
        yy, xx = np.nonzero(comp)
        w, h = xx.max() - xx.min() + 1, yy.max() - yy.min() + 1
        area = int(comp.sum())
        touches = xx.min() == 0 or yy.min() == 0 or xx.max() == comp.shape[1] - 1 or yy.max() == comp.shape[0] - 1
        interior = not touches or x0 == 0 or y0 == 0  # a blob cut by the window is bigger than a ball
        if touches and not (x0 == 0 or y0 == 0 or x1 == th.shape[1] or y1 == th.shape[0]):
            continue  # larger than the window: shoes, shafts, patches
        if not amin <= area <= amax:
            continue
        if not 0.6 <= w / max(h, 1) <= 1.7 or area / float(w * h) < 0.55:
            continue  # sticks, edges and irregular patches are not round
        contrast = float(th[y0:y1, x0:x1][comp].mean())
        out.append((float(xx.mean() + x0), float(yy.mean() + y0), float(np.sqrt(area / np.pi)), contrast))
    return sorted(out, key=lambda c: -c[3])[:max_n]


def _disk_mask(shape, cx, cy, r) -> np.ndarray:
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def ball_states(frames: list[np.ndarray], template: np.ndarray, cand, max_step: int = 3):
    """Per frame: does the candidate's disk look like the ball is there, gone, or neither?

    Two references are built from the template: the ball present (the template itself) and the
    ball gone (the disk filled with the median of the ring of grass around it). A frame is
    *present* if its disk matches the first, *gone* if it is close to the second, and neither if
    something else covers the spot (a leg, the club). Differences are RMS gray levels.

    The picture may drift (a hand-held or panning camera moved by ~10 px over one swing in a
    real clip), so the ball patch is *tracked*: each frame searches ``max_step`` pixels around
    where it matched in the previous frame, and the position is only updated on a good match, so
    it stays put while the club covers the ball or after the ball has left.

    Returns ``(present, gone, delta)`` where ``delta`` is the RMS difference between the two
    references.
    """
    cx, cy, r, _ = cand
    h, w = template.shape
    half = int(np.ceil(1.3 * r))
    x0, x1, y0, y1 = int(round(cx)) - half, int(round(cx)) + half + 1, int(round(cy)) - half, int(round(cy)) + half + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return np.zeros(len(frames), bool), np.zeros(len(frames), bool), 0.0  # too close to the edge
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    mask = ((xx ** 2 + yy ** 2) <= (1.3 * r) ** 2).astype(np.uint8)
    maskf = mask.astype(np.float32)
    n = float(mask.sum())
    t = template.astype(np.float32)
    ry, rx = np.ogrid[:h, :w]
    rr2 = (rx - cx) ** 2 + (ry - cy) ** 2
    ring = (rr2 >= (1.7 * r) ** 2) & (rr2 <= (2.6 * r) ** 2)
    fill = float(np.median(t[ring])) if ring.any() else float(np.median(t))
    ref = t[y0:y1, x0:x1]
    delta = float(np.sqrt((((ref - fill) ** 2) * mask).sum() / n))
    thr = STATE_FRACTION * delta
    present = np.zeros(len(frames), bool)
    gone = np.zeros(len(frames), bool)
    sx = sy = 0
    S = max_step
    for i, f in enumerate(frames):
        ya, yb, xa, xb = y0 + sy - S, y1 + sy + S, x0 + sx - S, x1 + sx + S
        if ya < 0 or xa < 0 or yb > h or xb > w:
            continue  # drifted out of the region: leave this and later frames "neither"
        win = f[ya:yb, xa:xb].astype(np.float32)
        ssd = cv2.matchTemplate(win, ref, cv2.TM_SQDIFF, mask=maskf)
        mn, _, loc, _ = cv2.minMaxLoc(ssd)
        if np.sqrt(mn / n) < thr:
            present[i] = True
            sx, sy = sx + loc[0] - S, sy + loc[1] - S  # follow the ball
        centre = f[y0 + sy:y1 + sy, x0 + sx:x1 + sx].astype(np.float32)
        gone[i] = float(np.sqrt((((centre - fill) ** 2) * maskf).sum() / n)) < thr
    return present, gone, delta


def judge_candidate(present: np.ndarray, gone: np.ndarray, delta: float, pre, post) -> tuple[bool, float]:
    """``(is_ball, score)``: present before the swing, bare grass at the end.

    ``pre`` and ``post`` index the series (slice or integer array). The score is the share of
    final frames that look like bare grass plus the share of pre-swing frames that look like the
    ball minus one, so 1.0 is a textbook ball and 0 is no evidence.
    """
    if delta < MIN_DELTA or not len(present[pre]) or not len(gone[post]):
        return False, 0.0
    pre_ok, post_gone = float(present[pre].mean()), float(gone[post].mean())
    return bool(pre_ok >= PRE_PRESENT and post_gone >= POST_GONE), pre_ok + post_gone - 1.0


def first_departure(present: np.ndarray, start: int, hold: int) -> int | None:
    """First index >= ``start`` that does not look like the ball is there, and never does again.

    "Never again" is checked over the next ``hold`` frames: the club arriving at the ball and the
    ball leaving both qualify; a one-frame flicker does not.
    """
    for t in range(start, len(present) - hold + 1):
        if not present[t] and not present[t:t + hold].any():
            return t
    return None


def locate_ball_impact(
    rois: list[np.ndarray], first_index: int, template: np.ndarray, torso: float,
    pre, post, start: int, hold: int,
) -> BallCue:
    """Core, on cropped gray frames: pick the blob that vanishes and report when it first departs.

    ``rois[i]`` is frame ``first_index + i``. ``pre`` and ``post`` index ``rois`` (a slice or an
    integer array): ``pre`` are frames where the ball is still on the tee (and that were not used
    to build ``template``), ``post`` the last stretch of the window; ``start`` is where to look
    for the departure.
    """
    cands = find_candidates(template, torso)
    best = None
    for c in cands:
        present, gone, delta = ball_states(rois, template, c)
        ok, score = judge_candidate(present, gone, delta, pre, post)
        if ok and (best is None or score > best[1]):
            best = (c, score, present)
    if best is None:
        return BallCue(None, candidates=len(cands), note=(
            "no blob looked like a ball that is there before the swing and bare grass after it"
            if cands else "no ball-sized bright round blob found near the feet"))
    c, score, present = best
    t = first_departure(present, start, hold)
    if t is None:
        return BallCue(None, candidates=len(cands), note="the ball patch never departed for good")
    return BallCue(first_index + t, (c[0], c[1]), c[2], score, len(cands))


# --------------------------------------------------------------------------- orchestration

def ball_roi(seq: PoseSequence, address: int, frame_shape: tuple[int, int, int]):
    """Search rectangle ``(x0, y0, x1, y1)`` and the torso length (px), or ``(None, None)``."""
    k = seq.keypoints[max(0, address - 10):address + 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # joints missing -> NaN -> no region
        ank = np.nanmedian(k[:, [J["left_ankle"], J["right_ankle"]], :2], axis=(0, 1))
        hands = np.nanmedian(k[:, [J["left_wrist"], J["right_wrist"]], :2], axis=(0, 1))
        hip = np.nanmedian(k[:, [J["left_hip"], J["right_hip"]], :2], axis=(0, 1))
        sh = np.nanmedian(k[:, [J["left_shoulder"], J["right_shoulder"]], :2], axis=(0, 1))
    torso = float(np.linalg.norm(sh - hip))
    if not (np.isfinite([*ank, *hands, *hip, torso]).all() and torso > 0):
        return None, None
    s = np.sign(hands[0] - hip[0])
    if s == 0:
        return None, None
    h, w = frame_shape[:2]
    xa, xb = ank[0] - 0.3 * torso * s, ank[0] + 3.0 * torso * s
    x0, x1 = int(max(0, min(xa, xb))), int(min(w, max(xa, xb)))
    y0, y1 = int(max(0, ank[1] - 0.8 * torso)), int(min(h, ank[1] + 0.6 * torso))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, None
    return (x0, y0, x1, y1), torso


def ball_cue(video, seq: PoseSequence, frames: dict[str, int | None]) -> BallCue:
    """Find the frame the ball leaves the tee, given the clip, its smoothed keypoints and phases.

    ``frames`` needs ``address``, ``takeaway``, ``top`` and ``impact`` (the hand-based estimate, used only to
    bound the window and to sanity-check the answer). Returns ``BallCue(None, ...)`` with a note
    when the picture does not support an answer.
    """
    a, top, imp = frames.get("address"), frames.get("top"), frames.get("impact")
    tk = frames.get("takeaway")
    if None in (a, tk, top, imp) or not a < tk < top < imp:
        return BallCue(None, note="needs address, takeaway, top and impact (in order) to know where to look")
    down, back = imp - top, top - tk
    # clean frames: the second half of the backswing, club well away from the ball
    w0, w1 = tk + int(0.5 * back), top - 2
    if w1 - w0 + 1 < 6:
        return BallCue(None, note=f"backswing too short ({back} frames) to get ball frames without the club")
    end = min(len(seq) - 1, imp + int(0.8 * down) + 5)
    first = w0
    roi = None
    crops: list[np.ndarray] = []
    for i, f in iter_frames(video, 0, end + 1):
        if roi is None:
            roi, torso = ball_roi(seq, a, f.shape)
            if roi is None:
                return BallCue(None, note="could not place a search region (feet or hands not tracked)")
        if i >= first:
            x0, y0, x1, y1 = roi
            crops.append(cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY))
    if not crops:
        return BallCue(None, note="no frames decoded")
    rois = crops
    win = np.arange(0, w1 - w0 + 1)  # indices (into rois) of the clean frames
    template = np.median(np.stack([rois[i] for i in win[::2]]), axis=0).astype(np.uint8)
    pre = win[1::2]  # the other half: unchanged ball frames that were not used for the template
    post = np.arange(len(rois) - max(3, int(0.3 * (len(rois) - (top - first)))), len(rois))
    hold = max(3, int(round(0.08 * down)))
    cue = locate_ball_impact(rois, first, template, torso, pre, post, top - first, hold)
    if cue.frame is not None:
        cue.center = (cue.center[0] + roi[0], cue.center[1] + roi[1])
        if abs(cue.frame - imp) > 0.6 * down:  # a ball cue far from the hand estimate is not trusted
            cue.note = (f"ball cue (frame {cue.frame}) is {abs(cue.frame - imp)} frames from the hand-based impact, "
                        f"more than 60% of the {down}-frame downswing: not trusted")
            cue.frame = None
    return cue


# --------------------------------------------------------------------------- using the cue

def refine_impact(phases, cue: BallCue):
    """Return ``(phases, note)`` with impact taken from the ball cue when that is sound.

    The cue replaces the hand-based impact only if it falls after the top (and before the
    finish, when there is one) and the corrected phases still look like a real swing. Otherwise
    the hand-based phases come back unchanged, with a note saying why.
    """
    from dataclasses import replace

    from swing.phases.detect import plausibility_issues

    if cue.frame is None:
        return phases, f"impact from the hands: {cue.note}" if cue.note else "impact from the hands"
    f = dict(phases.frames)
    top, fin = f.get("top"), f.get("finish")
    if top is None or not cue.frame > top or (fin is not None and not cue.frame < fin):
        return phases, f"ball cue (frame {cue.frame}) is not between the top and the finish: impact from the hands"
    f["impact"] = cue.frame
    issues = plausibility_issues(f, phases.fps)
    if issues:
        return phases, "ball cue gives timings that cannot be a real swing (" + "; ".join(issues) + "): impact from the hands"
    return replace(phases, frames=f), f"impact from the ball leaving the tee (hand-based was {phases.frames.get('impact')})"
