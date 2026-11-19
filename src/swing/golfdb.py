"""Reading GolfDB's labels, converting them to this project's labels, and scoring against them.

GolfDB (McNally et al., CVPR Workshops 2019; https://github.com/wmcnally/golfdb) labels 1,400 swing
clips cut from 580 YouTube videos with eight events. Only the label file (``golfDB.mat``) is used
here; the videos are other people's uploads and are not part of this repository. See docs/golfdb.md.

Event order in ``events``: start, address, toe-up, mid-backswing, top, mid-downswing, impact,
mid-follow-through, finish, end. Frame numbers refer to the source video sampled at 30 fps.

Mapping to this project's phases (see the paper's definitions):

* address = GolfDB address ("the frame before movement in the backswing is noticeable") and
  takeaway = the next frame, because here address is the last still frame and takeaway the first
  frame after it;
* shaft_parallel = toe-up ("shaft parallel with the ground during the backswing");
* top = top ("the moment the golf club changes directions");
* impact = impact ("the moment the clubhead touches the golf ball").

Finish is **not** mapped: GolfDB's finish is the moment *before the final pose is relaxed*, the end
of the held finish, where this project's is the start of it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from swing.labels import PhaseLabels

EVENT_NAMES = ("start", "address", "toe_up", "mid_backswing", "top", "mid_downswing", "impact",
               "mid_follow_through", "finish", "end")
GOLFDB_FPS = 30.0
SCORED = ("takeaway", "shaft_parallel", "top", "impact")  # phases with a sound GolfDB counterpart


@dataclass(frozen=True)
class GolfDBSample:
    id: int
    youtube_id: str
    player: str
    sex: str
    club: str
    view: str  # "down-the-line", "face-on" or "other"
    slow: bool  # slow motion (re-timed to 30 fps) rather than real time
    events: tuple[int, ...]  # the ten event frames, in source-video frames at 30 fps
    bbox: tuple[float, ...]
    split: int

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.youtube_id}"

    @property
    def address_to_impact(self) -> int:
        return self.events[6] - self.events[1]


def load_golfdb(path: str | Path) -> list[GolfDBSample]:
    """Read ``golfDB.mat``. Uses SciPy's data-only MATLAB reader; no code in the file is run."""
    from scipy.io import loadmat

    raw = loadmat(str(path))["golfDB"][0]
    out = []
    for r in raw:
        out.append(GolfDBSample(
            id=int(r["id"].ravel()[0]), youtube_id=str(r["youtube_id"].ravel()[0]),
            player=str(r["player"].ravel()[0]), sex=str(r["sex"].ravel()[0]), club=str(r["club"].ravel()[0]),
            view=str(r["view"].ravel()[0]), slow=bool(int(r["slow"].ravel()[0])),
            events=tuple(int(v) for v in r["events"].ravel()), bbox=tuple(float(v) for v in r["bbox"].ravel()),
            split=int(r["split"].ravel()[0])))
    return out


def valid(sample: GolfDBSample) -> bool:
    """Ten events, in non-decreasing order (one sample in the dataset is not)."""
    e = sample.events
    return len(e) == 10 and all(b >= a for a, b in zip(e, e[1:]))


# --------------------------------------------------------------------------- frame bookkeeping

def to_native(frame30: float, native_fps: float) -> int:
    """GolfDB frame (30 fps) -> frame index in a video that runs at ``native_fps``."""
    return int(round(frame30 * native_fps / GOLFDB_FPS))


def to_30(frame_native: float, native_fps: float) -> int:
    """Frame index in a ``native_fps`` video -> GolfDB's 30 fps frame number."""
    return int(round(frame_native * GOLFDB_FPS / native_fps))


def phase_frames(sample: GolfDBSample, relative_to_start: bool = False) -> dict[str, int]:
    """This project's phases from a sample's events, in 30 fps frames.

    ``relative_to_start`` subtracts the sample's start frame, for a clip trimmed to the sample.
    """
    off = sample.events[0] if relative_to_start else 0
    e = sample.events
    return {"address": e[1] - off, "takeaway": e[1] + 1 - off, "shaft_parallel": e[2] - off,
            "top": e[4] - off, "impact": e[6] - off}


def to_labels(sample: GolfDBSample, video_name: str, fps: float, frame_count: int,
              relative_to_start: bool = False) -> PhaseLabels:
    """A ``PhaseLabels`` for a video holding this sample, with the mapped phases (not finish)."""
    lab = PhaseLabels(video_name, fps, frame_count)
    for phase, f in phase_frames(sample, relative_to_start).items():
        if 0 <= f < frame_count:
            lab.set(phase, f)
    lab.notes = (f"GolfDB sample {sample.id} ({sample.player}, {sample.club}, {sample.view}, "
                 f"{'slow-motion' if sample.slow else 'real-time'}); labels by the GolfDB annotators, "
                 "mapped to this project's phases. Finish is not mapped (different definition). Frames are "
                 f"{'relative to the sample start' if relative_to_start else 'GolfDB source-video frames'} at 30 fps.")
    return lab


# --------------------------------------------------------------------------- shortlist

CELLS = (("real-time", "down-the-line"), ("real-time", "face-on"),
         ("slow-motion", "down-the-line"), ("slow-motion", "face-on"))


def _cell(sample: GolfDBSample) -> tuple[str, str]:
    return ("slow-motion" if sample.slow else "real-time", sample.view)


def shortlist(samples: list[GolfDBSample], clips_per_cell: int = 6, min_slow_frames: int = 0) -> list[dict]:
    """Videos to consider, per (speed, view) cell, taking the videos with the most qualifying clips.

    Only down-the-line and face-on clips qualify ("other" views are skipped), and ``min_slow_frames``
    can require slow-motion clips to span at least that many frames from address to impact (more
    frames, more temporal detail). Within a cell the videos with the most clips come first, so few
    videos cover the target; ties break on the YouTube id, so the list is deterministic.
    """
    out = []
    for cell in CELLS:
        by_video: dict[str, list[GolfDBSample]] = {}
        for s in samples:
            if not valid(s) or _cell(s) != cell:
                continue
            if s.slow and s.address_to_impact < min_slow_frames:
                continue
            by_video.setdefault(s.youtube_id, []).append(s)
        ranked = sorted(by_video.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        got = 0
        for vid, clips in ranked:
            if got >= clips_per_cell:
                break
            got += len(clips)
            out.append({
                "cell": f"{cell[0]}, {cell[1]}", "youtube_id": vid, "url": clips[0].url, "player": clips[0].player,
                "clubs": sorted({c.club for c in clips}), "clips": len(clips), "sample_ids": [c.id for c in clips],
                "address_to_impact_frames": sorted(c.address_to_impact for c in clips),
            })
    return out


# --------------------------------------------------------------------------- scoring

def score_sample(sample: GolfDBSample, detected_native: dict[str, int | None], native_fps: float,
                 relative_to_start: bool = False) -> dict[str, int | None]:
    """Per scored phase: detected minus GolfDB label, in 30 fps frames (``None`` if not detected).

    ``detected_native`` holds this project's phase frames as indices into the video that was
    analysed (running at ``native_fps``).
    """
    label = phase_frames(sample, relative_to_start)
    out = {}
    for p in SCORED:
        d = detected_native.get(p)
        out[p] = None if d is None else to_30(d, native_fps) - label[p]
    return out


def aggregate(rows: list[dict]) -> dict:
    """Summarise per-clip errors.

    ``rows``: dicts with ``slow`` (bool) and ``errors`` (phase -> frames or None). Returns, per group
    (``all``, ``real-time``, ``slow-motion``) and phase: n scored, n missing, median signed error
    (bias), median and mean absolute error, and the share within 2 and 5 frames.
    """
    groups = {"all": rows, "real-time": [r for r in rows if not r["slow"]], "slow-motion": [r for r in rows if r["slow"]]}
    out = {}
    for g, rs in groups.items():
        out[g] = {}
        for p in SCORED:
            vals = [r["errors"].get(p) for r in rs]
            have = np.array([v for v in vals if v is not None], float)
            missing = sum(v is None for v in vals)
            if len(have) == 0:
                out[g][p] = {"n": 0, "missing": missing}
                continue
            a = np.abs(have)
            out[g][p] = {"n": int(len(have)), "missing": int(missing), "bias": float(np.median(have)),
                         "median_abs": float(np.median(a)), "mean_abs": float(a.mean()),
                         "within_2": float((a <= 2).mean()), "within_5": float((a <= 5).mean())}
    return out


def dataset_summary(samples: list[GolfDBSample]) -> dict:
    """Counts used to judge what the dataset can supply."""
    return {
        "samples": len(samples), "videos": len({s.youtube_id for s in samples}),
        "players": len({s.player for s in samples}),
        "slow": dict(Counter(s.slow for s in samples)), "view": dict(Counter(s.view for s in samples)),
        "club": dict(Counter(s.club for s in samples)), "invalid": sum(not valid(s) for s in samples),
    }


# --------------------------------------------------------------------------- analysing a sample

def detect_for_sample(seq, slow: bool, slow_factor: float = 4.0) -> dict[str, int | None]:
    """Run smoothing and phase detection on a sample's keypoints; return phases in the analysed video's frames.

    ``seq`` is the ``PoseSequence`` of the sample's window of the video (``start_frame`` set). A
    slow-motion sample is a clip re-timed to play slowly, so its real capture rate is higher than
    the file says: the stamped rate is multiplied by ``slow_factor``. GolfDB's own label counts
    put the typical slowdown near 4x (median address-to-impact 132 frames against 34 for
    real-time clips); that is a prior, not a measurement of any one clip.
    """
    from dataclasses import replace

    from swing.phases import detect_phases
    from swing.smoothing import smooth_sequence

    fps = seq.fps * (slow_factor if slow else 1.0)
    smooth = smooth_sequence(replace(seq, fps=fps))
    return dict(detect_phases(smooth).frames)
