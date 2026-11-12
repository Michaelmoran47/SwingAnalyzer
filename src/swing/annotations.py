"""Hand-clicked wrist positions on chosen frames: a small keypoint ground truth.

``labels.py`` records *when* phases happen. This records *where* the wrists are, so a
backend's wrist can be scored as a distance to a human's click instead of judged by eye.

Pixel coordinates are in the decoded frame (after orientation), the same space as the
keypoint ``.npz`` files. Frame indices are decoded-stream indices, as in ``labels.py``.
Each click carries ``visible``: False means "I can see no hand here (hidden behind the
body), this click is my best guess". Hidden guesses are scored separately because they
are not ground truth.

Scope: wrists only, a few dozen frames of one swing, one annotator. It measures how far a
backend's wrist is from where one person put it. It is not an accuracy benchmark.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from swing.pose import J, PoseSequence

WRISTS: tuple[str, ...] = ("left_wrist", "right_wrist")  # the golfer's own left / right


@dataclass
class Point:
    x: float
    y: float
    visible: bool = True


@dataclass
class WristAnnotations:
    video: str
    fps: float
    frame_count: int
    points: dict[int, dict[str, Point]] = field(default_factory=dict)
    notes: str = ""
    _history: list[tuple[int, str, Point | None]] = field(default_factory=list, repr=False)

    def _check(self, frame: int, wrist: str) -> None:
        if wrist not in WRISTS:
            raise ValueError(f"unknown wrist {wrist!r}")
        if not 0 <= frame < self.frame_count:
            raise ValueError(f"frame {frame} outside 0..{self.frame_count - 1}")

    def get(self, frame: int, wrist: str) -> Point | None:
        return self.points.get(frame, {}).get(wrist)

    def set(self, frame: int, wrist: str, x: float, y: float, visible: bool = True) -> None:
        self._check(frame, wrist)
        self._history.append((frame, wrist, self.get(frame, wrist)))
        self.points.setdefault(frame, {})[wrist] = Point(float(x), float(y), visible)

    def set_visible(self, frame: int, wrist: str, visible: bool) -> bool:
        """Flip the visible flag of an existing click; False if there is no click."""
        p = self.get(frame, wrist)
        if p is None:
            return False
        self.set(frame, wrist, p.x, p.y, visible)
        return True

    def clear(self, frame: int, wrist: str) -> None:
        p = self.get(frame, wrist)
        if p is None:
            return
        self._history.append((frame, wrist, p))
        del self.points[frame][wrist]
        if not self.points[frame]:
            del self.points[frame]

    def undo(self) -> bool:
        if not self._history:
            return False
        frame, wrist, previous = self._history.pop()
        if previous is None:
            self.points.get(frame, {}).pop(wrist, None)
            if frame in self.points and not self.points[frame]:
                del self.points[frame]
        else:
            self.points.setdefault(frame, {})[wrist] = previous
        return True

    def complete_frames(self) -> list[int]:
        """Frames with both wrists clicked."""
        return sorted(f for f, d in self.points.items() if all(w in d for w in WRISTS))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "video": self.video,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "notes": self.notes,
            "wrists": {
                str(f): {w: {"x": round(p.x, 1), "y": round(p.y, 1), "visible": p.visible}
                         for w, p in ((w, self.points[f][w]) for w in WRISTS if w in self.points[f])}
                for f in sorted(self.points)
            },
        }
        path.write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "WristAnnotations":
        d = json.loads(Path(path).read_text())
        out = cls(d["video"], float(d["fps"]), int(d["frame_count"]), notes=d.get("notes", ""))
        for f, wd in d["wrists"].items():
            unknown = set(wd) - set(WRISTS)
            if unknown:
                raise ValueError(f"unknown wrists in {path}: {sorted(unknown)}")
            out.points[int(f)] = {w: Point(float(p["x"]), float(p["y"]), bool(p["visible"]))
                                  for w, p in wd.items()}
        return out


def default_annotation_path(video: str | Path, annotations_dir: str | Path = "data/annotations") -> Path:
    return Path(annotations_dir) / f"{Path(video).stem}_wrists.json"


def sample_frames(start: int, end: int, step: int) -> list[int]:
    """Frames ``start, start+step, ...`` up to and including ``end``. Fixed by the rule
    alone, never by what any pose model output, so the sample is not picked to flatter one."""
    if step < 1:
        raise ValueError("step must be >= 1")
    return list(range(start, end + 1, step))


@dataclass
class WristError:
    frame: int
    wrist: str
    visible: bool
    px: float  # distance between the estimate and the click; NaN if the backend had no wrist


def wrist_errors(ann: WristAnnotations, seq: PoseSequence) -> list[WristError]:
    """Distance (px) from each clicked wrist to the backend's *raw* wrist, per clicked frame.

    Frames outside the sequence's range, and wrists the backend did not output, give NaN.
    """
    out = []
    for f in sorted(ann.points):
        for w, p in ann.points[f].items():
            i = f - seq.start_frame
            est = seq.keypoints[i, J[w], :2] if 0 <= i < len(seq) else np.full(2, np.nan)
            out.append(WristError(f, w, p.visible,
                                  float(np.hypot(est[0] - p.x, est[1] - p.y))))
    return out


def summarise(errors: list[WristError], torso_px: float) -> dict[str, float]:
    """Counts and distance statistics (torso lengths) over the errors given."""
    px = np.array([e.px for e in errors], float)
    ok = px[np.isfinite(px)]
    res: dict[str, float] = {"n": len(px), "missing": int(len(px) - len(ok))}
    if len(ok):
        t = ok / torso_px
        res.update(median=float(np.median(t)), mean=float(t.mean()),
                   p90=float(np.percentile(t, 90)), max=float(t.max()))
    return res
