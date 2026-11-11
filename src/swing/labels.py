"""Hand-labelled phase frames: the ground truth phase detection is validated on.

Frame indices refer to the decoded stream (the same indices ``swing.io.iter_frames``
yields), not to header frame numbers. One JSON file per clip.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

PHASES: tuple[str, ...] = ("address", "takeaway", "top", "impact", "finish")


@dataclass
class PhaseLabels:
    video: str
    fps: float
    frame_count: int
    labels: dict[str, int] = field(default_factory=dict)
    notes: str = ""
    _history: list[tuple[str, int | None]] = field(default_factory=list, repr=False)

    def set(self, phase: str, frame: int) -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}")
        if not 0 <= frame < self.frame_count:
            raise ValueError(f"frame {frame} outside 0..{self.frame_count - 1}")
        self._history.append((phase, self.labels.get(phase)))
        self.labels[phase] = frame

    def clear(self, phase: str) -> None:
        if phase in self.labels:
            self._history.append((phase, self.labels.pop(phase)))

    def undo(self) -> bool:
        if not self._history:
            return False
        phase, previous = self._history.pop()
        if previous is None:
            self.labels.pop(phase, None)
        else:
            self.labels[phase] = previous
        return True

    def phase_at(self, frame: int) -> str | None:
        for p, f in self.labels.items():
            if f == frame:
                return p
        return None

    def problems(self) -> list[str]:
        """Human-readable issues: out-of-order phases. Empty list means fine."""
        out = []
        present = [(p, self.labels[p]) for p in PHASES if p in self.labels]
        for (p1, f1), (p2, f2) in zip(present, present[1:]):
            if f2 <= f1:
                out.append(f"{p2} ({f2}) is not after {p1} ({f1})")
        return out

    @property
    def complete(self) -> bool:
        return all(p in self.labels for p in PHASES) and not self.problems()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "video": self.video,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "labels": {p: self.labels[p] for p in PHASES if p in self.labels},
            "notes": self.notes,
        }
        path.write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "PhaseLabels":
        d = json.loads(Path(path).read_text())
        unknown = set(d["labels"]) - set(PHASES)
        if unknown:
            raise ValueError(f"unknown phases in {path}: {sorted(unknown)}")
        return cls(d["video"], float(d["fps"]), int(d["frame_count"]),
                   {k: int(v) for k, v in d["labels"].items()}, d.get("notes", ""))


def default_label_path(video: str | Path, labels_dir: str | Path = "data/labels") -> Path:
    return Path(labels_dir) / f"{Path(video).stem}.json"
