"""Compare detected phase frames with hand labels."""

from __future__ import annotations

from dataclasses import dataclass

from swing.labels import PHASES, PhaseLabels
from swing.phases.detect import PhaseResult


@dataclass(frozen=True)
class PhaseError:
    phase: str
    detected: int | None
    labelled: int | None
    error_frames: int | None  # detected - labelled (positive = detected late)
    error_ms: float | None


def compare(result: PhaseResult, labels: PhaseLabels) -> list[PhaseError]:
    out = []
    for p in PHASES:
        d, l = result.frames.get(p), labels.labels.get(p)
        err = None if d is None or l is None else d - l
        out.append(PhaseError(p, d, l, err, None if err is None else 1000.0 * err / result.fps))
    return out
