"""Compare detected phase frames with hand labels."""

from __future__ import annotations

from dataclasses import dataclass

from swing.labels import PHASES, PhaseLabels
from swing.phases.detect import PhaseResult


# Phases scored against hand labels. Address and finish stay threshold-defined (see
# docs/phase_definitions.md): they are gradual events. Takeaway joined once it was detected
# from whole-body motion and labelled on three swings.
VALIDATED_PHASES: tuple[str, ...] = ("takeaway", "top", "impact")
# Image-based estimates with no labels behind them yet: shown, compared if labelled, never called scored.
EXPERIMENTAL_PHASES: tuple[str, ...] = ("shaft_parallel",)


@dataclass(frozen=True)
class PhaseError:
    phase: str
    detected: int | None
    labelled: int | None
    error_frames: int | None  # detected - labelled (positive = detected late)
    error_ms: float | None
    validated: bool = False
    acceptable: tuple[int, int] | None = None  # a labelled range: any frame inside is correct

    @property
    def label_text(self) -> str:
        if self.acceptable:
            return f"{self.acceptable[0]}-{self.acceptable[1]}"
        return "-" if self.labelled is None else str(self.labelled)


def compare(result: PhaseResult, labels: PhaseLabels) -> list[PhaseError]:
    """Per phase: detected minus label. With a labelled range the error is 0 inside it, else the
    distance to its nearest edge."""
    out = []
    for p in PHASES:
        d, l, rng = result.frames.get(p), labels.labels.get(p), labels.ranges.get(p)
        if d is None or (l is None and rng is None):
            err = None
        elif rng is not None:
            err = 0 if rng[0] <= d <= rng[1] else (d - rng[0] if d < rng[0] else d - rng[1])
        else:
            err = d - l
        out.append(PhaseError(p, d, l, err, None if err is None else 1000.0 * err / result.fps,
                              p in VALIDATED_PHASES, rng))
    return out
