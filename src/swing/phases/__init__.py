"""Rule-based swing phase detection."""

from swing.phases.detect import (PhaseParams, PhaseResult, detect_phases,
                                 detect_phases_from_hands, plausibility_issues)
from swing.phases.evaluate import EXPERIMENTAL_PHASES, PhaseError, VALIDATED_PHASES, compare

__all__ = ["PhaseParams", "PhaseResult", "PhaseError", "EXPERIMENTAL_PHASES", "VALIDATED_PHASES", "compare", "detect_phases",
           "detect_phases_from_hands", "plausibility_issues"]
