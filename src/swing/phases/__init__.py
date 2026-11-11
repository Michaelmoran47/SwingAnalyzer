"""Rule-based swing phase detection."""

from swing.phases.detect import (PhaseParams, PhaseResult, detect_phases,
                                 detect_phases_from_hands)
from swing.phases.evaluate import PhaseError, compare

__all__ = ["PhaseParams", "PhaseResult", "PhaseError", "compare", "detect_phases",
           "detect_phases_from_hands"]
