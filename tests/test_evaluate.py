import pytest

from swing.labels import PhaseLabels
from swing.phases import VALIDATED_PHASES, PhaseResult, compare


def test_errors_and_validated_flag():
    res = PhaseResult({"address": 100, "takeaway": 120, "top": 300, "impact": 360, "finish": None},
                      fps=240.0)
    lab = PhaseLabels("v", 240.0, 1000,
                      {"address": 90, "takeaway": 110, "top": 297, "impact": 364, "finish": 500})
    rows = {e.phase: e for e in compare(res, lab)}
    assert rows["top"].error_frames == 3 and rows["top"].error_ms == pytest.approx(12.5)
    assert rows["impact"].error_frames == -4
    assert rows["finish"].error_frames is None and rows["finish"].error_ms is None
    assert {p for p, e in rows.items() if e.validated} == set(VALIDATED_PHASES) == {"top", "impact"}


def test_missing_label_gives_no_error():
    res = PhaseResult({"top": 300}, fps=240.0)
    rows = {e.phase: e for e in compare(res, PhaseLabels("v", 240.0, 1000))}
    assert rows["top"].error_frames is None
