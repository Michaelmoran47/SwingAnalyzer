"""Clips whose timeline is too slow for the phase rules (found when a 30 fps re-timed clip was uploaded).

IMG_6003 was a ~240 fps swing stamped as 29.9 fps over 65 s. Phase detection ran anyway and put top
and impact on the same frame. It must refuse instead, and a stated capture rate must work.
"""

from dataclasses import replace

import numpy as np
import pytest

from swing.cli import main
from swing.demo import analyze, phase_table
from swing.phases import PhaseParams, detect_phases, detect_phases_from_hands
from swing.pose import PoseSequence
from test_demo import APP, _scaled_seq, demo_dirs  # noqa: F401  (fixture re-used)
from test_phases import synthetic_hands

SLOW = 29.9


def _slow_seq():
    seq, truth = _scaled_seq()
    return replace(seq, fps=SLOW), truth


def test_detector_refuses_below_min_fps_and_says_how_to_proceed():
    hands, _ = synthetic_hands()
    r = detect_phases_from_hands(hands, SLOW, 400.0)
    assert all(v is None for v in r.frames.values())
    assert "29.9" in r.warnings[0] and "--fps" in r.warnings[0]


def test_detector_still_runs_at_the_minimum_rate():
    hands, _ = synthetic_hands()
    r = detect_phases_from_hands(hands, PhaseParams().min_fps, 400.0)
    assert r.frames["top"] is not None  # 100 fps is allowed


def test_metrics_explain_that_the_frame_rate_is_the_reason():
    a = analyze(*(_slow_seq()[0], "dtl"))
    assert all(v is None for v in a.phases.frames.values())
    for name in ("tempo_ratio", "spine_angle_address_deg", "hip_toward_ball_impact_torso"):
        assert a.metrics.values[name] is None
        assert "frame rate" in a.metrics.unavailable[name]


def test_capture_fps_override_gives_the_same_answer_as_a_correctly_stamped_clip():
    slow, truth = _slow_seq()
    right, _ = _scaled_seq()
    fixed = analyze(slow, "dtl", capture_fps=240.0)
    ref = analyze(right, "dtl")
    assert fixed.phases.frames == ref.phases.frames
    assert abs(fixed.phases.frames["top"] - truth["top"]) <= 3
    assert fixed.stamped_fps == SLOW and fixed.smooth.fps == 240.0
    assert fixed.metrics.values["tempo_ratio"] == pytest.approx(ref.metrics.values["tempo_ratio"])
    assert ref.stamped_fps is None


def test_override_equal_to_stamped_rate_is_not_flagged_as_an_override():
    right, _ = _scaled_seq()
    assert analyze(right, "dtl", capture_fps=right.fps).stamped_fps is None


def test_cli_refuses_then_accepts_fps(tmp_path, capsys):
    slow, truth = _slow_seq()
    path = tmp_path / "slow.npz"
    slow.save(path)
    assert main(["phases", str(path)]) == 1
    out = capsys.readouterr().out
    assert "not detected" in out and "--fps" in out
    assert main(["phases", str(path), "--fps", "240"]) == 0
    out = capsys.readouterr().out
    top_line = next(line for line in out.splitlines() if line.startswith("top"))
    assert abs(int(top_line.split()[1]) - truth["top"]) <= 3
    assert main(["metrics", str(path), "--view", "dtl", "--fps", "240"]) == 0
    assert "tempo_ratio" in capsys.readouterr().out


# ---- the real app -----------------------------------------------------------

@pytest.fixture
def slow_dirs(demo_dirs):  # noqa: F811
    base, truth = demo_dirs
    slow, _ = _slow_seq()
    slow.save(base / "outputs" / "swing.npz")
    return base, truth


def _run():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(APP), default_timeout=180).run()


def _fps_box(at):
    return next(s for s in at.sidebar.selectbox if s.label == "Capture frame rate")


def test_app_shows_error_and_computes_nothing_at_a_low_frame_rate(slow_dirs):
    at = _run()
    assert not at.exception
    assert any("not computed" in e.value for e in at.error)
    phases = at.dataframe[0].value.set_index("phase")
    assert phases["frame"].isna().all()


def test_app_capture_rate_control_unlocks_phases_and_says_it_is_your_claim(slow_dirs):
    _, truth = slow_dirs
    at = _run()
    _fps_box(at).set_value("240").run()
    assert not at.exception
    assert not at.error
    assert any("Timings assume" in i.value and "240" in i.value for i in at.info)
    phases = at.dataframe[0].value.set_index("phase")
    assert abs(phases.loc["top", "frame"] - truth["top"]) <= 3
    assert any("(set by you)" in c.value for c in at.caption)


def test_app_normal_clip_has_no_error_or_override_notice(demo_dirs):  # noqa: F811
    at = _run()
    assert not at.error and not any("Timings assume" in i.value for i in at.info)
