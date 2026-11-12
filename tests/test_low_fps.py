"""Low frame rates and time-stretched clips.

Two different things were confused at first:

* A genuinely low frame rate (a normal 30 fps phone video). The rules cope: see
  scripts/framerate_sweep.py. The detector runs down to ``PhaseParams.min_fps`` (15).
* A *time-stretched* clip: 240 fps footage stamped as 30 fps (IMG_6003). The swing then spans
  ~8x too many frames for the millisecond-scale windows, top and impact landed on one frame,
  and nothing said so. The detector now flags timings that cannot be a real-time swing.
"""

from dataclasses import replace

import numpy as np
import pytest

from swing.cli import main
from swing.demo import analyze
from swing.phases import PhaseParams, detect_phases_from_hands, plausibility_issues
from swing.pose import PoseSequence
from test_demo import APP, _scaled_seq, demo_dirs  # noqa: F401  (fixture re-used)
from test_phases import synthetic_hands

STAMPED = 29.9  # what the re-timed file says


def _stretched_seq():
    """A 240 fps-scale swing whose file says 29.9 fps: the IMG_6003 situation."""
    seq, truth = _scaled_seq()
    return replace(seq, fps=STAMPED), truth


# ---- genuinely low frame rates work -----------------------------------------

@pytest.mark.parametrize("step,fps", [(8, 30.0), (16, 15.0)])
def test_real_time_swing_at_low_frame_rate_is_detected_within_a_frame(step, fps):
    hands, truth = synthetic_hands()
    sub = hands[::step]
    r = detect_phases_from_hands(sub, fps, 400.0)
    assert r.plausible and r.warnings == []
    for phase in ("top", "impact"):
        assert abs(r.frames[phase] * step - truth[phase]) <= step + 1  # within about one frame
    f = r.frames
    assert f["address"] < f["takeaway"] < f["top"] < f["impact"] < f["finish"]


def test_detector_does_not_run_below_the_tested_floor():
    hands, _ = synthetic_hands()
    r = detect_phases_from_hands(hands[::24], 10.0, 400.0)
    assert all(v is None for v in r.frames.values()) and not r.plausible
    assert "10.0" in r.warnings[0] and "--fps" in r.warnings[0]


def test_detector_runs_at_exactly_the_floor():
    hands, _ = synthetic_hands()
    r = detect_phases_from_hands(hands[::16], PhaseParams().min_fps, 400.0)
    assert r.frames["top"] is not None


# ---- time-stretched clips are flagged, and a stated capture rate fixes them --

def test_time_stretched_clip_is_flagged_not_silently_analysed():
    a = analyze(_stretched_seq()[0], "dtl")
    assert not a.phases.plausible
    msg = a.phases.warnings[-1]
    assert "real-time swing" in msg and "--fps" in msg
    assert a.metrics.values["tempo_ratio"] is None  # not reported as if it were real


def test_capture_fps_override_gives_the_same_answer_as_a_correctly_stamped_clip():
    stretched, truth = _stretched_seq()
    right, _ = _scaled_seq()
    fixed = analyze(stretched, "dtl", capture_fps=240.0)
    ref = analyze(right, "dtl")
    assert fixed.phases.plausible and fixed.phases.frames == ref.phases.frames
    assert abs(fixed.phases.frames["top"] - truth["top"]) <= 3
    assert fixed.stamped_fps == STAMPED and fixed.smooth.fps == 240.0
    assert fixed.metrics.values["tempo_ratio"] == pytest.approx(ref.metrics.values["tempo_ratio"])
    assert ref.stamped_fps is None and ref.phases.plausible


def test_override_equal_to_stamped_rate_is_not_flagged_as_an_override():
    right, _ = _scaled_seq()
    assert analyze(right, "dtl", capture_fps=right.fps).stamped_fps is None


# ---- the plausibility rule itself -------------------------------------------

def test_plausibility_rule_catches_each_failure_and_passes_a_normal_swing():
    ok = {"address": 100, "takeaway": 140, "top": 250, "impact": 320, "finish": 380}
    assert plausibility_issues(ok, 240.0) == []
    same = dict(ok, impact=250)  # top == impact (the IMG_6003 symptom)
    assert "share a frame" in plausibility_issues(same, 240.0)[0]
    assert any("backswing" in i for i in plausibility_issues(ok, 30.0))  # 110 frames at 30 fps = 3.7 s
    slow_down = {"address": 0, "takeaway": 10, "top": 40, "impact": 100, "finish": 120}
    assert any("downswing" in i for i in plausibility_issues(slow_down, 30.0))  # 2 s
    assert plausibility_issues({"address": None, "takeaway": None, "top": None,
                                "impact": None, "finish": None}, 30.0) == []


def test_metrics_say_why_when_the_detector_did_not_run(tmp_path):
    hands, _ = synthetic_hands()
    seq, _ = _scaled_seq()
    a = analyze(replace(seq, fps=10.0), "dtl")
    for name in ("tempo_ratio", "spine_angle_address_deg", "hip_toward_ball_impact_torso"):
        assert a.metrics.values[name] is None
        assert "frame rate" in a.metrics.unavailable[name]


def test_cli_flags_a_stretched_clip_then_accepts_fps(tmp_path, capsys):
    stretched, truth = _stretched_seq()
    path = tmp_path / "stretched.npz"
    stretched.save(path)
    main(["phases", str(path)])
    out = capsys.readouterr().out
    assert "real-time swing" in out and "--fps" in out
    assert main(["phases", str(path), "--fps", "240"]) == 0
    out = capsys.readouterr().out
    assert "real-time swing" not in out
    top_line = next(line for line in out.splitlines() if line.startswith("top"))
    assert abs(int(top_line.split()[1]) - truth["top"]) <= 3
    assert main(["metrics", str(path), "--view", "dtl", "--fps", "240"]) == 0
    assert "tempo_ratio" in capsys.readouterr().out


# ---- the real app -----------------------------------------------------------

@pytest.fixture
def stretched_dirs(demo_dirs):  # noqa: F811
    base, truth = demo_dirs
    stretched, _ = _stretched_seq()
    stretched.save(base / "outputs" / "swing.npz")
    return base, truth


def _run():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(APP), default_timeout=180).run()


def _fps_box(at):
    return next(s for s in at.sidebar.selectbox if s.label == "Capture frame rate")


def test_app_warns_that_a_stretched_clip_looks_slowed_down(stretched_dirs):
    at = _run()
    assert not at.exception
    assert any("real-time swing" in w.value and "Capture frame rate" in w.value for w in at.warning)


def test_app_capture_rate_control_clears_the_warning_and_says_it_is_your_claim(stretched_dirs):
    _, truth = stretched_dirs
    at = _run()
    _fps_box(at).set_value("240").run()
    assert not at.exception
    assert not any("real-time swing" in w.value for w in at.warning)
    assert any("Timings assume" in i.value and "240" in i.value for i in at.info)
    phases = at.dataframe[0].value.set_index("phase")
    assert abs(phases.loc["top", "frame"] - truth["top"]) <= 3
    assert any("(set by you)" in c.value for c in at.caption)


def test_app_normal_clip_shows_no_slowdown_warning_or_override_notice(demo_dirs):  # noqa: F811
    at = _run()
    assert not any("real-time swing" in w.value for w in at.warning)
    assert not at.error and not any("Timings assume" in i.value for i in at.info)
