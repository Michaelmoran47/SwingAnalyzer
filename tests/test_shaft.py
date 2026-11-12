"""Shaft-parallel in the backswing: image line finder, crossing logic, and the plumbing around it."""

import cv2
import numpy as np
import pytest

from swing.demo import Analysis, apply_shaft_cue, phase_table, shaft_cue_cached
from swing.labels import CORE_PHASES, PHASES, PhaseLabels
from swing.phases import PhaseResult
from swing.pose import J, PoseSequence, empty_pose
from swing.shaft import (ShaftCue, crossing_from_readings, isotonic_increasing, refine_shaft,
                         shaft_cue, shaft_elevation)
from test_demo import APP, demo_dirs  # noqa: F401  (fixture re-used by the app test)

TORSO = 100.0
HANDS = (300.0, 300.0)
ELBOW = (220.0, 300.0)  # to the left, so a shaft pointing right is "away from the forearm"


def _img(seed=0):
    rng = np.random.default_rng(seed)
    g = np.clip(80 + rng.normal(0, 3, (480, 640)), 0, 255)
    return g


def _frame_with_shaft(elevation_deg, length=80, extra=None, seed=0):
    g = _img(seed)
    if elevation_deg is not None:
        a = np.radians(elevation_deg)
        end = (HANDS[0] + length * np.cos(a), HANDS[1] - length * np.sin(a))
        cv2.line(g, (int(HANDS[0]), int(HANDS[1])), (int(end[0]), int(end[1])), 240, 3)
    if extra:
        extra(g)
    return cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


# ---- the monotone fit ---------------------------------------------------------------

def test_isotonic_is_non_decreasing_and_keeps_monotone_input():
    y = np.array([-60, -50, -20, 0, 10, 55.0])
    assert np.allclose(isotonic_increasing(y), y)
    z = isotonic_increasing(np.array([5, 3, 4, 1, 9.0]))
    assert (np.diff(z) >= 0).all()


def test_one_stray_reading_does_not_drag_the_fit_negative():
    # the real shape on a clip: a few near-horizontal readings, one stray -25, then clearly up
    y = np.array([-60, -58, -55, 8, 4, 1, 0, -25, 53, 55, 58.0])
    fit = isotonic_increasing(y)
    assert fit[3] >= 0 and (np.diff(fit) >= 0).all()  # a mean-based fit pooled these to about -2


# ---- the crossing ---------------------------------------------------------------------

def test_clean_ramp_crosses_where_it_should_and_the_bracket_is_tight():
    f = np.arange(0, 101, 2)
    e = -60 + 120 * f / 100.0  # crosses zero at frame 50
    cue = crossing_from_readings(f, e, np.ones_like(e), backswing_frames=100)
    assert abs(cue.frame - 50) <= 1 and cue.bracket[1] - cue.bracket[0] <= 4
    assert not cue.low_confidence


def test_a_gap_in_the_readings_gives_a_wide_bracket_and_a_midpoint_estimate():
    f = np.array([0, 4, 8, 12, 16, 20, 80, 84, 88, 92, 96, 100])
    e = np.array([-60, -62, -61, -63, -60, -59, 52, 53, 54, 55, 56, 58.0])
    cue = crossing_from_readings(f, e, np.ones_like(e), backswing_frames=80)
    assert cue.bracket == (20, 80) and 45 <= cue.frame <= 55
    assert cue.low_confidence and "wide" in cue.note  # 60 frames of an 80-frame backswing


def test_too_few_readings_on_either_side_gives_no_answer():
    f = np.arange(10)
    only_below = crossing_from_readings(f, np.full(10, -50.0), np.ones(10), 100)
    assert only_below.frame is None and "too few" in only_below.note
    one_above = crossing_from_readings(np.arange(8), np.array([-50, -50, -50, -50, -50, 40, 0, 0.0]), np.ones(8), 100)
    assert one_above.frame is None


def test_a_stray_negative_after_the_crossing_does_not_move_the_estimate_much():
    f = np.arange(0, 60, 2)
    e = -60 + 4 * f.astype(float)
    e = np.clip(e, -60, 60)
    clean = crossing_from_readings(f, e, np.ones_like(e), 60).frame
    e2 = e.copy()
    e2[20] = -25.0  # a false line well after the crossing
    noisy = crossing_from_readings(f, e2, np.ones_like(e2), 60).frame
    assert abs(noisy - clean) <= 3


# ---- the line finder on images ---------------------------------------------------------

@pytest.mark.parametrize("elevation", [-45.0, 30.0, 70.0])
def test_finds_a_shaft_and_its_elevation(elevation):
    r = shaft_elevation(_frame_with_shaft(elevation), HANDS, ELBOW, TORSO)
    assert r is not None and abs(r[0] - elevation) <= 4
    assert r[1] >= 0.5 * TORSO


def test_no_line_means_no_reading():
    assert shaft_elevation(_frame_with_shaft(None), HANDS, ELBOW, TORSO) is None


def test_a_background_edge_running_right_through_the_hands_is_not_the_shaft():
    def through(g):
        cv2.line(g, (150, 300), (500, 300), 240, 3)  # a long horizontal edge, both sides of the hands
    assert shaft_elevation(_frame_with_shaft(None, extra=through), HANDS, ELBOW, TORSO) is None


def test_a_line_running_back_along_the_forearm_is_not_the_shaft():
    def forearm(g):
        cv2.line(g, (int(HANDS[0]), int(HANDS[1])), (int(ELBOW[0] - 10), int(ELBOW[1])), 240, 3)  # toward the elbow
    assert shaft_elevation(_frame_with_shaft(None, extra=forearm), HANDS, ELBOW, TORSO) is None


def test_a_short_line_is_too_short_to_be_the_shaft():
    assert shaft_elevation(_frame_with_shaft(20.0, length=30), HANDS, ELBOW, TORSO) is None  # 0.3 torso


def test_a_line_that_does_not_pass_through_the_hands_is_ignored():
    def away(g):
        cv2.line(g, (300, 200), (380, 160), 240, 3)  # 100 px above the hands
    assert shaft_elevation(_frame_with_shaft(None, extra=away), HANDS, ELBOW, TORSO) is None


# ---- a whole synthetic clip -------------------------------------------------------------

TK, TOP, N = 20, 100, 130


def _seq():
    k = np.stack([empty_pose() for _ in range(N)])
    k[:, :, 2] = 0.9
    for name, p in (("left_wrist", HANDS), ("right_wrist", HANDS), ("left_elbow", ELBOW), ("right_elbow", ELBOW),
                    ("left_shoulder", (260, 150)), ("right_shoulder", (340, 150)),
                    ("left_hip", (270, 250)), ("right_hip", (330, 250))):
        k[:, J[name], :2] = p
    return PoseSequence(k.astype(np.float32), 240.0, 640, 480, "t")


@pytest.fixture
def shaft_video(tmp_path):
    path = tmp_path / "club.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 240.0, (640, 480))
    for t in range(N):
        e = -60 + 120 * np.clip((t - TK) / (TOP - TK), 0, 1)  # head rises from below the hands to above; level at t=60
        w.write(_frame_with_shaft(e, seed=t))
    w.release()
    return path


def test_shaft_cue_finds_the_frame_the_shaft_is_level(shaft_video):
    cue = shaft_cue(shaft_video, _seq(), {"takeaway": TK, "top": TOP})
    assert cue.frame is not None and abs(cue.frame - 60) <= 5, cue
    assert cue.bracket[0] <= cue.frame <= cue.bracket[1]
    assert cue.readings > 20


def test_shaft_cue_says_why_when_there_is_no_shaft(tmp_path):
    path = tmp_path / "blank.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 240.0, (640, 480))
    for t in range(N):
        w.write(_frame_with_shaft(None, seed=t))
    w.release()
    cue = shaft_cue(path, _seq(), {"takeaway": TK, "top": TOP})
    assert cue.frame is None and "no shaft line" in cue.note


@pytest.mark.parametrize("frames", [{"takeaway": None, "top": 100}, {"takeaway": 100, "top": 20}])
def test_shaft_cue_needs_ordered_takeaway_and_top(shaft_video, frames):
    cue = shaft_cue(shaft_video, _seq(), frames)
    assert cue.frame is None and "takeaway and the top" in cue.note


# ---- plumbing ----------------------------------------------------------------------------

def _phases(**over):
    f = {"address": 19, "takeaway": 20, "shaft_parallel": None, "top": 100, "impact": 120, "finish": 129}
    f.update(over)
    return PhaseResult(f, 240.0)


def test_refine_shaft_fills_the_phase_only_between_takeaway_and_top():
    ph, note = refine_shaft(_phases(), ShaftCue(60, (56, 64), 30, False, "between frames 56 and 64"))
    assert ph.frames["shaft_parallel"] == 60 and "estimate" in ph.sources["shaft_parallel"] and "56 and 64" in note
    for bad in (15, 20, 100, 110):
        original = _phases()
        ph2, note2 = refine_shaft(original, ShaftCue(bad))
        assert ph2 is original and "not between" in note2
    original = _phases()
    ph3, note3 = refine_shaft(original, ShaftCue(None, note="blurred"))
    assert ph3 is original and "not found" in note3 and "blurred" in note3


def test_shaft_parallel_is_a_label_phase_between_takeaway_and_top_but_not_required():
    assert PHASES.index("takeaway") < PHASES.index("shaft_parallel") < PHASES.index("top")
    assert "shaft_parallel" not in CORE_PHASES
    lab = PhaseLabels("v", 240.0, 1000)
    for p, f in (("address", 50), ("takeaway", 92), ("top", 220), ("impact", 276), ("finish", 400)):
        lab.set(p, f)
    assert lab.complete  # a label set without shaft_parallel is still complete
    lab.set("shaft_parallel", 150)
    assert not lab.problems()
    lab.set("shaft_parallel", 300)  # after the top
    assert lab.problems()


def test_label_tool_key_6_marks_shaft_parallel_and_1_to_5_are_unchanged(synthetic_video, tmp_path, monkeypatch):
    import cv2 as _cv2

    from swing import label_tool

    keys = iter([ord("d")] * 3 + [ord("6"), ord("d"), ord("1"), ord("d"), ord("3"), 27])
    monkeypatch.setattr(_cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(_cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(_cv2, "waitKeyEx", lambda d=0: next(keys))
    monkeypatch.setattr(_cv2, "getWindowProperty", lambda *a: 1.0)
    monkeypatch.setattr(_cv2, "destroyAllWindows", lambda: None)
    labels = label_tool.run(synthetic_video, tmp_path / "l.json")
    assert labels.labels == {"shaft_parallel": 3, "address": 4, "top": 5}


def test_apply_shaft_cue_and_the_phase_table_label_it_experimental():
    seq = _seq()
    a = Analysis(seq, seq, _phases(), None, "dtl")
    b = apply_shaft_cue(a, ShaftCue(60, (56, 64), 30, False, "between frames 56 and 64"))
    rows = {r["phase"]: r for r in phase_table(b)}
    assert rows["shaft_parallel"]["frame"] == 60 and "experimental" in rows["shaft_parallel"]["status"]
    assert rows["top"]["status"].startswith("scored")
    assert "between frames 56 and 64" in b.shaft_note
    assert apply_shaft_cue(a, None) is a
    unfound = {r["phase"]: r for r in phase_table(a)}["shaft_parallel"]
    assert unfound["frame"] is None and unfound["from"] == "not estimated"


def test_shaft_cue_cached_runs_once(shaft_video, tmp_path, monkeypatch):
    import swing.shaft as shaft

    seq = _seq()
    a = Analysis(seq, seq, _phases(), None, "dtl")
    first = shaft_cue_cached(shaft_video, a, tmp_path / "cache")
    assert first.frame is not None
    monkeypatch.setattr(shaft, "iter_frames", lambda *x, **k: (_ for _ in ()).throw(AssertionError("decoded again")))
    assert shaft_cue_cached(shaft_video, a, tmp_path / "cache") == first
    [entry] = (tmp_path / "cache").glob("*.shaft.json")
    entry.write_text("garbage")
    monkeypatch.undo()
    assert shaft_cue_cached(shaft_video, a, tmp_path / "cache").frame == first.frame  # rebuilt


def test_app_offers_shaft_parallel_off_by_default_and_survives_turning_it_on(demo_dirs):  # noqa: F811
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=180).run()
    box = next(c for c in at.sidebar.checkbox if "shaft-parallel" in c.label)
    assert box.value is False and "experimental" in box.label.lower()
    box.check().run()
    assert not at.exception
    assert any("Shaft-parallel (experimental, unvalidated)" in c.value for c in at.caption)
