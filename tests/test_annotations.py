import numpy as np
import pytest

from swing.annotations import (WristAnnotations, sample_frames, summarise, wrist_errors)
from swing.pose import J, PoseSequence


def _ann():
    return WristAnnotations("v.mp4", 240.0, 100)


def test_set_clear_undo_roundtrip(tmp_path):
    a = _ann()
    a.set(10, "left_wrist", 100.04, 200.0)
    a.set(10, "right_wrist", 5, 6, visible=False)
    a.set(10, "left_wrist", 101, 201)  # replace
    assert a.undo() and a.get(10, "left_wrist").x == pytest.approx(100.04)
    a.clear(10, "right_wrist")
    assert a.get(10, "right_wrist") is None and a.undo()
    assert a.get(10, "right_wrist").visible is False
    a.save(tmp_path / "a.json")
    b = WristAnnotations.load(tmp_path / "a.json")
    assert b.points[10]["left_wrist"].x == pytest.approx(100.0, abs=0.1)
    assert b.points[10]["right_wrist"].visible is False
    assert b.complete_frames() == [10]


def test_undo_of_first_click_removes_frame():
    a = _ann()
    a.set(3, "left_wrist", 1, 1)
    assert a.undo() and a.points == {} and not a.undo()


def test_rejects_bad_frame_and_wrist():
    a = _ann()
    with pytest.raises(ValueError):
        a.set(100, "left_wrist", 0, 0)
    with pytest.raises(ValueError):
        a.set(1, "left_hip", 0, 0)


def test_set_visible_needs_a_click():
    a = _ann()
    assert not a.set_visible(1, "left_wrist", False)
    a.set(1, "left_wrist", 0, 0)
    assert a.set_visible(1, "left_wrist", False) and not a.get(1, "left_wrist").visible


def test_sample_frames_is_a_pure_rule():
    assert sample_frames(731, 761, 10) == [731, 741, 751, 761]
    assert sample_frames(5, 5, 3) == [5]
    with pytest.raises(ValueError):
        sample_frames(0, 10, 0)


def _seq(start=50):
    k = np.full((10, 17, 3), np.nan, np.float32)
    k[:, J["left_wrist"]] = [100, 200, 0.9]
    k[3, J["left_wrist"]] = np.nan  # frame 53: backend had no wrist
    k[:, J["right_wrist"]] = [30, 40, 0.9]
    return PoseSequence(k, 240.0, 320, 240, "x", start_frame=start)


def test_wrist_errors_distance_missing_and_out_of_range():
    a = _ann()
    a.set(50, "left_wrist", 103, 204)      # 5 px
    a.set(53, "left_wrist", 0, 0)          # backend missing
    a.set(52, "right_wrist", 30, 40, visible=False)  # 0 px, hidden guess
    a.set(80, "left_wrist", 0, 0)          # outside the sequence
    e = {(x.frame, x.wrist): x for x in wrist_errors(a, _seq())}
    assert e[(50, "left_wrist")].px == pytest.approx(5.0)
    assert np.isnan(e[(53, "left_wrist")].px) and np.isnan(e[(80, "left_wrist")].px)
    assert e[(52, "right_wrist")].px == 0 and not e[(52, "right_wrist")].visible


def test_summarise_in_torso_lengths_and_missing():
    a = _ann()
    for f, off in ((50, 10), (51, 20), (52, 30)):
        a.set(f, "left_wrist", 100 + off, 200)
    a.set(53, "left_wrist", 0, 0)
    s = summarise(wrist_errors(a, _seq()), torso_px=10.0)
    assert (s["n"], s["missing"]) == (4, 1)
    assert s["median"] == pytest.approx(2.0) and s["max"] == pytest.approx(3.0)
    assert summarise([], 10.0) == {"n": 0, "missing": 0}
