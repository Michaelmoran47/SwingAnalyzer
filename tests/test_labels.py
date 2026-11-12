import pytest

from swing.labels import PHASES, PhaseLabels, default_label_path


def _l():
    return PhaseLabels("clip.MOV", 240.0, 1000)


def test_set_and_roundtrip(tmp_path):
    lab = _l()
    for i, p in enumerate(PHASES):
        lab.set(p, 100 + i * 50)
    assert lab.complete
    path = tmp_path / "x" / "clip.json"
    lab.save(path)
    back = PhaseLabels.load(path)
    assert back.labels == lab.labels
    assert (back.video, back.fps, back.frame_count) == ("clip.MOV", 240.0, 1000)


def test_out_of_range_and_unknown_phase():
    lab = _l()
    with pytest.raises(ValueError):
        lab.set("impact", 1000)
    with pytest.raises(ValueError):
        lab.set("impact", -1)
    with pytest.raises(ValueError):
        lab.set("backswing", 5)


def test_order_problems_detected():
    lab = _l()
    lab.set("top", 500)
    lab.set("impact", 400)
    assert not lab.complete
    assert lab.problems() == ["impact (400) is not after top (500)"]


def test_undo_restores_previous_value_and_removal():
    lab = _l()
    lab.set("impact", 600)
    lab.set("impact", 610)
    assert lab.undo() and lab.labels["impact"] == 600
    assert lab.undo() and "impact" not in lab.labels
    assert not lab.undo()


def test_clear_and_undo_clear():
    lab = _l()
    lab.set("top", 300)
    lab.clear("top")
    assert lab.labels == {}
    lab.undo()
    assert lab.labels == {"top": 300}


def test_phase_at():
    lab = _l()
    lab.set("address", 20)
    assert lab.phase_at(20) == "address" and lab.phase_at(21) is None


def test_load_rejects_unknown_phase(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"video":"v","fps":240,"frame_count":10,"labels":{"bogus":1}}')
    with pytest.raises(ValueError):
        PhaseLabels.load(p)


def test_default_path():
    assert default_label_path("data/raw/IMG_1.MOV").as_posix() == "data/labels/IMG_1.json"


# ---- acceptable ranges (a golfer who pauses at the top) ------------------------------

def test_range_roundtrip_and_reference(tmp_path):
    lab = _l()
    lab.set("takeaway", 92)
    lab.set_range("top", 190, 250)
    assert lab.reference("top") == 220 and lab.reference("takeaway") == 92 and lab.reference("impact") is None
    path = tmp_path / "c.json"
    lab.save(path)
    back = PhaseLabels.load(path)
    assert back.ranges == {"top": (190, 250)} and back.labels == {"takeaway": 92}


def test_files_without_ranges_still_load_and_save_without_the_key(tmp_path):
    lab = _l()
    lab.set("top", 300)
    lab.save(tmp_path / "a.json")
    assert "ranges" not in (tmp_path / "a.json").read_text()
    assert PhaseLabels.load(tmp_path / "a.json").ranges == {}


@pytest.mark.parametrize("lo,hi", [(250, 190), (-1, 10), (0, 1000)])
def test_bad_ranges_rejected(lo, hi):
    with pytest.raises(ValueError):
        _l().set_range("top", lo, hi)
    with pytest.raises(ValueError):
        _l().set_range("backswing", 1, 2)


def test_order_check_and_completeness_use_ranges():
    lab = _l()
    lab.set("takeaway", 300)
    lab.set_range("top", 190, 250)  # middle 220 is before the takeaway
    assert lab.problems()
    lab.set("takeaway", 92)
    assert not lab.problems()
    for p, f in (("address", 50), ("impact", 400), ("finish", 500)):
        lab.set(p, f)
    assert lab.complete


def test_compare_scores_a_range_as_zero_inside_and_distance_outside():
    from swing.phases import PhaseResult, compare

    lab = PhaseLabels("v", 240.0, 1000, {"takeaway": 92})
    lab.set_range("top", 190, 250)
    rows = {e.phase: e for e in compare(PhaseResult({"takeaway": 129, "top": 202, "impact": 270}, 240.0), lab)}
    assert rows["top"].error_frames == 0 and rows["top"].label_text == "190-250"
    assert rows["takeaway"].error_frames == 37 and rows["takeaway"].label_text == "92"
    early = compare(PhaseResult({"top": 180}, 240.0), lab)[2]
    late = compare(PhaseResult({"top": 260}, 240.0), lab)[2]
    assert early.error_frames == -10 and late.error_frames == 10
