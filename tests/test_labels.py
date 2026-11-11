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
