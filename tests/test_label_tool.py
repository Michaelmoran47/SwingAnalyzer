import cv2

from swing import label_tool
from swing.labels import PhaseLabels


def _drive(monkeypatch, keys, shown):
    it = iter(keys)
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda w, img: shown.append(img.shape))
    monkeypatch.setattr(cv2, "waitKeyEx", lambda d=0: next(it))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1.0)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)


def test_scripted_session_saves_labels(synthetic_video, tmp_path, monkeypatch):
    shown = []
    # 3x right -> frame 3; '1' labels address; down (+10) -> 13; '4' labels impact;
    # left -> 12; '4' relabels impact; 'u' undoes that; Esc quits.
    keys = [ord("d")] * 3 + [ord("1"), 2621440, ord("4"), ord("a"), ord("4"), ord("u"), 27]
    _drive(monkeypatch, keys, shown)
    path = tmp_path / "labels.json"
    labels = label_tool.run(synthetic_video, path)
    assert labels.labels == {"address": 3, "impact": 13}
    assert PhaseLabels.load(path).labels == {"address": 3, "impact": 13}
    assert len(shown) == len(keys)


def test_clamps_at_ends_and_resumes(synthetic_video, tmp_path, monkeypatch):
    path = tmp_path / "labels.json"
    _drive(monkeypatch, [ord("a"), ord("1"), 27], [])
    assert label_tool.run(synthetic_video, path).labels == {"address": 0}
    # second session resumes and can delete via backspace on the labelled frame
    _drive(monkeypatch, [8, 27], [])
    assert label_tool.run(synthetic_video, path).labels == {}


def test_refuses_mismatched_label_file(synthetic_video, tmp_path, monkeypatch):
    import pytest
    path = tmp_path / "labels.json"
    PhaseLabels("x", 120.0, 999).save(path)
    _drive(monkeypatch, [27], [])
    with pytest.raises(ValueError):
        label_tool.run(synthetic_video, path)
