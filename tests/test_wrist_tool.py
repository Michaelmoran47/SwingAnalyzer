import cv2

from swing import wrist_tool
from swing.annotations import WristAnnotations


def _drive(monkeypatch, script):
    """``script`` items: a key code, or ('click', x, y) delivered through the mouse callback."""
    cb = {}
    it = iter(script)
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda w, img: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda w, f: cb.update(f=f))

    def wait(d=0):
        item = next(it)
        if isinstance(item, tuple):
            cb["f"](cv2.EVENT_LBUTTONDOWN, item[1], item[2], 0, None)
            return -1
        return item

    monkeypatch.setattr(cv2, "waitKeyEx", wait)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1.0)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)


def test_clicks_become_full_resolution_coordinates(synthetic_video, tmp_path, monkeypatch):
    # clip is 320x240; crop rows 40..200 and max_height 80 -> scale 0.5
    path = tmp_path / "w.json"
    _drive(monkeypatch, [("click", 50, 20), ("click", 100, 60), ord("v"), 27])
    ann = wrist_tool.run(synthetic_video, path, samples=[5, 15], crop_y=(40, 200), max_height=80)
    assert (ann.get(5, "left_wrist").x, ann.get(5, "left_wrist").y) == (100.0, 80.0)
    right = ann.get(5, "right_wrist")
    assert (right.x, right.y) == (200.0, 160.0) and right.visible is False  # v hit the selected wrist
    assert WristAnnotations.load(path).complete_frames() == [5]


def test_navigation_between_samples_and_undo(synthetic_video, tmp_path, monkeypatch):
    path = tmp_path / "w.json"
    # n -> frame 15; click left; 2 selects right; click; u undoes right; p -> back to 5; Esc
    _drive(monkeypatch, [ord("n"), ("click", 10, 10), ord("2"), ("click", 20, 20), ord("u"),
                         ord("p"), ord("n"), 27])
    ann = wrist_tool.run(synthetic_video, path, samples=[5, 15])
    assert ann.get(15, "left_wrist") is not None and ann.get(15, "right_wrist") is None
    assert ann.get(5, "left_wrist") is None


def test_refuses_mismatched_file(synthetic_video, tmp_path, monkeypatch):
    import pytest
    path = tmp_path / "w.json"
    WristAnnotations("clip.mp4", 120.0, 99).save(path)
    _drive(monkeypatch, [27])
    with pytest.raises(ValueError, match="refusing"):
        wrist_tool.run(synthetic_video, path, samples=[1])
