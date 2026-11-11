import pytest

from swing.io import iter_frames, probe_video


def test_probe(synthetic_video):
    info = probe_video(synthetic_video)
    assert info.fps == pytest.approx(120.0)
    assert info.frame_count == 60
    assert (info.width, info.height) == (320, 240)
    assert info.time_of(60) == pytest.approx(0.5)
    assert info.warnings == ()


def test_iter_range(synthetic_video):
    idx = [i for i, _ in iter_frames(synthetic_video, 10, 15)]
    assert idx == [10, 11, 12, 13, 14]


def test_iter_all(synthetic_video):
    assert sum(1 for _ in iter_frames(synthetic_video)) == 60


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe_video(tmp_path / "nope.mp4")
