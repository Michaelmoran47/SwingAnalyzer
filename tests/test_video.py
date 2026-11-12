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


def test_start_frame_is_the_same_image_as_in_a_run_from_zero(synthetic_video):
    """Frame n must be the n-th decoded frame whatever ``start`` is.

    A synthetic file cannot reproduce the real-file bug (cap.set seeking landed 50
    frames late on a file whose header disagreed with its stream), so this guards the
    contract; the real file was checked by hand (docs/rtmpose.md).
    """
    import numpy as np

    from_zero = {i: f for i, f in iter_frames(synthetic_video)}
    for start in (1, 10, 37):
        for i, f in iter_frames(synthetic_video, start, start + 3):
            assert np.array_equal(f, from_zero[i]), (start, i)


def test_start_past_the_end_yields_nothing(synthetic_video):
    assert list(iter_frames(synthetic_video, 500)) == []


def test_iter_all(synthetic_video):
    assert sum(1 for _ in iter_frames(synthetic_video)) == 60


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe_video(tmp_path / "nope.mp4")
