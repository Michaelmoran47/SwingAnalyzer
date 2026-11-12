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


# ---- on-disk frame cache -----------------------------------------------------

def test_frame_cache_roundtrip_and_reuse(synthetic_video, tmp_path, monkeypatch):
    from swing.io import cached_jpeg_frames, load_jpeg_frames
    import swing.io.video as video

    cache = tmp_path / "cache"
    first, hit1 = cached_jpeg_frames(synthetic_video, 0.5, cache)
    assert not hit1 and len(first) == 60
    assert first == load_jpeg_frames(synthetic_video, 0.5)  # identical bytes
    # second call must not decode at all
    monkeypatch.setattr(video, "iter_frames", lambda *a, **k: (_ for _ in ()).throw(AssertionError("decoded")))
    second, hit2 = cached_jpeg_frames(synthetic_video, 0.5, cache)
    assert hit2 and second == first


def test_frame_cache_invalidates_on_scale_and_file_change(synthetic_video, tmp_path):
    import os

    from swing.io import cached_jpeg_frames

    cache = tmp_path / "cache"
    cached_jpeg_frames(synthetic_video, 0.5, cache)
    _, hit = cached_jpeg_frames(synthetic_video, 0.25, cache)  # different scale: new entry
    assert not hit
    assert len(list(cache.glob("*.frames.npz"))) == 1  # the stale one was removed
    st = synthetic_video.stat()
    os.utime(synthetic_video, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))  # file "edited"
    _, hit = cached_jpeg_frames(synthetic_video, 0.25, cache)
    assert not hit


def test_frame_cache_rebuilds_from_corrupt_file(synthetic_video, tmp_path):
    from swing.io import cached_jpeg_frames

    cache = tmp_path / "cache"
    good, _ = cached_jpeg_frames(synthetic_video, 0.5, cache)
    [entry] = cache.glob("*.frames.npz")
    entry.write_bytes(entry.read_bytes()[:200])  # truncated
    again, hit = cached_jpeg_frames(synthetic_video, 0.5, cache)
    assert not hit and again == good
    _, hit = cached_jpeg_frames(synthetic_video, 0.5, cache)
    assert hit  # and it was repaired
    assert not list(cache.glob("*.tmp"))


# ---- variable frame rate detection ------------------------------------------

def test_variable_rate_ignores_a_single_odd_gap():
    import numpy as np

    from swing.io.video import is_variable_rate

    stamps = np.arange(450) * 33.3
    stamps[200:] -= 27.0  # one 6 ms gap among 449 (what a screen recording had)
    assert not is_variable_rate(stamps)


def test_variable_rate_detects_a_real_time_ramp():
    import numpy as np

    from swing.io.video import is_variable_rate

    gaps = np.r_[np.full(300, 4.2), np.full(300, 33.3), np.full(300, 4.2)]  # fast, slow, fast
    assert is_variable_rate(np.r_[0, np.cumsum(gaps)])


def test_constant_rate_and_tiny_inputs_are_not_variable():
    import numpy as np

    from swing.io.video import is_variable_rate

    assert not is_variable_rate(np.arange(100) * 4.1667)
    assert not is_variable_rate([0.0, 4.0])
    assert not is_variable_rate([0.0])
