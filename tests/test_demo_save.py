"""Keeping an analysed upload so it appears under "Saved analysis"."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from swing.demo import find_clips, safe_clip_name, save_upload
from swing.io import cached_jpeg_frames, load_jpeg_frames, probe_video, store_jpeg_frames
from swing.pose import PoseSequence
from test_demo import APP, _scaled_seq, demo_dirs  # noqa: F401  (fixture re-used)


def test_save_upload_copies_video_and_keypoints(tmp_path):
    seq, _ = _scaled_seq()
    src = tmp_path / "up" / "My Swing (1).MOV"
    src.parent.mkdir()
    src.write_bytes(b"video-bytes")
    v, k = save_upload(src, seq, "My Swing (1)", tmp_path / "raw", tmp_path / "out")
    assert v == tmp_path / "raw" / "My_Swing_1.mov" and v.read_bytes() == b"video-bytes"
    np.testing.assert_array_equal(PoseSequence.load(k).keypoints, seq.keypoints)
    assert not list((tmp_path / "raw").glob("*.tmp")) and not list((tmp_path / "out").glob("*.tmp*"))
    [clip] = find_clips(tmp_path / "raw", tmp_path / "out", tmp_path / "labels")
    assert clip.video == v and clip.keypoints == (k,)  # appears under "Saved analysis"


@pytest.mark.parametrize("existing", ["raw/s.mov", "raw/s.MP4", "out/s.npz"])
def test_save_upload_never_overwrites_and_writes_nothing_else(tmp_path, existing):
    seq, _ = _scaled_seq()
    src = tmp_path / "in.mov"
    src.write_bytes(b"new")
    target = tmp_path / existing
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"precious")
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    with pytest.raises(FileExistsError):
        save_upload(src, seq, "s", tmp_path / "raw", tmp_path / "out")
    assert target.read_bytes() == b"precious"
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before


def test_save_upload_rolls_back_video_if_keypoints_cannot_be_saved(tmp_path, monkeypatch):
    seq, _ = _scaled_seq()
    src = tmp_path / "in.mov"
    src.write_bytes(b"v")
    real = os.replace

    def flaky(a, b):
        if str(b).endswith(".npz"):
            raise OSError("disk full")
        return real(a, b)

    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(OSError):
        save_upload(src, seq, "s", tmp_path / "raw", tmp_path / "out")
    assert not list((tmp_path / "raw").glob("*")) and not list((tmp_path / "out").glob("*"))


def test_safe_name_and_empty_name_rejected(tmp_path):
    assert safe_clip_name("  evil name!.mov ") == "evil_name"
    assert safe_clip_name("../../x") == "x"  # path parts dropped
    assert safe_clip_name("***") == ""
    seq, _ = _scaled_seq()
    with pytest.raises(ValueError):
        save_upload(tmp_path / "x.mov", seq, "***", tmp_path / "raw", tmp_path / "out")


def test_find_clips_gives_each_file_to_the_longest_matching_clip(tmp_path):
    raw, out, lab = tmp_path / "raw", tmp_path / "out", tmp_path / "lab"
    for d in (raw, out, lab):
        d.mkdir()
    for n in ("swing.mov", "swing_2.mov"):
        (raw / n).write_bytes(b"")
    for n in ("swing.npz", "swing_rtmpose.npz", "swing_2.npz", "swing_2_rtmpose.npz"):
        (out / n).write_bytes(b"")
    clips = {c.video.stem: sorted(k.name for k in c.keypoints) for c in find_clips(raw, out, lab)}
    assert clips["swing"] == ["swing.npz", "swing_rtmpose.npz"]
    assert clips["swing_2"] == ["swing_2.npz", "swing_2_rtmpose.npz"]


def test_store_jpeg_frames_registers_a_clip_without_decoding(synthetic_video, tmp_path, monkeypatch):
    import swing.io.video as video

    frames = load_jpeg_frames(synthetic_video, 0.5)
    store_jpeg_frames(synthetic_video, 0.5, tmp_path / "c", frames)
    monkeypatch.setattr(video, "iter_frames", lambda *a, **k: (_ for _ in ()).throw(AssertionError("decoded")))
    got, hit = cached_jpeg_frames(synthetic_video, 0.5, tmp_path / "c")
    assert hit and got == frames


def test_app_upload_save_button_end_to_end(demo_dirs, tmp_path_factory):  # noqa: F811
    """Pretend an upload was analysed, press Save, and find it under Saved analysis."""
    from streamlit.testing.v1 import AppTest

    base, _ = demo_dirs
    seq, _ = _scaled_seq()
    up_dir = tmp_path_factory.mktemp("upload")
    clip = up_dir / "Phone Clip.mp4"
    shutil.copyfile(base / "raw" / "swing.mp4", clip)

    at = AppTest.from_file(str(APP), default_timeout=180)
    at.session_state["upload"] = dict(dir=str(up_dir), video=clip, info=probe_video(clip), seq=seq,
                                      backend="mediapipe", source="uploaded: Phone Clip.mp4", labels=None)
    at.run()
    at.sidebar.radio[0].set_value("Upload a video").run()
    assert not at.exception
    assert at.sidebar.text_input[0].value == "Phone_Clip"
    at.sidebar.button[-1].click().run()
    assert not at.exception
    assert (base / "raw" / "Phone_Clip.mp4").exists() and (base / "outputs" / "Phone_Clip.npz").exists()
    assert list((base / "outputs" / ".frame_cache").glob("Phone_Clip_*.frames.npz"))
    assert any("Saved as" in s.value for s in at.sidebar.success)
    # it now appears under Saved analysis; the upload's temp path wrote no cache of its own
    at.sidebar.radio[0].set_value("Saved analysis").run()
    assert not at.exception
    assert "Phone_Clip.mp4" in list(at.sidebar.selectbox[0].options)
    assert not list((base / "outputs" / ".frame_cache").glob("*Phone Clip*"))
