import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from swing.demo import (analyze, find_clips, hand_figure, metric_table, overlay_rgb, phase_table,
                        usable_keypoint_files)
from swing.labels import PhaseLabels
from swing.pose import PoseSequence
from test_phases import FPS, _seq_from_hands, synthetic_hands

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def _scaled_seq(**kw):
    """Synthetic swing, coordinates scaled 0.1 so a small video can hold it."""
    hands, truth = synthetic_hands()
    seq = _seq_from_hands(hands, **kw)
    k = seq.keypoints.copy()
    k[:, :, :2] *= 0.1
    return PoseSequence(k, seq.fps, 108, 192, "synthetic", seq.start_frame), truth


# ---- pure layer ------------------------------------------------------------

def test_analyze_tables_and_figure():
    seq, truth = _scaled_seq()
    a = analyze(seq, "dtl")
    rows = {r["phase"]: r for r in phase_table(a)}
    assert abs(rows["top"]["frame"] - truth["top"]) <= 3
    assert abs(rows["impact"]["frame"] - truth["impact"]) <= 3
    assert rows["top"]["status"].startswith("scored")
    assert rows["address"]["status"].startswith("threshold-defined")
    assert rows["top"]["your label"] is None  # no labels supplied
    metrics = {r["metric"]: r for r in metric_table(a)}
    assert metrics["tempo_ratio"]["value"] is not None
    assert metrics["hip_slide_impact_torso"]["value"] is None
    assert "unavailable" in metrics["hip_slide_impact_torso"]["what it is"]
    assert hand_figure(a) is not None


def test_phase_table_shows_label_errors():
    seq, truth = _scaled_seq()
    labels = PhaseLabels("v", FPS, len(seq), {"top": truth["top"] + 12, "impact": truth["impact"]})
    rows = {r["phase"]: r for r in phase_table(analyze(seq, "dtl", labels=labels))}
    assert rows["top"]["your label"] == truth["top"] + 12
    assert rows["top"]["error (ms)"] < 0  # detected earlier than the label
    assert rows["takeaway"]["error (ms)"] is None  # not labelled


def test_overlay_rgb_draws_and_keeps_size():
    seq, _ = _scaled_seq()
    img = np.full((192, 108, 3), 40, np.uint8)
    jpeg = cv2.imencode(".jpg", img)[1].tobytes()
    out = overlay_rgb(jpeg, seq.keypoints[100], 1.0)
    assert out.shape == (192, 108, 3) and (out != cv2.cvtColor(cv2.imdecode(
        np.frombuffer(jpeg, np.uint8), 1), cv2.COLOR_BGR2RGB)).any()


def test_usable_keypoint_files_skips_partial_and_smoothed(tmp_path):
    seq, _ = _scaled_seq()
    full = tmp_path / "clip.npz"
    seq.save(full)
    seq.save(tmp_path / "clip_smooth.npz")
    PoseSequence(seq.keypoints[:50], seq.fps, 108, 192, "x", start_frame=10).save(tmp_path / "clip_part.npz")
    (tmp_path / "clip_junk.npz").write_bytes(b"nope")
    ok, skipped = usable_keypoint_files(sorted(tmp_path.glob("*.npz")), len(seq))
    assert [k.path.name for k in ok] == ["clip.npz"]
    reasons = {p.name: r for p, r in skipped}
    assert "smoothed" in reasons["clip_smooth.npz"]
    assert "not the whole clip" in reasons["clip_part.npz"]
    assert "unreadable" in reasons["clip_junk.npz"]


def test_find_clips(tmp_path):
    raw, out, lab = tmp_path / "raw", tmp_path / "out", tmp_path / "lab"
    for d in (raw, out, lab):
        d.mkdir()
    (raw / "a.MOV").write_bytes(b"")
    (raw / "notes.txt").write_text("x")
    (out / "a.npz").write_bytes(b"")
    (out / "a_rtmpose.npz").write_bytes(b"")
    (out / "b.npz").write_bytes(b"")
    (lab / "a.json").write_text("{}")
    [c] = find_clips(raw, out, lab)
    assert c.video.name == "a.MOV" and len(c.keypoints) == 2 and c.labels is not None
    assert find_clips(tmp_path / "missing", out, lab) == []


# ---- the real app, headless -----------------------------------------------

@pytest.fixture
def demo_dirs(tmp_path, monkeypatch):
    seq, truth = _scaled_seq()
    raw, out, lab = tmp_path / "raw", tmp_path / "outputs", tmp_path / "labels"
    for d in (raw, out, lab):
        d.mkdir()
    w = cv2.VideoWriter(str(raw / "swing.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (108, 192))
    for i in range(len(seq)):
        w.write(np.full((192, 108, 3), i % 200, np.uint8))
    w.release()
    seq.save(out / "swing.npz")
    seq.save(out / "swing_smooth.npz")
    PhaseLabels("swing.mp4", FPS, len(seq), {"top": truth["top"], "impact": truth["impact"]}).save(lab / "swing.json")
    for var, d in (("SWING_RAW_DIR", raw), ("SWING_OUT_DIR", out), ("SWING_LABELS_DIR", lab)):
        monkeypatch.setenv(var, str(d))
    return tmp_path, truth


def tmp_path_of(demo_dirs):
    return demo_dirs[0]


def _run():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(APP), default_timeout=180).run()


def test_app_renders_saved_analysis(demo_dirs):
    _, truth = demo_dirs
    at = _run()
    assert not at.exception
    assert at.title[0].value == "Swing Analyzer"
    assert any("Unvalidated" in w.value for w in at.warning)
    assert len(at.tabs) == 5
    phases = at.dataframe[0].value.set_index("phase")
    assert abs(phases.loc["top", "frame"] - truth["top"]) <= 3
    assert phases.loc["top", "your label"] == truth["top"]
    assert any("swing_smooth.npz" in c.value and "smoothed" in c.value for c in at.sidebar.caption)
    cache_files = list((tmp_path_of(demo_dirs) / "outputs" / ".frame_cache").glob("*.frames.npz"))
    assert len(cache_files) == 1  # decoded frames were saved for the next launch


def test_app_view_switch_changes_available_metrics(demo_dirs):
    at = _run()
    dtl = at.dataframe[1].value.set_index("metric")
    assert "hip_toward_ball_impact_torso" in dtl.index  # listed (value or reason) for dtl
    assert "unavailable" in dtl.loc["hip_slide_impact_torso", "what it is"]
    at.sidebar.radio[1].set_value("fo").run()
    assert not at.exception
    fo = at.dataframe[1].value.set_index("metric")
    assert "hip_toward_ball_impact_torso" not in fo.index
    assert "hip_slide_impact_torso" in fo.index


def test_app_without_any_clip_shows_guidance(tmp_path, monkeypatch):
    for var in ("SWING_RAW_DIR", "SWING_OUT_DIR", "SWING_LABELS_DIR"):
        monkeypatch.setenv(var, str(tmp_path / "empty"))
    at = _run()
    assert not at.exception
    assert any("No videos found" in i.value for i in at.info)
