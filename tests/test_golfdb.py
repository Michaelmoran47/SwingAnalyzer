"""GolfDB label reading, mapping to this project's phases, shortlist, and scoring (no video needed)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from swing.golfdb import (SCORED, aggregate, dataset_summary, detect_for_sample, load_golfdb, phase_frames,
                          score_sample, shortlist, to_30, to_labels, to_native, valid)
from test_phases import _seq_from_hands, synthetic_hands

FIELDS = [("id", "O"), ("youtube_id", "O"), ("player", "O"), ("sex", "O"), ("club", "O"), ("view", "O"),
          ("slow", "O"), ("events", "O"), ("bbox", "O"), ("split", "O")]
GOOD = [408, 455, 473, 476, 490, 495, 498, 501, 514, 545]  # a real sample's events


def _write_mat(path, specs):
    """specs: (youtube_id, player, club, view, slow, events). Same layout as the real golfDB.mat."""
    arr = np.empty((1, len(specs)), dtype=FIELDS)
    for i, (yt, player, club, view, slow, ev) in enumerate(specs):
        arr[0, i] = (np.array([[i]], np.uint8), np.array([yt]), np.array([player]), np.array(["f"]),
                     np.array([club]), np.array([view]), np.array([[int(slow)]], np.uint8),
                     np.array([ev], np.int32), np.array([[0.1, 0.0, 0.5, 0.98]]), np.array([[1]], np.uint8))
    savemat(str(path), {"golfDB": arr})


def _ev(a2i, start=100):
    """Events with address-to-impact = a2i frames, in a plausible proportion."""
    addr = start + 10
    imp = addr + a2i
    top = imp - max(2, a2i // 4)
    tu = addr + (top - addr) // 2
    return [start, addr, tu, tu + 1, top, top + 1, imp, imp + 1, imp + 10, imp + 30]


# ---- loading ------------------------------------------------------------------------

def test_load_roundtrip_and_fields(tmp_path):
    p = tmp_path / "g.mat"
    _write_mat(p, [("abcDEF12345", "SANDRA GAL", "driver", "down-the-line", False, GOOD),
                   ("zzz", "X Y", "iron", "face-on", True, _ev(130))])
    s = load_golfdb(p)
    assert len(s) == 2
    a = s[0]
    assert (a.youtube_id, a.player, a.club, a.view, a.slow) == ("abcDEF12345", "SANDRA GAL", "driver", "down-the-line", False)
    assert a.events == tuple(GOOD) and a.address_to_impact == 498 - 455
    assert a.url == "https://www.youtube.com/watch?v=abcDEF12345"
    assert s[1].slow is True and dataset_summary(s)["slow"] == {False: 1, True: 1}


def test_valid_flags_out_of_order_events(tmp_path):
    bad = list(GOOD)
    bad[4], bad[6] = bad[6], bad[4]  # top after impact
    p = tmp_path / "g.mat"
    _write_mat(p, [("a", "p", "driver", "face-on", False, GOOD), ("b", "p", "driver", "face-on", False, bad)])
    s = load_golfdb(p)
    assert valid(s[0]) and not valid(s[1]) and dataset_summary(s)["invalid"] == 1


# ---- mapping to this project's phases ------------------------------------------------

def _sample(tmp_path, ev=GOOD, slow=False, view="down-the-line"):
    p = tmp_path / "one.mat"
    _write_mat(p, [("vid", "P", "driver", view, slow, ev)])
    return load_golfdb(p)[0]


def test_phase_mapping_and_takeaway_is_the_frame_after_address(tmp_path):
    f = phase_frames(_sample(tmp_path))
    assert f == {"address": 455, "takeaway": 456, "shaft_parallel": 473, "top": 490, "impact": 498}
    assert "finish" not in f  # a different definition: deliberately not mapped
    rel = phase_frames(_sample(tmp_path), relative_to_start=True)
    assert rel["address"] == 455 - 408 and rel["impact"] == 498 - 408


def test_to_labels_sets_mapped_phases_inside_the_video_and_says_where_they_came_from(tmp_path):
    s = _sample(tmp_path)
    lab = to_labels(s, "vid.mp4", 30.0, 600)
    assert lab.labels == {"address": 455, "takeaway": 456, "shaft_parallel": 473, "top": 490, "impact": 498}
    assert "GolfDB sample 0" in lab.notes and "Finish is not mapped" in lab.notes
    assert not lab.problems()
    short = to_labels(s, "vid.mp4", 30.0, 480)  # a video too short for top/impact: those are skipped, not invented
    assert "impact" not in short.labels and "top" not in short.labels


@pytest.mark.parametrize("fps", [24.0, 30.0, 60.0, 120.0])
def test_frame_conversion_roundtrips(fps):
    for f in (0, 1, 100, 455, 498):
        assert to_30(to_native(f, fps), fps) == f
    assert to_native(30, 60.0) == 60 and to_30(60, 60.0) == 30


# ---- shortlist --------------------------------------------------------------------------

def test_shortlist_prefers_videos_with_most_clips_per_cell_and_skips_other_views_and_invalid(tmp_path):
    bad = list(GOOD)
    bad[4], bad[6] = bad[6], bad[4]
    specs = []
    for _ in range(3):
        specs.append(("rtDTL_many", "A", "driver", "down-the-line", False, _ev(34)))
    specs.append(("rtDTL_few", "B", "iron", "down-the-line", False, _ev(34)))
    specs.append(("rtDTL_few", "B", "iron", "down-the-line", False, _ev(34)))
    specs += [("other_view", "C", "driver", "other", False, _ev(34))] * 4  # never qualifies
    specs += [("invalid", "D", "driver", "face-on", False, bad)] * 4  # never qualifies
    specs += [("slowFO", "E", "iron", "face-on", True, _ev(130)), ("slowFO", "E", "iron", "face-on", True, _ev(140))]
    p = tmp_path / "g.mat"
    _write_mat(p, specs)
    sl = shortlist(load_golfdb(p), clips_per_cell=3)
    cells = {x["cell"]: [] for x in sl}
    for x in sl:
        cells[x["cell"]].append(x)
    assert [x["youtube_id"] for x in cells["real-time, down-the-line"]] == ["rtDTL_many"]  # 3 clips reach the target
    assert [x["clips"] for x in cells["slow-motion, face-on"]] == [2]
    assert not any(x["youtube_id"] in ("other_view", "invalid") for x in sl)
    assert sl[0]["url"].startswith("https://www.youtube.com/watch?v=")


def test_shortlist_slow_frame_filter_and_deterministic_ties(tmp_path):
    specs = [("bbb", "P", "driver", "face-on", True, _ev(100)), ("aaa", "P", "driver", "face-on", True, _ev(100)),
             ("ccc", "P", "driver", "face-on", True, _ev(200))]
    p = tmp_path / "g.mat"
    _write_mat(p, specs)
    s = load_golfdb(p)
    ids = [x["youtube_id"] for x in shortlist(s, clips_per_cell=1)]
    assert ids == ["aaa"]  # equal clip counts: ties break on the id
    long_only = [x["youtube_id"] for x in shortlist(s, clips_per_cell=5, min_slow_frames=150)]
    assert long_only == ["ccc"]


# ---- scoring ---------------------------------------------------------------------------------

def test_score_sample_is_in_30_fps_frames_whatever_the_video_rate(tmp_path):
    s = _sample(tmp_path)
    label = phase_frames(s)
    detected_60 = {p: 2 * label[p] + 4 for p in SCORED}  # 2 frames late at 30 fps, expressed at 60 fps
    err = score_sample(s, detected_60, 60.0)
    assert all(err[p] == 2 for p in SCORED)
    assert score_sample(s, {"top": None}, 30.0)["top"] is None


def test_aggregate_statistics_and_groups():
    rows = [
        {"slow": False, "errors": {"takeaway": 3, "shaft_parallel": None, "top": -1, "impact": 0}},
        {"slow": False, "errors": {"takeaway": -5, "shaft_parallel": None, "top": 1, "impact": 2}},
        {"slow": True, "errors": {"takeaway": 10, "shaft_parallel": 4, "top": None, "impact": -8}},
    ]
    agg = aggregate(rows)
    t = agg["real-time"]["takeaway"]
    assert t["n"] == 2 and t["bias"] == -1.0 and t["median_abs"] == 4.0 and t["mean_abs"] == 4.0
    assert t["within_2"] == 0.0 and t["within_5"] == 1.0
    assert agg["real-time"]["shaft_parallel"] == {"n": 0, "missing": 2}
    assert agg["slow-motion"]["top"]["n"] == 0 and agg["slow-motion"]["top"]["missing"] == 1
    assert agg["all"]["impact"]["n"] == 3 and agg["all"]["impact"]["within_2"] == pytest.approx(2 / 3)


def test_detect_for_sample_then_score_on_a_synthetic_swing(tmp_path):
    """A 240 fps synthetic swing, subsampled to 30 fps, run through the same path the harness uses."""
    hands, truth = synthetic_hands()
    sub = hands[::8]
    off = 100  # the window starts 100 frames into the video
    seq = replace(_seq_from_hands(sub, start_frame=off), fps=30.0)
    ev = [off, off + truth["start"] // 8, off + (truth["start"] + truth["top"]) // 16,
          off + (truth["start"] + truth["top"]) // 16 + 1, off + truth["top"] // 8, off + truth["top"] // 8 + 1,
          off + truth["impact"] // 8, off + truth["impact"] // 8 + 1, off + truth["impact"] // 8 + 15,
          off + truth["impact"] // 8 + 30]
    p = tmp_path / "g.mat"
    _write_mat(p, [("vid", "P", "driver", "down-the-line", False, ev)])
    sample = load_golfdb(p)[0]
    frames = detect_for_sample(seq, slow=False)
    err = score_sample(sample, frames, native_fps=30.0)
    assert abs(err["top"]) <= 2 and abs(err["impact"]) <= 2
    assert err["takeaway"] is not None and abs(err["takeaway"]) <= 12
    assert err["shaft_parallel"] is None  # not estimated unless asked for


def test_slow_samples_are_analysed_at_the_assumed_capture_rate():
    hands, _ = synthetic_hands()
    seq = replace(_seq_from_hands(hands[::2]), fps=30.0)  # a 120 fps swing stamped as 30 fps
    as_real = detect_for_sample(seq, slow=False)
    as_slow = detect_for_sample(seq, slow=True, slow_factor=4.0)
    assert as_slow["top"] is not None and as_slow["impact"] is not None
    assert abs(as_slow["top"] - as_real["top"]) <= 3  # the frames agree; the time base is what differs


# ---- the real file, when it is here ----------------------------------------------------------

REAL = Path(__file__).resolve().parent.parent / "data" / "external" / "golfdb" / "golfDB.mat"


@pytest.mark.skipif(not REAL.exists(), reason="the GolfDB label file is not downloaded")
def test_real_label_file_has_the_documented_shape():
    s = load_golfdb(REAL)
    summary = dataset_summary(s)
    assert summary["samples"] == 1400 and summary["videos"] == 580 and summary["players"] == 246
    assert summary["slow"] == {False: 758, True: 642} and summary["invalid"] == 1
    assert summary["view"] == {"down-the-line": 585, "face-on": 461, "other": 354}
    sl = shortlist(s, clips_per_cell=6)
    assert 8 <= len(sl) <= 12 and sum(x["clips"] for x in sl) >= 24
