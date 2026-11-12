"""The ball-leaves-the-tee impact cue, on synthetic frames with the decoys real footage has."""

import cv2
import numpy as np
import pytest

from swing.impact_cue import (BallCue, ball_cue, ball_roi, ball_states, find_candidates,
                              first_departure, judge_candidate, locate_ball_impact)
from swing.pose import J, PoseSequence, empty_pose
from test_demo import demo_dirs  # noqa: F401  (fixture re-used by the app tests)

TORSO = 100.0  # px; the ball is then ~8-9 px across
H, W = 120, 300
BALL = (200, 60)  # (x, y) of the ball centre in the search region
FRAMES = 80
K = 55  # the club covers the ball at this frame; the ball is gone from K+1


def _grass(seed=0):
    rng = np.random.default_rng(seed)
    g = 90 + rng.normal(0, 4, (H, W))
    return np.clip(g, 0, 255)


def _scene(*, drift=0.0, ball=True, leg_over_decoy=True, never_leaves=False, leg_gray=200, seed=0):
    """Gray frames: grass, a ball that leaves at K, plus decoys that never vanish.

    Decoys: a big white 'shoe', a thin white 'alignment stick', a small dirt fleck, a second
    white ball-sized spot that is *covered by a dark 'leg'* in the last frames (so it changes,
    but never becomes bare grass).
    """
    frames = []
    base = _grass(seed)
    for t in range(FRAMES):
        f = base.copy()
        f += np.random.default_rng(seed + 1 + t).normal(0, 2.5, f.shape)  # sensor noise
        cv2.ellipse(f, (60, 80), (22, 9), 0, 0, 360, 235, -1)            # shoe
        cv2.line(f, (120, 20), (150, 100), 225, 2)                        # stick
        cv2.circle(f, (30, 30), 2, 170, -1)                               # fleck
        cv2.circle(f, (250, 40), 4, 235, -1)                              # decoy spot
        if ball and not (t > K) and not never_leaves or (ball and never_leaves):
            cv2.circle(f, BALL, 4, 240, -1)
        if t == K:                                                        # club head at the ball
            cv2.rectangle(f, (BALL[0] - 9, BALL[1] - 6), (BALL[0] + 9, BALL[1] + 6), 30, -1)
        if leg_over_decoy and t >= K + 8:
            cv2.rectangle(f, (232, 25), (270, 60), leg_gray, -1)          # leg over the decoy spot
        f = np.clip(f, 0, 255).astype(np.uint8)
        dx = int(round(drift * t))
        if dx:
            M = np.float32([[1, 0, dx], [0, 1, 0]])
            f = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        frames.append(f)
    return frames


def _locate(frames):
    clean = np.arange(0, 30)
    template = np.median(np.stack([frames[i] for i in clean[::2]]), axis=0).astype(np.uint8)
    pre = clean[1::2]
    post = np.arange(len(frames) - 20, len(frames))
    return locate_ball_impact(frames, 0, template, TORSO, pre, post, start=30, hold=3)


# ---- candidates ---------------------------------------------------------------

def test_candidates_include_the_ball_and_reject_shoe_and_stick():
    template = np.median(np.stack(_scene()[:10]), axis=0).astype(np.uint8)
    cands = find_candidates(template, TORSO)
    centres = [(round(c[0]), round(c[1])) for c in cands]
    assert any(abs(x - BALL[0]) <= 2 and abs(y - BALL[1]) <= 2 for x, y in centres)
    assert not any(abs(x - 60) <= 6 and abs(y - 80) <= 6 for x, y in centres)   # shoe is too big
    assert not any(abs(x - 135) <= 15 and 25 <= y <= 95 for x, y in centres)    # thin stick is not round


# ---- the vanishing test picks the real ball and the frame -----------------------

def test_finds_the_ball_and_the_frame_the_club_arrives():
    cue = _locate(_scene())
    assert cue.frame == K
    assert abs(cue.center[0] - BALL[0]) <= 2 and abs(cue.center[1] - BALL[1]) <= 2
    assert cue.score == pytest.approx(1.0, abs=0.15)


@pytest.mark.parametrize("leg_gray", [200, 15])  # white trousers, a dark shoe
def test_a_decoy_covered_by_a_leg_is_not_mistaken_for_the_ball(leg_gray):
    cue = _locate(_scene(leg_over_decoy=True, leg_gray=leg_gray))
    assert abs(cue.center[0] - BALL[0]) <= 2  # not the decoy spot at x=250


@pytest.mark.parametrize("leg_gray", [200, 15])
def test_no_ball_but_a_leg_over_a_decoy_gives_no_answer(leg_gray):
    assert _locate(_scene(ball=False, leg_over_decoy=True, leg_gray=leg_gray)).frame is None


@pytest.mark.parametrize("drift", [0.1, 0.15])  # 4 to 9 px over the clip: a panning camera
def test_follows_a_drifting_picture(drift):
    cue = _locate(_scene(drift=drift))
    assert cue.frame is not None and abs(cue.frame - K) <= 1


def test_no_ball_means_no_answer_not_a_guess():
    cue = _locate(_scene(ball=False, leg_over_decoy=False))
    assert cue.frame is None and cue.note


def test_a_ball_that_never_leaves_gives_no_answer():
    cue = _locate(_scene(never_leaves=True, leg_over_decoy=False))
    assert cue.frame is None


# ---- the small judging pieces -----------------------------------------------------

def test_first_departure_ignores_a_one_frame_flicker():
    present = np.array([True] * 10 + [False] + [True] * 5 + [False] * 8)
    assert first_departure(present, 0, hold=3) == 16  # not the flicker at 10


def test_first_departure_none_when_it_never_leaves():
    assert first_departure(np.ones(20, bool), 0, hold=3) is None


def test_judge_needs_a_visible_change_and_grass_at_the_end():
    present = np.array([True] * 10 + [False] * 10)
    gone = np.array([False] * 10 + [True] * 10)
    assert judge_candidate(present, gone, 60.0, slice(0, 10), slice(12, 20))[0]
    assert not judge_candidate(present, gone, 10.0, slice(0, 10), slice(12, 20))[0]  # a dirt fleck
    covered = np.zeros(20, bool)  # something else over the spot, never bare grass
    assert not judge_candidate(present, covered, 60.0, slice(0, 10), slice(12, 20))[0]


def test_ball_states_reports_edge_candidates_as_unusable():
    template = _grass().astype(np.uint8)
    present, gone, delta = ball_states([template], template, (2.0, 2.0, 4.0, 100.0))
    assert not present.any() and not gone.any() and delta == 0.0


# ---- the search region --------------------------------------------------------------

def _pose_seq(hands_x, hip_x=100.0):
    k = np.stack([empty_pose() for _ in range(40)])
    k[:, :, 2] = 0.9
    for name, p in (("left_ankle", (95, 200)), ("right_ankle", (105, 200)), ("left_wrist", (hands_x, 120)),
                    ("right_wrist", (hands_x, 120)), ("left_hip", (hip_x - 5, 150)), ("right_hip", (hip_x + 5, 150)),
                    ("left_shoulder", (hip_x - 20, 50)), ("right_shoulder", (hip_x + 20, 50))):
        k[:, J[name], :2] = p
    return PoseSequence(k.astype(np.float32), 240.0, 640, 480, "t")


def test_search_region_extends_toward_the_ball_whichever_side_it_is():
    right = _pose_seq(hands_x=140.0)  # hands ahead of the hips on the right of the image
    left = _pose_seq(hands_x=60.0)
    r_roi, torso = ball_roi(right, 30, (480, 640, 3))
    l_roi, _ = ball_roi(left, 30, (480, 640, 3))
    assert torso == pytest.approx(100.0, abs=1)
    assert r_roi[2] > 300 and r_roi[0] <= 100          # reaches ~3 torso lengths to the right
    assert l_roi[0] == 0 and l_roi[2] >= 100           # reaches to the left edge instead


def test_search_region_needs_tracked_feet_and_a_clear_direction():
    assert ball_roi(_pose_seq(hands_x=100.0), 30, (480, 640, 3)) == (None, None)  # hands over the hips
    seq = _pose_seq(hands_x=140.0)
    seq.keypoints[:, J["left_ankle"], :2] = np.nan
    seq.keypoints[:, J["right_ankle"], :2] = np.nan
    assert ball_roi(seq, 30, (480, 640, 3)) == (None, None)


# ---- the whole function, on a small synthetic video ---------------------------------

@pytest.fixture
def synthetic_swing_video(tmp_path):
    """640x480 video: a ball near the feet that leaves at frame 92; sequence matching it."""
    path = tmp_path / "swing.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 240.0, (640, 480))
    rng = np.random.default_rng(3)
    base = np.clip(90 + rng.normal(0, 3, (480, 640)), 0, 255)
    for t in range(130):
        g = base + np.random.default_rng(10 + t).normal(0, 2, base.shape)
        if t <= 92:
            cv2.circle(g, (270, 205), 5, 245, -1)
        if t == 92:
            cv2.rectangle(g, (258, 197), (282, 213), 30, -1)
        frame = cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        w.write(frame)
    w.release()
    return path


PHASES = {"address": 10, "takeaway": 20, "top": 60, "impact": 90, "finish": 120}


def test_ball_cue_end_to_end(synthetic_swing_video):
    seq = _pose_seq(hands_x=140.0)
    seq = PoseSequence(np.tile(seq.keypoints[:1], (130, 1, 1)), 240.0, 640, 480, "t")
    cue = ball_cue(synthetic_swing_video, seq, PHASES)
    assert cue.frame is not None and abs(cue.frame - 92) <= 2  # mp4 compression blurs the edges slightly
    assert cue.note == ""


def test_ball_cue_refuses_a_frame_far_from_the_hand_estimate(synthetic_swing_video):
    seq = _pose_seq(hands_x=140.0)
    seq = PoseSequence(np.tile(seq.keypoints[:1], (130, 1, 1)), 240.0, 640, 480, "t")
    # hand-based impact 23 frames after the ball leaves, with a 35-frame downswing: 23 > 0.6 * 35
    far = {"address": 10, "takeaway": 20, "top": 80, "impact": 115, "finish": 125}
    cue = ball_cue(synthetic_swing_video, seq, far)
    assert cue.frame is None and "not trusted" in cue.note


@pytest.mark.parametrize("frames,expect", [
    ({"address": 10, "takeaway": None, "top": 60, "impact": 90}, "needs address, takeaway, top and impact"),
    ({"address": 10, "takeaway": 20, "top": 90, "impact": 60}, "needs address, takeaway, top and impact"),
])
def test_ball_cue_needs_ordered_phases(synthetic_swing_video, frames, expect):
    seq = _pose_seq(hands_x=140.0)
    seq = PoseSequence(np.tile(seq.keypoints[:1], (130, 1, 1)), 240.0, 640, 480, "t")
    cue = ball_cue(synthetic_swing_video, seq, frames)
    assert cue.frame is None and expect in cue.note and isinstance(cue, BallCue)


# ---- using the cue: refine_impact, apply_ball_cue, caching, the app ----------------------

def _phases(**over):
    from swing.phases import PhaseResult

    f = {"address": 100, "takeaway": 140, "top": 250, "impact": 320, "finish": 380}
    f.update(over)
    return PhaseResult(f, 240.0)


def test_refine_impact_takes_a_sound_cue_and_reports_the_change():
    from swing.impact_cue import refine_impact

    ph, note = refine_impact(_phases(), BallCue(326))
    assert ph.frames["impact"] == 326 and "ball leaving the tee" in note and "320" in note


@pytest.mark.parametrize("cue_frame,why", [(None, "impact from the hands"), (240, "not between the top and the finish"),
                                           (385, "not between the top and the finish"),
                                           (251, "cannot be a real swing")])  # a 1-frame downswing
def test_refine_impact_keeps_the_hand_estimate_when_the_cue_is_not_sound(cue_frame, why):
    from swing.impact_cue import refine_impact

    original = _phases()
    ph, note = refine_impact(original, BallCue(cue_frame, note="no ball"))
    assert ph is original and why in note


def test_apply_ball_cue_recomputes_metrics_from_the_new_impact():
    from swing.demo import analyze, apply_ball_cue
    from test_demo import _scaled_seq

    seq, truth = _scaled_seq()
    a = analyze(seq, "dtl")
    later = a.phases.frames["impact"] + 5
    b = apply_ball_cue(a, BallCue(later))
    assert b.phases.frames["impact"] == later and b.impact_from_ball
    assert b.metrics.values["downswing_s"] == pytest.approx(a.metrics.values["downswing_s"] + 5 / 240.0)
    assert apply_ball_cue(a, None) is a
    c = apply_ball_cue(a, BallCue(None, note="no ball"))
    assert not c.impact_from_ball and c.phases.frames == a.phases.frames and "no ball" in c.impact_note


def test_ball_cue_cached_runs_once_and_survives_a_bad_cache_file(synthetic_swing_video, tmp_path, monkeypatch):
    import swing.impact_cue as ic
    from dataclasses import replace

    from swing.demo import Analysis, ball_cue_cached
    from swing.phases import PhaseResult

    seq = _pose_seq(hands_x=140.0)
    seq = PoseSequence(np.tile(seq.keypoints[:1], (130, 1, 1)), 240.0, 640, 480, "t")
    a = Analysis(seq, seq, PhaseResult(dict(PHASES), 240.0), None, "dtl")
    cache = tmp_path / "cue"
    first = ball_cue_cached(synthetic_swing_video, a, cache)
    assert first.frame is not None and abs(first.frame - 92) <= 2
    monkeypatch.setattr(ic, "iter_frames", lambda *x, **k: (_ for _ in ()).throw(AssertionError("decoded again")))
    second = ball_cue_cached(synthetic_swing_video, a, cache)
    assert second == first
    [entry] = cache.glob("*.cue.json")
    entry.write_text("not json")
    monkeypatch.undo()
    third = ball_cue_cached(synthetic_swing_video, a, cache)
    assert third.frame == first.frame  # rebuilt


def test_app_says_why_when_no_ball_is_found_and_keeps_hand_impact(demo_dirs):
    from streamlit.testing.v1 import AppTest

    from test_demo import APP

    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception
    assert any("Find impact from the ball" in c.label for c in at.sidebar.checkbox)
    caps = [c.value for c in at.caption] + [s.value for s in at.success]
    assert any(v.startswith("Impact: impact from the hands") for v in caps)


def test_app_ball_cue_checkbox_turns_it_off(demo_dirs):
    from streamlit.testing.v1 import AppTest

    from test_demo import APP

    at = AppTest.from_file(str(APP), default_timeout=180).run()
    box = next(c for c in at.sidebar.checkbox if "Find impact from the ball" in c.label)
    box.uncheck().run()
    assert not at.exception
    assert not any(v.value.startswith("Impact:") for v in list(at.caption) + list(at.success))
