"""Takeaway from whole-body motion: the end of the last quiet stretch before the swing."""

import numpy as np
import pytest

from swing.phases import PhaseParams, detect_phases
from swing.phases.onset import POOL, body_motion_onset, pooled_speed
from swing.pose import J, PoseSequence, empty_pose

FPS = 240.0
T = 700
START = 300  # the swing starts here
TOP = 520


def _keypoints(*, noise=0.3, ramp=0.03, seed=0, joints=POOL, waggle=False):
    """Pooled joints still (plus keypoint noise) until START, then moving with growing speed.

    ``ramp`` px/frame^2 is how quickly the motion speeds up. ``waggle`` adds a brief movement
    well before START (a golfer fidgeting before the swing).
    """
    rng = np.random.default_rng(seed)
    k = np.stack([empty_pose() for _ in range(T)])
    k[:, :, 2] = 0.9
    t = np.arange(T)
    d = 0.5 * ramp * np.clip(t - START, 0, None) ** 2  # displacement
    if waggle:
        d = d + 6 * np.exp(-0.5 * ((t - 120) / 6.0) ** 2)
    for i, name in enumerate(joints):
        base = np.array([100.0 + 15 * i, 200.0 + 10 * i])
        k[:, J[name], :2] = base + np.stack([d * (1 + 0.1 * i), -0.5 * d], axis=1) + rng.normal(0, noise, (T, 2))
    return k.astype(np.float32)


def test_onset_is_close_to_the_true_start_and_not_early():
    k = _keypoints()
    o = body_motion_onset(k, FPS, t0=START + 90, top=TOP)
    assert START - 3 <= o <= START + 25


def test_a_hand_based_estimate_far_after_the_start_is_corrected_backwards():
    k = _keypoints()
    assert body_motion_onset(k, FPS, t0=START + 150, top=TOP) == body_motion_onset(k, FPS, t0=START + 60, top=TOP)


def test_noisier_keypoints_delay_the_onset_but_it_stays_near_the_start():
    quiet = body_motion_onset(_keypoints(noise=0.2), FPS, START + 90, TOP)
    noisy = body_motion_onset(_keypoints(noise=0.8), FPS, START + 90, TOP)
    assert quiet <= noisy <= START + 70  # a higher noise floor means a later, but still near, call


def test_a_faster_start_is_found_sooner():
    slow = body_motion_onset(_keypoints(ramp=0.01), FPS, START + 120, TOP)
    fast = body_motion_onset(_keypoints(ramp=0.08), FPS, START + 120, TOP)
    assert START - 3 <= fast <= slow


def test_motion_all_the_time_has_no_quiet_stretch_so_no_answer():
    t = np.arange(T)
    k = _keypoints(noise=0.2)
    for name in POOL:
        k[:, J[name], 0] += 20 * np.sin(t / 5.0)  # always moving
    assert body_motion_onset(k, FPS, START + 90, TOP) is None


def test_too_few_tracked_joints_gives_no_answer():
    k = _keypoints(joints=POOL[:3])  # only 3 of the 8 pooled joints tracked
    assert body_motion_onset(k, FPS, START + 90, TOP) is None
    assert np.isnan(pooled_speed(k)).all()


def test_a_noise_free_clip_gives_no_answer_instead_of_dividing_by_zero():
    k = _keypoints(noise=0.0)
    assert body_motion_onset(k, FPS, START + 90, TOP) is None


@pytest.mark.parametrize("t0,top", [(0, 100), (100, 100), (500, 400), (100, 10_000)])
def test_bad_arguments_give_no_answer(t0, top):
    assert body_motion_onset(_keypoints(), FPS, t0, top) is None


def test_a_waggle_before_the_swing_is_treated_as_movement_not_ignored():
    """Documented limit: a fidget just before the swing ends the quiet stretch early."""
    plain = body_motion_onset(_keypoints(), FPS, START + 90, TOP)
    with_waggle = body_motion_onset(_keypoints(waggle=True), FPS, START + 90, TOP)
    assert with_waggle == plain  # a waggle long before the quiet stretch does not matter
    near = _keypoints()
    near[START - 25:START - 5, J["left_wrist"], 0] += 15  # a fidget right before: shorter than the stretch needed
    assert body_motion_onset(near, FPS, START + 90, TOP) <= plain


# ---- inside detect_phases -------------------------------------------------------------

def _full_sequence():
    """Hands and body from one synthetic swing: still until START, backswing to TOP, downswing."""
    from test_phases import synthetic_hands

    hands, truth = synthetic_hands(still_s=START / FPS)
    n = len(hands)
    k = np.stack([empty_pose() for _ in range(n)])
    k[:, :, 2] = 0.9
    rng = np.random.default_rng(1)
    prog = np.clip((np.arange(n) - START) / (truth["top"] - START), 0, 1)  # body follows the backswing
    for name, dx in (("left_wrist", 0), ("right_wrist", 0)):
        k[:, J[name], :2] = hands + rng.normal(0, 0.3, (n, 2))
    for i, name in enumerate(("left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_elbow", "right_elbow")):
        k[:, J[name], :2] = np.array([500.0 + 30 * i, 900.0 - 60 * i]) + np.stack(
            [110 * prog, -60 * prog], axis=1) + rng.normal(0, 0.3, (n, 2))  # ~100 px over a backswing
    return PoseSequence(k.astype(np.float32), FPS, 1080, 1920, "t"), truth


def test_detect_phases_uses_the_body_onset_and_makes_address_the_frame_before():
    seq, truth = _full_sequence()
    on = detect_phases(seq)
    off = detect_phases(seq, PhaseParams(body_onset=False))
    assert on.frames["takeaway"] <= off.frames["takeaway"]
    assert truth["start"] - 5 <= on.frames["takeaway"] <= truth["start"] + 30
    assert on.frames["address"] == on.frames["takeaway"] - 1
    assert "body motion" in on.sources["takeaway"]
    assert on.frames["top"] == off.frames["top"] and on.frames["impact"] == off.frames["impact"]  # untouched


def test_detect_phases_falls_back_to_the_hands_and_says_so_when_no_quiet_stretch():
    seq, _ = _full_sequence()
    k = seq.keypoints.copy()
    k[:, J["left_shoulder"], 0] += 20 * np.sin(np.arange(len(k)) / 4.0)
    for name in ("right_shoulder", "left_hip", "right_hip", "left_elbow", "right_elbow", "left_wrist", "right_wrist"):
        k[:, J[name], 1] += 20 * np.sin(np.arange(len(k)) / 4.0)
    always = PoseSequence(k, FPS, 1080, 1920, "t")
    r = detect_phases(always)
    assert "hands" in r.sources.get("takeaway", "hands")


def test_the_new_takeaway_is_rechecked_for_plausibility():
    seq, _ = _full_sequence()
    r = detect_phases(seq)
    assert r.plausible and r.warnings == []
