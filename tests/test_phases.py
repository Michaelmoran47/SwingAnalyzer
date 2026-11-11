import numpy as np
import pytest

from swing.phases.detect import (_interp_short_gaps, detect_phases,
                                 detect_phases_from_hands, hand_signal)
from swing.pose import J, PoseSequence, empty_pose

FPS = 240.0
TORSO = 400.0


def _ease_in_out(n):  # smooth 0..1 (min-jerk-like)
    u = np.linspace(0, 1, n)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def _ease_in(n):  # accelerating 0..1 (fast at the end, like a downswing)
    return np.linspace(0, 1, n) ** 3


def _ease_out(n):
    return 1 - (1 - np.linspace(0, 1, n)) ** 3


def synthetic_hands(still_s=1.0, back_s=0.8, down_s=0.25, follow_s=0.4, hold_s=1.0):
    """Image coords (y down). Returns hands (T, 2) and the true frame indices."""
    n = lambda s: int(round(s * FPS))
    addr = np.array([500.0, 1000.0])
    top = np.array([300.0, 400.0])  # hands high: small y
    imp = np.array([500.0, 1020.0])  # hands lowest at impact
    fin = np.array([650.0, 350.0])
    seg = [
        np.tile(addr, (n(still_s), 1)),
        addr + np.outer(_ease_in_out(n(back_s)), top - addr),
        top + np.outer(_ease_in(n(down_s)), imp - top),
        imp + np.outer(_ease_out(n(follow_s)), fin - imp),
        np.tile(fin, (n(hold_s), 1)),
    ]
    hands = np.vstack(seg)
    truth = {
        "start": n(still_s),
        "top": n(still_s) + n(back_s) - 1,
        "impact": n(still_s) + n(back_s) + n(down_s) - 1,
        "follow_end": n(still_s) + n(back_s) + n(down_s) + n(follow_s) - 1,
    }
    return hands, truth


def test_detects_all_phases_in_order_close_to_truth():
    hands, t = synthetic_hands()
    r = detect_phases_from_hands(hands, FPS, TORSO)
    assert r.warnings == []
    f = r.frames
    assert abs(f["top"] - t["top"]) <= 3
    assert abs(f["impact"] - t["impact"]) <= 3
    # address/takeaway are defined by thresholds on a smoothly starting motion,
    # so allow up to 0.15 s after the true start, never before it
    assert 0 <= f["address"] - t["start"] <= 0.15 * FPS
    assert f["address"] <= f["takeaway"] <= f["top"]
    assert t["impact"] < f["finish"] <= t["follow_end"] + int(0.3 * FPS)
    assert f["address"] < f["takeaway"] < f["top"] < f["impact"] < f["finish"]


def test_robust_to_noise_and_offset_start():
    hands, t = synthetic_hands(still_s=2.5)
    rng = np.random.default_rng(0)
    hands = hands + rng.normal(0, 0.3, hands.shape)  # sub-pixel noise after smoothing
    r = detect_phases_from_hands(hands, FPS, TORSO)
    assert abs(r.frames["impact"] - t["impact"]) <= 3
    assert abs(r.frames["top"] - t["top"]) <= 5


def test_short_gap_after_impact_is_bridged():
    hands, t = synthetic_hands()
    gap = hands.copy()
    gap[t["impact"] + 2: t["impact"] + 30] = np.nan  # ~0.12 s blur gap
    filled = detect_phases_from_hands(_interp_short_gaps(gap, int(0.3 * FPS)), FPS, TORSO)
    assert abs(filled.frames["impact"] - t["impact"]) <= 5
    assert filled.frames["finish"] is not None


def test_missing_tail_gives_none_and_reason_not_a_guess():
    hands, t = synthetic_hands()
    hands[t["follow_end"] - 20:] = np.nan  # keypoints lost for the finish
    r = detect_phases_from_hands(hands, FPS, TORSO)
    assert r.frames["finish"] is None
    assert any("finish" in w for w in r.warnings)
    assert abs(r.frames["top"] - t["top"]) <= 3


def test_no_motion_returns_nothing():
    hands = np.tile([500.0, 1000.0], (500, 1))
    r = detect_phases_from_hands(hands, FPS, TORSO)
    assert all(v is None for v in r.frames.values()) and r.warnings


def test_all_nan_returns_nothing():
    r = detect_phases_from_hands(np.full((300, 2), np.nan), FPS, TORSO)
    assert all(v is None for v in r.frames.values()) and r.warnings


def test_no_still_period_before_backswing():
    hands, t = synthetic_hands()
    hands = hands[int(1.0 * FPS) + 100:]  # clip starts mid-backswing
    r = detect_phases_from_hands(hands, FPS, TORSO)
    assert r.frames["address"] is None
    assert any("address" in w for w in r.warnings)


def _seq_from_hands(hands, start_frame=0, drop_left=False):
    T = len(hands)
    k = np.stack([empty_pose() for _ in range(T)])
    k[:, :, 2] = 0.9
    for name in ("left_wrist", "right_wrist"):
        k[:, J[name], :2] = hands
    for name, dy in (("left_shoulder", -400), ("right_shoulder", -400),
                     ("left_hip", 0), ("right_hip", 0)):
        k[:, J[name], :2] = np.array([500.0, 1000.0 + dy])
    if drop_left:
        k[:, J["left_wrist"], :2] = np.nan
    return PoseSequence(k.astype(np.float32), FPS, 1080, 1920, "t", start_frame)


def test_detect_phases_uses_either_wrist_and_shifts_by_start_frame():
    hands, t = synthetic_hands()
    both = detect_phases(_seq_from_hands(hands, start_frame=100))
    one = detect_phases(_seq_from_hands(hands, start_frame=100, drop_left=True))
    assert both.frames == one.frames
    assert abs(both.frames["impact"] - (t["impact"] + 100)) <= 3


def test_hand_signal_means_wrists():
    hands, _ = synthetic_hands()
    seq = _seq_from_hands(hands)
    seq.keypoints[:, J["right_wrist"], 0] += 100
    np.testing.assert_allclose(hand_signal(seq)[:, 0], hands[:, 0] + 50, atol=1e-3)
