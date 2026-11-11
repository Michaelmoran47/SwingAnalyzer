import numpy as np
import pytest

from swing.metrics import compute_metrics, tempo
from swing.phases import PhaseResult
from swing.pose import J, PoseSequence

FPS = 240.0
T = 400
TORSO = 300.0
FRAMES = {"address": 100, "takeaway": 140, "top": 250, "impact": 320, "finish": 380}
KEYS = [0, 100, 250, 320, T - 1]


def _series(vals):
    """Piecewise-linear per-frame series through the phase keyframes."""
    return np.interp(np.arange(T), KEYS, vals)


def make_seq(*, flip=1, spine=(30, 30, 30, 30, 30), hip_dx=(0,) * 5, head_dx=(0,) * 5,
             sh_w=(120,) * 5, hip_w=(90,) * 5, hands_ahead=100.0, start_frame=0):
    """Skeleton sequence. ``flip=-1`` mirrors the scene (camera on the other side).

    Golfer's left hip / shoulder / ear sit at +x for flip=1. Spine leans toward
    +x (the ball side in a dtl view). Wrists are ``hands_ahead`` px ahead of the hips.
    """
    k = np.full((T, 17, 3), np.nan, np.float32)
    k[:, :, 2] = 0.9
    a = np.radians(_series(spine))
    hip = np.stack([500 + _series(hip_dx), np.full(T, 1000.0)], axis=1)
    sh = hip + TORSO * np.stack([np.sin(a), -np.cos(a)], axis=1)
    head = sh + np.stack([_series(head_dx), np.full(T, -80.0)], axis=1)
    hw, sw = _series(hip_w) / 2, _series(sh_w) / 2
    z = np.zeros(T)

    def put(name, p):
        k[:, J[name], :2] = p
    put("left_hip", hip + np.stack([hw, z], 1)); put("right_hip", hip - np.stack([hw, z], 1))
    put("left_shoulder", sh + np.stack([sw, z], 1)); put("right_shoulder", sh - np.stack([sw, z], 1))
    put("left_ear", head + np.array([30.0, 0])); put("right_ear", head - np.array([30.0, 0]))
    wr = hip + np.array([hands_ahead, -50.0])
    put("left_wrist", wr); put("right_wrist", wr)
    k[:, :, 0] = 500 + flip * (k[:, :, 0] - 500)  # mirror about x=500
    seq = PoseSequence(k, FPS, 1080, 1920, "synthetic", start_frame=start_frame)
    ph = PhaseResult({p: f + start_frame for p, f in FRAMES.items()}, FPS)
    return seq, ph


# ---- tempo -----------------------------------------------------------------

def test_tempo_values():
    t = tempo(FRAMES, FPS)
    assert t["backswing_s"] == pytest.approx(110 / FPS)
    assert t["downswing_s"] == pytest.approx(70 / FPS)
    assert t["tempo_ratio"] == pytest.approx(110 / 70)


@pytest.mark.parametrize("bad", [
    {"takeaway": None, "top": 250, "impact": 320},
    {"takeaway": 300, "top": 250, "impact": 320},  # out of order
])
def test_tempo_none_when_unusable(bad):
    assert tempo(bad, FPS) == {"backswing_s": None, "downswing_s": None, "tempo_ratio": None}


# ---- spine angle -----------------------------------------------------------

@pytest.mark.parametrize("flip", [1, -1])
def test_spine_angle_unsigned_and_mirror_invariant(flip):
    seq, ph = make_seq(flip=flip, spine=(35, 35, 33, 25, 25))
    v = compute_metrics(seq, ph, "dtl").values
    assert v["spine_angle_address_deg"] == pytest.approx(35, abs=0.01)
    assert v["spine_angle_top_deg"] == pytest.approx(33, abs=0.01)
    assert v["spine_angle_impact_deg"] == pytest.approx(25, abs=0.01)
    assert v["spine_angle_change_to_impact_deg"] == pytest.approx(-10, abs=0.01)


def test_spine_angle_in_both_views():
    seq, ph = make_seq(spine=(20,) * 5)
    for view in ("dtl", "fo"):
        assert compute_metrics(seq, ph, view).values["spine_angle_address_deg"] == pytest.approx(20, abs=0.01)


# ---- dtl: early extension indicator ---------------------------------------

@pytest.mark.parametrize("flip", [1, -1])
def test_hip_toward_ball_positive_when_hips_move_to_ball(flip):
    # hips (in scene coordinates before mirroring) move +x = toward the ball at impact
    seq, ph = make_seq(flip=flip, hip_dx=(0, 0, -15, 0.12 * TORSO, 0.12 * TORSO))
    v = compute_metrics(seq, ph, "dtl").values
    assert v["hip_toward_ball_impact_torso"] == pytest.approx(0.12, abs=1e-3)
    assert v["hip_toward_ball_top_torso"] == pytest.approx(-15 / TORSO, abs=1e-3)


def test_hip_toward_ball_negative_when_hips_move_away():
    seq, ph = make_seq(hip_dx=(0, 0, 0, -0.1 * TORSO, -0.1 * TORSO))
    assert compute_metrics(seq, ph, "dtl").values["hip_toward_ball_impact_torso"] == pytest.approx(-0.1, abs=1e-3)


def test_dtl_ambiguous_ball_direction_is_unavailable_not_guessed():
    seq, ph = make_seq(hands_ahead=0.0)
    r = compute_metrics(seq, ph, "dtl")
    assert r.values["hip_toward_ball_impact_torso"] is None
    assert "ball" in r.unavailable["hip_toward_ball_impact_torso"]


# ---- fo: hip slide, head sway, rotation proxies ----------------------------

@pytest.mark.parametrize("flip", [1, -1])
def test_hip_slide_toward_target_positive(flip):
    seq, ph = make_seq(flip=flip, hip_dx=(0, 0, -0.05 * TORSO, 0.08 * TORSO, 0.08 * TORSO))
    v = compute_metrics(seq, ph, "fo").values
    assert v["hip_slide_impact_torso"] == pytest.approx(0.08, abs=1e-3)
    assert v["hip_slide_top_torso"] == pytest.approx(-0.05, abs=1e-3)


@pytest.mark.parametrize("flip", [1, -1])
def test_head_sway(flip):
    seq, ph = make_seq(flip=flip, head_dx=(0, 0, -0.06 * TORSO, 0.02 * TORSO, 0.02 * TORSO))
    v = compute_metrics(seq, ph, "fo").values
    assert v["head_sway_top_torso"] == pytest.approx(-0.06, abs=1e-3)
    assert v["head_sway_impact_torso"] == pytest.approx(0.02, abs=1e-3)
    assert v["head_sway_max_excursion_torso"] == pytest.approx(0.06, abs=1e-3)


def test_rotation_proxies_follow_cosine():
    # shoulder width shrinks to cos(45 deg), hip width to cos(30 deg) at the top
    seq, ph = make_seq(sh_w=(120, 120, 120 * np.cos(np.radians(45)), 120, 120),
                       hip_w=(90, 90, 90 * np.cos(np.radians(30)), 90, 90))
    v = compute_metrics(seq, ph, "fo").values
    assert v["shoulder_turn_proxy_top_deg"] == pytest.approx(45, abs=0.05)
    assert v["hip_turn_proxy_top_deg"] == pytest.approx(30, abs=0.05)


def test_head_metrics_unavailable_when_ears_missing():
    seq, ph = make_seq()
    seq.keypoints[:, J["left_ear"], :2] = np.nan
    r = compute_metrics(seq, ph, "fo")
    assert r.values["head_sway_impact_torso"] is None
    assert r.values["hip_slide_impact_torso"] is not None  # others unaffected


# ---- view gating, missing phases, offsets ---------------------------------

def test_metrics_gated_by_view():
    seq, ph = make_seq()
    dtl, fo = compute_metrics(seq, ph, "dtl"), compute_metrics(seq, ph, "fo")
    assert dtl.values["hip_slide_impact_torso"] is None and "face-on" in dtl.unavailable["hip_slide_impact_torso"]
    assert dtl.values["shoulder_turn_proxy_top_deg"] is None
    assert "hip_toward_ball_impact_torso" not in fo.values
    assert fo.values["shoulder_turn_proxy_top_deg"] is not None


def test_invalid_view():
    seq, ph = make_seq()
    with pytest.raises(ValueError):
        compute_metrics(seq, ph, "overhead")


def test_missing_phase_reports_reason_instead_of_guessing():
    seq, ph = make_seq()
    ph.frames["address"] = None
    r = compute_metrics(seq, ph, "fo")
    assert r.values["spine_angle_address_deg"] is None
    assert r.values["hip_slide_impact_torso"] is None
    assert "address" in r.unavailable["hip_slide_impact_torso"]


def test_sequence_start_frame_offset():
    seq, ph = make_seq(start_frame=500, spine=(35, 35, 33, 25, 25))
    assert compute_metrics(seq, ph, "dtl").values["spine_angle_impact_deg"] == pytest.approx(25, abs=0.01)
