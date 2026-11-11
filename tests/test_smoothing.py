import numpy as np
import pytest

from swing.pose import PoseSequence
from swing.smoothing import fill_gaps, savgol_smooth, smooth_sequence

FPS = 240.0


def _kps(x, y=None, conf=0.9):
    """(T, 17, 3) with joint 0 carrying the signal, the rest constant."""
    T = len(x)
    k = np.zeros((T, 17, 3), np.float32)
    k[:, :, 2] = conf
    k[:, 0, 0] = x
    k[:, 0, 1] = x if y is None else y
    return k


def test_fill_short_gap_linear_and_flagged():
    x = np.arange(50, dtype=np.float32)
    k = _kps(x)
    k[20:23, 0, :2] = np.nan
    out, filled = fill_gaps(k, FPS, max_gap_s=0.05)  # 12 frames allowed
    np.testing.assert_allclose(out[20:23, 0, 0], [20, 21, 22], atol=1e-4)
    assert filled[20:23, 0].all() and filled.sum() == 3


def test_fill_leaves_long_gap_and_edges_nan():
    k = _kps(np.arange(60, dtype=np.float32))
    k[10:40, 0, :2] = np.nan  # 30 frames > 12
    k[:2, 0, :2] = np.nan  # no left neighbour
    out, filled = fill_gaps(k, FPS, max_gap_s=0.05)
    assert np.isnan(out[10:40, 0, :2]).all()
    assert np.isnan(out[:2, 0, :2]).all()
    assert not filled[:, 0].any()


def test_low_confidence_is_masked_then_filled_but_confidence_kept():
    k = _kps(np.arange(30, dtype=np.float32))
    k[10, 0, 0] = 999.0  # garbage value
    k[10, 0, 2] = 0.05
    out, filled = fill_gaps(k, FPS, min_conf=0.3)
    assert out[10, 0, 0] == pytest.approx(10.0)
    assert out[10, 0, 2] == pytest.approx(0.05)  # not inflated
    assert filled[10, 0]


def test_nan_confidence_becomes_zero():
    k = _kps(np.arange(30, dtype=np.float32))
    k[5, 0, :] = np.nan
    out, _ = fill_gaps(k, FPS)
    assert out[5, 0, 2] == 0.0


def test_smoothing_reduces_noise_error():
    rng = np.random.default_rng(1)
    t = np.arange(600) / FPS
    clean = 200 * np.sin(2 * np.pi * 1.5 * t)
    noisy = clean + rng.normal(0, 3, t.size)
    out = savgol_smooth(_kps(noisy), FPS)[:, 0, 0]
    rmse_raw = np.sqrt(np.mean((noisy - clean) ** 2))
    rmse_smooth = np.sqrt(np.mean((out - clean) ** 2))
    assert rmse_smooth < 0.5 * rmse_raw


def test_smoothing_has_no_time_lag():
    t = np.arange(600) / FPS
    pulse = 100 * np.exp(-0.5 * ((t - 1.25) / 0.05) ** 2)  # fast peak at 1.25 s
    out = savgol_smooth(_kps(pulse), FPS)[:, 0, 0]
    assert abs(int(np.argmax(out)) - int(np.argmax(pulse))) <= 1


def test_smoothing_preserves_linear_motion():
    x = np.linspace(0, 300, 400)
    out = savgol_smooth(_kps(x), FPS)[:, 0, 0]
    np.testing.assert_allclose(out, x, atol=1e-2)


def test_short_runs_untouched_and_nan_safe():
    k = _kps(np.arange(40, dtype=np.float32))
    k[5, 0, :2] = np.nan
    k[6:, 0, :2] += 3  # run after the NaN is shorter than the window? 34 > 14, fine
    out = savgol_smooth(k, FPS)
    assert np.isnan(out[5, 0, :2]).all()
    short = _kps(np.arange(5, dtype=np.float32))
    np.testing.assert_array_equal(savgol_smooth(short, FPS), short)


def test_smooth_sequence_end_to_end_keeps_shape_and_conf():
    rng = np.random.default_rng(2)
    k = _kps(np.linspace(0, 100, 300) + rng.normal(0, 2, 300))
    k[100:104, 0, :2] = np.nan
    seq = PoseSequence(k, FPS, 1080, 1920, "t", 0)
    out = smooth_sequence(seq)
    assert out.keypoints.shape == k.shape
    assert out.filled[100:104, 0].all()
    assert np.isfinite(out.keypoints[:, 0, :2]).all()
    np.testing.assert_array_equal(out.keypoints[:, :, 2], k[:, :, 2])
