import numpy as np
import pytest

from swing.pose import get_estimator
from swing.pose.rtmpose_backend import RTMPoseEstimator, select_person


def _person(cx, size, score):
    """17 keypoints spread over a square of side ``size`` centred at (cx, cx)."""
    rng = np.random.default_rng(int(cx))
    kp = cx + (rng.random((17, 2)) - 0.5) * size
    return kp, np.full(17, score)


def test_picks_largest_confident_person():
    small, s_small = _person(100, 50, 0.9)
    big, s_big = _person(400, 300, 0.8)
    out = select_person(np.stack([small, big]), np.stack([s_small, s_big]))
    np.testing.assert_allclose(out[:, :2], big, rtol=1e-5)
    assert np.allclose(out[:, 2], 0.8)


def test_large_but_low_confidence_loses_to_confident_medium():
    a, sa = _person(100, 200, 0.05)  # big box, junk scores
    b, sb = _person(400, 150, 0.9)
    out = select_person(np.stack([a, b]), np.stack([sa, sb]))
    np.testing.assert_allclose(out[:, :2], b, rtol=1e-5)


def test_no_person_gives_all_nan():
    out = select_person(np.zeros((0, 17, 2)), np.zeros((0, 17)))
    assert out.shape == (17, 3) and np.isnan(out).all()


def test_scores_clipped_to_unit_interval():
    kp, sc = _person(300, 100, 1.0)
    sc[0], sc[1] = 1.7, -0.2
    out = select_person(kp[None], sc[None])
    assert out[0, 2] == 1.0 and out[1, 2] == 0.0


def test_bad_shapes_rejected():
    with pytest.raises(ValueError):
        select_person(np.zeros((1, 15, 2)), np.zeros((1, 15)))


def test_estimator_with_injected_model():
    kp, sc = _person(300, 120, 0.7)
    calls = []

    def fake(frame):
        calls.append(frame.shape)
        return kp[None], sc[None]

    est = RTMPoseEstimator(model=fake)
    out = est.estimate(np.zeros((480, 640, 3), np.uint8), 0.0)
    assert calls == [(480, 640, 3)] and out.shape == (17, 3)
    assert est.name == "rtmpose"


def test_factory_unknown_backend_lists_rtmpose():
    with pytest.raises(ValueError, match="rtmpose"):
        get_estimator("nope")


def test_cuda_requested_without_cuda_provider_fails_loudly(monkeypatch):
    import onnxruntime as ort

    from swing.pose.rtmpose_backend import check_device

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        check_device("cuda")
    check_device("cpu")  # never raises
    monkeypatch.setattr(ort, "get_available_providers",
                        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    check_device("cuda")  # fine when available


def test_estimator_construction_checks_device(monkeypatch):
    import onnxruntime as ort

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError):
        RTMPoseEstimator(device="cuda")
    # an injected model bypasses model loading, so no device check is needed
    RTMPoseEstimator(device="cuda", model=lambda f: (np.zeros((0, 17, 2)), np.zeros((0, 17))))
