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



class _FakeSession:
    """Stands in for an onnxruntime session; ``after_run`` is what it reports post-run."""

    def __init__(self, after_run):
        self._providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._after_run = after_run

    def get_inputs(self):
        class _In:
            name, shape = "x", [1, 3, "h", 8]

        return [_In()]

    def run(self, _outputs, feed):
        assert feed["x"].shape == (1, 3, 1, 8)  # dynamic dims become 1
        self._providers = self._after_run

    def get_providers(self):
        return self._providers


class _FakeSub:
    def __init__(self, after_run):
        self.session = _FakeSession(after_run)


class _FakeBody:
    def __init__(self, det_after, pose_after):
        self.det_model, self.pose_model = _FakeSub(det_after), _FakeSub(pose_after)


_GPU = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_verify_gpu_execution_accepts_sessions_that_stay_on_gpu():
    from swing.pose.rtmpose_backend import verify_gpu_execution

    verify_gpu_execution(_FakeBody(_GPU, _GPU))


@pytest.mark.parametrize("which", ["det_model", "pose_model"])
def test_verify_gpu_execution_catches_silent_cpu_fallback(which):
    """onnxruntime retries a failed GPU run on the CPU and relabels the session."""
    from swing.pose.rtmpose_backend import verify_gpu_execution

    cpu = ["CPUExecutionProvider"]
    body = _FakeBody(cpu if which == "det_model" else _GPU, cpu if which == "pose_model" else _GPU)
    with pytest.raises(RuntimeError, match=f"{which} ran on .*CPUExecutionProvider"):
        verify_gpu_execution(body)


def test_verify_gpu_execution_refuses_to_guess_when_rtmlib_api_changes():
    from swing.pose.rtmpose_backend import verify_gpu_execution

    with pytest.raises(RuntimeError, match="cannot verify"):
        verify_gpu_execution(object())

def test_estimator_construction_checks_device(monkeypatch):
    import onnxruntime as ort

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError):
        RTMPoseEstimator(device="cuda")
    # an injected model bypasses model loading, so no device check is needed
    RTMPoseEstimator(device="cuda", model=lambda f: (np.zeros((0, 17, 2)), np.zeros((0, 17))))
