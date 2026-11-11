import numpy as np
import pytest

from swing.pose import J, JOINTS, PoseSequence, empty_pose
from swing.pose.mediapipe_backend import blaze_to_coco


def test_blaze_mapping_picks_correct_landmarks():
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 0] = np.arange(33) / 100.0  # x encodes the BlazePose index
    lm[:, 3] = 0.9
    out = blaze_to_coco(lm, width=1000, height=500)
    assert out.shape == (17, 3)
    assert out[J["nose"], 0] == pytest.approx(0)
    assert out[J["left_shoulder"], 0] == pytest.approx(110)  # index 11
    assert out[J["right_ankle"], 0] == pytest.approx(280)  # index 28
    assert out[J["left_wrist"], 0] == pytest.approx(150)  # index 15
    assert np.allclose(out[:, 2], 0.9)


def test_blaze_scales_y_by_height():
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 1] = 0.5
    out = blaze_to_coco(lm, width=1000, height=500)
    assert np.allclose(out[:, 1], 250)


def test_blaze_bad_shape():
    with pytest.raises(ValueError):
        blaze_to_coco(np.zeros((17, 4)), 10, 10)


def test_sequence_roundtrip(tmp_path):
    kps = np.random.default_rng(0).random((5, len(JOINTS), 3)).astype(np.float32)
    kps[2] = empty_pose()
    seq = PoseSequence(kps, fps=240.0, width=1080, height=1920, backend="x", start_frame=7)
    seq.save(tmp_path / "k.npz")
    back = PoseSequence.load(tmp_path / "k.npz")
    np.testing.assert_array_equal(back.keypoints, kps)
    assert (back.fps, back.width, back.height, back.backend, back.start_frame) == (
        240.0, 1080, 1920, "x", 7)
    assert back.detected.tolist() == [True, True, False, True, True]
