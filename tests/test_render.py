import cv2
import numpy as np

from swing.io import probe_video
from swing.pose import J, PoseSequence, empty_pose
from swing.render import draw_skeleton, write_overlay_video


def _pose():
    p = empty_pose()
    p[J["left_shoulder"]] = (100, 80, 0.9)
    p[J["left_elbow"]] = (120, 130, 0.9)
    p[J["right_shoulder"]] = (160, 80, 0.1)  # below threshold
    return p


def test_draw_marks_confident_joints_only():
    frame = np.zeros((240, 320, 3), np.uint8)
    draw_skeleton(frame, _pose(), min_conf=0.3)
    assert frame[80, 100].any()  # left shoulder drawn
    assert not frame[80, 160].any()  # low-confidence right shoulder skipped


def test_draw_all_nan_is_noop():
    frame = np.zeros((240, 320, 3), np.uint8)
    draw_skeleton(frame, empty_pose())
    assert not frame.any()


def test_write_overlay_video(synthetic_video, tmp_path):
    kps = np.stack([_pose()] * 20)
    seq = PoseSequence(kps, fps=120.0, width=320, height=240, backend="t", start_frame=5)
    out = tmp_path / "o.mp4"
    assert write_overlay_video(synthetic_video, seq, out) == 20
    info = probe_video(out)
    assert info.frame_count == 20
    assert info.fps == 120.0
