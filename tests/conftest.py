import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_video(tmp_path):
    """A 60-frame 240x320 clip at 120 fps; frame i has brightness encoding i."""
    path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 120.0, (320, 240))
    assert w.isOpened()
    for i in range(60):
        w.write(np.full((240, 320, 3), i * 4, dtype=np.uint8))
    w.release()
    return path
