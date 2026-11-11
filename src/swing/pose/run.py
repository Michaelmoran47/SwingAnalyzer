"""Run an estimator over a video."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from swing.io import iter_frames, probe_video
from swing.pose.base import NUM_JOINTS, PoseEstimator, PoseSequence

BACKENDS = ("mediapipe",)  # rtmpose arrives in a later milestone


def get_estimator(name: str, **kwargs) -> PoseEstimator:
    if name == "mediapipe":
        from swing.pose.mediapipe_backend import MediaPipeEstimator

        return MediaPipeEstimator(**kwargs)
    raise ValueError(f"Unknown pose backend {name!r}; available: {', '.join(BACKENDS)}")


def estimate_video(
    path: str | Path,
    estimator: PoseEstimator,
    start: int = 0,
    end: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> PoseSequence:
    info = probe_video(path)
    stop = info.frame_count if end is None else min(end, info.frame_count)
    poses: list[np.ndarray] = []
    for idx, frame in iter_frames(path, start, stop):
        poses.append(estimator.estimate(frame, timestamp_ms=info.time_of(idx) * 1000.0))
        if progress:
            progress(idx - start + 1, stop - start)
    kps = np.stack(poses) if poses else np.empty((0, NUM_JOINTS, 3), np.float32)
    return PoseSequence(kps, info.fps, info.width, info.height, estimator.name, start)
