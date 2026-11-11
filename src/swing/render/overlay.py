"""Skeleton overlay rendering."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from swing.io import iter_frames
from swing.pose.base import J, PoseSequence

_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
]
# BGR. Lead side (left, for a right-handed golfer) orange, trail side cyan.
_LEFT, _RIGHT, _CENTER = (0, 140, 255), (255, 200, 0), (200, 200, 200)


def _color(a: str, b: str) -> tuple[int, int, int]:
    sides = {n.split("_")[0] for n in (a, b) if "_" in n}
    if sides == {"left"}:
        return _LEFT
    if sides == {"right"}:
        return _RIGHT
    return _CENTER


def draw_skeleton(frame: np.ndarray, pose: np.ndarray, min_conf: float = 0.3) -> np.ndarray:
    """Draw ``(17, 3)`` keypoints onto ``frame`` in place and return it.

    Joints that are NaN or below ``min_conf`` (and edges touching them) are skipped.
    """
    ok = np.isfinite(pose[:, :2]).all(axis=1) & (pose[:, 2] >= min_conf)
    thick = max(2, round(min(frame.shape[:2]) / 360))
    for a, b in _EDGES:
        ia, ib = J[a], J[b]
        if ok[ia] and ok[ib]:
            pa = tuple(int(round(v)) for v in pose[ia, :2])
            pb = tuple(int(round(v)) for v in pose[ib, :2])
            cv2.line(frame, pa, pb, _color(a, b), thick, cv2.LINE_AA)
    for i in np.flatnonzero(ok):
        p = tuple(int(round(v)) for v in pose[i, :2])
        cv2.circle(frame, p, thick + 1, (255, 255, 255), -1, cv2.LINE_AA)
    return frame


def write_overlay_video(video: str | Path, seq: PoseSequence, out: str | Path,
                        min_conf: float = 0.3) -> int:
    """Write ``video`` with skeletons drawn for the frames covered by ``seq``.

    Output keeps the source frame rate, so slo-mo clips stay slo-mo.
    Returns the number of frames written.
    """
    writer = None
    n = 0
    try:
        stop = seq.start_frame + len(seq)
        for idx, frame in iter_frames(video, seq.start_frame, stop):
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(out), cv2.VideoWriter_fourcc(*"mp4v"), seq.fps, (w, h)
                )
                if not writer.isOpened():
                    raise OSError(f"Could not open {out} for writing")
            draw_skeleton(frame, seq.keypoints[idx - seq.start_frame], min_conf)
            writer.write(frame)
            n += 1
    finally:
        if writer is not None:
            writer.release()
    return n
