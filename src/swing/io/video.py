"""Video ingest via OpenCV.

Timestamps are derived from ``frame_index / fps``. For iPhone slo-mo clips the
container metadata can be misleading (slow-motion is stored with a time-ramp
edit list), so :func:`probe_video` reports what OpenCV decodes and
``VideoInfo.warnings`` flags anything suspicious. Verify with ``ffprobe`` if in
doubt -- see docs/capture_protocol.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

# Below this, tempo / phase timing resolution is too coarse for a golf swing
# (impact region lasts only a few ms).
MIN_RECOMMENDED_FPS = 100.0


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int  # after orientation correction
    height: int
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def time_of(self, frame_index: int) -> float:
        """Timestamp in seconds of a frame, assuming constant frame rate."""
        return frame_index / self.fps


def _open(path: Path) -> cv2.VideoCapture:
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(
            f"OpenCV could not open {path}. iPhone HEVC .MOV may need a "
            "decoder; try converting to H.264 with ffmpeg."
        )
    # Apply the rotation metadata (iPhone portrait clips) when supported.
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return cap


def probe_video(path: str | Path) -> VideoInfo:
    path = Path(path)
    cap = _open(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Could not decode a first frame from {path}")
        height, width = frame.shape[:2]  # decoded size, orientation applied
    finally:
        cap.release()

    warnings: list[str] = []
    if fps <= 0:
        warnings.append("Container reports no frame rate; timestamps unreliable.")
    elif fps < MIN_RECOMMENDED_FPS:
        warnings.append(
            f"{fps:.1f} fps is below the recommended {MIN_RECOMMENDED_FPS:.0f}; "
            "was the clip recorded in slo-mo or exported at normal speed?"
        )
    if count <= 0:
        warnings.append("Container reports no frame count.")
    return VideoInfo(path, fps, count, width, height, tuple(warnings))


def iter_frames(
    path: str | Path, start: int = 0, end: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, BGR frame)`` for ``start <= index < end``."""
    cap = _open(Path(path))
    try:
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        while end is None or idx < end:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()
