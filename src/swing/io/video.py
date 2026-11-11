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
    """Measure fps and frame count from the decoded stream, not the header.

    iPhone slo-mo headers can disagree with the stream (observed: header said
    227.8 fps / 1635 frames, the decoded stream was 240 fps / 1551 frames), so
    we walk every frame with ``grab()`` (no pixel conversion) and use the
    presentation timestamps.
    """
    path = Path(path)
    cap = _open(path)
    warnings: list[str] = []
    try:
        header_fps = float(cap.get(cv2.CAP_PROP_FPS))
        header_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Could not decode a first frame from {path}")
        height, width = frame.shape[:2]  # decoded size, orientation applied
        stamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
        while cap.grab():
            stamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    finally:
        cap.release()

    count = len(stamps)
    fps = 0.0
    if count > 1 and stamps[-1] > stamps[0]:
        # Mean rate over the clip; per-frame ms stamps are quantised so a
        # median of single gaps would be noisy.
        fps = (count - 1) * 1000.0 / (stamps[-1] - stamps[0])
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        if max(gaps) > 2.0 * min(g for g in gaps if g > 0):
            warnings.append(
                "Frame spacing varies by more than 2x: variable frame rate, "
                "likely the iPhone slo-mo time ramp. Trim to the full-rate segment."
            )
    else:
        fps = header_fps
        warnings.append("Could not read frame timestamps; using header fps.")
    if header_fps > 0 and abs(header_fps - fps) / max(fps, 1e-9) > 0.02:
        warnings.append(
            f"Header says {header_fps:.1f} fps / {header_count} frames but the "
            f"decoded stream is {fps:.1f} fps / {count} frames; using the stream."
        )
    if fps <= 0:
        warnings.append("No frame rate available; timestamps unreliable.")
    elif fps < MIN_RECOMMENDED_FPS:
        warnings.append(
            f"{fps:.1f} fps is below the recommended {MIN_RECOMMENDED_FPS:.0f}; "
            "was the clip recorded in slo-mo or exported at normal speed?"
        )
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
