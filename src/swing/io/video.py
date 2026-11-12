"""Video ingest via OpenCV.

Timestamps are derived from ``frame_index / fps``. For iPhone slo-mo clips the
container metadata can be misleading (slow-motion is stored with a time-ramp
edit list), so :func:`probe_video` reports what OpenCV decodes and
``VideoInfo.warnings`` flags anything suspicious. Verify with ``ffprobe`` if in
doubt -- see docs/capture_protocol.md.
"""

from __future__ import annotations

import hashlib
import os
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


def is_variable_rate(stamps) -> bool:
    """True if frame spacing varies by more than 2x across the clip (a time ramp).

    Uses the 2nd and 98th percentile of the gaps, so a single odd gap (a dropped or
    duplicated frame, rounding at the ends) does not count as a variable rate.
    """
    gaps = np.diff(np.asarray(stamps, dtype=float))
    gaps = gaps[gaps > 0]
    if len(gaps) < 2:
        return False
    lo, hi = np.percentile(gaps, [2, 98]) if len(gaps) >= 20 else (gaps.min(), gaps.max())
    return bool(hi > 2.0 * lo)


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
        if is_variable_rate(stamps):
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
        # Skip by decoding, never by cap.set(CAP_PROP_POS_FRAMES): seeking trusts the
        # container header, and on IMG_6004.MOV (header 227.8 fps / 1635 frames, stream
        # 239.5 fps / 1551 frames) it landed on the wrong frame (start=985 gave frame
        # 1035, start=500 gave 515). Frame indices must mean "n-th decoded frame",
        # the same as `swing label` and a run from frame 0.
        for _ in range(start):
            if not cap.grab():
                return
        idx = start
        while end is None or idx < end:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def load_jpeg_frames(
    path: str | Path, scale: float, start: int = 0, end: int | None = None, quality: int = 90
) -> list[bytes]:
    """Decode ``[start, end)`` once, downscale, keep JPEG bytes (full-res clips need GBs raw)."""
    out = []
    for _, frame in iter_frames(path, start, end):
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        out.append(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes())
    return out


_CACHE_VERSION = 1


def _cache_key(path: Path, scale: float, quality: int) -> str:
    st = path.stat()
    raw = f"v{_CACHE_VERSION}|{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{scale:.6f}|{quality}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def store_jpeg_frames(
    path: str | Path, scale: float, cache_dir: str | Path, frames: list[bytes], quality: int = 90
) -> Path:
    """Write already-decoded ``frames`` of ``path`` into the cache (what ``cached_jpeg_frames`` reads).

    Lets a caller that decoded a clip some other way (for example an upload that was
    analysed from a temporary file) register it under the file's final location without
    decoding again. Removes entries for the same video name with a different key.
    """
    import numpy as np

    path, cache_dir = Path(path), Path(cache_dir)
    entry = cache_dir / f"{path.stem}_{_cache_key(path, scale, quality)}.frames.npz"
    cache_dir.mkdir(parents=True, exist_ok=True)
    offsets = np.cumsum([0] + [len(f) for f in frames]).astype(np.int64)
    tmp = entry.with_name(entry.name + f".{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:  # file handle: np.savez would append ".npz" to a name
            np.savez(fh, data=np.frombuffer(b"".join(frames), np.uint8), offsets=offsets)
        os.replace(tmp, entry)  # atomic: readers never see a half-written file
    finally:
        tmp.unlink(missing_ok=True)
    for old in cache_dir.glob(f"{path.stem}_*.frames.npz"):
        if old != entry:
            old.unlink(missing_ok=True)
    return entry


def cached_jpeg_frames(
    path: str | Path, scale: float, cache_dir: str | Path, quality: int = 90
) -> tuple[list[bytes], bool]:
    """Like :func:`load_jpeg_frames` for a whole clip, but saved to ``cache_dir``.

    Returns ``(frames, from_cache)``. The cache entry is keyed on the file's path,
    size and modification time plus scale and quality, so editing or replacing the
    video invalidates it. Entries for the same video with a different key are
    removed. A missing, truncated or unreadable cache file is silently rebuilt.
    The cache is safe to delete at any time.
    """
    import numpy as np

    path, cache_dir = Path(path), Path(cache_dir)
    key = _cache_key(path, scale, quality)
    entry = cache_dir / f"{path.stem}_{key}.frames.npz"
    if entry.is_file():
        try:
            with np.load(entry, allow_pickle=False) as d:
                data, offsets = d["data"].tobytes(), d["offsets"]
            return [data[a:b] for a, b in zip(offsets[:-1], offsets[1:])], True
        except Exception:
            pass  # corrupt or truncated: rebuild below
    frames = load_jpeg_frames(path, scale, quality=quality)
    store_jpeg_frames(path, scale, cache_dir, frames, quality)
    return frames, False

