"""Backend-agnostic pose types.

All backends emit the COCO-17 joint set so downstream code (smoothing, phases,
metrics) never depends on a particular model. Coordinates are image pixels
(x right, y down); confidence is in [0, 1]. Missing detections are NaN.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

JOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
J = {name: i for i, name in enumerate(JOINTS)}
NUM_JOINTS = len(JOINTS)

# Left/right are the subject's anatomical sides. For the right-handed golfer
# this project targets, the lead side is the left.


class PoseEstimator(ABC):
    """Single-frame pose estimator. Stateful backends may track across frames."""

    name: str

    @abstractmethod
    def estimate(self, frame_bgr: np.ndarray, timestamp_ms: float) -> np.ndarray:
        """Return ``(17, 3)`` array of ``x_px, y_px, confidence`` (NaN if no person)."""

    def close(self) -> None:  # pragma: no cover - default no-op
        pass

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def empty_pose() -> np.ndarray:
    return np.full((NUM_JOINTS, 3), np.nan, dtype=np.float32)


@dataclass
class PoseSequence:
    """Keypoints for a whole clip: ``keypoints`` is ``(T, 17, 3)``."""

    keypoints: np.ndarray
    fps: float
    width: int
    height: int
    backend: str
    start_frame: int = 0
    # Optional (T, 17) bool: joint position was interpolated, not measured.
    filled: np.ndarray | None = None

    def __len__(self) -> int:
        return self.keypoints.shape[0]

    @property
    def detected(self) -> np.ndarray:
        """Boolean ``(T,)``: frame has at least one finite keypoint."""
        return np.isfinite(self.keypoints[:, :, 0]).any(axis=1)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            keypoints=self.keypoints,
            fps=self.fps,
            width=self.width,
            height=self.height,
            backend=self.backend,
            start_frame=self.start_frame,
            joints=np.array(JOINTS),
            filled=(np.zeros(self.keypoints.shape[:2], bool)
                    if self.filled is None else self.filled),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PoseSequence":
        with np.load(path, allow_pickle=False) as d:
            if tuple(d["joints"]) != JOINTS:
                raise ValueError("Joint order in file does not match this version.")
            return cls(
                keypoints=d["keypoints"],
                fps=float(d["fps"]),
                width=int(d["width"]),
                height=int(d["height"]),
                backend=str(d["backend"]),
                start_frame=int(d["start_frame"]),
                filled=d["filled"] if "filled" in d.files else None,
            )
