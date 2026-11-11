"""MediaPipe BlazePose backend (default).

Uses the legacy ``mp.solutions.pose`` API. Complexity 0/1 models ship in the
wheel; complexity 2 (heavy, the default here) is fetched by MediaPipe itself on
first use. BlazePose yields 33 landmarks; we keep the 17 that map onto COCO.
"""

from __future__ import annotations

import numpy as np

from swing.pose.base import J, NUM_JOINTS, PoseEstimator, empty_pose

# COCO joint name -> BlazePose landmark index.
_BLAZE_INDEX = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}
_SRC = np.array([_BLAZE_INDEX[n] for n in sorted(_BLAZE_INDEX, key=J.get)])


def blaze_to_coco(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert ``(33, 4)`` normalised ``x, y, z, visibility`` to ``(17, 3)`` px.

    Pure function so the mapping is testable without MediaPipe installed.
    """
    lm = np.asarray(landmarks, dtype=np.float32)
    if lm.shape != (33, 4):
        raise ValueError(f"expected (33, 4) BlazePose landmarks, got {lm.shape}")
    out = np.empty((NUM_JOINTS, 3), dtype=np.float32)
    sel = lm[_SRC]
    out[:, 0] = sel[:, 0] * width
    out[:, 1] = sel[:, 1] * height
    out[:, 2] = sel[:, 3]
    return out


class MediaPipeEstimator(PoseEstimator):
    name = "mediapipe"

    def __init__(self, model_complexity: int = 2, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None:
        try:
            import mediapipe as mp
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "MediaPipe backend needs the 'mediapipe' extra: pip install -e .[mediapipe]"
            ) from e
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=False,  # smoothing is its own, testable stage
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def estimate(self, frame_bgr: np.ndarray, timestamp_ms: float) -> np.ndarray:
        import cv2

        h, w = frame_bgr.shape[:2]
        result = self._pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks is None:
            return empty_pose()
        lm = np.array(
            [[p.x, p.y, p.z, p.visibility] for p in result.pose_landmarks.landmark],
            dtype=np.float32,
        )
        return blaze_to_coco(lm, w, h)

    def close(self) -> None:
        self._pose.close()
