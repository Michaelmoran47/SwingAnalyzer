"""Per-frame geometric building blocks. Pure functions of keypoint arrays.

Keypoints are ``(T, 17, 3)`` (x_px, y_px, confidence), image y pointing down.
Everything here is a 2D image-plane quantity; none of it is a true 3D angle.
NaN in, NaN out.
"""

from __future__ import annotations

import warnings

import numpy as np

from swing.pose.base import J


def pt(k: np.ndarray, name: str) -> np.ndarray:
    return k[:, J[name], :2]


def mid(k: np.ndarray, a: str, b: str) -> np.ndarray:
    return (pt(k, a) + pt(k, b)) / 2.0


def spine_vector(k: np.ndarray) -> np.ndarray:
    """Hip-mid to shoulder-mid, ``(T, 2)``."""
    return mid(k, "left_shoulder", "right_shoulder") - mid(k, "left_hip", "right_hip")


def spine_angle_deg(k: np.ndarray) -> np.ndarray:
    """Unsigned angle of the spine line from vertical in the image plane, degrees.

    0 = upright. From a down-the-line camera this is the forward bend; from a
    face-on camera it is the lateral tilt.
    """
    v = spine_vector(k)
    return np.degrees(np.arctan2(np.abs(v[:, 0]), -v[:, 1]))


def head_point(k: np.ndarray) -> np.ndarray:
    """Midpoint of the two ears. NaN unless both ears are present.

    Ears move less than the nose when the head turns, which matters for sway.
    """
    return mid(k, "left_ear", "right_ear")


def width(k: np.ndarray, left: str, right: str) -> np.ndarray:
    """Projected distance between two joints (shoulder or hip width), px."""
    return np.linalg.norm(pt(k, left) - pt(k, right), axis=1)


def reference_pose(k: np.ndarray, frame: int, fps: float, window_s: float = 0.2) -> np.ndarray:
    """Median pose ``(1, 17, 3)`` over the ``window_s`` ending at ``frame``.

    A median over a short window is steadier than a single frame.
    """
    lo = max(0, frame - int(round(window_s * fps)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # joint never visible -> NaN
        return np.nanmedian(k[lo:frame + 1], axis=0, keepdims=True)


def torso_length(ref: np.ndarray) -> float:
    """Projected hip-mid to shoulder-mid length of a ``(1, 17, 3)`` pose, px."""
    return float(np.linalg.norm(spine_vector(ref), axis=1)[0])


def arccos_ratio_deg(w: np.ndarray, w_ref: float) -> np.ndarray:
    """Rotation proxy: ``arccos(w / w_ref)`` in degrees, ratio clipped to [0, 1].

    Assumes the segment is square to the camera when ``w_ref`` was measured and
    rotates about the vertical axis. Unsigned, and ambiguous past 90 degrees.
    A proxy, not a measured rotation.
    """
    return np.degrees(np.arccos(np.clip(w / w_ref, 0.0, 1.0)))
