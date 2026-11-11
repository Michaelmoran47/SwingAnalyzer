"""RTMPose backend via rtmlib + ONNX Runtime.

Top-down: a YOLOX person detector, then RTMPose on the best box. Output is
COCO-17, the same joint set the rest of the pipeline uses, so no remapping.

The default device is ``cpu`` (works everywhere). Use ``device="cuda"`` with
``onnxruntime-gpu`` installed on a machine with an NVIDIA GPU.

Confidence is NOT comparable with MediaPipe's visibility: RTMPose scores come
from SimCC peak values, so a ``min_conf`` tuned for one backend should not be
assumed right for the other. The benchmark script reports both.

Models are downloaded by rtmlib from download.openmmlab.com on first use and
cached under ``~/.cache/rtmlib``.
"""

from __future__ import annotations

import numpy as np

from swing.pose.base import NUM_JOINTS, PoseEstimator, empty_pose


def select_person(keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Pick the main subject from ``(N, 17, 2)`` keypoints / ``(N, 17)`` scores.

    The subject is the person whose keypoint bounding box is largest, weighted
    by mean score (the golfer is the foreground figure). Returns ``(17, 3)``
    ``x, y, confidence`` with confidence clipped to [0, 1], or all-NaN if nobody
    was found.
    """
    kp = np.asarray(keypoints, dtype=np.float32)
    sc = np.asarray(scores, dtype=np.float32)
    if kp.ndim != 3 or kp.shape[0] == 0:
        return empty_pose()
    if kp.shape[1:] != (NUM_JOINTS, 2) or sc.shape != kp.shape[:2]:
        raise ValueError(f"unexpected shapes {kp.shape}, {sc.shape}")
    extent = kp.max(axis=1) - kp.min(axis=1)  # (N, 2)
    best = int(np.argmax(extent[:, 0] * extent[:, 1] * sc.mean(axis=1)))
    out = np.empty((NUM_JOINTS, 3), dtype=np.float32)
    out[:, :2] = kp[best]
    out[:, 2] = np.clip(sc[best], 0.0, 1.0)
    return out


def check_device(device: str) -> None:
    """Fail loudly if a CUDA device is requested but ONNX Runtime cannot use it.

    Without this, onnxruntime quietly runs on the CPU (about 15x slower) and the
    run looks like it worked.
    """
    if not device.startswith("cuda"):
        return
    import onnxruntime as ort

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            f"device={device!r} requested but ONNX Runtime has no CUDAExecutionProvider "
            f"(available: {', '.join(providers)}). Install onnxruntime-gpu (and remove "
            "plain onnxruntime) with a matching CUDA/cuDNN, or use --device cpu."
        )


class RTMPoseEstimator(PoseEstimator):
    name = "rtmpose"

    def __init__(self, mode: str = "balanced", device: str = "cpu", model=None) -> None:
        """``mode``: rtmlib preset (``lightweight`` | ``balanced`` | ``performance``).

        ``model`` lets tests inject a callable returning ``(keypoints, scores)``.
        """
        if model is None:
            check_device(device)
            try:
                from rtmlib import Body
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "RTMPose backend needs the 'rtmpose' extra: pip install -e .[rtmpose]"
                ) from e
            model = Body(mode=mode, to_openpose=False, backend="onnxruntime", device=device)
        self._model = model
        self.mode, self.device = mode, device

    def estimate(self, frame_bgr: np.ndarray, timestamp_ms: float) -> np.ndarray:
        keypoints, scores = self._model(frame_bgr)
        return select_person(keypoints, scores)
