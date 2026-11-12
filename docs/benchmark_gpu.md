# Backend benchmark: IMG_6004.MOV (dtl)

Generated 2026-11-11 by `scripts/benchmark_backends.py`. One clip, one swing: a comparison on this footage, **not** an accuracy ranking (no ground-truth keypoints exist).

Swing window = your hand labels takeaway..finish, frames 731-1117. Machine: Intel64 Family 6 Model 94 Stepping 3, GenuineIntel, Windows 10.

**GPU machine (added by hand, the script only prints the CPU string):** Intel Core i7-6700 @ 3.40 GHz; NVIDIA GeForce RTX 4060 (8 GB), driver 560.94 (CUDA 12.6 per `nvidia-smi`); Python 3.11.9; onnxruntime-gpu 1.22.0; CUDA 12.9 runtime/cuBLAS and cuDNN 9.10.2.21 installed as pip `nvidia-*-cu12` packages; rtmlib 0.0.16; mediapipe 0.10.21 (assumed CPU-only: the legacy `mp.solutions.pose` API; GPU use was not monitored). Read the mediapipe s/frame below as CPU time on this machine and the rtmpose s/frame as GPU time.

## Speed (this machine)

ONNX Runtime providers available: TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider; rtmpose device = `cuda`, mode = `performance`. Per-frame time at full video resolution IMG_6004.MOV, 30 frames after warm-up, starting mid-swing.

| backend | s/frame | fps |
|---|---|---|
| mediapipe | 0.088 | 11.4 |
| rtmpose | 0.069 | 14.5 |

## Completeness in the swing window

Share of frames where the joint is usable. *Raw* = finite position with confidence >= 0.3 (same number for both backends, whose confidence scales differ). *After smoothing* = finite position after gap filling (gaps up to 0.1 s interpolated). **"Usable" means the model reported a position with confidence, not that the position is correct**: a backend can be confident and wrong, so check with `scripts/compare_frames.py` (see docs/rtmpose.md).

| backend | joint | raw usable | after smoothing |
|---|---|---|---|
| mediapipe | left_wrist | 84.0% | 88.6% |
| mediapipe | right_wrist | 85.3% | 86.0% |
| mediapipe | left_hip | 100.0% | 100.0% |
| mediapipe | right_hip | 100.0% | 100.0% |
| rtmpose | left_wrist | 100.0% | 100.0% |
| rtmpose | right_wrist | 100.0% | 100.0% |
| rtmpose | left_hip | 100.0% | 100.0% |
| rtmpose | right_hip | 100.0% | 100.0% |

## Raw jitter (whole clip)

RMS second difference of the raw position, px/frameÂ². Includes low-confidence frames. Lower = steadier, but a backend can be steady and wrong.

| backend | left_wrist | right_wrist | left_hip | right_hip |
|---|---|---|---|---|
| mediapipe | 11.47 | 9.13 | 3.10 | 2.98 |
| rtmpose | 5.82 | 8.29 | 5.37 | 5.04 |

## Disagreement between mediapipe and rtmpose

Median distance between the two backends' estimates of the same joint, over swing-window frames where both have confidence >= 0.3. In torso lengths (torso = 344 px). A difference, **not** an error: either may be wrong.

| joint | frames compared | median distance (px) | (torso lengths) |
|---|---|---|---|
| left_wrist | 325 | 17.1 | 0.050 |
| right_wrist | 330 | 12.1 | 0.035 |
| left_hip | 387 | 11.8 | 0.034 |
| right_hip | 387 | 10.5 | 0.031 |

## Downstream: phases

Detected frames from each backend's smoothed keypoints. **Only top and impact are scored** against your hand labels (see docs/phase_definitions.md); the others are threshold-defined. Error = detected - label in ms.

| phase | label | mediapipe frame (err ms) | rtmpose frame (err ms) |
|---|---|---|---|
| address | 681 | 786 (+438) * | 780 (+413) * |
| takeaway | 731 | 800 (+288) * | 795 (+267) * |
| top | 922 | 919 (-13) | 918 (-17) |
| impact | 995 | 999 (+17) | 994 (-4) |
| finish | 1117 | 1077 (-167) * | 1068 (-205) * |

\* threshold-defined, not scored.

## Downstream: metrics (dtl)

Same code, different keypoints. Differences here show how much the choice of backend moves each number; neither column is validated.

| metric | mediapipe | rtmpose |
|---|---|---|
| backswing_s | 0.497 | 0.514 |
| downswing_s | 0.334 | 0.317 |
| tempo_ratio | 1.488 | 1.618 |
| spine_angle_address_deg | 35.110 | 36.015 |
| spine_angle_top_deg | 27.610 | 27.538 |
| spine_angle_impact_deg | 29.934 | 33.810 |
| spine_angle_change_to_impact_deg | -5.176 | -2.205 |
| hip_toward_ball_top_torso | 0.085 | 0.086 |
| hip_toward_ball_impact_torso | 0.114 | 0.084 |
