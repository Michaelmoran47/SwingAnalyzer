# Backend benchmark: IMG_6004.MOV (dtl)

Generated 2026-11-11 by `scripts/benchmark_backends.py`. One clip, one swing: a comparison on this footage, **not** an accuracy ranking (no ground-truth keypoints exist).

Swing window = your hand labels takeaway..finish, frames 731-1117. Machine: Intel64 Family 6 Model 140 Stepping 1, GenuineIntel, Windows 10.

## Speed (this machine)

ONNX Runtime providers available: AzureExecutionProvider, CPUExecutionProvider; rtmpose device = `cpu`. Per-frame time at full video resolution IMG_6004.MOV, 8 frames after warm-up, starting mid-swing.

| backend | s/frame | fps |
|---|---|---|
| mediapipe | 0.062 | 16.1 |
| rtmpose | 0.412 | 2.4 |

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

RMS second difference of the raw position, px/frame². Includes low-confidence frames. Lower = steadier, but a backend can be steady and wrong.

| backend | left_wrist | right_wrist | left_hip | right_hip |
|---|---|---|---|---|
| mediapipe | 11.47 | 9.13 | 3.10 | 2.98 |
| rtmpose | 6.27 | 7.09 | 5.00 | 4.74 |

## Disagreement between mediapipe and rtmpose

Median distance between the two backends' estimates of the same joint, over swing-window frames where both have confidence >= 0.3. In torso lengths (torso = 344 px). A difference, **not** an error: either may be wrong.

| joint | frames compared | median distance (px) | (torso lengths) |
|---|---|---|---|
| left_wrist | 325 | 16.8 | 0.049 |
| right_wrist | 330 | 12.5 | 0.036 |
| left_hip | 387 | 13.6 | 0.039 |
| right_hip | 387 | 13.4 | 0.039 |

## Downstream: phases

Detected frames from each backend's smoothed keypoints. **Only top and impact are scored** against your hand labels (see docs/phase_definitions.md); the others are threshold-defined. Error = detected - label in ms.

| phase | label | mediapipe frame (err ms) | rtmpose frame (err ms) |
|---|---|---|---|
| address | 681 | 786 (+438) * | 774 (+388) * |
| takeaway | 731 | 800 (+288) * | 793 (+259) * |
| top | 922 | 919 (-13) | 919 (-13) |
| impact | 995 | 999 (+17) | 996 (+4) |
| finish | 1117 | 1077 (-167) * | 1102 (-63) * |

\* threshold-defined, not scored.

## Downstream: metrics (dtl)

Same code, different keypoints. Differences here show how much the choice of backend moves each number; neither column is validated.

| metric | mediapipe | rtmpose |
|---|---|---|
| backswing_s | 0.497 | 0.526 |
| downswing_s | 0.334 | 0.322 |
| tempo_ratio | 1.488 | 1.636 |
| spine_angle_address_deg | 35.110 | 36.367 |
| spine_angle_top_deg | 27.610 | 28.875 |
| spine_angle_impact_deg | 29.934 | 32.868 |
| spine_angle_change_to_impact_deg | -5.176 | -3.499 |
| hip_toward_ball_top_torso | 0.085 | 0.098 |
| hip_toward_ball_impact_torso | 0.114 | 0.103 |
