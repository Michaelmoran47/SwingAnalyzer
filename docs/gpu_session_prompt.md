# Prompt for a Claude Code session on the GPU computer

Paste everything below the line into a new session opened in the cloned repo.

---

You are continuing work on SwingAnalyzer, a golf swing analysis portfolio
project (Python, OpenCV, pose estimation) that I built with Claude Code on my
laptop. This computer has an NVIDIA GPU; the laptop does not. Read README.md,
docs/rtmpose.md, docs/benchmark.md, docs/phase_definitions.md and
docs/metrics.md first. They record what exists, what was measured, and what is
unvalidated.

## Project in one paragraph
Phone slo-mo video (240 fps, right-handed golfer) goes through pose estimation
(PoseEstimator interface: MediaPipe default, RTMPose via rtmlib/ONNX), then
smoothing, rule-based phase detection from the wrist trajectory, and view-aware
metrics. Everything is a library plus a `swing` CLI. Milestones 0-5 are done on
one clip, `data/raw/IMG_6004.MOV` (down-the-line, 1551 decoded frames,
~239.5 fps), hand-labelled in `data/labels/IMG_6004.json`.

## Rules I want you to keep
- **Never invent metrics or results.** Every number in the README or docs must
  come from a script in this repo. Say plainly what is not validated.
- One labelled swing is a sanity check, not an accuracy result. Do not tune
  thresholds against this one clip. Only top and impact are scored against hand
  labels; address/takeaway/finish are threshold-defined (docs/phase_definitions.md).
- Do not edit `data/labels/*.json` (my ground truth).
- There are no ground-truth keypoints, so compare backends on behaviour
  (speed, completeness, jitter, disagreement, downstream effect), never "accuracy".
- Pose confidence is not reliable: RTMPose reported 0.57-0.77 on wrists that
  were visibly wrong. Check by eye with `scripts/compare_frames.py`.
- Run tests (`pytest`) before committing. Commit small, push to `main`.

> **Correction (added after the GPU session):** the wrist findings in this section were
> read off images that were about 50 frames later than their labels, because
> `iter_frames(start>0)` seeked wrongly on this file. They are withdrawn; see
> "Correction: frame-seek bug" in docs/rtmpose.md.

## What was found on the laptop (CPU only)
- Balanced-mode RTMPose and MediaPipe both put the wrists on the torso/hip in
  frames ~1000-1030 (post-impact motion blur); hands are really high near the club.
- RTMPose `--rtm-mode performance` put the wrists on the hands in six checked
  frames (990-1045), but is ~1.85 s/frame on CPU (about 48 min for the clip).
- The balanced-vs-MediaPipe comparison exposed and fixed a bug in the address rule.

## Your tasks (in order)
1. Set up: `python -m venv .venv`, activate, then
   `pip install -e ".[mediapipe,dev]" rtmlib onnxruntime-gpu`
   (use onnxruntime-gpu, not the `rtmpose` extra; the two onnxruntime packages
   conflict). Verify `CUDAExecutionProvider` appears in
   `onnxruntime.get_available_providers()`. If not, diagnose (driver, CUDA 12,
   cuDNN 9) and tell me what to install. Do not silently use CPU: the backend
   now raises an error if `--device cuda` is unavailable.
2. `data/raw/IMG_6004.MOV` is gitignored. Check it exists; if not, ask me to
   copy it. Do not download substitute video.
3. Produce keypoints (outputs/ is gitignored):
   - `swing overlay data/raw/IMG_6004.MOV --keypoints outputs/IMG_6004.npz -o outputs/mp.mp4`
   - `swing overlay data/raw/IMG_6004.MOV --backend rtmpose --device cuda --rtm-mode performance --keypoints outputs/IMG_6004_rtmpose_perf.npz -o outputs/perf.mp4`
   The first RTMPose run downloads ~550 MB of models from download.openmmlab.com
   (OpenMMLab's official host); that is allowed.
4. Benchmark with GPU timing, output to a new file so the CPU report stays:
   `python scripts/benchmark_backends.py --npz mediapipe=outputs/IMG_6004.npz rtmpose=outputs/IMG_6004_rtmpose_perf.npz --labels data/labels/IMG_6004.json --view dtl --video data/raw/IMG_6004.MOV --time-frames 30 --device cuda --rtm-mode performance --out docs/benchmark_gpu.md`
   Add the GPU model, driver, CUDA/cuDNN and onnxruntime-gpu versions to the
   report by hand (the script only prints the CPU string).
5. Whole-clip performance-mode results: run `swing phases` (with
   `--labels data/labels/IMG_6004.json`) and `swing metrics --view dtl` on
   outputs/IMG_6004_rtmpose_perf.npz. Compare against the MediaPipe and
   balanced numbers in docs/benchmark.md. Run
   `scripts/compare_frames.py` on frames 985-1045 to confirm by eye that wrists
   stay on the hands over the whole follow-through, not just six frames.
6. Update docs/rtmpose.md with the findings (clearly labelled by machine), commit
   docs/benchmark_gpu.md and the doc update, run `pytest`, push.

## When done, report
GPU speed (s/frame, fps) for MediaPipe and RTMPose performance mode; whether
wrists stay on the hands post-impact; how top/impact errors and metrics change;
and anything surprising. If something is ambiguous, ask me instead of guessing.
