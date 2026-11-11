# SwingAnalyzer

Golf swing analysis from phone video: pose estimation → smoothing → phase
detection → metrics, as a core Python library with a CLI (Streamlit demo planned).

**Status: milestones 0-5.** Ingest, skeleton overlay,
smoothing, rule-based phase detection and view-aware metrics work and are
unit-tested. Validation so far is one hand-labelled swing, so nothing here is an
accuracy result. Both pose backends (MediaPipe, RTMPose) are implemented and
compared on that clip ([docs/rtmpose.md](docs/rtmpose.md)). The Streamlit demo and
multi-clip validation are not built yet. No results are reported in this README: every
number that appears here will come from a script in this repo.

## Scope

- Capture: iPhone slo-mo (120–240 fps), down-the-line and face-on, right-handed
  golfer. See [docs/capture_protocol.md](docs/capture_protocol.md).
- Pose backends behind a `PoseEstimator` interface: MediaPipe BlazePose (default,
  implemented); RTMPose via rtmlib/ONNX (implemented, benchmarked on one clip).
- Planned metrics: tempo, spine angle, head sway, hip slide, early extension, and
  rotation as a labelled 2D proxy (not a true angle).
- Metrics will be pure functions of keypoints, unit-tested on synthetic skeletons.

## Install

Python 3.10+.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -e ".[mediapipe,rtmpose,dev]"
```

MediaPipe pinned to 0.10.21 for the legacy `mp.solutions.pose` API. The default
heavy model (complexity 2) is downloaded by MediaPipe on first use.

## Usage

```bash
swing info data/raw/2026-11-11_7i_dtl_01.MOV
swing overlay data/raw/2026-11-11_7i_dtl_01.MOV -o outputs/overlay.mp4 --keypoints outputs/kps.npz
swing label data/raw/2026-11-11_7i_dtl_01.MOV   # hand-label phases, see docs/labelling.md
```

The overlay keeps the source frame rate, so slo-mo stays slo-mo. Lead (left) side
is drawn orange, trail (right) side cyan. Low-confidence joints are hidden.

## Layout

```
src/swing/
  io/         video probing and frame iteration
  pose/       PoseEstimator interface, COCO-17 schema, MediaPipe backend
  render/     skeleton overlay
  smoothing/  gap filling and Savitzky-Golay smoothing
  phases/     rule-based phase detection and scoring against labels
  metrics/    tempo, spine angle, head sway, hip slide, hip-toward-ball, rotation proxies
  compare/    (later)
tests/        pytest; synthetic videos and skeletons, no real footage needed
data/         raw/ is gitignored; labels/ for hand-labelled phase frames
docs/
```

## Tests

```bash
pytest
```

## Roadmap

0. Scaffold and capture protocol ✅
1. Ingest + MediaPipe overlay ✅ (tried on one real clip)
2. Smoothing ✅ (`--smooth`; effect measured by `scripts/smoothing_report.py`)
3. Rule-based phase detection ✅ implemented (`swing phases`). Top and impact are scored against hand labels; address, takeaway and finish are threshold-defined, see [docs/phase_definitions.md](docs/phase_definitions.md). Validation so far: one labelled swing, so a sanity check, not an accuracy result
4. Metrics v1 ✅ implemented and unit-tested, view-aware (`swing metrics`), see [docs/metrics.md](docs/metrics.md). Not validated against reference measurements
5. RTMPose backend and benchmark ✅ implemented; compared on one clip, see [docs/rtmpose.md](docs/rtmpose.md) and [docs/benchmark.md](docs/benchmark.md). Not an accuracy ranking
6. Streamlit demo
7. Stretch: SwingNet phase model, 3D lifting
