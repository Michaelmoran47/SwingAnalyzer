# Streamlit demo

A local web page for looking at one swing: skeleton overlay frame by frame, detected
phases, metrics, and the hand trajectory. Code: `app/streamlit_app.py` (UI) and
`src/swing/demo.py` (analysis, unit-tested without Streamlit).

```bash
pip install -e ".[mediapipe,demo]"
streamlit run app/streamlit_app.py
```

Run it from the repo root. The page opens at `http://127.0.0.1:8501`.

## Two ways in

- **Saved analysis.** Lists videos in `data/raw/` and, for each, the keypoint files in
  `outputs/` whose name starts with the clip name (made with
  `swing overlay <clip> --keypoints outputs/<clip>.npz`, any backend). Only raw keypoints
  covering the whole clip are offered; already-smoothed and partial files are skipped
  with the reason shown. Hand labels from `data/labels/<clip>.json` are used if present.
- **Upload a video.** Runs MediaPipe on the uploaded file, on this machine (about 15
  frames per second on a laptop CPU; a 240 fps clip takes a couple of minutes). The file
  is kept only in a temporary folder for the session. RTMPose is not offered here: use
  the CLI, then pick the saved analysis.

The first view of a clip decodes every frame once (about 1.5 minutes for a 1551-frame
1080x1920 clip on a laptop); later views are instant.

## What the page says about trust

The banner and tables repeat what the docs say: one swing, 2D geometry, only top and
impact compared with hand labels, address/takeaway/finish threshold-defined, metrics with
no reference measurement, confidence not an error flag. Metrics that cannot be measured
from the chosen camera view are listed as unavailable with the reason. You choose the
view (down-the-line or face-on); the app does not guess it.

## Privacy and configuration

- `.streamlit/config.toml` makes the server listen on `127.0.0.1` only and turns off
  usage statistics. Streamlit's default is to listen on every network interface, which
  would serve your swing frames to anyone on the network. Do not change this without a
  reason.
- Nothing is uploaded anywhere. Folders can be changed with `SWING_RAW_DIR`,
  `SWING_OUT_DIR` and `SWING_LABELS_DIR`.
- The raw video and keypoints are gitignored, so there is no public hosted demo.
  Publishing one needs a clip you are happy to share.

## Dependency note

Streamlit is pinned to 1.40.2 and protobuf to below 5. Newer Streamlit releases need
protobuf 5.26 or later, and MediaPipe 0.10.21 needs protobuf below 5; installing the
newest Streamlit alongside MediaPipe makes MediaPipe fail with
`'FieldDescriptor' object has no attribute 'label'`. The pins keep both working
(`pip check` is clean).

## Tests

`tests/test_demo.py` covers the analysis layer and runs the real app headlessly
(Streamlit's `AppTest`) against a synthetic clip: it renders, shows the caution banner,
lists the right phase frames, switches metrics with the camera view, skips stale keypoint
files, and shows guidance when no clip exists. The live-upload path (MediaPipe on an
uploaded file) is not covered by a test. Checked by hand in a browser on `IMG_6004`: frame
jump, overlay, and all tabs.
