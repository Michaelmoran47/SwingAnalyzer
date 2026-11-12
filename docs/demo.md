# Streamlit demo

A local web page for looking at one swing: skeleton overlay frame by frame, detected
phases, metrics, and the hand trajectory. Code: `app/streamlit_app.py` (UI) and
`src/swing/demo.py` (analysis, unit-tested without Streamlit).

```bash
pip install -e ".[mediapipe,demo]"
streamlit run app/streamlit_app.py
```

Run it from the repo root. The page opens at `http://127.0.0.1:8501`.

On Windows, `scripts/run_demo.bat` does the same and opens the browser for you; close its
window to stop the demo. To make a desktop shortcut to it, right-click the file, choose
Show more options, then Send to, then Desktop (create shortcut).

## Two ways in

- **Saved analysis.** Lists videos in `data/raw/` and, for each, the keypoint files in
  `outputs/` whose name starts with the clip name (made with
  `swing overlay <clip> --keypoints outputs/<clip>.npz`, any backend). Only raw keypoints
  covering the whole clip are offered; already-smoothed and partial files are skipped
  with the reason shown. Hand labels from `data/labels/<clip>.json` are used if present.
- **Upload a video.** Runs MediaPipe on the uploaded file, on this machine (about 15
  frames per second on a laptop CPU; a 240 fps clip takes a couple of minutes). RTMPose
  is not offered here: use the CLI, then pick the saved analysis. By default nothing is
  kept: the video sits in a temporary folder and the keypoints live in the page session,
  so they are lost when you restart the server.
  - **Keep this analysis.** After an upload is analysed, the sidebar offers a name and a
    **Save clip and keypoints** button. It copies the video to `data/raw/` and the raw
    keypoints to `outputs/` (and stores the decoded frames in the cache, so no second
    decode), after which the clip appears under *Saved analysis* and opens quickly.
    Nothing is saved unless you press it. It never overwrites: if the name is taken
    (any video extension, or an existing keypoint file) it says so and writes nothing.
    The name is reduced to letters, digits, dot, dash and underscore.

Keypoint files are matched to videos by name: `<clip>.npz` or `<clip>_<anything>.npz`.
When one name is a prefix of another (`swing` and `swing_2`), each file goes to the clip
with the longest matching name.

The first view of a clip decodes every frame once and saves the shrunken frames to
`outputs/.frame_cache/` (measured on `IMG_6004`, 1551 frames of 1080x1920: 46.5 s the
first time, 0.2 s from the cache, 70 MB on disk). After that, launches open in a few
seconds (reading the video's frame timestamps still takes a moment). The cache is keyed
on the file's path, size and modification time, so replacing the video rebuilds it; a
damaged cache file is rebuilt automatically. It is safe to delete the folder at any time.

## Capture frame rate

The sidebar has a **Capture frame rate** control, by default the rate stamped in the file.
Below 100 fps the page shows an error and computes no phases or metrics (see
[capture_protocol.md](capture_protocol.md#clips-that-lose-their-real-timeline)). If you
know the clip was recorded in slo-mo and re-timed to play slowly, choose 240 or 120; the
page then says that all timings assume that rate and that this is your claim. The chosen
rate is also used for the times shown on the Frames tab and the plot.

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
  `SWING_OUT_DIR`, `SWING_LABELS_DIR` and `SWING_CACHE_DIR`.
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
files, and shows guidance when no clip exists. Saving an upload is tested end to end (session state is
set directly, since the test harness cannot upload a file). The upload widget and the live
MediaPipe run inside the page are not covered by a test. Checked by hand in a browser on `IMG_6004`: frame
jump, overlay, and all tabs.
