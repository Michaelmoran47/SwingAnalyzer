"""Swing Analyzer demo.  Run from the repo root:  streamlit run app/streamlit_app.py

Two ways in: a clip already analysed on this machine (any backend), or an
uploaded video analysed live with MediaPipe. Everything runs locally; nothing
is sent anywhere. Folders can be changed with SWING_RAW_DIR, SWING_OUT_DIR and
SWING_LABELS_DIR.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from swing.demo import (analyze, apply_ball_cue, ball_cue_cached, find_clips, hand_figure, metric_table,
                        overlay_rgb, phase_table, safe_clip_name, save_upload, slowdown_scan,
                        usable_keypoint_files)
from swing.io import cached_jpeg_frames, load_jpeg_frames, probe_video, store_jpeg_frames
from swing.labels import PHASES, PhaseLabels
from swing.phases import PhaseParams
from swing.pose import J, PoseSequence

RAW = os.environ.get("SWING_RAW_DIR", "data/raw")
OUT = os.environ.get("SWING_OUT_DIR", "outputs")
LABELS = os.environ.get("SWING_LABELS_DIR", "data/labels")
IMG_HEIGHT = 720
# Decoded frames are saved here (safe to delete); the first view of a clip fills it.
CACHE = os.environ.get("SWING_CACHE_DIR", str(Path(OUT) / ".frame_cache"))
CUE_CACHE = os.environ.get("SWING_CUE_CACHE_DIR", str(Path(OUT) / ".cue_cache"))

st.set_page_config(page_title="Swing Analyzer", layout="wide")


@st.cache_data(show_spinner="Reading video timestamps...")
def cached_probe(path: str, mtime: float):
    return probe_video(path)


@st.cache_resource(show_spinner="Loading frames. A clip's first time decodes them and saves a cache "
                                "(about 1.5 minutes for a 1551-frame clip); later launches are quick...")
def cached_frames(path: str, mtime: float, scale: float, persist: bool):
    if persist:  # clips kept on disk: also save the frames for the next launch
        return cached_jpeg_frames(path, scale, CACHE)[0]
    return load_jpeg_frames(path, scale)  # uploads live in a temp folder: no disk cache


@st.cache_data
def cached_sequence(path: str, mtime: float):
    return PoseSequence.load(path)


def _mtime(p) -> float:
    return Path(p).stat().st_mtime


def pick_saved():
    clips = find_clips(RAW, OUT, LABELS)
    if not clips:
        st.info(f"No videos found in `{RAW}`. Put a clip there and analyse it with "
                "`swing overlay <clip> --keypoints outputs/<clip>.npz`, or use the upload option.")
        return None
    clip = st.sidebar.selectbox("Clip", clips, format_func=lambda c: c.video.name)
    info = cached_probe(str(clip.video), _mtime(clip.video))
    usable, skipped = usable_keypoint_files(clip.keypoints, info.frame_count)
    if not usable:
        st.info("No full-clip keypoint file for this clip yet. Create one with "
                f"`swing overlay {clip.video} --keypoints {OUT}/{clip.video.stem}.npz`.")
        for p, why in skipped:
            st.caption(f"skipped `{p.name}`: {why}")
        return None
    kp = st.sidebar.selectbox("Pose backend / keypoints", usable,
                              format_func=lambda k: f"{k.backend}  ({k.path.name})")
    for p, why in skipped:
        st.sidebar.caption(f"skipped `{p.name}`: {why}")
    labels = PhaseLabels.load(clip.labels) if clip.labels else None
    return dict(video=clip.video, info=info, seq=cached_sequence(str(kp.path), _mtime(kp.path)),
                backend=kp.backend, source=kp.path.name, labels=labels)


def pick_upload():
    up = st.sidebar.file_uploader("Video (iPhone slo-mo .MOV works best)", type=["mov", "mp4", "m4v"])
    state = st.session_state.get("upload")
    if up is not None and st.sidebar.button("Analyse with MediaPipe", type="primary"):
        if state:
            shutil.rmtree(state["dir"], ignore_errors=True)
        d = tempfile.mkdtemp(prefix="swing_upload_")
        path = Path(d) / up.name
        path.write_bytes(up.getbuffer())
        try:
            info = probe_video(path)
            from swing.pose import estimate_video, get_estimator

            bar = st.progress(0.0, text="Running MediaPipe pose estimation...")
            with get_estimator("mediapipe") as est:
                seq = estimate_video(path, est, progress=lambda i, n: bar.progress(i / n, text=f"pose {i}/{n}"))
            bar.empty()
            state = dict(dir=d, video=path, info=info, seq=seq, backend="mediapipe",
                         source=f"uploaded: {up.name}", labels=None)
            st.session_state["upload"] = state
        except Exception as e:
            shutil.rmtree(d, ignore_errors=True)
            st.error(f"Could not analyse that video: {e}")
            return None
    if not state:
        st.info("Upload a clip and press **Analyse**. The video is processed on this machine and "
                "kept only in a temporary folder for this session.")
        return None
    return state


def save_panel(data, jpegs, scale):
    """Let the user keep an analysed upload so it appears under "Saved analysis"."""
    st.sidebar.divider()
    st.sidebar.markdown("**Keep this analysis**")
    if data.get("saved_as"):
        st.sidebar.success(f"Saved as `{data['saved_as']}`. Choose *Saved analysis* above to open it any time.")
        return
    name = st.sidebar.text_input("Save as", value=safe_clip_name(Path(data["video"]).name),
                                 key=f"save_name_{data['dir']}")
    st.sidebar.caption(f"Copies the video to `{RAW}/` and the keypoints to `{OUT}/` on this machine. "
                       "Nothing is overwritten.")
    if st.sidebar.button("Save clip and keypoints"):
        try:
            video, _ = save_upload(data["video"], data["seq"], name, RAW, OUT)
            store_jpeg_frames(video, scale, CACHE, jpegs)  # frames are already decoded: no second pass
        except (FileExistsError, ValueError, OSError) as e:
            st.sidebar.error(str(e))
            return
        data["saved_as"] = video.stem
        st.rerun()


# ---------------------------------------------------------------- page

st.title("Swing Analyzer")
st.warning(
    "**Unvalidated demo.** Built and checked on one swing. Everything here is 2D image-plane "
    "geometry from pose keypoints, not a coaching verdict. Only *top* and *impact* are compared "
    "with hand labels; address, takeaway and finish follow threshold rules; metrics have no "
    "reference measurement. Pose confidence is not a reliable error flag."
)

mode = st.sidebar.radio("Source", ["Saved analysis", "Upload a video"])
view = st.sidebar.radio("Camera view of this clip", ["dtl", "fo"],
                        format_func=lambda v: {"dtl": "Down-the-line", "fo": "Face-on"}[v],
                        help="Decides which metrics are meaningful.")
window_s = st.sidebar.slider("Smoothing window (s)", 0.02, 0.15, 0.06, 0.01,
                             help="Savitzky-Golay, zero-phase. Default 0.06.")
min_conf = st.sidebar.slider("Minimum keypoint confidence", 0.0, 0.9, 0.3, 0.05,
                             help="Below this a joint counts as missing.")
use_ball = st.sidebar.checkbox(
    "Find impact from the ball", True,
    help="Looks in the video for the ball on the tee and takes impact as the first frame it departs. "
         "Wrist tracking alone can be several frames off. Needs a visible ball and a fairly steady camera; "
         "if no ball is found, impact stays hand-based and the page says why. Reads the video the first "
         "time (tens of seconds); the answer is remembered.")

data = pick_saved() if mode == "Saved analysis" else pick_upload()
if data is None:
    st.stop()

info, seq = data["info"], data["seq"]
for w in info.warnings:
    st.caption(f"video: {w}")

stamped = float(seq.fps)
FPS_CHOICES = {"stamped": f"As stamped in the file ({stamped:.1f} fps)"}
for n_fps in (60, 120, 240, 480, 960):  # real capture rates the file could have been slowed from
    if n_fps > stamped * 1.2:
        FPS_CHOICES[str(n_fps)] = f"{n_fps} fps capture (file slowed {n_fps / stamped:.1f}x)"
fps_choice = st.sidebar.selectbox(
    "Capture frame rate", list(FPS_CHOICES), format_func=FPS_CHOICES.get, key=f"fps_{data['source']}",
    help="Phase detection and metrics need the real capture rate. Leave this on the file's rate "
         "unless the file was re-timed (a swing that takes seconds instead of about a second).")
capture_fps = None if fps_choice == "stamped" else float(fps_choice)
a = analyze(seq, view, window_s, min_conf, data["labels"], capture_fps)
fps_used = a.smooth.fps
if use_ball and a.phases.plausible and all(
        a.phases.frames.get(p) is not None for p in ("address", "takeaway", "top", "impact")):
    with st.spinner("Looking for the ball in the video (first time only; the answer is remembered)..."):
        a = apply_ball_cue(a, ball_cue_cached(data["video"], a, CUE_CACHE))
MIN_FPS = PhaseParams().min_fps
if fps_used < MIN_FPS:
    st.error(f"This clip's timeline is {fps_used:.1f} fps, below the {MIN_FPS:.0f} fps the phase rules "
             "have been checked down to, so phases and metrics are **not computed**. " + a.phases.warnings[0])
elif not a.phases.plausible:
    st.warning("**The phase timings look wrong.** " + a.phases.warnings[-1])
if capture_fps is not None:
    st.info(f"Timings assume this clip was captured at {capture_fps:.0f} fps; the file says "
            f"{stamped:.1f}. If that is not true for this clip, every time, tempo and phase here is wrong.")
elif fps_used < 100 and a.phases.plausible:
    st.caption(f"Frame rate {fps_used:.0f} fps: one frame is {1000 / fps_used:.0f} ms, so phase times "
               "are only good to about one frame. Pose estimation on blurred frames is less reliable.")
if capture_fps is None and stamped < 100:
    with st.expander("Was this clip slowed down? What changes if it was", expanded=not a.phases.plausible):
        st.markdown(
            "A slow-motion clip keeps plenty of frames per swing, but when it has been re-timed to play "
            "slowly the file only stores the playback rate, so **how much it was slowed is unknown**. "
            "Each row re-runs the analysis as if the clip had been slowed by that factor. Rows marked "
            "*no* cannot be a real-time swing. The frames and the tempo ratio usually move little across "
            "the rows marked *yes*; the seconds scale directly with the assumption. This does not say "
            "which row is true: pick the capture rate your camera recorded at in the sidebar.")
        st.dataframe(slowdown_scan(seq, view, window_s, min_conf), hide_index=True, use_container_width=True)
scale = min(1.0, IMG_HEIGHT / info.height)
jpegs = cached_frames(str(data["video"]), _mtime(data["video"]), scale, mode == "Saved analysis")

if mode == "Upload a video":
    save_panel(data, jpegs, scale)

st.caption(f"{data['source']} | backend: **{data['backend']}** | {info.width}x{info.height}, "
           f"{fps_used:.1f} fps{' (set by you)' if a.stamped_fps else ''}, {info.frame_count} frames | "
           f"view: **{view}**")

tab_frames, tab_phases, tab_metrics, tab_traj, tab_about = st.tabs(
    ["Frames", "Phases", "Metrics", "Hand trajectory", "About"])

with tab_frames:
    n = len(jpegs)
    start = a.phases.frames.get("address") or 0
    if "frame" not in st.session_state or st.session_state.get("_clip") != data["source"]:
        st.session_state["frame"] = int(min(start, n - 1))
        st.session_state["_clip"] = data["source"]

    def _jump():
        f = a.phases.frames.get(st.session_state["_jump"])
        if f is not None:
            st.session_state["frame"] = int(min(f, n - 1))

    c1, c2 = st.columns([1, 2])
    with c2:
        st.selectbox("Jump to detected phase", [p for p in PHASES if a.phases.frames.get(p) is not None],
                     key="_jump", on_change=_jump, index=None, placeholder="choose a phase")
        st.slider("Frame", 0, n - 1, key="frame")
        show = st.checkbox("Show skeleton", True)
        use_raw = st.checkbox("Raw keypoints (not smoothed)", False)
    idx = st.session_state["frame"]
    src = a.raw if use_raw else a.smooth
    with c1:
        pose = src.keypoints[idx]
        filled = None if (use_raw or src.filled is None) else src.filled[idx]
        if show:
            st.image(overlay_rgb(jpegs[idx], pose, scale, filled, min_conf), use_container_width=True)
        else:
            st.image(jpegs[idx], use_container_width=True)
    with c2:
        near = [p for p in PHASES if a.phases.frames.get(p) is not None and abs(a.phases.frames[p] - idx) <= 2]
        st.markdown(f"**Frame {idx}**, t = {idx / fps_used:.3f} s" + (f" — detected **{near[0]}**" if near else ""))
        hl = a.labels.phase_at(idx) if a.labels else None
        if hl:
            st.markdown(f"Your hand label here: **{hl}**")
        conf = {k: float(pose[J[k], 2]) for k in ("left_wrist", "right_wrist", "left_hip", "right_hip")}
        st.caption("Keypoint confidence: " + ", ".join(f"{k.replace('_', ' ')} {v:.2f}" for k, v in conf.items())
                   + (" | interpolated joints are drawn when smoothing filled a gap" if not use_raw else ""))
        st.caption("Orange = lead (left) side, cyan = trail (right) side.")

with tab_phases:
    st.dataframe(phase_table(a), hide_index=True, use_container_width=True)
    if use_ball:
        (st.success if a.impact_from_ball else st.caption)(f"Impact: {a.impact_note}.")
    st.markdown(
        "**top** = highest hands before peak hand speed; **impact** = the first frame the ball leaves the tee "
        "when a ball is found in the picture, otherwise the lowest hands shortly after the peak speed. "
        "**address**, **takeaway** and **finish** are set by speed/displacement thresholds on the wrist "
        "trajectory (see `docs/phase_definitions.md`), so they can differ from what a person would pick "
        "by 0.1 to 0.4 s on the one labelled swing. Phases the rules cannot find are blank, not guessed."
    )
    for w in a.phases.warnings:
        st.caption(f"note: {w}")
    if a.labels is None:
        st.caption("No hand labels found for this clip, so no error column.")

with tab_metrics:
    rows = metric_table(a)
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.markdown(
        "Distances are in **torso lengths** (2D, at address). Metrics that cannot be measured from this camera "
        "view are listed as unavailable with the reason. There are no pass/fail thresholds: values are "
        "measurements. They have not been checked against any reference measurement. See `docs/metrics.md`."
    )

with tab_traj:
    # Streamlit runs every tab's code on each rerun; drawing takes ~0.5 s, so draw only on change.
    fig_key = (data["source"], id(seq), view, window_s, min_conf, data["labels"] is not None, fps_used,
               a.phases.frames.get("impact"))
    if st.session_state.get("_fig_key") != fig_key:
        st.session_state["_fig"], st.session_state["_fig_key"] = hand_figure(a), fig_key
    st.pyplot(st.session_state["_fig"])
    st.caption("Mean of the available wrists, after smoothing. Solid lines mark the hand-validated phases "
               "(top, impact); dashed lines mark threshold-defined ones.")

with tab_about:
    st.markdown(
        f"""
**Pipeline:** video → pose keypoints (`{data['backend']}`) → gap filling and zero-phase smoothing →
rule-based phase detection from the wrists → view-aware metrics.

**What is and is not checked** (details in the repo's `docs/`):
- One swing, one camera angle, one annotator. Nothing here is an accuracy result.
- Top and impact came within about 17 ms of hand labels on that swing.
- Address, takeaway and finish are threshold-defined and were not scored.
- Metrics are unit-tested on synthetic skeletons, not validated against measurements.
- Rotation values are 2D proxies, not angles.

**Privacy:** the video and keypoints stay on this machine.
"""
    )
