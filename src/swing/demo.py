"""Logic behind the Streamlit demo, kept free of Streamlit so it can be tested.

The demo shows an analysis of one swing. Every phase and metric is tagged with
how much it can be trusted (see docs/phase_definitions.md and docs/metrics.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from swing.labels import PHASES, PhaseLabels
from swing.metrics import MetricReport, compute_metrics
from swing.phases import PhaseResult, compare, detect_phases
from swing.phases.detect import hand_signal
from swing.phases.evaluate import VALIDATED_PHASES
from swing.pose import PoseSequence
from swing.render import draw_skeleton
from swing.smoothing import smooth_sequence

VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v"}

# metric -> (unit, what it is). Wording matches docs/metrics.md.
METRIC_INFO: dict[str, tuple[str, str]] = {
    "backswing_s": ("s", "takeaway to top; takeaway is threshold-defined"),
    "downswing_s": ("s", "top to impact"),
    "tempo_ratio": ("ratio", "backswing / downswing; depends on the takeaway rule"),
    "spine_angle_address_deg": ("deg", "spine from vertical in the image plane (forward bend from down-the-line, lateral tilt from face-on)"),
    "spine_angle_top_deg": ("deg", "spine from vertical at the top"),
    "spine_angle_impact_deg": ("deg", "spine from vertical at impact"),
    "spine_angle_change_to_impact_deg": ("deg", "impact minus address"),
    "hip_toward_ball_top_torso": ("torso lengths", "hip movement toward the ball from address; positive = toward the ball"),
    "hip_toward_ball_impact_torso": ("torso lengths", "indicator related to early extension, not a detector of it"),
    "hip_slide_top_torso": ("torso lengths", "hip movement toward the target; positive = toward the target"),
    "hip_slide_impact_torso": ("torso lengths", "hip movement toward the target at impact"),
    "head_sway_top_torso": ("torso lengths", "ear-midpoint movement toward the target"),
    "head_sway_impact_torso": ("torso lengths", "ear-midpoint movement toward the target at impact"),
    "head_sway_max_excursion_torso": ("torso lengths", "largest head movement from takeaway to impact"),
    "shoulder_turn_proxy_top_deg": ("deg, 2D proxy", "arccos of shoulder-width ratio; not a rotation angle"),
    "hip_turn_proxy_top_deg": ("deg, 2D proxy", "arccos of hip-width ratio; not a rotation angle"),
}


@dataclass
class Analysis:
    raw: PoseSequence
    smooth: PoseSequence
    phases: PhaseResult
    metrics: MetricReport
    view: str
    labels: PhaseLabels | None = None


def analyze(seq: PoseSequence, view: str, window_s: float = 0.06, min_conf: float = 0.3,
            labels: PhaseLabels | None = None) -> Analysis:
    smooth = smooth_sequence(seq, min_conf, window_s=window_s)
    phases = detect_phases(smooth)
    return Analysis(seq, smooth, phases, compute_metrics(smooth, phases, view), view, labels)


def phase_table(a: Analysis) -> list[dict]:
    errs = {e.phase: e for e in compare(a.phases, a.labels)} if a.labels else {}
    rows = []
    for p in PHASES:
        f = a.phases.frames.get(p)
        e = errs.get(p)
        rows.append({
            "phase": p,
            "frame": f,
            "time (s)": None if f is None else round(f / a.smooth.fps, 3),
            "status": "scored against your hand labels (one swing)" if p in VALIDATED_PHASES
                      else "threshold-defined (not scored)",
            "your label": None if e is None else e.labelled,
            "error (ms)": None if e is None or e.error_ms is None else round(e.error_ms, 1),
        })
    return rows


def metric_table(a: Analysis) -> list[dict]:
    rows = []
    for name, v in a.metrics.values.items():
        unit, note = METRIC_INFO.get(name, ("", ""))
        rows.append({
            "metric": name,
            "value": None if v is None else round(v, 3),
            "unit": unit,
            "what it is": note if v is not None else f"unavailable: {a.metrics.unavailable.get(name, '')}",
        })
    return rows


def hand_figure(a: Analysis):
    """Hand height and speed over time with the detected phases marked."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure

    s = a.smooth
    hands = hand_signal(s)
    t = (np.arange(len(s)) + s.start_frame) / s.fps
    speed = np.full(len(s), np.nan)
    speed[1:-1] = np.linalg.norm(hands[2:] - hands[:-2], axis=1) * s.fps / 2
    fig = Figure(figsize=(9, 4.2), layout="constrained")
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.plot(t, -hands[:, 1], color="#d95f02", lw=1.5)
    ax1.set_ylabel("hand height (px, up = higher)")
    ax2.plot(t, speed, color="#1b6ca8", lw=1.2)
    ax2.set_ylabel("hand speed (px/s)")
    ax2.set_xlabel("time (s)")
    for p in PHASES:
        f = a.phases.frames.get(p)
        if f is None:
            continue
        for ax in (ax1, ax2):
            ax.axvline(f / s.fps, color="#888", lw=0.8, ls="--" if p not in VALIDATED_PHASES else "-")
        ax1.text(f / s.fps, ax1.get_ylim()[1], p, rotation=90, va="top", ha="right", fontsize=8)
    return fig


def overlay_rgb(jpeg: bytes, pose: np.ndarray, scale: float, filled: np.ndarray | None = None,
                min_conf: float = 0.3) -> np.ndarray:
    """Decode a stored frame and draw ``(17, 3)`` full-resolution keypoints scaled by ``scale``."""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    p = pose.copy()
    p[:, :2] *= scale
    draw_skeleton(img, p, min_conf, filled)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@dataclass(frozen=True)
class ClipEntry:
    video: Path
    keypoints: tuple[Path, ...]
    labels: Path | None


def find_clips(raw_dir: str | Path, out_dir: str | Path, labels_dir: str | Path) -> list[ClipEntry]:
    """Videos in ``raw_dir`` with the keypoint files in ``out_dir`` that start with the stem."""
    raw, out, lab = Path(raw_dir), Path(out_dir), Path(labels_dir)
    entries = []
    for v in sorted(raw.glob("*")) if raw.is_dir() else []:
        if v.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        kps = tuple(sorted(out.glob(f"{v.stem}*.npz"))) if out.is_dir() else ()
        label = lab / f"{v.stem}.json"
        entries.append(ClipEntry(v, kps, label if label.exists() else None))
    return entries


@dataclass(frozen=True)
class KeypointFile:
    path: Path
    backend: str
    start_frame: int
    n_frames: int


def describe_keypoints(path: str | Path) -> KeypointFile:
    """Read only the header fields of a keypoint ``.npz`` (cheap)."""
    with np.load(path, allow_pickle=False) as d:
        return KeypointFile(Path(path), str(d["backend"]), int(d["start_frame"]), int(d["keypoints"].shape[0]))


def usable_keypoint_files(paths, frame_count: int) -> tuple[list[KeypointFile], list[tuple[Path, str]]]:
    """Split keypoint files into ones the demo can analyse and ones it skips (with reasons).

    Usable = raw keypoints covering the whole clip. Skipped: files whose name says
    they are already smoothed (smoothing twice changes the result), partial files
    (a window, not the whole swing) and files for a different clip length.
    """
    ok, skipped = [], []
    for p in paths:
        p = Path(p)
        if "smooth" in p.stem.lower():
            skipped.append((p, "already smoothed; the demo smooths raw keypoints itself"))
            continue
        try:
            k = describe_keypoints(p)
        except Exception as e:  # unreadable or not a keypoint file
            skipped.append((p, f"unreadable: {e}"))
            continue
        if k.start_frame != 0 or k.n_frames != frame_count:
            skipped.append((p, f"covers frames {k.start_frame}-{k.start_frame + k.n_frames - 1}, "
                               f"not the whole clip (0-{frame_count - 1})"))
        else:
            ok.append(k)
    return ok, skipped
