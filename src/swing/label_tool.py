"""Frame-stepping window for hand-labelling swing phases (OpenCV GUI)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from swing.io import load_jpeg_frames, probe_video
from swing.labels import PHASES, PhaseLabels, default_label_path
from swing.pose import PoseSequence
from swing.render import draw_skeleton

# cv2.waitKeyEx codes for arrow / page keys (Windows; Linux/mac codes also listed).
_LEFT, _RIGHT = {2424832, 65361, 63234}, {2555904, 65363, 63235}
_UP, _DOWN = {2490368, 65362, 63232}, {2621440, 65364, 63233}
_PGUP, _PGDN = {2162688, 65365}, {2228224, 65366}

HELP = [
    "left/right or a/d : +-1 frame",
    "up/down or w/s    : +-10 frames",
    "PgUp/PgDn or q/e  : +-50 frames",
    "1-5 set label: " + " ".join(f"{i + 1}={p}" for i, p in enumerate(PHASES)),
    "backspace: clear label here | u: undo",
    "k: toggle skeleton | h: toggle help | Esc: save & quit",
]


def _text(img, s, org, color=(255, 255, 255), size=0.6):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def run(video: str | Path, label_path: str | Path | None = None,
        keypoints: str | Path | None = None, max_height: int = 900) -> PhaseLabels:
    video = Path(video)
    info = probe_video(video)
    label_path = Path(label_path) if label_path else default_label_path(video)
    labels = (PhaseLabels.load(label_path) if label_path.exists()
              else PhaseLabels(video.name, info.fps, info.frame_count))
    if labels.frame_count != info.frame_count:
        raise ValueError(
            f"{label_path} was made for {labels.frame_count} frames, video has "
            f"{info.frame_count}; refusing to mix them.")

    scale = min(1.0, max_height / info.height)
    print(f"decoding {info.frame_count} frames...", flush=True)
    jpegs = load_jpeg_frames(video, scale, quality=92)
    seq = PoseSequence.load(keypoints) if keypoints else None
    show_skel, show_help, idx = seq is not None, True, 0
    win = f"label: {video.name}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def save() -> None:
        labels.save(label_path)

    while True:
        frame = cv2.imdecode(np.frombuffer(jpegs[idx], np.uint8), cv2.IMREAD_COLOR)
        if show_skel and seq is not None and 0 <= idx - seq.start_frame < len(seq):
            i = idx - seq.start_frame
            pose = seq.keypoints[i].copy()
            pose[:, :2] *= scale
            draw_skeleton(frame, pose, 0.3, None if seq.filled is None else seq.filled[i])
        here = labels.phase_at(idx)
        _text(frame, f"frame {idx}/{info.frame_count - 1}   t={idx / info.fps:.3f}s", (10, 28))
        if here:
            _text(frame, f"LABEL: {here.upper()}", (10, 58), (0, 255, 255), 0.8)
        y = 90
        for p in PHASES:
            f = labels.labels.get(p)
            _text(frame, f"{PHASES.index(p) + 1} {p}: {'-' if f is None else f}", (10, y),
                  (0, 255, 0) if f is not None else (180, 180, 180), 0.5)
            y += 22
        for msg in labels.problems():
            _text(frame, "! " + msg, (10, y), (0, 0, 255), 0.5)
            y += 22
        if show_help:
            for n, line in enumerate(HELP):
                _text(frame, line, (10, frame.shape[0] - 12 - 20 * (len(HELP) - 1 - n)), size=0.45)
        cv2.imshow(win, frame)

        key = cv2.waitKeyEx(0)
        step = 0
        if key in _LEFT or key == ord("a"): step = -1
        elif key in _RIGHT or key == ord("d"): step = 1
        elif key in _UP or key == ord("w"): step = -10
        elif key in _DOWN or key == ord("s"): step = 10
        elif key in _PGUP or key == ord("q"): step = -50
        elif key in _PGDN or key == ord("e"): step = 50
        elif ord("1") <= key <= ord(str(len(PHASES))):
            labels.set(PHASES[key - ord("1")], idx)
            save()
        elif key in (8, 127, 3014656):  # backspace / delete
            here = labels.phase_at(idx)
            if here:
                labels.clear(here)
                save()
        elif key == ord("u"):
            if labels.undo():
                save()
        elif key == ord("k") and seq is not None:
            show_skel = not show_skel
        elif key == ord("h"):
            show_help = not show_help
        elif key == 27:
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        idx = int(np.clip(idx + step, 0, info.frame_count - 1))

    cv2.destroyAllWindows()
    save()
    return labels
