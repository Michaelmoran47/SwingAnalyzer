"""Click where the wrists are on chosen frames (OpenCV GUI).

Deliberately shows no pose-model output, so your clicks are not anchored on any backend.
Frames come from a fixed rule (every ``step``-th frame in a range), not from where the
models disagree.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from swing.annotations import (WRISTS, WristAnnotations, default_annotation_path,
                               sample_frames)
from swing.io import iter_frames, probe_video

_LEFT, _RIGHT = {2424832, 65361, 63234}, {2555904, 65363, 63235}
_UP, _DOWN = {2490368, 65362, 63232}, {2621440, 65364, 63233}

HELP = [
    "click: place the SELECTED wrist (it then switches to the other one)",
    "1 = golfer's LEFT wrist (lead arm)   2 = golfer's RIGHT wrist",
    "v: flip the wrist you just clicked (or just selected) between visible and HIDDEN guess",
    "n/p: next/previous sample frame   left/right or a/d: +-1   up/down: +-10",
    "backspace: clear selected wrist here | u: undo | h: help | Esc: save & quit",
]
_COL = {"left_wrist": (255, 120, 0), "right_wrist": (0, 200, 255)}  # BGR


def _text(img, s, org, color=(255, 255, 255), size=0.6):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def run(video: str | Path, path: str | Path | None = None, samples: list[int] | None = None,
        crop_y: tuple[int, int] | None = None, max_height: int = 900) -> WristAnnotations:
    video = Path(video)
    info = probe_video(video)
    path = Path(path) if path else default_annotation_path(video)
    ann = (WristAnnotations.load(path) if path.exists()
           else WristAnnotations(video.name, info.fps, info.frame_count))
    if ann.frame_count != info.frame_count:
        raise ValueError(f"{path} was made for {ann.frame_count} frames, video has "
                         f"{info.frame_count}; refusing to mix them.")
    samples = sorted(set(samples)) if samples else list(range(info.frame_count))
    y0, y1 = crop_y if crop_y else (0, info.height)
    y0, y1 = max(0, y0), min(info.height, y1)
    scale = min(1.0, max_height / (y1 - y0))

    # Only the frames we need, decoded once: the sample frames plus their +-1/+-10 neighbours
    # would be most of the clip, so keep the whole range between the first and last sample.
    lo, hi = samples[0], samples[-1]
    print(f"decoding frames {lo}-{hi}...", flush=True)
    jpegs: dict[int, bytes] = {}
    for i, frame in iter_frames(video, lo, hi + 1):
        crop = frame[y0:y1]
        small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        jpegs[i] = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()

    state = {"idx": lo, "sel": WRISTS[0], "last": None, "dirty": True, "help": True}
    win = f"wrists: {video.name}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, mx, my, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ann.set(state["idx"], state["sel"], mx / scale, my / scale + y0)
        ann.save(path)
        state["last"] = state["sel"]
        state["sel"] = WRISTS[1] if state["sel"] == WRISTS[0] else WRISTS[0]
        state["dirty"] = True

    cv2.setMouseCallback(win, on_mouse)

    def render() -> None:
        idx = state["idx"]
        frame = cv2.imdecode(np.frombuffer(jpegs[idx], np.uint8), cv2.IMREAD_COLOR)
        for w in WRISTS:
            p = ann.get(idx, w)
            if p is None:
                continue
            c = (int(p.x * scale), int((p.y - y0) * scale))
            if p.visible:
                cv2.circle(frame, c, 6, _COL[w], -1)
            else:
                cv2.circle(frame, c, 8, _COL[w], 2)  # hollow = hidden guess
        done = sum(1 for s in samples if s in ann.complete_frames())
        _text(frame, f"frame {idx}   t={idx / info.fps:.3f}s   sample frames done {done}/{len(samples)}",
              (10, 28))
        y = 58
        for n, w in enumerate(WRISTS, 1):
            p = ann.get(idx, w)
            status = "-" if p is None else ("visible" if p.visible else "HIDDEN guess")
            mark = ">" if state["sel"] == w else " "
            _text(frame, f"{mark}{n} {w}: {status}", (10, y), _COL[w], 0.55)
            y += 24
        if state["help"]:
            for n, line in enumerate(HELP):
                _text(frame, line, (10, frame.shape[0] - 12 - 20 * (len(HELP) - 1 - n)), size=0.45)
        cv2.imshow(win, frame)

    def move_to(target: int) -> None:
        state["idx"] = int(np.clip(target, lo, hi))
        state["sel"], state["last"] = WRISTS[0], None

    while True:
        if state["dirty"]:
            render()
            state["dirty"] = False
        key = cv2.waitKeyEx(30)
        if key == -1:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue
        idx = state["idx"]
        if key in _LEFT or key == ord("a"): move_to(idx - 1)
        elif key in _RIGHT or key == ord("d"): move_to(idx + 1)
        elif key in _UP or key == ord("w"): move_to(idx - 10)
        elif key in _DOWN or key == ord("s"): move_to(idx + 10)
        elif key == ord("n"):
            nxt = [s for s in samples if s > idx]
            if nxt: move_to(nxt[0])
        elif key == ord("p"):
            prv = [s for s in samples if s < idx]
            if prv: move_to(prv[-1])
        elif key in (ord("1"), ord("2")):
            state["sel"] = state["last"] = WRISTS[key - ord("1")]
        elif key == ord("v"):
            w = state["last"] or state["sel"]
            p = ann.get(idx, w)
            if p is not None:
                ann.set_visible(idx, w, not p.visible)
                ann.save(path)
        elif key in (8, 127, 3014656):
            ann.clear(idx, state["sel"])
            ann.save(path)
        elif key == ord("u"):
            if ann.undo():
                ann.save(path)
        elif key == ord("h"):
            state["help"] = not state["help"]
        elif key == 27:
            break
        state["dirty"] = True

    cv2.destroyAllWindows()
    ann.save(path)
    return ann


__all__ = ["run", "sample_frames"]
