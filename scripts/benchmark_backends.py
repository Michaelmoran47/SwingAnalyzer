"""Compare pose backends on one clip. Writes a markdown report.

Usage:
  python scripts/benchmark_backends.py \
      --npz mediapipe=outputs/IMG_6004.npz rtmpose=outputs/IMG_6004_rtmpose.npz \
      --labels data/labels/IMG_6004.json --view dtl \
      --video data/raw/IMG_6004.MOV --time-frames 8 --out docs/benchmark.md

There are NO ground-truth keypoints, so this does not say which backend is more
*accurate*. It reports what can be measured without them:

* speed on this machine (CPU/GPU is printed, never assumed)
* completeness: share of wrist/hip joint-frames usable, before and after smoothing
* jitter of the raw output
* inter-backend disagreement (a difference, not an error)
* downstream effect: phase frames (top/impact scored against hand labels) and
  metric values from each backend

Confidence scales differ between backends (BlazePose visibility vs RTMPose SimCC
score), so "usable" uses the same numeric ``--min-conf`` for both only as a
convenience; the table says so.
"""

from __future__ import annotations

import argparse
import datetime
import platform
import time
from pathlib import Path

import numpy as np

from swing.labels import PhaseLabels
from swing.metrics import compute_metrics
from swing.phases import compare, detect_phases
from swing.pose import J, PoseSequence
from swing.smoothing import smooth_sequence

KEY_JOINTS = ["left_wrist", "right_wrist", "left_hip", "right_hip"]


def usable_fraction(k, joint, lo, hi, min_conf):
    seg = k[lo:hi, J[joint]]
    ok = np.isfinite(seg[:, :2]).all(axis=1) & (np.nan_to_num(seg[:, 2]) >= min_conf)
    return float(ok.mean())


def finite_fraction(k, joint, lo, hi):
    return float(np.isfinite(k[lo:hi, J[joint], 0]).mean())


def raw_jitter(k, joint):
    """RMS second difference (px/frame^2) over frames where 3 consecutive are finite."""
    p = k[:, J[joint], :2]
    d2 = p[2:] - 2 * p[1:-1] + p[:-2]
    d2 = d2[np.isfinite(d2).all(axis=1)]
    return float(np.sqrt((d2 ** 2).sum(axis=1).mean())) if len(d2) else float("nan")


def time_backend(name, video, start, n, device):
    from swing.io import iter_frames
    from swing.pose import get_estimator

    frames = [f for _, f in iter_frames(video, start, start + n + 1)]
    kw = {"device": device} if name == "rtmpose" else {}
    with get_estimator(name, **kw) as est:
        est.estimate(frames[0], 0.0)  # warm-up (model init, first-run allocations)
        t = time.perf_counter()
        for f in frames[1:]:
            est.estimate(f, 0.0)
        return (time.perf_counter() - t) / (len(frames) - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True, help="backend=path.npz (raw keypoints)")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--view", required=True, choices=["dtl", "fo"])
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--video", help="needed for --time-frames")
    ap.add_argument("--time-frames", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="device for rtmpose timing")
    ap.add_argument("--out", default="docs/benchmark.md")
    a = ap.parse_args()

    labels = PhaseLabels.load(a.labels)
    lo, hi = labels.labels["takeaway"], labels.labels["finish"] + 1  # swing window (hand labels)
    seqs = {}
    for item in a.npz:
        name, path = item.split("=", 1)
        seqs[name] = PoseSequence.load(path)
    smooth = {n: smooth_sequence(s, a.min_conf) for n, s in seqs.items()}

    L = [f"# Backend benchmark: {labels.video} ({a.view})", "",
         f"Generated {datetime.date.today()} by `scripts/benchmark_backends.py`. "
         f"One clip, one swing: a comparison on this footage, **not** an accuracy ranking "
         f"(no ground-truth keypoints exist).", "",
         f"Swing window = your hand labels takeaway..finish, frames {lo}-{hi - 1}. "
         f"Machine: {platform.processor() or platform.machine()}, {platform.system()} "
         f"{platform.release()}.", ""]

    if a.time_frames:
        import onnxruntime as ort
        L += ["## Speed (this machine)", "",
              f"ONNX Runtime providers available: {', '.join(ort.get_available_providers())}; "
              f"rtmpose device = `{a.device}`. Per-frame time at full video resolution "
              f"{labels.video}, {a.time_frames} frames after warm-up, starting mid-swing.", "",
              "| backend | s/frame | fps |", "|---|---|---|"]
        for n in seqs:
            dt = time_backend(n, a.video, 900, a.time_frames, a.device)
            L.append(f"| {n} | {dt:.3f} | {1 / dt:.1f} |")
        L.append("")

    L += ["## Completeness in the swing window", "",
          f"Share of frames where the joint is usable. *Raw* = finite position with "
          f"confidence >= {a.min_conf} (same number for both backends, whose confidence "
          f"scales differ). *After smoothing* = finite position after gap filling "
          f"(gaps up to 0.1 s interpolated). **\"Usable\" means the model reported a position "
          f"with confidence, not that the position is correct**: a backend can be confident "
          f"and wrong, so check with `scripts/compare_frames.py` (see docs/rtmpose.md).", "",
          "| backend | joint | raw usable | after smoothing |", "|---|---|---|---|"]
    for n in seqs:
        for j in KEY_JOINTS:
            L.append(f"| {n} | {j} | {usable_fraction(seqs[n].keypoints, j, lo, hi, a.min_conf):.1%} "
                     f"| {finite_fraction(smooth[n].keypoints, j, lo, hi):.1%} |")
    L.append("")

    L += ["## Raw jitter (whole clip)", "",
          "RMS second difference of the raw position, px/frame². Includes low-confidence "
          "frames. Lower = steadier, but a backend can be steady and wrong.", "",
          "| backend | " + " | ".join(KEY_JOINTS) + " |", "|---|" + "---|" * len(KEY_JOINTS)]
    for n in seqs:
        L.append(f"| {n} | " + " | ".join(f"{raw_jitter(seqs[n].keypoints, j):.2f}" for j in KEY_JOINTS) + " |")
    L.append("")

    names = list(seqs)
    if len(names) == 2:
        k1, k2 = seqs[names[0]].keypoints, seqs[names[1]].keypoints
        T = min(len(k1), len(k2))
        torso = float(np.nanmedian(np.linalg.norm(
            (k1[:T, J["left_shoulder"], :2] + k1[:T, J["right_shoulder"], :2]) / 2
            - (k1[:T, J["left_hip"], :2] + k1[:T, J["right_hip"], :2]) / 2, axis=1)))
        L += [f"## Disagreement between {names[0]} and {names[1]}", "",
              f"Median distance between the two backends' estimates of the same joint, over "
              f"swing-window frames where both have confidence >= {a.min_conf}. In torso lengths "
              f"(torso = {torso:.0f} px). A difference, **not** an error: either may be wrong.", "",
              "| joint | frames compared | median distance (px) | (torso lengths) |", "|---|---|---|---|"]
        for j in KEY_JOINTS:
            seg1, seg2 = k1[lo:hi, J[j]], k2[lo:hi, J[j]]
            ok = (np.nan_to_num(seg1[:, 2]) >= a.min_conf) & (np.nan_to_num(seg2[:, 2]) >= a.min_conf) \
                & np.isfinite(seg1[:, 0]) & np.isfinite(seg2[:, 0])
            d = np.linalg.norm(seg1[ok, :2] - seg2[ok, :2], axis=1)
            L.append(f"| {j} | {int(ok.sum())} | {np.median(d):.1f} | {np.median(d) / torso:.3f} |"
                     if ok.any() else f"| {j} | 0 | - | - |")
        L.append("")

    L += ["## Downstream: phases", "",
          "Detected frames from each backend's smoothed keypoints. **Only top and impact are "
          "scored** against your hand labels (see docs/phase_definitions.md); the others are "
          "threshold-defined. Error = detected - label in ms.", "",
          "| phase | label | " + " | ".join(f"{n} frame (err ms)" for n in names) + " |",
          "|---|---|" + "---|" * len(names)]
    res = {n: detect_phases(smooth[n]) for n in names}
    errs = {n: {e.phase: e for e in compare(res[n], labels)} for n in names}
    for p in ["address", "takeaway", "top", "impact", "finish"]:
        cells = []
        for n in names:
            e = errs[n][p]
            cells.append("-" if e.detected is None else
                         f"{e.detected} ({e.error_ms:+.0f})" + ("" if e.validated else " *"))
        L.append(f"| {p} | {labels.labels.get(p, '-')} | " + " | ".join(cells) + " |")
    L += ["", "\\* threshold-defined, not scored.", ""]

    L += [f"## Downstream: metrics ({a.view})", "",
          "Same code, different keypoints. Differences here show how much the choice of "
          "backend moves each number; neither column is validated.", "",
          "| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    reps = {n: compute_metrics(smooth[n], res[n], a.view) for n in names}
    for m in reps[names[0]].values:
        vals = [reps[n].values.get(m) for n in names]
        if all(v is None for v in vals):
            continue
        L.append(f"| {m} | " + " | ".join("-" if v is None else f"{v:.3f}" for v in vals) + " |")
    L.append("")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
