"""Command-line interface: ``swing info|overlay``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from swing.io import probe_video


def _cmd_info(args: argparse.Namespace) -> int:
    info = probe_video(args.video)
    print(f"path:       {info.path}")
    print(f"resolution: {info.width}x{info.height} (orientation applied)")
    print(f"fps:        {info.fps:.3f}")
    print(f"frames:     {info.frame_count}")
    print(f"duration:   {info.duration_s:.3f} s (of recorded time)")
    for w in info.warnings:
        print(f"warning:    {w}")
    return 0


def _cmd_label(args: argparse.Namespace) -> int:
    from swing.label_tool import run

    labels = run(args.video, args.labels, args.keypoints, args.max_height)
    print(f"labels: {labels.labels}")
    for msg in labels.problems():
        print(f"warning: {msg}")
    return 0


def _cmd_phases(args: argparse.Namespace) -> int:
    from swing.labels import PhaseLabels
    from swing.phases import compare, detect_phases
    from swing.pose import PoseSequence
    from swing.smoothing import smooth_sequence

    seq = PoseSequence.load(args.keypoints)
    if not args.no_smooth:
        seq = smooth_sequence(seq)
    res = detect_phases(seq)
    labels = PhaseLabels.load(args.labels) if args.labels else None
    rows = {e.phase: e for e in compare(res, labels)} if labels else {}
    print(f"{'phase':10} {'frame':>6} {'time_s':>7}" + (f" {'label':>6} {'err_fr':>7} {'err_ms':>8}  scored" if labels else ""))
    for ph, f in res.frames.items():
        line = f"{ph:10} {'-' if f is None else f:>6} {'-' if f is None else format(f / seq.fps, '.3f'):>7}"
        if labels:
            e = rows[ph]
            line += (f" {'-' if e.labelled is None else e.labelled:>6}"
                     f" {'-' if e.error_frames is None else format(e.error_frames, '+d'):>7}"
                     f" {'-' if e.error_ms is None else format(e.error_ms, '+.1f'):>8}"
                     f"  {'yes' if e.validated else 'no (threshold-defined)'}")
        print(line)
    for w in res.warnings:
        print(f"warning: {w}")
    return 0 if all(v is not None for v in res.frames.values()) else 1


def _cmd_metrics(args: argparse.Namespace) -> int:
    import json

    from swing.metrics import compute_metrics
    from swing.phases import detect_phases
    from swing.pose import PoseSequence
    from swing.smoothing import smooth_sequence

    seq = PoseSequence.load(args.keypoints)
    if not args.no_smooth:
        seq = smooth_sequence(seq)
    phases = detect_phases(seq)
    rep = compute_metrics(seq, phases, args.view)
    print(f"view: {args.view}   phases: " + ", ".join(
        f"{p}={'-' if f is None else f}" for p, f in phases.frames.items()))
    for name, v in rep.values.items():
        if v is not None:
            print(f"  {name:34s} {v:9.3f}")
    for name, why in rep.unavailable.items():
        print(f"  {name:34s} unavailable: {why}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"view": rep.view, "phases": phases.frames, "values": rep.values,
             "unavailable": rep.unavailable}, indent=2) + "\n")
        print(f"json -> {args.json}")
    return 0


def _cmd_overlay(args: argparse.Namespace) -> int:
    from swing.pose import PoseSequence, estimate_video, get_estimator
    from swing.render import write_overlay_video

    info = probe_video(args.video)
    for w in info.warnings:
        print(f"warning: {w}", file=sys.stderr)

    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"\rpose {done}/{total}", end="", file=sys.stderr, flush=True)

    if args.load_keypoints:
        seq = PoseSequence.load(args.load_keypoints)
    else:
        kw = ({"device": args.device, "mode": args.rtm_mode}
              if args.backend == "rtmpose" else {})
        with get_estimator(args.backend, **kw) as est:
            seq = estimate_video(args.video, est, args.start, args.end, progress)
        print(file=sys.stderr)

    if args.smooth:
        from swing.smoothing import smooth_sequence

        seq = smooth_sequence(seq, args.min_conf, window_s=args.smooth_window)
    det = int(seq.detected.sum())
    print(f"detected a person in {det}/{len(seq)} frames")
    if args.keypoints:
        Path(args.keypoints).parent.mkdir(parents=True, exist_ok=True)
        seq.save(args.keypoints)
        print(f"keypoints -> {args.keypoints}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = write_overlay_video(args.video, seq, out, args.min_conf, args.playback_fps)
    print(f"overlay ({n} frames @ {args.playback_fps or seq.fps:g} fps playback) -> {out}")
    return 0 if det else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="swing", description="Golf swing analyzer")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("info", help="print video properties")
    pi.add_argument("video")
    pi.set_defaults(func=_cmd_info)

    po = sub.add_parser("overlay", help="render a skeleton overlay video")
    po.add_argument("video")
    po.add_argument("-o", "--output", default="outputs/overlay.mp4")
    po.add_argument("--backend", default="mediapipe")
    po.add_argument("--device", default="cpu", help="rtmpose only: cpu or cuda")
    po.add_argument("--rtm-mode", default="balanced",
                    choices=["lightweight", "balanced", "performance"], help="rtmpose model size")
    po.add_argument("--keypoints", help="also save keypoints to this .npz")
    po.add_argument("--start", type=int, default=0, help="first frame index")
    po.add_argument("--end", type=int, default=None, help="end frame index (exclusive)")
    po.add_argument("--load-keypoints", help="reuse a saved .npz instead of running pose")
    po.add_argument("--playback-fps", type=float, default=None,
                    help="output playback rate (e.g. 30 for smooth slow-mo); default = source fps")
    po.add_argument("--smooth", action="store_true", help="fill gaps and smooth keypoints")
    po.add_argument("--smooth-window", type=float, default=0.06,
                    help="Savitzky-Golay window in seconds (default 0.06)")
    po.add_argument("--min-conf", type=float, default=0.3)
    po.set_defaults(func=_cmd_overlay)

    pl = sub.add_parser("label", help="hand-label swing phases frame by frame")
    pl.add_argument("video")
    pl.add_argument("--labels", help="label file (default data/labels/<video stem>.json)")
    pl.add_argument("--keypoints", help="optional .npz to show a skeleton (toggle with k)")
    pl.add_argument("--max-height", type=int, default=900, help="window height in px")
    pl.set_defaults(func=_cmd_label)

    pp = sub.add_parser("phases", help="detect swing phases from a RAW keypoints .npz")
    pp.add_argument("keypoints", help=".npz from `swing overlay --keypoints` (unsmoothed)")
    pp.add_argument("--labels", help="hand-label JSON to score against")
    pp.add_argument("--no-smooth", action="store_true", help="input is already smoothed")
    pp.set_defaults(func=_cmd_phases)

    pm = sub.add_parser("metrics", help="compute swing metrics from a RAW keypoints .npz")
    pm.add_argument("keypoints")
    pm.add_argument("--view", required=True, choices=["dtl", "fo"],
                    help="camera view of this clip: down-the-line or face-on")
    pm.add_argument("--no-smooth", action="store_true", help="input is already smoothed")
    pm.add_argument("--json", help="also write results to this file")
    pm.set_defaults(func=_cmd_metrics)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
