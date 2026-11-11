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


def _cmd_overlay(args: argparse.Namespace) -> int:
    from swing.pose import estimate_video, get_estimator
    from swing.render import write_overlay_video

    info = probe_video(args.video)
    for w in info.warnings:
        print(f"warning: {w}", file=sys.stderr)

    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"\rpose {done}/{total}", end="", file=sys.stderr, flush=True)

    with get_estimator(args.backend) as est:
        seq = estimate_video(args.video, est, args.start, args.end, progress)
    print(file=sys.stderr)

    det = int(seq.detected.sum())
    print(f"detected a person in {det}/{len(seq)} frames")
    if args.keypoints:
        seq.save(args.keypoints)
        print(f"keypoints -> {args.keypoints}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = write_overlay_video(args.video, seq, out, args.min_conf)
    print(f"overlay ({n} frames @ {seq.fps:g} fps) -> {out}")
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
    po.add_argument("--keypoints", help="also save keypoints to this .npz")
    po.add_argument("--start", type=int, default=0, help="first frame index")
    po.add_argument("--end", type=int, default=None, help="end frame index (exclusive)")
    po.add_argument("--min-conf", type=float, default=0.3)
    po.set_defaults(func=_cmd_overlay)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
