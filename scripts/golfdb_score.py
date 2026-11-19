"""Score this project's detectors against GolfDB labels, on videos you have fetched yourself.

Usage:
  python scripts/golfdb_score.py --videos data/external/golfdb/videos [--mat data/external/golfdb/golfDB.mat]
        [--backend mediapipe] [--slow-factor 4] [--limit N]

Expects one file per YouTube video, named ``<youtube_id>.<ext>`` (mp4, mkv, webm or mov), holding the
whole video. For every GolfDB sample whose video is present, it runs pose estimation on the
sample's window, detects phases, and compares takeaway, shaft-parallel, top and impact with the
mapped labels in GolfDB's 30 fps frames (docs/golfdb.md). Finish is not scored (different definition).

Slow-motion samples are re-timed clips: their capture rate is taken as the file's rate times
``--slow-factor`` (default 4, GolfDB's typical slowdown). shaft_parallel is only present if you add
``--shaft`` (experimental, reads the video; slow). Results are written to outputs/golfdb_scores.json.

The frame numbers in GolfDB refer to its own 30 fps sampling of each video. A video at another frame
rate is mapped by time; if yours was re-encoded or trimmed differently from theirs, labels will not
line up, and the scores say nothing about the detector.
"""

import argparse
import json
from pathlib import Path

from swing.golfdb import aggregate, detect_for_sample, load_golfdb, score_sample, to_native, valid
from swing.io import probe_video
from swing.pose import estimate_video, get_estimator

ap = argparse.ArgumentParser()
ap.add_argument("--videos", required=True)
ap.add_argument("--mat", default="data/external/golfdb/golfDB.mat")
ap.add_argument("--backend", default="mediapipe")
ap.add_argument("--slow-factor", type=float, default=4.0)
ap.add_argument("--limit", type=int, default=0, help="stop after this many samples")
ap.add_argument("--shaft", action="store_true", help="also estimate shaft-parallel (experimental)")
a = ap.parse_args()

files = {p.stem: p for p in Path(a.videos).iterdir() if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}}
samples = [s for s in load_golfdb(a.mat) if valid(s) and s.youtube_id in files]
if a.limit:
    samples = samples[:a.limit]
print(f"{len(files)} videos found, {len(samples)} labelled samples to score")
rows = []
with get_estimator(a.backend) as est:
    for s in samples:
        path = files[s.youtube_id]
        info = probe_video(path)
        pad = 15
        lo = max(0, to_native(s.events[0] - pad, info.fps))
        hi = min(info.frame_count, to_native(s.events[9] + pad, info.fps))
        seq = estimate_video(path, est, lo, hi)
        frames = detect_for_sample(seq, s.slow, a.slow_factor)
        if a.shaft:
            from dataclasses import replace

            from swing.phases import PhaseResult
            from swing.shaft import refine_shaft, shaft_cue
            from swing.smoothing import smooth_sequence

            sm = smooth_sequence(replace(seq, fps=seq.fps * (a.slow_factor if s.slow else 1.0)))
            res, _ = refine_shaft(PhaseResult(frames, sm.fps), shaft_cue(path, sm, frames))
            frames = res.frames
        errors = score_sample(s, frames, info.fps)
        rows.append({"id": s.id, "youtube_id": s.youtube_id, "slow": s.slow, "view": s.view, "club": s.club,
                     "errors": errors})
        print(f"sample {s.id:4d} {'slow' if s.slow else 'real'} {s.view:13s} {s.club:7s} errors (30 fps frames): {errors}")
summary = aggregate(rows)
Path("outputs").mkdir(exist_ok=True)
Path("outputs/golfdb_scores.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
for g, ph in summary.items():
    print(f"\n[{g}]")
    for p, v in ph.items():
        print(f"  {p:15s} " + (f"n={v['n']} missing={v['missing']} bias={v['bias']:+.1f} median|e|={v['median_abs']:.1f} "
                               f"mean|e|={v['mean_abs']:.1f} within2={v['within_2']:.0%} within5={v['within_5']:.0%}"
                               if v["n"] else f"no scores (missing={v['missing']})"))
