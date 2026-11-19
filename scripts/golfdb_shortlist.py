"""Shortlist GolfDB videos worth considering, from the label file only (no video is fetched).

Usage:
  python scripts/golfdb_shortlist.py [--mat data/external/golfdb/golfDB.mat] [--clips-per-cell 6]
                                     [--min-slow-frames 0]

Picks, for each of four cells (real-time / slow-motion x down-the-line / face-on), the videos with
the most clips, until the cell has about ``--clips-per-cell`` clips, so a handful of videos cover the
set. Writes data/external/golfdb/shortlist.md and .json next to the labels (both git-ignored).

The YouTube links come from the dataset; some videos may have been removed or made private since
2019, and they are other people's uploads (see docs/golfdb.md before downloading anything).
"""

import argparse
import json
from pathlib import Path

from swing.golfdb import dataset_summary, load_golfdb, shortlist

ap = argparse.ArgumentParser()
ap.add_argument("--mat", default="data/external/golfdb/golfDB.mat")
ap.add_argument("--clips-per-cell", type=int, default=6)
ap.add_argument("--min-slow-frames", type=int, default=0,
                help="slow-motion clips must span at least this many frames from address to impact")
a = ap.parse_args()

samples = load_golfdb(a.mat)
sl = shortlist(samples, a.clips_per_cell, a.min_slow_frames)
out = Path(a.mat).parent
(out / "shortlist.json").write_text(json.dumps(sl, indent=2))
lines = ["# GolfDB shortlist", "", f"{len(sl)} videos, {sum(x['clips'] for x in sl)} clips "
         f"(target {a.clips_per_cell} per cell). Summary of the dataset: `{dataset_summary(samples)}`", "",
         "| cell | player | clubs | clips | address-to-impact (30 fps frames) | video |", "|---|---|---|---|---|---|"]
for x in sl:
    lines.append(f"| {x['cell']} | {x['player']} | {', '.join(x['clubs'])} | {x['clips']} | "
                 f"{x['address_to_impact_frames']} | {x['url']} |")
(out / "shortlist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nwritten to {out / 'shortlist.md'} and shortlist.json")
