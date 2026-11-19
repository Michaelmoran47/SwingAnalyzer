# GolfDB: labels as a test set

[GolfDB](https://github.com/wmcnally/golfdb) (McNally et al., CVPR Workshops 2019) labels 1,400
swing clips cut from 580 YouTube videos of professionals (mostly PGA, LPGA and Champions Tour), with
eight events each. It is the only source of labels made by other people, so it can test the
detectors on far more swings than the three of ours. **Only the label file is used here.** The
videos are other people's uploads and are not part of this repository.

## What the label file holds

Downloaded to `data/external/golfdb/golfDB.mat` (117,644 bytes, SHA-256
`36523fcef016d93334691ea058d643bcf0d3a4a8c70b733f3b4cc0b8e5e0cf0a`; git-ignored). It is read with
SciPy's data-only MATLAB reader. A `.pkl` also exists in the repo but a pickle can run code when
opened, so it is not used.

- 1,400 samples, 580 YouTube videos, 246 players. 758 real-time and 642 slow-motion; 585
  down-the-line, 461 face-on, 354 "other"; 952 drivers, 229 irons, 162 fairway woods, 34 hybrids,
  23 wedges. One sample has its events out of order and is skipped.
- Per sample: the YouTube id, player, club, view, a slow-motion flag, ten frame numbers (start,
  address, toe-up, mid-backswing, top, mid-downswing, impact, mid-follow-through, finish, end) and a
  bounding box. Frame numbers refer to the source video sampled at **30 fps**.

## How its events map to this project's phases

| GolfDB (paper's definition) | Here |
|---|---|
| Address: "the frame before movement in the backswing is noticeable" | address; **takeaway = the next frame** |
| Toe-up: "shaft parallel with the ground during the backswing" | **shaft_parallel** |
| Top: "the moment the golf club changes directions" | top (a club-direction change; ours is hands-highest, so a long pause is judged differently) |
| Impact: "the moment the clubhead touches the golf ball" | impact |
| Finish: "just before the golfer's final pose is relaxed" | **not mapped**: it is the end of the held finish, ours is the start |
| Mid-backswing, mid-downswing, mid-follow-through | not used |

## What the label counts say about slow-motion clips

Median address-to-impact frames at 30 fps: 34 for real-time clips and 132 for slow-motion ones, and
top-to-impact 8 versus 30. So the typical slow-motion clip in this dataset is about **4 times** longer
than real time (my inference from the counts, with a spread of roughly 3 to 5). That is a prior for
the slowed-clip scan and for `--slow-factor`, not a measurement of any one clip. The slow flag is
not perfect: one slow-flagged clip spans only 47 frames.

## How to use it

1. `python scripts/golfdb_shortlist.py` lists a handful of videos (about 9 videos cover 32 clips
   across real-time/slow-motion x down-the-line/face-on) and writes `shortlist.md` next to the
   labels. It fetches nothing. Some videos may have been removed since 2019.
2. Obtaining the videos is your decision (see below). Put each whole video at
   `data/external/golfdb/videos/<youtube_id>.mp4`.
3. `python scripts/golfdb_score.py --videos data/external/golfdb/videos` runs pose estimation on each
   labelled sample's window, detects phases, and prints per-clip errors and a summary (bias, median
   and mean absolute error, share within 2 and 5 frames) for takeaway, shaft_parallel (only with
   `--shaft`), top and impact, split into real-time and slow-motion. Slow clips use the file's rate
   times `--slow-factor` (default 4).

The importer is `src/swing/golfdb.py` (`load_golfdb`, `to_labels`, `shortlist`, `score_sample`,
`aggregate`), tested on a synthetic label file with the same layout and against the real file when
present. **The scoring harness has not been run on real GolfDB video**, because none has been fetched;
the pieces it uses are tested, the end-to-end run is not.

## Cautions

- **Frame alignment.** GolfDB's frame numbers are for its own 30 fps sampling. A video at another
  rate is mapped by time; if your copy was re-encoded or trimmed differently, the labels will not line
  up, and the scores then say nothing about the detectors. Check a few clips by eye first.
- **Label noise.** One annotator chose each frame "at their own discretion"; the paper notes the exact
  impact moment was rarely captured at 30 fps and reports no agreement between annotators. Expect a
  frame or two of noise on real-time clips and more on slow-motion ones.
- **Footage.** Many clips are TV broadcast with panning cameras, mostly drivers, and 30 fps. The
  ball cue and shaft estimate were developed on slow-motion indoor and range clips and are untested
  on this kind of footage.
- **Rights.** The code is CC BY-NC 4.0. No licence is stated for the dataset or the videos, which are
  copyrighted uploads, and downloading from YouTube is against its terms. Private research use is the
  realistic line: keep any video local (the folder is git-ignored), do not redistribute it, and cite
  the paper if you report results.
