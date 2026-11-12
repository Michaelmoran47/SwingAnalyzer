# Shaft parallel (backswing) - experimental

**Status: experimental and unvalidated.** There are no hand labels for it yet, and on the three
clips it was tried on I think it is late by 5 to 25 frames (my reading of the frames, not ground
truth). It is off by default. It is built so that your labels can measure it.

Definition: in the backswing, the moment the club shaft is parallel to the ground. It is its own
checkpoint, separate from the takeaway (the start of the swing, see
[phase_definitions.md](phase_definitions.md)).

## How to turn it on and label it

- CLI: `swing phases <keypoints.npz> --video <clip> --shaft [--fps N] [--labels file.json]`
- Demo: the sidebar option *Estimate shaft-parallel (experimental)*.
- Label it with `swing label <clip>`, key **6**. Keys 1-5 are unchanged (address, takeaway, top,
  impact, finish). It is optional: a label set without it is still complete.
- Check by eye: `python scripts/shaft_check.py <clip> <keypoints.npz> [--fps N]` draws the detected
  shaft on the frames around the estimate.

## Why it has to come from the picture

The pose model has no club keypoint. In the image the shaft is parallel to the ground when the club
head is level with the hands, which holds in any view if the camera is level, so the checkpoint is
the frame where the shaft's **elevation** (the angle of the line from the hands to the club head above
the horizontal) crosses zero on the way up. At address it is about -45 to -60 degrees on the clips
tried; at the top, about +55 to +60.

## Method (`src/swing/shaft.py`)

1. For frames between takeaway and top, find the strongest straight edge that passes through the hands,
   extends away from them, does not run back along the forearm, and does not carry on behind the
   hands (a background edge does; a shaft ends at the grip). Canny plus a probabilistic Hough
   transform in a window around the hands.
2. Fit a non-decreasing curve (the shaft only rotates up in a backswing) with each pool taking its
   weighted **median**, so one stray line cannot drag it. A mean-based fit let one false reading move
   the crossing by about 20 frames.
3. Report the frame where the fit crosses zero, interpolated, **with the bracket**: the last reading
   below horizontal and the first above it. The page shows the bracket next to the estimate.

## What it found, and why not to trust it yet

| Clip | Estimate (bracket) | What the frames show (my reading) |
|---|---|---|
| `IMG_6004` | 839 (811-863) | At 835, 839 and 843 the club is already clearly above horizontal, so the estimate is probably late; the shaft was not detected there (thin gray shaft on a gray wall, motion-blurred) |
| Recording | 140 (140-156) | The crossing looks like about 133-135; the readings near horizontal at 132-140 are partly false horizontal lines, and the estimate moved from 132 to 140 when one line filter was added |
| `IMG_6003` | 1293 (1264-1318) | A 54-frame gap with no usable reading; the estimate is the middle of the gap |

Limits, plainly:

- **The shaft is blurred and often undetected exactly where it crosses horizontal**, because the club
  is moving fastest there. The estimate is then an interpolation inside a gap, and the bracket is the
  honest statement of what is known.
- **Small changes to the filters moved the estimate by about 8 frames**, and without labels there is
  no way to say which version is closer. Further tuning now would be guessing.
- **It measures the angle in the image, not in 3D.** It is right when the camera is level and the shaft
  is not pointing along the camera's line of sight; a perfectly down-the-line shaft at that moment
  would look short and its angle would be unreliable.
- A low-contrast shaft (gray on gray), a hidden club, or hands out of the window give no reading.

When you have labelled shaft-parallel on a few clips, `swing phases ... --shaft --labels ...` will show
the error per clip (`shaft_parallel` is listed as experimental until it has earned a place in the scored
set).
