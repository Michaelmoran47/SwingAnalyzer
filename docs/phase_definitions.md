# Phase definitions

Phases are detected from the wrist trajectory (mean of the available wrists,
smoothed; see `src/swing/phases/detect.py`). They fall into two groups with
different standing.

## Hand-validated: top, impact

These are discrete, reproducible events, so a hand label is a trustworthy
reference and the detector is scored against it.

| Phase | Rule |
|---|---|
| **impact** | When the clip shows the ball on the tee: the first frame the ball leaves it, from the picture ([impact_cue.md](impact_cue.md)). Otherwise, or if that cue is not sound: lowest hand position (max image y) within 0.20 s after peak hand speed. |
| **top** | Highest hand position (min image y) before the peak hand speed. |

Only these two count toward any accuracy claim in this project. The hand-only impact can be several
frames off (-6 frames on one slow-motion clip, where the hands stall while the club swings through);
the ball cue exists because of that.

## Threshold-defined: address, takeaway, finish

These events are gradual. On the first labelled clip, the hands creep for about
0.4 s before the swing proper, and at the moment a human would call "takeaway"
the hands had moved 4.7 px, which is on the order of the pose model's own
resting noise (about 3.4 px). Finish is an asymptote: the hands settle
gradually, so no single speed threshold picks one frame. A hand label for these
is therefore not a reliable reference, and the detector's threshold *is* the
definition. Values are relative, so they do not depend on resolution, distance
or frame rate.

| Phase | Rule (defaults in `PhaseParams`) |
|---|---|
| **address** | Scanning back from the top (past the slow region around it), the last frame with hand speed below **2 %** of peak speed. |
| **takeaway** | First frame after address where the hands are more than **5 %** of torso length from their address position. |
| **finish** | First frame after impact where hand speed stays below **10 %** of peak for **0.15 s**. |

Consequences to keep in mind:

- Address/takeaway/finish times are consistent between swings (same rule, same
  clip conditions), which is what comparisons and tempo need, but they are not
  claimed to match what a human sees.
- Tempo (takeaway→top versus top→impact) depends on the takeaway rule. On the
  first clip, a human takeaway label gives about 2.6 : 1 and the threshold rule
  gives about 1.5 : 1. Report the rule alongside any tempo number.
- The rule parameters were set from first principles, not tuned to any clip.
  Changing them changes these phases and any metric built on them.

## Frame rate and slowed-down clips

Two different problems look alike and need different handling.

**A genuinely low frame rate** (a normal 30 or 60 fps phone video). The rules cope.
`scripts/framerate_sweep.py` keeps every k-th frame of the 240 fps keypoints of `IMG_6004`
(every possible offset), runs smoothing and phase detection, and scores top and impact against
your hand labels. Error = detected minus label, in ms (min / median / max over the offsets);
phases were found at every offset and rate.

| fps | 1 frame | top, MediaPipe | impact, MediaPipe | top, RTMPose balanced | impact, RTMPose balanced |
|---|---|---|---|---|---|
| 239.5 | 4.2 ms | -13 / -13 / -13 | +17 / +17 / +17 | -13 / -13 / -13 | +4 / +4 / +4 |
| 119.7 | 8.4 ms | -21 / -19 / -17 | +17 / +19 / +21 | -13 / -10 / -8 | +4 / +6 / +8 |
| 59.9 | 16.7 ms | -21 / -15 / -8 | 0 / +23 / +29 | -29 / -13 / -4 | -4 / +2 / +8 |
| 29.9 | 33.4 ms | -29 / -15 / 0 | 0 / +15 / +29 | -25 / -6 / +17 | -13 / +2 / +17 |
| 23.9 | 41.8 ms | -38 / -19 / 0 | -4 / +15 / +33 | -29 / -6 / +17 | -21 / -2 / +17 |
| 20.0 | 50.1 ms | -42 / -19 / +4 | -21 / +2 / +25 | -33 / -10 / +13 | -21 / +2 / +25 |
| 15.0 | 66.8 ms | -46 / -15 / +17 | -25 / +6 / +38 | -38 / -6 / +25 | -33 / -2 / +29 |

So the error stays within about one frame interval down to 15 fps, which is why
`PhaseParams.min_fps` is 15 and detection is not attempted below it. **This is optimistic.**
The keypoints came from sharp 240 fps frames; real 30 fps video has longer exposures, more
motion blur at the fast part of the swing, and worse pose estimates there. It is also one swing
with one set of labels, so it shows how the rules behave with fewer samples, not how accurate
they are in general. The finish rule (a 0.15 s hold below a speed threshold) is only a few frames
at 30 fps and was seen to fire right after impact on one low-rate clip; it is not scored.

**A time-stretched clip**: a high-frame-rate capture re-timed to play slowly, for example 240 fps
footage stamped as 30 fps (the first re-timed upload, `IMG_6003`). The swing then spans about 8x
more frames than a real-time swing, the millisecond-scale windows become a few frames, and an
earlier version put top and impact on the same frame without saying so. Now:

- `plausibility_issues` checks the result: phases must be in order and not share a frame,
  the backswing (takeaway to top) must last 0.2 to 3.0 s and the downswing (top to impact) 0.08
  to 1.5 s. The ranges are wide on purpose, they exist to catch a slowed clip or a failed
  detection, not to judge a swing. They are assumptions about swings, not measured here.
- When the check fails, the result is flagged (`PhaseResult.plausible`), a warning says what is
  wrong and how to give the capture rate (`--fps`, or the demo's control), and all time-based
  metrics are withheld. Geometry metrics are also withheld if the phases are out of order.
- A slowed clip whose slowdown is modest (say 2x) can still pass these ranges. The check is a
  safety net, not proof that the timeline is real.

## Visual re-check with corrected frames

An earlier by-eye diagnosis of address and takeaway used images from `iter_frames(start=660)`,
which seeked to the wrong frame (see "Correction: frame-seek bug" in [rtmpose.md](rtmpose.md)),
so its statement that the detector's address (frame 786) was "wrong by any definition" is
withdrawn. Re-checked with the fixed `iter_frames` (frames 681, 720, 731, 786, 800, 1077,
1117, 1181 of `IMG_6004`, by eye, not quantified):

- 681 and 720 look like address; at 731 the club has moved very slightly; by 786 the club
  head has visibly left its position behind the ball. So the hand labels (address 681,
  takeaway 731) are reasonable, and the detector's address (786) and takeaway (800) are
  later because the hands move little while the club starts moving.
- Finish: from 1077 on the hands stay in roughly the same place while the club keeps
  swinging round (1077, 1117, 1181), which matches the hands-versus-club difference
  between the detector's finish (1077) and the hand label (1117).

The decision to treat address, takeaway and finish as threshold-defined rests on the
keypoint numbers above (hand displacement of 4.7 px at the takeaway label, close to pose
noise) and on this hands-versus-club difference, not on the earlier misaligned images.

