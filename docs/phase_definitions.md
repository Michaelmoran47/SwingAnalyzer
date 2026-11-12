# Phase definitions

Phases are detected from the wrist trajectory (mean of the available wrists,
smoothed; see `src/swing/phases/detect.py`). They fall into two groups with
different standing.

## Scored against your hand labels: takeaway, top, impact

| Phase | Rule |
|---|---|
| **takeaway** | The start of the swing: the first frame after the **last quiet stretch** before it. Eight joints (shoulders, hips, elbows, wrists) are pooled into one mean speed; a frame is quiet if that speed is within 3x this clip's own noise level (the median of the quietest 25 % of frames before the top); the stretch must last 0.10 s (at least 10 frames). Falls back to the wrist rule below if no quiet stretch is found. Code: `src/swing/phases/onset.py`. |
| **top** | Highest hand position (min image y) before the peak hand speed. If the golfer pauses at the top, any frame in the pause is correct: label it as a range. |
| **impact** | When the clip shows the ball on the tee: the first frame the ball leaves it, from the picture ([impact_cue.md](impact_cue.md)). Otherwise, or if that cue is not sound: lowest hand position (max image y) within 0.20 s after peak hand speed. |

Why takeaway moved: the old rule used the wrists alone, with a threshold relative to the *peak
downswing speed*. It ignored the small early motion of the shoulders, hips and arms that a person
sees as the start of the swing, and was **late on all three labelled swings (+69, +37, +96 frames)**.
The hand-only impact could be several frames off (-6 on a slow-motion clip, where the hands stall
while the club swings through), which is why the ball cue exists.

### Errors against your labels (detected minus label, frames)

| Clip | takeaway (hand rule, before) | takeaway (body motion) | top | impact (hands, then ball) |
|---|---|---|---|---|
| `IMG_6004` (240 fps) | +69 | -16 | -3 | +4, then 0 |
| Recording (slow-motion screen capture) | +37 | -28 | 0 (inside your 190-250 range) | -6, then 0 |
| `IMG_6003` (re-timed, 240 fps assumed) | +96 | +15 | -7 | no label (hands 1472, ball 1467) |

Mean absolute takeaway error fell from 67 to 20 frames, and it is no longer always late; the
typical miss is now +-15-30 frames (about 60-120 ms at 240 fps). Top is within 7 frames on all
three, impact within 0 once the ball cue applies.

**How far to trust this.** Three swings, two golfers. The takeaway settings were chosen among nine
combinations (below) on those same three clips, so these errors are optimistic. Neighbouring
settings give similar results, which is the reason to believe the choice is not a fluke, but the
spread across settings is as large as the improvement over the old rule on one clip.

`kappa` (quiet = within kappa x noise) and the quiet-stretch length, error in frames
(IMG_6004 / Recording / IMG_6003; "none" = no quiet stretch found), from a prototype whose results
differ from the integrated detector by 1-2 frames:

| kappa | 0.10 s | 0.15 s | 0.25 s |
|---|---|---|---|
| 1.5 | -222 / -36 / -94 | -221 / none / -75 | -223 / none / -73 |
| 2.0 | -27 / -27 / -4 | -28 / -26 / -58 | -27 / none / -45 |
| **3.0** | **-14 / -26 / +17** (chosen) | -13 / -25 / +22 | -6 / none / +33 |

Limits: a waggle or forward press right before the swing counts as movement, and a motion
detector cannot tell it from a takeaway (the recording is 28 frames early, which is probably this).
A clip needs a quiet stretch before the swing; one that starts mid-motion falls back to the wrists.
Shaft-parallel (the end of the takeaway in coaching language) is not detected or labelled yet.

## Threshold-defined, not scored: address, finish

| Phase | Rule (defaults in `PhaseParams`) |
|---|---|
| **address** | The frame before the takeaway (the last still frame). Your own address labels sit earlier, where the golfer settled (IMG_6004: 681 vs 714), so they are not comparable and address is not scored. |
| **finish** | First frame after impact where hand speed stays below **10 %** of peak for **0.15 s**. Finish is gradual, so no single threshold picks one frame; it differed from your label by -167 ms on `IMG_6004`. |

Values are relative, so they do not depend on resolution, distance or frame rate.

If the body-motion rule finds no quiet stretch, takeaway and address fall back to the wrists:
address = last frame where hand speed was below **2 %** of peak, scanning back from the top (past the
slow region around it); takeaway = first frame after address where the hands moved more than **5 %**
of torso length. That fallback is the rule that was late.

Consequences to keep in mind:

- Tempo (takeaway to top versus top to impact) now uses the earlier, body-motion takeaway. On
  `IMG_6004` that gives about 2.7 to 1, close to the 2.6 to 1 you get from your own labels; the old
  rule gave 1.5 to 1. Report the rule alongside any tempo number.
- The rule parameters were set from first principles and, for takeaway, checked on three labelled
  swings. Changing them changes address and any metric built on them.

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

(Superseded for takeaway: it is now detected from whole-body motion and scored, see the top of
this page. This re-check is kept as history.) The earlier decision to treat address, takeaway and
finish as threshold-defined rested on the
keypoint numbers above (hand displacement of 4.7 px at the takeaway label, close to pose
noise) and on this hands-versus-club difference, not on the earlier misaligned images.

