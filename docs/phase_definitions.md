# Phase definitions

Phases are detected from the wrist trajectory (mean of the available wrists,
smoothed; see `src/swing/phases/detect.py`). They fall into two groups with
different standing.

## Hand-validated: top, impact

These are discrete, reproducible events, so a hand label is a trustworthy
reference and the detector is scored against it.

| Phase | Rule |
|---|---|
| **impact** | Lowest hand position (max image y) within 0.20 s after peak hand speed. |
| **top** | Highest hand position (min image y) before the peak hand speed. |

Only these two count toward any accuracy claim in this project.

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

## Minimum frame rate

The rules use windows of tens of milliseconds (0.2 s after the peak for impact, a 0.15 s hold
for finish, a 60 ms smoother) and were built on 240 fps footage. Below 100 fps
(`PhaseParams.min_fps`) detection returns no phases and a warning instead of guessing. On a
30 fps re-timed clip, an earlier version ran anyway and put top and impact on the same frame.
Telling the detector the real capture rate fixes the time base (`--fps`); it does not make
the rules valid for a clip that was genuinely captured at 30 fps, which is not supported.

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

