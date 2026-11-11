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
