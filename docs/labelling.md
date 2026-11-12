# Labelling swing phases

Hand labels are the ground truth the rule-based phase detector is validated
against, so consistency matters more than speed. Use the same definitions for
every clip, and label each clip in one sitting.

## Run the tool

```
swing label data/raw/<clip>.MOV
```

Optional: `--keypoints outputs/<clip>_smooth.npz` shows the skeleton (toggle
with `k`). Labelling without it avoids biasing your choice toward what the
pose model sees.

Labels autosave to `data/labels/<clip>.json`. Reopening a clip resumes where you
left off. Frame numbers are indices into the *decoded* stream, the same ones the
code uses (see `swing info`), not the header frame count.

| Key | Action |
|---|---|
| ← / → or `a` / `d` | ±1 frame |
| ↑ / ↓ or `w` / `s` | ±10 frames |
| PgUp / PgDn or `q` / `e` | ±50 frames |
| `1`–`5` | set address, takeaway, top, impact, finish at this frame |
| Backspace / Delete | clear the label on this frame |
| `u` | undo last label change |
| `k` / `h` | toggle skeleton / help |
| Esc | save and quit |

## What needs labelling

Label **takeaway**, **top** and **impact**: the phases the detector is scored on. Address and
finish are gradual events defined by threshold (see [phase_definitions.md](phase_definitions.md));
you may label them for reference, but they are not scored.

If the golfer **pauses at the top**, any frame in the pause is correct. Record it as a range by
editing the label JSON, for example `"ranges": {"top": [190, 250]}`; the scorer then counts any
detected frame inside the range as zero error. (The label tool marks single frames only.)

## Definitions

Work with +-1 frame (about 4 ms at 240 fps) in mind: pick a rule and apply it the same way each time.

1. **address**: the last frame where everything is still, just before the swing starts.
2. **takeaway**: **the start of the swing**: the first frame where the club or body clearly moves
   away from the set position. This is not the shaft-parallel position; shaft-parallel (the end
   of the takeaway in coaching language, "toe-up" in swing datasets) is its own checkpoint and
   is not marked in this project yet.
3. **top**: the top of the backswing, where it stops and the downswing begins. If the golfer pauses,
   any frame in the pause is acceptable (see ranges above).
4. **impact**: the frame where the club head contacts the ball. If contact falls between two frames,
   take the later frame (the first where the ball has visibly moved, or the club is past the ball).
5. **finish**: the first frame where the follow-through is complete and the golfer holds a stable
   finish, even if the club is still settling.

Phases must be in this order with strictly increasing frame numbers; the tool flags violations
in red.

## Notes

- Fast frames near impact are motion-blurred. Step through them in ±1
  increments; do not rely on a single frame.
- Add a free-text `notes` field in the JSON by hand if a clip is unusual
  (mishit, partial swing, practice swing, occlusion).
