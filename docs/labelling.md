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

Label **top** and **impact**. They are the phases the detector is scored on.
Address, takeaway and finish are gradual events that the detector defines by
threshold (see [phase_definitions.md](phase_definitions.md)); you may still
label them for reference, but they are not scored.

## Definitions

**All phases are defined by the hands (the grip / wrists), not the club head or
the body.** The pose model tracks wrists, not the club, and the detector uses
the wrist trajectory, so the labels have to mean the same thing. Watch the
golfer's hands and ignore the club head, even when it moves first.

Work with ±1 frame (about 4 ms at 240 fps) in mind: pick a rule and apply it
the same way each time.

1. **address**: the last frame where the hands are still before the backswing
   begins. Slow creeping counts as moving; if the hands are visibly drifting,
   you are past address.
2. **takeaway**: the first frame where the hands have clearly moved away from
   their address position. The club head will already be moving; ignore it.
3. **top**: the frame where the hands reach their highest point (smallest
   height in the image) before the downswing.
4. **impact**: the frame where the club head contacts the ball. If contact falls
   between two frames, take the later frame. (The one phase read from the ball,
   because the hands' position there is only a proxy.)
5. **finish**: the first frame where the hands stop (no longer rising or
   drifting) in the follow-through, even if the club and body are still settling.

Phases must be in this order with strictly increasing frame numbers; the tool
flags violations in red.

## Notes

- Fast frames near impact are motion-blurred. Step through them in ±1
  increments; do not rely on a single frame.
- Add a free-text `notes` field in the JSON by hand if a clip is unusual
  (mishit, partial swing, practice swing, occlusion).
