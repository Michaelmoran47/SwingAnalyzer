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

## Definitions

Work with ±1 frame (about 4 ms at 240 fps) in mind: pick a rule and apply it
the same way each time.

1. **address**: the last frame where the golfer and club are still, just before
   the takeaway begins.
2. **takeaway**: the first frame where the club or hands visibly start moving
   away from the ball.
3. **top**: the frame where the backswing stops and the downswing begins. The
   hands reach their highest point and reverse direction.
4. **impact**: the frame where the club head contacts the ball. If contact
   falls between two frames, take the later frame (the first where the ball
   has visibly moved, or the club is past the ball position).
5. **finish**: the first frame where the follow-through is complete and the
   golfer holds a stable finish position (hands have stopped rising).

Phases must be in this order with strictly increasing frame numbers; the tool
flags violations in red.

## Notes

- Fast frames near impact are motion-blurred. Step through them in ±1
  increments; do not rely on a single frame.
- Add a free-text `notes` field in the JSON by hand if a clip is unusual
  (mishit, partial swing, practice swing, occlusion).
