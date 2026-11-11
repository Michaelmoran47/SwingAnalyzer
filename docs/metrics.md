# Metrics v1

Command: `swing metrics outputs/<clip>.npz --view dtl|fo [--json out.json]`
(input is the **raw** keypoints `.npz`; smoothing and phase detection are applied
first). Implementation: `src/swing/metrics/`. Each metric is a pure function of
keypoints (plus phase frames) and is unit-tested on synthetic skeletons with
known answers, including mirrored cameras.

**Status: implemented and unit-tested; not validated against any reference
measurement.** Values from a real clip show the pipeline runs and look
plausible, nothing more. There are no pass/fail thresholds: every number is a
measurement, not a verdict.

## Which view each metric needs

A single camera sees one plane, so many metrics only make sense from one view.
Asking for a metric that does not apply returns "unavailable" with the reason.

| Metric | dtl | fo |
|---|---|---|
| tempo (`backswing_s`, `downswing_s`, `tempo_ratio`) | yes | yes |
| spine angle (address, top, impact, change to impact) | forward bend | lateral tilt |
| `hip_toward_ball_{top,impact}_torso` (early-extension indicator) | yes | no |
| `hip_slide_{top,impact}_torso` | no | yes |
| `head_sway_{top,impact,max_excursion}_torso` | no | yes |
| `shoulder_turn_proxy_top_deg`, `hip_turn_proxy_top_deg` | no | yes |

## Definitions

Conventions: image coordinates (y down). Distances are in **torso lengths**: the
projected hip-mid to shoulder-mid length of the reference pose, which is the
median pose over the 0.2 s ending at the address frame. Directions come from the
keypoints at address, not from assumptions about where the camera is.

- **Tempo**: backswing = takeaway→top, downswing = top→impact, ratio = backswing /
  downswing. Takeaway is threshold-defined ([phase_definitions.md](phase_definitions.md)),
  so the ratio depends on that rule; quote the rule with the number.
- **Spine angle**: angle of the hip-mid→shoulder-mid line from vertical, in the
  image plane, degrees, unsigned.
- **hip_toward_ball**: hip-mid horizontal displacement from the reference, positive
  toward the ball. "Toward the ball" is the direction from the hips to the hands
  at address. A continuous indicator related to early extension, not a detector
  of it; combine with the spine-angle change to impact when reading it. If the
  hands are not clearly ahead of the hips the metric is unavailable.
- **hip_slide**: same displacement, positive toward the target. For a right-handed
  golfer facing the camera the target is the image direction from the right hip to
  the left hip. Unavailable if the hips are not clearly side by side.
- **head_sway**: displacement of the ear midpoint (both ears required), positive
  toward the target; `max_excursion` is the largest absolute value from takeaway
  to impact.
- **Rotation proxies**: `arccos(projected width / width at address)` for the
  shoulders and hips. **A 2D proxy, not a rotation angle.** It assumes the
  segment was square to the camera at address and turns about the vertical axis,
  is unsigned, and cannot distinguish past 90 degrees. Treat as an indicator of
  how much the segment narrowed.

## Known limits

- Everything is 2D image-plane geometry. Camera tilt or a camera not on the
  intended line biases every value.
- Down-the-line hip keypoints are noisy because the near hip hides the far one;
  `hip_toward_ball` inherits that, and its error is unmeasured.
- Phase frames come from the threshold rules for address/takeaway/finish. Only
  top and impact have been compared to hand labels, on one swing.
- Torso length is a 2D projection; it shrinks if the golfer bends toward or away
  from the camera.
