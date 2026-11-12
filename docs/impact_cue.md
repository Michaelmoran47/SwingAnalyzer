# Impact from the ball

Wrist tracking cannot see the club, so "lowest hands" can sit several frames away from contact.
On the slow-motion recording the hands stall for about 6 frames while the club swings through,
and the hand-based impact came out 6 frames early. The ball is a better cue: it sits still until
impact and is gone afterwards. Code: `src/swing/impact_cue.py`; try it with
`swing phases <keypoints.npz> --video <clip> [--fps N]` or the demo's *Find impact from the ball*
option (on by default).

## How it works

1. **Search region.** Near the feet, extending toward the ball (the side the hands are on at
   address) out to about 3 torso lengths. The ball was 2.1 torso lengths away on the recording.
2. **Clean background.** The median of the second half of the backswing: the club is far from
   the ball but the ball is still on the tee. At address the club head touches the ball and merges
   with it into one odd-shaped blob, so address frames cannot be used.
3. **Candidates.** Small bright round blobs, 0.04 to 0.14 torso lengths across, found with a
   top-hat filter. A blob's extent is taken at half its own peak response (a fixed threshold let
   the ball merge with bright grass into one huge region on both clips).
4. **Which blob is the ball.** Each candidate is compared, frame by frame, with two reference
   images: *ball there* (the background) and *bare grass* (the same patch with the ball painted
   out). The ball looks like the first before the swing and like the second at the end. A white
   shoe, an alignment stick or the tee does not turn into bare grass, and a leg crossing a spot
   matches neither. The patch is **tracked** from frame to frame (3 px per frame), because a
   hand-held or panning camera drifted the picture by about 10 px over one swing on the recording.
5. **Impact** = the first frame, from the top on, where the patch stops looking like *ball there*
   and never does again. That is the club arriving at the ball, a frame or so before the ball is
   clearly gone, which matches the labelling guide's "later frame" rule closely in the cases
   checked.
6. **Safety net.** The answer is used only if it is after the top, within 60 % of the downswing
   length (in frames) of the hand-based frame, and the corrected phases make sense. When the
   timings look like a real-time swing that includes the finish coming after impact and plausible
   durations. On a clip whose timeline looks slowed the finish and the durations are unreliable,
   so only the frame order address < takeaway < top < impact is required (the cue works on
   frames, not seconds) and a finish that falls before the corrected impact is dropped rather
   than shown. Otherwise the hand-based impact stays and the page says why.

## What it did on the three clips (one swing each)

| Clip | Your label | Hand-based | Ball cue |
|---|---|---|---|
| `IMG_6004` (240 fps) | 995 | 999 (+4) | 995 (0) |
| Recording (slow-motion screen capture; same at its stamped 30 fps and at an assumed 240) | 276 | 270 (-6) | 276 (0) |
| `IMG_6003` (re-timed, 240 fps assumed) | none | 1472 | 1467 |

For `IMG_6003` there is no label. Looking at the frames: at 1462-1466 the club approaches and
arrives behind the ball, at 1467 the club face meets the ball, at 1468 the ball is gone (only the
red mark's smear is left). The cue's 1467 agrees; the hand-based 1472 was about 5 frames after the
ball had left. That is a by-eye check on one clip.

**Read these with care.** The two labelled clips are not independent tests: four real bugs were
found and fixed while looking at exactly those clips (a cap on the candidates examined, merged
blobs, "changed" versus "gone", camera drift). `IMG_6003` is the closest thing to a held-out
check, and it is one swing judged by eye. An exact match on two clips says the method can work,
not how often it will.

## When it cannot work, and what it does then

It returns nothing, and impact stays hand-based, with the reason shown, when there is no ball
(indoor net, simulator, hidden or out of frame), no round blob passes the vanishing test, the
camera moves more than about 3 px between frames, or the ball cue contradicts the hand-based
frame. Known weak spots:

- **Something mid-gray covering the spot.** "Bare grass" is accepted within 40 % of the
  ball-versus-grass difference, so a cover close to the grass's gray level could be mistaken for
  the ball having left. White trousers and a dark shoe are rejected in the tests; mid-gray cloth
  was not tested on real footage.
- **A very dark or very light surface** where the ball has little contrast.
- **A ball struck out of the search region**, for example a driver teed far from the feet.
- **Heavy motion blur at low frame rates**, which was not tested (all three clips are slow motion).
- The cue reads the video once (tens of seconds); the answer is cached in `outputs/.cue_cache/`.
