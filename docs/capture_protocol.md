# Capture protocol

Consistent capture matters more than any model choice. Pose error from a bad
angle or motion blur will swamp everything downstream. Follow this for every
clip so results are comparable across sessions.

Scope: a **right-handed golfer** (lead side = left). Two views per swing.

## Equipment

- iPhone, rear main camera, on a tripod or fixed mount. Never hand-held.
- Slo-mo mode: **240 fps at 1080p** preferred, 120 fps acceptable. Below ~100 fps
  the impact window is too coarse for phase timing.
- Good, even daylight or bright indoor light. Fast shutter needed at 240 fps, so
  dim light means noisy frames and worse keypoints.

## Camera positions

Camera height ≈ hip height (about 0.9–1.0 m), lens parallel to the ground,
phone in **portrait** (the golfer is tall; portrait wastes less frame).
Keep the zoom at 1x and move the tripod instead.

| View | Position | Distance |
|---|---|---|
| **Down-the-line (DTL)** | Behind the golfer, on the extension of the target line through the ball, looking toward the target | ~3 m from the ball |
| **Face-on (FO)** | Square to the target line, facing the golfer's chest at address, roughly level with the ball | ~3 m from the ball |

The whole body plus club head at the top of the backswing and in the finish must
stay in frame for the entire swing. Leave margin; you can crop later but you
cannot recover cut-off limbs.

Mark the tripod and ball positions (tape or a tee) so later sessions repeat the
geometry. Note the distances in the clip's metadata (below).

## Clothing and background

- Fitted, contrasting clothing. Avoid baggy tops that hide the hips and shoulders.
- Plain background if possible; avoid other people walking through the frame.
- Shoes visible (ankle keypoints).

## Recording procedure

1. Start recording, golfer settles into the address position and stays still ~1 s
   (this gives a clean address reference frame).
2. Golfer hits one shot. Hold the finish ~1 s.
3. Stop recording. One swing per clip.
4. Record DTL and FO as separate swings (a single camera cannot capture both at
   once). Do not assume the two swings are identical; do not pair them for
   cross-view measurements unless you capture them simultaneously with two phones.

## File naming and layout

Raw video lives in `data/raw/` and is **gitignored**.

```
data/raw/<YYYY-MM-DD>_<club>_<view>_<nn>.MOV
e.g. data/raw/2026-11-11_7i_dtl_01.MOV
```

`view` is `dtl` or `fo`. Hand-labelled phase frames go in `data/labels/` (one
JSON per clip, added in milestone 3).

## Export notes

- AirDrop or copy the original `.MOV`; avoid "Optimize"/re-encoded exports.
- iPhone slo-mo is stored with a time-ramp (normal speed → slow-mo → normal).
  Make sure the clip is trimmed in Photos so the **whole swing falls inside the
  full-rate segment**, or the playback timeline will not match real time.
- iPhone clips are usually HEVC. If OpenCV cannot decode a clip, convert it:

  ```
  ffmpeg -i in.MOV -c:v libx264 -crf 14 -an out.mp4
  ```

- Check what OpenCV sees before analysing:

  ```
  swing info data/raw/<clip>.MOV
  ```

  The reported fps should be 120 or 240. If it says 30, the clip was exported at
  normal speed or the ramp has been flattened; real-time metrics such as tempo
  would be wrong, so re-export it.

## Clips that lose their real timeline

Everything time-based (phases, tempo, smoothing windows) assumes the file's frame
timestamps are real time. A slo-mo clip can be saved or exported **re-timed to play slowly**,
for example 240 fps footage stamped as 30 fps. Then the swing takes seconds instead of
about one second, and `swing info` reports about 30 fps even though the pixels came from a
240 fps capture.

How to tell: `swing info <clip>` reports a frame rate below 100, or the whole clip is far
longer than the swing (a backswing that takes several seconds is the giveaway).

What the project does about it:

- Phase detection **refuses to run below 100 fps** and says so, rather than returning
  numbers (what happened on the first re-timed upload: top and impact on the same frame).
- If you know the capture rate, give it: `swing phases clip.npz --fps 240`,
  `swing metrics clip.npz --view dtl --fps 240`, or the demo's **Capture frame rate**
  control. The keypoints are per frame, so only the time base changes. This is your claim
  about the clip and cannot be checked from the file: if it is wrong, every time and tempo
  is wrong.
- Better: keep the original file. AirDrop or copy the original `.MOV` from Photos rather
  than an exported or shared copy, and re-run `swing info` on it.

## Known limitations

- A single camera gives a 2D projection. DTL and FO angles are image-plane
  angles, not true joint angles. Rotation is reported as a labelled 2D proxy.
- Camera not exactly on the target line, or tilted, biases every angle.
