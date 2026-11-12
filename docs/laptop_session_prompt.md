# Handoff for a Claude Code session on the laptop (CPU only)

Paste everything below the line into a new session opened in the repo on the laptop.
Written after the GPU-computer session; it says where the project stands and what to
trust. Read it, then README.md and docs/rtmpose.md.

---

You are continuing work on SwingAnalyzer, a golf swing analysis portfolio project
(Python, OpenCV, pose estimation). This laptop has no GPU. The previous session ran on a
desktop with an RTX 4060 and finished the GPU runs, a bug fix, a three-backend
comparison and a hand-clicked wrist scoring. Everything is committed there; get those
commits first (see "Before anything else").

## Rules (unchanged, keep them)
- **Never invent metrics or results.** Every number in the README or docs must come from
  a script in this repo. Say plainly what is not validated.
- One labelled swing is a sanity check, not an accuracy result. Do not tune thresholds on
  this one clip. Only top and impact are scored against hand labels.
- Do not edit `data/labels/*.json` or `data/annotations/*.json`: they are the user's
  hand-made ground truth.
- No keypoint ground truth exists except the 78 wrist clicks (below). Compare backends on
  behaviour, and use the wrist clicks for wrist distance only.
- Pose confidence is unreliable on RTMPose. Check by eye with `scripts/compare_frames.py`.
- Run `pytest` before committing. Do not download substitute video.
- Run commands and launch tools yourself; the user only does the human steps (clicking,
  labelling). Never fill in ground truth yourself.
- Shell is PowerShell on Windows. `Activate.ps1` may be blocked by execution policy, so
  call `.\.venv\Scripts\swing.exe` and `.\.venv\Scripts\python.exe` directly.

## Before anything else
1. The GPU computer's commits may not be pushed (it had no GitHub credentials). Check
   `git log --oneline` for "Score wrists against hand-clicked wrists on 39 frames". If it
   is missing, ask the user to push from the GPU machine or copy the repo over.
2. `data/raw/IMG_6004.MOV` is gitignored; it should already be on this laptop.
3. `outputs/` and `*.npz` are gitignored. The GPU keypoints are not in git. Either copy
   them from the GPU machine (`outputs/IMG_6004.npz`, `IMG_6004_rtmpose_bal.npz`,
   `IMG_6004_rtmpose_perf.npz`) or regenerate; performance mode is about 48 minutes per
   clip on this CPU, balanced about 11 minutes, MediaPipe about 1.5 minutes.
4. CPU environment: install plain `onnxruntime` (the `rtmpose` extra). Do not install
   `onnxruntime-gpu` alongside it; they conflict. `--device cuda` fails loudly here, by
   design; use the default `--device cpu`.
5. Git has no identity configured on the GPU machine; if it is missing here too, pass
   `-c user.name=... -c user.email=...` per commit rather than changing config.

## Where the project stands
Milestones 0-5 done on one clip, `data/raw/IMG_6004.MOV` (down-the-line, 1551 frames,
~239.5 fps, right-handed golfer), hand-labelled in `data/labels/IMG_6004.json`. Milestone
6 (Streamlit demo) and multi-clip validation are not built. Details and numbers are in
docs/rtmpose.md (the single source; do not copy numbers around).

What happened in the GPU session:
- **GPU setup works** (onnxruntime-gpu 1.22.0 + cuDNN 9.10.2.21, see "GPU setup" in
  docs/rtmpose.md). The RTMPose backend verifies it really runs on the GPU and raises on
  silent CPU fallback. Irrelevant on this laptop unless you use `--device cuda`.
- **Frame-seek bug found and fixed** (`iter_frames(start>0)` landed ~50 frames late on
  this file). The old laptop wrist notes ("wrists on the torso, hands high") were read off
  misaligned images and are **withdrawn**; do not rely on them.
- **Three backends compared** (MediaPipe, RTMPose balanced, RTMPose performance) on
  phases, metrics and wrists.
- **New tool:** `swing annotate-wrists` plus `scripts/score_wrists.py`. The user clicked
  the wrists on 39 frames (every 10th, 731-1111) into
  `data/annotations/IMG_6004_wrists.json`. Scored against those clicks over the visible
  ones, performance mode is closest, then balanced, then MediaPipe (the MediaPipe gap is
  the clearer one; the balanced-versus-performance gap is small). One swing, one
  annotator, hidden-hand clicks are guesses. See docs/rtmpose.md for the table and caveats.
- Timings: GPU makes performance mode practical (about 3 minutes versus 48 on CPU).
  Results are essentially the same on CPU and GPU; balanced gave identical phases and spine
  angles within 0.06 degrees.

## Open items (the user has not chosen the next one)
1. **Streamlit demo (milestone 6).** A web page: upload or pick a clip, choose a backend,
   show overlay, phases, metrics (labelled unvalidated). On this laptop only MediaPipe
   (and balanced, slowly) is practical for live uploads. The simple option is a
   precomputed single-swing demo that loads saved keypoints. Ask which version they want.
   The raw video and keypoints are gitignored; a public demo needs a clip they agree to
   share. Do not add substitute footage.
2. **More footage and labels.** One swing is the project's main limit. A face-on clip would
   also show whether the hidden-hand frames (about 1015-1035) are just this camera angle.
   Use `swing label` for phases and `swing annotate-wrists` for wrists.
3. **Unexplained:** frame 921 right wrist is 53-62 px from the click for all three backends.
4. Nothing is pushed. Ask before pushing; GitHub credentials are the user's to set up.

If something is ambiguous, ask the user instead of guessing.
