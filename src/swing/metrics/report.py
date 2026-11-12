"""Metric definitions and the view-aware report.

Conventions
-----------
* ``view`` is ``"dtl"`` (down-the-line) or ``"fo"`` (face-on).
* Distances are in torso lengths (projected hip-mid to shoulder-mid length at
  address, 2D), so they do not depend on resolution or camera distance.
* Directions come from the data at address, not from assumptions about which
  side the camera is on:
    - toward-ball (dtl): the direction from the hips to the hands.
    - toward-target (fo): the image direction of the golfer's left hip from
      the right hip (right-handed golfer: lead side = left = target side).
* Phase frames address/takeaway/finish are threshold-defined (see
  docs/phase_definitions.md); every metric that uses them inherits that.
* No pass/fail thresholds. A value like ``hip_toward_ball_impact_torso`` is a
  measurement, not a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from swing.metrics import geometry as g
from swing.phases.detect import PhaseResult
from swing.pose.base import PoseSequence

VIEWS = ("dtl", "fo")
MIN_DIRECTION_TORSO = 0.05  # direction cue must exceed this to be trusted


@dataclass
class MetricReport:
    view: str
    values: dict[str, float | None] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)  # name -> reason

    def put(self, name: str, value) -> None:
        v = None if value is None or not np.isfinite(value) else float(value)
        if v is None:
            self.unavailable[name] = "keypoints missing at the needed frames"
        self.values[name] = v

    def skip(self, names, reason: str) -> None:
        for n in names:
            self.values[n] = None
            self.unavailable[n] = reason


def tempo(frames: dict[str, int | None], fps: float) -> dict[str, float | None]:
    """Backswing (takeaway to top), downswing (top to impact), and their ratio.

    Depends on the takeaway rule; quote the rule with the number.
    """
    t, top, imp = frames.get("takeaway"), frames.get("top"), frames.get("impact")
    if None in (t, top, imp) or not (t < top < imp):
        return {"backswing_s": None, "downswing_s": None, "tempo_ratio": None}
    b, d = (top - t) / fps, (imp - top) / fps
    return {"backswing_s": b, "downswing_s": d, "tempo_ratio": b / d}


def _direction(sign_value: float, torso: float) -> float | None:
    if not np.isfinite(sign_value) or abs(sign_value) < MIN_DIRECTION_TORSO * torso:
        return None
    return float(np.sign(sign_value))


def compute_metrics(seq: PoseSequence, phases: PhaseResult, view: str) -> MetricReport:
    """Compute all metrics that are meaningful for ``view`` from ``seq`` (smoothed)."""
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}")
    rep = MetricReport(view)
    k, fps = seq.keypoints, seq.fps
    fr = {p: (None if f is None else f - seq.start_frame) for p, f in phases.frames.items()}

    note = f" ({phases.warnings[0]})" if phases.warnings else ""
    for name, v in tempo(fr, fps).items():
        if v is None:
            rep.skip([name], "needs takeaway, top and impact, in that order" + note)
        else:
            rep.put(name, v)

    spine_names = ["spine_angle_address_deg", "spine_angle_top_deg",
                   "spine_angle_impact_deg", "spine_angle_change_to_impact_deg"]
    needed = [fr.get("address"), fr.get("top"), fr.get("impact")]
    if None in needed:
        rep.skip(spine_names, "needs address, top and impact" + note)
        view_names = {"dtl": ["hip_toward_ball_top_torso", "hip_toward_ball_impact_torso"],
                      "fo": ["hip_slide_top_torso", "hip_slide_impact_torso",
                             "head_sway_top_torso", "head_sway_impact_torso",
                             "head_sway_max_excursion_torso",
                             "shoulder_turn_proxy_top_deg", "hip_turn_proxy_top_deg"]}[view]
        rep.skip(view_names, "needs address, top and impact" + note)
        return rep
    a, top, imp = needed

    ref = g.reference_pose(k, a, fps)
    torso = g.torso_length(ref)
    if not np.isfinite(torso) or torso <= 0:
        rep.skip(spine_names, "torso not visible at address")
        return rep

    spine = g.spine_angle_deg(k)
    sa = float(g.spine_angle_deg(ref)[0])
    rep.put(spine_names[0], sa)
    rep.put(spine_names[1], spine[top])
    rep.put(spine_names[2], spine[imp])
    rep.put(spine_names[3], spine[imp] - sa)

    hips = g.mid(k, "left_hip", "right_hip")
    hip0 = g.mid(ref, "left_hip", "right_hip")[0]

    if view == "dtl":
        hands0 = g.mid(ref, "left_wrist", "right_wrist")[0]
        s = _direction(hands0[0] - hip0[0], torso)
        names = ["hip_toward_ball_top_torso", "hip_toward_ball_impact_torso"]
        if s is None:
            rep.skip(names, "cannot tell which way the ball is (hands not clearly ahead of hips at address)")
        else:
            rep.put(names[0], s * (hips[top, 0] - hip0[0]) / torso)
            rep.put(names[1], s * (hips[imp, 0] - hip0[0]) / torso)
        rep.skip(["hip_slide_top_torso", "hip_slide_impact_torso", "head_sway_top_torso",
                  "head_sway_impact_torso", "head_sway_max_excursion_torso",
                  "shoulder_turn_proxy_top_deg", "hip_turn_proxy_top_deg"],
                 "not measurable from a down-the-line view; use a face-on clip")
        return rep

    # face-on
    s = _direction(g.pt(ref, "left_hip")[0, 0] - g.pt(ref, "right_hip")[0, 0], torso)
    head = g.head_point(k)
    head0 = g.head_point(ref)[0]
    slide_names = ["hip_slide_top_torso", "hip_slide_impact_torso"]
    sway_names = ["head_sway_top_torso", "head_sway_impact_torso", "head_sway_max_excursion_torso"]
    if s is None:
        rep.skip(slide_names + sway_names, "cannot tell which way the target is (hips not clearly side by side)")
    else:
        rep.put(slide_names[0], s * (hips[top, 0] - hip0[0]) / torso)
        rep.put(slide_names[1], s * (hips[imp, 0] - hip0[0]) / torso)
        dx = s * (head[:, 0] - head0[0]) / torso
        rep.put(sway_names[0], dx[top])
        rep.put(sway_names[1], dx[imp])
        lo = fr.get("takeaway") if fr.get("takeaway") is not None else a
        seg = dx[lo:imp + 1]
        rep.put(sway_names[2], np.nanmax(np.abs(seg)) if np.isfinite(seg).any() else np.nan)
    sw0 = float(g.width(ref, "left_shoulder", "right_shoulder")[0])
    hw0 = float(g.width(ref, "left_hip", "right_hip")[0])
    rep.put("shoulder_turn_proxy_top_deg",
            g.arccos_ratio_deg(g.width(k, "left_shoulder", "right_shoulder"), sw0)[top])
    rep.put("hip_turn_proxy_top_deg",
            g.arccos_ratio_deg(g.width(k, "left_hip", "right_hip"), hw0)[top])
    return rep
