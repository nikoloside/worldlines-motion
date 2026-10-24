"""Deterministic procedural test clips: no model file, no network.

These are *parameters* for format and timing tests (walk in, squat, startle
jump), not captured human motion. ``wlmotion procedural walk_squat out.npz``.
"""

from __future__ import annotations

import math

import numpy as np

from ..clip import new_clip, smoothstep

FPS = 30
CASES = {"walk_squat": 6.0, "startle_jump": 4.0, "walk_startle": 5.0}


def ramp(t: float, a: float, b: float) -> float:
    return smoothstep((t - a) / (b - a))


def make_clip(case: str, fps: int = FPS) -> dict:
    """SMPL-X axis-angle controls in a Y-up world, facing +X while walking."""
    if case not in CASES or fps <= 0:
        raise ValueError(f"case must be one of {tuple(CASES)}, fps > 0")
    count = round(CASES[case] * fps)
    transl = np.zeros((count, 3), dtype=np.float32)
    global_orient = np.zeros((count, 3), dtype=np.float32)
    body_pose = np.zeros((count, 21, 3), dtype=np.float32)
    walking = case in ("walk_squat", "walk_startle")
    stop = 2.4 if case == "walk_squat" else 2.0
    for i in range(count):
        t = i / fps
        gait = (1.0 - ramp(t, stop - 0.3, stop)) if walking else 0.0
        cycle = 2.0 * math.pi * t / 0.8
        transl[i, 0] = -2.5 + 2.5 * ramp(t, 0, stop) if walking else 0.0
        global_orient[i, 1] = math.pi / 2 if walking else 0.0
        if case == "walk_squat":
            squat = ramp(t, 2.6, 3.3) * (1.0 - ramp(t, 4.15, 4.95))
            jump = alarm = 0.0
        else:
            trigger = 1.55 if case == "startle_jump" else 2.15
            anticipation = ramp(t, trigger - 0.18, trigger) * (1.0 - ramp(t, trigger, trigger + 0.12))
            u = (t - trigger) / 0.68
            jump = 0.32 * math.sin(math.pi * u) if 0.0 <= u <= 1.0 else 0.0
            squat = 0.15 * anticipation
            alarm = ramp(t, trigger - 0.05, trigger + 0.08) * (1.0 - ramp(t, trigger + 0.7, trigger + 1.35))
        transl[i, 1] = -0.39 * squat + jump
        # body_pose joints: 0/1 hips, 3/4 knees, 6/7 ankles, 15/16 shoulders, 17/18 elbows
        for side, phase, hip, knee, ankle, shoulder, elbow in ((1.0, 0.0, 0, 3, 6, 15, 17), (-1.0, math.pi, 1, 4, 7, 16, 18)):
            stride = math.sin(cycle + phase) * gait
            body_pose[i, hip, 0] = -1.05 * squat + 0.26 * stride
            body_pose[i, knee, 0] = 2.10 * squat + max(0.0, -stride) * 0.35
            body_pose[i, ankle, 0] = -1.05 * squat - max(0.0, -stride) * 0.16
            body_pose[i, shoulder, 0] = -0.30 * stride - 0.50 * alarm
            body_pose[i, shoulder, 2] = -side * 0.90 * alarm
            body_pose[i, elbow, 0] = -0.24 * alarm
        body_pose[i, 2, 0] = 0.20 * squat   # spine1 forward balance
        body_pose[i, 11, 0] = 0.12 * alarm  # neck response
    clip = new_clip(count, fps, source="procedural_test_only")
    clip["transl"], clip["global_orient"], clip["body_pose"] = transl, global_orient, body_pose.reshape(count, 63)
    clip["case"] = case
    return clip
