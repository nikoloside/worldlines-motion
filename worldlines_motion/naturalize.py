"""Motion clean-up passes that make recovered or generated SMPL-X clips read as natural.

Each pass is a pure function ``clip -> clip``; ``naturalize`` chains them.
The passes are deliberately conservative and reversible from the source
clip. Defaults are tuned for 30 fps single-person clips.

Passes
  smooth_rotations   quaternion-space Gaussian smoothing with velocity-aware width
  smooth_root        root translation smoothing (lighter than the rotations)
  joint_limits       clamp swing angles to anatomical ranges (knees/elbows hinge)
  ground             put the lowest foot joint on y=0 over the clip, remove per-frame sinking
  foot_lock          detect foot contacts, shift root so planted feet don't slide
  settle_ends        ease into / out of the clip (first & last frames reach rest velocity)
  breathing          tiny spine/chest oscillation so held poses are not frozen
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np

from .clip import (BODY_JOINT_NAMES, copy_clip, full_pose, quat_conj, quat_mul, quat_to_rotvec,
                   rotvec_to_quat, validate)
from .smplx_model import SMPLX

J = {name: i for i, name in enumerate(("pelvis",) + BODY_JOINT_NAMES)}  # 0..21 joint index
B = {name: i - 1 for name, i in J.items() if i > 0}  # body_pose index (0..20)

# Anatomical ranges for body joints as (min, max) per local axis in radians.
# SMPL-X local axes: X = flexion (hinge for knees/elbows is mostly X for knees, Y for elbows), see notes.
# Knees: X positive = flexion (bend back). Elbows: Y (left negative, right positive) = flexion.
JOINT_LIMITS: dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {
    "left_knee": ((-0.15, 2.6), (-0.35, 0.35), (-0.35, 0.35)),
    "right_knee": ((-0.15, 2.6), (-0.35, 0.35), (-0.35, 0.35)),
    "left_elbow": ((-0.6, 0.6), (-2.7, 0.15), (-0.6, 0.6)),
    "right_elbow": ((-0.6, 0.6), (-0.15, 2.7), (-0.6, 0.6)),
    "left_hip": ((-2.2, 1.0), (-0.9, 0.9), (-0.6, 1.2)),
    "right_hip": ((-2.2, 1.0), (-0.9, 0.9), (-1.2, 0.6)),
    "left_ankle": ((-0.8, 0.9), (-0.5, 0.5), (-0.5, 0.5)),
    "right_ankle": ((-0.8, 0.9), (-0.5, 0.5), (-0.5, 0.5)),
    "left_foot": ((-0.4, 0.6), (-0.2, 0.2), (-0.2, 0.2)),
    "right_foot": ((-0.4, 0.6), (-0.2, 0.2), (-0.2, 0.2)),
    "spine1": ((-0.5, 0.7), (-0.5, 0.5), (-0.45, 0.45)),
    "spine2": ((-0.4, 0.5), (-0.5, 0.5), (-0.4, 0.4)),
    "spine3": ((-0.4, 0.5), (-0.5, 0.5), (-0.4, 0.4)),
    "neck": ((-0.8, 0.9), (-1.0, 1.0), (-0.6, 0.6)),
    "head": ((-0.8, 0.8), (-1.0, 1.0), (-0.6, 0.6)),
    "left_collar": ((-0.4, 0.4), (-0.5, 0.5), (-0.5, 0.5)),
    "right_collar": ((-0.4, 0.4), (-0.5, 0.5), (-0.5, 0.5)),
    "left_wrist": ((-1.2, 1.2), (-1.2, 1.2), (-1.4, 1.4)),
    "right_wrist": ((-1.2, 1.2), (-1.2, 1.2), (-1.4, 1.4)),
}


@dataclass
class NaturalizeConfig:
    smooth_sigma: float = 1.4          # frames at 30 fps; scaled with fps
    root_sigma: float = 2.0
    velocity_adaptive: bool = True     # less smoothing where the joint moves fast (keeps impacts)
    joint_limits: bool = True
    limit_softness: float = 0.12       # radians of soft knee before the hard clamp
    ground: bool = True
    ground_percentile: float = 5.0     # lowest-foot percentile that defines the floor
    foot_lock: bool = True
    contact_height: float = 0.06       # m above floor
    contact_speed: float = 0.35        # m/s
    foot_lock_strength: float = 0.9
    settle_ends: bool = True
    settle_frames: int = 6
    breathing: float = 0.0             # 0 = off, 1 = visible chest rise (~1.5°)
    passes: list[str] = field(default_factory=lambda: [
        "smooth_rotations", "smooth_root", "joint_limits", "ground", "foot_lock", "settle_ends", "breathing"])

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict | str | None) -> "NaturalizeConfig":
        if data is None:
            return cls()
        if isinstance(data, str):
            data = json.loads(data)
        allowed = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**allowed)


# ---- helpers ---------------------------------------------------------------------------

def _gauss_kernel(sigma: float) -> np.ndarray:
    if sigma <= 0.05:
        return np.array([1.0])
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _smooth_1d(x: np.ndarray, sigma: float, weights: np.ndarray | None = None) -> np.ndarray:
    """Gaussian smoothing along axis 0 with edge replication. Optional per-frame weight (0..1 = keep raw)."""
    k = _gauss_kernel(sigma)
    if len(k) == 1:
        return x.copy()
    r = len(k) // 2
    pad = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    out = np.zeros_like(x, dtype=np.float64)
    for i, w in enumerate(k):
        out += w * pad[i:i + len(x)]
    if weights is not None:
        w = np.asarray(weights, np.float64).reshape(-1, *([1] * (x.ndim - 1)))
        out = out * (1 - w) + x * w
    return out


def _unwrap_quats(q: np.ndarray) -> np.ndarray:
    """Make consecutive quaternions share hemisphere so averaging is meaningful."""
    q = q.copy()
    for i in range(1, len(q)):
        if np.sum(q[i] * q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def smooth_rotvec_series(rv: np.ndarray, sigma: float, keep: np.ndarray | None = None) -> np.ndarray:
    """rv [T,3] axis-angle -> smoothed, via normalised quaternion averaging."""
    q = _unwrap_quats(rotvec_to_quat(rv))
    s = _smooth_1d(q, sigma, keep)
    s /= np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-9)
    return quat_to_rotvec(s)


def _fps_scale(clip: dict) -> float:
    return float(clip["fps"]) / 30.0


# ---- passes ----------------------------------------------------------------------------

def smooth_rotations(clip: dict, cfg: NaturalizeConfig, model: SMPLX | None = None) -> dict:
    n = validate(clip)
    out = copy_clip(clip)
    sigma = cfg.smooth_sigma * _fps_scale(clip)
    if sigma <= 0.05 or n < 3:
        return out
    pose = full_pose(clip).astype(np.float64)  # [T,55,3]
    smoothed = pose.copy()
    for j in range(22):
        keep = None
        if cfg.velocity_adaptive:
            q = _unwrap_quats(rotvec_to_quat(pose[:, j]))
            d = quat_to_rotvec(quat_mul(q[1:], quat_conj(q[:-1])))
            speed = np.linalg.norm(d, axis=1) * float(clip["fps"])  # rad/s
            speed = np.concatenate([speed[:1], speed])
            # 0 rad/s -> full smoothing, >= 8 rad/s -> ~70% raw (keeps hits/jumps crisp)
            keep = np.clip(speed / 8.0, 0, 1) * 0.7
            keep = _smooth_1d(keep, 1.0)
        smoothed[:, j] = smooth_rotvec_series(pose[:, j], sigma, keep)
    out["global_orient"] = smoothed[:, 0].astype(np.float32)
    out["body_pose"] = smoothed[:, 1:22].reshape(n, 63).astype(np.float32)
    return out


def smooth_root(clip: dict, cfg: NaturalizeConfig, model: SMPLX | None = None) -> dict:
    out = copy_clip(clip)
    sigma = cfg.root_sigma * _fps_scale(clip)
    if sigma > 0.05 and len(out["transl"]) > 2:
        out["transl"] = _smooth_1d(np.asarray(clip["transl"], np.float64), sigma).astype(np.float32)
    return out


def _soft_clamp(x: np.ndarray, lo: float, hi: float, soft: float) -> np.ndarray:
    """Hard range [lo,hi] with a smooth approach over `soft` radians."""
    if soft <= 0:
        return np.clip(x, lo, hi)
    y = x.copy()
    over = x > hi - soft
    y[over] = hi - soft + soft * np.tanh((x[over] - (hi - soft)) / soft)
    under = x < lo + soft
    y[under] = lo + soft - soft * np.tanh(((lo + soft) - x[under]) / soft)
    return y


def joint_limits(clip: dict, cfg: NaturalizeConfig, model: SMPLX | None = None) -> dict:
    n = validate(clip)
    out = copy_clip(clip)
    bp = np.asarray(out["body_pose"], np.float64).reshape(n, 21, 3)
    for name, limits in JOINT_LIMITS.items():
        k = B[name]
        for axis in range(3):
            lo, hi = limits[axis]
            bp[:, k, axis] = _soft_clamp(bp[:, k, axis], lo, hi, cfg.limit_softness)
    out["body_pose"] = bp.reshape(n, 63).astype(np.float32)
    return out


FOOT_JOINTS = (J["left_ankle"], J["left_foot"], J["right_ankle"], J["right_foot"])  # heel/toe pairs


def _foot_points(clip: dict, model: SMPLX) -> np.ndarray:
    """[T,4,3] left heel, left toe, right heel, right toe in world space."""
    rest = model.shaped(clip["betas"])
    joints = model.joints(rest, full_pose(clip), clip["transl"])
    return joints[:, list(FOOT_JOINTS)]


def _sole_offsets(clip: dict, model: SMPLX) -> np.ndarray:
    """Rest height of each foot point (L ankle, L toe, R ankle, R toe) above the model's sole."""
    rest = model.shaped(clip["betas"])
    j = model.joints_of(rest)
    sole = rest[:, 1].min()
    return np.array([j[k, 1] - sole for k in FOOT_JOINTS], np.float64)


def _floor_offset(clip: dict, model: SMPLX) -> float:
    return float(_sole_offsets(clip, model).min())


def _lowest_and_speed(clip: dict, model: SMPLX) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame: height of each foot point above y=0 ([T,4]), their horizontal speed ([T,4]), lowest point ([T])."""
    feet = _foot_points(clip, model).astype(np.float64)
    h = feet[:, :, 1] - _sole_offsets(clip, model)[None, :]
    vel = np.linalg.norm(np.gradient(feet, axis=0)[:, :, [0, 2]], axis=2) * float(clip["fps"])
    return h, vel, h.min(axis=1)


def estimate_floor(clip: dict, model: SMPLX, cfg: NaturalizeConfig | None = None) -> float:
    """Floor height: median of the lowest foot point over slow frames (planted feet define the floor)."""
    cfg = cfg or NaturalizeConfig()
    h, vel, low = _lowest_and_speed(clip, model)
    slow = vel[np.arange(len(h)), h.argmin(axis=1)] < cfg.contact_speed
    if slow.sum() >= max(3, len(h) // 10):
        return float(np.median(low[slow]))
    return float(np.percentile(low, cfg.ground_percentile))


def ground(clip: dict, cfg: NaturalizeConfig, model: SMPLX | None = None) -> dict:
    out = copy_clip(clip)
    if model is None:
        return out
    floor = estimate_floor(out, model, cfg)
    transl = np.asarray(out["transl"], np.float64)
    transl[:, 1] -= floor
    out["transl"] = transl.astype(np.float32)
    # never let a foot go through the floor; pull the root up gently where it would
    _, _, low = _lowest_and_speed(out, model)
    sink = np.minimum(low, 0.0)
    transl[:, 1] -= _smooth_1d(sink, 1.0 * _fps_scale(out))
    out["transl"] = transl.astype(np.float32)
    return out


def detect_contacts(clip: dict, model: SMPLX, height: float = 0.06, speed: float = 0.35, floor: float | None = None) -> np.ndarray:
    """[T,4] contact probability for L heel, L toe, R heel, R toe (heights relative to the clip's floor)."""
    h, vel, low = _lowest_and_speed(clip, model)
    if floor is None:
        floor = estimate_floor(clip, model)
    c_h = np.clip(1.0 - (h - floor) / height, 0, 1)
    c_v = np.clip(1.0 - vel / speed, 0, 1)
    c = np.sqrt(c_h * c_v)
    return _smooth_1d(c, 0.8 * float(clip["fps"]) / 30.0).astype(np.float32)


def contact_segments(contacts: np.ndarray, threshold: float = 0.5, min_len: int = 3) -> list[tuple[int, int, int]]:
    """(foot_point, start, end_exclusive) runs where contact probability stays above threshold."""
    segs = []
    n, k = contacts.shape
    for p in range(k):
        on = contacts[:, p] > threshold
        t = 0
        while t < n:
            if on[t]:
                s = t
                while t < n and on[t]:
                    t += 1
                if t - s >= min_len:
                    segs.append((p, s, t))
            else:
                t += 1
    return segs


def foot_lock(clip: dict, cfg: NaturalizeConfig, model: SMPLX | None = None) -> dict:
    """Remove foot sliding with a smooth root correction.

    Offline, segment based: during each contact run the foot should stay at
    the run's mean position, so the desired root correction is (mean - foot).
    Overlapping runs are averaged with their contact confidence, swing phases
    interpolate between neighbouring corrections, and the curve is smoothed so
    nothing pops. Only horizontal (X/Z) motion is touched; `ground` owns Y.
    """
    out = copy_clip(clip)
    if model is None:
        return out
    n = validate(out)
    fps = float(out["fps"])
    contacts = detect_contacts(out, model, cfg.contact_height, cfg.contact_speed)
    out["contacts"] = contacts
    feet = _foot_points(out, model).astype(np.float64)
    desired = np.zeros((n, 3), np.float64)
    weight = np.zeros(n, np.float64)
    for p, s, e in contact_segments(contacts):
        w = contacts[s:e, p].astype(np.float64)
        anchor = (feet[s:e, p] * w[:, None]).sum(0) / max(w.sum(), 1e-9)
        d = anchor[None, :] - feet[s:e, p]
        d[:, 1] = 0.0
        desired[s:e] += d * w[:, None]
        weight[s:e] += w
    if not np.any(weight > 0):
        return out
    has = weight > 0
    correction = np.zeros((n, 3), np.float64)
    correction[has] = desired[has] / weight[has, None]
    idx = np.arange(n)
    for axis in (0, 2):
        correction[:, axis] = np.interp(idx, idx[has], correction[has, axis])
    correction = _smooth_1d(correction, 1.2 * fps / 30.0)
    out["transl"] = (np.asarray(out["transl"], np.float64) + cfg.foot_lock_strength * correction).astype(np.float32)
    return out


def settle_ends(clip: dict, cfg: NaturalizeConfig, model: SMPLX | None = None) -> dict:
    """Blend the first/last frames toward their local mean so the clip starts and ends at rest."""
    n = validate(clip)
    out = copy_clip(clip)
    f = min(cfg.settle_frames, n // 3)
    if f < 2:
        return out
    for key in ("body_pose", "global_orient"):
        arr = np.asarray(out[key], np.float64)
        head = arr[:f].mean(0)
        tail = arr[-f:].mean(0)
        w = (1 - np.linspace(0, 1, f)) ** 2
        arr[:f] = arr[:f] * (1 - w[:, None]) + head * w[:, None]
        w = w[::-1]
        arr[-f:] = arr[-f:] * (1 - w[:, None]) + tail * w[:, None]
        out[key] = arr.astype(np.float32)
    return out


def breathing(clip: dict, cfg: NaturalizeConfig, model: SMPLX | None = None) -> dict:
    n = validate(clip)
    out = copy_clip(clip)
    if cfg.breathing <= 0:
        return out
    t = np.arange(n) / float(clip["fps"])
    wave = np.sin(2 * np.pi * t / 3.6)  # ~17 breaths / minute
    bp = np.asarray(out["body_pose"], np.float64).reshape(n, 21, 3)
    amp = 0.026 * cfg.breathing
    bp[:, B["spine2"], 0] -= amp * wave * 0.5
    bp[:, B["spine3"], 0] -= amp * wave * 0.5
    bp[:, B["neck"], 0] += amp * wave * 0.35
    bp[:, B["left_collar"], 2] -= amp * wave * 0.4
    bp[:, B["right_collar"], 2] += amp * wave * 0.4
    out["body_pose"] = bp.reshape(n, 63).astype(np.float32)
    return out


PASSES = {
    "smooth_rotations": smooth_rotations, "smooth_root": smooth_root, "joint_limits": joint_limits,
    "ground": ground, "foot_lock": foot_lock, "settle_ends": settle_ends, "breathing": breathing,
}


def naturalize(clip: dict, cfg: NaturalizeConfig | None = None, model: SMPLX | None = None) -> dict:
    cfg = cfg or NaturalizeConfig()
    out = copy_clip(clip)
    for name in cfg.passes:
        if name in ("joint_limits", "ground", "foot_lock", "settle_ends") and not getattr(cfg, name):
            continue
        if name == "breathing" and cfg.breathing <= 0:
            continue
        if name in ("ground", "foot_lock") and model is None:
            continue
        out = PASSES[name](out, cfg, model)
    out["naturalize"] = json.dumps(cfg.to_json())
    return out


# ---- diagnostics ------------------------------------------------------------------------

def metrics(clip: dict, model: SMPLX | None = None) -> dict:
    """Numbers to compare before/after: jitter, foot skate, floor penetration."""
    n = validate(clip)
    fps = float(clip["fps"])
    pose = full_pose(clip).astype(np.float64)[:, :22]
    q = np.stack([_unwrap_quats(rotvec_to_quat(pose[:, j])) for j in range(22)], 1)
    d1 = quat_to_rotvec(quat_mul(q[1:], quat_conj(q[:-1])))  # [T-1,22,3]
    ang_vel = np.linalg.norm(d1, axis=2) * fps
    jerk = np.abs(np.diff(ang_vel, n=2, axis=0)).mean() if n > 3 else 0.0
    out = {"frames": n, "fps": fps, "mean_angular_speed": float(ang_vel.mean()),
           "angular_jerk": float(jerk)}
    if model is not None:
        h, vel, low = _lowest_and_speed(clip, model)
        planted = h < 0.05
        out["foot_skate_m_per_s"] = float(vel[planted].mean()) if planted.any() else 0.0
        c = detect_contacts(clip, model)
        hard = c > 0.5
        out["contact_skate_m_per_s"] = float(vel[hard].mean()) if hard.any() else 0.0
        out["contact_ratio"] = float(hard.mean())
        out["floor_penetration_m"] = float(max(0.0, -h.min()))
        out["min_foot_height_m"] = float(h.min())
    return out
