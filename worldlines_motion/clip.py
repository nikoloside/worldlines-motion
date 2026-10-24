"""MotionClip: SMPL-X parameter clip, schema ``worldlines.smplx_motion.v1``.

Required arrays (Y-up world, metres, radians, axis-angle):
  transl[T,3]  global_orient[T,3]  body_pose[T,63]  betas[10]  fps
Optional: left_hand_pose[T,45], right_hand_pose[T,45], jaw_pose[T,3],
  contacts[T,4] (L heel, L toe, R heel, R toe in 0..1), source, case,
  physique (JSON string), hands (JSON string, raw finger keypoints).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "worldlines.smplx_motion.v1"
BODY_JOINTS = 21
NUM_BETAS = 10

# SMPL-X body joint order (index into body_pose // 3); pelvis is global_orient.
BODY_JOINT_NAMES = (
    "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)
JOINT_NAMES = ("pelvis",) + BODY_JOINT_NAMES + (
    "jaw", "left_eye_smplhf", "right_eye_smplhf",
    *(f"left_{f}{s}" for f in ("index", "middle", "pinky", "ring", "thumb") for s in (1, 2, 3)),
    *(f"right_{f}{s}" for f in ("index", "middle", "pinky", "ring", "thumb") for s in (1, 2, 3)),
)
# Parent of each of the 55 SMPL-X joints.
PARENTS = np.array([
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15,
    20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
    21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53,
], dtype=np.int64)

OPTIONAL_ARRAYS = {
    "left_hand_pose": 45, "right_hand_pose": 45, "jaw_pose": 3, "contacts": 4,
}
OPTIONAL_STRINGS = ("source", "case", "physique", "hands", "prompt", "video", "notes")


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def ramp(t: float, a: float, b: float) -> float:
    return smoothstep((t - a) / (b - a))


def validate(clip: dict) -> int:
    """Return frame count or raise ValueError."""
    if clip.get("schema") != SCHEMA:
        raise ValueError(f"unsupported schema {clip.get('schema')!r}")
    fps = float(clip["fps"])
    if not fps > 0:
        raise ValueError("fps must be positive")
    n = int(np.asarray(clip["transl"]).shape[0])
    for key, shape in (("transl", (n, 3)), ("global_orient", (n, 3)), ("body_pose", (n, 63)), ("betas", (NUM_BETAS,))):
        value = np.asarray(clip[key])
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"{key}: expected finite {shape}, got {value.shape}")
    for key, width in OPTIONAL_ARRAYS.items():
        if key in clip and clip[key] is not None:
            value = np.asarray(clip[key])
            if value.shape != (n, width) or not np.all(np.isfinite(value)):
                raise ValueError(f"{key}: expected finite {(n, width)}, got {value.shape}")
    return n


def new_clip(n: int, fps: float = 30.0, source: str = "unknown") -> dict:
    return {
        "schema": SCHEMA, "source": source, "fps": fps,
        "transl": np.zeros((n, 3), np.float32), "global_orient": np.zeros((n, 3), np.float32),
        "body_pose": np.zeros((n, 63), np.float32), "betas": np.zeros((NUM_BETAS,), np.float32),
    }


def save_clip(clip: dict, path: Path | str) -> Path:
    validate(clip)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, value in clip.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        out[key] = value
    np.savez_compressed(path, **out)
    return path


def load_clip(path: Path | str) -> dict:
    with np.load(Path(path), allow_pickle=False) as data:
        clip = {key: data[key].item() if data[key].ndim == 0 else data[key] for key in data.files}
    validate(clip)
    for key in ("physique", "hands"):
        if isinstance(clip.get(key), str):
            try:
                clip[key] = json.loads(clip[key])
            except json.JSONDecodeError:
                pass
    return clip


def copy_clip(clip: dict) -> dict:
    return {k: (v.copy() if isinstance(v, np.ndarray) else (json.loads(json.dumps(v)) if isinstance(v, (dict, list)) else v)) for k, v in clip.items()}


def to_json(clip: dict, precision: int = 5) -> dict[str, Any]:
    """JSON-serialisable form (arrays rounded) for the workbench."""
    n = validate(clip)
    out: dict[str, Any] = {"schema": SCHEMA, "fps": float(clip["fps"]), "n": n}
    for key in ("transl", "global_orient", "body_pose", "betas", *OPTIONAL_ARRAYS):
        if key in clip and clip[key] is not None:
            out[key] = np.round(np.asarray(clip[key], np.float64), precision).reshape(-1).tolist()
    for key in OPTIONAL_STRINGS:
        if key in clip:
            out[key] = clip[key]
    return out


def from_json(data: dict) -> dict:
    n = int(data["n"])
    clip = new_clip(n, float(data["fps"]), str(data.get("source", "json")))
    for key, width in (("transl", 3), ("global_orient", 3), ("body_pose", 63)):
        clip[key] = np.asarray(data[key], np.float32).reshape(n, width)
    clip["betas"] = np.asarray(data.get("betas", np.zeros(NUM_BETAS)), np.float32).reshape(NUM_BETAS)
    for key, width in OPTIONAL_ARRAYS.items():
        if data.get(key) is not None:
            clip[key] = np.asarray(data[key], np.float32).reshape(n, width)
    for key in OPTIONAL_STRINGS:
        if key in data:
            clip[key] = data[key]
    validate(clip)
    return clip


# ---- rotation helpers (numpy, batched) ------------------------------------------

def rotvec_to_quat(v: np.ndarray) -> np.ndarray:
    """[...,3] axis-angle -> [...,4] quaternion (x, y, z, w)."""
    v = np.asarray(v, np.float64)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    half = angle * 0.5
    small = angle[..., 0] < 1e-8
    s = np.where(small[..., None], 0.5 - angle ** 2 / 48.0, np.sin(half) / np.where(small[..., None], 1.0, angle))
    xyz = v * s
    w = np.cos(half)
    return np.concatenate([xyz, w], axis=-1)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(q[..., 3:4] < 0, -q, q)  # shortest arc
    xyz, w = q[..., :3], np.clip(q[..., 3], -1.0, 1.0)
    sin_half = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half[..., 0], w)
    small = sin_half[..., 0] < 1e-8
    scale = np.where(small[..., None], 2.0, angle[..., None] / np.where(small[..., None], 1.0, sin_half))
    return xyz * scale


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(np.asarray(a, np.float64), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b, np.float64), -1, 0)
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    return np.concatenate([-q[..., :3], q[..., 3:4]], axis=-1)


def rotvec_to_matrix(v: np.ndarray) -> np.ndarray:
    """[...,3] -> [...,3,3] (Rodrigues)."""
    v = np.asarray(v, np.float64)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    axis = v / np.where(angle < 1e-12, 1.0, angle)
    a = angle[..., 0]
    c, s = np.cos(a), np.sin(a)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    C = 1.0 - c
    m = np.empty(v.shape[:-1] + (3, 3), np.float64)
    m[..., 0, 0] = c + x * x * C
    m[..., 0, 1] = x * y * C - z * s
    m[..., 0, 2] = x * z * C + y * s
    m[..., 1, 0] = y * x * C + z * s
    m[..., 1, 1] = c + y * y * C
    m[..., 1, 2] = y * z * C - x * s
    m[..., 2, 0] = z * x * C - y * s
    m[..., 2, 1] = z * y * C + x * s
    m[..., 2, 2] = c + z * z * C
    return m


def matrix_to_rotvec(m: np.ndarray) -> np.ndarray:
    return quat_to_rotvec(matrix_to_quat(m))


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, np.float64)
    shape = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    q = np.empty((m.shape[0], 4), np.float64)
    for i, r in enumerate(m):
        t = np.trace(r)
        if t > 0:
            s = math.sqrt(t + 1.0) * 2
            q[i] = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            q[i] = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
        elif r[1, 1] > r[2, 2]:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            q[i] = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            q[i] = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    return q.reshape(shape + (4,))


def full_pose(clip: dict) -> np.ndarray:
    """[T,55,3] axis-angle including hands/jaw when present (else zeros)."""
    n = validate(clip)
    pose = np.zeros((n, 55, 3), np.float32)
    pose[:, 0] = clip["global_orient"]
    pose[:, 1:22] = np.asarray(clip["body_pose"]).reshape(n, 21, 3)
    if clip.get("jaw_pose") is not None:
        pose[:, 22] = clip["jaw_pose"]
    if clip.get("left_hand_pose") is not None:
        pose[:, 25:40] = np.asarray(clip["left_hand_pose"]).reshape(n, 15, 3)
    if clip.get("right_hand_pose") is not None:
        pose[:, 40:55] = np.asarray(clip["right_hand_pose"]).reshape(n, 15, 3)
    return pose


def resample(clip: dict, fps: float) -> dict:
    """Linear in translation, slerp-free (rotvec lerp, fine for small steps) resample."""
    n = validate(clip)
    src = float(clip["fps"])
    if abs(src - fps) < 1e-6:
        return copy_clip(clip)
    duration = (n - 1) / src
    m = max(2, int(round(duration * fps)) + 1)
    t_new = np.arange(m) / fps
    t_old = np.arange(n) / src
    out = copy_clip(clip)
    out["fps"] = fps

    def interp(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, np.float64)
        return np.stack([np.interp(t_new, t_old, arr[:, k]) for k in range(arr.shape[1])], axis=1).astype(np.float32)

    for key in ("transl", "global_orient", "body_pose", *OPTIONAL_ARRAYS):
        if key in out and out[key] is not None:
            out[key] = interp(out[key])
    return out


def concat(clips: list[dict], blend_frames: int = 6) -> dict:
    """Chain clips with a short cross-fade; root translation is made continuous."""
    if not clips:
        raise ValueError("no clips")
    out = copy_clip(clips[0])
    for nxt in clips[1:]:
        nxt = copy_clip(nxt)
        if abs(float(nxt["fps"]) - float(out["fps"])) > 1e-6:
            nxt = resample(nxt, float(out["fps"]))
        offset = out["transl"][-1] - nxt["transl"][0]
        offset[1] = 0.0
        nxt["transl"] = nxt["transl"] + offset
        b = min(blend_frames, len(out["transl"]) - 1, len(nxt["transl"]) - 1)
        for key in ("transl", "global_orient", "body_pose", *OPTIONAL_ARRAYS):
            if key not in out or out[key] is None or nxt.get(key) is None:
                if key in out and out[key] is not None:
                    out[key] = np.concatenate([out[key], np.zeros((len(nxt["transl"]), out[key].shape[1]), np.float32)])
                continue
            a, c = out[key], nxt[key]
            if b > 0:
                w = np.linspace(0, 1, b + 2)[1:-1, None]
                tail = a[-b:] * (1 - w) + c[:b] * w
                a = np.concatenate([a[:-b], tail])
                c = c[b:]
            out[key] = np.concatenate([a, c]).astype(np.float32)
    validate(out)
    return out
