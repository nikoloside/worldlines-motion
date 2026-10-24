"""Small, explicit MotionClip -> SMPL-X -> hair-anchor bridge.

Only the SMPL-X Python package and a separately licensed model file can
produce vertices. The built-in generator makes test *parameters*, not
MotionBricks output or a physically validated human motion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


FPS = 30
BODY_JOINTS = 21
HEAD_CHAIN = (3, 6, 9, 12, 15)  # SMPL-X indices, pelvis excluded
CASES = {"walk_squat": 6.0, "startle_jump": 4.0, "walk_startle": 5.0}


def smooth(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def ramp(t: float, a: float, b: float) -> float:
    return smooth((t - a) / (b - a))


def make_clip(case: str, fps: int = FPS) -> dict[str, np.ndarray | str | int]:
    """Generate deterministic SMPL-X axis-angle controls in a Y-up world."""
    if case not in CASES or fps <= 0:
        raise ValueError(f"case must be one of {tuple(CASES)}, fps > 0")
    count = round(CASES[case] * fps)
    transl = np.zeros((count, 3), dtype=np.float32)
    global_orient = np.zeros((count, 3), dtype=np.float32)
    body_pose = np.zeros((count, BODY_JOINTS, 3), dtype=np.float32)
    betas = np.zeros((10,), dtype=np.float32)
    for i in range(count):
        t = i / fps
        walking = case in ("walk_squat", "walk_startle")
        stop = 2.4 if case == "walk_squat" else 2.0
        gait = (1.0 - ramp(t, stop - 0.3, stop)) if walking else 0.0
        cycle = 2.0 * math.pi * t / 0.8
        transl[i, 0] = -2.5 + 2.5 * ramp(t, 0, stop) if walking else 0.0
        global_orient[i, 1] = math.pi / 2 if walking else 0.0  # face travel direction (+X)
        if case == "walk_squat":
            squat = ramp(t, 2.6, 3.3) * (1.0 - ramp(t, 4.15, 4.95))
            jump = 0.0
            alarm = 0.0
        else:
            trigger = 1.55 if case == "startle_jump" else 2.15
            anticipation = ramp(t, trigger - 0.18, trigger) * (1.0 - ramp(t, trigger, trigger + 0.12))
            u = (t - trigger) / 0.68
            jump = 0.32 * math.sin(math.pi * u) if 0.0 <= u <= 1.0 else 0.0
            squat = 0.15 * anticipation
            alarm = ramp(t, trigger - 0.05, trigger + 0.08) * (1.0 - ramp(t, trigger + 0.7, trigger + 1.35))
        transl[i, 1] = -0.39 * squat + jump
        # Index 0/1: left/right hip; 3/4: knees; 6/7: ankles.
        for side, phase, hip, knee, ankle, shoulder, elbow in (
            (1.0, 0.0, 0, 3, 6, 15, 17),
            (-1.0, math.pi, 1, 4, 7, 16, 18),
        ):
            stride = math.sin(cycle + phase) * gait
            body_pose[i, hip, 0] = -1.05 * squat + 0.26 * stride
            body_pose[i, knee, 0] = 2.10 * squat + max(0.0, -stride) * 0.35
            body_pose[i, ankle, 0] = -1.05 * squat - max(0.0, -stride) * 0.16
            body_pose[i, shoulder, 0] = -0.30 * stride - 0.50 * alarm
            body_pose[i, shoulder, 2] = -side * 0.90 * alarm
            body_pose[i, elbow, 0] = -0.24 * alarm
        body_pose[i, 2, 0] = 0.20 * squat  # spine1 forward balance
        body_pose[i, 11, 0] = 0.12 * alarm  # neck response
    return {
        "schema": "worldlines.smplx_motion.v1",
        "source": "procedural_test_only",
        "case": case,
        "fps": fps,
        "transl": transl,
        "global_orient": global_orient,
        "body_pose": body_pose.reshape(count, 63),
        "betas": betas,
    }


def validate(clip: dict) -> int:
    if clip.get("schema") != "worldlines.smplx_motion.v1":
        raise ValueError("unsupported schema")
    fps = int(clip["fps"])
    if fps <= 0:
        raise ValueError("fps must be positive")
    n = np.asarray(clip["transl"]).shape[0]
    for key, shape in (("transl", (n, 3)), ("global_orient", (n, 3)), ("body_pose", (n, 63)), ("betas", (10,))):
        value = np.asarray(clip[key])
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"{key}: expected finite {shape}, got {value.shape}")
    return n


def save_clip(clip: dict, path: Path) -> None:
    validate(clip)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **clip)


def load_clip(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        clip = {key: data[key].item() if data[key].ndim == 0 else data[key] for key in data.files}
    validate(clip)
    return clip


def head_quaternions(clip: dict) -> np.ndarray:
    """World head orientations in xyzw order, from the SMPL-X kinematic chain."""
    n = validate(clip)
    pose = np.asarray(clip["body_pose"]).reshape(n, BODY_JOINTS, 3)
    result = np.empty((n, 4), dtype=np.float32)
    for i in range(n):
        rot = rotvec_quat(clip["global_orient"][i])
        for smplx_joint in HEAD_CHAIN:
            rot = quat_mul(rot, rotvec_quat(pose[i, smplx_joint - 1]))
        result[i] = rot
    return result


def rotvec_quat(v: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    if angle < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    s = math.sin(angle / 2.0) / angle
    return np.array([v[0] * s, v[1] * s, v[2] * s, math.cos(angle / 2.0)], dtype=np.float64)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def render_model(clip: dict, model_path: Path, out: Path, gender: str = "male", scalp_indices: Path | None = None) -> None:
    """Run the actual SMPL-X model and export animated mesh + hair attachment data."""
    if not model_path.exists():
        raise FileNotFoundError(f"SMPL-X model asset missing: {model_path}")
    try:
        import smplx
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the official smplx Python package first") from exc
    n = validate(clip)
    model = smplx.create(str(model_path), model_type="smplx", gender=gender, use_pca=False, num_betas=10, batch_size=1)
    model.eval()
    all_vertices, all_head = [], []
    with torch.no_grad():
        for i in range(n):
            result = model(
                betas=torch.as_tensor(clip["betas"][None], dtype=torch.float32),
                transl=torch.as_tensor(clip["transl"][i:i + 1], dtype=torch.float32),
                global_orient=torch.as_tensor(clip["global_orient"][i:i + 1], dtype=torch.float32),
                body_pose=torch.as_tensor(clip["body_pose"][i:i + 1], dtype=torch.float32),
                return_verts=True,
            )
            all_vertices.append(result.vertices[0].cpu().numpy())
            all_head.append(result.joints[0, 15].cpu().numpy())
    vertices = np.stack(all_vertices).astype(np.float32)
    head_pos = np.stack(all_head).astype(np.float32)
    head_quat = head_quaternions(clip)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "smplx_animation.npz", vertices=vertices, faces=np.asarray(model.faces, dtype=np.int32), fps=clip["fps"])
    hair_data = {"schema": "worldlines.hair_attachment.v1", "fps": int(clip["fps"]), "space": "SMPL-X model world, Y-up", "head_position": head_pos.tolist(), "head_quaternion_xyzw": head_quat.tolist()}
    if scalp_indices is not None:
        ids = np.load(scalp_indices, allow_pickle=False)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(ids) == 0 or ids.min() < 0 or ids.max() >= vertices.shape[1]:
            raise ValueError("scalp indices must be valid 1-D vertex indices")
        hair_data["scalp_indices"] = ids.tolist()
        hair_data["scalp_vertices"] = vertices[:, ids].tolist()
    (out / "hair_attachment.json").write_text(json.dumps(hair_data, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("case", choices=CASES)
    gen.add_argument("output", type=Path)
    run = sub.add_parser("run")
    run.add_argument("clip", type=Path)
    run.add_argument("model", type=Path)
    run.add_argument("output_dir", type=Path)
    run.add_argument("--gender", choices=("male", "female", "neutral"), default="male")
    run.add_argument("--scalp-indices", type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        save_clip(make_clip(args.case), args.output)
        print(f"wrote {args.output}")
    else:
        render_model(load_clip(args.clip), args.model, args.output_dir, args.gender, args.scalp_indices)
        print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
