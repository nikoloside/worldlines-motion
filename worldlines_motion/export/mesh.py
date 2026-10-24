"""Bake a clip into per-frame vertices (``smplx_animation.npz``) plus a hair
attachment file (head position and orientation per frame).

Output contains model-derived geometry: keep it out of Git.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..clip import full_pose, quat_mul, rotvec_to_quat
from ..physique import make_rest
from ..smplx_model import SMPLX

HEAD_CHAIN = (3, 6, 9, 12, 15)  # spine1, spine2, spine3, neck, head
HEAD = 15


def bake_mesh(clip: dict, model: SMPLX, out_dir: Path | str, step: int = 1,
              pose_correctives: bool = True, progress=None) -> Path:
    """Write ``smplx_animation.npz`` and ``hair_attachment.json`` into *out_dir*."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    step = max(1, int(step))
    rest, profile = make_rest(model, clip.get("physique"))
    pose = full_pose(clip)
    frames = range(0, len(pose), step)
    verts, joints, head_q = [], [], []
    for n, i in enumerate(frames):
        v, j = model.skin(rest, pose[i][None], clip["transl"][i][None], pose_correctives=pose_correctives, return_joints=True)
        verts.append(v[0])
        joints.append(j[0])
        q = rotvec_to_quat(pose[i, 0])
        for k in HEAD_CHAIN:
            q = quat_mul(q, rotvec_to_quat(pose[i, k]))
        head_q.append(q)
        if progress:
            progress(n + 1, len(frames))
    verts, joints = np.stack(verts), np.stack(joints)
    fps = float(clip["fps"]) / step
    np.savez_compressed(out / "smplx_animation.npz", vertices=verts, faces=model.faces, joints=joints, fps=fps,
                        physique=json.dumps(profile))
    hair = {"schema": "worldlines.hair_attachment.v1", "fps": fps, "space": "SMPL-X model world, Y-up",
            "head_position": joints[:, HEAD].tolist(), "head_quaternion_xyzw": np.asarray(head_q).tolist()}
    (out / "hair_attachment.json").write_text(json.dumps(hair), encoding="utf-8")
    return out
