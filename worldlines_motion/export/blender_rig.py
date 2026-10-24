"""Prepare local-only SMPL-X mesh/weights for Blender from a licensed model (``wlmotion rig``).

Output contains restricted model-derived geometry. Keep it out of Git.
The resulting Blender rig has official topology, joint locations and LBS
weights. Per-frame pose corrective offsets are exported for a separate
Blender shape-key bake; these are local-only model-derived data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from smplx.lbs import batch_rodrigues, lbs

from ..clip import load_clip
from ..physique import PRESETS, make_muscular_rest, make_rest
from ..smplx_model import SMPLX


def prepare(model_path: Path, clip_path: Path, output_dir: Path,
            physique: str = "neutral", strength: float = 1.0, profile: dict | None = None) -> None:
    clip = load_clip(clip_path)
    if profile is None and isinstance(clip.get("physique"), dict):
        profile = clip["physique"]
    # The Julia NPZ has the official mesh/LBS tensors but omits hand PCA means;
    # smplx.create expects those means, so read the geometry directly instead.
    with np.load(model_path, allow_pickle=False) as model:
        template = np.asarray(model["v_template"], dtype=np.float32).copy()
        regressor = np.asarray(model["J_regressor"], dtype=np.float32)
        weights = np.asarray(model["weights"], dtype=np.float32)
        parents = np.asarray(model["kintree_table"][0], dtype=np.int64)
        faces = np.asarray(model["f"], dtype=np.int32)
        shapedirs = np.asarray(model["shapedirs"][:, :, :10], dtype=np.float32)
        posedirs = np.asarray(model["posedirs"], dtype=np.float32)
    if profile is not None:
        template, profile = make_rest(SMPLX(model_path), profile)
        betas = np.asarray(profile["betas"], np.float32)
        physique = "profile:" + json.dumps(profile, separators=(",", ":"))
    elif physique in PRESETS and physique != "neutral":
        template, profile = make_rest(SMPLX(model_path), PRESETS[physique])
        betas = np.asarray(profile["betas"], np.float32)
    elif physique == "muscular_male":
        template, betas = make_muscular_rest(template, shapedirs, regressor, weights, strength)
    elif physique == "neutral":
        betas = np.zeros(10, dtype=np.float32)
    else:
        raise ValueError(f"Unknown physique: {physique}")
    # Ground the rest body, preserving all relative joint and vertex positions.
    template[:, 1] -= template[:, 1].min()
    rest = regressor @ template
    if template.shape[0] != weights.shape[0] or weights.shape[1] != 55 or rest.shape != (55, 3):
        raise ValueError("Expected official SMPL-X 55-joint mesh and weights")
    colors = np.repeat(np.asarray([[195, 205, 212]], dtype=np.uint8), len(faces), axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "model_fixture.local.npz", vertices=template, faces=faces,
                        colors=colors, weights=weights, rest=rest, parents=parents,
                        source="licensed_smplx_model_lbs_only", physique=physique,
                        physique_strength=np.float32(strength), betas=betas)

    count = len(clip["transl"])
    frame_indices = np.array([0, count // 2, count - 1], dtype=np.int32)
    full_pose = np.zeros((count, 165), dtype=np.float32)
    full_pose[:, :3] = clip["global_orient"]
    full_pose[:, 3:66] = clip["body_pose"]
    pose = full_pose[frame_indices]
    template_t = torch.from_numpy(template)
    regressor_t = torch.from_numpy(regressor)
    weights_t = torch.from_numpy(weights)
    parents_t = torch.from_numpy(parents)
    shapedirs_t = torch.from_numpy(shapedirs)
    posedirs_t = torch.from_numpy(posedirs.reshape(-1, (55 - 1) * 9).T.copy())
    zero_posedirs = torch.zeros_like(posedirs_t)
    with torch.no_grad():
        rotations = batch_rodrigues(torch.from_numpy(full_pose.reshape(-1, 3))).reshape(count, 55, 3, 3)
        features = (rotations[:, 1:] - torch.eye(3)).reshape(count, -1)
        offsets = (features @ posedirs_t).reshape(count, len(template), 3).numpy()
        vertices, _ = lbs(
            betas=torch.zeros((len(frame_indices), 10), dtype=torch.float32),
            pose=torch.from_numpy(pose),
            v_template=template_t,
            shapedirs=shapedirs_t,
            posedirs=zero_posedirs,
            J_regressor=regressor_t,
            parents=parents_t,
            lbs_weights=weights_t,
        )
        full_vertices, _ = lbs(
            betas=torch.zeros((len(frame_indices), 10), dtype=torch.float32),
            pose=torch.from_numpy(pose),
            v_template=template_t,
            shapedirs=shapedirs_t,
            posedirs=posedirs_t,
            J_regressor=regressor_t,
            parents=parents_t,
            lbs_weights=weights_t,
        )
    expected = vertices.detach().cpu().numpy() + clip["transl"][frame_indices, None, :]
    np.savez_compressed(output_dir / "reference_lbs.local.npz", vertices=expected.astype(np.float32),
                        frame_indices=frame_indices, fps=int(clip["fps"]))
    np.savez_compressed(output_dir / "pose_offsets.local.npz", offsets=offsets.astype(np.float32), fps=int(clip["fps"]))
    expected_full = full_vertices.detach().cpu().numpy() + clip["transl"][frame_indices, None, :]
    np.savez_compressed(output_dir / "reference_full.local.npz", vertices=expected_full.astype(np.float32),
                        frame_indices=frame_indices, fps=int(clip["fps"]))
    print(f"Prepared SMPL-X mesh: {len(template)} vertices, {len(faces)} faces, 55 bones; physique={physique}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("clip", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--physique", choices=("neutral", *PRESETS), default="neutral")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--profile", help='physique profile JSON, e.g. \'{"muscular":1,"bulge":1}\' (overrides --physique)')
    args = parser.parse_args()
    prepare(args.model, args.clip, args.output_dir, args.physique, args.strength,
            json.loads(args.profile) if args.profile else None)


if __name__ == "__main__":
    main()
