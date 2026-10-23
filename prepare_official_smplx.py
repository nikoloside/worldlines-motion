"""Prepare local-only SMPL-X mesh/weights for Blender from a licensed model.

Output contains restricted model-derived geometry. Keep it out of Git.
The resulting Blender rig has official topology, joint locations and LBS
weights. Corrective pose blendshapes are not installed by this path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from smplx.lbs import lbs

from smplx_bridge import load_clip


def prepare(model_path: Path, clip_path: Path, output_dir: Path) -> None:
    import smplx

    clip = load_clip(clip_path)
    model = smplx.create(str(model_path), model_type="smplx", gender="male", use_pca=False, num_betas=10, batch_size=1)
    model.eval()
    template = model.v_template.detach().cpu().numpy().astype(np.float32)
    regressor = model.J_regressor.detach().cpu().numpy().astype(np.float32)
    weights = model.lbs_weights.detach().cpu().numpy().astype(np.float32)
    rest = regressor @ template
    parents = model.parents.detach().cpu().numpy().astype(np.int64)
    if template.shape[0] != weights.shape[0] or weights.shape[1] != 55 or rest.shape != (55, 3):
        raise ValueError("Expected official SMPL-X 55-joint mesh and weights")
    faces = np.asarray(model.faces, dtype=np.int32)
    colors = np.repeat(np.asarray([[195, 205, 212]], dtype=np.uint8), len(faces), axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "model_fixture.local.npz", vertices=template, faces=faces,
                        colors=colors, weights=weights, rest=rest, parents=parents,
                        source="licensed_smplx_model_lbs_only")

    count = len(clip["transl"])
    frame_indices = np.array([0, count // 2, count - 1], dtype=np.int32)
    pose = np.zeros((len(frame_indices), 165), dtype=np.float32)
    pose[:, :3] = clip["global_orient"][frame_indices]
    pose[:, 3:66] = clip["body_pose"][frame_indices]
    posedirs = torch.zeros_like(model.posedirs)
    with torch.no_grad():
        vertices, _ = lbs(
            betas=torch.zeros((len(frame_indices), model.num_betas), dtype=torch.float32),
            pose=torch.from_numpy(pose),
            v_template=model.v_template,
            shapedirs=model.shapedirs,
            posedirs=posedirs,
            J_regressor=model.J_regressor,
            parents=model.parents,
            lbs_weights=model.lbs_weights,
        )
    expected = vertices.detach().cpu().numpy() + clip["transl"][frame_indices, None, :]
    np.savez_compressed(output_dir / "reference_lbs.local.npz", vertices=expected.astype(np.float32),
                        frame_indices=frame_indices, fps=int(clip["fps"]))
    print(f"Prepared official SMPL-X mesh: {len(template)} vertices, {len(faces)} faces, 55 bones")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("clip", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    prepare(args.model, args.clip, args.output_dir)


if __name__ == "__main__":
    main()
