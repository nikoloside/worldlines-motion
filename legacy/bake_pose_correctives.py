"""Bake licensed SMPL-X pose corrections into local Blender shape keys.

Each integer frame gets an exact pre-skinning corrective, followed by the
existing official vertex weights and 55-bone Armature modifier. The result
matches the upstream SMPL-X LBS+pose-blendshape computation at 30 fps.
Do not commit the output .blend or model-derived offsets to an open repo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np


def bake(offsets_path: Path, reference_path: Path, output_path: Path, render_path: Path | None) -> None:
    with np.load(offsets_path, allow_pickle=False) as data:
        offsets = data["offsets"]
        fps = int(data["fps"])
    with np.load(reference_path, allow_pickle=False) as data:
        expected = data["vertices"]
        frame_indices = data["frame_indices"]
    body = bpy.data.objects["SMPL-X body — LBS only"]
    arm = bpy.data.objects["SMPL-X 55-bone Armature"]
    scene = bpy.context.scene
    frames, vertex_count, xyz = offsets.shape
    if xyz != 3 or vertex_count != len(body.data.vertices) or frames != scene.frame_end or fps != scene.render.fps:
        raise ValueError("Pose offsets do not match this SMPL-X mesh or animation")
    base = np.empty((vertex_count * 3,), dtype=np.float32)
    body.data.vertices.foreach_get("co", base)
    base = base.reshape(vertex_count, 3)
    body.shape_key_add(name="Basis", from_mix=False)
    for i in range(frames):
        key = body.shape_key_add(name=f"PoseCorrective_{i + 1:04d}", from_mix=False)
        key.data.foreach_set("co", np.ascontiguousarray(base + offsets[i]).ravel())
        key.value = 0.0
        if i > 0:
            key.keyframe_insert(data_path="value", frame=i)
        key.value = 1.0
        key.keyframe_insert(data_path="value", frame=i + 1)
        key.value = 0.0
        if i + 1 < frames:
            key.keyframe_insert(data_path="value", frame=i + 2)
        if (i + 1) % 30 == 0:
            print(f"Pose correctives: {i + 1}/{frames}")

    # Linear crossfades between adjacent integer-frame corrective keys.
    action = body.data.shape_keys.animation_data.action
    for curve in action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"

    for ref_index, source_frame in enumerate(frame_indices):
        scene.frame_set(int(source_frame) + 1)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = body.evaluated_get(depsgraph)
        evaluated_mesh = evaluated.to_mesh()
        actual = np.array([v.co[:] for v in evaluated_mesh.vertices], dtype=np.float32)
        evaluated.to_mesh_clear()
        error = float(np.max(np.abs(actual - expected[ref_index])))
        print(f"Full SMPL-X comparison frame {source_frame + 1}: max local vertex error {error:.6f} m")
        if error > 1e-3:
            raise RuntimeError("Blender mesh differs from SMPL-X pose-corrected reference")

    body.name = "SMPL-X body — full frame-corrected skin"
    body["source"] = "licensed_smplx_model_local_only"
    body["correctives"] = "one shape key per integer frame, before Armature modifier"
    scene.frame_set(min(frames, round(frames * 0.58)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path.resolve()))
    if render_path is not None:
        scene.render.filepath = str(render_path.resolve())
        bpy.ops.render.render(write_still=True)
    print(f"Saved full pose-corrected SMPL-X scene: {output_path}")


def main() -> None:
    args_after_separator = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("offsets", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--render", type=Path)
    args = parser.parse_args(args_after_separator)
    bake(args.offsets, args.reference, args.output, args.render)


if __name__ == "__main__":
    main()
