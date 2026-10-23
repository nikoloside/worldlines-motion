"""Build an editable Blender scene using the self-authored 55-bone test body.

This body uses SMPL-X joint names/order and synthetic vertex weights. It is
not the licensed SMPL-X body mesh, blendshapes, or official Blender add-on.

blender -b --factory-startup --python build_blender_scene.py -- \
  fixture.npz walk_squat.smplx.npz walk_squat.synthetic_rig.blend \
  walk_squat.synthetic_lbs.npz
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector


JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "jaw", "left_eye_smplhf", "right_eye_smplhf",
    *(f"left_{finger}{segment}" for finger in ("index", "middle", "pinky", "ring", "thumb") for segment in (1, 2, 3)),
    *(f"right_{finger}{segment}" for finger in ("index", "middle", "pinky", "ring", "thumb") for segment in (1, 2, 3)),
)


def rod_quat(rod):
    angle = float(np.linalg.norm(rod))
    return Quaternion(Vector(rod / angle), angle) if angle > 1e-9 else Quaternion()


def make_material(rgb):
    material = bpy.data.materials.new(name=f"Test body {rgb[0]}-{rgb[1]}-{rgb[2]}")
    material.diffuse_color = tuple(float(x) / 255 for x in rgb) + (1.0,)
    material.use_nodes = True
    material.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = material.diffuse_color
    material.node_tree.nodes.get("Principled BSDF").inputs["Roughness"].default_value = 0.8
    return material


def make_scene(fixture, clip, output, reference, render_path=None):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    rest = fixture["rest"]
    parents = fixture["parents"]
    vertices = fixture["vertices"]
    faces = fixture["faces"]
    colors = fixture["colors"]
    weights = fixture["weights"]
    is_official = "source" in fixture and str(fixture["source"]) == "licensed_smplx_model_lbs_only"
    assert len(JOINT_NAMES) == len(rest) == 55

    arm_data = bpy.data.armatures.new("SMPL-X joint layout")
    arm_name = "SMPL-X 55-bone Armature" if is_official else "Synthetic SMPL-X 55-bone Armature"
    arm = bpy.data.objects.new(arm_name, arm_data)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for i, name in enumerate(JOINT_NAMES):
        bone = arm_data.edit_bones.new(name)
        bone.head = tuple(float(x) for x in rest[i])
        children = [j for j, parent in enumerate(parents) if parent == i]
        tail = Vector(rest[children[0]]) if children else Vector(rest[i]) + Vector((0, 0.08, 0))
        if (tail - bone.head).length < 0.015:
            tail = bone.head + Vector((0, 0.08, 0))
        bone.tail = tail
    for i, name in enumerate(JOINT_NAMES):
        if parents[i] >= 0:
            arm_data.edit_bones[name].parent = arm_data.edit_bones[JOINT_NAMES[int(parents[i])]]
            arm_data.edit_bones[name].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")

    mesh_data = bpy.data.meshes.new("Licensed SMPL-X mesh" if is_official else "Synthetic whitebox mesh")
    mesh_data.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh_data.update()
    body = bpy.data.objects.new("SMPL-X body — LBS only" if is_official else "Self-authored test body — not official SMPL-X", mesh_data)
    bpy.context.collection.objects.link(body)
    palette = {}
    for face, color in zip(mesh_data.polygons, colors):
        key = tuple(int(c) for c in color)
        if key not in palette:
            palette[key] = len(palette)
            mesh_data.materials.append(make_material(key))
        face.material_index = palette[key]
    groups = [body.vertex_groups.new(name=name) for name in JOINT_NAMES]
    for vertex_index, row in enumerate(weights):
        for joint_index in np.flatnonzero(row > 1e-6):
            groups[int(joint_index)].add([vertex_index], float(row[joint_index]), "REPLACE")
    modifier = body.modifiers.new("55-joint linear skinning", "ARMATURE")
    modifier.object = arm
    modifier.use_vertex_groups = True
    modifier.use_bone_envelopes = False

    # Both objects share an object-space transform: source Y-up -> Blender Z-up.
    arm.rotation_euler.x = math.pi / 2
    body.rotation_euler.x = math.pi / 2
    bpy.context.view_layer.update()
    bone_orientations = [arm_data.bones[name].matrix_local.to_3x3() for name in JOINT_NAMES]
    count = len(clip["transl"])
    poses = np.zeros((count, 55, 3), dtype=np.float32)
    poses[:, 0] = clip["global_orient"]
    poses[:, 1:22] = clip["body_pose"].reshape(count, 21, 3)
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    root_basis_inverse = bone_orientations[0].inverted()
    for frame_index in range(count):
        for joint_index, name in enumerate(JOINT_NAMES):
            basis = bone_orientations[joint_index]
            q = rod_quat(poses[frame_index, joint_index])
            arm.pose.bones[name].rotation_quaternion = (basis.inverted() @ q.to_matrix() @ basis).to_quaternion()
            arm.pose.bones[name].keyframe_insert(data_path="rotation_quaternion", frame=frame_index + 1)
        arm.pose.bones[JOINT_NAMES[0]].location = root_basis_inverse @ Vector(clip["transl"][frame_index])
        arm.pose.bones[JOINT_NAMES[0]].keyframe_insert(data_path="location", frame=frame_index + 1)
    scene = bpy.context.scene
    scene.render.fps = int(clip["fps"])
    scene.frame_start = 1
    scene.frame_end = count

    # A ground plane and camera make the saved file inspectable immediately.
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.025))
    floor = bpy.context.object
    floor.name = "Ground"
    floor.data.materials.append(make_material((226, 234, 236)))
    camera_data = bpy.data.cameras.new("Stage camera")
    camera = bpy.data.objects.new("Stage camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (3.2, -7.8, 3.1)
    target = Vector((-0.8, 0, 0.95))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 5.1
    scene.camera = camera
    light_data = bpy.data.lights.new("Softbox", "AREA")
    light = bpy.data.objects.new("Softbox", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (0, -3, 5)
    light_data.energy = 850
    light_data.shape = "DISK"
    light_data.size = 5
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 960
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.world.color = (0.55, 0.55, 0.55)

    # Compare Blender's evaluated Armature mesh with the upstream LBS result.
    if reference is not None:
        expected = reference["vertices"]
        step = scene.render.fps // int(reference["fps"])
        frames = reference["frame_indices"] if "frame_indices" in reference else (0, count // 2 // step * step, (count - 1) // step * step)
        for ref_index, source_frame in enumerate(frames):
            scene.frame_set(source_frame + 1)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            evaluated = body.evaluated_get(depsgraph)
            evaluated_mesh = evaluated.to_mesh()
            actual = np.array([v.co[:] for v in evaluated_mesh.vertices], dtype=np.float32)
            evaluated.to_mesh_clear()
            expected_index = ref_index if "frame_indices" in reference else source_frame // step
            error = float(np.max(np.abs(actual - expected[expected_index])))
            print(f"LBS comparison frame {source_frame + 1}: max local vertex error {error:.6f} m")
            if error > 1e-3:
                raise RuntimeError("Blender armature does not match the reference LBS mesh")

    scene.frame_set(min(count, round(count * 0.58)))
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output.resolve()))
    if render_path:
        scene.render.filepath = str(render_path.resolve())
        bpy.ops.render.render(write_still=True)
    print(f"Saved editable skinned scene: {output}")


def main():
    args_after_separator = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("clip", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--render", type=Path)
    args = parser.parse_args(args_after_separator)
    with np.load(args.fixture, allow_pickle=False) as data:
        fixture = {key: data[key] for key in data.files}
    with np.load(args.clip, allow_pickle=False) as data:
        clip = {key: data[key] for key in data.files}
    with np.load(args.reference, allow_pickle=False) as data:
        reference = {key: data[key] for key in data.files}
    make_scene(fixture, clip, args.output, reference, args.render)


if __name__ == "__main__":
    main()
