"""Render three deterministic geometry-only motion tests with Blender.

Usage: blender -b -t 4 --python render_proxies.py -- CASE OUTPUT_DIR
CASE is walk_squat, startle_jump, or walk_startle.
These are blocking proxies, not MotionBricks, SMPL-X, or retargeted results.
"""

import math
import pathlib
import sys

import bpy
from mathutils import Vector


CASE = sys.argv[sys.argv.index("--") + 1]
OUT = pathlib.Path(sys.argv[sys.argv.index("--") + 2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
FPS = 12
DURATIONS = {"walk_squat": 6, "startle_jump": 4, "walk_startle": 5}
assert CASE in DURATIONS

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.render.resolution_x = 640
scene.render.resolution_y = 360
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "PNG"
scene.render.fps = FPS
scene.render.frame_start = 1
scene.render.frame_end = FPS * DURATIONS[CASE]
scene.view_settings.view_transform = "Standard"
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "MATERIAL"
scene.display.shading.show_shadows = True
scene.display.shading.show_cavity = True
scene.world.color = (0.72, 0.76, 0.81)


def material(name, rgb):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*rgb, 1)
    return m


BODY = material("proxy body", (0.23, 0.47, 0.71))
SKIN = material("proxy skin", (0.86, 0.68, 0.52))
PANTS = material("proxy pants", (0.13, 0.19, 0.30))
SHOES = material("proxy shoes", (0.93, 0.93, 0.88))
STAGE = material("floor", (0.52, 0.57, 0.61))
ACCENT = material("arrival mark", (0.90, 0.44, 0.22))


def sphere(name, radius, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8, radius=radius)
    ob = bpy.context.object
    ob.name = name
    ob.data.materials.append(mat)
    return ob


def limb(name, radius, mat):
    bpy.ops.mesh.primitive_cylinder_add(vertices=10, radius=radius, depth=1)
    ob = bpy.context.object
    ob.name = name
    ob.rotation_mode = "QUATERNION"
    ob.data.materials.append(mat)
    return ob


def block(name, loc, scale, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    ob = bpy.context.object
    ob.name = name
    ob.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    ob.data.materials.append(mat)
    return ob


block("floor", (0, 0, -0.08), (9, 3.5, 0.15), STAGE)
block("arrival stripe", (0, -0.35, 0.002), (0.055, 0.65, 0.004), ACCENT)
block("entry frame left", (-3.25, 0.65, 1.15), (0.12, 0.22, 2.3), PANTS)
block("entry frame top", (-2.75, 0.65, 2.27), (1.05, 0.22, 0.12), PANTS)

JOINTS = {
    "pelvis": (0.14, PANTS),
    "chest": (0.15, BODY),
    "head": (0.20, SKIN),
    "left_shoulder": (0.08, BODY),
    "right_shoulder": (0.08, BODY),
    "left_elbow": (0.06, SKIN),
    "right_elbow": (0.06, SKIN),
    "left_wrist": (0.06, SKIN),
    "right_wrist": (0.06, SKIN),
    "left_hip": (0.08, PANTS),
    "right_hip": (0.08, PANTS),
    "left_knee": (0.07, PANTS),
    "right_knee": (0.07, PANTS),
    "left_ankle": (0.06, SHOES),
    "right_ankle": (0.06, SHOES),
}
SEGMENTS = [
    ("pelvis", "chest", BODY, 0.18),
    ("chest", "head", SKIN, 0.10),
    ("left_shoulder", "left_elbow", BODY, 0.075),
    ("right_shoulder", "right_elbow", BODY, 0.075),
    ("left_elbow", "left_wrist", SKIN, 0.055),
    ("right_elbow", "right_wrist", SKIN, 0.055),
    ("left_hip", "left_knee", PANTS, 0.10),
    ("right_hip", "right_knee", PANTS, 0.10),
    ("left_knee", "left_ankle", PANTS, 0.075),
    ("right_knee", "right_ankle", PANTS, 0.075),
]
balls = {name: sphere(name, radius, mat) for name, (radius, mat) in JOINTS.items()}
cylinders = {(a, b): limb(f"{a}-{b}", radius, mat) for a, b, mat, radius in SEGMENTS}


def smooth(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def ramp(t, a, b):
    return smooth((t - a) / (b - a))


def pose(t):
    walking_end = 2.4 if CASE == "walk_squat" else 2.0
    walking = CASE in {"walk_squat", "walk_startle"}
    root_x = -2.5 + 2.5 * ramp(t, 0, walking_end) if walking else 0.0
    gait = (1 - ramp(t, walking_end - 0.3, walking_end)) if walking else 0.0
    cycle = 2 * math.pi * t / 0.80
    if CASE == "walk_squat":
        bend = ramp(t, 2.6, 3.25) * (1 - ramp(t, 4.2, 4.9))
        jump = 0.0
        alarm = 0.0
    else:
        trigger = 1.55 if CASE == "startle_jump" else 2.15
        crouch = ramp(t, trigger - 0.16, trigger) * (1 - ramp(t, trigger, trigger + 0.13))
        jump_u = (t - trigger) / 0.68
        jump = 0.36 * math.sin(math.pi * jump_u) if 0 <= jump_u <= 1 else 0.0
        bend = 0.20 * crouch + 0.15 * ramp(t, trigger + 0.68, trigger + 0.80) * (1 - ramp(t, trigger + 0.80, trigger + 1.1))
        alarm = ramp(t, trigger - 0.05, trigger + 0.08) * (1 - ramp(t, trigger + 0.75, trigger + 1.4))
    hip_z = 0.97 - 0.39 * bend + jump + 0.018 * abs(math.sin(cycle)) * gait
    torso_lean = 0.13 * bend - 0.07 * alarm
    p = {
        "pelvis": (root_x, 0, hip_z),
        "chest": (root_x + torso_lean, 0, hip_z + 0.47),
        "head": (root_x + torso_lean - 0.02 * alarm, 0, hip_z + 0.91),
    }
    for side, offset, phase in (("left", -0.13, 0), ("right", 0.13, math.pi)):
        step = math.sin(cycle + phase) * 0.28 * gait
        ankle_x = root_x + step * 1.12 + 0.03 * bend
        ankle_z = 0.10 + jump + max(0, math.cos(cycle + phase)) * 0.13 * gait
        knee_x = root_x + step * 0.35 + 0.25 * bend
        knee_z = 0.53 - 0.10 * bend + jump
        p[f"{side}_hip"] = (root_x, offset, hip_z - 0.08)
        p[f"{side}_knee"] = (knee_x, offset, knee_z)
        p[f"{side}_ankle"] = (ankle_x, offset, ankle_z)
        arm_swing = -math.sin(cycle + phase) * 0.18 * gait
        p[f"{side}_shoulder"] = (root_x + torso_lean, offset * 1.4, hip_z + 0.65)
        p[f"{side}_elbow"] = (root_x + torso_lean + arm_swing - 0.16 * alarm, offset * 1.75, hip_z + 0.37 + 0.23 * alarm)
        p[f"{side}_wrist"] = (root_x + torso_lean + arm_swing - 0.27 * alarm, offset * 1.9, hip_z + 0.12 + 0.53 * alarm)
    return p


axis = Vector((0, 0, 1))
for frame in range(scene.frame_start, scene.frame_end + 1):
    t = (frame - 1) / FPS
    p = pose(t)
    for name, obj in balls.items():
        obj.location = p[name]
        obj.keyframe_insert(data_path="location", frame=frame)
    for a, b, _, _ in SEGMENTS:
        obj = cylinders[(a, b)]
        pa, pb = Vector(p[a]), Vector(p[b])
        d = pb - pa
        obj.location = (pa + pb) * 0.5
        obj.rotation_quaternion = axis.rotation_difference(d.normalized())
        obj.scale.z = d.length
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        obj.keyframe_insert(data_path="scale", frame=frame)

bpy.ops.object.camera_add(location=(0, -9.5, 2.25))
camera = bpy.context.object
direction = Vector((0, 0, 1.05)) - camera.location
camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
camera.data.lens = 36
scene.camera = camera

scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / f"{CASE}.blend"))
scene.render.filepath = str(OUT / "frames" / "frame-")
(OUT / "frames").mkdir(exist_ok=True)
bpy.ops.render.render(animation=True)
