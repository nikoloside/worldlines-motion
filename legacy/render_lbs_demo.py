"""Render a self-authored whitebox character with upstream SMPL-X LBS.

The geometry and rest skeleton here are synthetic test fixtures. No SMPL-X
Body mesh, learned shape directions, or MotionBricks output is included.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from smplx.lbs import lbs

from smplx_bridge import head_quaternions, load_clip


JOINTS = 55
REST = np.zeros((JOINTS, 3), dtype=np.float32)
REST[:25] = np.array([
    (0, 1.00, 0), (-.16, .92, 0), (.16, .92, 0), (0, 1.17, 0),
    (-.16, .55, 0), (.16, .55, 0), (0, 1.35, 0),
    (-.16, .12, 0), (.16, .12, 0), (0, 1.51, 0),
    (-.16, .07, .17), (.16, .07, .17), (0, 1.67, 0),
    (-.13, 1.59, 0), (.13, 1.59, 0), (0, 1.81, 0),
    (-.31, 1.57, 0), (.31, 1.57, 0), (-.51, 1.28, 0),
    (.51, 1.28, 0), (-.52, 1.01, 0), (.52, 1.01, 0),
    (0, 1.76, .10), (-.045, 1.86, .14), (.045, 1.86, .14),
], dtype=np.float32)
PARENTS = np.array([
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
    12, 13, 14, 16, 17, 18, 19, 15, 15, 15,
    *([20, 25, 26] * 5), *([21, 40, 41] * 5),
], dtype=np.int64)
# The hand joints are unused in these clips. Keep their world rest points near
# the wrists and attach each finger's first joint directly to its wrist.
for side, wrist in ((-1, 20), (1, 21)):
    base = 25 if side == -1 else 40
    for finger in range(5):
        for segment in range(3):
            idx = base + finger * 3 + segment
            REST[idx] = REST[wrist] + np.array([side * .018 * (finger - 2), -.035 * (segment + 1), .025])
            PARENTS[idx] = wrist if segment == 0 else idx - 1

COLORS = {
    "coat": (58, 104, 142), "pants": (38, 55, 75),
    "skin": (214, 169, 132), "shoe": (225, 229, 225),
    "hair": (51, 49, 47), "accent": (233, 117, 74),
}


def sphere(rows=8, cols=12):
    verts = []
    for j in range(rows + 1):
        theta = math.pi * j / rows
        for i in range(cols):
            phi = 2 * math.pi * i / cols
            verts.append((math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)))
    faces = []
    for j in range(rows):
        for i in range(cols):
            a = j * cols + i
            b = j * cols + (i + 1) % cols
            c = (j + 1) * cols + i
            d = (j + 1) * cols + (i + 1) % cols
            if j > 0:
                faces.append((a, c, b))
            if j < rows - 1:
                faces.append((b, c, d))
    return np.asarray(verts, np.float32), np.asarray(faces, np.int32)


UNIT_V, UNIT_F = sphere()


def alignment(z_axis):
    z = z_axis / np.linalg.norm(z_axis)
    candidate = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(candidate, z))) > .95:
        candidate = np.array([0.0, 1.0, 0.0])
    x = np.cross(candidate, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack((x, y, z), axis=1)


def make_fixture():
    verts = []
    faces = []
    colors = []
    weights = []

    def part(a, b, width, depth, tone, stretch=1.25):
        start, end = REST[a], REST[b]
        vector = end - start
        length = np.linalg.norm(vector)
        rotation = alignment(vector)
        local = UNIT_V * np.array([width, depth, length * stretch / 2], np.float32)
        world = local @ rotation.T + (start + end) / 2
        base = len(verts)
        verts.extend(world.tolist())
        faces.extend((UNIT_F + base).tolist())
        colors.extend([COLORS[tone]] * len(UNIT_F))
        for v in world:
            alpha = np.clip(np.dot(v - start, vector) / (length * length), 0, 1)
            w = np.zeros(JOINTS, np.float32)
            w[a], w[b] = 1 - alpha, alpha
            weights.append(w)

    def ball(joint, radii, tone, center_shift=(0, 0, 0)):
        base = len(verts)
        world = UNIT_V * np.asarray(radii, np.float32) + REST[joint] + np.asarray(center_shift, np.float32)
        verts.extend(world.tolist())
        faces.extend((UNIT_F + base).tolist())
        colors.extend([COLORS[tone]] * len(UNIT_F))
        for _ in world:
            w = np.zeros(JOINTS, np.float32)
            w[joint] = 1
            weights.append(w)

    part(0, 3, .21, .12, "coat")
    part(3, 6, .20, .13, "coat")
    part(6, 9, .22, .14, "coat")
    part(9, 12, .095, .09, "skin")
    for hip, knee, ankle, foot in ((1, 4, 7, 10), (2, 5, 8, 11)):
        ball(hip, (.125, .14, .12), "pants")
        part(hip, knee, .115, .115, "pants")
        part(knee, ankle, .086, .09, "pants")
        part(ankle, foot, .084, .075, "shoe", stretch=1.5)
    for collar, shoulder, elbow, wrist in ((13, 16, 18, 20), (14, 17, 19, 21)):
        part(9, shoulder, .13, .11, "coat")
        part(shoulder, elbow, .095, .10, "coat")
        part(elbow, wrist, .065, .068, "skin")
        ball(wrist, (.075, .11, .05), "skin")
    ball(15, (.16, .21, .15), "skin", (0, .08, 0))
    ball(15, (.165, .105, .152), "hair", (0, .20, -.01))
    # Unrendered exact-joint vertices make the synthetic regressor deterministic.
    visible_count = len(verts)
    for j in range(JOINTS):
        verts.append(REST[j].tolist())
        w = np.zeros(JOINTS, np.float32)
        w[j] = 1
        weights.append(w)
    v = np.asarray(verts, np.float32)
    jreg = np.zeros((JOINTS, len(v)), np.float32)
    jreg[np.arange(JOINTS), visible_count + np.arange(JOINTS)] = 1
    return v, np.asarray(faces, np.int32), np.asarray(colors, np.uint8), np.asarray(weights, np.float32), jreg


def run_lbs(clip, frame_indices):
    template, faces, colors, weights, regressor = make_fixture()
    pose = np.zeros((len(frame_indices), JOINTS, 3), np.float32)
    pose[:, 0] = clip["global_orient"][frame_indices]
    pose[:, 1:22] = clip["body_pose"][frame_indices].reshape(-1, 21, 3)
    n = len(template)
    # No learned shape or pose corrections: this is a visible skinning fixture.
    shapedirs = torch.zeros((n, 3, 1), dtype=torch.float32)
    posedirs = torch.zeros(((JOINTS - 1) * 9, n * 3), dtype=torch.float32)
    with torch.no_grad():
        vertices, joints = lbs(
            betas=torch.zeros((len(frame_indices), 1), dtype=torch.float32),
            pose=torch.from_numpy(pose.reshape(len(frame_indices), -1)),
            v_template=torch.from_numpy(template), shapedirs=shapedirs,
            posedirs=posedirs, J_regressor=torch.from_numpy(regressor),
            parents=torch.from_numpy(PARENTS), lbs_weights=torch.from_numpy(weights),
        )
    root = clip["transl"][frame_indices].copy()
    vertices = vertices.numpy() + root[:, None, :]
    joints = joints.numpy() + root[:, None, :]
    return vertices, joints, faces, colors


def project(vertices, camera, target, scale, center):
    forward = target - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rel = vertices - target
    u = rel @ right * scale + center[0]
    v = -rel @ up * scale + center[1]
    depth = (vertices - camera) @ forward
    return np.stack((u, v, depth), axis=1)


def draw_view(draw, vertices, joints, faces, colors, box, camera, target, scale, caption):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill="#dfe7e9", outline="#9baeb7", width=2)
    center = ((x0 + x1) / 2, (y0 + y1) / 2 + 18)
    stage = []
    for x in np.arange(-3.5, 2.01, .5):
        stage.append((np.array([[x, 0, -1.5], [x, 0, 1.5]], np.float32), "#bdcbd0"))
    for z in np.arange(-1.5, 1.51, .5):
        stage.append((np.array([[-3.5, 0, z], [2.0, 0, z]], np.float32), "#bdcbd0"))
    for line, color in stage:
        p = project(line, camera, target, scale, center)
        draw.line((tuple(p[0, :2]), tuple(p[1, :2])), fill=color, width=1)
    p = project(vertices, camera, target, scale, center)
    tri = p[faces]
    depth = tri[:, :, 2].mean(axis=1)
    order = np.argsort(depth)[::-1]
    world_tri = vertices[faces]
    normals = np.cross(world_tri[:, 1] - world_tri[:, 0], world_tri[:, 2] - world_tri[:, 0])
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norm, 1e-7)
    light = np.array([-.35, .8, .48])
    light /= np.linalg.norm(light)
    shade = .54 + .42 * np.maximum(0, normals @ light)
    for k in order:
        coords = [tuple(x) for x in tri[k, :, :2]]
        if max(c[0] for c in coords) < x0 or min(c[0] for c in coords) > x1:
            continue
        if max(c[1] for c in coords) < y0 or min(c[1] for c in coords) > y1:
            continue
        color = tuple(int(max(0, min(255, n * shade[k]))) for n in colors[k])
        draw.polygon(coords, fill=color)
    head = project(joints[[15]], camera, target, scale, center)[0]
    hx, hy = head[:2]
    draw.ellipse((hx - 5, hy - 27, hx + 5, hy - 17), fill="#e95d4a", outline="white", width=1)
    draw.text((x0 + 18, y0 + 17), caption, fill="#244458", font=ImageFont.load_default(size=19))


def render_frame(vertices, joints, faces, colors, case, time_s, duration):
    image = Image.new("RGB", (1180, 650), "#f4f6f5")
    d = ImageDraw.Draw(image)
    d.text((35, 18), "WORLDLINES MOTION LAB  /  SMPL-X LBS TEST", fill="#1f3445", font=ImageFont.load_default(size=28))
    d.text((35, 53), f"{case.replace('_', ' ').upper()}     {time_s:04.2f}s / {duration:.1f}s", fill="#486778", font=ImageFont.load_default(size=19))
    d.text((35, 81), "SELF-AUTHORED TEST BODY  /  NO LICENSED SMPL-X MESH  /  NO MOTIONBRICKS", fill="#ac4a3b", font=ImageFont.load_default(size=15))
    draw_view(d, vertices, joints, faces, colors, (25, 115, 575, 575),
              np.array([0.5, 2.0, 6.0]), np.array([-1.0, .95, 0.0]), 112, "STAGE CAMERA")
    draw_view(d, vertices, joints, faces, colors, (605, 115, 1155, 575),
              np.array([5.5, 2.25, 4.0]), np.array([-1.0, .95, 0.0]), 112, "THREE-QUARTER CAMERA")
    d.rectangle((35, 603, 1145, 609), fill="#cfdbdf")
    d.rectangle((35, 603, 35 + int(1110 * min(time_s / duration, 1)), 609), fill="#e67854")
    d.text((35, 619), "Red marker = head / scalp anchor. Body motion uses upstream SMPL-X linear blend skinning.", fill="#496374", font=ImageFont.load_default(size=14))
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--preview-fps", type=int, default=15)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    clip = load_clip(args.clip)
    if clip["fps"] % args.preview_fps:
        parser.error("preview-fps must divide clip fps")
    step = clip["fps"] // args.preview_fps
    ids = np.arange(0, len(clip["transl"]), step, dtype=np.int32)
    if args.max_frames:
        ids = ids[:args.max_frames]
    vertices, joints, faces, colors = run_lbs(clip, ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scalp_indices = np.argsort(make_fixture()[0][:, 1])[-12:].astype(np.int32)
    scalp_vertices = vertices[:, scalp_indices]
    hair_attachment = {
        "schema": "worldlines.hair_attachment.v1",
        "source": "synthetic_fixture_upstream_lbs",
        "fps": args.preview_fps,
        "space": "Y-up meters",
        "head_position": joints[:, 15].tolist(),
        "head_quaternion_xyzw": head_quaternions(clip)[ids].tolist(),
        "scalp_indices": scalp_indices.tolist(),
        "scalp_vertices": scalp_vertices.tolist(),
    }
    (args.output_dir / f"{clip['case']}.hair_attachment.json").write_text(
        json.dumps(hair_attachment, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(args.output_dir / f"{clip['case']}.synthetic_lbs.npz",
                        vertices=vertices.astype(np.float32), joints=joints.astype(np.float32),
                        scalp_vertices=scalp_vertices.astype(np.float32), faces=faces,
                        fps=args.preview_fps, source="synthetic_fixture_upstream_lbs")
    with tempfile.TemporaryDirectory(prefix="worldlines-lbs-frames-") as folder:
        for k, frame in enumerate(ids):
            image = render_frame(vertices[k], joints[k], faces, colors, clip["case"],
                                 frame / clip["fps"], len(clip["transl"]) / clip["fps"])
            image.save(Path(folder) / f"{k:04d}.png")
            if k in (0, len(ids) // 2, len(ids) - 1):
                image.save(args.output_dir / f"{clip['case']}.frame-{k:03d}.png")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-framerate", str(args.preview_fps), "-i", str(Path(folder) / "%04d.png"),
                        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                        str(args.output_dir / f"{clip['case']}.synthetic_lbs.mp4")], check=True)
    print(args.output_dir / f"{clip['case']}.synthetic_lbs.mp4")


if __name__ == "__main__":
    main()
