"""Give a local SMPL-X character scene smooth shading and a closer review view.

blender -b input.blend --python polish_muscular_scene.py -- output.blend portrait.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def point_camera(camera: bpy.types.Object, position: tuple[float, float, float],
                 target: tuple[float, float, float]) -> None:
    camera.location = position
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("portrait", type=Path)
    args = parser.parse_args(argv)
    scene = bpy.context.scene
    body = bpy.data.objects["SMPL-X body — full frame-corrected skin"]
    camera = scene.camera
    for polygon in body.data.polygons:
        polygon.use_smooth = True

    scene.frame_set(min(scene.frame_end, 158))
    point_camera(camera, (4.6, -5.3, 2.65), (0.0, 0.0, 0.94))
    camera.data.ortho_scale = 2.55
    args.portrait.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.portrait.resolve())
    bpy.ops.render.render(write_still=True)

    point_camera(camera, (4.1, -7.8, 3.0), (0.0, 0.0, 0.94))
    camera.data.ortho_scale = 4.1
    scene.frame_set(min(scene.frame_end, 104))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
    print(f"Saved polished scene: {args.output}")


if __name__ == "__main__":
    main()
