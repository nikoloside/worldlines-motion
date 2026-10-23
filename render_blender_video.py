"""Render a local Blender motion scene to a compact review MP4.

blender -b scene.blend --python render_blender_video.py -- preview.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-fps", type=int, default=30)
    parser.add_argument("--preview-fps", type=int, default=15)
    parser.add_argument("--resolution-percent", type=int, default=75)
    args = parser.parse_args(argv)
    if args.source_fps % args.preview_fps:
        parser.error("preview-fps must divide source-fps")
    scene = bpy.context.scene
    if scene.render.fps != args.source_fps:
        raise ValueError(f"Scene fps is {scene.render.fps}, expected {args.source_fps}")
    scene.frame_step = args.source_fps // args.preview_fps
    scene.render.fps = args.preview_fps
    scene.render.resolution_percentage = args.resolution_percent
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.output.resolve())
    bpy.ops.render.render(animation=True)
    print(f"Rendered {args.output}")


if __name__ == "__main__":
    main()
