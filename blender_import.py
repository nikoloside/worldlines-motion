"""Run inside Blender with the official SMPL-X for Blender extension enabled.

blender -b --python blender_import.py -- clip.blender_smplx.npz output.blend

This uses the add-on's own Add Animation operator, mesh, rig, skin weights,
and corrective pose shapes. It does not create a substitute SMPL-X body.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np


def main() -> None:
    args_after_separator = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--correctives", action="store_true", help="Keyframe pose corrective weights on every frame")
    args = parser.parse_args(args_after_separator)

    with np.load(args.clip, allow_pickle=False) as data:
        required = {"poses", "trans", "betas", "gender", "mocap_framerate"}
        if not required.issubset(data.files):
            raise ValueError(f"Missing add-on NPZ fields: {required - set(data.files)}")
        frames = len(data["trans"])
        if data["poses"].shape != (frames, 165) or data["trans"].shape != (frames, 3):
            raise ValueError("Expected 55 SMPL-X joints and matching root translation frames")
        fps = int(data["mocap_framerate"])

    if not hasattr(bpy.ops.object, "smplx_add_animation"):
        raise RuntimeError("Enable the official 'SMPL-X for Blender' extension with its model data first")

    result = bpy.ops.object.smplx_add_animation(
        filepath=str(args.clip.resolve()),
        anim_format="AMASS",  # source coordinates are Y-up
        rest_position="SMPL-X",
        hand_reference="FLAT",
        target_framerate=fps,
        keyframe_corrective_pose_weights=args.correctives,
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"SMPL-X add-on import failed: {result}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
    print(f"Saved {frames} frames at {fps} fps to {args.output}")


if __name__ == "__main__":
    main()
