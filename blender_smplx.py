"""Export Worldlines test clips for the official SMPL-X Blender add-on.

The source clips use SMPL-X's Y-up world coordinates. In the add-on's
Add Animation dialog, select the AMASS orientation and Locked Head model.
No SMPL-X model data is bundled here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from smplx_bridge import load_clip, validate


SMPLX_JOINTS = 55
BODY_WITH_ROOT = 22 * 3


def addon_arrays(clip: dict, gender: str = "male") -> dict[str, np.ndarray]:
    """Pack a validated clip into the fields consumed by Add Animation."""
    frames = validate(clip)
    if gender not in {"male", "female", "neutral"}:
        raise ValueError("gender must be male, female, or neutral")

    poses = np.zeros((frames, SMPLX_JOINTS * 3), dtype=np.float32)
    poses[:, :3] = clip["global_orient"]
    poses[:, 3:BODY_WITH_ROOT] = clip["body_pose"]
    return {
        "poses": poses,
        "trans": np.asarray(clip["transl"], dtype=np.float32).copy(),
        "betas": np.asarray(clip["betas"], dtype=np.float32).copy(),
        "gender": np.asarray(gender),
        "mocap_framerate": np.asarray(int(clip["fps"]), dtype=np.int32),
    }


def export_clip(source: Path, destination: Path, gender: str = "male") -> None:
    arrays = addon_arrays(load_clip(source), gender)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Worldlines .smplx.npz")
    parser.add_argument("destination", type=Path, help="Blender add-on .npz")
    parser.add_argument("--gender", choices=("male", "female", "neutral"), default="male")
    args = parser.parse_args()
    export_clip(args.source, args.destination, args.gender)
    print(f"Wrote {args.destination} (SMPL-X 55 joints, Y-up; import with AMASS orientation)")


if __name__ == "__main__":
    main()
