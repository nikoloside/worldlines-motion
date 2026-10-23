"""Make a local walk/squat review clip with relaxed arms and male shape betas.

The regional physique sculpt is applied later by prepare_official_smplx.py;
the ten beta values alone do not reproduce the final review mesh.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from muscular_profile import MUSCULAR_MALE_BETAS
from smplx_bridge import load_clip, save_clip


def make_clip(source: Path, output: Path) -> None:
    clip = load_clip(source)
    if clip["case"] != "walk_squat":
        raise ValueError("This demo pose adjustment expects walk_squat")
    pose = clip["body_pose"].copy().reshape(-1, 21, 3)
    # Source clip starts in a T-pose. Relax the arms for a readable human demo.
    pose[:, 15, 2] -= 1.12
    pose[:, 16, 2] += 1.12
    pose[:, 17, 0] += 0.12
    pose[:, 18, 0] += 0.12
    clip["body_pose"] = pose.reshape(-1, 63)
    clip["betas"] = MUSCULAR_MALE_BETAS.copy()
    clip["source"] = "procedural_muscular_demo"
    save_clip(clip, output)
    print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    make_clip(args.source, args.output)


if __name__ == "__main__":
    main()
