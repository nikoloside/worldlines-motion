"""Behavior checks for the synthetic visual fixture using upstream SMPL-X LBS."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from smplx_bridge import load_clip


@unittest.skipUnless(importlib.util.find_spec("smplx") and importlib.util.find_spec("torch"), "smplx and torch not installed")
class SyntheticLBSTests(unittest.TestCase):
    def test_walk_squat_reaches_stage_and_lowers_head_with_feet_near_floor(self):
        from render_lbs_demo import run_lbs

        clip = load_clip(Path(__file__).with_name("walk_squat.smplx.npz"))
        frames = np.array([0, 72, 105, 179])  # start, stopped, squat, finish
        vertices, joints, faces, _ = run_lbs(clip, frames)
        self.assertEqual(vertices.shape[0], 4)
        self.assertGreater(len(faces), 100)
        self.assertGreater(float(joints[1, 0, 0] - joints[0, 0, 0]), 2.4)
        self.assertGreater(float(joints[1, 15, 1] - joints[2, 15, 1]), 0.35)
        self.assertTrue(np.all(np.abs(joints[2, [10, 11], 1]) < 0.15))
        self.assertAlmostEqual(float(joints[1, 15, 1]), float(joints[3, 15, 1]), delta=0.03)

    def test_startle_jump_leaves_ground_and_lands(self):
        from render_lbs_demo import run_lbs

        clip = load_clip(Path(__file__).with_name("startle_jump.smplx.npz"))
        frames = np.array([0, 57, 119])
        _, joints, _, _ = run_lbs(clip, frames)
        self.assertGreater(float(joints[1, 15, 1] - joints[0, 15, 1]), 0.25)
        self.assertGreater(float(joints[1, 10, 1] - joints[0, 10, 1]), 0.25)
        self.assertAlmostEqual(float(joints[-1, 10, 1]), float(joints[0, 10, 1]), delta=0.03)


if __name__ == "__main__":
    unittest.main()
