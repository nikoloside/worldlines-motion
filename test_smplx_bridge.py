"""Contract and action timing checks; real model smoke test needs model assets."""

import unittest

import numpy as np

from smplx_bridge import CASES, head_quaternions, load_clip, make_clip, validate


class BridgeTests(unittest.TestCase):
    def test_all_cases_have_smplx_layout_and_finite_head_orientation(self):
        for case, seconds in CASES.items():
            with self.subTest(case=case):
                clip = make_clip(case)
                self.assertEqual(validate(clip), round(seconds * clip["fps"]))
                self.assertEqual(clip["body_pose"].shape[1], 63)
                q = head_quaternions(clip)
                np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1, atol=1e-6)

    def test_action_landmarks(self):
        squat = make_clip("walk_squat")
        self.assertLess(squat["transl"][0, 0], -2.0)
        self.assertAlmostEqual(float(squat["transl"][-1, 0]), 0, places=3)
        self.assertLess(float(squat["transl"][round(3.5 * squat["fps"]), 1]), -0.2)
        jump = make_clip("startle_jump")
        self.assertGreater(float(jump["transl"][round(1.9 * jump["fps"]), 1]), 0.25)
        self.assertAlmostEqual(float(jump["transl"][0, 1]), 0, places=4)
        self.assertAlmostEqual(float(jump["transl"][-1, 1]), 0, places=4)

    def test_saved_clip_round_trip(self):
        from pathlib import Path
        for case in CASES:
            path = Path(__file__).with_name(f"{case}.smplx.npz")
            if path.exists():
                self.assertEqual(validate(load_clip(path)), round(CASES[case] * 30))


if __name__ == "__main__":
    unittest.main()
