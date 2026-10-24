import unittest

import numpy as np

from tests._helpers import model_or_skip
from worldlines_motion.physique import PRESETS, bulge_region, make_rest


class PhysiqueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = model_or_skip()

    def test_bulge_is_local_frontal_and_symmetric(self):
        m = self.model
        neutral, _ = make_rest(m, {})
        bulged, profile = make_rest(m, {"bulge": 1.0})
        self.assertEqual(profile["bulge"], 1.0)
        d = np.linalg.norm(bulged - neutral, axis=1)
        moved = np.flatnonzero(d > 1e-3)
        self.assertGreater(len(moved), 20)
        self.assertLess(len(moved), 400)
        joints = m.joints_of(neutral)
        hip_mid = (joints[1] + joints[2]) / 2
        pts = neutral[moved]
        self.assertTrue(np.all(np.abs(pts[:, 0] - hip_mid[0]) < 0.08))      # across hips
        self.assertTrue(np.all(pts[:, 1] < hip_mid[1] + 0.08))               # at/below hip joints
        self.assertTrue(np.all(pts[:, 2] > joints[0][2]))                    # front of the pelvis
        self.assertGreater(float((bulged - neutral)[moved][:, 2].mean()), 0)  # outward is forward
        peak = float(d.max())
        self.assertTrue(0.01 < peak < 0.05, peak)
        if m.vert_sym is not None:                                            # left/right mirror
            mirrored = d[m.vert_sym]
            self.assertLess(float(np.abs(mirrored - d).max()), 2e-3)

    def test_profiles_keep_topology_and_scale(self):
        m = self.model
        for name, profile in PRESETS.items():
            v, p = make_rest(m, profile)
            self.assertEqual(v.shape, (10475, 3), name)
            self.assertTrue(np.all(np.isfinite(v)))
            self.assertEqual(len(p["betas"]), 10)
        region = bulge_region(m.shaped(None), m.J_regressor, m.weights, size=1.0)
        self.assertAlmostEqual(float(region.max()), 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
