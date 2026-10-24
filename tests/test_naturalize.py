import unittest

import numpy as np

from tests._helpers import model_or_skip
from worldlines_motion import clip as C
from worldlines_motion.naturalize import B, JOINT_LIMITS, NaturalizeConfig, ground, joint_limits, metrics, naturalize


def walking_clip(n=90, fps=30.0, drift=0.25):
    """March in place (one foot always planted) while the root drifts sideways: pure foot skate."""
    c = C.new_clip(n, fps, "synthetic")
    t = np.arange(n) / fps
    bp = c["body_pose"].reshape(n, 21, 3)
    phase = np.sin(2 * np.pi * t / 1.0)
    left, right = np.clip(phase, 0, None), np.clip(-phase, 0, None)
    bp[:, B["left_hip"], 0] = -0.8 * left
    bp[:, B["left_knee"], 0] = 1.4 * left
    bp[:, B["right_hip"], 0] = -0.8 * right
    bp[:, B["right_knee"], 0] = 1.4 * right
    bp[:, B["left_shoulder"], 2] = -1.2
    bp[:, B["right_shoulder"], 2] = 1.2
    c["body_pose"] = bp.reshape(n, 63)
    c["transl"][:, 0] = drift * t
    return c


class NaturalizeTests(unittest.TestCase):
    def test_smoothing_reduces_jerk_without_model(self):
        rng = np.random.default_rng(0)
        c = walking_clip()
        noisy = C.copy_clip(c)
        noisy["body_pose"] = (noisy["body_pose"] + rng.normal(0, 0.05, noisy["body_pose"].shape)).astype(np.float32)
        cfg = NaturalizeConfig(ground=False, foot_lock=False)
        out = naturalize(noisy, cfg, model=None)
        self.assertLess(metrics(out)["angular_jerk"], metrics(noisy)["angular_jerk"] * 0.6)
        self.assertLess(np.abs(out["body_pose"] - c["body_pose"]).mean(), np.abs(noisy["body_pose"] - c["body_pose"]).mean())
        self.assertIn("naturalize", out)

    def test_joint_limits_clamp_hyperextension(self):
        c = C.new_clip(4)
        bp = c["body_pose"].reshape(4, 21, 3)
        bp[:, B["left_knee"], 0] = -1.0      # knee bent the wrong way
        bp[:, B["right_elbow"], 1] = -1.5    # elbow past straight
        c["body_pose"] = bp.reshape(4, 63)
        out = joint_limits(c, NaturalizeConfig())
        bp2 = out["body_pose"].reshape(4, 21, 3)
        self.assertGreaterEqual(float(bp2[0, B["left_knee"], 0]), JOINT_LIMITS["left_knee"][0][0] - 1e-6)
        self.assertGreaterEqual(float(bp2[0, B["right_elbow"], 1]), JOINT_LIMITS["right_elbow"][1][0] - 1e-6)

    def test_ground_and_foot_lock_with_model(self):
        model = model_or_skip()
        c = walking_clip()
        c["transl"][:, 1] = 0.3  # floating
        out = naturalize(c, NaturalizeConfig(), model)
        m = metrics(out, model)
        self.assertLess(abs(m["min_foot_height_m"]), 0.02)
        self.assertEqual(out["contacts"].shape, (90, 4))
        before = metrics(ground(c, NaturalizeConfig(), model), model)
        self.assertGreater(before["contact_ratio"], 0.05)
        self.assertLess(m["contact_skate_m_per_s"], before["contact_skate_m_per_s"] * 0.7)


if __name__ == "__main__":
    unittest.main()
