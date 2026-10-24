import tempfile
import unittest
from pathlib import Path

import numpy as np

from worldlines_motion import clip as C


class ClipTests(unittest.TestCase):
    def test_roundtrip_and_json(self):
        c = C.new_clip(12, 24.0, "test")
        c["body_pose"][:, 3] = np.linspace(0, 1, 12)
        c["physique"] = {"bulge": 1.0}
        with tempfile.TemporaryDirectory() as d:
            p = C.save_clip(c, Path(d) / "a.npz")
            back = C.load_clip(p)
        self.assertEqual(back["physique"], {"bulge": 1.0})
        np.testing.assert_allclose(back["body_pose"], c["body_pose"])
        j = C.to_json(back)
        again = C.from_json(j)
        np.testing.assert_allclose(again["body_pose"], c["body_pose"], atol=1e-5)
        self.assertEqual(again["fps"], 24.0)

    def test_validate_rejects_bad_shapes(self):
        c = C.new_clip(3)
        c["body_pose"] = np.zeros((3, 60), np.float32)
        with self.assertRaises(ValueError):
            C.validate(c)

    def test_rotation_helpers(self):
        rng = np.random.default_rng(1)
        v = rng.normal(size=(50, 3))
        v = v / np.linalg.norm(v, axis=1, keepdims=True) * rng.uniform(0.0, 3.0, (50, 1))  # angles below pi
        q = C.rotvec_to_quat(v)
        np.testing.assert_allclose(C.quat_to_rotvec(q), v, atol=1e-6)
        m = C.rotvec_to_matrix(v)
        np.testing.assert_allclose(C.matrix_to_rotvec(m), v, atol=1e-6)
        np.testing.assert_allclose(np.einsum("nij,nkj->nik", m, m), np.broadcast_to(np.eye(3), (50, 3, 3)), atol=1e-6)
        a, b = C.rotvec_to_quat(v[:1]), C.rotvec_to_quat(v[1:2])
        mab = C.rotvec_to_matrix(v[:1]) @ C.rotvec_to_matrix(v[1:2])
        np.testing.assert_allclose(C.rotvec_to_matrix(C.quat_to_rotvec(C.quat_mul(a, b))), mab, atol=1e-6)

    def test_resample_and_concat(self):
        a = C.new_clip(31, 30.0)
        a["transl"][:, 0] = np.linspace(0, 1, 31)
        b = C.resample(a, 15.0)
        self.assertEqual(len(b["transl"]), 16)
        self.assertAlmostEqual(float(b["transl"][-1, 0]), 1.0, places=5)
        c = C.concat([a, a], blend_frames=4)
        self.assertEqual(len(c["transl"]), 31 + 31 - 4)
        self.assertGreater(float(c["transl"][-1, 0]), 1.9)
        self.assertTrue(np.all(np.diff(c["transl"][:, 0]) >= -1e-6))


if __name__ == "__main__":
    unittest.main()
