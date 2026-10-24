import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests._helpers import model_or_skip
from worldlines_motion.clip import load_clip, save_clip, validate
from worldlines_motion.export.blender_addon import addon_arrays, export_clip
from worldlines_motion.export.mesh import bake_mesh
from worldlines_motion.source.procedural import CASES, make_clip


class ProceduralTests(unittest.TestCase):
    def test_cases_validate_and_time(self):
        for case, seconds in CASES.items():
            clip = make_clip(case)
            self.assertEqual(validate(clip), round(seconds * 30))
            self.assertEqual(clip["case"], case)
        jump = make_clip("startle_jump")
        self.assertGreater(float(jump["transl"][round(1.9 * jump["fps"]), 1]), 0.25)
        self.assertAlmostEqual(float(jump["transl"][-1, 1]), 0, places=4)

    def test_matches_shipped_demo_clip(self):
        shipped = Path(__file__).resolve().parents[1] / "demos" / "clips" / "walk_squat.smplx.npz"
        if not shipped.exists():
            self.skipTest("demo clip missing")
        a, b = make_clip("walk_squat"), load_clip(shipped)
        for k in ("transl", "global_orient", "body_pose"):
            np.testing.assert_allclose(a[k], b[k], atol=1e-6)


class BlenderAddonTests(unittest.TestCase):
    def test_addon_arrays_contract(self):
        clip = make_clip("walk_startle")
        arrays = addon_arrays(clip, "male")
        n = len(clip["transl"])
        self.assertEqual(arrays["poses"].shape, (n, 165))
        np.testing.assert_array_equal(arrays["poses"][:, :3], clip["global_orient"])
        np.testing.assert_array_equal(arrays["poses"][:, 3:66], clip["body_pose"])
        self.assertTrue(np.all(arrays["poses"][:, 66:] == 0))
        self.assertEqual(int(arrays["mocap_framerate"]), 30)
        with self.assertRaises(ValueError):
            addon_arrays(clip, "robot")

    def test_export_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            src = save_clip(make_clip("startle_jump"), Path(d) / "in.smplx.npz")
            dst = Path(d) / "out.blender_smplx.npz"
            export_clip(src, dst)
            with np.load(dst) as z:
                self.assertEqual(set(z.files), {"poses", "trans", "betas", "gender", "mocap_framerate"})
                self.assertEqual(str(z["gender"]), "male")


class MeshBakeTests(unittest.TestCase):
    def test_bake_writes_animation_and_hair(self):
        model = model_or_skip()
        clip = make_clip("startle_jump")
        with tempfile.TemporaryDirectory() as d:
            out = bake_mesh(clip, model, d, step=30)
            with np.load(out / "smplx_animation.npz") as z:
                self.assertEqual(z["vertices"].shape[1:], (10475, 3))
                self.assertEqual(z["vertices"].shape[0], 4)
                self.assertAlmostEqual(float(z["fps"]), 1.0)
            self.assertTrue((out / "hair_attachment.json").exists())


if __name__ == "__main__":
    unittest.main()
