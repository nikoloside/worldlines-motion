"""Verify the bridge preserves pose, root motion, and add-on NPZ contract."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from blender_smplx import addon_arrays, export_clip
from smplx_bridge import CASES, load_clip, make_clip, save_clip


class BlenderExportTests(unittest.TestCase):
    def test_all_actions_pack_55_joints_without_changing_body_motion(self):
        for case in CASES:
            with self.subTest(case=case):
                clip = make_clip(case)
                result = addon_arrays(clip)
                frames = len(clip["transl"])
                self.assertEqual(result["poses"].shape, (frames, 165))
                np.testing.assert_array_equal(result["poses"][:, :3], clip["global_orient"])
                np.testing.assert_array_equal(result["poses"][:, 3:66], clip["body_pose"])
                np.testing.assert_array_equal(result["poses"][:, 66:], 0)
                np.testing.assert_array_equal(result["trans"], clip["transl"])
                self.assertEqual(result["gender"].item(), "male")
                self.assertEqual(int(result["mocap_framerate"]), 30)

    def test_saved_npz_loads_without_pickle_and_has_official_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.npz"
            destination = Path(directory) / "blender.npz"
            save_clip(make_clip("startle_jump"), source)
            export_clip(source, destination)
            with np.load(destination, allow_pickle=False) as data:
                self.assertEqual(set(data.files), {"poses", "trans", "betas", "gender", "mocap_framerate"})
                self.assertEqual(data["poses"].shape, (120, 165))
                self.assertGreater(float(data["trans"][57, 1]), 0.25)
                self.assertEqual(str(data["gender"]), "male")


if __name__ == "__main__":
    unittest.main()
