"""Exercise the mesh and hair export boundary against the real SMPL-X API module.

The model object is replaced because registered model assets are not bundled.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from smplx_bridge import load_clip, render_model


@unittest.skipUnless(importlib.util.find_spec("smplx") and importlib.util.find_spec("torch"), "smplx and torch not installed")
class ExportContractTests(unittest.TestCase):
    def test_mesh_and_hair_attachment_frame_alignment(self):
        import smplx
        import torch

        class TestModel:
            faces = np.array([[0, 1, 2]], dtype=np.int32)

            def eval(self):
                return self

            def __call__(self, *, transl, **kwargs):
                base = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]])
                vertices = base + transl[:, None, :]
                joints = torch.zeros((1, 16, 3), dtype=torch.float32)
                joints[:, 15, :] = transl + torch.tensor([[0.0, 1.5, 0.0]])
                return type("Output", (), {"vertices": vertices, "joints": joints})()

        clip = load_clip(Path(__file__).resolve().parents[2] / "demos" / "clips" / "startle_jump.smplx.npz")
        with tempfile.TemporaryDirectory(prefix="worldlines-smplx-test-") as folder:
            root = Path(folder)
            model_file = root / "SMPLX_MALE.npz"
            model_file.touch()
            scalp_file = root / "scalp.npy"
            np.save(scalp_file, np.array([0, 2], dtype=np.int32))
            with patch.object(smplx, "create", return_value=TestModel()):
                render_model(clip, model_file, root / "out", "male", scalp_file)
            with np.load(root / "out" / "smplx_animation.npz") as animation:
                self.assertEqual(animation["vertices"].shape, (120, 3, 3))
                self.assertEqual(animation["faces"].shape, (1, 3))
                self.assertEqual(int(animation["fps"]), 30)
            hair = json.loads((root / "out" / "hair_attachment.json").read_text())
            self.assertEqual(len(hair["head_position"]), 120)
            self.assertEqual(len(hair["head_quaternion_xyzw"]), 120)
            self.assertEqual(len(hair["scalp_vertices"]), 120)
            self.assertEqual(len(hair["scalp_vertices"][0]), 2)


if __name__ == "__main__":
    unittest.main()
