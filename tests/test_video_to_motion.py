import unittest

import numpy as np

from tests._helpers import model_or_skip
from worldlines_motion import clip as C
from worldlines_motion.source.gvhmr_client import gvhmr_to_clip
from worldlines_motion.source.landmarks_to_smplx import MP, LandmarkSolver

# SMPL-X joint -> MediaPipe landmark used to fabricate landmarks from a posed skeleton.
SMPLX_TO_MP = {16: "l_shoulder", 17: "r_shoulder", 18: "l_elbow", 19: "r_elbow", 20: "l_wrist", 21: "r_wrist",
               1: "l_hip", 2: "r_hip", 4: "l_knee", 5: "r_knee", 7: "l_ankle", 8: "r_ankle", 10: "l_foot", 11: "r_foot"}


class GvhmrClientTests(unittest.TestCase):
    def test_payload_to_clip_faces_camera(self):
        n = 5
        payload = {"n": n, "fps": 30, "global_orient": [0, np.pi / 2, 0] * n, "body_pose": [0.0] * 63 * n,
                   "transl": [float(i) for i in range(n) for _ in range(3)], "incam_global_orient0": [np.pi, 0, 0]}  # facing the camera
        clip = gvhmr_to_clip(payload)
        self.assertEqual(C.validate(clip), n)
        R = C.rotvec_to_matrix(clip["global_orient"][0].astype(np.float64))
        fwd = R @ np.array([0, 0, 1.0])
        self.assertAlmostEqual(float(fwd[2]), 1.0, places=4)  # frame 0 faces +Z
        self.assertEqual(clip["source"], "gvhmr")


class LandmarkSolverTests(unittest.TestCase):
    def test_recovers_pose_from_synthetic_landmarks(self):
        model = model_or_skip()
        solver = LandmarkSolver(model)
        rng = np.random.default_rng(3)
        pose = np.zeros((1, 55, 3))
        pose[0, 4, 0] = 0.9     # left knee flexion
        pose[0, 18, 1] = -1.1   # left elbow flexion
        pose[0, 19, 1] = 1.1
        pose[0, 16, 2] = -0.8   # arms down
        pose[0, 17, 2] = 0.8
        pose[0, 1, 0] = -0.5    # left hip forward
        joints = model.joints(solver.rest_verts, pose)[0]
        lm = np.zeros((33, 3))
        for j, name in SMPLX_TO_MP.items():
            lm[MP[name]] = joints[j]
        head = joints[15]
        lm[MP["l_ear"]] = head + [0.07, 0.0, -0.02]; lm[MP["r_ear"]] = head + [-0.07, 0.0, -0.02]
        lm[MP["l_eye"]] = head + [0.03, 0.02, 0.08]; lm[MP["r_eye"]] = head + [-0.03, 0.02, 0.08]
        lm[MP["nose"]] = head + [0, 0, 0.1]
        lm += rng.normal(0, 0.002, lm.shape)
        go, bp = solver.solve(lm)
        rec = np.zeros((1, 55, 3)); rec[0, 0] = go; rec[0, 1:22] = bp.reshape(21, 3)
        rj = model.joints(solver.rest_verts, rec)[0]
        for j in (4, 7, 18, 20, 21):
            self.assertLess(float(np.linalg.norm(rj[j] - joints[j])), 0.04, f"joint {j}")


if __name__ == "__main__":
    unittest.main()
