"""Solve 3-D body landmarks (MediaPipe Pose layout) onto the SMPL-X skeleton.

Per frame: pelvis and chest frames from hip/shoulder lines, limbs by aiming
each bone at its measured direction (upper limbs also use the bend plane so
elbows and knees hinge the right way), head from ears/eyes. Output is local
axis-angle per SMPL-X joint, i.e. ``global_orient`` + ``body_pose``.

Landmark frame: metres, Y up, +Z toward the camera (subject facing the camera
faces +Z, like the SMPL-X rest pose). Convert MediaPipe world landmarks with
``(x, -y, -z)``.
"""

from __future__ import annotations

import numpy as np

from ..clip import matrix_to_rotvec
from ..smplx_model import SMPLX

MP = dict(nose=0, l_eye=2, r_eye=5, l_ear=7, r_ear=8, l_shoulder=11, r_shoulder=12, l_elbow=13, r_elbow=14,
          l_wrist=15, r_wrist=16, l_pinky=17, r_pinky=18, l_index=19, r_index=20, l_hip=23, r_hip=24,
          l_knee=25, r_knee=26, l_ankle=27, r_ankle=28, l_heel=29, r_heel=30, l_foot=31, r_foot=32)

# SMPL-X joints
PELVIS, L_HIP, R_HIP, SPINE1, L_KNEE, R_KNEE, SPINE2, L_ANKLE, R_ANKLE, SPINE3, L_FOOT, R_FOOT, NECK, \
    L_COLLAR, R_COLLAR, HEAD, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = range(22)
PARENT = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _basis(x: np.ndarray, y_hint: np.ndarray) -> np.ndarray:
    """Rotation matrix whose columns are X along x, Y close to y_hint, Z = X x Y."""
    X = _unit(x)
    Z = _unit(np.cross(X, y_hint))
    Y = np.cross(Z, X)
    return np.stack([X, Y, Z], axis=1)


def _shortest_arc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = _unit(a), _unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        axis = _unit(np.cross(a, [1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = _unit(np.cross(a, [0.0, 1.0, 0.0]))
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * K @ K
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def _slerp_mat(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    from ..clip import rotvec_to_matrix
    rel = matrix_to_rotvec(a.T @ b)
    return a @ rotvec_to_matrix(rel * t)


class LandmarkSolver:
    def __init__(self, model: SMPLX, betas: np.ndarray | None = None):
        self.model = model
        self.rest_verts = model.shaped(betas)
        self.rest = model.joints_of(self.rest_verts)[:22].astype(np.float64)
        R = self.rest
        self.rest_dir = {j: _unit(R[c] - R[j]) for j, c in ((L_HIP, L_KNEE), (R_HIP, R_KNEE), (L_KNEE, L_ANKLE), (R_KNEE, R_ANKLE),
                                                         (L_ANKLE, L_FOOT), (R_ANKLE, R_FOOT), (L_SHOULDER, L_ELBOW), (R_SHOULDER, R_ELBOW),
                                                         (L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST), (NECK, HEAD))}
        up = _unit((R[L_SHOULDER] + R[R_SHOULDER]) / 2 - (R[L_HIP] + R[R_HIP]) / 2)
        self.rest_pelvis = _basis(R[L_HIP] - R[R_HIP], up)
        self.rest_chest = _basis(R[L_SHOULDER] - R[R_SHOULDER], up)
        self.rest_head = self.rest_chest.copy()
        self.forward = np.array([0.0, 0.0, 1.0])

    # ------------------------------------------------------------------------------------
    def solve(self, lm: np.ndarray, vis: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """lm [33,3] -> (global_orient[3], body_pose[63])."""
        lm = np.asarray(lm, np.float64)
        vis = np.ones(33) if vis is None else np.asarray(vis)
        ok = lambda *ids: all(vis[MP[i]] > 0.35 for i in ids)  # noqa: E731
        P = lambda k: lm[MP[k]]  # noqa: E731
        mid_hip = (P("l_hip") + P("r_hip")) / 2
        mid_sh = (P("l_shoulder") + P("r_shoulder")) / 2
        up = mid_sh - mid_hip
        G = [np.eye(3) for _ in range(22)]
        pelvis = _basis(P("l_hip") - P("r_hip"), up) @ self.rest_pelvis.T
        chest = _basis(P("l_shoulder") - P("r_shoulder"), up) @ self.rest_chest.T
        G[PELVIS] = pelvis
        G[SPINE1] = _slerp_mat(pelvis, chest, 0.35)
        G[SPINE2] = _slerp_mat(pelvis, chest, 0.7)
        G[SPINE3] = chest
        G[L_COLLAR] = G[R_COLLAR] = chest
        # head
        if ok("l_ear", "r_ear", "l_eye", "r_eye"):
            ear_x = P("l_ear") - P("r_ear")
            fwd = (P("l_eye") + P("r_eye")) / 2 - (P("l_ear") + P("r_ear")) / 2
            head = _basis(ear_x, np.cross(fwd, ear_x)) @ self.rest_head.T
        else:
            head = chest
        G[NECK] = _slerp_mat(chest, head, 0.4)
        G[HEAD] = head
        chest_fwd = chest @ self.forward
        pelvis_fwd = pelvis @ self.forward
        # limbs
        for side, s in (("l", "L"), ("r", "R")):
            sh, el, wr = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
            hp, kn, an, ft = f"{side}_hip", f"{side}_knee", f"{side}_ankle", f"{side}_foot"
            SH, EL, WR = (L_SHOULDER, L_ELBOW, L_WRIST) if side == "l" else (R_SHOULDER, R_ELBOW, R_WRIST)
            HP, KN, AN = (L_HIP, L_KNEE, L_ANKLE) if side == "l" else (R_HIP, R_KNEE, R_ANKLE)
            if ok(sh, el):
                G[SH] = self._aim_limb(SH, P(el) - P(sh), P(wr) - P(el) if ok(wr) else None, self.forward, chest_fwd)
            else:
                G[SH] = chest
            G[EL] = self._aim(EL, G[SH], P(wr) - P(el)) if ok(el, wr) else G[SH]
            G[WR] = G[EL]
            if ok(hp, kn):
                G[HP] = self._aim_limb(HP, P(kn) - P(hp), P(an) - P(kn) if ok(an) else None, -self.forward, -pelvis_fwd)
            else:
                G[HP] = pelvis
            G[KN] = self._aim(KN, G[HP], P(an) - P(kn)) if ok(kn, an) else G[HP]
            G[AN] = self._aim(AN, G[KN], P(ft) - P(an)) if ok(an, ft) else G[KN]
            G[L_FOOT if side == "l" else R_FOOT] = G[AN]
        # globals -> locals
        local = np.zeros((22, 3))
        for j in range(22):
            Lj = G[j] if PARENT[j] < 0 else G[PARENT[j]].T @ G[j]
            local[j] = matrix_to_rotvec(Lj)
        return local[0].astype(np.float32), local[1:].reshape(63).astype(np.float32)

    def _aim(self, joint: int, parent_global: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Global rotation for `joint` so its rest bone points along `target` (zero extra twist)."""
        if np.linalg.norm(target) < 1e-6:
            return parent_global
        current = parent_global @ self.rest_dir[joint]
        return _shortest_arc(current, target) @ parent_global

    def _aim_limb(self, joint: int, upper: np.ndarray, lower: np.ndarray | None, rest_bend: np.ndarray, fallback_bend: np.ndarray) -> np.ndarray:
        """Upper arm/thigh: aim along `upper`; recover twist from the bend plane when the limb is bent."""
        u = _unit(upper)
        swing = _shortest_arc(self.rest_dir[joint], u)  # absolute (from rest), no twist
        if lower is None:
            return swing
        l = _unit(lower)
        bend = l - u * np.dot(l, u)
        amount = np.linalg.norm(bend)
        if amount < 0.15:
            return swing
        rest_axis = self.rest_dir[joint]
        rb = rest_bend - rest_axis * np.dot(rest_bend, rest_axis)
        if np.linalg.norm(rb) < 1e-6:
            return swing
        q_rest = _basis(rest_axis, rb)
        q_now = _basis(u, bend / amount)
        plane = q_now @ q_rest.T
        w = min(1.0, max(0.0, (amount - 0.15) / 0.3))
        w = w * w * (3 - 2 * w)
        return _slerp_mat(swing, plane, w)
