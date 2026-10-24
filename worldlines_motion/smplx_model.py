"""Minimal numpy SMPL-X: load the official NPZ and run linear blend skinning.

No torch or the ``smplx`` package is needed. Model files are licensed
separately (https://smpl-x.is.tue.mpg.de/) and never ship with this repo.
Default lookup order: ``SMPLX_MODEL`` env var, ``models/SMPLX_<GENDER>.npz``
next to the repo, ``~/Downloads/SMPLX_<GENDER>.npz``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .clip import PARENTS, NUM_BETAS, rotvec_to_matrix

NUM_VERTS = 10475
NUM_JOINTS = 55
REPO = Path(__file__).resolve().parent.parent


def find_model(gender: str = "male", explicit: Path | str | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("SMPLX_MODEL"):
        candidates.append(Path(os.environ["SMPLX_MODEL"]))
    name = f"SMPLX_{gender.upper()}.npz"
    candidates += [REPO / "models" / name, Path.home() / "Downloads" / name,
                   REPO / "models" / "smplx" / name]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"SMPL-X model not found; tried {[str(c) for c in candidates]}. "
                            "Set SMPLX_MODEL or put the NPZ in models/.")


class SMPLX:
    """Official SMPL-X geometry with a numpy LBS evaluator."""

    def __init__(self, path: Path | str | None = None, gender: str = "male", num_betas: int = NUM_BETAS):
        self.path = find_model(gender, path)
        self.gender = gender
        with np.load(self.path, allow_pickle=False) as m:
            self.v_template = np.asarray(m["v_template"], np.float32)
            self.faces = np.asarray(m["f"], np.int32)
            self.J_regressor = np.asarray(m["J_regressor"], np.float32)
            self.weights = np.asarray(m["weights"], np.float32)
            self.parents = np.asarray(m["kintree_table"][0], np.int64)
            self.shapedirs = np.asarray(m["shapedirs"][:, :, :num_betas], np.float32)
            self.posedirs = np.asarray(m["posedirs"], np.float32)  # [V,3,486]
            self.vert_sym = np.asarray(m["vert_sym_idxs"], np.int64) if "vert_sym_idxs" in m.files else None
        self.parents[0] = -1
        if self.v_template.shape != (NUM_VERTS, 3) or self.weights.shape != (NUM_VERTS, NUM_JOINTS):
            raise ValueError("Expected the official 10,475-vertex / 55-joint SMPL-X model")
        assert np.array_equal(self.parents, PARENTS)
        self.posedirs_flat = self.posedirs.reshape(NUM_VERTS * 3, -1)  # [V*3, 486]
        self._nz = [np.flatnonzero(self.weights[:, j] > 1e-7) for j in range(NUM_JOINTS)]

    # ---- shape ---------------------------------------------------------------------------
    def shaped(self, betas: np.ndarray | None = None) -> np.ndarray:
        """Rest vertices for betas (no pose)."""
        v = self.v_template.copy()
        if betas is not None:
            b = np.asarray(betas, np.float32)[: self.shapedirs.shape[2]]
            v += np.einsum("vci,i->vc", self.shapedirs[:, :, : len(b)], b)
        return v

    def joints_of(self, verts: np.ndarray) -> np.ndarray:
        return self.J_regressor @ verts

    # ---- pose ------------------------------------------------------------------------------
    def joint_transforms(self, pose: np.ndarray, rest_joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """pose [T,55,3] -> world joint transforms [T,55,4,4] and posed joints [T,55,3]."""
        pose = np.asarray(pose, np.float64)
        T = pose.shape[0]
        rot = rotvec_to_matrix(pose)  # [T,55,3,3]
        rel = rest_joints.astype(np.float64).copy()
        rel[1:] -= rest_joints[self.parents[1:]]
        G = np.zeros((T, NUM_JOINTS, 4, 4), np.float64)
        G[:, :, 3, 3] = 1.0
        G[:, 0, :3, :3] = rot[:, 0]
        G[:, 0, :3, 3] = rel[0]
        for j in range(1, NUM_JOINTS):
            local = np.zeros((T, 4, 4), np.float64)
            local[:, 3, 3] = 1.0
            local[:, :3, :3] = rot[:, j]
            local[:, :3, 3] = rel[j]
            G[:, j] = G[:, self.parents[j]] @ local
        posed = G[:, :, :3, 3].copy()
        return G, posed

    def pose_offsets(self, pose: np.ndarray) -> np.ndarray:
        """Pose blendshape offsets [T,V,3] from pose [T,55,3]."""
        rot = rotvec_to_matrix(np.asarray(pose, np.float64))[:, 1:]  # [T,54,3,3]
        feat = (rot - np.eye(3)).reshape(rot.shape[0], -1).astype(np.float32)  # [T,486]
        return (feat @ self.posedirs_flat.T).reshape(-1, NUM_VERTS, 3)

    def skin(self, rest_verts: np.ndarray, pose: np.ndarray, transl: np.ndarray | None = None,
             pose_correctives: bool = True, return_joints: bool = False):
        """LBS. rest_verts [V,3] (already shaped/sculpted), pose [T,55,3], transl [T,3]."""
        pose = np.asarray(pose, np.float64)
        if pose.ndim == 2:
            pose = pose[None]
        T = pose.shape[0]
        rest_joints = self.joints_of(rest_verts)
        G, posed_joints = self.joint_transforms(pose, rest_joints)
        # Remove rest joint position: A = G - pack(G @ [J;0])
        A = G.copy()
        A[:, :, :3, 3] -= np.einsum("tjab,jb->tja", G[:, :, :3, :3], rest_joints.astype(np.float64))
        v = np.broadcast_to(rest_verts.astype(np.float64), (T, NUM_VERTS, 3)).copy()
        if pose_correctives:
            v = v + self.pose_offsets(pose)
        out = np.zeros((T, NUM_VERTS, 3), np.float64)
        W = self.weights.astype(np.float64)
        for j in range(NUM_JOINTS):
            idx = self._nz[j]
            if len(idx) == 0:
                continue
            Rj = A[:, j, :3, :3]  # [T,3,3]
            tj = A[:, j, :3, 3]   # [T,3]
            xformed = np.einsum("tab,tvb->tva", Rj, v[:, idx]) + tj[:, None, :]
            out[:, idx] += W[idx, j][None, :, None] * xformed
        if transl is not None:
            tr = np.asarray(transl, np.float64).reshape(T, 1, 3)
            out = out + tr
            posed_joints = posed_joints + tr
        if return_joints:
            return out.astype(np.float32), posed_joints.astype(np.float32)
        return out.astype(np.float32)

    def joints(self, rest_verts: np.ndarray, pose: np.ndarray, transl: np.ndarray | None = None) -> np.ndarray:
        """Posed joints only [T,55,3] (cheap; no vertex skinning)."""
        pose = np.asarray(pose, np.float64)
        if pose.ndim == 2:
            pose = pose[None]
        _, posed = self.joint_transforms(pose, self.joints_of(rest_verts))
        if transl is not None:
            posed = posed + np.asarray(transl, np.float64).reshape(-1, 1, 3)
        return posed.astype(np.float32)

    def vertex_normals(self, verts: np.ndarray) -> np.ndarray:
        f = self.faces
        n = np.zeros_like(verts, dtype=np.float64)
        tri = verts[f].astype(np.float64)
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        for k in range(3):
            np.add.at(n, f[:, k], fn)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        return n.astype(np.float32)

    def export_json(self, rest_verts: np.ndarray, max_influences: int = 8) -> dict:
        """Compact browser payload: rest mesh, faces, joints, sparse weights. Model-derived; local only."""
        rest_joints = self.joints_of(rest_verts)
        order = np.argsort(-self.weights, axis=1)[:, :max_influences]
        w = np.take_along_axis(self.weights, order, axis=1)
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
        return {
            "vertices": np.round(rest_verts.astype(np.float64), 5).reshape(-1).tolist(),
            "faces": self.faces.reshape(-1).tolist(),
            "joints": np.round(rest_joints.astype(np.float64), 5).reshape(-1).tolist(),
            "parents": self.parents.tolist(),
            "skin_index": order.astype(np.int32).reshape(-1).tolist(),
            "skin_weight": np.round(w.astype(np.float64), 5).reshape(-1).tolist(),
            "influences": max_influences,
        }
