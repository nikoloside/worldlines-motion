"""Rest-mesh physique sculpts on top of the ten SMPL-X shape betas.

Every sculpt keeps the official topology, joints and skinning weights, so the
result still animates with the standard SMPL-X pipeline. These regional
offsets are *not* official SMPL-X beta presets; a profile is stored with the
clip as ``physique`` so a scene can be rebuilt from parameters.

Profiles::

    {"betas": [...10 floats...], "muscular": 0..1, "bulge": 0..1,
     "bulge_size": 0.5..2 (scale of the region), "chest": 0..1, "waist": 0..1}
"""

from __future__ import annotations

import numpy as np

from .smplx_model import SMPLX

# A lean starting point in the first ten male shape components.
MUSCULAR_MALE_BETAS = np.array([-0.5, -1.8, 0.8, 0, 0, 0, -0.5, 0, 0, 0], dtype=np.float32)

DEFAULT_PROFILE = {"betas": [0.0] * 10, "muscular": 0.0, "bulge": 0.0, "bulge_size": 1.0, "chest": 0.0, "waist": 0.0}


def _gaussian(value: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((value - center) / sigma) ** 2)


def muscular_delta(body: np.ndarray, regressor: np.ndarray, weights: np.ndarray,
                   chest: float = 1.0, waist: float = 1.0, arms: float = 1.0) -> np.ndarray:
    """Chest/back widening, waist taper, deltoid + biceps growth. Rest-pose coords (Y up, +Z front)."""
    x, y, z = body.T
    delta = np.zeros_like(body)
    w = _gaussian(y, -0.235, 0.105) * np.clip((0.37 - np.abs(x)) / 0.16, 0, 1)
    delta[:, 0] -= 0.085 * waist * w * x
    delta[:, 2] -= 0.11 * waist * w * z
    c = _gaussian(y, -0.005, 0.105) * np.clip((0.26 - np.abs(x)) / 0.09, 0, 1)
    delta[:, 0] += 0.18 * chest * c * x
    joints = regressor @ body
    for shoulder_index, elbow_index in ((16, 18), (17, 19)):
        shoulder, elbow = joints[shoulder_index], joints[elbow_index]
        side = 1 if shoulder[0] > 0 else -1
        distance = side * (x - shoulder[0])
        arm_length = abs(elbow[0] - shoulder[0])
        t = np.clip(distance / arm_length, 0, 1)
        center_y = shoulder[1] * (1 - t) + elbow[1] * t
        center_z = shoulder[2] * (1 - t) + elbow[2] * t
        segment = _gaussian(t, 0.42, 0.27)
        skin = np.clip((weights[:, shoulder_index] + weights[:, elbow_index]) * 1.7, 0, 1)
        deltoid = _gaussian(t, 0.06, 0.18)
        gain = (0.48 * segment + 0.14 * deltoid) * skin * arms
        delta[:, 1] += gain * (y - center_y)
        delta[:, 2] += gain * (z - center_z)
    return delta


def bulge_region(body: np.ndarray, regressor: np.ndarray, weights: np.ndarray, size: float = 1.0) -> np.ndarray:
    """Per-vertex falloff (0..1) for the male groin bulge, located from the hip joints.

    Centre: midway between the hip joints, a little below them, on the front
    surface of the pelvis. The region is an anisotropic Gaussian: narrow across
    (x), taller than wide (y), restricted to front-facing skin (z ahead of the
    hip joints) and to vertices skinned mainly by pelvis/hips.
    """
    joints = regressor @ body
    pelvis, lhip, rhip = joints[0], joints[1], joints[2]
    centre = (lhip + rhip) / 2
    hip_width = abs(lhip[0] - rhip[0])
    cx, cy = centre[0], centre[1] - 0.22 * hip_width
    x, y, z = body.T
    front = z > pelvis[2] + 0.12 * hip_width
    sx, sy = 0.23 * hip_width * size, 0.36 * hip_width * size
    fall = _gaussian(x, cx, sx) * _gaussian(y, cy, sy)
    skinned = np.clip((weights[:, 0] + weights[:, 1] + weights[:, 2]) * 1.2, 0, 1)
    region = fall * skinned * front
    region[region < 0.02] = 0.0
    return region


def bulge_delta(body: np.ndarray, normals: np.ndarray, regressor: np.ndarray, weights: np.ndarray,
                amount: float = 1.0, size: float = 1.0) -> np.ndarray:
    """Push the groin region outward along a blend of the surface normal and +Z / -Y.

    ``amount`` 1.0 ≈ 2.8 cm peak offset at average male hip width.
    """
    region = bulge_region(body, regressor, weights, size)
    joints = regressor @ body
    hip_width = abs(joints[1][0] - joints[2][0])
    direction = 0.55 * normals + np.array([0.0, -0.35, 0.75], np.float32)
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
    peak = 0.165 * hip_width * amount
    return (peak * region)[:, None] * direction


def make_rest(model: SMPLX, profile: dict | None = None) -> tuple[np.ndarray, dict]:
    """Rest vertices for a physique profile. Returns (vertices[V,3], normalised profile)."""
    p = {**DEFAULT_PROFILE, **(profile or {})}
    betas = np.asarray(p.get("betas") or [0.0] * 10, np.float32)
    if betas.shape != (10,):
        raise ValueError("betas must have 10 values")
    muscular = float(np.clip(p.get("muscular", 0.0), 0.0, 1.0))
    bulge = float(np.clip(p.get("bulge", 0.0), 0.0, 1.5))
    bulge_size = float(np.clip(p.get("bulge_size", 1.0), 0.4, 2.5))
    chest = float(np.clip(p.get("chest", 0.0), 0.0, 1.5))
    waist = float(np.clip(p.get("waist", 0.0), 0.0, 1.5))
    body = model.shaped(betas)
    delta = np.zeros_like(body)
    if muscular > 0 or chest > 0 or waist > 0:
        delta += muscular_delta(body, model.J_regressor, model.weights,
                                chest=max(muscular, chest), waist=max(muscular, waist), arms=muscular)
    if bulge > 0:
        normals = model.vertex_normals(body)
        delta += bulge_delta(body, normals, model.J_regressor, model.weights, bulge, bulge_size)
    body = (body + delta).astype(np.float32)
    p.update(betas=betas.tolist(), muscular=muscular, bulge=bulge, bulge_size=bulge_size, chest=chest, waist=waist)
    return body, p


def make_muscular_rest(template, shapedirs, regressor, weights, strength: float = 1.0):
    """Backwards-compatible entry used by prepare_official_smplx.py."""
    if not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    betas = MUSCULAR_MALE_BETAS * np.float32(strength)
    body = np.asarray(template, np.float32).copy()
    body += np.einsum("vci,i->vc", shapedirs[:, :, :10], betas, optimize=True)
    if strength == 0:
        return body, betas
    body += np.float32(strength) * muscular_delta(body, regressor, weights)
    return body, betas


PRESETS = {
    "neutral": {**DEFAULT_PROFILE},
    "muscular_male": {**DEFAULT_PROFILE, "betas": MUSCULAR_MALE_BETAS.tolist(), "muscular": 1.0},
    "male_bulge": {**DEFAULT_PROFILE, "bulge": 1.0},
    "muscular_male_bulge": {**DEFAULT_PROFILE, "betas": MUSCULAR_MALE_BETAS.tolist(), "muscular": 1.0, "bulge": 1.0},
}
