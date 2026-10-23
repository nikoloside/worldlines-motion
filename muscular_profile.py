"""Optional local physique sculpt on a licensed SMPL-X male rest mesh.

This keeps the SMPL-X topology and 55-joint skinning weights. The sculpt is
outside the standard SMPL-X beta space, so it must not be described as an
official SMPL-X beta preset.
"""

from __future__ import annotations

import numpy as np


# A lean starting point in the first ten male shape components. The visible
# shoulder/arm/torso changes below are the custom part of this profile.
MUSCULAR_MALE_BETAS = np.array([-0.5, -1.8, 0.8, 0, 0, 0, -0.5, 0, 0, 0], dtype=np.float32)


def _gaussian(value: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((value - center) / sigma) ** 2)


def make_muscular_rest(
    template: np.ndarray,
    shapedirs: np.ndarray,
    regressor: np.ndarray,
    weights: np.ndarray,
    strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return custom muscular mesh and underlying ten SMPL-X shape betas.

    ``strength=0`` returns the unmodified template. For 0..1, beta shape and
    custom regional sculpt are blended together. The source model remains
    caller owned; this function never changes it in place.
    """
    if not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    if template.shape != (10475, 3) or shapedirs.shape[:2] != template.shape:
        raise ValueError("Expected the official SMPL-X 10,475-vertex model")
    betas = MUSCULAR_MALE_BETAS * np.float32(strength)
    body = np.asarray(template, dtype=np.float32).copy()
    body += np.einsum("vci,i->vc", shapedirs[:, :, :10], betas, optimize=True)
    if strength == 0:
        return body, betas

    # Coordinates are the source SMPL-X rest-pose system: Y up, X across the
    # T-pose arms, positive Z toward the face/chest.
    x, y, z = body.T
    delta = np.zeros_like(body)
    waist = _gaussian(y, -0.235, 0.105) * np.clip((0.37 - np.abs(x)) / 0.16, 0, 1)
    delta[:, 0] -= 0.085 * waist * x
    delta[:, 2] -= 0.11 * waist * z

    chest = _gaussian(y, -0.005, 0.105) * np.clip((0.26 - np.abs(x)) / 0.09, 0, 1)
    delta[:, 0] += 0.18 * chest * x

    joints = regressor @ body
    for shoulder_index, elbow_index in ((16, 18), (17, 19)):
        shoulder = joints[shoulder_index]
        elbow = joints[elbow_index]
        side = 1 if shoulder[0] > 0 else -1
        # Grow the deltoid and biceps around the existing arm centerline.
        distance = side * (x - shoulder[0])
        arm_length = abs(elbow[0] - shoulder[0])
        t = np.clip(distance / arm_length, 0, 1)
        center_y = shoulder[1] * (1 - t) + elbow[1] * t
        center_z = shoulder[2] * (1 - t) + elbow[2] * t
        segment = _gaussian(t, 0.42, 0.27)
        skin = np.clip((weights[:, shoulder_index] + weights[:, elbow_index]) * 1.7, 0, 1)
        deltoid = _gaussian(t, 0.06, 0.18)
        radius_gain = (0.48 * segment + 0.14 * deltoid) * skin
        delta[:, 1] += radius_gain * (y - center_y)
        delta[:, 2] += radius_gain * (z - center_z)

    body += np.float32(strength) * delta
    return body, betas
