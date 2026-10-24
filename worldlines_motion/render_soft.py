"""Tiny dependency-free software renderer (PIL) for quick mesh review images.

Orthographic, painter-sorted flat shading. Good enough to eyeball physique
sculpts and contact; not a production renderer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _look(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return rx @ ry


def render_mesh(verts: np.ndarray, faces: np.ndarray, out: Path | str, size: int = 640,
                yaw_deg: float = 0.0, pitch_deg: float = 0.0, center: np.ndarray | None = None,
                extent: float | None = None, color=(205, 190, 175), highlight: np.ndarray | None = None,
                ground: float | None = None) -> Path:
    """Render `verts`[V,3] (Y up, +Z front) looking from +Z, rotated by yaw/pitch."""
    R = _look(np.radians(yaw_deg), np.radians(pitch_deg))
    v = (verts - (center if center is not None else verts.mean(0))) @ R.T
    if extent is None:
        extent = float(np.abs(v[:, :2]).max()) * 1.05
    scale = size / (2 * extent)
    xy = np.stack([v[:, 0] * scale + size / 2, size / 2 - v[:, 1] * scale], axis=1)
    tri = v[faces]
    depth = tri[:, :, 2].mean(1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    light = np.array([0.35, 0.6, 0.72])
    light /= np.linalg.norm(light)
    shade = np.clip(n @ light, 0, 1) * 0.75 + 0.25
    img = Image.new("RGB", (size, size), (28, 26, 32))
    draw = ImageDraw.Draw(img)
    if ground is not None:
        gy = size / 2 - ((ground - (center[1] if center is not None else verts.mean(0)[1])) * scale)
        draw.line([(0, gy), (size, gy)], fill=(90, 90, 110), width=1)
    order = np.argsort(depth)
    front = n[:, 2] > 0
    fh = highlight[faces].mean(1) if highlight is not None else None
    for i in order:
        if not front[i]:
            continue
        s = shade[i]
        c = np.array(color, float)
        if fh is not None and fh[i] > 0.02:
            c = c * (1 - fh[i]) + np.array([255, 70, 120]) * fh[i]
        col = tuple(int(x) for x in np.clip(c * s, 0, 255))
        draw.polygon([tuple(xy[k]) for k in faces[i]], fill=col)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def side_by_side(paths: list[Path | str], out: Path | str, labels: list[str] | None = None) -> Path:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    h = max(i.height for i in imgs)
    w = sum(i.width for i in imgs)
    canvas = Image.new("RGB", (w, h), (28, 26, 32))
    x = 0
    draw = ImageDraw.Draw(canvas)
    for k, im in enumerate(imgs):
        canvas.paste(im, (x, 0))
        if labels:
            draw.text((x + 8, 8), labels[k], fill=(240, 240, 240))
        x += im.width
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out
