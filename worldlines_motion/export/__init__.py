"""Stage 4 · 导出 (export): turn a clip into something another tool can open.

  blender_addon.py   .npz for the official "SMPL-X for Blender" Add Animation dialog
  blender_import.py  run inside Blender to drive that dialog headlessly
  blender_rig.py     full-skin local rig data (mesh, weights, per-frame correctives)
  mesh.py            bake per-frame vertices + hair attachment (no Blender needed)
"""

from .blender_addon import addon_arrays, export_clip  # noqa: F401
from .mesh import bake_mesh  # noqa: F401
