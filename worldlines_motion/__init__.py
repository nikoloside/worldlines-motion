"""Worldlines Motion: prompt / video -> SMPL-X motion toolkit.

The package follows the production pipeline, one stage per module group:

  1. source/     来源   prompt -> fal video -> motion; video -> motion (GVHMR worker
                        or MediaPipe on CPU); procedural test clips
  2. naturalize  动作   clean-up passes: smoothing, joint limits, ground, foot lock
  3. physique    身体   rest-mesh sculpts (muscular, male bulge) on top of betas
  4. export/     导出   Blender add-on .npz, full-skin rig data, baked meshes

Shared foundations:
  clip         motion clip schema (worldlines.smplx_motion.v1), load/save/resample
  smplx_model  numpy SMPL-X loader + linear blend skinning (no torch needed)
  render_soft  dependency-free software renderer for review frames
  workbench/   local HTTP server + browser viewer that walks the four stages
  cli          ``wlmotion`` command line, one subcommand per stage
"""

__version__ = "0.3.0"
