"""Stage 1 · 来源 (source): get a motion clip from somewhere.

  prompt.py             prompt -> fal video -> video_to_motion -> naturalize
  video.py              video -> clip, picking the best available backend
  gvhmr_client.py       client for the GVHMR HQ worker (CUDA, local or remote)
  gvhmr_worker/         the worker itself (from prompt2rig, Apache-2.0)
  mediapipe_solver.py   CPU fallback: MediaPipe Pose landmarks -> SMPL-X
  landmarks_to_smplx.py per-bone aiming solver used by the MediaPipe path
  fal_video.py          fal.ai text/image-to-video helpers
  procedural.py         deterministic test clips (no model, no network)
"""

from .gvhmr_client import gvhmr_available, gvhmr_to_clip, run_gvhmr  # noqa: F401
from .video import video_to_motion  # noqa: F401


def prompt_to_motion(*args, **kwargs):
    """Lazy wrapper so importing the package does not require fal credentials."""
    from .prompt import prompt_to_motion as _p2m
    return _p2m(*args, **kwargs)
