"""Video -> SMPL-X motion.

Backends
  gvhmr      prompt2rig-style HQ worker (CUDA, remote or local): best quality
  mediapipe  CPU/Metal fallback that runs on a Mac: MediaPipe Pose world
             landmarks solved onto the SMPL-X skeleton (no depth-true root)
"""

from .gvhmr_client import gvhmr_available, gvhmr_to_clip, run_gvhmr  # noqa: F401
from .landmarks_to_smplx import LandmarkSolver  # noqa: F401


def video_to_motion(video, backend: str = "auto", **kwargs):
    """Return a clip dict from a video path or URL using the best available backend."""
    if backend in ("auto", "gvhmr") and gvhmr_available():
        return run_gvhmr(video, **kwargs)
    if backend == "gvhmr":
        raise RuntimeError("GVHMR worker is not reachable; set GVHMR_WORKER_URL or start the worker")
    from .mediapipe_solver import run_mediapipe
    return run_mediapipe(video, **kwargs)
