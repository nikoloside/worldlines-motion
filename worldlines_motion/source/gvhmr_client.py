"""Client for a prompt2rig-protocol GVHMR worker (``POST /process {video_url}``).

The worker (``gvhmr_worker/worker.py``, ported from prompt2rig, Apache-2.0)
needs CUDA. On a Mac point ``GVHMR_WORKER_URL`` at a Linux/Windows box or a
cloud GPU running it; the JSON it returns is already SMPL body parameters, so
the conversion into a clip is a reshape plus an optional heading fix.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from ..clip import new_clip, rotvec_to_matrix, matrix_to_rotvec

DEFAULT_URL = os.environ.get("GVHMR_WORKER_URL", "http://127.0.0.1:5178")


def gvhmr_available(url: str | None = None, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{url or DEFAULT_URL}/health", timeout=timeout) as r:
            return bool(json.load(r).get("ok"))
    except Exception:  # noqa: BLE001
        return False


def gvhmr_to_clip(result: dict, face_camera: bool = True) -> dict:
    """Worker JSON -> clip. ``face_camera`` yaws the world so frame 0 faces +Z like the camera view."""
    n = int(result["n"])
    fps = float(result.get("fps") or 30.0)
    clip = new_clip(n, fps, source="gvhmr")
    clip["global_orient"] = np.asarray(result["global_orient"], np.float32).reshape(n, 3)
    clip["body_pose"] = np.asarray(result["body_pose"], np.float32).reshape(n, -1)[:, :63]
    clip["transl"] = np.asarray(result["transl"], np.float32).reshape(n, 3)
    if result.get("betas") is not None:
        b = np.asarray(result["betas"], np.float32).reshape(-1)[:10]
        clip["betas"][: len(b)] = b
    if face_camera and result.get("incam_global_orient0") is not None:
        R0 = rotvec_to_matrix(np.asarray(clip["global_orient"][0], np.float64))
        f_world = R0 @ np.array([0.0, 0.0, 1.0])
        a, b, c = result["incam_global_orient0"]
        Rc = rotvec_to_matrix(np.array([a, b, c], np.float64))
        f_cam = Rc @ np.array([0.0, 0.0, 1.0])  # OpenCV axes: x right, y down, z forward
        target_yaw = np.arctan2(f_cam[0], -f_cam[2])
        yaw = target_yaw - np.arctan2(f_world[0], f_world[2])
        cy, sy = np.cos(yaw), np.sin(yaw)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        clip["global_orient"] = matrix_to_rotvec(Ry[None] @ rotvec_to_matrix(clip["global_orient"].astype(np.float64))).astype(np.float32)
        t = clip["transl"].astype(np.float64)
        clip["transl"] = ((t - t[0]) @ Ry.T + t[0]).astype(np.float32)
    if result.get("hands"):
        clip["hands"] = json.dumps({k: v for k, v in result["hands"].items() if k in ("method",)})
    clip["notes"] = f"gvhmr worker, elapsed {result.get('elapsed', 0):.1f}s"
    return clip


def run_gvhmr(video: str | Path, url: str | None = None, hands: str | None = None, timeout: float = 600.0,
              public_base: str | None = None, **_) -> dict:
    """Send a video URL to the worker. Local paths need ``public_base`` (a URL the worker can fetch)."""
    url = url or DEFAULT_URL
    video = str(video)
    if not video.startswith(("http://", "https://")):
        if not public_base:
            raise ValueError("GVHMR worker needs a URL; serve the file (workbench does) or pass public_base")
        video = public_base.rstrip("/") + "/" + Path(video).name
    body = json.dumps({"video_url": video, **({"hands": hands} if hands else {})}).encode()
    req = urllib.request.Request(f"{url}/process", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            result = json.load(r)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GVHMR worker error {exc.code}: {exc.read().decode(errors='replace')[:400]}") from exc
    return gvhmr_to_clip(result)
