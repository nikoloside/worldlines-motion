"""MediaPipe Pose -> SMPL-X clip. Runs on CPU / Apple silicon; no CUDA.

Quality is below GVHMR (single-camera depth is noisy, root depth is not
observed), so the result goes through ``naturalize`` by default. The model
file ``pose_landmarker_heavy.task`` is downloaded once into ``models/``.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

from ..clip import new_clip
from ..smplx_model import SMPLX, REPO
from .landmarks_to_smplx import LandmarkSolver, MP

MODEL_URLS = {
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def pose_model_path(variant: str = "full") -> Path:
    d = Path(os.environ.get("WLMOTION_MODELS", REPO / "models"))
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"pose_landmarker_{variant}.task"
    if not p.exists():
        urllib.request.urlretrieve(MODEL_URLS[variant], p)
    return p


def track_video(video: str | Path, variant: str = "full", max_frames: int | None = None, progress=None):
    """Return (world[T,33,3] in Y-up metres, image[T,33,2] normalised, vis[T,33], fps, (w,h), valid[T])."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(pose_model_path(variant))),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    world, image, vis, valid = [], [], [], []
    with vision.PoseLandmarker.create_from_options(opts) as det:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and i >= max_frames):
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / fps))
            if res.pose_world_landmarks:
                wl, il = res.pose_world_landmarks[0], res.pose_landmarks[0]
                world.append([(p.x, -p.y, -p.z) for p in wl])
                image.append([(p.x, p.y) for p in il])
                vis.append([(p.visibility if p.visibility is not None else 1.0) for p in wl])
                valid.append(True)
            else:
                world.append(world[-1] if world else [(0.0, 0.0, 0.0)] * 33)
                image.append(image[-1] if image else [(0.5, 0.5)] * 33)
                vis.append([0.0] * 33)
                valid.append(False)
            i += 1
            if progress and i % 15 == 0:
                progress(i, total)
    cap.release()
    return (np.asarray(world, np.float64), np.asarray(image, np.float64), np.asarray(vis, np.float64),
            float(fps), (w, h), np.asarray(valid, bool))


def run_mediapipe(video: str | Path, model: SMPLX | None = None, variant: str = "full", root_motion: bool = True,
                  max_frames: int | None = None, progress=None, **_) -> dict:
    model = model or SMPLX()
    world, image, vis, fps, (w, h), valid = track_video(video, variant, max_frames, progress)
    n = len(world)
    if n == 0 or not valid.any():
        raise RuntimeError("no person detected in the video")
    solver = LandmarkSolver(model)
    clip = new_clip(n, fps, source="mediapipe")
    clip["video"] = str(video)
    # Root: 2-D hip track in pixels -> metres via the torso length ratio (world metres / pixels).
    rest_j = solver.rest
    leg_rest = (np.linalg.norm(rest_j[1] - rest_j[4]) + np.linalg.norm(rest_j[4] - rest_j[7]))
    hip0 = None
    mpp = None
    for t in range(n):
        go, bp = solver.solve(world[t], vis[t])
        clip["global_orient"][t] = go
        clip["body_pose"][t] = bp
        if root_motion and valid[t]:
            hip_px = (image[t, MP["l_hip"]] + image[t, MP["r_hip"]]) / 2 * (w, h)
            sh_px = (image[t, MP["l_shoulder"]] + image[t, MP["r_shoulder"]]) / 2 * (w, h)
            mid_hip = (world[t, MP["l_hip"]] + world[t, MP["r_hip"]]) / 2
            mid_sh = (world[t, MP["l_shoulder"]] + world[t, MP["r_shoulder"]]) / 2
            ratio = np.linalg.norm(mid_sh - mid_hip) / max(1.0, np.linalg.norm(sh_px - hip_px))
            leg = (np.linalg.norm(world[t, 23] - world[t, 25]) + np.linalg.norm(world[t, 25] - world[t, 27])
                   + np.linalg.norm(world[t, 24] - world[t, 26]) + np.linalg.norm(world[t, 26] - world[t, 28])) / 2
            scale = leg_rest / max(leg, 1e-3)
            mpp = ratio if mpp is None else mpp + 0.05 * (ratio - mpp)
            if hip0 is None:
                hip0 = hip_px.copy()
            clip["transl"][t] = [(hip_px[0] - hip0[0]) * mpp * scale, -(hip_px[1] - hip0[1]) * mpp * scale, 0.0]
        elif t > 0:
            clip["transl"][t] = clip["transl"][t - 1]
    # fill frames with no detection by holding the previous pose (first valid frame backwards)
    first = int(np.argmax(valid))
    for t in range(first):
        clip["global_orient"][t] = clip["global_orient"][first]
        clip["body_pose"][t] = clip["body_pose"][first]
    for t in range(first + 1, n):
        if not valid[t]:
            clip["global_orient"][t] = clip["global_orient"][t - 1]
            clip["body_pose"][t] = clip["body_pose"][t - 1]
    clip["notes"] = f"mediapipe {variant}, {int(valid.sum())}/{n} frames tracked, {w}x{h}"
    if model is not None:
        # The 2D-derived root has no absolute height; drop the body onto y=0 so the raw clip is
        # already viewable. Jitter, skate and limits are still left for the naturalize stage.
        from ..naturalize import estimate_floor
        clip["transl"][:, 1] -= estimate_floor(clip, model)
    return clip
