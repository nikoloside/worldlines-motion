"""Prompt -> short performer video on fal (same models/prompts as prompt2rig).

Needs ``FAL_KEY`` in the environment or in ``.env`` next to the repo. Nothing
is cached here; the returned URL is fal's hosted MP4, download it with
``fetch``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

from ..smplx_model import REPO

VIDEO_MODELS = {
    "h3": {"t2v": "minimax/h3-max-turbo/text-to-video", "i2v": "minimax/h3-max-turbo/image-to-video"},
    "flux3": {"t2v": "blackforestlabs/flux-3-max/text-to-video", "i2v": "blackforestlabs/flux-3-max/image-to-video"},
}
IMAGE_MODEL = "fal-ai/flux/schnell"

PERFORMER_PROMPT = (
    "Full-body photograph of one young adult standing in a relaxed neutral pose facing the camera, arms relaxed at the sides, "
    "wearing a fitted plain light-grey t-shirt, black slim pants and white sneakers. The whole body is visible from head to feet "
    "with empty space above the head and below the feet, the person is centered. Plain seamless "
    "light-grey studio backdrop and floor, flat even soft lighting, sharp focus, 35mm photo.")
STAGING = ("They stay roughly in place facing the camera, with both arms clearly visible and held away from the torso, no turning around, no flips. "
           "Their entire body, including both feet, stays visible in frame the whole time. Static locked-off camera, no camera movement, no zoom, no cuts.")


def video_prompt(action: str, from_image: bool) -> str:
    if from_image:
        return f"The person {action}. {STAGING} Plain studio background."
    return (f"Wide full-body shot of one person wearing a plain grey t-shirt and dark pants in an empty light-grey studio. "
            f"The person {action}. {STAGING}")


def load_env() -> None:
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def fal_key() -> str:
    load_env()
    key = os.environ.get("FAL_KEY")
    if not key:
        raise RuntimeError("FAL_KEY is not set (put it in .env or the environment)")
    return key


def fal_run(model: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(f"https://fal.run/{model}", data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Key {fal_key()}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.load(r)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"fal {model} -> {exc.code}: {exc.read().decode(errors='replace')[:400]}") from exc
    out["_elapsed"] = time.time() - t0
    return out


def generate_performer(orientation: str = "portrait") -> str:
    out = fal_run(IMAGE_MODEL, {"prompt": PERFORMER_PROMPT, "num_inference_steps": 4, "enable_safety_checker": True,
                                "image_size": "portrait_16_9" if orientation == "portrait" else "landscape_16_9"})
    return out["images"][0]["url"]


def generate_video(action: str, model: str = "h3", duration: int = 5, resolution: str = "480P", image_url: str | None = None,
                   orientation: str = "portrait", seed: int | None = None) -> dict:
    """Return {video_url, model, elapsed}."""
    if model not in VIDEO_MODELS:
        raise ValueError(f"model must be one of {list(VIDEO_MODELS)}")
    mode = "i2v" if image_url else "t2v"
    aspect = "9:16" if orientation == "portrait" else "16:9"
    payload: dict = {"prompt": video_prompt(action, bool(image_url))[:4000]}
    if model == "h3":
        payload.update(duration=min(15, max(1, int(duration))), resolution=resolution if resolution in ("480P", "768P", "1080P") else "480P",
                       prompt_expansion_mode="disabled")
        if mode == "t2v":
            payload["aspect_ratio"] = aspect
    else:
        payload.update(duration=min(20, max(5, int(duration))), resolution="1080p" if resolution == "1080P" else "720p",
                       generate_audio=False, reasoning_level="low", aspect_ratio="auto" if image_url else aspect)
    if seed is not None:
        payload["seed"] = int(seed)
    if image_url:
        payload["image_url"] = image_url
    out = fal_run(VIDEO_MODELS[model][mode], payload)
    return {"video_url": out.get("video", {}).get("url"), "model": model, "mode": mode, "elapsed": out["_elapsed"], "timings": out.get("timings")}


def fetch(url: str, dest: Path | str) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest
