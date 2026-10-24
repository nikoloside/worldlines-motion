"""prompt -> fal video -> video_to_motion -> naturalize -> clip (+ optional physique tag)."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from ..clip import save_clip
from .fal_video import fetch, generate_performer, generate_video
from ..naturalize import NaturalizeConfig, naturalize
from ..physique import PRESETS
from ..smplx_model import REPO, SMPLX
from .video import video_to_motion

TAKES = Path(REPO / "takes")


def prompt_to_motion(prompt: str, out_dir: Path | str | None = None, video_model: str = "h3", duration: int = 5,
                     resolution: str = "480P", orientation: str = "portrait", performer: bool = True,
                     backend: str = "auto", natural: NaturalizeConfig | None = None, physique: str | dict | None = None,
                     model: SMPLX | None = None, public_base: str | None = None, log=print) -> dict:
    """Return {"clip": clip, "video": path, "npz": path, "timings": {...}}."""
    out_dir = Path(out_dir or TAKES)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{prompt}|{video_model}|{duration}|{resolution}|{orientation}|{performer}".encode()).hexdigest()[:10]
    stem = "".join(c if c.isalnum() else "_" for c in prompt.lower())[:40].strip("_") + "_" + key
    timings: dict[str, float] = {}
    t0 = time.time()
    image_url = generate_performer(orientation) if performer else None
    timings["performer"] = time.time() - t0
    t0 = time.time()
    gen = generate_video(prompt, video_model, duration, resolution, image_url, orientation)
    timings["video"] = time.time() - t0
    log(f"video {gen['model']} in {gen['elapsed']:.1f}s: {gen['video_url']}")
    video_path = fetch(gen["video_url"], out_dir / f"{stem}.mp4")
    t0 = time.time()
    model = model or SMPLX()
    source = gen["video_url"] if backend != "mediapipe" else video_path
    clip = video_to_motion(source if backend in ("auto", "gvhmr") else video_path, backend=backend, model=model, public_base=public_base)
    if clip["source"] == "mediapipe" and backend == "auto":
        clip["video"] = str(video_path)
    timings["motion"] = time.time() - t0
    log(f"motion via {clip['source']} in {timings['motion']:.1f}s ({len(clip['transl'])} frames)")
    clip["prompt"] = prompt
    clip["video"] = str(video_path)
    t0 = time.time()
    clip = naturalize(clip, natural or NaturalizeConfig(), model)
    timings["naturalize"] = time.time() - t0
    if physique:
        clip["physique"] = PRESETS[physique] if isinstance(physique, str) else physique
    npz = save_clip(clip, out_dir / f"{stem}.smplx.npz")
    return {"clip": clip, "video": video_path, "npz": npz, "timings": timings, "video_url": gen["video_url"]}
