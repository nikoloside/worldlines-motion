"""Stdlib HTTP server for the motion workbench (no Node, no FastAPI).

Static files from ``public/``; JSON API under ``/api``; uploaded or fetched
videos under ``/uploads`` so a remote GVHMR worker can pull them. Long jobs
(video->motion, prompt->motion) run in a thread and are polled by id.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import threading
import time
import traceback
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..clip import from_json, load_clip, save_clip, to_json, validate
from ..naturalize import NaturalizeConfig, metrics, naturalize
from ..physique import PRESETS, make_rest
from ..smplx_model import REPO, SMPLX, find_model
from ..source import gvhmr_available, video_to_motion

PUBLIC = Path(__file__).resolve().parent / "public"
UPLOADS = REPO / "takes" / "uploads"
TAKES = REPO / "takes"
CLIP_DIRS = [TAKES, REPO / "demos" / "clips", REPO / "demos" / "local-smplx"]

_state: dict = {"model": None, "model_cache": {}, "jobs": {}, "lock": threading.Lock()}


def get_model() -> SMPLX:
    with _state["lock"]:
        if _state["model"] is None:
            _state["model"] = SMPLX(gender=os.environ.get("SMPLX_GENDER", "male"))
        return _state["model"]


def model_payload(profile: dict | None, only_vertices: bool = False) -> dict:
    model = get_model()
    key = json.dumps(profile or {}, sort_keys=True)
    with _state["lock"]:
        cached = _state["model_cache"].get(key)
    if cached is None:
        rest, norm = make_rest(model, profile)
        cached = {"profile": norm, "rest": rest}
        with _state["lock"]:
            _state["model_cache"][key] = cached
            if len(_state["model_cache"]) > 12:
                _state["model_cache"].pop(next(iter(_state["model_cache"])))
    if only_vertices:
        return {"profile": cached["profile"], "vertices": np.round(cached["rest"].astype(np.float64), 5).reshape(-1).tolist(),
                "joints": np.round(model.joints_of(cached["rest"]).astype(np.float64), 5).reshape(-1).tolist()}
    out = model.export_json(cached["rest"])
    out["profile"] = cached["profile"]
    out["gender"] = model.gender
    out["sole_offset"] = float(cached["rest"][:, 1].min())
    return out


def list_clips() -> list[dict]:
    seen, out = set(), []
    for d in CLIP_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.npz")):
            if ".local." in p.name or "fixture" in p.name or "reference" in p.name or "pose_offsets" in p.name or "synthetic_lbs" in p.name:
                continue
            if p.name.endswith(".blender_smplx.npz"):
                continue
            rp = str(p.relative_to(REPO))
            if rp in seen:
                continue
            seen.add(rp)
            try:
                with np.load(p, allow_pickle=False) as z:
                    if "schema" not in z.files:
                        continue
                    n = int(z["transl"].shape[0])
                    fps = float(z["fps"])
                    src = str(z["source"]) if "source" in z.files else ""
            except Exception:  # noqa: BLE001
                continue
            out.append({"path": rp, "name": p.name, "frames": n, "fps": fps, "source": src, "mtime": p.stat().st_mtime})
    out.sort(key=lambda c: -c["mtime"])
    return out


def safe_clip_path(rel: str) -> Path:
    p = (REPO / rel).resolve()
    if REPO.resolve() not in p.parents or p.suffix != ".npz":
        raise ValueError("bad clip path")
    return p


def clip_json_with_metrics(clip: dict) -> dict:
    data = to_json(clip)
    try:
        data["metrics"] = metrics(clip, get_model())
    except Exception as exc:  # noqa: BLE001
        data["metrics"] = {"error": str(exc)}
    return data


# ---- jobs ----------------------------------------------------------------------------------

def start_job(kind: str, fn) -> str:
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "kind": kind, "status": "running", "progress": "", "started": time.time(), "result": None, "error": None}
    _state["jobs"][jid] = job

    def run():
        try:
            job["result"] = fn(job)
            job["status"] = "done"
        except Exception as exc:  # noqa: BLE001
            job["error"] = f"{type(exc).__name__}: {exc}"
            job["trace"] = traceback.format_exc()
            job["status"] = "error"
        job["elapsed"] = time.time() - job["started"]

    threading.Thread(target=run, daemon=True).start()
    return jid


def job_video_to_motion(job: dict, video: Path, backend: str, natural: dict | None, base_url: str) -> dict:
    model = get_model()

    def progress(i, n):
        job["progress"] = f"tracking {i}/{n}"

    source = base_url.rstrip("/") + "/uploads/" + video.name if backend in ("auto", "gvhmr") and gvhmr_available() else video
    clip = video_to_motion(source, backend=backend, model=model, progress=progress, public_base=base_url + "/uploads")
    clip["video"] = "/uploads/" + video.name
    raw = clip_json_with_metrics(clip)
    job["progress"] = "naturalizing"
    nat = naturalize(clip, NaturalizeConfig.from_json(natural), model)
    stem = video.stem
    save_clip(clip, TAKES / f"{stem}.raw.smplx.npz")
    path = save_clip(nat, TAKES / f"{stem}.smplx.npz")
    return {"raw": raw, "clip": clip_json_with_metrics(nat), "path": str(path.relative_to(REPO))}


def job_prompt_to_motion(job: dict, body: dict, base_url: str) -> dict:
    from ..source.prompt import prompt_to_motion

    def log(msg):
        job["progress"] = msg

    res = prompt_to_motion(body["prompt"], out_dir=UPLOADS, video_model=body.get("video_model", "h3"),
                           duration=int(body.get("duration", 5)), resolution=body.get("resolution", "480P"),
                           orientation=body.get("orientation", "portrait"), performer=bool(body.get("performer", True)),
                           backend=body.get("backend", "auto"), natural=NaturalizeConfig.from_json(body.get("natural")),
                           model=get_model(), public_base=base_url + "/uploads", log=log)
    clip = res["clip"]
    clip["video"] = "/uploads/" + Path(res["video"]).name
    path = save_clip(clip, TAKES / Path(res["npz"]).name)
    return {"clip": clip_json_with_metrics(clip), "path": str(path.relative_to(REPO)), "video_url": res["video_url"], "timings": res["timings"]}


# ---- handler --------------------------------------------------------------------------------

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(PUBLIC), **kw)

    def end_headers(self):
        if getattr(self, "_static", False):  # always revalidate the UI files while developing
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):  # quieter
        if "/api/job/" not in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def base_url(self) -> str:
        host = self.headers.get("Host") or f"127.0.0.1:{self.server.server_address[1]}"
        return f"http://{host}"

    def send_json(self, status: int, body) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/health":
                try:
                    model_path = str(find_model(os.environ.get("SMPLX_GENDER", "male")))
                except FileNotFoundError as exc:
                    model_path = None
                    err = str(exc)
                else:
                    err = None
                try:
                    import mediapipe  # noqa: F401
                    mp_ok = True
                except ImportError:
                    mp_ok = False
                from ..source.fal_video import load_env
                load_env()
                return self.send_json(200, {"smplx": model_path, "smplx_error": err, "gvhmr": gvhmr_available(),
                                            "gvhmr_url": os.environ.get("GVHMR_WORKER_URL", "http://127.0.0.1:5178"),
                                            "mediapipe": mp_ok, "fal": bool(os.environ.get("FAL_KEY")), "repo": str(REPO)})
            if u.path == "/api/model":
                profile = json.loads(q["physique"]) if q.get("physique") else None
                return self.send_json(200, model_payload(profile, q.get("only") == "vertices"))
            if u.path == "/api/presets":
                return self.send_json(200, PRESETS)
            if u.path == "/api/clips":
                return self.send_json(200, list_clips())
            if u.path == "/api/clip":
                clip = load_clip(safe_clip_path(q["path"]))
                return self.send_json(200, clip_json_with_metrics(clip))
            if u.path.startswith("/api/job/"):
                job = _state["jobs"].get(u.path.rsplit("/", 1)[1])
                if not job:
                    return self.send_json(404, {"error": "no such job"})
                return self.send_json(200, {k: v for k, v in job.items() if k != "trace" or job["status"] == "error"})
            if u.path == "/api/export/blender":
                from ..export.blender_addon import addon_arrays
                clip = load_clip(safe_clip_path(q["path"]))
                arrays = addon_arrays(clip, q.get("gender", "male"))
                import io
                buf = io.BytesIO()
                np.savez_compressed(buf, **arrays)
                data = buf.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", f'attachment; filename="{Path(q["path"]).stem}.blender_smplx.npz"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return self.wfile.write(data)
            if u.path.startswith("/uploads/"):
                name = Path(u.path).name
                p = UPLOADS / name
                if not p.is_file():
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(p.stat().st_size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with open(p, "rb") as f:
                    self.wfile.write(f.read())
                return
            if u.path.startswith("/api/"):
                return self.send_json(404, {"error": "unknown endpoint"})
            if u.path == "/":
                self.path = "/index.html"
            self._static = True
            return super().do_GET()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return self.send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):  # noqa: N802
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/naturalize":
                body = json.loads(self.read_body() or b"{}")
                clip = from_json(body["clip"])
                out = naturalize(clip, NaturalizeConfig.from_json(body.get("config")), get_model())
                return self.send_json(200, clip_json_with_metrics(out))
            if u.path == "/api/metrics":
                body = json.loads(self.read_body() or b"{}")
                return self.send_json(200, metrics(from_json(body["clip"]), get_model()))
            if u.path == "/api/save":
                body = json.loads(self.read_body() or b"{}")
                clip = from_json(body["clip"])
                name = re.sub(r"[^a-zA-Z0-9_\-一-鿿]+", "_", body.get("name") or "take")[:60]
                if body.get("physique"):
                    clip["physique"] = body["physique"]
                path = save_clip(clip, TAKES / f"{name}.smplx.npz")
                return self.send_json(200, {"path": str(path.relative_to(REPO))})
            if u.path == "/api/upload":
                name = re.sub(r"[^a-zA-Z0-9_.\-]+", "_", q.get("name") or "video.mp4")[:80]
                stem, ext = os.path.splitext(name)
                UPLOADS.mkdir(parents=True, exist_ok=True)
                dest = UPLOADS / f"{stem}_{uuid.uuid4().hex[:6]}{ext or '.mp4'}"
                dest.write_bytes(self.read_body())
                return self.send_json(200, {"name": dest.name, "url": "/uploads/" + dest.name})
            if u.path == "/api/video_to_motion":
                body = json.loads(self.read_body() or b"{}")
                video = UPLOADS / Path(body["name"]).name
                if not video.is_file():
                    return self.send_json(404, {"error": "upload first"})
                base = self.base_url()
                jid = start_job("video_to_motion", lambda job: job_video_to_motion(job, video, body.get("backend", "auto"), body.get("natural"), base))
                return self.send_json(202, {"job": jid})
            if u.path == "/api/prompt_to_motion":
                body = json.loads(self.read_body() or b"{}")
                if not body.get("prompt"):
                    return self.send_json(400, {"error": "prompt required"})
                base = self.base_url()
                jid = start_job("prompt_to_motion", lambda job: job_prompt_to_motion(job, body, base))
                return self.send_json(202, {"job": jid})
            return self.send_json(404, {"error": "unknown endpoint"})
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return self.send_json(500, {"error": f"{type(exc).__name__}: {exc}"})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("WLMOTION_PORT", 5190)))
    args = parser.parse_args(argv)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"worldlines motion workbench -> http://{args.host}:{args.port}  (repo {REPO})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
