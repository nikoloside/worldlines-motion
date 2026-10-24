"""``wlmotion``: one subcommand per pipeline stage.

  workbench                                   browser UI that walks the four stages

  1 来源 source
    prompt "does jumping jacks" -o takes/     prompt -> fal video -> motion -> naturalize
    video clip.mp4 -o out.smplx.npz           video -> motion (GVHMR worker, else MediaPipe)
    procedural walk_squat out.smplx.npz       deterministic test clip, no model needed
  2 动作 motion
    naturalize in.npz out.npz                 clean-up passes, metrics before/after
  3 身体 body
    physique in.npz out.npz --preset male_bulge   tag a clip with a physique profile
  4 导出 export
    blender clip.npz out.npz                  official SMPL-X Blender add-on animation
    mesh clip.npz out_dir                     baked vertices + hair attachment
    rig clip.npz out_dir --physique ...       full-skin local rig data for Blender
    render clip.npz out.png --frame 60        software render of one frame for review

Global: --model /path/to/SMPLX_MALE.npz (default $SMPLX_MODEL, models/, ~/Downloads), --gender.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .clip import full_pose, load_clip, save_clip
from .naturalize import NaturalizeConfig, metrics, naturalize
from .physique import PRESETS, make_rest
from .smplx_model import SMPLX, find_model


def _model(args) -> SMPLX:
    return SMPLX(getattr(args, "model", None), getattr(args, "gender", "male"))


def _print_metrics(label: str, m: dict) -> None:
    print(f"{label:>8}: " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))


def _progress(i, n):
    print(f"  {i}/{n}", end="\r")


# ---- workbench -------------------------------------------------------------------------------

def cmd_workbench(args):
    from .workbench.server import main as serve
    serve(["--port", str(args.port), "--host", args.host])


# ---- 1 source ----------------------------------------------------------------------------------

def cmd_prompt(args):
    from .source.prompt import prompt_to_motion
    res = prompt_to_motion(args.prompt, args.out, video_model=args.video_model, duration=args.duration,
                           resolution=args.resolution, orientation=args.orientation, performer=not args.t2v,
                           backend=args.backend, physique=args.preset, model=_model(args))
    print(f"video: {res['video']}\nclip:  {res['npz']}\ntimings: {json.dumps({k: round(v, 1) for k, v in res['timings'].items()})}")


def cmd_video(args):
    from .source import video_to_motion
    model = _model(args)
    clip = video_to_motion(args.video, backend=args.backend, model=model, variant=args.variant, progress=_progress)
    _print_metrics("raw", metrics(clip, model))
    if not args.raw:
        clip = naturalize(clip, NaturalizeConfig(), model)
        _print_metrics("natural", metrics(clip, model))
    if args.preset:
        clip["physique"] = PRESETS[args.preset]
    save_clip(clip, args.out)
    print(f"wrote {args.out} ({len(clip['transl'])} frames @ {clip['fps']} fps, source {clip['source']})")


def cmd_procedural(args):
    from .source.procedural import make_clip
    save_clip(make_clip(args.case), args.output)
    print(f"wrote {args.output}")


# ---- 2 motion ----------------------------------------------------------------------------------

def cmd_naturalize(args):
    model = _model(args)
    clip = load_clip(args.input)
    cfg = NaturalizeConfig.from_json(json.loads(args.config) if args.config else None)
    if args.breathing is not None:
        cfg.breathing = args.breathing
    _print_metrics("before", metrics(clip, model))
    out = naturalize(clip, cfg, model)
    _print_metrics("after", metrics(out, model))
    save_clip(out, args.output)
    print(f"wrote {args.output}")


# ---- 3 body ------------------------------------------------------------------------------------

def cmd_physique(args):
    clip = load_clip(args.input)
    profile = dict(PRESETS[args.preset]) if args.preset else dict(clip.get("physique") or PRESETS["neutral"])
    for key in ("muscular", "bulge", "bulge_size", "chest", "waist"):
        v = getattr(args, key)
        if v is not None:
            profile[key] = v
    if args.betas:
        profile["betas"] = [float(x) for x in args.betas.split(",")]
    _, profile = make_rest(_model(args), profile)
    clip["physique"] = profile
    clip["betas"] = np.asarray(profile["betas"], np.float32)
    save_clip(clip, args.output)
    print(f"wrote {args.output} with physique {json.dumps(profile)}")


# ---- 4 export ----------------------------------------------------------------------------------

def cmd_blender(args):
    from .export.blender_addon import export_clip
    export_clip(Path(args.input), Path(args.output), args.gender)
    print(f"wrote {args.output} (import with the SMPL-X Blender add-on, AMASS orientation)")


def cmd_mesh(args):
    from .export.mesh import bake_mesh
    out = bake_mesh(load_clip(args.input), _model(args), args.output, step=args.step,
                    pose_correctives=not args.no_correctives, progress=_progress)
    print(f"wrote {out}/smplx_animation.npz and hair_attachment.json")


def cmd_rig(args):
    from .export.blender_rig import prepare
    model_path = Path(args.model) if args.model else find_model(args.gender)
    prepare(model_path, Path(args.input), Path(args.output), args.physique, args.strength,
            json.loads(args.profile) if args.profile else None)


def cmd_render(args):
    from .render_soft import render_mesh
    model = _model(args)
    clip = load_clip(args.input)
    rest, _ = make_rest(model, clip.get("physique"))
    f = min(args.frame, len(clip["transl"]) - 1)
    v = model.skin(rest, full_pose(clip)[f][None], clip["transl"][f][None])[0]
    render_mesh(v, model.faces, args.output, args.size, args.yaw, args.pitch, ground=0.0)
    print(f"wrote {args.output} (frame {f})")


# ---- parser ------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wlmotion", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="SMPL-X NPZ (default: $SMPLX_MODEL, models/, ~/Downloads)")
    p.add_argument("--gender", default="male", choices=("male", "female", "neutral"))
    sub = p.add_subparsers(dest="cmd", required=True, metavar="command")
    backends = ("auto", "gvhmr", "mediapipe")

    s = sub.add_parser("workbench", help="browser workbench (http://127.0.0.1:5190)")
    s.add_argument("--port", type=int, default=5190); s.add_argument("--host", default="127.0.0.1"); s.set_defaults(fn=cmd_workbench)

    s = sub.add_parser("prompt", aliases=["p2m"], help="[1 source] prompt -> video -> motion")
    s.add_argument("prompt"); s.add_argument("-o", "--out", default="takes")
    s.add_argument("--video-model", default="h3", choices=("h3", "flux3")); s.add_argument("--duration", type=int, default=5)
    s.add_argument("--resolution", default="480P"); s.add_argument("--orientation", default="portrait", choices=("portrait", "landscape"))
    s.add_argument("--t2v", action="store_true", help="text-to-video without the performer image")
    s.add_argument("--backend", default="auto", choices=backends); s.add_argument("--preset", choices=tuple(PRESETS)); s.set_defaults(fn=cmd_prompt)

    s = sub.add_parser("video", aliases=["v2m"], help="[1 source] video -> motion")
    s.add_argument("video"); s.add_argument("-o", "--out", required=True)
    s.add_argument("--backend", default="auto", choices=backends); s.add_argument("--variant", default="full", choices=("lite", "full", "heavy"))
    s.add_argument("--raw", action="store_true", help="skip naturalize"); s.add_argument("--preset", choices=tuple(PRESETS)); s.set_defaults(fn=cmd_video)

    s = sub.add_parser("procedural", aliases=["generate"], help="[1 source] deterministic test clip")
    s.add_argument("case", choices=("walk_squat", "startle_jump", "walk_startle")); s.add_argument("output"); s.set_defaults(fn=cmd_procedural)

    s = sub.add_parser("naturalize", help="[2 motion] clean-up passes with metrics")
    s.add_argument("input"); s.add_argument("output"); s.add_argument("--config", help="JSON NaturalizeConfig overrides")
    s.add_argument("--breathing", type=float); s.set_defaults(fn=cmd_naturalize)

    s = sub.add_parser("physique", help="[3 body] attach a physique profile")
    s.add_argument("input"); s.add_argument("output"); s.add_argument("--preset", choices=tuple(PRESETS))
    for key in ("muscular", "bulge", "bulge_size", "chest", "waist"):
        s.add_argument("--" + key.replace("_", "-"), dest=key, type=float)
    s.add_argument("--betas"); s.set_defaults(fn=cmd_physique)

    s = sub.add_parser("blender", help="[4 export] SMPL-X Blender add-on .npz")
    s.add_argument("input"); s.add_argument("output"); s.set_defaults(fn=cmd_blender)
    s = sub.add_parser("mesh", help="[4 export] bake vertices + hair attachment")
    s.add_argument("input"); s.add_argument("output"); s.add_argument("--step", type=int, default=1); s.add_argument("--no-correctives", action="store_true"); s.set_defaults(fn=cmd_mesh)
    s = sub.add_parser("rig", help="[4 export] full-skin local rig data (needs torch + smplx)")
    s.add_argument("input"); s.add_argument("output"); s.add_argument("--physique", choices=("neutral", *PRESETS), default="neutral")
    s.add_argument("--strength", type=float, default=1.0); s.add_argument("--profile", help='physique profile JSON, overrides --physique'); s.set_defaults(fn=cmd_rig)
    s = sub.add_parser("render", help="[4 export] software render of one frame")
    s.add_argument("input"); s.add_argument("output"); s.add_argument("--frame", type=int, default=0); s.add_argument("--size", type=int, default=640)
    s.add_argument("--yaw", type=float, default=30.0); s.add_argument("--pitch", type=float, default=5.0); s.set_defaults(fn=cmd_render)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
