"""Draw diagnostic stick-figure MP4s from the test SMPL-X parameters.

This preview is deliberately labeled: it does not evaluate the SMPL-X mesh.
"""

import argparse
import math
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from smplx_bridge import load_clip


W, H = 640, 360
SCALE = 70


def point(x, y):
    return (round(W / 2 + SCALE * x), round(288 - SCALE * y))


def draw_frame(clip, i):
    t = i / clip["fps"]
    root_x, root_y, _ = clip["transl"][i]
    angles = clip["body_pose"][i].reshape(21, 3)
    pelvis = (float(root_x), 1.02 + float(root_y))
    chest = (pelvis[0] - 0.05 * math.sin(float(angles[2, 0])), pelvis[1] + 0.49)
    head = (chest[0], chest[1] + 0.37)
    im = Image.new("RGB", (W, H), "#edf1f4")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 289, W, H), fill="#667786")
    d.line((point(-2.5, 0)[0], 289, point(-2.5, 0)[0], 295), fill="#f4a340", width=5)
    d.line((point(0, 0)[0], 289, point(0, 0)[0], 295), fill="#f4a340", width=5)
    d.line((point(-2.5, 0.0)[0], 100, point(-2.5, 0.0)[0], 289), fill="#344755", width=6)
    d.line((point(-2.5, 0.0)[0], 100, point(-2.0, 0.0)[0], 100), fill="#344755", width=6)
    d.line((point(*pelvis), point(*chest)), fill="#315f8d", width=23)
    d.line((point(*chest), point(*head)), fill="#d7a27b", width=10)
    hx, hy = point(*head)
    d.ellipse((hx - 16, hy - 16, hx + 16, hy + 16), fill="#d7a27b", outline="#344755", width=2)
    # Scalp marker: a placeholder for the attachment transform.
    d.ellipse((hx - 4, hy - 20, hx + 4, hy - 12), fill="#e15554")
    for side, hip_id, knee_id, shoulder_id, elbow_id in (
        (-1, 0, 3, 15, 17), (1, 1, 4, 16, 18)
    ):
        hip = (pelvis[0] + side * 0.13, pelvis[1] - 0.08)
        hip_a = float(angles[hip_id, 0])
        knee_a = float(angles[knee_id, 0])
        knee = (hip[0] + 0.43 * math.sin(hip_a), hip[1] - 0.43 * math.cos(hip_a))
        ankle = (knee[0] + 0.44 * math.sin(hip_a + knee_a), knee[1] - 0.44 * math.cos(hip_a + knee_a))
        d.line((point(*hip), point(*knee)), fill="#253548", width=14)
        d.line((point(*knee), point(*ankle)), fill="#253548", width=11)
        d.ellipse((*[v - 5 for v in point(*knee)], *[v + 5 for v in point(*knee)]), fill="#f4a340")
        shoulder = (chest[0] + side * 0.16, chest[1] + 0.03)
        arm_a = float(angles[shoulder_id, 0])
        elbow = (shoulder[0] + side * 0.07 - 0.27 * math.sin(arm_a), shoulder[1] - 0.27 * math.cos(arm_a))
        wrist = (elbow[0] + side * 0.04, elbow[1] - 0.25)
        d.line((point(*shoulder), point(*elbow)), fill="#315f8d", width=10)
        d.line((point(*elbow), point(*wrist)), fill="#d7a27b", width=8)
    title = f"{clip['case']}   {t:04.2f}s / {len(clip['transl']) / clip['fps']:.1f}s"
    d.text((20, 16), title, fill="#172735", font=ImageFont.load_default(size=22))
    d.text((20, 43), "SMPL-X PARAMETER PREVIEW | no model mesh, no MotionBricks", fill="#a23e3c", font=ImageFont.load_default(size=13))
    d.text((20, 328), "orange: stage marks    red: head attachment marker", fill="white", font=ImageFont.load_default(size=14))
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clip", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    clip = load_clip(a.clip)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="worldlines-motion-") as tmp:
        for i in range(len(clip["transl"])):
            draw_frame(clip, i).save(Path(tmp) / f"{i:04d}.png")
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(clip["fps"]),
            "-i", str(Path(tmp) / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(a.output)
        ], check=True)
    print(a.output)


if __name__ == "__main__":
    main()
