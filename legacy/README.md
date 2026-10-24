# legacy · 0.1 时代的脚本

这里是 `worldlines-motion` 0.1（独立 motion lab）阶段的脚本和测试，原样冻结，只为复现 `demos/synthetic-lbs/` 与 `demos/blender/` 里的基线素材。新代码请用 `worldlines_motion/` 包和 `wlmotion` 命令，不要再从这里 import。

| 脚本 | 作用 | 现在的替代 |
|---|---|---|
| `smplx_bridge.py` | 0.1 的片段格式、程序化测试动作、torch 版网格烘焙 | `worldlines_motion.clip`、`source.procedural`、`export.mesh` |
| `blender_smplx.py` | 官方 Blender 插件 `.npz` 导出 | `worldlines_motion.export.blender_addon` |
| `render_lbs_demo.py`、`render_proxies.py`、`preview_clips.py` | 自制白模的 LBS 视频与预览 | `wlmotion render` |
| `build_blender_scene.py`、`bake_pose_correctives.py`、`polish_muscular_scene.py`、`render_blender_video.py` | Blender 内跑的建场景、烘焙修正、出片脚本 | 仍在用，配合 `wlmotion rig` |
| `make_muscular_demo_clip.py` | 肌肉男演示片段 | `wlmotion physique` |

运行方式（需要上游 `smplx` 源码路径）：

```sh
SMPLX_SOURCE="$HOME/.x-repo/github.com/vchoutas/smplx"
PYTHONPATH="$SMPLX_SOURCE:$PWD/legacy:$PWD" python3 -m unittest discover -s legacy/tests -p 'test_*.py'
```
