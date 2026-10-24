# SMPL-X LBS 白模动作演示

这是一个可观看、可重复的**接口实验**。蓝衣角色的网格、静止骨架、蒙皮权重均由本项目程序生成；动作参数采用 SMPL-X 的 21 个身体关节轴角顺序；逐帧形变调用已单独拉取的 `vchoutas/smplx` 仓库中的 `smplx.lbs.lbs`。没有使用官方 SMPL-X Body 模型文件、MotionBricks 生成动作或正式头发引擎。

| 动作 | 双机位视频 | 关键画面 | 数据 |
|---|---|---|---|
| 走入、停下、深蹲、起身 | [6 秒 MP4](walk_squat.synthetic_lbs.mp4) | [深蹲中段](walk_squat.frame-045.png) | [网格与关节](walk_squat.synthetic_lbs.npz) · [头皮锚点](walk_squat.hair_attachment.json) |
| 原地受惊跳起、落地 | [4 秒 MP4](startle_jump.synthetic_lbs.mp4) | [跳起中段](startle_jump.frame-030.png) | [网格与关节](startle_jump.synthetic_lbs.npz) · [头皮锚点](startle_jump.hair_attachment.json) |
| 走入、停下、受惊跳起 | [5 秒 MP4](walk_startle.synthetic_lbs.mp4) | [动作中段](walk_startle.frame-037.png) | [网格与关节](walk_startle.synthetic_lbs.npz) · [头皮锚点](walk_startle.hair_attachment.json) |

红点表示头部锚点。`hair_attachment.json` 包含同帧率的头关节位置、`xyzw` 四元数和 12 个头皮顶点位置；这验证了**传递给头发系统所需的数据格式与时序**，没有运行头发模拟。

## 本轮可量化结果

- 输出尺寸 `1180×650`，每秒 15 帧；三段分别为 90、60、75 帧，视频时长为 6、4、5 秒。
- 自制网格每帧 2431 个顶点。深蹲时头关节从约 1.81 m 降到 1.41 m，两脚参考关节约在地面上方 0.08 m。
- 受惊跳起时头关节最高约 2.13 m，较起始升高约 0.32 m；结束后回到起始高度。
- 所有帧的网格、头关节和头皮锚点数目一致，数值有限。行为检查见 [`test_synthetic_lbs.py`](../../test_synthetic_lbs.py)。

## 复现

在项目根目录运行，先按上游许可准备 `smplx` 源码，机器还需 `torch`、`numpy`、`Pillow`、`ffmpeg`：

```sh
SMPLX_SOURCE="$HOME/.x-repo/github.com/vchoutas/smplx"
PYTHONPATH="$SMPLX_SOURCE:$PWD/legacy:$PWD" python3 legacy/render_lbs_demo.py demos/clips/walk_squat.smplx.npz demos/synthetic-lbs
PYTHONPATH="$SMPLX_SOURCE:$PWD/legacy:$PWD" python3 -m unittest discover -s legacy/tests -p 'test_*.py' -v
```

**边界**：这是自制白模的官方 LBS 函数测试，不是官方 SMPL-X Body 网格推理。真实身型的脚滑、关节自然度、头发物理与 G1→人体重定向仍需在各自运行时验证。`worldlines-motion` 不含上游源码或人体模型资产。
