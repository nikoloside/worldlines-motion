# Blender 中的 SMPL-X 动作

## 已可打开的 Blender 场景（自制测试身型）

| 动作 | Blender 文件 | 预览 |
|---|---|---|
| 入画、深蹲、起身 | [walk_squat.synthetic_rig.blend](walk_squat.synthetic_rig.blend) | [三维视频](../synthetic-lbs/walk_squat.synthetic_lbs.mp4) |
| 受惊跳起 | [startle_jump.synthetic_rig.blend](startle_jump.synthetic_rig.blend) | [Blender 渲染帧](startle_jump.synthetic_rig.png) |
| 入画、受惊跳起 | [walk_startle.synthetic_rig.blend](walk_startle.synthetic_rig.blend) | [三维视频](../synthetic-lbs/walk_startle.synthetic_lbs.mp4) |

这三个场景有可编辑的 55 关节骨架、线性蒙皮权重、关键帧、灯光、地面和机位。Blender 后台重新打开成功；在首帧、中间帧和末帧与上游 SMPL-X LBS 比较，顶点误差均小于 1 毫米。**人体网格与权重是本项目自制测试素材，不是官方 SMPL-X 模型。**

用 `build_blender_scene.py` 可重新生成；`synthetic_fixture.npz` 是该白模的顶点、骨架和权重。

以下三段 `.npz` 可供官方 **SMPL-X for Blender** 插件的 **Add Animation** 导入：

| 动作 | 文件 | 帧数 |
|---|---|---:|
| 入画、停下、深蹲、起身 | [walk_squat.blender_smplx.npz](walk_squat.blender_smplx.npz) | 180 |
| 原地受惊、跳起、落地 | [startle_jump.blender_smplx.npz](startle_jump.blender_smplx.npz) | 120 |
| 入画、停下、受惊跳起 | [walk_startle.blender_smplx.npz](walk_startle.blender_smplx.npz) | 150 |

它们包含 `poses[T,165]`（SMPL-X 55 关节轴角）、`trans[T,3]`、`betas[10]`、`gender` 和 `mocap_framerate`。身体姿态和根位移来自本仓库的程序测试动作；手部、下颌和眼睛目前保持零姿态。它们不是逐帧网格缓存，也没有内置人体模型。

## 在 Blender 中导入

1. 使用 Blender 4.5 的正式版本或更新版本。从 [SMPL-X 官方站点](https://smpl-x.is.tue.mpg.de/)取得 **SMPL-X for Blender** 插件 ZIP；该 ZIP 带有插件所需的 Locked Head 身型。仅下载[代码仓库](https://gitlab.tuebingen.mpg.de/jtesch/smplx_blender_addon)不能创建 SMPL-X 人体。
2. Blender → Edit → Preferences → Get Extensions → Install from Disk，安装并启用插件。
3. 在 3D Viewport 侧栏的 **SMPL Models** 页点击 **Add Animation**，选择本目录的任一 `.npz`。
4. **Format 选 AMASS**：这里表示源动作使用 Y-up 坐标，尽管文件有完整的 SMPL-X 55 关节；身型版本选 **Locked Head**。Target framerate 设为 **30**。
5. 若最终渲染需要逐帧 SMPL-X 姿态修正，勾选 **Use keyframed corrective pose weights**。这会增加导入时间和 Blender 播放负担。

也可在插件安装并启用后批量生成 `.blend`：

```sh
blender -b --python blender_import.py -- \
  demos/blender/walk_squat.blender_smplx.npz \
  demos/blender/walk_squat.blend --correctives
```

插件会创建自己提供的 SMPL-X 网格、形状专属骨架、蒙皮权重和形态键，并把动作写到骨架。`--correctives` 才会逐帧记录姿态修正形态键。运行脚本前请在 Blender 中启用插件。

## 目前的验证边界

导出器已按官方插件当前 `operators/animation.py` 的字段检查与 55 关节规格验证，且三段文件都可由 NumPy 无 pickle 读取。Blender 4.5.0 Beta 在受限沙箱中启动失败，在正常权限下已成功生成并重新打开白模 `.blend`。官方 Blender 插件模型 ZIP 尚未安装，因此真实 SMPL-X 人体蒙皮与姿态修正仍待验收。仓库不分发受限的模型文件。
