# Worldlines Motion Lab

独立的角色动作实验项目。当前有标准动作和 SMPL-X 姿态接口；下一阶段是 MotionBricks 重定向与头发引擎集成。此项目不依赖某一部短剧的剧本或镜头目录。

## 可观看的三维白模测试

[三段双机位视频、关键帧和逐帧数据](demos/synthetic-lbs/README.md)已生成。蓝衣角色是本项目自制测试身型，逐帧调用 SMPL-X 上游的 LBS 蒙皮函数；红点是头部/头皮锚点。它验证了三维动作与附件数据链路，不代表已取得或运行官方 SMPL-X Body。

## Blender 官方 SMPL-X 蒙皮接口

[Blender 场景与导入步骤](demos/blender/README.md)已生成。仓库内三段 `.blend` 是可编辑的自制白模。另使用官网男性 SMPL-X 模型，在本机忽略目录 `demos/local-smplx/` 生成三段真实人体 `.full.blend` 和视频：10,475 顶点、55 关节、官方蒙皮权重和逐帧姿态修正。另有一个胸肩臂加厚、腰部收紧的肌肉男自定义塑形 Blender 动画与视频。Blender 首、中、末帧与上游完整 SMPL-X LBS 的最大顶点误差约 1 微米。模型派生文件不提交到开源仓库；肌肉塑形本身不是官方 beta 参数。

## 当前结果（2026-10-23）

这三个动作仍是**程序生成的测试**，不是 MotionBricks 输出或 G1→SMPL-X 重定向。官网 Julia 男性模型已用于本地 Blender 人体蒙皮验证；这个 NPZ 省略了 `smplx.create` 所需的手部 PCA 字段，所以 `prepare_official_smplx.py` 直接读取官网顶点、权重、骨架和姿态修正，并调用上游 `smplx.lbs` 做对照。头发引擎仓库/接口尚未定位，`hair_attachment.json` 仍是待对接的契约。

| 用例 | SMPL-X 参数 | 诊断视频 | 技术检查 |
|---|---|---|---|
| 男生入画、停下、深蹲、起身 | [walk_squat.smplx.npz](walk_squat.smplx.npz) | [walk_squat.preview.mp4](walk_squat.preview.mp4) | 根位移、髋膝踝曲线、蹲下时头部与头发根部高度 |
| 原地受惊、预备、跳起、落地 | [startle_jump.smplx.npz](startle_jump.smplx.npz) | [startle_jump.preview.mp4](startle_jump.preview.mp4) | 起跳和落地时刻、头部加速度、头发惯性输入 |
| 入画、停下、受惊跳起 | [walk_startle.smplx.npz](walk_startle.smplx.npz) | [walk_startle.preview.mp4](walk_startle.preview.mp4) | 行走到突发动作的过渡、停步脚接触 |

预览是参数诊断用的二维火柴人；它不显示 SMPL-X 实际身型、皮肤形变或头发物理。当前动作只证明格式与时序，尚不能以“无脚滑、关节自然”作为结论。

## 本机运行

```sh
python3 -m pip install -e .
SMPLX_SOURCE="$HOME/.x-repo/github.com/vchoutas/smplx"
PYTHONPATH="$SMPLX_SOURCE" python3 -m unittest discover -s . -p 'test_*.py' -v
# 取得适用许可和模型文件后，运行真实人体网格推理：
PYTHONPATH="$SMPLX_SOURCE" python3 smplx_bridge.py run \
  walk_squat.smplx.npz \
  /path/to/SMPLX_MALE.npz \
  /tmp/worldlines-walk-squat --gender male
```

成功后输出 `smplx_animation.npz`（`vertices[T,V,3]`、`faces[F,3]`、帧率）和 `hair_attachment.json`（每帧世界坐标头关节位置、`xyzw` 头部四元数）。如果头发引擎已有固定头皮顶点索引，给 `--scalp-indices /path/to/indices.npy`，会追加逐帧头皮顶点。模型文件必须由使用者按对应许可获取；仓库不包含模型文件。

## 整条工作链

1. **导演指令 → 时间事件**：把“从门口入画，停下，蹲下”拆成入画、停步、下蹲、保持、起身。保存每段起止时间和空间目标。剧情里的惊吓需另给触发事件（声音、道具、演员反应方向）。
2. **动作源 → 中立 `MotionClip`**：先用以上程序用例做基线；文字项目用导演事件驱动 MotionBricks，视频项目先恢复人物根轨迹、人体姿态和脚接触。两路都存为有帧率、骨架、坐标、来源的 `MotionClip`，保留原始帧。MotionBricks 当前公开预览主骨架是 34 关节 G1，需先解码为根位移、关节位置/旋转、脚接触，不能直接把 29 个机器人铰链角填入 SMPL-X 的 63 维身体轴角。[MotionBricks 动作格式](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/motionbricks/docs/motion_representation.md)
3. **G1 → 人体 → SMPL-X 拟合**：统一坐标系和尺度，匹配两侧髋、膝、踝、肩、肘、腕、腰、头；用逆运动学求 21 个 SMPL-X 局部轴角；再优化脚接地、步长、关节范围及动作连续性。无法对应的 G1 铰链、手部/脸部细节不能凭空恢复，需另一路表演或规则层补足。
4. **SMPL-X → 角色和头发**：将人物固定 `betas` 与动作逐帧送入官方模型；头发以头部四元数和头皮网格位置作为边界条件，另外输入速度、重力、风、碰撞体。头发在身体姿态计算后更新，并与身体、镜头用相同时间戳。当前只产出头发附件数据；具体引擎的坐标、单位、骨骼与头皮索引仍待映射。
5. **镜头合成与验收**：Blockout 管演员根轨迹、机位、焦段和场景；我们的动画层在同一时间轴上驱动 SMPL-X。逐项看脚滑、蹲起重心、跳起落地、头发滞后和穿模。Blockout 目前没有现成的逐帧 SMPL-X 轨道，因此先做外部合成，再考虑扩展其求值器。

## 下一道实际验收

真实 SMPL-X 网格与蒙皮已在 Blender 验证。下一步检查深蹲时脚底接地和关节自然度；拿到头发引擎后，从发根绑定提取头皮索引，导入 `hair_attachment.json`，测试落地后的发丝延迟与穿模。拿到可运行的 MotionBricks CUDA 环境后，用同三段条件生成 G1 输出，再比较真实的 G1→SMPL-X 拟合质量。

## 应用到具体影片

标准动作是技术基线。影片项目可以引用这些动作并增加角色身型、走位、道具接触和摄影机参数；具体剧本的改编意见留在影片项目内，不进入本项目的核心动作数据。

## 上游源码与许可边界

官方 [vchoutas/smplx](https://github.com/vchoutas/smplx) 已单独拉取在共享仓库目录 `~/.x-repo/github.com/vchoutas/smplx`，检出版本 `1265df7`。源码仓库不包含 `SMPLX_MALE.npz/.pkl`；本机从[官方站点](https://smpl-x.is.tue.mpg.de/)单独取得模型。`worldlines-motion` 不复制上游源码或模型。

上游仓库的 [LICENSE](https://github.com/vchoutas/smplx/blob/main/LICENSE) 对该仓库的软件与模型写有非商业用途及再分发限制。我们的基础包不自动安装 `smplx`；取得适用许可后可选择性安装 `worldlines-motion[smplx]`，或在本地测试时把官方源码目录加入 `PYTHONPATH`。发布开源工具前须核定实际使用的软件与模型许可。
