# Worldlines Motion Lab

独立的角色动作实验项目。当前有标准动作和 SMPL-X 姿态接口；下一阶段是 MotionBricks 重定向与头发引擎集成。此项目不依赖某一部短剧的剧本或镜头目录。

## 可观看的三维白模测试

[三段双机位视频、关键帧和逐帧数据](demos/synthetic-lbs/README.md)已生成。蓝衣角色是本项目自制测试身型，逐帧调用 SMPL-X 上游的 LBS 蒙皮函数；红点是头部/头皮锚点。它验证了三维动作与附件数据链路，不代表已取得或运行官方 SMPL-X Body。

## Blender 官方 SMPL-X 蒙皮接口

[三段可播放的 Blender 场景、官方插件动作文件及步骤](demos/blender/README.md)已生成。现有 `.blend` 使用自制白模、SMPL-X 的 55 关节顺序与线性蒙皮；已在 Blender 中抽样比对上游 LBS，顶点误差低于 1 毫米。它们不是官方 SMPL-X 人体。另有三段 `.npz` 已按官方 **SMPL-X for Blender** 插件格式封装；取得插件模型包后，可用其真实人体网格、骨架和姿态修正形态键导入。

## 当前结果（2026-10-23）

这三个文件是**程序动作写入 SMPL-X 参数格式**的测试，不是 MotionBricks 生成、G1→SMPL-X 重定向或 SMPL-X 网格渲染。`smplx_bridge.py` 已能用官方 `smplx.create` 接收这些参数，逐帧计算网格与头部位置，再导出头部四元数和可选头皮顶点；本机尚无 `SMPLX_MALE.npz/.pkl` 模型资产，因此实际网格推理没有跑通。头发引擎仓库/接口也尚未定位，所以 `hair_attachment.json` 是对接契约，尚未送入引擎。

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

拿到男性 SMPL-X 模型文件后，对三段参数运行官方前向推理：确认顶点、关节和头部锚点逐帧数量一致；抽查蹲下帧与跳跃帧的脚底位置。拿到头发引擎后，从它的静态头套或发根绑定数据提取顶点索引，导入 `hair_attachment.json`，用“跳起后落地”检查发丝延迟、穿模与坐标方向。拿到可运行的 MotionBricks CUDA 环境后，用相同三段条件生成 G1 输出，再开始真正的 G1→SMPL-X 拟合质量比较。

## 应用到具体影片

标准动作是技术基线。影片项目可以引用这些动作并增加角色身型、走位、道具接触和摄影机参数；具体剧本的改编意见留在影片项目内，不进入本项目的核心动作数据。

## 上游源码与许可边界

官方 [vchoutas/smplx](https://github.com/vchoutas/smplx) 已单独拉取在共享仓库目录 `~/.x-repo/github.com/vchoutas/smplx`，检出版本 `1265df7`。源码仓库不包含 `SMPLX_MALE.npz/.pkl`；模型需从[官方站点](https://smpl-x.is.tue.mpg.de/)注册取得。`worldlines-motion` 不复制上游源码或模型。

上游仓库的 [LICENSE](https://github.com/vchoutas/smplx/blob/main/LICENSE) 对该仓库的软件与模型写有非商业用途及再分发限制。我们的基础包不自动安装 `smplx`；取得适用许可后可选择性安装 `worldlines-motion[smplx]`，或在本地测试时把官方源码目录加入 `PYTHONPATH`。发布开源工具前须核定实际使用的软件与模型许可。
