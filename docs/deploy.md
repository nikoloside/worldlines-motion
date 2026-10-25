# 部署：哪台机器跑什么

整个流水线分两类进程：**工作台和命令行**（任何 Mac 或 Linux 都能跑，只要有 SMPL-X 模型文件），以及**GPU worker**（常驻在一台 NVIDIA 机器上，工作台通过 HTTP 调它）。worker 是可选的，没有它就回退到本机 CPU 路径，质量低一档。

| 进程 | 放在哪 | 为什么 |
|---|---|---|
| 工作台 `wlmotion workbench`、Blender 导出、审片渲染 | 任何 Mac（M2 本或隔壁 Mac Studio） | 纯 numpy + 浏览器，不需要显卡 |
| MediaPipe 视频解算（本机兜底） | 同上 | CPU 就行 |
| GVHMR worker（高质量视频 → SMPL） | **NVIDIA 机器** | 只有 CUDA 版本，约 5 GB 权重 |
| Kimodo worker（文字 → SMPL-X，待做） | **NVIDIA 机器**，和 GVHMR 放一起 | 需要 17 GB 显存，文字编码器挪到 CPU 可降到 3 GB；Mac 未验证 |
| MoMask（文字 → 关节，本机兜底，待做） | 任何 Mac | 小模型，CPU 几秒 |

结论：**两个 worker 都部署在 NVIDIA 机器上**，Mac Studio 只跑工作台。Mac Studio 的大内存对现有流程没有额外用处；HY-Motion 之类大模型虽然声称支持 macOS，没验证前不算进来。

## NVIDIA 机器上做什么

```sh
git clone https://github.com/nikoloside/worldlines-motion.git
cd worldlines-motion/worldlines_motion/source/gvhmr_worker
bash setup.sh                                  # venv、CUDA torch、GVHMR 权重
cp /path/to/SMPLX_NEUTRAL.npz body_models/smplx/
.venv/bin/python worker.py                     # 监听 127.0.0.1:5178
```

要让别的机器访问，改 `worker.py` 的监听地址为 `0.0.0.0`，或者用 ssh 隧道：在 Mac 上 `ssh -N -L 5178:127.0.0.1:5178 gpu-box`。

Kimodo worker 尚未写，计划和 GVHMR worker 同样的形状：`POST /generate {"prompt", "duration"}` 返回 `global_orient / body_pose / transl / fps`，客户端在 `worldlines_motion/source/` 里转成片段。

## Mac 上做什么

```sh
git clone https://github.com/nikoloside/worldlines-motion.git
cd worldlines-motion
python3 -m venv .venv && .venv/bin/pip install -e '.[video]'
cp .env.example .env            # 填 GVHMR_WORKER_URL=http://gpu-box:5178
PYTHONPATH=. .venv/bin/python -m worldlines_motion.workbench   # http://127.0.0.1:5190
```

SMPL-X 模型文件自己从官网下，放在 `~/Downloads/SMPLX_MALE.npz` 或用 `--model` 指定；仓库不带、也不要提交。

## 文字出动作：方向已定，尚未实装

现在第 ① 步的"文字"入口是 prompt2rig 搬来的"文字 → fal 生成视频 → 提动作"，**不是**我们要的。我们要的是文字直接出动作。决定如下：

- 撤掉 fal 这条路。视频由火山引擎 Seedance 在外面生成，从"视频"入口进来即可；`source/fal_video.py` 留作可选工具，不再出现在界面上。
- 文字入口改成真正的 text-to-motion，两个后端，和视频入口的 GVHMR / MediaPipe 一样的结构：
  - `kimodo`：NVIDIA 的 [Kimodo](https://github.com/nv-tlabs/kimodo)（2026-03），直接出 SMPL-X，支持多段文字排时间线。代码 Apache-2.0，SMPL-X 权重 NVIDIA R&D 许可，非商用可用。跑在 NVIDIA 机器上做 worker。
  - `momask`：[MoMask](https://github.com/EricGuo5513/momask-codes)，MIT，出 22 关节 20 fps，CPU 可跑，作为 Mac 本机兜底。
- 两者之间加一个共用的"关节 → SMPL-X 拟合"模块，用现有的 numpy LBS 做优化；MediaPipe 那条以后也能换到它。
- 出来的片段照常进 ② 动作、③ 身体、④ 导出。
