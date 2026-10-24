# GVHMR worker（CUDA）

这一目录是 [prompt2rig](https://github.com/blendi-remade/prompt2rig)（Apache-2.0，见 `LICENSE.prompt2rig` 与 `NOTICE`）的 `hq/` 工作进程原样移植：`worker.py` 把 GVHMR 常驻在 GPU 上，`POST /process {"video_url"}` 返回 SMPL 身体参数（`global_orient`、`body_pose`、`transl`、`fps`、可选 WiLoR 手指关键点）。`worldlines_motion.video_to_motion.gvhmr_client` 消费这一协议并转换成本项目的 `.smplx.npz` 片段。

需要 NVIDIA GPU。Mac 上不能运行它，但可以把 `GVHMR_WORKER_URL` 指向一台跑着它的 Linux/Windows 机器或云 GPU，工作台和命令行会自动优先使用它，否则回退到 MediaPipe 解算。

## 安装

Windows：`setup.ps1`（原脚本）。Linux：

```sh
cd worldlines_motion/video_to_motion/gvhmr_worker
bash setup.sh                # venv, CUDA torch, gvhmr + 约 5 GB 权重
cp /path/to/SMPLX_NEUTRAL.npz body_models/smplx/
.venv/bin/python worker.py   # 127.0.0.1:5178
```

GVHMR、SMPL-X、WiLoR/MANO 为研究 / 非商业许可，YOLO 为 AGPL-3.0；商业产品需另行授权或替换。
