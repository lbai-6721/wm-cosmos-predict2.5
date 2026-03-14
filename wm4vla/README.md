# wm4vla — World Model for VLA

基于 NVIDIA Cosmos-Predict2.5 的 skip-dynamics 像素级 world model 训练模块。

## 目录结构

```
wm4vla/
├── datasets/                          # 数据集类
│   ├── dataset_kinetix.py             # Kinetix skip-dynamics (128×128, 9帧)
│   ├── dataset_lerobot_libero.py      # LIBERO LeRobot parquet (256×256, 17帧)
│   └── dataset_libero.py             # LIBERO HDF5 (128×128, 13帧)
│
├── configs/                           # Hydra 配置注册
│   ├── experiments.py                 # 5个实验配置 (Kinetix / LIBERO / LeRobot)
│   └── data_registry.py              # 数据加载器注册
│
├── scripts/                           # 工具脚本
│   ├── precompute_libero_t5.py        # 预计算 T5 文本嵌入
│   ├── eval_world_model.py            # 离线评估 (PSNR/SSIM/LPIPS)
│   └── visualize_wm.py               # WM 输出可视化
│
└── doc/                               # 文档
    ├── train_wm_pixels.md             # 训练流程说明
    ├── project.md                     # 项目架构说明
    └── data.md                        # 数据格式说明
```

## 与 Cosmos 原始代码的关系

本模块通过桥接方式与 Cosmos 原始代码集成，不修改 Cosmos 核心逻辑：

- `cosmos_predict2/experiments/base/action.py` → 调用 `wm4vla.configs.experiments.register_wm4vla_experiments()`
- `cosmos_predict2/.../data.py` → 调用 `wm4vla.configs.data_registry.register_wm4vla_data()`
- `cosmos_predict2/.../datasets/dataset_*.py` → re-export shim（向后兼容）
- `scripts/*.py` → forwarding shim 到 `wm4vla/scripts/`

原始训练命令**完全不变**：

```bash
torchrun --nproc_per_node=4 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py -- \
  experiment=ac_libero_lerobot_256_pixels_2b
```

## 仍保留在 Cosmos 原文件中的修改

以下修改因为影响范围小且与 Cosmos 基础设施紧密耦合，保留在原位：

1. **`conditioner.py`** — `ActionConditionedConditionerConfig`（dropout_rate=0.0，纯条件训练）
2. **`rectified_flow_model.py`** — `guidance==0` 时跳过 uncond forward（推理 ~2× 加速）
