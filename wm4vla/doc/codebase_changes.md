# wm-cosmos-predict2.5 改造说明（wm4vla）

本文档记录 `wm-cosmos-predict2.5/wm4vla` 中为复现
`cosmos-predict2.5/wm4vla` teacher world model 训练流程所做的接线改动。

## 改造目标

- 在当前仓库中接入 action-conditioned skip-dynamics teacher world model 训练/评估路径。
- 训练方法、数据 batch 语义、action prefix 与 delay 条件接口与 `cosmos-predict2.5/wm4vla` 保持一致。
- 移除 Kinetix 与旧 HDF5 LIBERO teacher 训练系列。
- 新增 PI-LIBERO teacher experiments。

## 当前数据路径

当前只保留 parquet 数据流：

- `wm4vla/datasets/dataset_lerobot_libero.py`
- `wm4vla/datasets/dataset_pi_libero.py`

已移除旧系列：

- Kinetix pixel dataset
- official LIBERO HDF5 dataset

## Cosmos 接线

`cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py`
在 `register_training_and_val_data()` 末尾调用：

```python
from wm4vla.configs.data_registry import register_wm4vla_data

register_wm4vla_data()
```

因此 Hydra 可以注册 wm4vla dataloader。

`wm4vla/configs/action_conditioned/config.py` 是 teacher 训练配置入口，统一注册：

- wm4vla dataloader
- wm4vla tokenizer
- wm4vla teacher experiments

## 已注册 Teacher Experiments

LeRobot LIBERO：

- `ac_libero_lerobot_256_pixels_2b`
- `ac_libero_lerobot_256_pixels_2b_task0`
- `ac_libero_lerobot_256_pixels_2b_task01`

PI-LIBERO：

- `ac_pi_libero_256_pixels_2b_all`
- `ac_pi_libero_256_pixels_2b_10`
- `ac_pi_libero_256_pixels_2b_goal`
- `ac_pi_libero_256_pixels_2b_object`
- `ac_pi_libero_256_pixels_2b_spatial`

不再注册：

- `ac_kinetix_pixels_2b`
- `ac_libero_pixels_2b`

## 训练语义

主线 batch 接口为：

```text
video              : [2, 5, 3, 256, 256]
action             : [2, 8, 8]
delay_scalar       : [2, 1]
t5_text_embeddings : [2, 512, 1024]
```

其中 `action` 是 fixed 8-slot masked action prefix。每个 slot 为：

```text
[raw_action_7d, valid_mask]
```

`delay_scalar` 不拼进每个 action slot，而是作为独立归一化标量输入：

```text
(delay - 1) / (max_delay - 1)
```

这个接口与 `cosmos-predict2.5/wm4vla` 的 teacher world model 训练流程一致。
