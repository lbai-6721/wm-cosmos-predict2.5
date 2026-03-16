# World Model 训练说明（wm-cosmos-predict2.5，skip-dynamics）

本文档说明如何在 `wm-cosmos-predict2.5` 中训练 `wm4vla` 的像素级 skip-dynamics world model。  
预训练权重：`nvidia/Cosmos-Predict2.5-2B/base/post-trained`

---

## 1. 训练目标与核心思路

**Skip-dynamics 目标：**

给定当前观测 `o_t`、延迟 action `a_{t+d}` 与延迟值 `d`，预测 `o_{t+d+1}`。

```text
Kinetix: (o_t, a_{t+d}, d) -> o_{t+d+1}
LIBERO : (o_t, task, a_{t+d}, d) -> o_{t+d+1}
```

**两类数据的关键差异：**

| 项目 | Kinetix | LIBERO LeRobot |
|---|---|---|
| 文本条件 | 无（零向量占位） | 有（T5-11B 预计算嵌入） |
| `t5_text_embeddings` | `zeros(512,1024)` | `t5_embeddings.pkl` 查表 |
| 视频布局 | 9 帧（`state_t=3`） | 5 帧（`state_t=2`，paired 双视角 batch） |
| 分辨率 | 128x128 | 256x256 |
| `action_dim` | 7（6+delay） | 8（7+delay） |

**共同点：**
- `d in [0, max_delay-1]`，默认 `max_delay=5`
- action 输入为 `[action ; d/(max_delay-1)]`
- 每个目标帧重复 4 次填满一个 latent
- 训练采用纯条件：`dropout_rate=0.0`
- 推理采用纯条件：`guidance=0`（跳过 uncond 分支，速度更快）

---

## 2. 代码位置

| 文件 | 说明 |
|---|---|
| `wm4vla/datasets/dataset_kinetix.py` | Kinetix 9 帧数据集 |
| `wm4vla/datasets/dataset_lerobot_libero.py` | LIBERO LeRobot paired 5 帧数据集（含文本） |
| `wm4vla/configs/data_registry.py` | dataloader 注册（全量/task0/task01） |
| `wm4vla/configs/experiments.py` | 实验配置注册 |
| `cosmos_predict2/_src/predict2/action/configs/action_conditioned/conditioner.py` | 纯条件训练配置（dropout=0） |
| `cosmos_predict2/_src/predict2/action/models/action_conditioned_video2world_rectified_flow_model.py` | `guidance==0` 跳过 uncond 前向 |
| `scripts/precompute_libero_t5.py` | 预计算 T5 嵌入 |
| `scripts/eval_world_model.py` | 离线评估 |
| `scripts/visualize_wm.py` | 可视化输出 |

---

## 3. Kinetix 训练（无文本条件）

### 3.1 环境变量

```bash
export KINETIX_DATA_PIXELS_DIR=/path/to/data_pixels
export IMAGINAIRE_OUTPUT_ROOT=/home/kyji/storage_net/tmp/lbai/wm-cosmos-predict2.5/outputs/wm-output
# export WANDB_API_KEY=your_key
```

### 3.2 冒烟训练（1 GPU）

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py -- \
  experiment=ac_kinetix_pixels_2b \
  job.wandb_mode=disabled \
  '~dataloader_train.dataloaders'
```

### 3.3 正式训练（4 GPU）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py -- \
  experiment=ac_kinetix_pixels_2b \
  job.wandb_mode=disabled \
  dataloader_train.batch_size=32 \
  '~dataloader_train.dataloaders'
```

---

## 4. LIBERO LeRobot 训练（含文本条件）

### 4.1 数据与帧布局

推荐数据目录：

```text
/home/kyji/storage_net/tmp/lbai/wm-cosmos-predict2.5/lerobot/lerobot--libero_spatial_image@v2.0
```

paired 5 帧布局（`state_t=2`）：

| batch / 帧索引 | 内容 |
|---|---|
| batch0 / 0 | `cam1_t` |
| batch0 / 1-4 | `cam1_{t+d+1} x4`（评估取 frame 1） |
| batch1 / 0 | `cam2_t` |
| batch1 / 1-4 | `cam2_{t+d+1} x4`（评估取 frame 1） |

说明：`dataset_lerobot_libero.py` 返回 `video: [2, 3, 5, 256, 256]`、`action: [2, 1, 8]`、`t5_text_embeddings: [2, 512, 1024]`，两个视角在 batch 维强配对。

### 4.2 预计算文本嵌入（一次性）

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_libero_t5.py \
  --data-root /home/kyji/storage_net/tmp/lbai/wm-cosmos-predict2.5/lerobot/lerobot--libero_spatial_image@v2.0 \
  --t5-model /home/kyji/public/models/google-t5-11b
```

输出：`meta/t5_embeddings.pkl`

### 4.3 环境变量

```bash
export LEROBOT_LIBERO_DATA_ROOT=/home/kyji/storage_net/tmp/lbai/cosmos-predict2.5/lerobot/lerobot--libero_10_image@v2.0
export LEROBOT_LIBERO_T5_EMB_PATH=${LEROBOT_LIBERO_DATA_ROOT}/meta/t5_embeddings.pkl
export IMAGINAIRE_OUTPUT_ROOT=/home/kyji/storage_net/tmp/lbai/wm-cosmos-predict2.5/wm-output/output/re_new_v3
```

### 4.4 冒烟训练（1 GPU）

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12342 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py -- \
  experiment=ac_libero_lerobot_256_pixels_2b \
  job.wandb_mode=disabled \
  '~dataloader_train.dataloaders'
```

### 4.5 正式训练（2-4 GPU）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=12342 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py -- \
  experiment=ac_libero_lerobot_256_pixels_2b_task0 \
  job.wandb_mode=disabled \
  dataloader_train.batch_size=16 \
  trainer.max_iter=500000 \
  '~dataloader_train.dataloaders'
```

### 4.6 子任务训练

- 单任务：`experiment=ac_libero_lerobot_256_pixels_2b_task0`
- 双任务：`experiment=ac_libero_lerobot_256_pixels_2b_task01`

---

## 5. checkpoint 转换

```bash
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/<job_name>/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)

python scripts/convert_distcp_to_pt.py \
  $CHECKPOINTS_DIR/$CHECKPOINT_ITER/model \
  $CHECKPOINTS_DIR/$CHECKPOINT_ITER
```

输出：
- `model_ema_bf16.pt`（推荐推理）
- `model_ema_fp32.pt`

---

## 6. 评估与可视化

### 6.1 离线评估

```bash
python scripts/eval_world_model.py \
  --ckpt $CHECKPOINTS_DIR/$CHECKPOINT_ITER/model_ema_bf16.pt \
  --output outputs/eval_wm/results.json \
  --num-steps 35 \
  --samples-per-episode 20
```

### 6.2 可视化

```bash
python scripts/visualize_wm.py \
  --ckpt $CHECKPOINTS_DIR/$CHECKPOINT_ITER/model_ema_bf16.pt \
  --output outputs/wm_vis \
  --n-episodes 3 \
  --delay 1 \
  --num-steps 10
```

---

## 7. 与 wm4vla eval 对接

模型输出 `video_out: [2, 3, 5, 256, 256]`（值域 `[-1,1]`）时：

```python
cam1_pred = video_out[0, :, 1]
cam2_pred = video_out[1, :, 1]
obs["observation.images.image"] = cam1_pred
obs["observation.images.wrist_image"] = cam2_pred
```

## 8. B=2 推理采样注意事项

在 paired-view 5 帧方案中，`cam1/cam2` 不是拼在同一个时间轴，而是作为强配对 batch 输入 WM：

- `video[0]` 对应 `cam1`
- `video[1]` 对应 `cam2`

因此推理采样必须按整批 `B=2` 一起更新 latent。若采样器错误地只更新 `latents[0]`，会出现：

- `cam1_pred` 看起来正常
- `cam2_pred` 异常接近 `cam1_pred`
- 容易误判为取帧索引或列名错误

`wm-cosmos-predict2.5` 已完成此修复，采样 step 统一改为 batch 级更新（不再使用 `latents[0]` 路径）：

- `cosmos_predict2/_src/predict2/models/text2world_model.py`
- `cosmos_predict2/_src/predict2/models/text2world_wan2pt1_model.py`
- `cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py`
- `cosmos_predict2/_src/predict2/models/interpolator_model_rectified_flow.py`

这属于**推理路径修复**，不是训练目标变更。若 checkpoint 本身已按 paired 5 帧语义训练，通常不需要仅因该 bug 重训。
