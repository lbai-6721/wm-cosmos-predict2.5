# World Model 训练说明（Cosmos 像素 WM，skip-dynamics）

本文档对应 `wm4vla` 当前主线训练方式：使用 Cosmos-Predict2.5 在 paired-view LeRobot LIBERO 数据集上训练像素级 skip-dynamics world model。

预训练权重：`nvidia/Cosmos-Predict2.5-2B/base/post-trained`

## 1. 训练目标

给定当前观测、任务文本、固定长度 masked action prefix 和延迟 `d`，预测 `d` 步之后的未来观测：

```text
(view_t, task, [a_t ... a_{t+d-1}], d) -> view_{t+d}
```

这里的 `d` 不是额外 supervision，而是直接作为主模型条件参与去噪 / flow 预测。

## 2. 当前数据组织

`wm4vla/datasets/dataset_lerobot_libero.py` 会把 LeRobot LIBERO 的双视角 episode 转成强配对 batch。

每个 dataset item 都包含两个样本：

- `batch 0`: `cam1_t -> cam1_{t+d}`
- `batch 1`: `cam2_t -> cam2_{t+d}`

每个视角固定为 5 帧短视频：

- `frame 0`: 当前观测
- `frame 1-4`: 重复 4 次的未来观测

对应实验配置：

- `state_t=2`
- `min_num_conditional_frames=1`
- `max_num_conditional_frames=1`

## 3. 条件如何进入模型

当前训练依赖四类条件：

- 视频条件：`frame 0`
- 文本条件：任务描述的 T5 嵌入
- 动作条件：固定长度 `max_delay=8` 的 masked action prefix
- 延迟条件：归一化到 `[0,1]` 的 `delay_scalar`

动作和延迟进入网络的路径是：

1. dataset 生成 `action: [max_delay, 8]` 和 `delay_scalar: [1]`
2. 网络把 `action` 编成 action summary，把 `delay_scalar` 编成 delay embedding
3. 这两路条件分别加到 `t_embedding_B_T_D` 和 `adaln_lora_B_T_3D`
4. 所有 Transformer block 和 final layer 都通过 AdaLN 调制消费这些条件

也就是说，`action` / `delay` 不是直接拼到视频 token 上，而是改写 timestep/AdaLN 条件路径。

## 4. 当前保留的实验

- `ac_libero_lerobot_256_pixels_2b`：全量任务
- `ac_libero_lerobot_256_pixels_2b_task0`：仅 task 0
- `ac_libero_lerobot_256_pixels_2b_task01`：task 0 + task 1

共同设置：

- 分辨率：`256x256`
- 动作槽数：`8`
- `dropout_rate=0.0`
- 推理推荐 `guidance=0`
- delay curriculum 只提升 `sampled_delay_max`，不改变模型 action 维度

## 5. 预计算文本嵌入

推荐先生成 `meta/t5_embeddings.pkl`：

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/precompute_libero_t5.py \
  --data-root /home/kyji/storage_net/tmp/lbai/cosmos-predict2.5/lerobot/lerobot--libero_10_image@v2.0 \
  --t5-model /home/kyji/public/models/google-t5-11b
```

生成后设置：

```bash
export LEROBOT_LIBERO_T5_EMB_PATH=${LEROBOT_LIBERO_DATA_ROOT}/meta/t5_embeddings.pkl
```

若不设置或文件不存在，dataset 会退回零文本嵌入。

## 6. 环境变量

```bash
export HF_ENDPOINT=https://hf-mirror.com
export LEROBOT_LIBERO_DATA_ROOT=/home/kyji/storage_net/tmp/lbai/cosmos-predict2.5/lerobot/lerobot--libero_10_image@v2.0
export LEROBOT_LIBERO_T5_EMB_PATH=${LEROBOT_LIBERO_DATA_ROOT}/meta/t5_embeddings.pkl
export IMAGINAIRE_OUTPUT_ROOT=/home/kyji/storage_net/tmp/lbai/cosmos-predict2.5/outputs/wm-output/test_04090139
# export WANDB_API_KEY=your_key
```

## 7. 冒烟测试

全集：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12342 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py -- \
  experiment=ac_libero_lerobot_256_pixels_2b \
  job.wandb_mode=disabled \
  '~dataloader_train.dataloaders'
```

单任务 task0：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12342 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py -- \
  experiment=ac_libero_lerobot_256_pixels_2b_task0 \
  job.wandb_mode=disabled \
  '~dataloader_train.dataloaders'
```

## 8. 课程学习训练

推荐 delay curriculum：

```text
[1,2] -> [1,3] -> [1,4] -> [1,5] -> [1,6] -> [1,7] -> [1,8]
```

每个阶段训练 `2000` iter，只调：

- `dataloader_train.dataset.sampled_delay_max`
- `dataloader_val.dataset.sampled_delay_max`
- `trainer.max_iter`

直接使用脚本：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
BATCH_SIZE=24 \
MASTER_PORT=12341 \
EXPERIMENT=ac_libero_lerobot_256_pixels_2b \
bash wm4vla/scripts/train_wm_delay_curriculum.sh
```

若你要跑 `task0` 或 `task01`，把 `EXPERIMENT` 改成对应实验名即可。脚本默认实验是 `task0`。

手动启动第一阶段示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=12342 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py -- \
  experiment=ac_libero_lerobot_256_pixels_2b \
  job.wandb_mode=disabled \
  dataloader_train.batch_size=28 \
  trainer.max_iter=2000 \
  dataloader_train.dataset.sampled_delay_max=2 \
  dataloader_val.dataset.sampled_delay_max=2 \
  '~dataloader_train.dataloaders'
```

## 9. checkpoint 转换

```bash
CHECKPOINTS_DIR=/home/kyji/storage_net/tmp/lbai/tmp/wm4lva-output/wm-output/wm-output/distill-output/libero-10-task0-6gpu/cosmos_interactive/cosmos3_interactive/dmd2_trigflow_distill_wm_libero_lerobot_256_task0/checkpoints
CHECKPOINT_ITER=$(cat "${CHECKPOINTS_DIR}/latest_checkpoint.txt")

python scripts/convert_distcp_to_pt.py \
  "${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}/model" \
  "${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}"
```

输出：

- `model_ema_bf16.pt`
- `model_ema_fp32.pt`

## 10. 评估与可视化

离线评估：

```bash
python scripts/eval_world_model.py \
  --ckpt "${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}/model_ema_bf16.pt" \
  --output outputs/eval_wm/results.json \
  --num-steps 35 \
  --samples-per-episode 20
```

可视化：

```bash
python scripts/visualize_wm.py \
  --ckpt "${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}/model_ema_bf16.pt" \
  --output outputs/wm_vis \
  --n-episodes 3 \
  --delay 1 \
  --num-steps 10
```

## 11. wm4vla 集成时如何取预测帧

```python
# video_out: [2, 3, 5, 256, 256]
cam1_pred = video_out[0, :, 1]
cam2_pred = video_out[1, :, 1]
```
