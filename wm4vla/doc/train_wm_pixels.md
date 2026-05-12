# World Model 训练说明（Cosmos 像素 WM，skip-dynamics）

本文档说明如何使用 Cosmos 视频扩散模型训练用于 `wm4vla` 的像素级 skip-dynamics world model。
预训练权重：nvidia/Cosmos-Predict2.5-2B/base/post-trained

---

## 1. 训练目标与核心思路

**Skip-dynamics 设计：**

给定当前观测 `o_t`、固定长度掩码动作序列 `[a_t,\ldots,a_{t+d-1}]`、延迟值 `d`，预测延迟后的未来观测；LIBERO 还额外加入任务描述作为文本条件：

```
LIBERO： (o_t,  task,  [a_t…a_{t+d-1}],  d)   →  o_{t+d}   # 有文本条件（T5-11B）
```

**共同点：**
- 训练时 `d ∈ [1, max_delay]`（默认 `max_delay=8`），eval 支持 `d ∈ [0, max_delay]`
- 课程学习训练时保持固定 action 槽数不变，通过 `sampled_delay_max` 逐阶段提升 delay 采样上限
- `d=0` 时不调用 WM，直接使用真实观测
- action 条件为固定长度 `max_delay` 的动作序列，使用 mask 标记前 `d` 个有效动作；当前 LIBERO 输入为 `action: [max_delay, 8] = [action(7); mask]`
- delay 不再拼到每个 action slot；当前实现单独传入归一化 `delay_scalar: [1] = (d - 1) / (max_delay - 1)`
- action prefix 通过 per-slot MLP、Temporal MLP Mixer、masked pooling 汇总，再经 MLP head 注入 timestep / AdaLN 调制路径
- 预测帧重复 4 次填充一个 latent，推理时取第一帧（paired-view LIBERO 对应 frame 1）
- 纯条件训练：`dropout_rate=0.0`；纯条件推理：`guidance=0`（跳过 uncond 前向，约 2× 加速）

---

## 2. 相关文件

| 文件 | 说明 |
|---|---|
| `wm4vla/datasets/dataset_lerobot_libero.py` | LIBERO LeRobot v2.0 parquet 数据集（主用，含文本条件）；构造 `action` 和 `delay_scalar` |
| `wm4vla/conditioning/action_sequence.py` | 固定长度 masked action prefix 与归一化 delay scalar 的打包逻辑 |
| `wm4vla/configs/data_registry.py` | 注册 wm4vla dataloaders，设置 `t5_emb_path` |
| `cosmos_predict2/_src/predict2/action/configs/action_conditioned/conditioner.py` | 条件器（text + video + action + delay），纯条件训练 |
| `cosmos_predict2/_src/predict2/action/networks/action_conditioned_minimal_v1_lvg_dit.py` | action prefix MLP 编码、Temporal MLP Mixer、masked pooling 与 AdaLN 调制注入 |
| `wm4vla/configs/experiments.py` | wm4vla 实验 config |
| `scripts/precompute_libero_t5.py` | 预计算 T5 文本嵌入（LIBERO 专用，一次性） |
| `scripts/eval_world_model.py` | 离线评估 WM（PSNR/SSIM，按 delay 分组） |
| `scripts/visualize_wm.py` | 可视化 WM 输出（生成成对 5 帧 mp4，`guidance=0`） |
| `wm4vla/scripts/train_wm_delay_curriculum.sh` | 课程学习训练入口：按 `[1,2] → [1,8]` 分阶段提升 delay 采样上限 |

---

## 3. LIBERO-Spatial 训练（含文本条件）

### 3.1 数据说明

| 项 | 值 |
|---|---|
| 数据路径 | `lerobot/lerobot--libero_spatial_image@v2.0` |
| 格式 | LeRobot v2.0 parquet（每 episode 一个文件） |
| 总 episodes | ~500（10 tasks × ~50 demos） |
| 训练/验证分割 | `val_ratio=0.1, seed=0`（episode 级别，~450 train / ~50 val） |
| 图像 | cam1 + cam2 各 **256×256** uint8 |
| action | 7-dim，已归一化 [-1,1] |
| 网络 action 输入 | `action: [8,8]`：每槽 `[action(7) ; mask]` |
| 网络 delay 输入 | `delay_scalar: [1]`：归一化全局 delay，单独送入 delay MLP |
| 任务文本 | 10 种，见 `meta/tasks.jsonl` |

**训练窗口数量：**
```
有效 start_t/episode = T - max_delay ≈ 230 - 5 = 225
训练 episodes ≈ 450
总训练窗口 ≈ 450 × 225 = 101,250
```

| 总 batch_size | iter/epoch |
|---|---|
| 4（2卡×2） | ~25,312 |
| 8（2卡×4） | ~12,656 |
| 32（4卡×8） | ~3,164 |

**视频帧布局（强配对 batch，每个视角 5 帧，state_t=2，num_conditional_frames=1）：**

| 帧索引 | latent | 内容 | 说明 |
|---|---|---|---|
| 0 | latent 0（cond） | `view_t` | 当前视角条件帧 |
| 1–4 | latent 1（**pred**） | `view_{t+d}` ×4 | 预测未来帧（**取 frame 1**） |

其中每个窗口 `(episode, start_t, d)` 会生成一对强配对样本：

| batch 索引 | 视角 | 内容 |
|---|---|---|
| 0 | cam1 | `cam1_t -> cam1_{t+d}` |
| 1 | cam2 | `cam2_t -> cam2_{t+d}` |

### 3.2 Action prefix 与 MLP 调制

当前代码中 action prefix 和 delay 是两路条件，delay 不附加到 action slot：

1. dataset 采样 `d` 后取 `a_t ... a_{t+d-1}`，并调用 `pack_masked_action_sequence` 打包成固定 8 槽。
2. 前 `d` 个槽写入真实 7 维 action，并把最后 1 维 mask 置为 `1`；剩余槽保持全 `0`。
3. `delay_scalar` 单独由 `normalize_delay_scalar` 生成，shape 为 `[1]`，DataLoader 后为 `[B,1]`。
4. 网络先用 `action_slot_embedder` 对每个 slot 做 MLP 编码，再加入 slot position embedding。
5. 两层 `ActionMLPTemporalMixerBlock` 在有效 action 槽上做 token 维和 channel 维 MLP mixing。
6. `action_pooling_score` 对有效槽做 masked softmax pooling，得到一个 action summary。
7. action summary 经 `action_summary_to_B_D` / `action_summary_to_B_3D` 两个 MLP head，delay scalar 经 `delay_embedder_B_D` / `delay_embedder_B_3D` 两个 MLP head。
8. 两路 embedding 分别加到 `t_embedding_B_T_D` 和 `adaln_lora_B_T_3D`，由所有 DiT block 和 final layer 的 AdaLN shift / scale / gate 消费。

### 3.3 文本条件控制

`LeRobotLiberoDataset` 通过 `t5_emb_path` 参数控制文本编码行为：

| `t5_emb_path` 值 | 行为 |
|---|---|
| `None`（默认） | 自动查找 `<data_root>/meta/t5_embeddings.pkl`，找不到则 warn + 零张量 |
| `""`（空字符串） | 显式禁用文本条件，始终使用零张量 |
| `"/path/to/t5_embeddings.pkl"` | 从指定路径加载 |

支持三种方式传入：
1. **环境变量**（推荐，训练前设置一次）
2. **Hydra 命令行覆盖**
3. **自动检测**（pkl 放在默认位置即可）

#### Step 0 — 预计算 T5 嵌入（只需一次）

```bash
# （可选）下载编码器模型到本地
huggingface-cli download google-t5/t5-11b \
  --local-dir /home/kyji/public/models/google-t5-11b \
  --local-dir-use-symlinks False

# 使用本地模型编码（--t5-model 指定本地目录，无需网络）
CUDA_VISIBLE_DEVICES=7 python scripts/precompute_libero_t5.py \
    --data-root /home/kyji/storage_net/tmp/lbai/cosmos-predict2.5/lerobot/lerobot--libero_10_image@v2.0 \
    --t5-model /home/kyji/public/models/google-t5-11b
```

输出：`meta/t5_embeddings.pkl`（10 条任务，每条 `(1,512,1024) bfloat16`）

> pkl 的 key 为任务描述字符串，dataset 通过 `task_index → tasks.jsonl → 描述字符串 → embedding` 查表。

### 3.4 环境变量

```bash
export HF_ENDPOINT=https://hf-mirror.com
export PI_LIBERO_DATA_ROOT=/data1/vla-data/physical-intelligence/libero
export IMAGINAIRE_OUTPUT_ROOT=/home/jikangye/workspace/tmp/lbai/tmp/wm4vla-outputs/wm-cosmos-predict-outputs/reconstruct_20260511
export WANDB_API_KEY=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

# 文本条件方式一：通过环境变量指定（可选，None 时自动检测默认路径）
#export LEROBOT_LIBERO_T5_EMB_PATH=${LEROBOT_LIBERO_DATA_ROOT}/meta/t5_embeddings.pkl

# LightVAE 训练 / 评估 / 可视化共用
export LIGHTVAE_PTH=/home/jikangye/workspace/tmp/lbai/tmp/wm4vla-outputs/vae/delay5_8_30000/lightvae_step_0030000.safetensors
export LIGHTX2V_ROOT=/home/jikangye/workspace/tmp/lbai/LightX2V
LIGHTVAE_ARGS=(
  'tokenizer=wan2pt1_lightvae_tokenizer'
  "model.config.tokenizer.vae_pth=${LIGHTVAE_PTH}"
  "model.config.tokenizer.lightx2v_root=${LIGHTX2V_ROOT}"
  'model.config.tokenizer.use_batched_vae=true'
)
```

后续训练命令统一使用 `wm4vla/configs/action_conditioned/config.py`，并在命令末尾追加 `"${LIGHTVAE_ARGS[@]}"`。

### 3.5 冒烟测试（1 GPU）

**情况 A — 有文本条件（pkl 已生成）：**
```bash
# 方式1：pkl 放在默认位置，无需额外参数（自动检测）
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12342 -m scripts.train \
  --config=wm4vla/configs/action_conditioned/config.py -- \
  experiment=ac_libero_lerobot_256_pixels_2b \
  job.wandb_mode=disabled \
  '~dataloader_train.dataloaders' \
  "${LIGHTVAE_ARGS[@]}"

# 方式2：Hydra 显式覆盖路径
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12342 -m scripts.train \
  --config=wm4vla/configs/action_conditioned/config.py -- \
  experiment=ac_libero_lerobot_256_pixels_2b \
  job.wandb_mode=disabled \
  'dataloader_train.dataset.t5_emb_path=/path/to/meta/t5_embeddings.pkl' \
  '~dataloader_train.dataloaders' \
  "${LIGHTVAE_ARGS[@]}"
```

**情况 B — 无文本条件（不使用 T5，或 pkl 未生成）：**
```bash
# 显式禁用：t5_emb_path="" 强制使用零张量
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12342 -m scripts.train \
  --config=wm4vla/configs/action_conditioned/config.py -- \
  experiment=ac_libero_lerobot_256_pixels_2b \
  job.wandb_mode=disabled \
  'dataloader_train.dataset.t5_emb_path=""' \
  '~dataloader_train.dataloaders' \
  "${LIGHTVAE_ARGS[@]}"
```

### 3.6 课程学习正式训练（2–4 GPU）

课程学习阶段定义：

```text
[1,2] → [1,3] → [1,4] → [1,5] → [1,6] → [1,7] → [1,8]
```

每个阶段训练 `2000 iterations`。为了保证 checkpoint 可以平滑续训：
- 模型 action 输入仍固定为 8 槽，不改 `num_action_per_chunk`
- 只通过 `dataloader_{train,val}.dataset.sampled_delay_max` 控制当前阶段的 delay 采样上限
- 每次使用同一个 `experiment=ac_libero_lerobot_256_pixels_2b`，并把 `trainer.max_iter` 设为累计 iteration，训练会从同一路径下的 `latest_checkpoint.txt` 自动 resume

推荐直接运行课程学习脚本：

```
EXPERIMENT=ac_pi_libero_256_pixels_2b_10 \
  bash wm4vla/scripts/train_wm_delay_curriculum_pi_libero.sh "${LIGHTVAE_ARGS[@]}"
```

可选的 5 个 experiment 是：

  - ac_pi_libero_256_pixels_2b_all
  - ac_pi_libero_256_pixels_2b_10
  - ac_pi_libero_256_pixels_2b_goal
  - ac_pi_libero_256_pixels_2b_object
  - ac_pi_libero_256_pixels_2b_spatial


```bash
bash wm4vla/scripts/train_wm_delay_curriculum.sh trainer.logging_iter=5 "${LIGHTVAE_ARGS[@]}"
```

该脚本会顺序执行：
- `[1,2]` 训练到 `2000 iter`
- `[1,3]` 训练到 `4000 iter`
- `[1,4]` 训练到 `6000 iter`
- `[1,5]` 训练到 `8000 iter`
- `[1,6]` 训练到 `10000 iter`
- `[1,7]` 训练到 `12000 iter`
- `[1,8]` 训练到 `14000 iter`

如果你要手动启动，下面是前两个阶段的等价命令：

```bash
# Stage 1: delay ∈ [1,2]
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 --master_port=12342 -m scripts.train \
  --config=wm4vla/configs/action_conditioned/config.py -- \
  experiment=ac_libero_lerobot_256_pixels_2b \
  job.wandb_mode=disabled \
  dataloader_train.batch_size=28 \
  trainer.max_iter=2000 \
  dataloader_train.dataset.sampled_delay_max=2 \
  dataloader_val.dataset.sampled_delay_max=2 \
  '~dataloader_train.dataloaders' \
  "${LIGHTVAE_ARGS[@]}"

# Stage 2: delay ∈ [1,3]
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 --master_port=12342 -m scripts.train \
  --config=wm4vla/configs/action_conditioned/config.py -- \
  experiment=ac_libero_lerobot_256_pixels_2b \
  job.wandb_mode=disabled \
  dataloader_train.batch_size=28 \
  trainer.max_iter=4000 \
  dataloader_train.dataset.sampled_delay_max=3 \
  dataloader_val.dataset.sampled_delay_max=3 \
  '~dataloader_train.dataloaders' \
  "${LIGHTVAE_ARGS[@]}"
```

后续阶段继续把 `sampled_delay_max` 改成 `4,5,6,7,8`，并把 `trainer.max_iter` 改成 `6000,8000,10000,12000,14000` 即可。

> 注意：256×256 latent 32×32，显存约为 128×128 的 4 倍。A100 80G 建议 batch_size ≤ 8/卡。

### 3.7 前 2 个任务联合训练（task_index=0 和 1）

训练前 2 个任务，用于验证多任务文本条件效果，数据量约为全集的 1/5：

| 项 | 值 |
|---|---|
| 实验配置 | `ac_libero_lerobot_256_pixels_2b_task01` |
| 训练 episodes | ~100（task 0 + task 1 各约 50） |
| 训练窗口 | ~100 × 225 ≈ **22,500** |
| 文本条件 | task 0 / task 1 各自的 T5 描述嵌入 |
| checkpoint 路径 | `.../2b_libero_lerobot_256_skip_dynamics_dual_cam_task01/` |

每个 batch 中 task 0 和 task 1 的样本**随机混合**，模型学习根据不同 task_description 区分行为。

```bash
# 冒烟测试（1 GPU）
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12342 -m scripts.train \
  --config=wm4vla/configs/action_conditioned/config.py -- \
  experiment=ac_libero_lerobot_256_pixels_2b_task01 \
  job.wandb_mode=disabled \
  '~dataloader_train.dataloaders' \
  "${LIGHTVAE_ARGS[@]}"

# 正式训练（4 GPU）
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 --master_port=12342 -m scripts.train \
  --config=wm4vla/configs/action_conditioned/config.py -- \
  experiment=ac_libero_lerobot_256_pixels_2b_task01 \
  job.wandb_mode=disabled \
  dataloader_train.batch_size=24 \
  '~dataloader_train.dataloaders' \
  "${LIGHTVAE_ARGS[@]}"
```

| 总 batch_size | iter/epoch（task01） | 说明 |
|---|---|---|
| 8（1卡×8） | ~2,813 | 冒烟用 |
| 32（4卡×8） | ~703 | 正式训练 |

评估：

```bash
python scripts/eval_world_model.py \
    --ckpt .../2b_libero_lerobot_256_skip_dynamics_dual_cam_task01/checkpoints/.../model_ema_bf16.pt \
    --task-indices 0 1 \
    --experiment ac_libero_lerobot_256_pixels_2b_task01 \
    --t5-emb-path /path/to/meta/t5_embeddings.pkl \
    --tokenizer-backend lightvae \
    --tokenizer-vae-pth "${LIGHTVAE_PTH}" \
    --lightx2v-root "${LIGHTX2V_ROOT}" \
    --output outputs/eval_wm/task01.json
```

### 3.8 单任务快速测试（task_index=0）

在正式训练全部 10 个任务前，可先只训练 task 0 快速验证效果：

**Task 0**: `"pick up the black bowl next to the cookie box and place it on the plate"`

数据量约为全集的 1/10：~50 demos × 225 windows ≈ **11,250 训练窗口**（全集约 101,250）

实验配置：`ac_libero_lerobot_256_pixels_2b_task0`（使用 `task_indices=[0]` 过滤，其余参数与全集相同）

```bash
# 冒烟测试（1 GPU）
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12342 -m scripts.train \
  --config=wm4vla/configs/action_conditioned/config.py -- \
  experiment=ac_libero_lerobot_256_pixels_2b_task0 \
  job.wandb_mode=disabled \
  '~dataloader_train.dataloaders' \
  "${LIGHTVAE_ARGS[@]}"

# 正式训练（4 GPU）
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=12342 -m scripts.train \
  --config=wm4vla/configs/action_conditioned/config.py -- \
  experiment=ac_libero_lerobot_256_pixels_2b_task0 \
  job.wandb_mode=disabled \
  dataloader_train.batch_size=16 \
  '~dataloader_train.dataloaders' \
  "${LIGHTVAE_ARGS[@]}"
```

| 总 batch_size | iter/epoch（task0） | 说明 |
|---|---|---|
| 8（1卡×8） | ~1,406 | 冒烟用 |
| 32（4卡×8） | ~352 | 快速完整训练 |

checkpoint 路径（区别于全集）：
```
outputs/.../2b_libero_spatial_lerobot_256_skip_dynamics_dual_cam_task0/checkpoints/
```

若测试通过，直接切换到 `ac_libero_lerobot_256_pixels_2b` 即可训练全部 10 个任务，**无需修改其他代码**。

### 3.9 checkpoint 转换

```bash
# 全集训练（experiment=ac_libero_lerobot_256_pixels_2b）
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/\
2b_libero_spatial_lerobot_256_skip_dynamics_dual_cam/checkpoints
# 单任务（task0）
# CHECKPOINTS_DIR=.../2b_libero_spatial_lerobot_256_skip_dynamics_dual_cam_task0/checkpoints
# 两任务（task01）
# CHECKPOINTS_DIR=.../2b_libero_lerobot_256_skip_dynamics_dual_cam_task01/checkpoints


CHECKPOINTS_DIR=/home/jikangye/workspace/tmp/lbai/tmp/wm4vla-outputs/wm-cosmos-predict-outputs/reconstruct_20260511/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_pi_libero_256_skip_dynamics_dual_cam_10/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)

python scripts/convert_distcp_to_pt.py \
  $CHECKPOINTS_DIR/$CHECKPOINT_ITER/model \
  $CHECKPOINTS_DIR/$CHECKPOINT_ITER
```

生成：
- `model_ema_bf16.pt`：推理用（推荐）
- `model_ema_fp32.pt`：精度更高

---

## 4. 评估

### 4.1 离线指标评估（PSNR / SSIM）

```bash
python scripts/eval_world_model.py \
    --ckpt /home/jikangye/workspace/tmp/lbai/tmp/wm4vla-outputs/wm-cosmos-predict-outputs/reconstruct_20260511/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_pi_libero_256_skip_dynamics_dual_cam_10/checkpoints/iter_000010000/model_ema_bf16.pt \
    --output outputs/eval_wm/results.json \
    --tokenizer-backend lightvae \
    --tokenizer-vae-pth "${LIGHTVAE_PTH}" \
    --lightx2v-root "${LIGHTX2V_ROOT}" \
    --num-steps 35 \
    --samples-per-episode 20
```

协议：val split（seed=0, val_ratio=0.1），每 episode 20 个 start_t，d ∈ {1,2,3,4} 分组统计。

### 4.2 可视化

```bash
python scripts/visualize_wm.py \
    --ckpt /home/jikangye/workspace/tmp/lbai/tmp/wm4vla-outputs/wm-cosmos-predict-outputs/reconstruct_20260511/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_pi_libero_256_skip_dynamics_dual_cam_10/checkpoints/iter_000010000/model_ema_bf16.pt \
    --output outputs/wm_vis/260512 \
    --tokenizer-backend lightvae \
    --tokenizer-vae-pth "${LIGHTVAE_PTH}" \
    --lightx2v-root "${LIGHTX2V_ROOT}" \
    --n-episodes 3 \
    --delay 4 \
    --num-steps 35     # 快速预览；正式用 35
```

输出：每次推理生成两个原始 5 帧 mp4（`cam1` / `cam2`）。`--with-gt` 附加对应 GT 对比帧。
若训练时使用 LightVAE，评估和可视化必须使用同一套 tokenizer 参数。

---

## 5. wm4vla eval 集成

eval 时 WM 输出帧提取（强配对 batch，5 帧布局，state_t=2）：

```python
# video_out: [2, 3, 5, 256, 256], float [-1, 1]
cam1_pred = video_out[0, :, 1]    # batch 0, predicted cam1（pixel frame 1）
cam2_pred = video_out[1, :, 1]    # batch 1, predicted cam2（pixel frame 1）

obs["observation.images.image"]       = cam1_pred  # 256×256，直接传入
obs["observation.images.wrist_image"] = cam2_pred  # 无需 resize
```

---

## 6. 改动总结

| 文件 | 改动 |
|---|---|
| `wm4vla/datasets/dataset_lerobot_libero.py` | 强配对双视角 5 帧布局（state_t=2）；返回成对 batch 的 `video/action/delay_scalar/t5_text_embeddings` |
| `wm4vla/conditioning/action_sequence.py` | 打包固定长度 masked action prefix；生成归一化 `delay_scalar` |
| `cosmos_predict2/_src/predict2/action/configs/action_conditioned/conditioner.py` | `ActionConditionedConditionerConfig` dropout_rate=0.0（纯条件训练）；透传 `action` 和 `delay_scalar` |
| `wm4vla/configs/data_registry.py` | 注册 task0 / task01 / 全集三种数据加载器；支持 `LEROBOT_LIBERO_T5_EMB_PATH` |
| `cosmos_predict2/_src/predict2/action/networks/action_conditioned_minimal_v1_lvg_dit.py` | action prefix 通过 MLP mixer + masked pooling 编码，并注入 timestep / AdaLN 调制路径 |
| `wm4vla/configs/experiments.py` | paired-view LIBERO 使用 `state_t=2`、`num_conditional_frames=1`、`action_dim=8`、`num_action_per_chunk=8` |
| `cosmos_predict2/_src/predict2/action/models/action_conditioned_video2world_rectified_flow_model.py` | `guidance==0` 时跳过 uncond forward（推理约 2× 加速） |
| `scripts/precompute_libero_t5.py` | **新建**：`--t5-model` 支持本地目录直接加载，保存 `meta/t5_embeddings.pkl` |
| `scripts/eval_world_model.py` | 5 帧 paired-view 布局；按 delay 构造 `action` / `delay_scalar`；真实任务描述 prompt；预计算嵌入注入 |
| `wm4vla/scripts/visualize_wm.py` | 成对 5 帧视角视频可视化；输出 `cam1/cam2` 两个 mp4 |
