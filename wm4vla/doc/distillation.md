# WM4VLA 蒸馏训练指南

本文档说明如何对 wm4vla 训练的 teacher 模型进行 DMD2（Distribution Matching Distillation 2）蒸馏，以加速推理速度（目标：从 50 步缩减到 1-4 步），适配 VLA 策略闭环控制的异步推理场景。

## 蒸馏算法概述

DMD2 交替进行两个更新步骤：

- **Generator step（Student 更新）**：Student 网络用 Teacher 的 score function 计算分布匹配损失，学习 Teacher 的生成分布。
- **Critic step（Fake Score 更新）**：Fake Score 网络（结构与 Student 相同）充当判别器，区分真实数据分布与 Student 生成的假样本分布。

Teacher 网络在整个蒸馏过程中保持冻结。

## 环境要求

与 teacher 训练相同，见 [train_wm_pixels.md](./train_wm_pixels.md)。

额外需要的环境变量（根据任务添加）：

```bash
export HF_ENDPOINT=https://hf-mirror.com

# 所有训练输出的根目录
export IMAGINAIRE_OUTPUT_ROOT=/home/kyji/storage_net/tmp/lbai/wm-cosmos-predict2.5/wm-output/output/wm-distill-output/distill_final_mlp_20260409

# LIBERO 数据路径（若做 LIBERO 蒸馏）
export LEROBOT_LIBERO_DATA_ROOT=/home/kyji/storage_net/tmp/lbai/cosmos-predict2.5/lerobot/lerobot--libero_10_image@v2.0
export LEROBOT_LIBERO_T5_EMB_PATH=${LEROBOT_LIBERO_DATA_ROOT}/meta/t5_embeddings.pkl

# Kinetix 数据路径（若做 Kinetix 蒸馏）
export KINETIX_DATA_PIXELS_DIR=/path/to/kinetix/data_pixels
```

## Teacher Checkpoint 格式

蒸馏框架支持直接加载 `.pt` 格式 checkpoint（`model_ema_bf16.pt`）。

若 teacher 训练输出为 DCP 目录，先运行转换脚本：

```bash
CHECKPOINT_DIR=/path/to/teacher/checkpoints/iter_XXXXXX

python scripts/convert_distcp_to_pt.py \
  ${CHECKPOINT_DIR}/model \
  ${CHECKPOINT_DIR}
# 输出：${CHECKPOINT_DIR}/model_ema_bf16.pt
```

## 关键配置差异说明

相比标准 Cosmos Predict2.5 蒸馏，wm4vla teacher 训练方式有以下差异，蒸馏时必须对应设置：

| 配置项 | 原因 | 必须设置 |
|--------|------|---------|
| `tokenizer="wan2pt1_tokenizer"` | wm4vla teacher 训练用 wan2pt1 VAE（DEFAULT_CHECKPOINT 中会将 wan2pt2 覆盖为 wan2pt1） | ✅ 必须 override |
| `multiply_noise_by_video_len=False` | wm4vla 训练 `adjust_video_noise=False`，噪声乘数为 1.0 | ✅ 必须 override |
| `use_clean_cond_timesteps` | 取决于 teacher 训练时的 `conditional_frame_timestep`：`-1.0` → `False`；`0.1`（wm-output 原始 teacher）→ `True` | ✅ 必须与 teacher 匹配 |
| `teacher_guidance=0` | Teacher 纯条件训练，无 CFG，节省显存 | ✅ 必须 |
| `text_encoder_config=None` | 使用数据中预计算的 T5 embedding | ✅ 必须 |
| `action` / `delay_scalar` 接口 | 当前 teacher 使用 masked action prefix `action:[B,8,8]` + 独立 `delay_scalar:[B,1]`，不再把 delay 拼进单个 action 向量 | ✅ 必须匹配 |
| `num_action_per_chunk=8` | Teacher 的 action encoder 现在消费固定长度 8 槽 prefix，而不是 `T=1` 的单步 action | ✅ LIBERO 必须 |
| `fsdp_shard_size=4` | 4-GPU 单节点 | ✅ 必须 |
| 采样器按整批更新 latent | paired 方案下 `B=2` 表示两个视角；若只更新 `latents[0]`，会导致 `cam2` 输出异常 | ✅ 必须 |

### B=2 paired 采样修复

paired 5 帧方案下，蒸馏评估/推理同样依赖底层采样器对整个 batch 做 step 更新。  
`wm-cosmos-predict2.5` 已修复以下文件中的单样本采样路径（`latents[0]`）：

- `cosmos_predict2/_src/predict2/models/text2world_model.py`
- `cosmos_predict2/_src/predict2/models/text2world_wan2pt1_model.py`
- `cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py`
- `cosmos_predict2/_src/predict2/models/interpolator_model_rectified_flow.py`

修复后统一为 batch 级 step 更新，保证 `video_out[0]` 与 `video_out[1]` 分别对应 `cam1/cam2` 的独立预测语义。

## 蒸馏实验配置

实验配置在 `cosmos_predict2/_src/interactive/configs/registry_experiment/experiments_dmd2_ac_predict2p5.py`。

目前已注册以下实验：

### LIBERO task0（256×256，paired 5 帧，state_t=2）

```
实验名：dmd2_trigflow_distill_wm_libero_lerobot_256_task0
Teacher：iter_000008000/model_ema_bf16.pt（task0 任务）
数据：lerobot_libero_dual_cam_256_task0_train
条件帧：1 帧（每个 paired 视角样本各自的 view_t）
action_dim：8（7-dim raw action + valid-mask）
num_action_per_chunk：8
delay_scalar：单独输入，不再拼入 action 最后一维
```

### Kinetix（128×128，9 帧，state_t=3）

```
实验名：dmd2_trigflow_distill_wm_kinetix_128_9frame
Teacher：/path/to/kinetix/model_ema_bf16.pt（需填入实际路径）
数据：kinetix_5frame_128_train
条件帧：2 帧（blank + obs_t）
action_dim：7
```

## 启动蒸馏训练

入口配置文件：`cosmos_predict2/_src/interactive/configs/registry_predict2p5.py`

注意：当前蒸馏侧必须与 teacher 训练接口完全一致，即 dataset / conditioner / student / teacher / fake-score 都要走：

```text
action       : [B, 8, 8]   # masked action prefix
delay_scalar : [B, 1]
```

不能再使用旧版：

```text
action : [B, 1, 8]  # [a_{t+d} ; normalized_delay]
```

**LIBERO task0（4 GPU）：**

```bash
cd /home/kyji/storage_net/tmp/lbai/wm-cosmos-predict2.5

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  --master_port=12340 \
  -m scripts.train \
  --config=cosmos_predict2/_src/interactive/configs/registry_predict2p5.py \
  -- \
  experiment=dmd2_trigflow_distill_wm_libero_lerobot_256_task0 \
  dataloader_train.batch_size=4
```

**Kinetix（4 GPU）：**

```bash
cd /home/kyji/storage_net/tmp/lbai/wm-cosmos-predict2.5

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  --master_port=12342 \
  -m scripts.train \
  --config=cosmos_predict2/_src/interactive/configs/registry_predict2p5.py \
  -- \
  experiment=dmd2_trigflow_distill_wm_kinetix_128_9frame \
  job.wandb_mode=disabled
```

## 蒸馏输出与 Checkpoint 转换

蒸馏训练输出为 DCP 格式，保存在：

```
${IMAGINAIRE_OUTPUT_ROOT}/cosmos3_interactive/<experiment_name>/checkpoints/iter_XXXXXX/
```

转换为 `.pt` 供推理使用：

```bash
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT}/cosmos3_interactive/dmd2_trigflow_distill_wm_libero_lerobot_256_task0/checkpoints

CHECKPOINTS_DIR=/home/kyji/storage_net/tmp/lbai/cosmos-predict2.5-new/output/wm-distill-output/distill_v3_light/cosmos_interactive/cosmos3_interactive/dmd2_trigflow_distill_wm_libero_lerobot_256_task0/checkpoints
CHECKPOINT_ITER=iter_000011000
#CHECKPOINT_ITER=$(cat ${CHECKPOINTS_DIR}/latest_checkpoint.txt)

python scripts/convert_distcp_to_pt.py \
  ${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}/model \
  ${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}
# 输出：${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}/model_ema_bf16.pt
```

## 评估蒸馏效果

### 第一步：转换蒸馏 Checkpoint

蒸馏训练完成后，先将 DCP 目录转换为 `.pt` 文件：

```bash
CHECKPOINTS_DIR=/home/kyji/storage_net/tmp/lbai/cosmos-predict2.5-new/output/wm-distill-output/distill_v3_op/cosmos_interactive/cosmos3_interactive/dmd2_trigflow_distill_wm_libero_lerobot_256_task0/checkpoints
CHECKPOINT_ITER=$(cat ${CHECKPOINTS_DIR}/latest_checkpoint.txt)

CUDA_VISIBLE_DEVICES=7 python scripts/convert_distcp_to_pt.py \
  ${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}/model \
  ${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}
# 输出：${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}/model_ema_bf16.pt
```

### 第二步：运行评估

评估脚本：`wm4vla/scripts/eval_distilled_world_model.py`

**快速验证（单步，1 GPU）：**

```bash
cd /home/kyji/storage_net/tmp/lbai/wm-cosmos-predict2.5

CUDA_VISIBLE_DEVICES=7 python wm4vla/scripts/eval_distilled_world_model.py \
    --ckpt ${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}/model_ema_bf16.pt \
    --task-indices 0 \
    --num-steps 1 \
    --t5-emb-path ${LEROBOT_LIBERO_T5_EMB_PATH} \
    --samples-per-episode 5 \
    --output outputs/eval_distill/task0_step1_quick.json
```

**完整评估（1、2、4 步各一组，对比质量-速度曲线）：**

```bash
CUDA_VISIBLE_DEVICES=0 python wm4vla/scripts/eval_distilled_world_model.py \
    --ckpt ${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}/model_ema_bf16.pt \
    --task-indices 0 \
    --num-steps 1 2 4 \
    --t5-emb-path ${LEROBOT_LIBERO_T5_EMB_PATH} \
    --output outputs/eval_distill/task0_steps124.json
```

**保存预测 vs GT 对比图（仅第一个步数）：**

```bash
CUDA_VISIBLE_DEVICES=0 python wm4vla/scripts/eval_distilled_world_model.py \
    --ckpt ${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}/model_ema_bf16.pt \
    --task-indices 0 \
    --num-steps 1 \
    --t5-emb-path ${LEROBOT_LIBERO_T5_EMB_PATH} \
    --save-images outputs/eval_distill/images_step1_nn \
    --output outputs/eval_distill/task0_step1_nn.json

CUDA_VISIBLE_DEVICES=5 python wm4vla/scripts/eval_distilled_world_model.py \
    --ckpt /home/kyji/storage_net/tmp/lbai/cosmos-predict2.5-new/output/wm-distill-output/distill_v3_light/cosmos_interactive/cosmos3_interactive/dmd2_trigflow_distill_wm_libero_lerobot_256_task0/checkpoints/iter_000011000/model_ema_bf16.pt \
    --task-indices 0 \
    --num-steps 1 \
    --t5-emb-path ${LEROBOT_LIBERO_T5_EMB_PATH} \
    --save-images outputs/eval_distill/images_step1_light_test_env \
    --output outputs/eval_distill/task0_step1_light_test_env.json
```
  
### 输出结果格式

JSON 结果按 `steps=N` → `d=D` 两级索引：

```json
{
  "meta": {
    "ckpt": "/path/to/model_ema_bf16.pt",
    "num_steps_evaluated": [1, 2, 4]
  },
  "results": {
    "steps=1": {
      "d=1": {"n": 120, "cam1_psnr": 22.1, "cam2_psnr": 21.8, "avg_psnr": 21.9,
               "avg_ssim": 0.72, "avg_inference_ms": 180.0},
      "d=2": {...},
      "d=3": {...},
      "d=4": {...}
    },
    "steps=2": {...},
    "steps=4": {...}
  }
}
```

## 调整提示

- **batch_size**：当前设置 LIBERO=2，Kinetix=4。如显存不足可降低；如显存充裕可适当调高。
- **student_update_freq=5**：每 5 步更新一次 Critic，其余步骤更新 Student。可根据训练稳定性调整。
- **训练步数**：建议先跑 1000~5000 步观察生成质量，再决定是否继续训练。
- **LIBERO T5 embedding**：确保 `LEROBOT_LIBERO_T5_EMB_PATH` 指向已预计算的 `t5_embeddings.pkl`，否则无法加载数据。
