# WM4VLA 蒸馏训练指南

本文档说明如何对 wm4vla teacher 做 DMD2 蒸馏，以获得 1-4 步推理的 student world model。

当前支持三类数据：

- `lerobot task0`：旧单任务链路，保留兼容
- `pi_libero`：新增主线，支持 `all / libero_10 / libero_goal / libero_object / libero_spatial`
- `metaworld_mt50`：MetaWorld MT50 单视角链路，支持 `mt50 / task0`

## 蒸馏算法概述

DMD2 交替进行两个更新步骤：

- `Generator step`：student 用 teacher score 计算分布匹配损失
- `Critic step`：fake-score 网络区分真实样本和 student 生成样本

teacher 在蒸馏全过程中冻结。

## 环境变量

```bash
export HF_ENDPOINT=https://hf-mirror.com
export IMAGINAIRE_OUTPUT_ROOT=/mnt/cpfs/yangboxue/vla/wm4vla/data/distill/metaworld/delay8_14000

export PI_LIBERO_DATA_ROOT=/mnt/storage/users/kyji_data/tmp/lbai/cosmos-predict2.5/physical-intelligence/libero
export PI_LIBERO_T5_EMB_PATH=${PI_LIBERO_DATA_ROOT}/meta/t5_embeddings.pkl

export WM4VLA_PI_LIBERO_TEACHER_CKPT_ALL=/path/to/ac_pi_libero_256_pixels_2b_all/model_ema_bf16.pt
export WM4VLA_PI_LIBERO_TEACHER_CKPT_10=/home/kyji/storage_net/tmp/lbai/cosmos-predict2.5/outputs/wm-output/benchmark/pi_libero_10/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_pi_libero_256_skip_dynamics_dual_cam_10/checkpoints/iter_000014000/model_ema_bf16.pt
export WM4VLA_PI_LIBERO_TEACHER_CKPT_GOAL=/path/to/ac_pi_libero_256_pixels_2b_goal/model_ema_bf16.pt
export WM4VLA_PI_LIBERO_TEACHER_CKPT_OBJECT=/path/to/ac_pi_libero_256_pixels_2b_object/model_ema_bf16.pt
export WM4VLA_PI_LIBERO_TEACHER_CKPT_SPATIAL=/path/to/ac_pi_libero_256_pixels_2b_spatial/model_ema_bf16.pt

# 兼容旧 task0
export LEROBOT_LIBERO_DATA_ROOT=/home/kyji/storage_net/tmp/lbai/cosmos-predict2.5/lerobot/lerobot--libero_10_image@v2.0
export LEROBOT_LIBERO_T5_EMB_PATH=${LEROBOT_LIBERO_DATA_ROOT}/meta/t5_embeddings.pkl
export WM4VLA_LIBERO_TASK0_TEACHER_CKPT=/path/to/ac_libero_lerobot_256_pixels_2b_task0/model_ema_bf16.pt

# 可选：Kinetix
export WM4VLA_KINETIX_TEACHER_CKPT=/path/to/kinetix/model_ema_bf16.pt

# MetaWorld MT50
export METAWORLD_DATA_ROOT=/mnt/cpfs/yangboxue/vla/wm4vla/data/dataset/lerobot/metaworld_mt50
export METAWORLD_T5_EMB_PATH=${METAWORLD_DATA_ROOT}/meta/t5_embeddings.pkl
export WM4VLA_METAWORLD_MT50_TEACHER_CKPT=/mnt/cpfs/yangboxue/vla/wm4vla/data/cosmos-predict-output/metaworld_delay8/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_metaworld_mt50_256_skip_dynamics_single_cam/checkpoints/iter_000014000/model_ema_bf16.pt
# export WM4VLA_METAWORLD_TASK0_TEACHER_CKPT=/path/to/ac_metaworld_mt50_256_pixels_2b_task0/model_ema_bf16.pt
export LIGHTVAE_PTH=/mnt/cpfs/yangboxue/vla/wm4vla/LightX2V/save_results/wan21_lightvae_distill_metaworld/exports/lightvae_step_0013000.safetensors
export LIGHTX2V_ROOT=/mnt/cpfs/yangboxue/vla/wm4vla/LightX2V
```

## Teacher Checkpoint 格式

蒸馏框架直接使用 `.pt` 格式的 `model_ema_bf16.pt`。

若 teacher 只有 DCP 目录，先转换：

```bash
CHECKPOINT_DIR=/path/to/teacher/checkpoints/iter_XXXXXX

python scripts/convert_distcp_to_pt.py \
  ${CHECKPOINT_DIR}/model \
  ${CHECKPOINT_DIR}
```

## 关键配置约束

相对标准 Cosmos Distillation，wm4vla teacher 的约束必须保持一致：

| 配置项 | 要求 |
|---|---|
| `tokenizer` | LIBERO 固定为 `wan2pt1_tokenizer`；MetaWorld 默认为 `wan2pt1_lightvae_tokenizer`，需与 teacher 保持一致 |
| `multiply_noise_by_video_len` | 固定为 `False` |
| `use_clean_cond_timesteps` | 固定为 `False` |
| `teacher_guidance` | 固定为 `0` |
| `text_encoder_config` | 固定为 `None`，使用预计算 T5 |
| `action` 接口 | LIBERO 为 `action: [B, 8, 8]`，MetaWorld 为 `action: [B, 8, 5]` |
| `delay_scalar` 接口 | `delay_scalar: [B, 1]` |
| `num_action_per_chunk` | 固定为 `8` |
| `fsdp_shard_size` | 当前配置固定为 `4` |

旧版单步 action 语义不再是主线：

```text
action : [B, 1, 8]  # [a_{t+d}; normalized_delay]
```

当前主线必须使用：

```text
action       : [B, 8, slot_dim]  # LIBERO slot_dim=8, MetaWorld slot_dim=5
delay_scalar : [B, 1]
```

## 已注册实验

### 旧实验

- `dmd2_trigflow_distill_wm_libero_lerobot_256_task0`
- `dmd2_trigflow_distill_wm_kinetix_128_9frame`

### 新增 `pi_libero` 实验

| 实验名 | 数据集 |
|---|---|
| `dmd2_trigflow_distill_wm_pi_libero_256_all` | `pi_libero_all_256_train` |
| `dmd2_trigflow_distill_wm_pi_libero_256_10` | `pi_libero_10_256_train` |
| `dmd2_trigflow_distill_wm_pi_libero_256_goal` | `pi_libero_goal_256_train` |
| `dmd2_trigflow_distill_wm_pi_libero_256_object` | `pi_libero_object_256_train` |
| `dmd2_trigflow_distill_wm_pi_libero_256_spatial` | `pi_libero_spatial_256_train` |

### 新增 `metaworld_mt50` 实验

| 实验名 | 数据集 |
|---|---|
| `dmd2_trigflow_distill_wm_metaworld_mt50_256` | `metaworld_mt50_256_train` |
| `dmd2_trigflow_distill_wm_metaworld_mt50_256_task0` | `metaworld_mt50_256_task0_train` |

## 推荐启动方式

优先使用统一脚本：

```bash
cd /home/kyji/storage_net/tmp/lbai/wm-cosmos-predict2.5

BENCHMARK=all     bash wm4vla/scripts/train_distill_pi_libero.sh
BENCHMARK=10      bash wm4vla/scripts/train_distill_pi_libero.sh
BENCHMARK=goal    bash wm4vla/scripts/train_distill_pi_libero.sh
BENCHMARK=object  bash wm4vla/scripts/train_distill_pi_libero.sh
BENCHMARK=spatial bash wm4vla/scripts/train_distill_pi_libero.sh
```

MetaWorld：

```bash
cd /home/kyji/storage_net/tmp/lbai/wm-cosmos-predict2.5

BENCHMARK=mt50  bash wm4vla/scripts/train_distill_metaworld.sh
BENCHMARK=task0 bash wm4vla/scripts/train_distill_metaworld.sh
```

常用覆盖方式：

```bash
BENCHMARK=10 \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
MASTER_PORT=12340 \
BATCH_SIZE=2 \
bash wm4vla/scripts/train_distill_pi_libero.sh
```

也可以直接指定 teacher checkpoint：

```bash
BENCHMARK=goal \
TEACHER_CKPT=/path/to/model_ema_bf16.pt \
bash wm4vla/scripts/train_distill_pi_libero.sh
```

MetaWorld 也支持同样的覆盖：

```bash
BENCHMARK=mt50 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=4 \
MASTER_PORT=12341 \
BATCH_SIZE=16 \
TEACHER_CKPT=/path/to/model_ema_bf16.pt \
LIGHTVAE_PTH=/mnt/cpfs/yangboxue/vla/wm4vla/LightX2V/save_results/wan21_lightvae_distill_metaworld/exports/lightvae_step_0013000.safetensors \
LIGHTX2V_ROOT=/mnt/cpfs/yangboxue/vla/wm4vla/LightX2V \
bash wm4vla/scripts/train_distill_metaworld.sh
```

## 直接 torchrun 启动

```bash
cd /home/kyji/storage_net/tmp/lbai/wm-cosmos-predict2.5

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  --master_port=12340 \
  -m scripts.train \
  --config=cosmos_predict2/_src/interactive/configs/registry_predict2p5.py \
  -- \
  experiment=dmd2_trigflow_distill_wm_pi_libero_256_10 \
  model.config.teacher_load_from.load_path=${WM4VLA_PI_LIBERO_TEACHER_CKPT_10}
```

MetaWorld：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  --master_port=12341 \
  -m scripts.train \
  --config=cosmos_predict2/_src/interactive/configs/registry_predict2p5.py \
  -- \
  experiment=dmd2_trigflow_distill_wm_metaworld_mt50_256 \
  model.config.teacher_load_from.load_path=${WM4VLA_METAWORLD_MT50_TEACHER_CKPT} \
  tokenizer=wan2pt1_lightvae_tokenizer \
  +model.config.tokenizer.vae_pth=${LIGHTVAE_PTH} \
  +model.config.tokenizer.lightx2v_root=${LIGHTX2V_ROOT} \
  model.config.tokenizer.use_batched_vae=true
```

旧 `task0` 兼容命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 \
  --master_port=12340 \
  -m scripts.train \
  --config=cosmos_predict2/_src/interactive/configs/registry_predict2p5.py \
  -- \
  experiment=dmd2_trigflow_distill_wm_libero_lerobot_256_task0 \
  model.config.teacher_load_from.load_path=${WM4VLA_LIBERO_TASK0_TEACHER_CKPT}
```

## 输出与 Checkpoint 转换

蒸馏输出保存在：

```text
${IMAGINAIRE_OUTPUT_ROOT}/cosmos3_interactive/<experiment_name>/checkpoints/iter_XXXXXX/
```

转换为 `.pt`：

```bash
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT}/cosmos3_interactive/dmd2_trigflow_distill_wm_pi_libero_256_10/checkpoints
CHECKPOINT_ITER=$(cat ${CHECKPOINTS_DIR}/latest_checkpoint.txt)

python scripts/convert_distcp_to_pt.py \
  ${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}/model \
  ${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}
```

## 评估蒸馏效果

### 评估 `pi_libero`

```bash
CUDA_VISIBLE_DEVICES=0 python wm4vla/scripts/eval_distilled_world_model.py \
  --ckpt ${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}/model_ema_bf16.pt \
  --benchmark libero_10 \
  --num-steps 1 2 4 \
  --t5-emb-path ${PI_LIBERO_T5_EMB_PATH} \
  --output outputs/eval_distill/pi_libero_10_steps124.json
```

### 评估旧 `task0`

```bash
CUDA_VISIBLE_DEVICES=0 python wm4vla/scripts/eval_distilled_world_model.py \
  --ckpt ${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}/model_ema_bf16.pt \
  --task-indices 0 \
  --num-steps 1 \
  --t5-emb-path ${LEROBOT_LIBERO_T5_EMB_PATH} \
  --output outputs/eval_distill/task0_step1.json
```

### 保存对比图

```bash
CUDA_VISIBLE_DEVICES=0 python wm4vla/scripts/eval_distilled_world_model.py \
  --ckpt ${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}/model_ema_bf16.pt \
  --benchmark libero_spatial \
  --num-steps 1 \
  --save-images outputs/eval_distill/pi_libero_spatial_images \
  --output outputs/eval_distill/pi_libero_spatial_step1.json
```

### 评估 `metaworld_mt50`（LightVAE 单视角）

MetaWorld 使用单视角 `observation.image`，蒸馏训练默认使用 `wan2pt1_lightvae_tokenizer`。评估时使用单独脚本：

```bash
CUDA_VISIBLE_DEVICES=0 python wm4vla/scripts/eval_distilled_metaworld.py \
  --ckpt ${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}/model_ema_bf16.pt \
  --benchmark mt50 \
  --num-steps 1 2 4 \
  --t5-emb-path ${METAWORLD_T5_EMB_PATH} \
  --lightvae-pth ${LIGHTVAE_PTH} \
  --lightx2v-root ${LIGHTX2V_ROOT} \
  --output outputs/eval_distill/metaworld_mt50_steps124.json
```

只评估 task0：

```bash
CUDA_VISIBLE_DEVICES=0 python wm4vla/scripts/eval_distilled_metaworld.py \
  --ckpt ${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}/model_ema_bf16.pt \
  --benchmark task0 \
  --num-steps 1 \
  --t5-emb-path ${METAWORLD_T5_EMB_PATH} \
  --lightvae-pth ${LIGHTVAE_PTH} \
  --lightx2v-root ${LIGHTX2V_ROOT} \
  --output outputs/eval_distill/metaworld_task0_step1.json
```

保存单视角预测/GT 对比图：

```bash
CUDA_VISIBLE_DEVICES=0 python wm4vla/scripts/eval_distilled_metaworld.py \
  --ckpt /mnt/cpfs/yangboxue/vla/wm4vla/data/distill/metaworld/delay8_14000/cosmos_interactive/cosmos3_interactive/dmd2_trigflow_distill_wm_metaworld_mt50_256/checkpoints/iter_000007500/model_ema_bf16.pt \
  --benchmark mt50 \
  --num-steps 1 \
  --save-images /mnt/cpfs/yangboxue/vla/wm4vla/data/eval_wm/eval_distill/metaworld_images \
  --output /mnt/cpfs/yangboxue/vla/wm4vla/data/eval_wm/eval_distill/metaworld__step1.json
```
