# 说明

本仓库仅用于蒸馏 student model，不用于训练 teacher model。

当前推荐路径分两类：

- `pi_libero`：新增主线，支持 `all / libero_10 / libero_goal / libero_object / libero_spatial`
- `lerobot task0`：保留旧单任务蒸馏入口，便于兼容已有 checkpoint 和评估脚本

## 激活环境

```bash
source /home/jikangye/workspace/tmp/lbai/wm-cosmos-predict2.5/.venv/bin/activate
```

或重新创建：

```bash
uv python install
uv sync --extra=cu128
source .venv/bin/activate

uv pip install pyarrow
uv pip install h5py
```

## 环境变量

```bash
export HF_ENDPOINT=https://hf-mirror.com

# 所有蒸馏输出的根目录
export IMAGINAIRE_OUTPUT_ROOT=/home/kyji/storage_net/tmp/lbai/tmp/wm4lva-output/wm-output/distill-output/wm-cosmos-predict2.5/20260416_20000_pi_libero_10

# pi_libero 主线路径
export PI_LIBERO_DATA_ROOT=/mnt/storage/users/kyji_data/tmp/lbai/cosmos-predict2.5/physical-intelligence/libero
export PI_LIBERO_T5_EMB_PATH=${PI_LIBERO_DATA_ROOT}/meta/t5_embeddings.pkl

# 对应 benchmark/all 的 teacher checkpoint
export WM4VLA_PI_LIBERO_TEACHER_CKPT_ALL=/path/to/ac_pi_libero_256_pixels_2b_all/model_ema_bf16.pt
export WM4VLA_PI_LIBERO_TEACHER_CKPT_10=/home/kyji/storage_net/tmp/lbai/cosmos-predict2.5/outputs/wm-output/benchmark/pi_libero_10/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_pi_libero_256_skip_dynamics_dual_cam_10/checkpoints/iter_000020000/model_ema_bf16.pt
export WM4VLA_PI_LIBERO_TEACHER_CKPT_GOAL=/path/to/ac_pi_libero_256_pixels_2b_goal/model_ema_bf16.pt
export WM4VLA_PI_LIBERO_TEACHER_CKPT_OBJECT=/path/to/ac_pi_libero_256_pixels_2b_object/model_ema_bf16.pt
export WM4VLA_PI_LIBERO_TEACHER_CKPT_SPATIAL=/path/to/ac_pi_libero_256_pixels_2b_spatial/model_ema_bf16.pt

# 兼容旧 task0 蒸馏
export LEROBOT_LIBERO_DATA_ROOT=/home/kyji/storage_net/tmp/lbai/cosmos-predict2.5/lerobot/lerobot--libero_10_image@v2.0
export LEROBOT_LIBERO_T5_EMB_PATH=${LEROBOT_LIBERO_DATA_ROOT}/meta/t5_embeddings.pkl
export WM4VLA_LIBERO_TASK0_TEACHER_CKPT=/path/to/ac_libero_lerobot_256_pixels_2b_task0/model_ema_bf16.pt
```

## Teacher Checkpoint 格式

蒸馏框架直接加载 `.pt` checkpoint（`model_ema_bf16.pt`）。

若 teacher 输出是 DCP 目录，先转换：

```bash
CHECKPOINT_DIR=/path/to/teacher/checkpoints/iter_XXXXXX

python scripts/convert_distcp_to_pt.py \
  ${CHECKPOINT_DIR}/model \
  ${CHECKPOINT_DIR}
```

## 蒸馏实验名

`pi_libero` 新增 5 个实验：

- `dmd2_trigflow_distill_wm_pi_libero_256_all`
- `dmd2_trigflow_distill_wm_pi_libero_256_10`
- `dmd2_trigflow_distill_wm_pi_libero_256_goal`
- `dmd2_trigflow_distill_wm_pi_libero_256_object`
- `dmd2_trigflow_distill_wm_pi_libero_256_spatial`

保留旧实验：

- `dmd2_trigflow_distill_wm_libero_lerobot_256_task0`
- `dmd2_trigflow_distill_wm_kinetix_128_9frame`

## 推荐启动方式

推荐直接使用统一脚本：

```bash
cd /home/kyji/storage_net/tmp/lbai/wm-cosmos-predict2.5

BENCHMARK=all bash wm4vla/scripts/train_distill_pi_libero.sh
BENCHMARK=10  bash wm4vla/scripts/train_distill_pi_libero.sh
BENCHMARK=goal bash wm4vla/scripts/train_distill_pi_libero.sh
BENCHMARK=object bash wm4vla/scripts/train_distill_pi_libero.sh
BENCHMARK=spatial bash wm4vla/scripts/train_distill_pi_libero.sh
```

常用覆盖项：

```bash
BENCHMARK=10 \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
MASTER_PORT=12340 \
BATCH_SIZE=2 \
bash wm4vla/scripts/train_distill_pi_libero.sh
```

如果不想依赖环境变量，也可以直接传 teacher checkpoint：

```bash
BENCHMARK=goal \
TEACHER_CKPT=/path/to/model_ema_bf16.pt \
bash wm4vla/scripts/train_distill_pi_libero.sh
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

旧 `task0` 兼容入口：

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

## 蒸馏输出与 Checkpoint 转换

蒸馏训练输出为 DCP 格式：

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

## 评估

评估 `pi_libero` benchmark：

```bash
export PI_LIBERO_DATA_ROOT=/mnt/storage/users/kyji_data/tmp/lbai/cosmos-predict2.5/physical-intelligence/libero
export PI_LIBERO_T5_EMB_PATH=${PI_LIBERO_DATA_ROOT}/meta/t5_embeddings.pkl

CUDA_VISIBLE_DEVICES=7 python wm4vla/scripts/eval_distilled_world_model.py \
  --ckpt /home/kyji/storage_net/tmp/lbai/tmp/wm4lva-output/wm-output/wm-output/distill-output/benchmark/pi_libero_10_delay8_only_20000/cosmos_interactive/cosmos3_interactive/dmd2_trigflow_distill_wm_pi_libero_256_10/checkpoints/iter_000001500/model_ema_bf16.pt \
  --benchmark libero_10 \
  --num-steps 1 \
  --t5-emb-path ${PI_LIBERO_T5_EMB_PATH} \
  --save-images outputs/eval_distill/pi_libero_10_delay8_only_wan_steps1_20000_1500_imgs \
  --output outputs/eval_distill/pi_libero_10_delay8_only_wan_steps1_20000_1500.json 
```

评估旧 `task0`：

```bash
CUDA_VISIBLE_DEVICES=0 python wm4vla/scripts/eval_distilled_world_model.py \
  --ckpt ${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}/model_ema_bf16.pt \
  --task-indices 0 \
  --num-steps 1 \
  --t5-emb-path ${LEROBOT_LIBERO_T5_EMB_PATH} \
  --save-images outputs/eval_distill/task0_imgs \
  --output outputs/eval_distill/task0_step1.json
```
