# 说明
本仓库仅用于蒸馏学生模型，不用于训练教师模型，详细参考说明文档 wm-cosmos-predict2.5/wm4vla/doc/distillation.md

## 激活环境

与训练教师模型环境一致
```
source /home/jikangye/workspace/tmp/lbai/cosmos-predict2.5/.venv/bin/activate
```

## 环境变量

```
export HF_ENDPOINT=https://hf-mirror.com

# 所有训练输出的根目录
export IMAGINAIRE_OUTPUT_ROOT=/home/kyji/storage_net/tmp/lbai/tmp/wm4lva-output/wm-output/wm-output/distill-output/wm-cosmos-predcit2.5/distill_libero_10_task0_test

# LIBERO 数据路径（根据实验设置数据集路径）
export LEROBOT_LIBERO_DATA_ROOT=/home/kyji/storage_net/tmp/lbai/cosmos-predict2.5/lerobot/lerobot--libero_10_image@v2.0
export LEROBOT_LIBERO_T5_EMB_PATH=${LEROBOT_LIBERO_DATA_ROOT}/meta/t5_embeddings.pkl
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

## 修改teacher checkpoint路径
LIBERO 配置在 cosmos_predict2/_src/interactive/configs/registry_experiment/experiments_dmd2_ac_predict2p5.py:247 

## 蒸馏
```
**LIBERO task0（4 GPU）：**

```bash
cd wm-cosmos-predict2.5

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 \
  --master_port=12340 \
  -m scripts.train \
  --config=cosmos_predict2/_src/interactive/configs/registry_predict2p5.py \
  -- \
  experiment=dmd2_trigflow_distill_wm_libero_lerobot_256_task0  \
  model.config.teacher_load_from.load_path=/home/kyji/storage_net/tmp/lbai/cosmos-predict2.5/outputs/wm-output/wm4vla-action-sequence-temporal-mlp-pool-6gpu-124567-20260407/checkpoint/model_ema_bf16.pt
```
## 蒸馏输出与 Checkpoint 转换

蒸馏训练输出为 DCP 格式，保存在：

```
${IMAGINAIRE_OUTPUT_ROOT}/cosmos3_interactive/<experiment_name>/checkpoints/iter_XXXXXX/
```

转换为 `.pt` 供推理使用：

```bash
CHECKPOINTS_DIR=/home/kyji/storage_net/tmp/lbai/cosmos-predict2.5-new/output/wm-distill-output/distill_v3_light/cosmos_interactive/cosmos3_interactive/dmd2_trigflow_distill_wm_libero_lerobot_256_task0/checkpoints
CHECKPOINT_ITER=$(cat ${CHECKPOINTS_DIR}/latest_checkpoint.txt)

python scripts/convert_distcp_to_pt.py \
  ${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}/model \
  ${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}
# 输出：${CHECKPOINTS_DIR}/${CHECKPOINT_ITER}/model_ema_bf16.pt
```

## 使用默认wan2pt1 vae评估
```
CUDA_VISIBLE_DEVICES=0 python wm4vla/scripts/eval_distilled_world_model.py \
    --ckpt /home/kyji/storage_net/tmp/lbai/cosmos-predict2.5/outputs/wm-output/wm-distill-output/distill_final_mlp_20260410_afternoon/cosmos_interactive/cosmos3_interactive/dmd2_trigflow_distill_wm_libero_lerobot_256_task0/checkpoints/iter_000002000/model_ema_bf16.pt \
    --task-indices 0 \
    --num-steps 1 \
    --t5-emb-path ${LEROBOT_LIBERO_T5_EMB_PATH} \
    --save-images outputs/eval_distill/images_20260413_afternoon \
    --output outputs/eval_distill/20260413_afternoon.json
```

## 使用lightvae评估
先在当前仓库上一级clone lightvae仓库
```
git clone https://github.com/ModelTC/LightX2V.git
```

```
CUDA_VISIBLE_DEVICES=4 python wm4vla/scripts/eval_world_model.py \
  --ckpt /home/kyji/storage_net/tmp/lbai/cosmos-predict2.5/outputs/wm-output/wm-distill-output/distill_final_mlp_20260410_afternoon/cosmos_interactive/cosmos3_interactive/dmd2_trigflow_distill_wm_libero_lerobot_256_task0/checkpoints/iter_000002000/model_ema_bf16.pt \
  --tokenizer-backend lightvae \
  --tokenizer-vae-pth /home/kyji/public/models/lightx2v/vae/lightvaew2_1.pth \
  --task-indices 0 \
  --num-steps 1 \
  --t5-emb-path ${LEROBOT_LIBERO_T5_EMB_PATH} \
  --lightx2v-root /home/kyji/storage_net/tmp/lbai/LightX2V \
  --save-images outputs/eval_distill/task0_lightvae_20260413 \
  --output outputs/eval_wm/task0_lightvae_20260413.json
```