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
export IMAGINAIRE_OUTPUT_ROOT=/home/kyji/storage_net/tmp/lbai/wm-cosmos-predict2.5/wm-output/output/wm-distill-output/distill_v3_light

# LIBERO 数据路径（若做 LIBERO 蒸馏）
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

## 蒸馏
```
**LIBERO task0（4 GPU）：**

```bash
cd wm-cosmos-predict2.5

CUDA_VISIBLE_DEVICES=0,2,3,4 torchrun \
  --nproc_per_node=4 \
  --master_port=12340 \
  -m scripts.train \
  --config=cosmos_predict2/_src/interactive/configs/registry_predict2p5.py \
  -- experiment=dmd2_trigflow_distill_wm_libero_lerobot_256_task0 \
  job.wandb_mode=disabled
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

## 评估wm
```
CUDA_VISIBLE_DEVICES=5 python wm4vla/scripts/eval_distilled_world_model.py \
    --ckpt /home/kyji/storage_net/tmp/lbai/cosmos-predict2.5-new/output/wm-distill-output/distill_v3_light/cosmos_interactive/cosmos3_interactive/dmd2_trigflow_distill_wm_libero_lerobot_256_task0/checkpoints/iter_000011000/model_ema_bf16.pt \
    --task-indices 0 \
    --num-steps 1 \
    --t5-emb-path ${LEROBOT_LIBERO_T5_EMB_PATH} \
    --save-images outputs/eval_distill/images_step1_light_test_env \
    --output outputs/eval_distill/task0_step1_light_test_env.json
```