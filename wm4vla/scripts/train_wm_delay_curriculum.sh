#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG:=wm4vla/configs/action_conditioned/config.py}"
: "${EXPERIMENT:=ac_pi_libero_256_pixels_2b_10}"
: "${MASTER_PORT:=12341}"
: "${NPROC_PER_NODE:=4}"
: "${CUDA_VISIBLE_DEVICES:=4,5,6,7}"
: "${BATCH_SIZE:=24}"
: "${CURRICULUM_DELAY_START:=2}"
: "${CURRICULUM_DELAY_END:=8}"
: "${STAGE_ITERS:=2000}"

if (( CURRICULUM_DELAY_START < 1 )); then
  echo "CURRICULUM_DELAY_START must be >= 1" >&2
  exit 1
fi

if (( CURRICULUM_DELAY_END < CURRICULUM_DELAY_START )); then
  echo "CURRICULUM_DELAY_END must be >= CURRICULUM_DELAY_START" >&2
  exit 1
fi

for delay_max in $(seq "${CURRICULUM_DELAY_START}" "${CURRICULUM_DELAY_END}"); do
  stage_index=$((delay_max - CURRICULUM_DELAY_START + 1))
  target_iter=$((stage_index * STAGE_ITERS))

  echo "[curriculum] delay range [1,${delay_max}] -> trainer.max_iter=${target_iter}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" -m scripts.train \
    --config="${CONFIG}" -- \
    experiment="${EXPERIMENT}" \
    dataloader_train.batch_size="${BATCH_SIZE}" \
    trainer.max_iter="${target_iter}" \
    "dataloader_train.dataset.sampled_delay_max=${delay_max}" \
    "dataloader_val.dataset.sampled_delay_max=${delay_max}" \
    '~dataloader_train.dataloaders' \
    "$@"
done
