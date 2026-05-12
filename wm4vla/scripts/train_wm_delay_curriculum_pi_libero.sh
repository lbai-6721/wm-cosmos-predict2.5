#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG:=wm4vla/configs/action_conditioned/config.py}"
: "${PI_LIBERO_DATA_ROOT:=/mnt/storage/users/kyji_data/tmp/lbai/cosmos-predict2.5/physical-intelligence/libero}"
: "${PI_LIBERO_T5_EMB_PATH:=${PI_LIBERO_DATA_ROOT}/meta/t5_embeddings.pkl}"
: "${EXPERIMENT:=ac_pi_libero_256_pixels_2b_all}"
: "${MASTER_PORT:=12341}"
: "${NPROC_PER_NODE:=4}"
: "${CUDA_VISIBLE_DEVICES:=1,2,3,4}"
: "${BATCH_SIZE:=24}"
: "${CURRICULUM_DELAY_START:=2}"
: "${CURRICULUM_DELAY_END:=8}"
: "${STAGE_ITERS:=2000}"
: "${POST_CURRICULUM_MAX_ITER:=999999999}"

export PI_LIBERO_DATA_ROOT
export PI_LIBERO_T5_EMB_PATH

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

  echo "[curriculum][pi_libero] delay range [1,${delay_max}] -> trainer.max_iter=${target_iter}"
  echo "[curriculum][pi_libero] experiment=${EXPERIMENT}"
  echo "[curriculum][pi_libero] data_root=${PI_LIBERO_DATA_ROOT}"

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

final_curriculum_iter=$(((CURRICULUM_DELAY_END - CURRICULUM_DELAY_START + 1) * STAGE_ITERS))

if (( POST_CURRICULUM_MAX_ITER > final_curriculum_iter )); then
  echo "[curriculum][pi_libero] continue delay range [1,${CURRICULUM_DELAY_END}] -> trainer.max_iter=${POST_CURRICULUM_MAX_ITER}"
  echo "[curriculum][pi_libero] experiment=${EXPERIMENT}"
  echo "[curriculum][pi_libero] data_root=${PI_LIBERO_DATA_ROOT}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" -m scripts.train \
    --config="${CONFIG}" -- \
    experiment="${EXPERIMENT}" \
    dataloader_train.batch_size="${BATCH_SIZE}" \
    trainer.max_iter="${POST_CURRICULUM_MAX_ITER}" \
    "dataloader_train.dataset.sampled_delay_max=${CURRICULUM_DELAY_END}" \
    "dataloader_val.dataset.sampled_delay_max=${CURRICULUM_DELAY_END}" \
    '~dataloader_train.dataloaders' \
    "$@"
fi
