#!/usr/bin/env bash
# export IMAGINAIRE_OUTPUT_ROOT=/home/kyji/storage_net/tmp/lbai/tmp/wm4lva-output/wm-output/wm-output/cosmos-predict-output/delay8_only_train_with_lightvae
# bash wm4vla/scripts/train_wm_delay8_pi_libero.sh
set -euo pipefail

: "${CONFIG:=wm4vla/configs/action_conditioned/config.py}"
: "${PI_LIBERO_DATA_ROOT:=/mnt/storage/users/kyji_data/tmp/lbai/cosmos-predict2.5/physical-intelligence/libero}"
: "${PI_LIBERO_T5_EMB_PATH:=${PI_LIBERO_DATA_ROOT}/meta/t5_embeddings.pkl}"
: "${EXPERIMENT:=ac_pi_libero_256_pixels_2b_10}"
: "${MASTER_PORT:=12341}"
: "${NPROC_PER_NODE:=4}"
: "${CUDA_VISIBLE_DEVICES:=1,2,3,4}"
: "${BATCH_SIZE:=24}"
: "${FIXED_DELAY:=8}"
: "${MODEL_MAX_DELAY:=8}"
: "${MAX_ITER:=50000}"
: "${TOKENIZER_BACKEND:=lightvae}"
: "${LIGHTVAE_PTH:=/home/kyji/public/models/lightx2v/vae/lightvaew2_1.pth}"
: "${LIGHTX2V_ROOT:=/home/kyji/storage_net/tmp/lbai/LightX2V}"
: "${USE_BATCHED_VAE:=true}"

export PI_LIBERO_DATA_ROOT
export PI_LIBERO_T5_EMB_PATH

if (( FIXED_DELAY < 1 )); then
  echo "FIXED_DELAY must be >= 1" >&2
  exit 1
fi

if (( FIXED_DELAY > MODEL_MAX_DELAY )); then
  echo "FIXED_DELAY must be <= MODEL_MAX_DELAY (${MODEL_MAX_DELAY})" >&2
  exit 1
fi

if [[ "${TOKENIZER_BACKEND}" != "lightvae" && "${TOKENIZER_BACKEND}" != "wan2pt1" ]]; then
  echo "TOKENIZER_BACKEND must be one of: lightvae, wan2pt1" >&2
  exit 1
fi

TOKENIZER_ARGS=()
if [[ "${TOKENIZER_BACKEND}" == "lightvae" ]]; then
  TOKENIZER_ARGS+=(
    "tokenizer=wan2pt1_lightvae_tokenizer"
    "model.config.tokenizer.vae_pth=${LIGHTVAE_PTH}"
    "model.config.tokenizer.lightx2v_root=${LIGHTX2V_ROOT}"
    "model.config.tokenizer.use_batched_vae=${USE_BATCHED_VAE}"
  )
fi

echo "[delay8][pi_libero] experiment=${EXPERIMENT}"
echo "[delay8][pi_libero] data_root=${PI_LIBERO_DATA_ROOT}"
echo "[delay8][pi_libero] fixed_delay=${FIXED_DELAY}"
echo "[delay8][pi_libero] trainer.max_iter=${MAX_ITER}"
echo "[delay8][pi_libero] config=${CONFIG}"
echo "[delay8][pi_libero] tokenizer_backend=${TOKENIZER_BACKEND}"
if [[ "${TOKENIZER_BACKEND}" == "lightvae" ]]; then
  echo "[delay8][pi_libero] lightvae_pth=${LIGHTVAE_PTH}"
  echo "[delay8][pi_libero] lightx2v_root=${LIGHTX2V_ROOT}"
  echo "[delay8][pi_libero] use_batched_vae=${USE_BATCHED_VAE}"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" -m scripts.train \
  --config="${CONFIG}" -- \
  experiment="${EXPERIMENT}" \
  dataloader_train.batch_size="${BATCH_SIZE}" \
  trainer.max_iter="${MAX_ITER}" \
  "dataloader_train.dataset.fixed_delay=${FIXED_DELAY}" \
  "dataloader_val.dataset.fixed_delay=${FIXED_DELAY}" \
  "dataloader_train.dataset.sampled_delay_max=${FIXED_DELAY}" \
  "dataloader_val.dataset.sampled_delay_max=${FIXED_DELAY}" \
  '~dataloader_train.dataloaders' \
  "${TOKENIZER_ARGS[@]}" \
  "$@"
