#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG:=cosmos_predict2/_src/interactive/configs/registry_predict2p5.py}"
: "${METAWORLD_DATA_ROOT:=/mnt/cpfs/yangboxue/vla/wm4vla/data/dataset/lerobot/metaworld_mt50}"
: "${METAWORLD_T5_EMB_PATH:=${METAWORLD_DATA_ROOT}/meta/t5_embeddings.pkl}"
: "${BENCHMARK:=mt50}"
: "${MASTER_PORT:=12344}"
: "${NPROC_PER_NODE:=4}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
: "${BATCH_SIZE:=16}"
: "${TOKENIZER_BACKEND:=lightvae}"
: "${LIGHTVAE_PTH:=/mnt/cpfs/yangboxue/vla/wm4vla/LightX2V/save_results/wan21_lightvae_distill_metaworld/exports/lightvae_step_0013000.safetensors}"
: "${LIGHTX2V_ROOT:=/mnt/cpfs/yangboxue/vla/wm4vla/LightX2V}"
: "${USE_BATCHED_VAE:=true}"
: "${WAN21_VAE_PTH:=hf://nvidia/Cosmos-Predict2.5-2B/tokenizer.pth}"

case "${BENCHMARK}" in
  mt50|all)
    DEFAULT_EXPERIMENT="dmd2_trigflow_distill_wm_metaworld_mt50_256"
    TEACHER_ENV_VAR="WM4VLA_METAWORLD_MT50_TEACHER_CKPT"
    ;;
  task0)
    DEFAULT_EXPERIMENT="dmd2_trigflow_distill_wm_metaworld_mt50_256_task0"
    TEACHER_ENV_VAR="WM4VLA_METAWORLD_TASK0_TEACHER_CKPT"
    ;;
  *)
    echo "Unsupported BENCHMARK=${BENCHMARK}" >&2
    echo "Expected one of: mt50/all, task0" >&2
    exit 1
    ;;
esac

: "${EXPERIMENT:=${DEFAULT_EXPERIMENT}}"
TEACHER_CKPT="${TEACHER_CKPT:-${!TEACHER_ENV_VAR:-}}"

if [[ -z "${TEACHER_CKPT}" ]]; then
  echo "Teacher checkpoint is required." >&2
  echo "Set TEACHER_CKPT or ${TEACHER_ENV_VAR} before launching distillation." >&2
  exit 1
fi

export METAWORLD_DATA_ROOT
export METAWORLD_T5_EMB_PATH

if [[ "${TOKENIZER_BACKEND}" != "lightvae" && "${TOKENIZER_BACKEND}" != "wan2pt1" ]]; then
  echo "TOKENIZER_BACKEND must be one of: lightvae, wan2pt1" >&2
  exit 1
fi

TOKENIZER_ARGS=()
if [[ "${TOKENIZER_BACKEND}" == "lightvae" ]]; then
  TOKENIZER_ARGS+=(
    "tokenizer=wan2pt1_lightvae_tokenizer"
    "+model.config.tokenizer.vae_pth=${LIGHTVAE_PTH}"
    "+model.config.tokenizer.lightx2v_root=${LIGHTX2V_ROOT}"
    "model.config.tokenizer.use_batched_vae=${USE_BATCHED_VAE}"
  )
else
  TOKENIZER_ARGS+=(
    "tokenizer=wan2pt1_tokenizer"
    "+model.config.tokenizer.vae_pth=${WAN21_VAE_PTH}"
  )
fi

echo "[distill][metaworld] benchmark=${BENCHMARK}"
echo "[distill][metaworld] experiment=${EXPERIMENT}"
echo "[distill][metaworld] data_root=${METAWORLD_DATA_ROOT}"
echo "[distill][metaworld] t5_emb_path=${METAWORLD_T5_EMB_PATH}"
echo "[distill][metaworld] teacher_ckpt=${TEACHER_CKPT}"
echo "[distill][metaworld] tokenizer_backend=${TOKENIZER_BACKEND}"
if [[ "${TOKENIZER_BACKEND}" == "lightvae" ]]; then
  echo "[distill][metaworld] lightvae_pth=${LIGHTVAE_PTH}"
  echo "[distill][metaworld] lightx2v_root=${LIGHTX2V_ROOT}"
  echo "[distill][metaworld] use_batched_vae=${USE_BATCHED_VAE}"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
  -m scripts.train \
  --config="${CONFIG}" -- \
  experiment="${EXPERIMENT}" \
  dataloader_train.batch_size="${BATCH_SIZE}" \
  "model.config.teacher_load_from.load_path=${TEACHER_CKPT}" \
  "${TOKENIZER_ARGS[@]}" \
  "$@"
