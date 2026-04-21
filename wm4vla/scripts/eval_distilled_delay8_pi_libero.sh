#!/usr/bin/env bash
set -euo pipefail

: "${PI_LIBERO_DATA_ROOT:=/mnt/storage/users/kyji_data/tmp/lbai/cosmos-predict2.5/physical-intelligence/libero}"
: "${PI_LIBERO_T5_EMB_PATH:=${PI_LIBERO_DATA_ROOT}/meta/t5_embeddings.pkl}"
: "${BENCHMARK:=10}"
: "${FIXED_DELAY:=8}"
: "${MODEL_MAX_DELAY:=8}"
: "${CUDA_VISIBLE_DEVICES:=1}"
: "${TOKENIZER_BACKEND:=wan2pt1}"
: "${TOKENIZER_VAE_PTH:=}"
: "${LIGHTVAE_PTH:=/home/kyji/public/models/lightx2v/vae/lightvaew2_1.pth}"
: "${LIGHTX2V_ROOT:=}"
: "${NUM_STEPS:=1}"
: "${SAMPLES_PER_EPISODE:=20}"
: "${SEED:=42}"
: "${OUTPUT:=}"
: "${SAVE_IMAGES:=}"
: "${EXPERIMENT:=}"
: "${CKPT:=}"
: "${TASK_INDICES:=}"

if [[ -z "${CKPT}" ]]; then
  echo "CKPT is required." >&2
  echo "Set CKPT=/path/to/model_ema_bf16.pt before launching evaluation." >&2
  exit 1
fi

if (( FIXED_DELAY < 1 )); then
  echo "FIXED_DELAY must be >= 1" >&2
  exit 1
fi

if (( FIXED_DELAY > MODEL_MAX_DELAY )); then
  echo "FIXED_DELAY must be <= MODEL_MAX_DELAY (${MODEL_MAX_DELAY})" >&2
  exit 1
fi

case "${BENCHMARK}" in
  all)
    EVAL_BENCHMARK="all"
    ;;
  10|libero_10)
    EVAL_BENCHMARK="libero_10"
    ;;
  goal|libero_goal)
    EVAL_BENCHMARK="libero_goal"
    ;;
  object|libero_object)
    EVAL_BENCHMARK="libero_object"
    ;;
  spatial|libero_spatial)
    EVAL_BENCHMARK="libero_spatial"
    ;;
  *)
    echo "Unsupported BENCHMARK=${BENCHMARK}" >&2
    echo "Expected one of: all, 10/libero_10, goal/libero_goal, object/libero_object, spatial/libero_spatial" >&2
    exit 1
    ;;
esac

case "${TOKENIZER_BACKEND}" in
  wan2pt1|lightvae)
    ;;
  *)
    echo "Unsupported TOKENIZER_BACKEND=${TOKENIZER_BACKEND}" >&2
    echo "Expected one of: wan2pt1, lightvae" >&2
    exit 1
    ;;
esac

read -r -a NUM_STEPS_ARR <<< "${NUM_STEPS}"

CMD=(
  python wm4vla/scripts/eval_distilled_world_model.py
  --ckpt "${CKPT}"
  --data-root "${PI_LIBERO_DATA_ROOT}"
  --dataset-format pi_libero
  --benchmark "${EVAL_BENCHMARK}"
  --t5-emb-path "${PI_LIBERO_T5_EMB_PATH}"
  --num-steps "${NUM_STEPS_ARR[@]}"
  --delays "${FIXED_DELAY}"
  --samples-per-episode "${SAMPLES_PER_EPISODE}"
  --seed "${SEED}"
  --tokenizer-backend "${TOKENIZER_BACKEND}"
)

if [[ -n "${TOKENIZER_VAE_PTH}" ]]; then
  CMD+=(--tokenizer-vae-pth "${TOKENIZER_VAE_PTH}")
elif [[ "${TOKENIZER_BACKEND}" == "lightvae" && -n "${LIGHTVAE_PTH}" ]]; then
  CMD+=(--lightvae-pth "${LIGHTVAE_PTH}")
fi

if [[ -n "${LIGHTX2V_ROOT}" ]]; then
  CMD+=(--lightx2v-root "${LIGHTX2V_ROOT}")
fi

if [[ -n "${OUTPUT}" ]]; then
  CMD+=(--output "${OUTPUT}")
fi

if [[ -n "${SAVE_IMAGES}" ]]; then
  CMD+=(--save-images "${SAVE_IMAGES}")
fi

if [[ -n "${EXPERIMENT}" ]]; then
  CMD+=(--experiment "${EXPERIMENT}")
fi

if [[ -n "${TASK_INDICES}" ]]; then
  read -r -a TASK_INDICES_ARR <<< "${TASK_INDICES}"
  CMD+=(--task-indices "${TASK_INDICES_ARR[@]}")
fi

export PI_LIBERO_DATA_ROOT
export PI_LIBERO_T5_EMB_PATH
export CUDA_VISIBLE_DEVICES

echo "[eval][distill][delay8][pi_libero] benchmark=${BENCHMARK}"
echo "[eval][distill][delay8][pi_libero] eval_benchmark=${EVAL_BENCHMARK}"
echo "[eval][distill][delay8][pi_libero] fixed_delay=${FIXED_DELAY}"
echo "[eval][distill][delay8][pi_libero] ckpt=${CKPT}"
echo "[eval][distill][delay8][pi_libero] data_root=${PI_LIBERO_DATA_ROOT}"
echo "[eval][distill][delay8][pi_libero] tokenizer_backend=${TOKENIZER_BACKEND}"
if [[ -n "${OUTPUT}" ]]; then
  echo "[eval][distill][delay8][pi_libero] output=${OUTPUT}"
fi
if [[ -n "${SAVE_IMAGES}" ]]; then
  echo "[eval][distill][delay8][pi_libero] save_images=${SAVE_IMAGES}"
fi

exec "${CMD[@]}" "$@"
