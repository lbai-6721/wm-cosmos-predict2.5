#!/usr/bin/env bash
# export CKPT=/home/kyji/storage_net/tmp/lbai/tmp/wm4lva-output/wm-output/wm-output/cosmos-predict-output/delay8_only/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_pi_libero_256_skip_dynamics_dual_cam_10/checkpoints/iter_000030000/model_ema_bf16.pt
# export CUDA_VISIBLE_DEVICES=7
# bash wm4vla/scripts/eval_wm_delay8_pi_libero.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${PI_LIBERO_DATA_ROOT:=/mnt/storage/users/kyji_data/tmp/lbai/cosmos-predict2.5/physical-intelligence/libero}"
: "${PI_LIBERO_T5_EMB_PATH:=${PI_LIBERO_DATA_ROOT}/meta/t5_embeddings.pkl}"
: "${EXPERIMENT:=ac_pi_libero_256_pixels_2b_10}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${CKPT:?Please set CKPT=/path/to/model_ema_bf16.pt}"
: "${NUM_STEPS:=35}"
: "${SAMPLES_PER_EPISODE:=5}"
: "${GUIDANCE:=0.0}"
: "${SEED:=42}"
: "${OUTPUT:=/home/kyji/storage_net/tmp/lbai/tmp/wm4lva-output/wm-output/wm-output/eval_wm/${EXPERIMENT}_delay8_30000.json}"
: "${SAVE_IMAGES:=/home/kyji/storage_net/tmp/lbai/tmp/wm4lva-output/wm-output/wm-output/eval_wm/${EXPERIMENT}_delay8_images_30000}"

export PI_LIBERO_DATA_ROOT
export PI_LIBERO_T5_EMB_PATH

for arg in "$@"; do
  if [[ "${arg}" == "--delays" ]]; then
    echo "This script is fixed to delay=8; do not pass --delays." >&2
    exit 1
  fi
done

echo "[eval][delay8][pi_libero] experiment=${EXPERIMENT}"
echo "[eval][delay8][pi_libero] data_root=${PI_LIBERO_DATA_ROOT}"
echo "[eval][delay8][pi_libero] ckpt=${CKPT}"
echo "[eval][delay8][pi_libero] num_steps=${NUM_STEPS}"
echo "[eval][delay8][pi_libero] samples_per_episode=${SAMPLES_PER_EPISODE}"
echo "[eval][delay8][pi_libero] output=${OUTPUT}"

cd "${REPO_ROOT}"

cmd=(
  python scripts/eval_world_model.py
  --ckpt "${CKPT}"
  --experiment "${EXPERIMENT}"
  --t5-emb-path "${PI_LIBERO_T5_EMB_PATH}"
  --num-steps "${NUM_STEPS}"
  --guidance "${GUIDANCE}"
  --samples-per-episode "${SAMPLES_PER_EPISODE}"
  --seed "${SEED}"
  --delays 8
)

if [[ -n "${OUTPUT}" ]]; then
  cmd+=(--output "${OUTPUT}")
fi

if [[ -n "${SAVE_IMAGES}" ]]; then
  cmd+=(--save-images "${SAVE_IMAGES}")
fi

cmd+=("$@")

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}"
