#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG:=cosmos_predict2/_src/interactive/configs/registry_predict2p5.py}"
: "${PI_LIBERO_DATA_ROOT:=/mnt/storage/users/kyji_data/tmp/lbai/cosmos-predict2.5/physical-intelligence/libero}"
: "${PI_LIBERO_T5_EMB_PATH:=${PI_LIBERO_DATA_ROOT}/meta/t5_embeddings.pkl}"
: "${BENCHMARK:=all}"
: "${MASTER_PORT:=12340}"
: "${NPROC_PER_NODE:=4}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
: "${BATCH_SIZE:=16}"

case "${BENCHMARK}" in
  all)
    DEFAULT_EXPERIMENT="dmd2_trigflow_distill_wm_pi_libero_256_all"
    TEACHER_ENV_VAR="WM4VLA_PI_LIBERO_TEACHER_CKPT_ALL"
    ;;
  10|libero_10)
    DEFAULT_EXPERIMENT="dmd2_trigflow_distill_wm_pi_libero_256_10"
    TEACHER_ENV_VAR="WM4VLA_PI_LIBERO_TEACHER_CKPT_10"
    ;;
  goal|libero_goal)
    DEFAULT_EXPERIMENT="dmd2_trigflow_distill_wm_pi_libero_256_goal"
    TEACHER_ENV_VAR="WM4VLA_PI_LIBERO_TEACHER_CKPT_GOAL"
    ;;
  object|libero_object)
    DEFAULT_EXPERIMENT="dmd2_trigflow_distill_wm_pi_libero_256_object"
    TEACHER_ENV_VAR="WM4VLA_PI_LIBERO_TEACHER_CKPT_OBJECT"
    ;;
  spatial|libero_spatial)
    DEFAULT_EXPERIMENT="dmd2_trigflow_distill_wm_pi_libero_256_spatial"
    TEACHER_ENV_VAR="WM4VLA_PI_LIBERO_TEACHER_CKPT_SPATIAL"
    ;;
  *)
    echo "Unsupported BENCHMARK=${BENCHMARK}" >&2
    echo "Expected one of: all, 10/libero_10, goal/libero_goal, object/libero_object, spatial/libero_spatial" >&2
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

export PI_LIBERO_DATA_ROOT
export PI_LIBERO_T5_EMB_PATH

echo "[distill][pi_libero] benchmark=${BENCHMARK}"
echo "[distill][pi_libero] experiment=${EXPERIMENT}"
echo "[distill][pi_libero] data_root=${PI_LIBERO_DATA_ROOT}"
echo "[distill][pi_libero] teacher_ckpt=${TEACHER_CKPT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
  -m scripts.train \
  --config="${CONFIG}" -- \
  experiment="${EXPERIMENT}" \
  dataloader_train.batch_size="${BATCH_SIZE}" \
  "model.config.teacher_load_from.load_path=${TEACHER_CKPT}" \
  "$@"
