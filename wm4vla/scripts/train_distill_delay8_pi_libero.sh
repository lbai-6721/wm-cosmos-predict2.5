#!/usr/bin/env bash
set -euo pipefail

: "${BENCHMARK:=10}"
: "${FIXED_DELAY:=8}"
: "${MODEL_MAX_DELAY:=8}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if (( FIXED_DELAY < 1 )); then
  echo "FIXED_DELAY must be >= 1" >&2
  exit 1
fi

if (( FIXED_DELAY > MODEL_MAX_DELAY )); then
  echo "FIXED_DELAY must be <= MODEL_MAX_DELAY (${MODEL_MAX_DELAY})" >&2
  exit 1
fi

echo "[distill][delay8][pi_libero] benchmark=${BENCHMARK}"
echo "[distill][delay8][pi_libero] fixed_delay=${FIXED_DELAY}"

BENCHMARK="${BENCHMARK}" \
exec bash "${SCRIPT_DIR}/train_distill_pi_libero.sh" \
  "$@" \
  "dataloader_train.dataset.fixed_delay=${FIXED_DELAY}" \
  "dataloader_val.dataset.fixed_delay=${FIXED_DELAY}" \
  "dataloader_train.dataset.sampled_delay_max=${FIXED_DELAY}" \
  "dataloader_val.dataset.sampled_delay_max=${FIXED_DELAY}"
