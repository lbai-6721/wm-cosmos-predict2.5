#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/vla-data/physical-intelligence"
KEEP_DIR="libero"
EXECUTE=false

usage() {
  cat <<'EOF'
Usage:
  bash scripts/delete_physical_intelligence_except_libero.sh
  bash scripts/delete_physical_intelligence_except_libero.sh --execute

Default mode is dry-run. It lists immediate child directories under
/data1/vla-data/physical-intelligence that would be deleted, excluding libero.

Pass --execute to actually delete those directories.
EOF
}

case "${1:-}" in
  "")
    ;;
  "--dry-run")
    ;;
  "--execute")
    EXECUTE=true
    ;;
  "-h"|"--help")
    usage
    exit 0
    ;;
  *)
    echo "Unknown argument: $1" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ ! -d "${ROOT}" ]]; then
  echo "Target root does not exist or is not a directory: ${ROOT}" >&2
  exit 1
fi

ROOT_REAL="$(realpath "${ROOT}")"
EXPECTED_REAL="$(realpath -m "/data1/vla-data/physical-intelligence")"

if [[ "${ROOT_REAL}" != "${EXPECTED_REAL}" ]]; then
  echo "Safety check failed: resolved root is ${ROOT_REAL}, expected ${EXPECTED_REAL}" >&2
  exit 1
fi

mapfile -d '' TARGETS < <(
  find "${ROOT}" -mindepth 1 -maxdepth 1 -type d ! -name "${KEEP_DIR}" -print0 | sort -z
)

if (( ${#TARGETS[@]} == 0 )); then
  echo "No directories to delete under ${ROOT}; keeping ${KEEP_DIR}."
  exit 0
fi

echo "Directories selected for deletion under ${ROOT}:"
for path in "${TARGETS[@]}"; do
  echo "  ${path}"
done

if [[ "${EXECUTE}" != true ]]; then
  echo
  echo "Dry-run only. Re-run with --execute to delete the directories above."
  exit 0
fi

for path in "${TARGETS[@]}"; do
  if [[ "$(basename "${path}")" == "${KEEP_DIR}" ]]; then
    echo "Refusing to delete kept directory: ${path}" >&2
    exit 1
  fi
  rm -rf --one-file-system -- "${path}"
done

echo "Deleted ${#TARGETS[@]} directories. Kept ${ROOT}/${KEEP_DIR}."
