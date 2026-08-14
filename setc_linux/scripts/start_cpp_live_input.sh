#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${PACKAGE_DIR}/build_live_observer}"
SERVER="${SERVER:-${BUILD_DIR}/bin/omnivggt_stream_server}"

MAX_INFLIGHT_GROUPS="${MAX_INFLIGHT_GROUPS:-3}"
HISTORY_KEEP_GROUPS="${HISTORY_KEEP_GROUPS:?set HISTORY_KEEP_GROUPS to the requested retention count}"
IMAGE_DIR="${IMAGE_DIR:-${PACKAGE_DIR}/data2}"
OUTPUT_DIR="${OUTPUT_DIR:-${PACKAGE_DIR}/outputs/data2_cpp_linux_live_input}"
MODEL="${MODEL:-${PACKAGE_DIR}/models/omnivggt_observer_s1_700x434_bf16_unfrozen_torch270.pt}"
PAIR_MODEL="${PAIR_MODEL:-${PACKAGE_DIR}/models/omnivggt_observer_s2_700x700_bf16_unfrozen_torch270.pt}"

if [[ "${MAX_INFLIGHT_GROUPS}" != "3" ]]; then
    echo "MAX_INFLIGHT_GROUPS must be 3" >&2
    exit 1
fi
if [[ ! -x "${SERVER}" ]]; then
    echo "[ERROR] Server binary was not found: ${SERVER}" >&2
    exit 1
fi

exec "${SERVER}" \
    --image-dir "${IMAGE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --model-pair "${PAIR_MODEL}" \
    --pair-letterbox \
    --input-group-size 3 \
    --input-group-stride 1 \
    --group-anchor-index 1 \
    --history-keep-groups "${HISTORY_KEEP_GROUPS}" \
    --device cuda \
    --dtype bf16
