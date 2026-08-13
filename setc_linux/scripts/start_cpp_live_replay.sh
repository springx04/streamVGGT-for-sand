#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${PACKAGE_DIR}/build_live_observer}"
BIN_DIR="${BIN_DIR:-${BUILD_DIR}/bin}"

LIBTORCH_VALUE="${LIBTORCH_ROOT:-${LIBTORCH:-}}"
if [[ -n "${LIBTORCH_VALUE}" && -d "${LIBTORCH_VALUE}/lib" ]]; then
    export LD_LIBRARY_PATH="${LIBTORCH_VALUE}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME:-}/lib64" ]]; then
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
# Ubuntu packages OpenCV 4.x under the multiarch library directory.  Keep
# this optional: custom OpenCV builds can still be supplied through LD_LIBRARY_PATH.
if [[ -d "/usr/lib/x86_64-linux-gnu" ]]; then
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

SERVER="${SERVER:-${BIN_DIR}/omnivggt_stream_server}"
VIEWER="${VIEWER:-${BIN_DIR}/omnivggt_live_viewer}"
MODEL="${MODEL:-${PACKAGE_DIR}/models/omnivggt_observer_s1_700x434_bf16_unfrozen_torch270.pt}"
PAIR_MODEL="${PAIR_MODEL:-${PACKAGE_DIR}/models/omnivggt_observer_s2_700x700_bf16_unfrozen_torch270.pt}"
GROUP_MODEL="${GROUP_MODEL:-${PACKAGE_DIR}/models/omnivggt_observer_b3s1_406x252_bf16_unfrozen_torch270.pt}"
INPUT_GROUP_SIZE="${INPUT_GROUP_SIZE:-3}"
INPUT_GROUP_STRIDE="${INPUT_GROUP_STRIDE:-1}"
GROUP_ANCHOR_INDEX="${GROUP_ANCHOR_INDEX:-1}"
GROUP_MODEL_WIDTH="${GROUP_MODEL_WIDTH:-406}"
GROUP_MODEL_HEIGHT="${GROUP_MODEL_HEIGHT:-252}"
IMAGE_DIR="${IMAGE_DIR:-${PACKAGE_DIR}/data2}"
OUTPUT_DIR="${OUTPUT_DIR:-${PACKAGE_DIR}/outputs/data2_cpp_linux_live_replay}"
TARGET_WIDTH="${TARGET_WIDTH:-700}"
TARGET_SIZE="${TARGET_SIZE:-700}"
CANVAS_WIDTH="${CANVAS_WIDTH:-770}"
CANVAS_HEIGHT="${CANVAS_HEIGHT:-630}"
FIRST_MODEL_WIDTH="${FIRST_MODEL_WIDTH:-700}"
FIRST_MODEL_HEIGHT="${FIRST_MODEL_HEIGHT:-434}"
QUEUE_CAPACITY="${QUEUE_CAPACITY:-1024}"
PORT="${PORT:-37651}"
SERVER_STARTUP_DELAY="${SERVER_STARTUP_DELAY:-7}"
SERVER_LOG="${SERVER_LOG:-${OUTPUT_DIR}/server.log}"

required_files=("${SERVER}" "${VIEWER}")
if [[ "${INPUT_GROUP_SIZE}" == "3" && -n "${USE_GROUP_MODEL:-}" ]]; then
    required_files+=("${GROUP_MODEL}")
else
    required_files+=("${MODEL}" "${PAIR_MODEL}")
fi
for required_file in "${required_files[@]}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "[ERROR] Required file was not found: ${required_file}" >&2
        exit 1
    fi
done
if [[ ! -d "${IMAGE_DIR}" ]]; then
    echo "[ERROR] Image directory was not found: ${IMAGE_DIR}" >&2
    exit 1
fi
mkdir -p "${OUTPUT_DIR}"

server_pid=""
cleanup() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting OmniVGGT Linux C++ live replay"
echo "  dataset: ${IMAGE_DIR}"
echo "  output:  ${OUTPUT_DIR}"
if [[ "${INPUT_GROUP_SIZE}" == "3" ]]; then
    echo "  input:   groups of 3, stride=${INPUT_GROUP_STRIDE}, anchor=${GROUP_ANCHOR_INDEX}"
    if [[ -n "${USE_GROUP_MODEL:-}" ]]; then
        echo "  group:   B=3,S=1 ${GROUP_MODEL_WIDTH}x${GROUP_MODEL_HEIGHT} experimental group geometry"
    else
        echo "  group:   three-image observation fusion, then standard S=2 stream update"
    fi
else
    echo "  model:   ${MODEL}"
    echo "  pair:    ${PAIR_MODEL}"
fi
echo "  viewer:  ${VIEWER}"
echo "  log:     ${SERVER_LOG}"

server_args=(
    --image-dir "${IMAGE_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --target-size "${TARGET_SIZE}"
    --target-width "${TARGET_WIDTH}"
    --canvas-width "${CANVAS_WIDTH}"
    --canvas-height "${CANVAS_HEIGHT}"
    --first-model-width "${FIRST_MODEL_WIDTH}"
    --first-model-height "${FIRST_MODEL_HEIGHT}"
    --device cuda
    --dtype bf16
    --min_conf 0.0
    --queue-capacity "${QUEUE_CAPACITY}"
    --port "${PORT}"
    --no-save-debug
)
if [[ "${INPUT_GROUP_SIZE}" == "3" && -n "${USE_GROUP_MODEL:-}" ]]; then
    server_args+=(
        --model-group3 "${GROUP_MODEL}"
        --input-group-size 3
        --input-group-stride "${INPUT_GROUP_STRIDE}"
        --group-anchor-index "${GROUP_ANCHOR_INDEX}"
        --group-model-width "${GROUP_MODEL_WIDTH}"
        --group-model-height "${GROUP_MODEL_HEIGHT}"
    )
elif [[ "${INPUT_GROUP_SIZE}" == "3" ]]; then
    server_args+=(
        --model "${MODEL}"
        --model-pair "${PAIR_MODEL}"
        --pair-letterbox
        --input-group-size 3
        --input-group-stride "${INPUT_GROUP_STRIDE}"
        --group-anchor-index "${GROUP_ANCHOR_INDEX}"
    )
else
    server_args+=(--model "${MODEL}" --model-pair "${PAIR_MODEL}" --pair-letterbox)
fi

"${SERVER}" "${server_args[@]}" >"${SERVER_LOG}" 2>&1 &
server_pid=$!

sleep "${SERVER_STARTUP_DELAY}"
if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "[ERROR] C++ server exited during startup. See ${SERVER_LOG}" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
fi

echo "Server is running with PID ${server_pid}; close the viewer with q or Esc."
"${VIEWER}" --host 127.0.0.1 --port "${PORT}" --display-max-points 0
