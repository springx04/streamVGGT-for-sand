#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build}"
LIBTORCH_VALUE="${LIBTORCH_ROOT:-${LIBTORCH:-}}"
ENABLE_LIVE_OBSERVER="${OMNIVGGT_ENABLE_LIVE_OBSERVER:-OFF}"

if [[ -z "${LIBTORCH_VALUE}" ]]; then
    echo "[ERROR] Set LIBTORCH or LIBTORCH_ROOT to a Linux LibTorch directory." >&2
    exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
    echo "[ERROR] cmake was not found in PATH." >&2
    exit 1
fi
if [[ -z "${CMAKE_CXX_COMPILER:-}" ]]; then
    if command -v g++ >/dev/null 2>&1; then
        CMAKE_CXX_COMPILER="$(command -v g++)"
    else
        echo "[ERROR] g++ was not found in PATH." >&2
        exit 1
    fi
fi

cmake_args=(
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}"
    "-DLIBTORCH_ROOT=${LIBTORCH_VALUE}"
    "-DOMNIVGGT_ENABLE_LIVE_OBSERVER=${ENABLE_LIVE_OBSERVER}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
)
if [[ -n "${OpenCV_DIR:-}" ]]; then
    cmake_args+=("-DOpenCV_DIR=${OpenCV_DIR}")
elif [[ -f "/usr/lib/x86_64-linux-gnu/cmake/opencv4/OpenCVConfig.cmake" ]]; then
    cmake_args+=("-DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4")
fi
if [[ -n "${CMAKE_GENERATOR:-}" ]]; then
    cmake_args+=("-G" "${CMAKE_GENERATOR}")
fi

echo "Configuring Linux C++ build: ${BUILD_DIR}"
cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" "${cmake_args[@]}"
echo "Building OmniVGGT C++ targets"
cmake --build "${BUILD_DIR}" --parallel
echo "Build complete: ${BUILD_DIR}/bin"
