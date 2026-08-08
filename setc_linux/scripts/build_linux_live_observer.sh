#!/usr/bin/env bash
set -euo pipefail

export OMNIVGGT_ENABLE_LIVE_OBSERVER=ON
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/build_linux.sh" "$@"
