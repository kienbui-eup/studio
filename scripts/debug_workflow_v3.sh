#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
PORT="${PORT:-8190}"
HOST="${HOST:-127.0.0.1}"
LOG_DIR="${ROOT_DIR}/output/debug"
LOG_FILE="${LOG_DIR}/workflow-v3-debug.log"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Missing virtualenv python: ${VENV_PYTHON}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "Starting ComfyUI debug server for workflow v3"
echo "Host: ${HOST}"
echo "Port: ${PORT}"
echo "Log:  ${LOG_FILE}"

exec "${VENV_PYTHON}" "${ROOT_DIR}/main.py" \
  --listen "${HOST}" \
  --port "${PORT}" \
  --verbose DEBUG \
  --log-stdout \
  --disable-auto-launch 2>&1 | tee "${LOG_FILE}"
