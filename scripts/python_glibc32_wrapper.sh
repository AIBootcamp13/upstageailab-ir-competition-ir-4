#!/usr/bin/env bash
# Wrapper that runs the project virtualenv's python with glibc 2.32 loader.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

GLIBC_ROOT="${GLIBC_ROOT:-/opt/glibc-2.32}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.4}"
LD_PATHS_DEFAULT="${GLIBC_ROOT}/lib:${CUDA_ROOT}/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
LD_LIBRARY_PATH_GLIBC="${FLASH_ATTN_LD_LIBRARY_PATH:-${LD_PATHS_DEFAULT}}"
LD_LOADER="${GLIBC_ROOT}/lib/ld-linux-x86-64.so.2"
PYTHON_BIN="${FLASH_ATTN_PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"

if [[ ! -x "${LD_LOADER}" ]]; then
  echo "[flash-attn wrapper] glibc 로더를 찾을 수 없습니다: ${LD_LOADER}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[flash-attn wrapper] 파이썬 인터프리터를 찾을 수 없습니다: ${PYTHON_BIN}" >&2
  exit 1
fi

exec "${LD_LOADER}" --library-path "${LD_LIBRARY_PATH_GLIBC}" "${PYTHON_BIN}" "$@"
