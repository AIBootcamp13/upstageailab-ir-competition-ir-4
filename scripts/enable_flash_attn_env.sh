#!/usr/bin/env bash
# Source this script to configure environment variables for flash-attn usage with uv run.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "이 스크립트는 source로 실행해야 합니다: source scripts/enable_flash_attn_env.sh" >&2
  exit 1
fi

_flash_attn_old_opts="$(set +o)"
set -euo pipefail
trap 'eval "${_flash_attn_old_opts}"; unset _flash_attn_old_opts' RETURN

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

GLIBC_ROOT_DEFAULT="/opt/glibc-2.32"
CUDA_ROOT_DEFAULT="/usr/local/cuda-12.4"

export GLIBC_ROOT="${GLIBC_ROOT:-${GLIBC_ROOT_DEFAULT}}"
export CUDA_ROOT="${CUDA_ROOT:-${CUDA_ROOT_DEFAULT}}"

if [[ ! -x "${GLIBC_ROOT}/lib/ld-linux-x86-64.so.2" ]]; then
  echo "[flash-attn env] glibc 2.32 로더를 찾을 수 없습니다: ${GLIBC_ROOT}/lib/ld-linux-x86-64.so.2" >&2
  return 1
fi

FLASH_ATTN_LD_LIBRARY_PATH="${GLIBC_ROOT}/lib:${CUDA_ROOT}/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
export FLASH_ATTN_LD_LIBRARY_PATH

FLASH_ATTN_PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${FLASH_ATTN_PYTHON_BIN}" ]]; then
  echo "[flash-attn env] 프로젝트 가상환경 파이썬을 찾을 수 없습니다: ${FLASH_ATTN_PYTHON_BIN}" >&2
  echo "uv sync 등을 이용해 .venv를 먼저 준비해주세요." >&2
  return 1
fi
export FLASH_ATTN_PYTHON_BIN

PYTHON_WRAPPER="${SCRIPT_DIR}/python_glibc32_wrapper.sh"
if [[ ! -x "${PYTHON_WRAPPER}" ]]; then
  echo "[flash-attn env] 파이썬 래퍼 스크립트가 없습니다: ${PYTHON_WRAPPER}" >&2
  return 1
fi

FLASH_ATTN_UV_BIN="$(command -v uv)"
if [[ -z "${FLASH_ATTN_UV_BIN}" ]]; then
  echo "[flash-attn env] uv 명령을 찾을 수 없습니다." >&2
  return 1
fi

uv() {
  if [[ "$#" -gt 0 && "$1" == "run" ]]; then
    shift
    local run_opts=()
    while [[ "$#" -gt 0 ]]; do
      case "$1" in
        --)
          run_opts+=("$1")
          shift
          break
          ;;
        --*)
          run_opts+=("$1")
          shift
          ;;
        python|python3)
          shift
          command "${FLASH_ATTN_UV_BIN}" run "${run_opts[@]}" "${PYTHON_WRAPPER}" "$@"
          return $?
          ;;
        *)
          break
          ;;
      esac
    done

    if [[ "$#" -gt 0 ]]; then
      local candidate="$1"
      if [[ "${candidate}" == *.py || -f "${candidate}" ]]; then
        command "${FLASH_ATTN_UV_BIN}" run "${run_opts[@]}" "${PYTHON_WRAPPER}" "$@"
        return $?
      fi
    fi

    command "${FLASH_ATTN_UV_BIN}" run "${run_opts[@]}" "$@"
    return $?
  fi

  command "${FLASH_ATTN_UV_BIN}" "$@"
}

echo "[flash-attn env] 준비 완료. 'uv run python ...' 또는 'uv run <script>.py' 명령이 glibc 2.32 래퍼를 사용합니다."
echo "[flash-attn env] uv 함수가 재정의되었습니다. 현재 래퍼: ${PYTHON_WRAPPER}"
