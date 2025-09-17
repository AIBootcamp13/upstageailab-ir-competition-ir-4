#!/usr/bin/env bash
set -euo pipefail

# 모델이 너무 커서 CUDA_OUT_OF_MEMORY 에러가 발생한다면, 모델을 아래 중에 더 작은 걸로 변경해보세요!
# Qwen/Qwen3-Reranker-0.6B
# Qwen/Qwen3-Reranker-4B
# Qwen/Qwen3-Reranker-8B
# 혹은 batch_size를 1로 줄여보세요!

# flash attention 관련 에러가 생기면, --use_flash_attention을 지우고 실행해보세요!

# 현재 코드는 20개의 일상대화를 모두 포함한 상태에서 수행되었고, "멀티턴 대화의 모든 사용자 발화를 단순히 공백으로 연결하여 하나의 긴 쿼리로 변환하는 전략"을 사용하고 있습니다.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

UV_BIN="$(command -v uv)"
if [[ -z "${UV_BIN}" ]]; then
  echo "uv 명령을 찾을 수 없습니다. uv를 우선 설치해주세요." >&2
  exit 1
fi

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "${PYTHON_BIN} 경로의 프로젝트 가상환경을 찾을 수 없습니다." >&2
  echo "uv sync 등을 이용해 가상환경을 먼저 생성해주세요." >&2
  exit 1
fi

GLIBC_ROOT="/opt/glibc-2.32"
CUDA_ROOT="/usr/local/cuda-12.4"
LD_PATHS="${GLIBC_ROOT}/lib:${CUDA_ROOT}/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
LD_LOADER="${GLIBC_ROOT}/lib/ld-linux-x86-64.so.2"

if [[ ! -x "${LD_LOADER}" ]]; then
  echo "${LD_LOADER} 를 찾을 수 없습니다. glibc 2.32가 설치되어 있는지 확인해주세요." >&2
  exit 1
fi

# uv 환경이 최신 상태인지 확인
"${UV_BIN}" sync --frozen || {
  echo "uv sync 실행에 실패했습니다." >&2
  exit 1
}

CUDA_HOME="${CUDA_ROOT}" \
LD_LIBRARY_PATH="${LD_PATHS}" \
"${LD_LOADER}" --library-path "${LD_PATHS}" \
  "${PYTHON_BIN}" reranker2.py \
    --model_name Qwen/Qwen3-Reranker-4B \
    --documents_path ../../input/data/documents.jsonl \
    --eval_path ../../input/data/eval.jsonl \
    --scores_path ./similarity_scores.csv \
    --output_path ./reranked_submission.csv \
    --top_k 3 \
    --batch_size 1 \
    --use_flash_attention

# --batch_size 16 \
# --use_flash_attention

echo "Reranking 완료!"
echo "결과 파일: reranked_submission.csv"
echo "점수 파일: similarity_scores.csv"
