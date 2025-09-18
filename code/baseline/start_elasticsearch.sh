#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash code/baseline/start_elasticsearch.sh [VERSION] [WAIT_SECONDS]
# Examples:
#   bash code/baseline/start_elasticsearch.sh           # 자동 탐지 후 시작
#   bash code/baseline/start_elasticsearch.sh 9.0.3     # 9.0.3로 시작
#   bash code/baseline/start_elasticsearch.sh 9.0.3 60  # 60초 대기 포함

VERSION_ARG="${1:-}"
WAIT_SECONDS="${2:-30}"

cd ~

# ES 디렉터리 결정
if [[ -n "${VERSION_ARG}" ]]; then
  ES_DIR="elasticsearch-${VERSION_ARG}"
else
  if [[ -d "elasticsearch-9.0.3" ]]; then
    ES_DIR="elasticsearch-9.0.3"
  elif [[ -d "elasticsearch-8.8.0" ]]; then
    ES_DIR="elasticsearch-8.8.0"
  else
    # 가장 최신 버전 자동 선택 (폴더명 버전 기준)
    ES_DIR=$(ls -d elasticsearch-* 2>/dev/null | sort -V | tail -n1 || true)
    if [[ -z "${ES_DIR}" ]]; then
      echo "[start] 설치된 Elasticsearch 디렉터리를 찾지 못했습니다." >&2
      exit 1
    fi
  fi
fi

if [[ ! -d "${ES_DIR}" ]]; then
  echo "[start] 대상 디렉터리가 없습니다: ~/${ES_DIR}" >&2
  exit 1
fi

echo "[start] Target directory: ~/${ES_DIR}"

# 이미 실행 중인지 확인
if pgrep -f "org.elasticsearch.bootstrap.Elasticsearch" >/dev/null 2>&1; then
  echo "[start] 이미 실행 중인 Elasticsearch 프로세스가 있습니다. 중복 실행을 피하기 위해 종료합니다."
  exit 0
fi

# daemon 사용자 존재 시 해당 계정으로 실행
if id -u daemon >/dev/null 2>&1; then
  echo "[start] Starting as daemon user..."
  sudo -u daemon "./${ES_DIR}/bin/elasticsearch" -d
else
  echo "[start] Starting as current user..."
  "./${ES_DIR}/bin/elasticsearch" -d
fi

if [[ "${WAIT_SECONDS}" =~ ^[0-9]+$ ]] && [[ "${WAIT_SECONDS}" -gt 0 ]]; then
  echo "[start] Waiting ${WAIT_SECONDS}s for Elasticsearch to boot..."
  sleep "${WAIT_SECONDS}"
fi

echo "[start] Done. 상태 확인 예시:"
echo "  curl --cacert ~/${ES_DIR}/config/certs/http_ca.crt https://localhost:9200 -u elastic:<비밀번호>"
echo "[start] 로그: tail -f ~/${ES_DIR}/logs/elasticsearch.log"

