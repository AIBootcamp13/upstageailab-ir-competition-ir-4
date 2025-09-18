#!/usr/bin/env bash
set -euo pipefail

echo "[stop] Running Elasticsearch stop routine..."

# 우선 실행 중인 ES 프로세스 PID 탐색 (일반/daemon 사용자 모두 고려)
PIDS=$(pgrep -f "org.elasticsearch.bootstrap.Elasticsearch" || true)

if [ -z "${PIDS}" ]; then
  echo "[stop] Elasticsearch process not found."
  exit 0
fi

echo "[stop] Found PIDs: ${PIDS}"

echo "[stop] Sending SIGTERM..."
kill ${PIDS} || true

# 최대 30초 대기하며 종료 확인
for i in $(seq 1 30); do
  sleep 1
  if ! pgrep -f "org.elasticsearch.bootstrap.Elasticsearch" >/dev/null 2>&1; then
    echo "[stop] Elasticsearch stopped gracefully."
    exit 0
  fi
done

echo "[stop] Forcing SIGKILL..."
kill -9 ${PIDS} || true

if pgrep -f "org.elasticsearch.bootstrap.Elasticsearch" >/dev/null 2>&1; then
  echo "[stop] Failed to stop Elasticsearch (still running)." >&2
  exit 1
fi

echo "[stop] Elasticsearch stopped."

