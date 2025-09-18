#!/usr/bin/env bash
set -euo pipefail

ES_VERSION="9.0.3"
ES_DIR="elasticsearch-${ES_VERSION}"
ES_TARBALL="${ES_DIR}-linux-x86_64.tar.gz"
ES_URL="https://artifacts.elastic.co/downloads/elasticsearch/${ES_TARBALL}"

echo "[install] Target version: ${ES_VERSION}"

cd ~

if [ -d "${ES_DIR}" ]; then
  echo "[install] ${ES_DIR} already exists. Skipping download/extract."
else
  echo "[install] Downloading ${ES_URL}..."
  wget -q --show-progress "${ES_URL}"
  echo "[install] Extracting ${ES_TARBALL}..."
  tar -xzf "${ES_TARBALL}"
fi

# Configure JVM heap to 2g via jvm.options.d
echo "[install] Setting JVM heap size to 2g..."
MEM_OPTS_DIR="./${ES_DIR}/config/jvm.options.d"
MEM_OPTS_FILE="${MEM_OPTS_DIR}/memory.options"
mkdir -p "${MEM_OPTS_DIR}"
if [ -f "${MEM_OPTS_FILE}" ]; then
  # Normalize existing Xms/Xmx to 2g
  sed -i -E 's/^(-Xms)[0-9]+[mMgG]/\12g/; s/^(-Xmx)[0-9]+[mMgG]/\12g/' "${MEM_OPTS_FILE}"
else
  printf "# Custom heap settings for memory optimization\n-Xms2g\n-Xmx2g\n" > "${MEM_OPTS_FILE}"
fi

if id -u daemon >/dev/null 2>&1; then
  echo "[install] Changing ownership to daemon:daemon"
  chown -R daemon:daemon "${ES_DIR}" || true
else
  echo "[install] 'daemon' user not found. Skipping chown."
fi

echo "[install] Installing analysis-nori plugin (batch mode)..."
"./${ES_DIR}/bin/elasticsearch-plugin" install --batch analysis-nori || true

echo "[install] Starting Elasticsearch as daemon user (if available)..."
if id -u daemon >/dev/null 2>&1; then
  sudo -u daemon "./${ES_DIR}/bin/elasticsearch" -d
else
  "./${ES_DIR}/bin/elasticsearch" -d
fi

echo "[install] Waiting 60 seconds for ES to boot..."
sleep 60

echo "[install] Resetting elastic password (interactive output will display new password)..."
if id -u daemon >/dev/null 2>&1; then
  sudo -u daemon "./${ES_DIR}/bin/elasticsearch-reset-password" -u elastic || true
else
  "./${ES_DIR}/bin/elasticsearch-reset-password" -u elastic || true
fi

echo "[install] Done."
