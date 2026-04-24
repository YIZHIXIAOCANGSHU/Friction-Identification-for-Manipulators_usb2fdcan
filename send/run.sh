#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERFACE="can0"
NOM_BITRATE="${NOM_BITRATE:-1000000}"
DATA_BITRATE="${DATA_BITRATE:-5000000}"
MST_ID="${MST_ID:-0x11}"
LISTEN_DURATION="${LISTEN_DURATION:-0}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 未找到，请先安装 Python 3。" >&2
  exit 1
fi

if ! command -v ip >/dev/null 2>&1; then
  echo "ip 命令未找到，请先安装 iproute2。" >&2
  exit 1
fi

if [[ "${SKIP_CAN_CONFIG:-0}" != "1" ]]; then
  if ! ip link show "${INTERFACE}" >/dev/null 2>&1; then
    echo "接口 ${INTERFACE} 不存在。" >&2
    exit 1
  fi

  echo "配置 ${INTERFACE} 为 CAN FD: bitrate=${NOM_BITRATE}, dbitrate=${DATA_BITRATE}"
  sudo ip link set "${INTERFACE}" down || true
  sudo ip link set "${INTERFACE}" type can bitrate "${NOM_BITRATE}" dbitrate "${DATA_BITRATE}" fd on
  sudo ip link set "${INTERFACE}" up
fi

cd "${ROOT_DIR}"

exec python3 "${ROOT_DIR}/scripts/enable_dm8009_id1.py" \
  --mst-id "${MST_ID}" \
  --listen-duration "${LISTEN_DURATION}" \
  "$@"
