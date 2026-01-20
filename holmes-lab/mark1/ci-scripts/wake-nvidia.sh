#!/usr/bin/env bash
set -euo pipefail

# Wake the NVIDIA box from the NUC via Wake-on-LAN (WOL).

MAC="${MAC_OVERRIDE:-10:7C:61:3D:E7:8F}"                 # NVIDIA NIC MAC
BROADCAST_IP="${BROADCAST_IP_OVERRIDE:-192.168.1.255}"   # broadcast for your LAN
LOG_FILE="${LOG_FILE_OVERRIDE:-$HOME/wake-nvidia.log}"
WOL_SEND_TIMEOUT_SECONDS="${WOL_SEND_TIMEOUT_SECONDS:-3}"

echo "$(date): Sending WOL for mark1nvidia (MAC=$MAC, broadcast=$BROADCAST_IP)" >> "$LOG_FILE"

if ! command -v wakeonlan >/dev/null 2>&1; then
  echo "wakeonlan not found. Install it on the NUC (for Debian/Ubuntu: sudo apt-get install wakeonlan)." >&2
  exit 1
fi

if command -v timeout >/dev/null 2>&1; then
  # wakeonlan is usually instantaneous; this just protects the orchestrator from rare hangs.
  timeout "${WOL_SEND_TIMEOUT_SECONDS}s" wakeonlan -i "$BROADCAST_IP" "$MAC" || true
else
  wakeonlan -i "$BROADCAST_IP" "$MAC"
fi

