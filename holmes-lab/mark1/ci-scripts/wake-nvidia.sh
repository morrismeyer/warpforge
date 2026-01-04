#!/usr/bin/env bash
set -euo pipefail

# Wake the NVIDIA box from the NUC via Wake-on-LAN (WOL).

MAC="${MAC_OVERRIDE:-10:ff:e0:c4:be:75}"                 # NVIDIA NIC MAC
BROADCAST_IP="${BROADCAST_IP_OVERRIDE:-192.168.1.255}"   # broadcast for your LAN
LOG_FILE="${LOG_FILE_OVERRIDE:-$HOME/wake-nvidia.log}"

echo "$(date): Sending WOL for mark1nvidia (MAC=$MAC, broadcast=$BROADCAST_IP)" >> "$LOG_FILE"

if ! command -v wakeonlan >/dev/null 2>&1; then
  echo "wakeonlan not found. Install it on the NUC (for Debian/Ubuntu: sudo apt-get install wakeonlan)." >&2
  exit 1
fi

wakeonlan -i "$BROADCAST_IP" "$MAC"

