#!/usr/bin/env bash
# Wake mark1nvidia from nuc via magic packet

MAC="10:ff:e0:c4:be:75"   # <-- your nvidia MAC

echo "$(date): Sending WOL for mark1nvidia ($MAC)" >> ~/wake-nvidia.log

# Use directed broadcast on the LAN
wakeonlan -i 192.168.1.160 "$MAC"

