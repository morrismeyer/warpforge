#!/usr/bin/env bash
# Wake mark1nvidia from nuc via magic packet

MAC="30:56:0f:20:67:79 "   # <-- your amd MAC

echo "$(date): Sending WOL for mark1amd ($MAC)" >> ~/wake-amd.log

# Use directed broadcast on the LAN
wakeonlan -i 192.168.1.162 "$MAC"

