#!/usr/bin/env bash
set -euo pipefail

# Control automatic shutdown behavior for Holmes Lab GPU boxes.
#
# Usage:
#   ./shutdown-mode.sh status    # Show current mode
#   ./shutdown-mode.sh disable   # Disable auto-shutdown (machines stay on)
#   ./shutdown-mode.sh enable    # Enable auto-shutdown (default CI behavior)
#
# This script manages a flag file that the orchestration scripts check.
# When shutdown is disabled, GPU boxes will stay running after WOL + build/test.
#
# Useful for:
#   - Testing Mellanox cross-connect
#   - Extended debugging sessions
#   - Manual testing without CI interference

FLAG_DIR="${FLAG_DIR_OVERRIDE:-$HOME/.holmes-lab}"
FLAG_FILE="${FLAG_DIR}/no-shutdown"

usage() {
    cat <<EOF
Holmes Lab Shutdown Mode Control

Usage: $(basename "$0") <command>

Commands:
    status    Show current shutdown mode
    disable   Disable auto-shutdown (GPU boxes stay running after tests)
    enable    Enable auto-shutdown (GPU boxes power off after successful tests)

Current flag file: ${FLAG_FILE}

Examples:
    $(basename "$0") disable   # Before Mellanox testing
    $(basename "$0") status    # Check current mode
    $(basename "$0") enable    # Re-enable when done testing
EOF
    exit 1
}

show_status() {
    if [[ -f "$FLAG_FILE" ]]; then
        echo "Shutdown mode: DISABLED"
        echo "  GPU boxes will stay running after WOL + build/test"
        echo "  Flag file: ${FLAG_FILE}"
        echo ""
        echo "  To re-enable auto-shutdown: $(basename "$0") enable"
    else
        echo "Shutdown mode: ENABLED (default)"
        echo "  GPU boxes will power off after successful build/test"
        echo ""
        echo "  To disable auto-shutdown: $(basename "$0") disable"
    fi
}

disable_shutdown() {
    mkdir -p "$FLAG_DIR"
    touch "$FLAG_FILE"
    echo "Auto-shutdown DISABLED"
    echo "  GPU boxes will stay running after WOL + build/test"
    echo "  Flag file created: ${FLAG_FILE}"
    echo ""
    echo "  Remember to re-enable when done: $(basename "$0") enable"
}

enable_shutdown() {
    if [[ -f "$FLAG_FILE" ]]; then
        rm -f "$FLAG_FILE"
        echo "Auto-shutdown ENABLED"
        echo "  GPU boxes will power off after successful build/test"
        echo "  Flag file removed: ${FLAG_FILE}"
    else
        echo "Auto-shutdown already ENABLED (no flag file present)"
    fi
}

# Main
if [[ $# -lt 1 ]]; then
    usage
fi

case "${1:-}" in
    status)
        show_status
        ;;
    disable|off|stop)
        disable_shutdown
        ;;
    enable|on|start)
        enable_shutdown
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown command: $1" >&2
        echo ""
        usage
        ;;
esac
