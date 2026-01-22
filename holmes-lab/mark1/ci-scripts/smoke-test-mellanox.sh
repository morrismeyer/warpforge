#!/usr/bin/env bash
set -euo pipefail

# Mellanox Cross-Connect Smoke Test
#
# Tests RDMA connectivity between NVIDIA and AMD boxes via Mellanox NICs.
# Intended to run ON the NUC as orchestrator.
#
# Flow:
# 1. Wake both GPU boxes
# 2. Wait for SSH on both
# 3. Sync warpforge repo on both
# 4. Start server on NVIDIA box
# 5. Run client on AMD box
# 6. Collect and display results
#
# Prerequisites:
# - Both boxes have Mellanox NICs configured with IP addresses
# - UCX/RDMA drivers installed and configured on both boxes
# - SSH keys set up for passwordless access

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# -------- CONFIG (override via environment variables) --------

# GPU box hostnames (must be in SSH config or /etc/hosts)
NVIDIA_HOST="${NVIDIA_HOST_OVERRIDE:-nvidia}"
AMD_HOST="${AMD_HOST_OVERRIDE:-amd}"

# Mellanox interface IPs (the cross-connect network, NOT the management network)
# These should be the IPs assigned to the Mellanox NICs
NVIDIA_MELLANOX_IP="${NVIDIA_MELLANOX_IP_OVERRIDE:-192.168.2.1}"
AMD_MELLANOX_IP="${AMD_MELLANOX_IP_OVERRIDE:-192.168.2.2}"

# MAC addresses for Wake-on-LAN (management NICs)
NVIDIA_MAC="${NVIDIA_MAC_OVERRIDE:-10:7C:61:3D:E7:8F}"
AMD_MAC="${AMD_MAC_OVERRIDE:-10:7C:61:63:CF:BD}"
BROADCAST_IP="${BROADCAST_IP_OVERRIDE:-192.168.1.255}"

# Path to the repo on GPU boxes
REMOTE_REPO_DIR="${REMOTE_REPO_DIR_OVERRIDE:-/home/actions/surfworks/warpforge}"

# WarpForge repo URL for cloning if missing
WARP_FORGE_REPO_URL="${WARP_FORGE_REPO_URL_OVERRIDE:-git@github.com:morrismeyer/warpforge.git}"

# Branch to use
BRANCH="${GITHUB_REF_NAME:-main}"

# Smoke test port
SMOKE_TEST_PORT="${SMOKE_TEST_PORT:-18515}"

# SSH settings
SSH_CONNECT_TIMEOUT_SECONDS="${SSH_CONNECT_TIMEOUT_SECONDS:-5}"
SSH_POLL_INTERVAL_SECONDS="${SSH_POLL_INTERVAL_SECONDS:-10}"
SSH_MAX_WAIT_SECONDS="${SSH_MAX_WAIT_SECONDS:-120}"

# Whether to wake boxes before testing (set to 0 if boxes are already running)
WAKE_BOXES="${WAKE_BOXES:-1}"

# Skip box if unreachable
SKIP_IF_UNREACHABLE="${SKIP_IF_UNREACHABLE:-0}"

# Logging
LOG_ROOT="${LOG_ROOT_OVERRIDE:-$HOME/build-logs}"
RUN_ID="${GITHUB_RUN_ID:-manual-$(date +%Y%m%d-%H%M%S)}"

# -------- END CONFIG --------

mkdir -p "$LOG_ROOT"
LOG_FILE="$LOG_ROOT/mellanox-smoke-${RUN_ID}.log"

# Server process tracking
SERVER_PID=""
SERVER_SSH_PID=""

log() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$timestamp] $*" | tee -a "$LOG_FILE"
}

log_section() {
    echo "" | tee -a "$LOG_FILE"
    log "════════════════════════════════════════════════════════════════════"
    log "  $*"
    log "════════════════════════════════════════════════════════════════════"
}

cleanup() {
    log "Cleaning up..."

    # Kill server process if running
    if [[ -n "$SERVER_SSH_PID" ]] && kill -0 "$SERVER_SSH_PID" 2>/dev/null; then
        log "Stopping server process (SSH PID: $SERVER_SSH_PID)..."
        kill "$SERVER_SSH_PID" 2>/dev/null || true
        wait "$SERVER_SSH_PID" 2>/dev/null || true
    fi

    # Also try to kill server on remote
    ssh -o ConnectTimeout=5 -o BatchMode=yes "$NVIDIA_HOST" \
        "pkill -f MellanoxSmokeTest || true" 2>/dev/null || true

    log "Cleanup complete"
}

trap cleanup EXIT

wake_host() {
    local mac=$1
    local name=$2

    if [[ "$WAKE_BOXES" -ne 1 ]]; then
        log "WAKE_BOXES=0, skipping wake for $name"
        return 0
    fi

    log "Sending Wake-on-LAN to $name (MAC: $mac)..."
    if command -v wakeonlan >/dev/null 2>&1; then
        wakeonlan -i "$BROADCAST_IP" "$mac" 2>&1 | tee -a "$LOG_FILE" || true
    elif command -v etherwake >/dev/null 2>&1; then
        sudo etherwake -i eth0 "$mac" 2>&1 | tee -a "$LOG_FILE" || true
    else
        log "WARN: No wake-on-lan tool found (wakeonlan or etherwake)"
    fi
}

wait_for_ssh() {
    local host=$1
    local max_wait=$2

    log "Waiting for SSH on $host (max ${max_wait}s)..."
    local start_ts
    start_ts="$(date +%s)"
    local attempt=0

    while true; do
        attempt=$((attempt + 1))
        if ssh -o ConnectTimeout="${SSH_CONNECT_TIMEOUT_SECONDS}" \
               -o BatchMode=yes \
               -o StrictHostKeyChecking=accept-new \
               "$host" "echo up" >/dev/null 2>&1; then
            log "$host is reachable via SSH"
            return 0
        fi

        local now_ts elapsed
        now_ts="$(date +%s)"
        elapsed=$((now_ts - start_ts))

        if (( elapsed >= max_wait )); then
            log "ERROR: $host did not become reachable within ${max_wait}s"
            return 1
        fi

        log "  ...waiting for $host (attempt $attempt, elapsed ${elapsed}s)"
        sleep "${SSH_POLL_INTERVAL_SECONDS}"
    done
}

sync_repo() {
    local host=$1

    log "Syncing warpforge repo on $host..."

    ssh "$host" bash -lc "
        set -euo pipefail

        echo '[remote] Preparing repo in ${REMOTE_REPO_DIR}'
        mkdir -p '${REMOTE_REPO_DIR}'
        cd '${REMOTE_REPO_DIR}'

        if [ ! -d .git ]; then
            echo '[remote] Cloning repo...'
            git clone '${WARP_FORGE_REPO_URL}' .
        fi

        echo '[remote] Fetching origin...'
        git fetch origin

        if git show-ref --verify --quiet 'refs/remotes/origin/${BRANCH}'; then
            if git show-ref --verify --quiet 'refs/heads/${BRANCH}'; then
                git checkout '${BRANCH}'
            else
                git checkout -b '${BRANCH}' 'origin/${BRANCH}'
            fi
            git reset --hard 'origin/${BRANCH}'
        else
            echo '[remote] ERROR: origin/${BRANCH} does not exist' >&2
            exit 1
        fi

        echo '[remote] Repo synced to ${BRANCH}'
    " 2>&1 | tee -a "$LOG_FILE"
}

check_mellanox_connectivity() {
    local from_host=$1
    local to_ip=$2
    local to_name=$3

    log "Testing basic IP connectivity from $from_host to $to_name ($to_ip)..."

    if ssh "$from_host" "ping -c 3 -W 2 '$to_ip'" 2>&1 | tee -a "$LOG_FILE"; then
        log "PASS: $from_host can reach $to_name via Mellanox ($to_ip)"
        return 0
    else
        log "FAIL: $from_host cannot reach $to_name via Mellanox ($to_ip)"
        return 1
    fi
}

check_rdma_devices() {
    local host=$1

    log "Checking RDMA devices on $host..."

    ssh "$host" bash -lc "
        set -euo pipefail

        echo '=== RDMA Devices ==='
        if [ -d /sys/class/infiniband ]; then
            ls -la /sys/class/infiniband/
        else
            echo 'No /sys/class/infiniband directory found'
        fi

        echo ''
        echo '=== ibstat (if available) ==='
        if command -v ibstat >/dev/null 2>&1; then
            ibstat || echo 'ibstat failed'
        else
            echo 'ibstat not installed'
        fi

        echo ''
        echo '=== UCX info (if available) ==='
        if command -v ucx_info >/dev/null 2>&1; then
            ucx_info -d | head -50 || echo 'ucx_info failed'
        else
            echo 'ucx_info not installed'
        fi
    " 2>&1 | tee -a "$LOG_FILE"
}

start_server() {
    log "Starting Mellanox smoke test server on $NVIDIA_HOST..."
    log "Server will listen on port $SMOKE_TEST_PORT"

    # Start server in background via SSH
    # The server will bind to all interfaces, clients connect via Mellanox IP
    ssh "$NVIDIA_HOST" bash -lc "
        cd '${REMOTE_REPO_DIR}'
        echo '[server] Building warpforge-io...'
        ./gradlew :warpforge-io:compileJava --no-daemon -q 2>&1

        echo '[server] Starting Mellanox smoke test server on port ${SMOKE_TEST_PORT}...'
        echo '[server] Waiting for client connection...'
        ./gradlew :warpforge-io:mellanoxServer -Pport=${SMOKE_TEST_PORT} --no-daemon 2>&1
    " 2>&1 | tee -a "$LOG_FILE" &

    SERVER_SSH_PID=$!
    log "Server SSH process started (PID: $SERVER_SSH_PID)"

    # Give server time to start
    log "Waiting for server to initialize (10 seconds)..."
    sleep 10

    # Check if SSH process is still running
    if ! kill -0 "$SERVER_SSH_PID" 2>/dev/null; then
        log "ERROR: Server process died immediately"
        wait "$SERVER_SSH_PID" || true
        return 1
    fi

    log "Server appears to be running"
    return 0
}

run_client() {
    log "Starting Mellanox smoke test client on $AMD_HOST..."
    log "Client will connect to $NVIDIA_HOST via Mellanox IP: $NVIDIA_MELLANOX_IP:$SMOKE_TEST_PORT"

    ssh "$AMD_HOST" bash -lc "
        cd '${REMOTE_REPO_DIR}'
        echo '[client] Building warpforge-io...'
        ./gradlew :warpforge-io:compileJava --no-daemon -q 2>&1

        echo '[client] Running Mellanox smoke test client...'
        echo '[client] Connecting to server at ${NVIDIA_MELLANOX_IP}:${SMOKE_TEST_PORT}...'
        ./gradlew :warpforge-io:mellanoxClient -Phost=${NVIDIA_MELLANOX_IP} -Pport=${SMOKE_TEST_PORT} --no-daemon 2>&1
    " 2>&1 | tee -a "$LOG_FILE"
}

# ======== MAIN ========

log_section "MELLANOX CROSS-CONNECT SMOKE TEST"
log "Run ID: $RUN_ID"
log "Log file: $LOG_FILE"
log ""
log "Configuration:"
log "  NVIDIA host: $NVIDIA_HOST (Mellanox IP: $NVIDIA_MELLANOX_IP)"
log "  AMD host: $AMD_HOST (Mellanox IP: $AMD_MELLANOX_IP)"
log "  Branch: $BRANCH"
log "  Remote repo: $REMOTE_REPO_DIR"
log "  Smoke test port: $SMOKE_TEST_PORT"

# Check for shutdown mode
NO_SHUTDOWN_FLAG="${HOME}/.holmes-lab/no-shutdown"
if [[ -f "$NO_SHUTDOWN_FLAG" ]]; then
    log ""
    log "NOTE: Auto-shutdown is DISABLED (flag file present)"
    log "      GPU boxes will stay running after test"
fi

# Phase 1: Wake both boxes
log_section "PHASE 1: WAKE GPU BOXES"

wake_host "$NVIDIA_MAC" "$NVIDIA_HOST"
wake_host "$AMD_MAC" "$AMD_HOST"

# Phase 2: Wait for SSH on both
log_section "PHASE 2: WAIT FOR SSH"

nvidia_ssh_ok=0
amd_ssh_ok=0

if wait_for_ssh "$NVIDIA_HOST" "$SSH_MAX_WAIT_SECONDS"; then
    nvidia_ssh_ok=1
else
    if [[ "$SKIP_IF_UNREACHABLE" -eq 1 ]]; then
        log "WARN: NVIDIA box unreachable, but SKIP_IF_UNREACHABLE=1"
    else
        log "FATAL: NVIDIA box unreachable and SKIP_IF_UNREACHABLE=0"
        exit 1
    fi
fi

if wait_for_ssh "$AMD_HOST" "$SSH_MAX_WAIT_SECONDS"; then
    amd_ssh_ok=1
else
    if [[ "$SKIP_IF_UNREACHABLE" -eq 1 ]]; then
        log "WARN: AMD box unreachable, but SKIP_IF_UNREACHABLE=1"
    else
        log "FATAL: AMD box unreachable and SKIP_IF_UNREACHABLE=0"
        exit 1
    fi
fi

if [[ "$nvidia_ssh_ok" -ne 1 || "$amd_ssh_ok" -ne 1 ]]; then
    log "FATAL: Both boxes must be reachable for cross-connect test"
    exit 1
fi

log "Both boxes are reachable via SSH"

# Phase 3: Sync repos
log_section "PHASE 3: SYNC REPOSITORIES"

sync_repo "$NVIDIA_HOST"
sync_repo "$AMD_HOST"

log "Repositories synced on both boxes"

# Phase 4: Check RDMA devices
log_section "PHASE 4: CHECK RDMA DEVICES"

check_rdma_devices "$NVIDIA_HOST"
check_rdma_devices "$AMD_HOST"

# Phase 5: Test Mellanox IP connectivity
log_section "PHASE 5: TEST MELLANOX IP CONNECTIVITY"

mlx_connectivity_ok=1

if ! check_mellanox_connectivity "$NVIDIA_HOST" "$AMD_MELLANOX_IP" "AMD"; then
    mlx_connectivity_ok=0
fi

if ! check_mellanox_connectivity "$AMD_HOST" "$NVIDIA_MELLANOX_IP" "NVIDIA"; then
    mlx_connectivity_ok=0
fi

if [[ "$mlx_connectivity_ok" -ne 1 ]]; then
    log ""
    log "FATAL: Mellanox IP connectivity failed"
    log "Please verify:"
    log "  1. Mellanox NICs are configured with correct IPs"
    log "  2. Both NICs are link-up (check 'ip link' and 'ibstat')"
    log "  3. No firewall blocking traffic on the cross-connect"
    exit 1
fi

log "Mellanox IP connectivity verified"

# Phase 6: Run RDMA smoke test
log_section "PHASE 6: RDMA SMOKE TEST"

# Start server on NVIDIA box (background)
if ! start_server; then
    log "FATAL: Failed to start server"
    exit 1
fi

# Give server a moment to fully initialize
sleep 5

# Run client on AMD box
log ""
log "Running client test..."
set +e
run_client
client_status=$?
set -e

# Wait a moment for server to finish
sleep 2

# Check results
log_section "TEST RESULTS"

if [[ "$client_status" -eq 0 ]]; then
    log ""
    log "╔══════════════════════════════════════════════════════════════════╗"
    log "║                    ALL TESTS PASSED                              ║"
    log "║                                                                  ║"
    log "║  Mellanox cross-connect is functional!                          ║"
    log "║  RDMA connectivity verified between NVIDIA and AMD boxes.       ║"
    log "╚══════════════════════════════════════════════════════════════════╝"
    log ""
    exit 0
else
    log ""
    log "╔══════════════════════════════════════════════════════════════════╗"
    log "║                    TEST FAILED                                   ║"
    log "║                                                                  ║"
    log "║  Check log file for details: $LOG_FILE"
    log "╚══════════════════════════════════════════════════════════════════╝"
    log ""
    exit 1
fi
