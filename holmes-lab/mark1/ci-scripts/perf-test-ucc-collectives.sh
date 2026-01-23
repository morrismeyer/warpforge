#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Two-Node UCC Collective Performance Test
# =============================================================================
#
# Comprehensive performance testing for UCC collective operations over
# Mellanox 100GbE cross-connect between NVIDIA and AMD GPU boxes.
#
# Based on smoke-test-mellanox.sh pattern - runs from NUC orchestrator.
#
# Flow:
# 1. Wake both GPU boxes (if not already running)
# 2. Wait for SSH on both
# 3. Sync warpforge repo on both
# 4. Validate RDMA connectivity
# 5. Run performance benchmarks (multiple message sizes, iterations)
# 6. Collect and analyze results
# 7. Compare against RDMA baseline
#
# Usage:
#   ./perf-test-ucc-collectives.sh [OPTIONS]
#
# Options:
#   --quick          Run quick benchmark (fewer iterations, fewer sizes)
#   --iterations N   Override default iteration count
#   --sizes LIST     Comma-separated list of message sizes (bytes)
#   --warmup N       Number of warmup iterations
#   --no-wake        Skip Wake-on-LAN (boxes already running)
#   --continuous     Run continuously until interrupted (overnight mode)
#   --interval SECS  Seconds between continuous runs (default: 300)
#
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# -------- CONFIG (override via environment variables) --------

# GPU box hostnames
NVIDIA_HOST="${NVIDIA_HOST_OVERRIDE:-nvidia}"
AMD_HOST="${AMD_HOST_OVERRIDE:-amd}"

# Mellanox interface IPs (cross-connect network)
NVIDIA_MELLANOX_IP="${NVIDIA_MELLANOX_IP_OVERRIDE:-192.168.2.1}"
AMD_MELLANOX_IP="${AMD_MELLANOX_IP_OVERRIDE:-192.168.2.2}"

# MAC addresses for Wake-on-LAN
NVIDIA_MAC="${NVIDIA_MAC_OVERRIDE:-10:7C:61:3D:E7:8F}"
AMD_MAC="${AMD_MAC_OVERRIDE:-10:7C:61:63:CF:BD}"
BROADCAST_IP="${BROADCAST_IP_OVERRIDE:-192.168.1.255}"

# Path to the repo on GPU boxes
REMOTE_REPO_DIR="${REMOTE_REPO_DIR_OVERRIDE:-/home/actions/surfworks/warpforge}"

# WarpForge repo URL for cloning if missing
WARP_FORGE_REPO_URL="${WARP_FORGE_REPO_URL_OVERRIDE:-git@github.com:morrismeyer/warpforge.git}"

# Branch to use
BRANCH="${GITHUB_REF_NAME:-main}"

# UCC test port
UCC_TEST_PORT="${UCC_TEST_PORT:-29500}"

# SSH settings
SSH_CONNECT_TIMEOUT_SECONDS="${SSH_CONNECT_TIMEOUT_SECONDS:-5}"
SSH_POLL_INTERVAL_SECONDS="${SSH_POLL_INTERVAL_SECONDS:-10}"
SSH_MAX_WAIT_SECONDS="${SSH_MAX_WAIT_SECONDS:-120}"

# Performance test settings
DEFAULT_ITERATIONS=100
DEFAULT_WARMUP=10
DEFAULT_SIZES="1024,4096,16384,65536,262144,1048576,4194304,16777216,67108864"
# 1KB, 4KB, 16KB, 64KB, 256KB, 1MB, 4MB, 16MB, 64MB

# Continuous mode settings
CONTINUOUS_INTERVAL=300  # 5 minutes between runs

# Logging
LOG_ROOT="${LOG_ROOT_OVERRIDE:-$HOME/build-logs}"
RUN_ID="${GITHUB_RUN_ID:-perf-$(date +%Y%m%d-%H%M%S)}"

# -------- END CONFIG --------

# Parse command line options
QUICK_MODE=0
ITERATIONS=$DEFAULT_ITERATIONS
WARMUP=$DEFAULT_WARMUP
SIZES=$DEFAULT_SIZES
WAKE_BOXES=1
CONTINUOUS_MODE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            QUICK_MODE=1
            ITERATIONS=10
            WARMUP=2
            SIZES="65536,1048576,16777216"  # 64KB, 1MB, 16MB
            shift
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --sizes)
            SIZES="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --no-wake)
            WAKE_BOXES=0
            shift
            ;;
        --continuous)
            CONTINUOUS_MODE=1
            shift
            ;;
        --interval)
            CONTINUOUS_INTERVAL="$2"
            shift 2
            ;;
        --help|-h)
            head -50 "$0" | tail -n +2 | grep -E "^#" | sed 's/^# *//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$LOG_ROOT"
LOG_FILE="$LOG_ROOT/ucc-perf-${RUN_ID}.log"
RESULTS_FILE="$LOG_ROOT/ucc-perf-${RUN_ID}-results.csv"

# Server process tracking
SERVER_SSH_PID=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "[$timestamp] $*" | tee -a "$LOG_FILE"
}

log_section() {
    echo "" | tee -a "$LOG_FILE"
    log "${CYAN}════════════════════════════════════════════════════════════════════${NC}"
    log "${CYAN}  $*${NC}"
    log "${CYAN}════════════════════════════════════════════════════════════════════${NC}"
}

log_success() {
    log "${GREEN}[PASS]${NC} $*"
}

log_warning() {
    log "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    log "${RED}[FAIL]${NC} $*"
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
        "pkill -f UccPerfTest || pkill -f TwoNodeCollectiveTest || true" 2>/dev/null || true

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
        log_warning "No wake-on-lan tool found (wakeonlan or etherwake)"
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
            log_error "$host did not become reachable within ${max_wait}s"
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

    log "Testing connectivity from $from_host to $to_name ($to_ip)..."

    if ssh "$from_host" "ping -c 3 -W 2 '$to_ip'" 2>&1 | tee -a "$LOG_FILE"; then
        log_success "$from_host can reach $to_name via Mellanox ($to_ip)"
        return 0
    else
        log_error "$from_host cannot reach $to_name via Mellanox ($to_ip)"
        return 1
    fi
}

build_warpforge() {
    local host=$1

    log "Building warpforge on $host..."

    ssh "$host" bash -lc "
        cd '${REMOTE_REPO_DIR}'
        echo '[remote] Building warpforge-io...'
        ./gradlew :warpforge-io:compileJava :warpforge-core:compileJava --no-daemon -q 2>&1
        echo '[remote] Build complete'
    " 2>&1 | tee -a "$LOG_FILE"
}

start_perf_server() {
    local size_bytes=$1

    log "Starting UCC performance test server on $NVIDIA_HOST..."
    log "  Rank: 0, Master: ${NVIDIA_MELLANOX_IP}:${UCC_TEST_PORT}"
    log "  Size: $size_bytes bytes, Iterations: $ITERATIONS, Warmup: $WARMUP"

    ssh "$NVIDIA_HOST" bash -lc "
        cd '${REMOTE_REPO_DIR}'
        ./gradlew :warpforge-io:uccPerfTest \
            --args='--rank 0 --world-size 2 --master ${NVIDIA_MELLANOX_IP} --port ${UCC_TEST_PORT} --size ${size_bytes} --iterations ${ITERATIONS} --warmup ${WARMUP}' \
            --no-daemon 2>&1
    " 2>&1 | tee -a "$LOG_FILE" &

    SERVER_SSH_PID=$!
    log "Server SSH process started (PID: $SERVER_SSH_PID)"

    # Wait for server to be ready
    log "Waiting for server to initialize (15 seconds)..."
    sleep 15

    if ! kill -0 "$SERVER_SSH_PID" 2>/dev/null; then
        log_error "Server process died immediately"
        wait "$SERVER_SSH_PID" || true
        return 1
    fi

    log "Server appears to be running"
    return 0
}

run_perf_client() {
    local size_bytes=$1

    log "Starting UCC performance test client on $AMD_HOST..."
    log "  Rank: 1, Master: ${NVIDIA_MELLANOX_IP}:${UCC_TEST_PORT}"

    local output
    output=$(ssh "$AMD_HOST" bash -lc "
        cd '${REMOTE_REPO_DIR}'
        ./gradlew :warpforge-io:uccPerfTest \
            --args='--rank 1 --world-size 2 --master ${NVIDIA_MELLANOX_IP} --port ${UCC_TEST_PORT} --size ${size_bytes} --iterations ${ITERATIONS} --warmup ${WARMUP}' \
            --no-daemon 2>&1
    " 2>&1)

    echo "$output" | tee -a "$LOG_FILE"

    # Extract performance data from output
    echo "$output"
}

wait_for_server() {
    if [[ -n "$SERVER_SSH_PID" ]] && kill -0 "$SERVER_SSH_PID" 2>/dev/null; then
        log "Waiting for server to finish..."
        wait "$SERVER_SSH_PID" 2>/dev/null || true
    fi
    SERVER_SSH_PID=""
}

parse_results() {
    local output=$1
    local size_bytes=$2
    local op_name=$3

    # Extract bandwidth (Gbps) from output - adjust pattern based on Java test output
    local bw
    bw=$(echo "$output" | grep -i "${op_name}.*Gbps" | grep -oE '[0-9]+\.[0-9]+\s*Gbps' | head -1 | grep -oE '[0-9]+\.[0-9]+')

    # Extract latency (ms) from output
    local lat_ms
    lat_ms=$(echo "$output" | grep -i "${op_name}.*ms" | grep -oE '[0-9]+\.[0-9]+\s*ms' | head -1 | grep -oE '[0-9]+\.[0-9]+')

    if [[ -n "$bw" ]]; then
        echo "$size_bytes,$op_name,$bw,$lat_ms" >> "$RESULTS_FILE"
    fi
}

run_performance_tests() {
    log_section "RUNNING PERFORMANCE TESTS"

    # Initialize results CSV
    echo "size_bytes,operation,bandwidth_gbps,latency_ms" > "$RESULTS_FILE"

    # Convert comma-separated sizes to array
    IFS=',' read -ra SIZE_ARRAY <<< "$SIZES"

    for size in "${SIZE_ARRAY[@]}"; do
        log ""
        log "═══════════════════════════════════════════════════════"
        log "  Testing message size: $size bytes"
        log "═══════════════════════════════════════════════════════"

        # Start server
        if ! start_perf_server "$size"; then
            log_error "Failed to start server for size $size"
            continue
        fi

        # Give server extra time to be ready
        sleep 5

        # Run client and capture output
        set +e
        local output
        output=$(run_perf_client "$size")
        local client_status=$?
        set -e

        # Wait for server to complete
        wait_for_server

        if [[ $client_status -eq 0 ]]; then
            log_success "Test completed for size $size"

            # Parse and record results
            parse_results "$output" "$size" "AllReduce"
            parse_results "$output" "$size" "AllGather"
            parse_results "$output" "$size" "Broadcast"
            parse_results "$output" "$size" "ReduceScatter"
        else
            log_error "Test failed for size $size"
        fi

        # Brief pause between sizes
        sleep 2
    done
}

print_summary() {
    log_section "PERFORMANCE SUMMARY"

    if [[ -f "$RESULTS_FILE" ]]; then
        log ""
        log "Results saved to: $RESULTS_FILE"
        log ""

        # Print results table
        log "Operation         | Size        | Bandwidth   | Latency"
        log "------------------|-------------|-------------|----------"

        tail -n +2 "$RESULTS_FILE" | while IFS=',' read -r size op bw lat; do
            # Format size for display
            local size_display
            if (( size >= 1073741824 )); then
                size_display="$(echo "scale=1; $size / 1073741824" | bc)GB"
            elif (( size >= 1048576 )); then
                size_display="$(echo "scale=1; $size / 1048576" | bc)MB"
            elif (( size >= 1024 )); then
                size_display="$(echo "scale=1; $size / 1024" | bc)KB"
            else
                size_display="${size}B"
            fi

            printf "%-17s | %-11s | %-11s | %s\n" "$op" "$size_display" "${bw:-N/A} Gbps" "${lat:-N/A} ms"
        done | tee -a "$LOG_FILE"

        log ""

        # Calculate peak bandwidth
        local peak_bw
        peak_bw=$(tail -n +2 "$RESULTS_FILE" | cut -d',' -f3 | sort -rn | head -1)

        if [[ -n "$peak_bw" ]]; then
            log "Peak Bandwidth: ${peak_bw} Gbps"

            # Compare against RDMA baseline (100 Gbps target)
            local efficiency
            efficiency=$(echo "scale=1; $peak_bw * 100 / 100" | bc)
            log "Efficiency vs 100GbE line rate: ${efficiency}%"

            if (( $(echo "$peak_bw >= 80" | bc -l) )); then
                log_success "Performance is within acceptable range (>= 80 Gbps)"
            elif (( $(echo "$peak_bw >= 50" | bc -l) )); then
                log_warning "Performance is moderate (50-80 Gbps) - potential for improvement"
            else
                log_error "Performance is low (< 50 Gbps) - investigation needed"
            fi
        fi
    else
        log_warning "No results file found"
    fi
}

run_single_iteration() {
    local iteration_num=$1

    log_section "ITERATION $iteration_num"
    log "Starting at $(date)"

    # Phase 1: Check/wake boxes
    if [[ "$WAKE_BOXES" -eq 1 ]]; then
        log_section "PHASE 1: WAKE GPU BOXES"
        wake_host "$NVIDIA_MAC" "$NVIDIA_HOST"
        wake_host "$AMD_MAC" "$AMD_HOST"
    fi

    # Phase 2: Wait for SSH
    log_section "PHASE 2: WAIT FOR SSH"

    if ! wait_for_ssh "$NVIDIA_HOST" "$SSH_MAX_WAIT_SECONDS"; then
        log_error "NVIDIA box unreachable"
        return 1
    fi

    if ! wait_for_ssh "$AMD_HOST" "$SSH_MAX_WAIT_SECONDS"; then
        log_error "AMD box unreachable"
        return 1
    fi

    log_success "Both boxes are reachable via SSH"

    # Phase 3: Sync repos
    log_section "PHASE 3: SYNC REPOSITORIES"
    sync_repo "$NVIDIA_HOST"
    sync_repo "$AMD_HOST"

    # Phase 4: Build
    log_section "PHASE 4: BUILD WARPFORGE"
    build_warpforge "$NVIDIA_HOST" &
    local nvidia_build_pid=$!
    build_warpforge "$AMD_HOST" &
    local amd_build_pid=$!

    wait $nvidia_build_pid
    wait $amd_build_pid
    log_success "Build complete on both boxes"

    # Phase 5: Check connectivity
    log_section "PHASE 5: VERIFY MELLANOX CONNECTIVITY"

    if ! check_mellanox_connectivity "$NVIDIA_HOST" "$AMD_MELLANOX_IP" "AMD"; then
        return 1
    fi

    if ! check_mellanox_connectivity "$AMD_HOST" "$NVIDIA_MELLANOX_IP" "NVIDIA"; then
        return 1
    fi

    # Phase 6: Run performance tests
    run_performance_tests

    # Phase 7: Print summary
    print_summary

    log ""
    log "Iteration $iteration_num completed at $(date)"
    return 0
}

# ======== MAIN ========

log_section "UCC COLLECTIVE PERFORMANCE TEST"
log "Run ID: $RUN_ID"
log "Log file: $LOG_FILE"
log "Results file: $RESULTS_FILE"
log ""
log "Configuration:"
log "  NVIDIA host: $NVIDIA_HOST (Mellanox IP: $NVIDIA_MELLANOX_IP)"
log "  AMD host: $AMD_HOST (Mellanox IP: $AMD_MELLANOX_IP)"
log "  Branch: $BRANCH"
log "  Remote repo: $REMOTE_REPO_DIR"
log "  UCC test port: $UCC_TEST_PORT"
log ""
log "Test Parameters:"
log "  Iterations: $ITERATIONS"
log "  Warmup: $WARMUP"
log "  Message sizes: $SIZES"
log "  Quick mode: $QUICK_MODE"
log "  Continuous mode: $CONTINUOUS_MODE"

if [[ "$CONTINUOUS_MODE" -eq 1 ]]; then
    log ""
    log "═══════════════════════════════════════════════════════════"
    log "  CONTINUOUS MODE - Running until interrupted (Ctrl+C)"
    log "  Interval between runs: ${CONTINUOUS_INTERVAL}s"
    log "═══════════════════════════════════════════════════════════"

    iteration=1
    while true; do
        run_single_iteration $iteration || log_error "Iteration $iteration failed"

        log ""
        log "Sleeping ${CONTINUOUS_INTERVAL}s before next iteration..."
        log "Press Ctrl+C to stop"
        sleep "$CONTINUOUS_INTERVAL"

        iteration=$((iteration + 1))
    done
else
    run_single_iteration 1

    if [[ -f "$RESULTS_FILE" ]]; then
        log ""
        log "═══════════════════════════════════════════════════════════"
        log "  ALL TESTS COMPLETE"
        log "═══════════════════════════════════════════════════════════"
        exit 0
    else
        log_error "No results generated"
        exit 1
    fi
fi
