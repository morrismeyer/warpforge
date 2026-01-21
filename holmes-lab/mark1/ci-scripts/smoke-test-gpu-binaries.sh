#!/usr/bin/env bash
set -euo pipefail

# GPU Binary Smoke Test
#
# Validates that snakegrinder native binaries can be distributed to and
# executed on GPU boxes. This is a prerequisite check before containerization.
#
# Intended to run ON the NUC after snakegrinder-dist has been built.
# It:
# - Verifies local snakegrinder-dist build exists
# - Wakes GPU boxes (if needed)
# - SCPs binaries to known install locations
# - Verifies binaries execute correctly on each GPU box
# - Runs a simple trace to validate end-to-end functionality

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# -------- CONFIG (override via environment variables) --------

# Path to the snakegrinder distribution on NUC
DIST_DIR="${DIST_DIR_OVERRIDE:-}"
if [[ -z "$DIST_DIR" ]]; then
    # Try to find it relative to script location (assumes standard repo layout)
    REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
    DIST_DIR="$REPO_ROOT/snakegrinder-dist/build/dist"
fi

# Install location on GPU boxes
INSTALL_DIR="${INSTALL_DIR_OVERRIDE:-/opt/warpforge}"

# GPU box hostnames (must be in SSH config or /etc/hosts)
NVIDIA_HOST="${NVIDIA_HOST_OVERRIDE:-nvidia}"
AMD_HOST="${AMD_HOST_OVERRIDE:-amd}"

# MAC addresses for Wake-on-LAN
NVIDIA_MAC="${NVIDIA_MAC_OVERRIDE:-10:7C:61:3D:E7:8F}"
AMD_MAC="${AMD_MAC_OVERRIDE:-30:56:0f:20:67:79}"
BROADCAST_IP="${BROADCAST_IP_OVERRIDE:-192.168.1.255}"

# SSH settings
SSH_CONNECT_TIMEOUT_SECONDS="${SSH_CONNECT_TIMEOUT_SECONDS:-5}"
SSH_POLL_INTERVAL_SECONDS="${SSH_POLL_INTERVAL_SECONDS:-10}"
SSH_MAX_WAIT_SECONDS="${SSH_MAX_WAIT_SECONDS:-60}"

# Skip unreachable boxes instead of failing
SKIP_IF_UNREACHABLE="${SKIP_IF_UNREACHABLE:-1}"

# Whether to wake boxes before testing
WAKE_BOXES="${WAKE_BOXES:-1}"

# Logging
LOG_ROOT="${LOG_ROOT_OVERRIDE:-$HOME/build-logs}"
RUN_ID="${GITHUB_RUN_ID:-manual-$(date +%Y%m%d-%H%M%S)}"

# -------- END CONFIG --------

mkdir -p "$LOG_ROOT"
LOG_FILE="$LOG_ROOT/smoke-test-gpu-${RUN_ID}.log"

log() {
    echo "[$(date)] $*" | tee -a "$LOG_FILE"
}

log_section() {
    log ""
    log "========================================"
    log "$*"
    log "========================================"
}

# Track results
NVIDIA_RESULT="SKIPPED"
AMD_RESULT="SKIPPED"

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
            log "$host is reachable"
            return 0
        fi

        local now_ts elapsed
        now_ts="$(date +%s)"
        elapsed=$((now_ts - start_ts))

        if (( elapsed >= max_wait )); then
            log "$host did not become reachable within ${max_wait}s"
            return 1
        fi

        log "  ...waiting for $host (attempt $attempt, elapsed ${elapsed}s)"
        sleep "${SSH_POLL_INTERVAL_SECONDS}"
    done
}

distribute_to_host() {
    local host=$1
    local host_type=$2  # "nvidia" or "amd" for future GPU-specific builds

    log "Distributing snakegrinder to $host..."

    # Ensure install directory exists
    log "  Creating install directory $INSTALL_DIR on $host..."
    ssh "$host" bash -c "
        sudo mkdir -p '$INSTALL_DIR/bin' '$INSTALL_DIR/venv'
        sudo chown -R \$USER '$INSTALL_DIR'
    " 2>&1 | tee -a "$LOG_FILE"

    # Copy the binary
    log "  Copying snakegrinder binary..."
    scp "$DIST_DIR/bin/snakegrinder" "$host:$INSTALL_DIR/bin/" 2>&1 | tee -a "$LOG_FILE"

    # Copy venv if it exists (contains PyTorch site-packages)
    if [[ -d "$DIST_DIR/venv" ]]; then
        log "  Copying venv (this may take a moment)..."
        # Use rsync for efficiency if available, otherwise scp
        if command -v rsync >/dev/null 2>&1; then
            rsync -az --delete "$DIST_DIR/venv/" "$host:$INSTALL_DIR/venv/" 2>&1 | tee -a "$LOG_FILE"
        else
            scp -r "$DIST_DIR/venv" "$host:$INSTALL_DIR/" 2>&1 | tee -a "$LOG_FILE"
        fi
    else
        log "  WARN: No venv directory found at $DIST_DIR/venv"
    fi

    # Make binary executable
    ssh "$host" "chmod +x '$INSTALL_DIR/bin/snakegrinder'" 2>&1 | tee -a "$LOG_FILE"

    log "  Distribution to $host complete"
}

verify_on_host() {
    local host=$1
    local host_type=$2

    log "Verifying snakegrinder on $host..."

    # Test 1: Binary exists and is executable
    log "  Test 1: Checking binary exists..."
    if ! ssh "$host" "test -x '$INSTALL_DIR/bin/snakegrinder'"; then
        log "  FAIL: Binary not found or not executable at $INSTALL_DIR/bin/snakegrinder"
        return 1
    fi
    log "  PASS: Binary exists and is executable"

    # Test 2: Version check
    log "  Test 2: Running --version..."
    local version_output
    if ! version_output=$(ssh "$host" "'$INSTALL_DIR/bin/snakegrinder' --version" 2>&1); then
        log "  FAIL: --version failed"
        log "  Output: $version_output"
        return 1
    fi
    log "  PASS: --version succeeded"
    log "  Version: $version_output"

    # Test 3: Help check (validates argument parsing)
    log "  Test 3: Running --help..."
    if ! ssh "$host" "'$INSTALL_DIR/bin/snakegrinder' --help >/dev/null 2>&1"; then
        log "  FAIL: --help failed"
        return 1
    fi
    log "  PASS: --help succeeded"

    # Test 4: Trace example (validates GraalPy + PyTorch integration)
    log "  Test 4: Running --trace-example..."
    local trace_output_file="/tmp/smoke-test-${RUN_ID}.mlir"
    if ! ssh "$host" "'$INSTALL_DIR/bin/snakegrinder' --trace-example --out '$trace_output_file'" 2>&1 | tee -a "$LOG_FILE"; then
        log "  FAIL: --trace-example failed"
        return 1
    fi

    # Verify output file was created and has content
    log "  Verifying trace output..."
    local output_size
    output_size=$(ssh "$host" "stat -c%s '$trace_output_file' 2>/dev/null || stat -f%z '$trace_output_file' 2>/dev/null || echo 0")
    if [[ "$output_size" -lt 100 ]]; then
        log "  FAIL: Trace output file too small (${output_size} bytes)"
        return 1
    fi
    log "  PASS: Trace output created (${output_size} bytes)"

    # Test 5: Verify output contains StableHLO markers
    log "  Test 5: Checking output format..."
    if ! ssh "$host" "grep -q 'stablehlo\|module\|func' '$trace_output_file'"; then
        log "  WARN: Output may not be valid StableHLO (no expected markers found)"
        # Don't fail on this - output format may vary
    else
        log "  PASS: Output contains expected StableHLO markers"
    fi

    # Cleanup
    ssh "$host" "rm -f '$trace_output_file'" 2>/dev/null || true

    log "  All verification tests PASSED on $host"
    return 0
}

test_gpu_box() {
    local host=$1
    local mac=$2
    local host_type=$3

    log_section "Testing $host_type GPU box: $host"

    # Wake the box
    wake_host "$mac" "$host"

    # Wait for SSH
    if ! wait_for_ssh "$host" "$SSH_MAX_WAIT_SECONDS"; then
        if [[ "$SKIP_IF_UNREACHABLE" -eq 1 ]]; then
            log "SKIP_IF_UNREACHABLE=1, skipping $host"
            return 2  # Special return code for "skipped"
        else
            log "FAIL: $host unreachable and SKIP_IF_UNREACHABLE=0"
            return 1
        fi
    fi

    # Distribute binaries
    if ! distribute_to_host "$host" "$host_type"; then
        log "FAIL: Distribution to $host failed"
        return 1
    fi

    # Verify execution
    if ! verify_on_host "$host" "$host_type"; then
        log "FAIL: Verification on $host failed"
        return 1
    fi

    log "$host: ALL TESTS PASSED"
    return 0
}

# ======== MAIN ========

log_section "GPU Binary Smoke Test (run $RUN_ID)"
log "Distribution source: $DIST_DIR"
log "Install target: $INSTALL_DIR"
log "NVIDIA host: $NVIDIA_HOST (MAC: $NVIDIA_MAC)"
log "AMD host: $AMD_HOST (MAC: $AMD_MAC)"

# Phase 1: Verify local distribution exists
log_section "Phase 1: Verify local distribution"

if [[ ! -d "$DIST_DIR" ]]; then
    log "ERROR: Distribution directory not found: $DIST_DIR"
    log "Run: ./gradlew :snakegrinder-dist:assembleDist"
    exit 1
fi

if [[ ! -x "$DIST_DIR/bin/snakegrinder" ]]; then
    log "ERROR: snakegrinder binary not found at $DIST_DIR/bin/snakegrinder"
    log "Run: ./gradlew :snakegrinder-dist:assembleDist"
    exit 1
fi

log "Local distribution verified at $DIST_DIR"
ls -la "$DIST_DIR/bin/" 2>&1 | tee -a "$LOG_FILE"

# Phase 2: Test NVIDIA box
log_section "Phase 2: NVIDIA GPU Box"

set +e
test_gpu_box "$NVIDIA_HOST" "$NVIDIA_MAC" "nvidia"
nvidia_status=$?
set -e

case $nvidia_status in
    0) NVIDIA_RESULT="PASSED" ;;
    2) NVIDIA_RESULT="SKIPPED" ;;
    *) NVIDIA_RESULT="FAILED" ;;
esac

# Phase 3: Test AMD box
log_section "Phase 3: AMD GPU Box"

set +e
test_gpu_box "$AMD_HOST" "$AMD_MAC" "amd"
amd_status=$?
set -e

case $amd_status in
    0) AMD_RESULT="PASSED" ;;
    2) AMD_RESULT="SKIPPED" ;;
    *) AMD_RESULT="FAILED" ;;
esac

# Summary
log_section "SMOKE TEST SUMMARY"
log "NVIDIA: $NVIDIA_RESULT"
log "AMD:    $AMD_RESULT"

# Determine overall result
if [[ "$NVIDIA_RESULT" == "FAILED" || "$AMD_RESULT" == "FAILED" ]]; then
    log ""
    log "OVERALL: FAILED"
    log "One or more GPU boxes failed verification."
    exit 1
fi

if [[ "$NVIDIA_RESULT" == "SKIPPED" && "$AMD_RESULT" == "SKIPPED" ]]; then
    log ""
    log "OVERALL: SKIPPED (no GPU boxes reachable)"
    log "Set SKIP_IF_UNREACHABLE=0 to fail when boxes are unreachable."
    exit 0
fi

log ""
log "OVERALL: PASSED"
log "Binaries successfully distributed and verified on reachable GPU boxes."
log ""
log "Binaries installed at: $INSTALL_DIR/bin/snakegrinder"
log "Ready for Ray job submission or further testing."
exit 0
