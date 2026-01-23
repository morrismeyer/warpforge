#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# UCX Throughput Validation Script
# =============================================================================
#
# Validates UCX performance as a baseline for WarpForge UCC operations.
# Compares against raw RDMA verbs (perftest) to measure UCX overhead.
#
# This establishes the UCX-layer baseline that warpforge-io is built on.
#
# Prerequisites:
#   - UCX libraries installed with ucx_perftest
#   - RDMA-capable NIC (Mellanox ConnectX-5 or similar)
#   - Two nodes connected via InfiniBand/RoCE
#
# Usage:
#   # On server node:
#   ./validate-ucx-throughput.sh server
#
#   # On client node:
#   ./validate-ucx-throughput.sh client <server-ip>
#
#   # Compare UCX vs raw verbs:
#   ./validate-ucx-throughput.sh compare <server-ip>
#
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
MESSAGE_SIZE="${UCX_MSG_SIZE:-1048576}"        # 1MB default
ITERATIONS="${UCX_ITERATIONS:-10000}"
WARMUP="${UCX_WARMUP:-1000}"
OUTSTANDING="${UCX_OUTSTANDING:-16}"           # Outstanding operations for pipelining

# UCX transport selection
# Options: rc_mlx5, rc_verbs, ud_mlx5, ud_verbs, dc_mlx5
UCX_TRANSPORT="${UCX_TRANSPORT:-rc_mlx5}"

# Target performance (as percentage of raw verbs baseline)
RAW_BASELINE_GBPS="${RAW_BASELINE_GBPS:-95}"   # Expected raw verbs performance
UCX_TARGET_PERCENT="${UCX_TARGET_PERCENT:-90}" # UCX should achieve 90% of raw

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $*"
}

log_error() {
    echo -e "${RED}[✗]${NC} $*"
}

log_section() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $*${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
}

# =============================================================================
# Prerequisite Checks
# =============================================================================

check_prerequisites() {
    log "Checking UCX prerequisites..."

    # Check for ucx_perftest
    if ! command -v ucx_perftest &> /dev/null; then
        log_error "ucx_perftest not found"
        log "UCX may not be installed or not in PATH"
        log ""
        log "Install UCX from source:"
        log "  git clone https://github.com/openucx/ucx.git"
        log "  cd ucx && ./autogen.sh && ./configure --prefix=/usr/local"
        log "  make -j && sudo make install"
        log ""
        log "Or install from package:"
        log "  sudo apt install ucx-utils"
        exit 1
    fi
    log_success "ucx_perftest found: $(which ucx_perftest)"

    # Show UCX version
    local ucx_version
    ucx_version=$(ucx_perftest -v 2>&1 | head -1 || echo "unknown")
    log "UCX version: $ucx_version"

    # Check for ucx_info
    if command -v ucx_info &> /dev/null; then
        log ""
        log "Available UCX transports:"
        ucx_info -d 2>/dev/null | grep -E "Transport:|Device:" | head -20 || true
    fi

    # Check for RDMA devices
    if command -v ibv_devices &> /dev/null; then
        log ""
        log "RDMA devices:"
        ibv_devices 2>/dev/null || true
    fi
}

# =============================================================================
# UCX Performance Tests
# =============================================================================

# Test UCT layer (lowest UCX layer, closest to hardware)
run_uct_bandwidth() {
    local server_ip="$1"
    local test_name="$2"
    local test_type="$3"

    log_section "UCT Layer: $test_name"
    log "Transport: $UCX_TRANSPORT"
    log "Message size: $MESSAGE_SIZE bytes"
    log "Iterations: $ITERATIONS"
    log ""

    # UCT layer test with specified transport
    local output
    output=$(ucx_perftest "$server_ip" \
        -t "$test_type" \
        -s "$MESSAGE_SIZE" \
        -n "$ITERATIONS" \
        -w "$WARMUP" \
        -O "$OUTSTANDING" \
        -x "$UCX_TRANSPORT" \
        -D zcopy \
        2>&1) || true

    echo "$output"

    # Parse bandwidth from output
    local bw
    bw=$(echo "$output" | grep -E "^\s*$MESSAGE_SIZE\s+" | awk '{print $NF}' | tail -1)

    if [[ -n "$bw" ]]; then
        log_success "$test_name: ${bw} Gbps (UCT layer)"
        echo "$bw"
    else
        log_warning "Could not parse bandwidth from $test_name"
        echo "0"
    fi
}

# Test UCP layer (higher-level UCX API used by UCC)
run_ucp_bandwidth() {
    local server_ip="$1"
    local test_name="$2"
    local test_type="$3"

    log_section "UCP Layer: $test_name"
    log "Message size: $MESSAGE_SIZE bytes"
    log "Iterations: $ITERATIONS"
    log ""

    # Set transport via environment for UCP tests
    local output
    output=$(UCX_TLS="$UCX_TRANSPORT" ucx_perftest "$server_ip" \
        -t "$test_type" \
        -s "$MESSAGE_SIZE" \
        -n "$ITERATIONS" \
        -w "$WARMUP" \
        -O "$OUTSTANDING" \
        -D zcopy \
        2>&1) || true

    echo "$output"

    # Parse bandwidth
    local bw
    bw=$(echo "$output" | grep -E "^\s*$MESSAGE_SIZE\s+" | awk '{print $NF}' | tail -1)

    if [[ -n "$bw" ]]; then
        log_success "$test_name: ${bw} Gbps (UCP layer)"
        echo "$bw"
    else
        log_warning "Could not parse bandwidth from $test_name"
        echo "0"
    fi
}

# =============================================================================
# Server Mode
# =============================================================================

run_server() {
    log_section "UCX Performance Test Server"
    log "Waiting for client connections..."
    log ""
    log "Client should run: $0 client <this-ip>"
    log ""

    # Run ucx_perftest in server mode (no hostname = server)
    UCX_TLS="$UCX_TRANSPORT" ucx_perftest \
        -t put_bw \
        -s "$MESSAGE_SIZE" \
        -n "$ITERATIONS" \
        -w "$WARMUP" \
        -O "$OUTSTANDING"
}

# =============================================================================
# Client Mode - Comprehensive Tests
# =============================================================================

run_client() {
    local server_ip="$1"

    log_section "UCX Performance Test Client"
    log "Server: $server_ip"
    log "Transport: $UCX_TRANSPORT"
    log "Message size: $MESSAGE_SIZE bytes"
    log "Iterations: $ITERATIONS"
    log "Outstanding ops: $OUTSTANDING"
    log ""

    local results_file
    results_file=$(mktemp)

    # =========================================================================
    # UCP Layer Tests (what UCC uses)
    # =========================================================================

    log_section "UCP LAYER TESTS (Used by UCC)"

    # PUT bandwidth (one-sided RDMA write)
    log ""
    log "--- PUT Bandwidth (RDMA Write) ---"
    local put_bw
    put_bw=$(UCX_TLS="$UCX_TRANSPORT" ucx_perftest "$server_ip" \
        -t put_bw \
        -s "$MESSAGE_SIZE" \
        -n "$ITERATIONS" \
        -w "$WARMUP" \
        -O "$OUTSTANDING" \
        -D zcopy \
        2>&1 | tee /dev/stderr | grep -E "^\s*$MESSAGE_SIZE\s+" | awk '{print $NF}' | tail -1) || put_bw="0"
    echo "UCP_PUT_BW=$put_bw" >> "$results_file"

    sleep 1

    # GET bandwidth (one-sided RDMA read)
    log ""
    log "--- GET Bandwidth (RDMA Read) ---"
    local get_bw
    get_bw=$(UCX_TLS="$UCX_TRANSPORT" ucx_perftest "$server_ip" \
        -t get_bw \
        -s "$MESSAGE_SIZE" \
        -n "$ITERATIONS" \
        -w "$WARMUP" \
        -O "$OUTSTANDING" \
        -D zcopy \
        2>&1 | tee /dev/stderr | grep -E "^\s*$MESSAGE_SIZE\s+" | awk '{print $NF}' | tail -1) || get_bw="0"
    echo "UCP_GET_BW=$get_bw" >> "$results_file"

    sleep 1

    # Active Message bandwidth (what UCC collectives use)
    log ""
    log "--- Active Message Bandwidth ---"
    local am_bw
    am_bw=$(UCX_TLS="$UCX_TRANSPORT" ucx_perftest "$server_ip" \
        -t am_bw \
        -s "$MESSAGE_SIZE" \
        -n "$ITERATIONS" \
        -w "$WARMUP" \
        -O "$OUTSTANDING" \
        -D zcopy \
        2>&1 | tee /dev/stderr | grep -E "^\s*$MESSAGE_SIZE\s+" | awk '{print $NF}' | tail -1) || am_bw="0"
    echo "UCP_AM_BW=$am_bw" >> "$results_file"

    sleep 1

    # Tag matching bandwidth (MPI-like send/recv)
    log ""
    log "--- Tag Matching Bandwidth ---"
    local tag_bw
    tag_bw=$(UCX_TLS="$UCX_TRANSPORT" ucx_perftest "$server_ip" \
        -t tag_bw \
        -s "$MESSAGE_SIZE" \
        -n "$ITERATIONS" \
        -w "$WARMUP" \
        -O "$OUTSTANDING" \
        -D zcopy \
        2>&1 | tee /dev/stderr | grep -E "^\s*$MESSAGE_SIZE\s+" | awk '{print $NF}' | tail -1) || tag_bw="0"
    echo "UCP_TAG_BW=$tag_bw" >> "$results_file"

    # =========================================================================
    # Summary
    # =========================================================================

    log_section "UCX PERFORMANCE SUMMARY"

    cat "$results_file"
    echo ""

    # Find peak bandwidth
    local peak_bw=0
    while IFS='=' read -r key value; do
        if (( $(echo "$value > $peak_bw" | bc -l 2>/dev/null || echo 0) )); then
            peak_bw="$value"
        fi
    done < "$results_file"

    log "Peak UCX bandwidth: ${peak_bw} Gbps"

    # Compare against expected raw verbs baseline
    local expected_ucx
    expected_ucx=$(echo "scale=1; $RAW_BASELINE_GBPS * $UCX_TARGET_PERCENT / 100" | bc)

    log "Expected (${UCX_TARGET_PERCENT}% of ${RAW_BASELINE_GBPS} Gbps raw): ${expected_ucx} Gbps"

    if (( $(echo "$peak_bw >= $expected_ucx" | bc -l 2>/dev/null || echo 0) )); then
        log_success "UCX performance meets target (>= ${expected_ucx} Gbps)"
    else
        log_warning "UCX performance below target (< ${expected_ucx} Gbps)"
        log "This may indicate configuration or hardware issues"
    fi

    # Calculate overhead vs raw verbs
    if (( $(echo "$peak_bw > 0" | bc -l 2>/dev/null || echo 0) )); then
        local overhead
        overhead=$(echo "scale=1; (1 - $peak_bw / $RAW_BASELINE_GBPS) * 100" | bc)
        log "UCX overhead vs raw verbs: ${overhead}%"
    fi

    rm -f "$results_file"
}

# =============================================================================
# Compare UCX vs Raw Verbs
# =============================================================================

run_comparison() {
    local server_ip="$1"

    log_section "UCX vs Raw Verbs Comparison"
    log "This test compares UCX performance against raw InfiniBand verbs"
    log ""

    # Check if perftest tools are available
    if ! command -v ib_write_bw &> /dev/null; then
        log_warning "perftest tools not installed, skipping raw verbs test"
        log "Install with: sudo apt install perftest"
        run_client "$server_ip"
        return
    fi

    local results_file
    results_file=$(mktemp)

    # -------------------------------------------------------------------------
    # Raw Verbs Test (ib_write_bw)
    # -------------------------------------------------------------------------
    log_section "RAW VERBS BASELINE (ib_write_bw)"

    local raw_bw
    raw_bw=$(ib_write_bw \
        --size="$MESSAGE_SIZE" \
        --iters="$ITERATIONS" \
        --report_gbits \
        "$server_ip" 2>&1 | tee /dev/stderr | grep -E "^\s*$MESSAGE_SIZE\s+" | awk '{print $4}') || raw_bw="0"

    echo "RAW_WRITE_BW=$raw_bw" >> "$results_file"
    log_success "Raw RDMA Write: ${raw_bw} Gbps"

    sleep 2

    # -------------------------------------------------------------------------
    # UCX UCP PUT Test
    # -------------------------------------------------------------------------
    log_section "UCX UCP PUT BANDWIDTH"

    local ucx_put_bw
    ucx_put_bw=$(UCX_TLS="$UCX_TRANSPORT" ucx_perftest "$server_ip" \
        -t put_bw \
        -s "$MESSAGE_SIZE" \
        -n "$ITERATIONS" \
        -w "$WARMUP" \
        -O "$OUTSTANDING" \
        -D zcopy \
        2>&1 | tee /dev/stderr | grep -E "^\s*$MESSAGE_SIZE\s+" | awk '{print $NF}' | tail -1) || ucx_put_bw="0"

    echo "UCX_PUT_BW=$ucx_put_bw" >> "$results_file"
    log_success "UCX PUT: ${ucx_put_bw} Gbps"

    sleep 2

    # -------------------------------------------------------------------------
    # UCX UCP Active Message Test
    # -------------------------------------------------------------------------
    log_section "UCX UCP ACTIVE MESSAGE BANDWIDTH"

    local ucx_am_bw
    ucx_am_bw=$(UCX_TLS="$UCX_TRANSPORT" ucx_perftest "$server_ip" \
        -t am_bw \
        -s "$MESSAGE_SIZE" \
        -n "$ITERATIONS" \
        -w "$WARMUP" \
        -O "$OUTSTANDING" \
        -D zcopy \
        2>&1 | tee /dev/stderr | grep -E "^\s*$MESSAGE_SIZE\s+" | awk '{print $NF}' | tail -1) || ucx_am_bw="0"

    echo "UCX_AM_BW=$ucx_am_bw" >> "$results_file"
    log_success "UCX Active Message: ${ucx_am_bw} Gbps"

    # -------------------------------------------------------------------------
    # Comparison Summary
    # -------------------------------------------------------------------------
    log_section "PERFORMANCE COMPARISON"

    echo ""
    echo "Results:"
    cat "$results_file"
    echo ""

    if [[ -n "$raw_bw" && "$raw_bw" != "0" ]]; then
        local put_efficiency am_efficiency
        put_efficiency=$(echo "scale=1; $ucx_put_bw * 100 / $raw_bw" | bc 2>/dev/null || echo "N/A")
        am_efficiency=$(echo "scale=1; $ucx_am_bw * 100 / $raw_bw" | bc 2>/dev/null || echo "N/A")

        echo ""
        echo "Efficiency vs Raw Verbs:"
        echo "  Raw RDMA Write:     ${raw_bw} Gbps (baseline)"
        echo "  UCX PUT:            ${ucx_put_bw} Gbps (${put_efficiency}% of raw)"
        echo "  UCX Active Message: ${ucx_am_bw} Gbps (${am_efficiency}% of raw)"
        echo ""

        # UCX overhead analysis
        local put_overhead am_overhead
        put_overhead=$(echo "scale=1; 100 - $put_efficiency" | bc 2>/dev/null || echo "N/A")
        am_overhead=$(echo "scale=1; 100 - $am_efficiency" | bc 2>/dev/null || echo "N/A")

        echo "UCX Overhead:"
        echo "  PUT overhead:            ${put_overhead}%"
        echo "  Active Message overhead: ${am_overhead}%"
        echo ""

        if (( $(echo "$put_efficiency >= 90" | bc -l 2>/dev/null || echo 0) )); then
            log_success "UCX PUT achieves >= 90% of raw verbs performance"
        else
            log_warning "UCX PUT below 90% of raw verbs - investigate configuration"
        fi
    fi

    rm -f "$results_file"
}

# =============================================================================
# Latency Tests
# =============================================================================

run_latency() {
    local server_ip="$1"

    log_section "UCX Latency Tests"

    # Small message latency
    log ""
    log "--- PUT Latency (8 bytes) ---"
    UCX_TLS="$UCX_TRANSPORT" ucx_perftest "$server_ip" \
        -t put_lat \
        -s 8 \
        -n 10000 \
        -w 1000 \
        2>&1 | tail -5

    log ""
    log "--- Active Message Latency (8 bytes) ---"
    UCX_TLS="$UCX_TRANSPORT" ucx_perftest "$server_ip" \
        -t am_lat \
        -s 8 \
        -n 10000 \
        -w 1000 \
        2>&1 | tail -5
}

# =============================================================================
# Show UCX Configuration
# =============================================================================

show_config() {
    log_section "UCX Configuration"

    if command -v ucx_info &> /dev/null; then
        log "UCX Build Configuration:"
        ucx_info -v 2>&1 | head -5

        log ""
        log "Available Transports:"
        ucx_info -d 2>&1 | grep -E "^#|Transport:" | head -30

        log ""
        log "Available Devices:"
        ucx_info -d 2>&1 | grep -E "Device:" | head -20
    else
        log_warning "ucx_info not found"
    fi

    log ""
    log "Current UCX Environment:"
    env | grep -E "^UCX_" | sort || echo "No UCX environment variables set"
}

# =============================================================================
# Usage
# =============================================================================

usage() {
    cat << EOF
UCX Throughput Validation Script

Usage: $0 <command> [options]

Commands:
    server              Start server mode (wait for client)
    client <server-ip>  Run UCX bandwidth tests against server
    compare <server-ip> Compare UCX vs raw verbs performance
    latency <server-ip> Run latency tests
    config              Show UCX configuration

Environment Variables:
    UCX_MSG_SIZE        Message size in bytes (default: 1048576 = 1MB)
    UCX_ITERATIONS      Number of iterations (default: 10000)
    UCX_WARMUP          Warmup iterations (default: 1000)
    UCX_OUTSTANDING     Outstanding operations (default: 16)
    UCX_TRANSPORT       Transport to use (default: rc_mlx5)
                        Options: rc_mlx5, rc_verbs, ud_mlx5, dc_mlx5

Examples:
    # On server node:
    $0 server

    # On client node (basic test):
    $0 client 192.168.2.1

    # Compare UCX vs raw verbs:
    $0 compare 192.168.2.1

    # Test with specific transport:
    UCX_TRANSPORT=dc_mlx5 $0 client 192.168.2.1

    # Test larger messages:
    UCX_MSG_SIZE=16777216 $0 client 192.168.2.1
EOF
}

# =============================================================================
# Main
# =============================================================================

main() {
    if [[ $# -lt 1 ]]; then
        usage
        exit 1
    fi

    local command="$1"
    shift

    case "$command" in
        server)
            check_prerequisites
            run_server
            ;;
        client)
            if [[ $# -lt 1 ]]; then
                log_error "Client mode requires server IP"
                usage
                exit 1
            fi
            check_prerequisites
            run_client "$1"
            ;;
        compare)
            if [[ $# -lt 1 ]]; then
                log_error "Compare mode requires server IP"
                usage
                exit 1
            fi
            check_prerequisites
            run_comparison "$1"
            ;;
        latency)
            if [[ $# -lt 1 ]]; then
                log_error "Latency mode requires server IP"
                usage
                exit 1
            fi
            check_prerequisites
            run_latency "$1"
            ;;
        config)
            check_prerequisites
            show_config
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

main "$@"
