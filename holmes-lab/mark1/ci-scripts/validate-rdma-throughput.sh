#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# RDMA Throughput Validation Script
# =============================================================================
#
# Validates that Mellanox 100GbE hardware can achieve near line-rate throughput
# using native Linux tools (perftest package).
#
# This establishes the baseline BEFORE attempting Java-level optimizations.
#
# Prerequisites:
#   - perftest package: sudo apt install perftest
#   - RDMA-capable NIC (Mellanox ConnectX-5 or similar)
#   - Two nodes connected via InfiniBand/RoCE
#
# Usage:
#   # On server node:
#   ./validate-rdma-throughput.sh server
#
#   # On client node:
#   ./validate-rdma-throughput.sh client <server-ip>
#
#   # Run full validation between two nodes (from orchestrator):
#   ./validate-rdma-throughput.sh full <server-ip> <client-ssh>
#
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
MESSAGE_SIZE="${RDMA_MSG_SIZE:-1048576}"      # 1MB default
ITERATIONS="${RDMA_ITERATIONS:-5000}"
TARGET_GBPS="${RDMA_TARGET_GBPS:-95}"         # Target: 95 Gbps on 100GbE
MIN_ACCEPTABLE_GBPS="${RDMA_MIN_GBPS:-90}"    # Fail if below 90 Gbps

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# =============================================================================
# Prerequisite Checks
# =============================================================================

check_prerequisites() {
    log "Checking prerequisites..."

    # Check for perftest tools
    local tools=("ib_write_bw" "ib_read_bw" "ib_send_bw" "ib_write_lat")
    local missing=()

    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing+=("$tool")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing perftest tools: ${missing[*]}"
        log "Install with: sudo apt install perftest"
        exit 1
    fi
    log_success "perftest tools available"

    # Check for RDMA devices
    if ! command -v ibv_devices &> /dev/null; then
        log_error "ibv_devices not found. Install: sudo apt install ibverbs-utils"
        exit 1
    fi

    local devices
    devices=$(ibv_devices 2>/dev/null | grep -v "device" | awk '{print $1}' | grep -v "^$" || true)

    if [[ -z "$devices" ]]; then
        log_error "No RDMA devices found!"
        log "Check: lspci | grep -i mellanox"
        log "Check: ibstat"
        exit 1
    fi

    log_success "RDMA devices found:"
    ibv_devices 2>/dev/null | head -10

    # Show device details
    log "Device details:"
    ibstat 2>/dev/null | head -30 || true
}

# =============================================================================
# Server Mode
# =============================================================================

run_server() {
    log "Starting RDMA server mode..."
    log "Waiting for client connections on all tests..."

    # Run servers for each test type
    # These will wait for client connection

    log ""
    log "=========================================="
    log "  RDMA WRITE BANDWIDTH SERVER"
    log "=========================================="
    log "Run on client: ib_write_bw --size=$MESSAGE_SIZE --iters=$ITERATIONS --report_gbits <this-ip>"
    ib_write_bw --size="$MESSAGE_SIZE" --iters="$ITERATIONS" --report_gbits

    log ""
    log "=========================================="
    log "  RDMA READ BANDWIDTH SERVER"
    log "=========================================="
    log "Run on client: ib_read_bw --size=$MESSAGE_SIZE --iters=$ITERATIONS --report_gbits <this-ip>"
    ib_read_bw --size="$MESSAGE_SIZE" --iters="$ITERATIONS" --report_gbits

    log ""
    log "=========================================="
    log "  SEND BANDWIDTH SERVER"
    log "=========================================="
    log "Run on client: ib_send_bw --size=$MESSAGE_SIZE --iters=$ITERATIONS --report_gbits <this-ip>"
    ib_send_bw --size="$MESSAGE_SIZE" --iters="$ITERATIONS" --report_gbits

    log ""
    log "=========================================="
    log "  WRITE LATENCY SERVER"
    log "=========================================="
    log "Run on client: ib_write_lat --size=8 --iters=10000 <this-ip>"
    ib_write_lat --size=8 --iters=10000
}

# =============================================================================
# Client Mode - Run Tests and Validate Results
# =============================================================================

run_client() {
    local server_ip="$1"

    log "Starting RDMA client mode..."
    log "Server: $server_ip"
    log "Message size: $MESSAGE_SIZE bytes"
    log "Iterations: $ITERATIONS"
    log "Target: ${TARGET_GBPS} Gbps"
    log ""

    local results_file
    results_file=$(mktemp)
    local all_passed=true

    # -------------------------------------------------------------------------
    # Test 1: RDMA Write Bandwidth
    # -------------------------------------------------------------------------
    log "=========================================="
    log "  TEST 1: RDMA WRITE BANDWIDTH"
    log "=========================================="

    local write_output
    write_output=$(ib_write_bw --size="$MESSAGE_SIZE" --iters="$ITERATIONS" --report_gbits "$server_ip" 2>&1)
    echo "$write_output"

    local write_bw
    write_bw=$(echo "$write_output" | grep -E "^\s*$MESSAGE_SIZE" | awk '{print $4}')

    if [[ -n "$write_bw" ]]; then
        echo "WRITE_BW=$write_bw" >> "$results_file"
        if (( $(echo "$write_bw >= $MIN_ACCEPTABLE_GBPS" | bc -l) )); then
            log_success "RDMA Write: ${write_bw} Gbps"
        else
            log_error "RDMA Write: ${write_bw} Gbps (below ${MIN_ACCEPTABLE_GBPS} Gbps minimum)"
            all_passed=false
        fi
    else
        log_error "Failed to parse RDMA Write bandwidth"
        all_passed=false
    fi

    sleep 2  # Brief pause between tests

    # -------------------------------------------------------------------------
    # Test 2: RDMA Read Bandwidth
    # -------------------------------------------------------------------------
    log ""
    log "=========================================="
    log "  TEST 2: RDMA READ BANDWIDTH"
    log "=========================================="

    local read_output
    read_output=$(ib_read_bw --size="$MESSAGE_SIZE" --iters="$ITERATIONS" --report_gbits "$server_ip" 2>&1)
    echo "$read_output"

    local read_bw
    read_bw=$(echo "$read_output" | grep -E "^\s*$MESSAGE_SIZE" | awk '{print $4}')

    if [[ -n "$read_bw" ]]; then
        echo "READ_BW=$read_bw" >> "$results_file"
        if (( $(echo "$read_bw >= $MIN_ACCEPTABLE_GBPS" | bc -l) )); then
            log_success "RDMA Read: ${read_bw} Gbps"
        else
            log_error "RDMA Read: ${read_bw} Gbps (below ${MIN_ACCEPTABLE_GBPS} Gbps minimum)"
            all_passed=false
        fi
    else
        log_error "Failed to parse RDMA Read bandwidth"
        all_passed=false
    fi

    sleep 2

    # -------------------------------------------------------------------------
    # Test 3: Send Bandwidth
    # -------------------------------------------------------------------------
    log ""
    log "=========================================="
    log "  TEST 3: SEND BANDWIDTH"
    log "=========================================="

    local send_output
    send_output=$(ib_send_bw --size="$MESSAGE_SIZE" --iters="$ITERATIONS" --report_gbits "$server_ip" 2>&1)
    echo "$send_output"

    local send_bw
    send_bw=$(echo "$send_output" | grep -E "^\s*$MESSAGE_SIZE" | awk '{print $4}')

    if [[ -n "$send_bw" ]]; then
        echo "SEND_BW=$send_bw" >> "$results_file"
        # Send is typically ~5-10% lower due to two-sided overhead
        local send_min=$((MIN_ACCEPTABLE_GBPS - 10))
        if (( $(echo "$send_bw >= $send_min" | bc -l) )); then
            log_success "Send: ${send_bw} Gbps"
        else
            log_error "Send: ${send_bw} Gbps (below ${send_min} Gbps minimum)"
            all_passed=false
        fi
    else
        log_error "Failed to parse Send bandwidth"
        all_passed=false
    fi

    sleep 2

    # -------------------------------------------------------------------------
    # Test 4: Write Latency
    # -------------------------------------------------------------------------
    log ""
    log "=========================================="
    log "  TEST 4: WRITE LATENCY"
    log "=========================================="

    local lat_output
    lat_output=$(ib_write_lat --size=8 --iters=10000 "$server_ip" 2>&1)
    echo "$lat_output"

    local avg_lat
    avg_lat=$(echo "$lat_output" | grep -E "^\s*8\s+" | awk '{print $6}')

    if [[ -n "$avg_lat" ]]; then
        echo "WRITE_LAT_US=$avg_lat" >> "$results_file"
        # ConnectX-5 should achieve < 2us latency
        if (( $(echo "$avg_lat < 5.0" | bc -l) )); then
            log_success "Write Latency: ${avg_lat} μs"
        else
            log_warning "Write Latency: ${avg_lat} μs (higher than expected 2μs)"
        fi
    else
        log_warning "Failed to parse Write latency"
    fi

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    log ""
    log "=========================================="
    log "  VALIDATION SUMMARY"
    log "=========================================="

    cat "$results_file"
    log ""

    if $all_passed; then
        log_success "All bandwidth tests PASSED (>= ${MIN_ACCEPTABLE_GBPS} Gbps)"
        log ""
        log "Hardware baseline validated. Proceed with Java-level optimization."
        rm -f "$results_file"
        exit 0
    else
        log_error "Some tests FAILED to meet minimum threshold"
        log ""
        log "Troubleshooting steps:"
        log "  1. Check link status: ibstat"
        log "  2. Check for errors: ethtool -S <interface> | grep error"
        log "  3. Verify MTU: ip link show"
        log "  4. Check PFC/ECN: mlnx_qos -i <interface>"
        log "  5. Verify NUMA locality: numactl --hardware"
        rm -f "$results_file"
        exit 1
    fi
}

# =============================================================================
# Quick Single-Node Test (Loopback)
# =============================================================================

run_loopback() {
    log "Running loopback test (single-node validation)..."
    log "This validates the NIC and driver, but not network throughput."
    log ""

    # Start server in background
    ib_write_bw --size="$MESSAGE_SIZE" --iters=1000 --report_gbits &
    local server_pid=$!
    sleep 2

    # Run client against localhost
    local output
    output=$(ib_write_bw --size="$MESSAGE_SIZE" --iters=1000 --report_gbits localhost 2>&1) || true
    echo "$output"

    wait $server_pid 2>/dev/null || true

    local bw
    bw=$(echo "$output" | grep -E "^\s*$MESSAGE_SIZE" | awk '{print $4}')

    if [[ -n "$bw" ]]; then
        log_success "Loopback bandwidth: ${bw} Gbps"
        log "Note: Loopback tests PCIe and NIC, not network fabric."
    else
        log_error "Loopback test failed"
        exit 1
    fi
}

# =============================================================================
# Bidirectional Test
# =============================================================================

run_bidirectional() {
    local server_ip="$1"

    log "Running bidirectional bandwidth test..."
    log "This measures full-duplex throughput."
    log ""

    local output
    output=$(ib_write_bw --size="$MESSAGE_SIZE" --iters="$ITERATIONS" --report_gbits --bidirectional "$server_ip" 2>&1)
    echo "$output"

    log_success "Bidirectional test complete"
}

# =============================================================================
# System Info
# =============================================================================

show_system_info() {
    log "=========================================="
    log "  SYSTEM INFORMATION"
    log "=========================================="

    log ""
    log "RDMA Devices:"
    ibv_devices 2>/dev/null || echo "ibv_devices failed"

    log ""
    log "Device Status:"
    ibstat 2>/dev/null | head -40 || echo "ibstat failed"

    log ""
    log "PCIe Info (Mellanox):"
    lspci -vvv 2>/dev/null | grep -A 20 -i mellanox | head -30 || echo "No Mellanox device found"

    log ""
    log "NUMA Topology:"
    numactl --hardware 2>/dev/null || echo "numactl not available"

    log ""
    log "Network Interfaces:"
    ip -br link show 2>/dev/null | grep -E "ib|roce|eth" || ip link show

    log ""
    log "Kernel Modules:"
    lsmod | grep -E "mlx|ib_|rdma" || echo "No RDMA modules loaded"
}

# =============================================================================
# Main
# =============================================================================

usage() {
    cat << EOF
RDMA Throughput Validation Script

Usage: $0 <command> [options]

Commands:
    server              Start server mode (wait for client)
    client <server-ip>  Run client tests against server
    loopback            Run loopback test (single node)
    bidir <server-ip>   Run bidirectional test
    info                Show system RDMA information
    check               Check prerequisites only

Environment Variables:
    RDMA_MSG_SIZE       Message size in bytes (default: 1048576 = 1MB)
    RDMA_ITERATIONS     Number of iterations (default: 5000)
    RDMA_TARGET_GBPS    Target throughput in Gbps (default: 95)
    RDMA_MIN_GBPS       Minimum acceptable Gbps (default: 90)

Examples:
    # On server node:
    $0 server

    # On client node:
    $0 client 192.168.1.100

    # Quick loopback test:
    $0 loopback

    # Show system info:
    $0 info
EOF
}

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
        loopback)
            check_prerequisites
            run_loopback
            ;;
        bidir|bidirectional)
            if [[ $# -lt 1 ]]; then
                log_error "Bidirectional mode requires server IP"
                usage
                exit 1
            fi
            check_prerequisites
            run_bidirectional "$1"
            ;;
        info)
            show_system_info
            ;;
        check)
            check_prerequisites
            log_success "All prerequisites satisfied"
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

main "$@"
