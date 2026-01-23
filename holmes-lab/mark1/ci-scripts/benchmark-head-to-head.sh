#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Head-to-Head Benchmark: C Baseline vs Java Implementation
# =============================================================================
#
# Runs the UCC C baseline and warpforge-io Java implementation back-to-back
# on the same hardware, then compares results to measure Java FFM overhead.
#
# Prerequisites:
#   - Both mark1nvidia and mark1amd nodes accessible via SSH
#   - UCX/UCC installed on both nodes
#   - WarpForge repository synced on both nodes
#
# Usage:
#   ./benchmark-head-to-head.sh                  # Full benchmark
#   ./benchmark-head-to-head.sh --quick          # Quick test (fewer iterations)
#   ./benchmark-head-to-head.sh --java-only      # Skip C baseline
#   ./benchmark-head-to-head.sh --c-only         # Skip Java test
#
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Nodes
NVIDIA_HOST="${NVIDIA_HOST:-mark1nvidia}"
AMD_HOST="${AMD_HOST:-mark1amd}"

# Test configuration
MESSAGE_SIZE="${MESSAGE_SIZE:-16777216}"          # 16MB default
ITERATIONS="${ITERATIONS:-100}"
WARMUP="${WARMUP:-10}"

# Mode flags
RUN_C_BASELINE=true
RUN_JAVA=true
RUN_NATIVE=false
QUICK_MODE=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Results storage
RESULTS_DIR="${SCRIPT_DIR}/../results/head-to-head"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="${RESULTS_DIR}/benchmark_${TIMESTAMP}.txt"

# =============================================================================
# Utility Functions
# =============================================================================

log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
success() { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠${NC} $*"; }
error() { echo -e "${RED}[$(date '+%H:%M:%S')] ✗${NC} $*" >&2; }

ssh_cmd() {
    local host="$1"
    shift
    ssh -o ConnectTimeout=10 -o BatchMode=yes "$host" "$@"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --quick)
                QUICK_MODE=true
                ITERATIONS=10
                WARMUP=2
                shift
                ;;
            --java-only)
                RUN_C_BASELINE=false
                shift
                ;;
            --c-only)
                RUN_JAVA=false
                RUN_NATIVE=false
                shift
                ;;
            --native)
                RUN_NATIVE=true
                shift
                ;;
            --native-only)
                RUN_JAVA=false
                RUN_C_BASELINE=false
                RUN_NATIVE=true
                shift
                ;;
            --size)
                MESSAGE_SIZE="$2"
                shift 2
                ;;
            --iterations)
                ITERATIONS="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --quick          Quick test with fewer iterations"
                echo "  --java-only      Skip C baseline benchmark"
                echo "  --c-only         Skip Java benchmark"
                echo "  --native         Also run native-image compiled Java"
                echo "  --native-only    Only run native-image compiled Java"
                echo "  --size BYTES     Message size (default: 16777216)"
                echo "  --iterations N   Number of iterations (default: 100)"
                echo "  -h, --help       Show this help"
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

setup_results_dir() {
    mkdir -p "$RESULTS_DIR"
}

# =============================================================================
# C Baseline Benchmark
# =============================================================================

build_c_baseline() {
    log "Building C baseline on both nodes..."

    local baseline_dir="holmes-lab/mark1/ucx-baseline"

    # Build on NVIDIA node
    log "  Building on $NVIDIA_HOST..."
    ssh_cmd "$NVIDIA_HOST" "cd ~/warpforge && git pull && cd $baseline_dir && make clean && make ucc" || {
        warn "C baseline build failed on $NVIDIA_HOST - UCC may not be installed"
        return 1
    }

    # Build on AMD node
    log "  Building on $AMD_HOST..."
    ssh_cmd "$AMD_HOST" "cd ~/warpforge && git pull && cd $baseline_dir && make clean && make ucc" || {
        warn "C baseline build failed on $AMD_HOST - UCC may not be installed"
        return 1
    }

    success "C baseline built on both nodes"
    return 0
}

run_c_baseline() {
    log "Running UCC C baseline benchmark..."

    local baseline_dir="~/warpforge/holmes-lab/mark1/ucx-baseline"
    local nvidia_ip=$(ssh_cmd "$NVIDIA_HOST" "hostname -I | awk '{print \$1}'")
    local amd_ip=$(ssh_cmd "$AMD_HOST" "hostname -I | awk '{print \$1}'")

    log "  NVIDIA IP: $nvidia_ip"
    log "  AMD IP: $amd_ip"

    # Start server on NVIDIA (rank 0)
    log "  Starting rank 0 on $NVIDIA_HOST..."
    ssh_cmd "$NVIDIA_HOST" "
        cd $baseline_dir
        ./ucc_collective_benchmark \
            -r 0 -n 2 \
            -a $amd_ip \
            -s $MESSAGE_SIZE \
            -i $ITERATIONS \
            -w $WARMUP \
            -t all
    " > "${RESULTS_DIR}/c_baseline_${TIMESTAMP}.txt" 2>&1 &
    local nvidia_pid=$!

    # Give server time to start
    sleep 2

    # Start client on AMD (rank 1)
    log "  Starting rank 1 on $AMD_HOST..."
    ssh_cmd "$AMD_HOST" "
        cd $baseline_dir
        ./ucc_collective_benchmark \
            -r 1 -n 2 \
            -a $nvidia_ip \
            -s $MESSAGE_SIZE \
            -i $ITERATIONS \
            -w $WARMUP \
            -t all
    " >> "${RESULTS_DIR}/c_baseline_${TIMESTAMP}.txt" 2>&1 &
    local amd_pid=$!

    # Wait for both to complete
    wait $nvidia_pid || warn "NVIDIA C baseline process exited with error"
    wait $amd_pid || warn "AMD C baseline process exited with error"

    success "C baseline benchmark complete"

    # Show results
    log "C Baseline Results:"
    echo ""
    cat "${RESULTS_DIR}/c_baseline_${TIMESTAMP}.txt"
    echo ""
}

# =============================================================================
# Java Implementation Benchmark
# =============================================================================

run_java_benchmark() {
    log "Running warpforge-io Java benchmark..."

    # Build warpforge-io on NVIDIA node (master)
    log "  Building warpforge-io..."
    ssh_cmd "$NVIDIA_HOST" "cd ~/warpforge && ./gradlew :warpforge-io:assemble" || {
        error "Failed to build warpforge-io"
        return 1
    }

    # Get IPs
    local nvidia_ip=$(ssh_cmd "$NVIDIA_HOST" "hostname -I | awk '{print \$1}'")

    # Start master on NVIDIA
    log "  Starting Java master on $NVIDIA_HOST..."
    ssh_cmd "$NVIDIA_HOST" "
        cd ~/warpforge
        ./gradlew :warpforge-io:uccPerfMaster \
            -Psize=$MESSAGE_SIZE \
            -Piterations=$ITERATIONS
    " > "${RESULTS_DIR}/java_${TIMESTAMP}.txt" 2>&1 &
    local master_pid=$!

    # Give master time to start
    sleep 5

    # Start worker on AMD
    log "  Starting Java worker on $AMD_HOST..."
    ssh_cmd "$AMD_HOST" "
        cd ~/warpforge
        ./gradlew :warpforge-io:uccPerfWorker \
            -Psize=$MESSAGE_SIZE \
            -Piterations=$ITERATIONS
    " >> "${RESULTS_DIR}/java_${TIMESTAMP}.txt" 2>&1 &
    local worker_pid=$!

    # Wait for both
    wait $master_pid || warn "Java master process exited with error"
    wait $worker_pid || warn "Java worker process exited with error"

    success "Java benchmark complete"

    # Show results
    log "Java Implementation Results:"
    echo ""
    cat "${RESULTS_DIR}/java_${TIMESTAMP}.txt"
    echo ""
}

# =============================================================================
# Native Image Benchmark (AOT-compiled Java via GraalVM native-image)
# =============================================================================

build_native_image() {
    log "Building native-image executable on both nodes..."

    # Build on NVIDIA node
    log "  Building native-image on $NVIDIA_HOST..."
    ssh_cmd "$NVIDIA_HOST" "cd ~/warpforge && ./gradlew :warpforge-io:nativeCompile --no-configuration-cache" || {
        error "Native-image build failed on $NVIDIA_HOST"
        return 1
    }

    # Build on AMD node
    log "  Building native-image on $AMD_HOST..."
    ssh_cmd "$AMD_HOST" "cd ~/warpforge && ./gradlew :warpforge-io:nativeCompile --no-configuration-cache" || {
        error "Native-image build failed on $AMD_HOST"
        return 1
    }

    success "Native-image built on both nodes"
    return 0
}

run_native_benchmark() {
    log "Running native-image compiled Java benchmark..."

    # Get IPs
    local nvidia_ip=$(ssh_cmd "$NVIDIA_HOST" "hostname -I | awk '{print \$1}'")

    local exe_path="~/warpforge/warpforge-io/build/native/nativeCompile/ucc-perf-test"
    local lib_path="${OPENUCX_PATH:-/usr/local}/lib:${UCC_PATH:-/usr/local}/lib"

    # Start master on NVIDIA
    log "  Starting native master on $NVIDIA_HOST..."
    ssh_cmd "$NVIDIA_HOST" "
        export LD_LIBRARY_PATH=\"$lib_path:\$LD_LIBRARY_PATH\"
        export UCC_CLS=basic
        export UCC_TLS=ucp
        export UCC_TL_UCP_ALLREDUCE_ALG=ring
        export UCX_RNDV_SCHEME=get_zcopy
        export UCX_RNDV_THRESH=8192

        $exe_path \
            --rank 0 --world-size 2 \
            --master $nvidia_ip --port 29500 \
            --size $MESSAGE_SIZE \
            --iterations $ITERATIONS \
            --warmup $WARMUP
    " > "${RESULTS_DIR}/native_${TIMESTAMP}.txt" 2>&1 &
    local master_pid=$!

    # Give master time to start
    sleep 3

    # Start worker on AMD
    log "  Starting native worker on $AMD_HOST..."
    ssh_cmd "$AMD_HOST" "
        export LD_LIBRARY_PATH=\"$lib_path:\$LD_LIBRARY_PATH\"
        export UCC_CLS=basic
        export UCC_TLS=ucp
        export UCC_TL_UCP_ALLREDUCE_ALG=ring
        export UCX_RNDV_SCHEME=get_zcopy
        export UCX_RNDV_THRESH=8192

        $exe_path \
            --rank 1 --world-size 2 \
            --master $nvidia_ip --port 29500 \
            --size $MESSAGE_SIZE \
            --iterations $ITERATIONS \
            --warmup $WARMUP
    " >> "${RESULTS_DIR}/native_${TIMESTAMP}.txt" 2>&1 &
    local worker_pid=$!

    # Wait for both
    wait $master_pid || warn "Native master process exited with error"
    wait $worker_pid || warn "Native worker process exited with error"

    success "Native-image benchmark complete"

    # Show results
    log "Native-image Java Results:"
    echo ""
    cat "${RESULTS_DIR}/native_${TIMESTAMP}.txt"
    echo ""
}

# =============================================================================
# Results Comparison
# =============================================================================

compare_results() {
    log "Comparing results..."

    {
        echo "=============================================================================="
        echo "Head-to-Head Benchmark Results"
        echo "=============================================================================="
        echo ""
        echo "Configuration:"
        echo "  Message Size: $MESSAGE_SIZE bytes ($(echo "scale=2; $MESSAGE_SIZE/1048576" | bc) MB)"
        echo "  Iterations: $ITERATIONS"
        echo "  Warmup: $WARMUP"
        echo "  Date: $(date)"
        echo ""

        if [[ "$RUN_C_BASELINE" == "true" ]] && [[ -f "${RESULTS_DIR}/c_baseline_${TIMESTAMP}.txt" ]]; then
            echo "------------------------------------------------------------------------------"
            echo "C Baseline (UCC Collective Benchmark)"
            echo "------------------------------------------------------------------------------"
            cat "${RESULTS_DIR}/c_baseline_${TIMESTAMP}.txt"
            echo ""
        fi

        if [[ "$RUN_JAVA" == "true" ]] && [[ -f "${RESULTS_DIR}/java_${TIMESTAMP}.txt" ]]; then
            echo "------------------------------------------------------------------------------"
            echo "Java Implementation (warpforge-io UCC on JVM)"
            echo "------------------------------------------------------------------------------"
            cat "${RESULTS_DIR}/java_${TIMESTAMP}.txt"
            echo ""
        fi

        if [[ "$RUN_NATIVE" == "true" ]] && [[ -f "${RESULTS_DIR}/native_${TIMESTAMP}.txt" ]]; then
            echo "------------------------------------------------------------------------------"
            echo "Native-Image Java (warpforge-io AOT compiled)"
            echo "------------------------------------------------------------------------------"
            cat "${RESULTS_DIR}/native_${TIMESTAMP}.txt"
            echo ""
        fi

        echo "=============================================================================="
        echo "Analysis"
        echo "=============================================================================="
        echo ""

        # Extract throughput values for comparison if both ran
        if [[ "$RUN_C_BASELINE" == "true" ]] && [[ "$RUN_JAVA" == "true" ]]; then
            echo "To calculate overhead: (C_Gbps - Java_Gbps) / C_Gbps * 100"
            echo ""
            echo "Target: Java should achieve >85% of C baseline throughput"
            echo ""
        fi

        if [[ "$RUN_NATIVE" == "true" ]]; then
            echo "Native-image eliminates JVM startup and reduces FFM overhead."
            echo "Expected: Native should be within 5-10% of C baseline."
            echo ""
        fi

    } | tee "$RESULT_FILE"

    success "Results saved to: $RESULT_FILE"
}

# =============================================================================
# Main
# =============================================================================

main() {
    parse_args "$@"

    echo ""
    log "Head-to-Head Benchmark: C Baseline vs Java Implementation"
    log "==========================================================="
    echo ""

    if [[ "$QUICK_MODE" == "true" ]]; then
        warn "Quick mode enabled (iterations=$ITERATIONS, warmup=$WARMUP)"
        echo ""
    fi

    setup_results_dir

    # Check connectivity
    log "Checking node connectivity..."
    ssh_cmd "$NVIDIA_HOST" "echo 'NVIDIA node OK'" || {
        error "Cannot connect to $NVIDIA_HOST"
        exit 1
    }
    ssh_cmd "$AMD_HOST" "echo 'AMD node OK'" || {
        error "Cannot connect to $AMD_HOST"
        exit 1
    }
    success "Both nodes accessible"
    echo ""

    # Run C baseline
    if [[ "$RUN_C_BASELINE" == "true" ]]; then
        if build_c_baseline; then
            run_c_baseline
        else
            warn "Skipping C baseline due to build failure"
            RUN_C_BASELINE=false
        fi
        echo ""
    fi

    # Run Java benchmark
    if [[ "$RUN_JAVA" == "true" ]]; then
        run_java_benchmark
        echo ""
    fi

    # Run native-image benchmark
    if [[ "$RUN_NATIVE" == "true" ]]; then
        if build_native_image; then
            run_native_benchmark
        else
            warn "Skipping native-image due to build failure"
            RUN_NATIVE=false
        fi
        echo ""
    fi

    # Compare and summarize
    compare_results

    echo ""
    success "Benchmark complete!"
}

main "$@"
