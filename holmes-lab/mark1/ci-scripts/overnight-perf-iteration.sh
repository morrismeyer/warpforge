#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Overnight Performance Iteration Script
# =============================================================================
#
# Continuously runs UCC collective performance tests and tracks improvements
# over time. Designed to run overnight unattended.
#
# Features:
# - Runs performance tests at regular intervals
# - Tracks results in a cumulative log
# - Computes statistics across runs
# - Identifies performance trends
# - Generates summary report
#
# Usage:
#   ./overnight-perf-iteration.sh [OPTIONS]
#
# Options:
#   --duration HOURS   Run for N hours (default: 8, overnight)
#   --interval MINS    Minutes between runs (default: 15)
#   --quick            Use quick mode for individual tests
#   --report-only      Just generate report from existing data
#
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PERF_TEST_SCRIPT="$SCRIPT_DIR/perf-test-ucc-collectives.sh"

# Configuration
DURATION_HOURS="${DURATION_HOURS:-8}"
INTERVAL_MINS="${INTERVAL_MINS:-15}"
QUICK_MODE=0
REPORT_ONLY=0

# Data storage
DATA_DIR="$HOME/build-logs/overnight-perf"
RUN_ID="overnight-$(date +%Y%m%d-%H%M%S)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration)
            DURATION_HOURS="$2"
            shift 2
            ;;
        --interval)
            INTERVAL_MINS="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=1
            shift
            ;;
        --report-only)
            REPORT_ONLY=1
            shift
            ;;
        --help|-h)
            head -30 "$0" | tail -n +2 | grep -E "^#" | sed 's/^# *//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$DATA_DIR"

# Files
CUMULATIVE_LOG="$DATA_DIR/cumulative-results.csv"
ITERATION_LOG="$DATA_DIR/iteration-log.txt"
SUMMARY_REPORT="$DATA_DIR/summary-report.txt"
CURRENT_RUN_LOG="$DATA_DIR/${RUN_ID}.log"

log() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "[$timestamp] $*" | tee -a "$CURRENT_RUN_LOG"
}

log_section() {
    echo "" | tee -a "$CURRENT_RUN_LOG"
    log "${CYAN}════════════════════════════════════════════════════════════════════${NC}"
    log "${CYAN}  $*${NC}"
    log "${CYAN}════════════════════════════════════════════════════════════════════${NC}"
}

# Initialize cumulative log if it doesn't exist
initialize_cumulative_log() {
    if [[ ! -f "$CUMULATIVE_LOG" ]]; then
        echo "timestamp,iteration,size_bytes,operation,bandwidth_gbps,latency_ms" > "$CUMULATIVE_LOG"
    fi
}

# Parse results from a single run
parse_and_append_results() {
    local results_file=$1
    local iteration=$2
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    if [[ -f "$results_file" ]]; then
        # Skip header and append to cumulative log
        tail -n +2 "$results_file" | while IFS=',' read -r size op bw lat; do
            echo "$timestamp,$iteration,$size,$op,$bw,$lat" >> "$CUMULATIVE_LOG"
        done
    fi
}

# Calculate statistics for a specific operation and size
calculate_stats() {
    local op=$1
    local size=$2

    # Extract bandwidth values for this operation/size
    local values
    values=$(grep ",$size,$op," "$CUMULATIVE_LOG" | cut -d',' -f5 | grep -v "^$" | sort -n)

    if [[ -z "$values" ]]; then
        echo "N/A"
        return
    fi

    local count mean min max
    count=$(echo "$values" | wc -l | tr -d ' ')

    if [[ $count -eq 0 ]]; then
        echo "N/A"
        return
    fi

    min=$(echo "$values" | head -1)
    max=$(echo "$values" | tail -1)
    mean=$(echo "$values" | awk '{sum+=$1} END {printf "%.2f", sum/NR}')

    # Calculate std dev
    local stddev
    stddev=$(echo "$values" | awk -v mean="$mean" '{sum+=($1-mean)^2} END {printf "%.2f", sqrt(sum/NR)}')

    echo "mean=$mean min=$min max=$max stddev=$stddev n=$count"
}

# Generate summary report
generate_report() {
    log_section "GENERATING PERFORMANCE SUMMARY REPORT"

    {
        echo "═════════════════════════════════════════════════════════════════════"
        echo "  OVERNIGHT UCC COLLECTIVE PERFORMANCE REPORT"
        echo "  Generated: $(date)"
        echo "═════════════════════════════════════════════════════════════════════"
        echo ""

        # Get unique operations and sizes
        local operations sizes
        operations=$(tail -n +2 "$CUMULATIVE_LOG" | cut -d',' -f4 | sort -u)
        sizes=$(tail -n +2 "$CUMULATIVE_LOG" | cut -d',' -f3 | sort -un)

        # Summary table
        echo "BANDWIDTH SUMMARY (Gbps)"
        echo "────────────────────────────────────────────────────────────────────"
        printf "%-15s %-10s %-10s %-10s %-10s %-10s %-5s\n" "Operation" "Size" "Mean" "Min" "Max" "StdDev" "N"
        echo "────────────────────────────────────────────────────────────────────"

        for op in $operations; do
            for size in $sizes; do
                local stats
                stats=$(calculate_stats "$op" "$size")
                if [[ "$stats" != "N/A" ]]; then
                    # Format size
                    local size_fmt
                    if (( size >= 1073741824 )); then
                        size_fmt="$(echo "scale=0; $size / 1073741824" | bc)GB"
                    elif (( size >= 1048576 )); then
                        size_fmt="$(echo "scale=0; $size / 1048576" | bc)MB"
                    elif (( size >= 1024 )); then
                        size_fmt="$(echo "scale=0; $size / 1024" | bc)KB"
                    else
                        size_fmt="${size}B"
                    fi

                    # Parse stats
                    local mean min max stddev n
                    mean=$(echo "$stats" | grep -oE 'mean=[0-9.]+' | cut -d= -f2)
                    min=$(echo "$stats" | grep -oE 'min=[0-9.]+' | cut -d= -f2)
                    max=$(echo "$stats" | grep -oE 'max=[0-9.]+' | cut -d= -f2)
                    stddev=$(echo "$stats" | grep -oE 'stddev=[0-9.]+' | cut -d= -f2)
                    n=$(echo "$stats" | grep -oE 'n=[0-9]+' | cut -d= -f2)

                    printf "%-15s %-10s %-10s %-10s %-10s %-10s %-5s\n" \
                        "$op" "$size_fmt" "$mean" "$min" "$max" "$stddev" "$n"
                fi
            done
        done

        echo ""
        echo "────────────────────────────────────────────────────────────────────"

        # Performance trend analysis
        echo ""
        echo "PERFORMANCE TRENDS"
        echo "────────────────────────────────────────────────────────────────────"

        # Check for improvement/regression over time
        local total_runs
        total_runs=$(tail -n +2 "$CUMULATIVE_LOG" | cut -d',' -f2 | sort -un | wc -l | tr -d ' ')

        echo "Total test iterations: $total_runs"
        echo ""

        # Find best and worst runs for large message AllReduce (most important metric)
        local best_allreduce worst_allreduce
        best_allreduce=$(grep ",16777216,AllReduce," "$CUMULATIVE_LOG" 2>/dev/null | cut -d',' -f5 | sort -rn | head -1)
        worst_allreduce=$(grep ",16777216,AllReduce," "$CUMULATIVE_LOG" 2>/dev/null | cut -d',' -f5 | sort -n | head -1)

        if [[ -n "$best_allreduce" && -n "$worst_allreduce" ]]; then
            echo "AllReduce 16MB Performance Range:"
            echo "  Best:  ${best_allreduce} Gbps"
            echo "  Worst: ${worst_allreduce} Gbps"

            local variance
            variance=$(echo "scale=1; ($best_allreduce - $worst_allreduce) * 100 / $best_allreduce" | bc)
            echo "  Variance: ${variance}%"

            if (( $(echo "$variance > 20" | bc -l) )); then
                echo "  Status: HIGH VARIANCE - performance is unstable"
            elif (( $(echo "$variance > 10" | bc -l) )); then
                echo "  Status: MODERATE VARIANCE - some instability"
            else
                echo "  Status: STABLE - performance is consistent"
            fi
        fi

        echo ""
        echo "────────────────────────────────────────────────────────────────────"

        # Recommendations
        echo ""
        echo "OPTIMIZATION RECOMMENDATIONS"
        echo "────────────────────────────────────────────────────────────────────"

        # Analyze peak bandwidth achieved
        local peak_bw
        peak_bw=$(tail -n +2 "$CUMULATIVE_LOG" | cut -d',' -f5 | sort -rn | head -1)

        if [[ -n "$peak_bw" ]]; then
            local efficiency
            efficiency=$(echo "scale=1; $peak_bw * 100 / 100" | bc)

            echo "Peak bandwidth achieved: ${peak_bw} Gbps (${efficiency}% of 100GbE)"
            echo ""

            if (( $(echo "$peak_bw < 50" | bc -l) )); then
                echo "CRITICAL: Bandwidth below 50% of line rate"
                echo ""
                echo "Suggested optimizations:"
                echo "  1. Check UCX transport layer selection (UCX_TLS)"
                echo "  2. Verify RDMA device is being used (not TCP)"
                echo "  3. Check for CPU bottleneck (use perf record)"
                echo "  4. Review UCC algorithm selection for message sizes"
                echo "  5. Consider increasing message sizes to amortize latency"
            elif (( $(echo "$peak_bw < 80" | bc -l) )); then
                echo "MODERATE: Bandwidth at 50-80% of line rate"
                echo ""
                echo "Suggested optimizations:"
                echo "  1. Tune UCC algorithm selection (UCC_TL_UCP_*_ALG)"
                echo "  2. Consider pipelining multiple operations"
                echo "  3. Profile FFM overhead with async-profiler"
                echo "  4. Review memory allocation patterns"
            else
                echo "GOOD: Bandwidth at 80%+ of line rate"
                echo ""
                echo "Fine-tuning options:"
                echo "  1. Test with larger message sizes for higher throughput"
                echo "  2. Profile latency for small messages"
                echo "  3. Consider asynchronous operation pipelining"
            fi
        fi

        echo ""
        echo "═════════════════════════════════════════════════════════════════════"
        echo "  Data files: $DATA_DIR"
        echo "═════════════════════════════════════════════════════════════════════"

    } > "$SUMMARY_REPORT"

    cat "$SUMMARY_REPORT" | tee -a "$CURRENT_RUN_LOG"
}

# Main execution loop
run_overnight() {
    local start_time duration_secs interval_secs iteration elapsed
    start_time=$(date +%s)
    duration_secs=$((DURATION_HOURS * 3600))
    interval_secs=$((INTERVAL_MINS * 60))
    iteration=1

    log_section "OVERNIGHT PERFORMANCE ITERATION"
    log "Run ID: $RUN_ID"
    log "Duration: ${DURATION_HOURS} hours"
    log "Interval: ${INTERVAL_MINS} minutes"
    log "Data directory: $DATA_DIR"
    log "Quick mode: $QUICK_MODE"
    log ""
    log "Starting at $(date)"
    log "Will run until $(date -d "+${DURATION_HOURS} hours" 2>/dev/null || date -v+${DURATION_HOURS}H)"

    initialize_cumulative_log

    while true; do
        elapsed=$(($(date +%s) - start_time))

        if (( elapsed >= duration_secs )); then
            log ""
            log "Duration limit reached. Stopping."
            break
        fi

        log ""
        log_section "ITERATION $iteration"
        log "Elapsed: $((elapsed / 60)) minutes of $((duration_secs / 60)) minutes"

        # Run performance test
        local test_args=""
        if [[ "$QUICK_MODE" -eq 1 ]]; then
            test_args="--quick"
        fi
        test_args="$test_args --no-wake"  # Boxes are already on

        set +e
        "$PERF_TEST_SCRIPT" $test_args 2>&1 | tee -a "$CURRENT_RUN_LOG"
        local test_status=$?
        set -e

        if [[ $test_status -eq 0 ]]; then
            # Find the latest results file
            local latest_results
            latest_results=$(ls -t "$HOME/build-logs"/ucc-perf-*-results.csv 2>/dev/null | head -1)

            if [[ -f "$latest_results" ]]; then
                parse_and_append_results "$latest_results" "$iteration"
                log "Results appended to cumulative log"
            fi
        else
            log "Test iteration $iteration failed"
            echo "$(date '+%Y-%m-%d %H:%M:%S'),$iteration,FAILED" >> "$ITERATION_LOG"
        fi

        iteration=$((iteration + 1))

        # Check if we should continue
        elapsed=$(($(date +%s) - start_time))
        if (( elapsed + interval_secs >= duration_secs )); then
            log ""
            log "Not enough time for another iteration. Stopping."
            break
        fi

        log ""
        log "Sleeping ${INTERVAL_MINS} minutes until next iteration..."
        log "Press Ctrl+C to stop and generate report"
        sleep "$interval_secs"
    done

    # Generate final report
    generate_report

    log ""
    log_section "OVERNIGHT RUN COMPLETE"
    log "Total iterations: $((iteration - 1))"
    log "Summary report: $SUMMARY_REPORT"
    log "Cumulative data: $CUMULATIVE_LOG"
    log "Finished at $(date)"
}

# Handle Ctrl+C gracefully
trap 'log ""; log "Interrupted. Generating report..."; generate_report; exit 0' INT TERM

# Main
if [[ "$REPORT_ONLY" -eq 1 ]]; then
    generate_report
else
    run_overnight
fi
