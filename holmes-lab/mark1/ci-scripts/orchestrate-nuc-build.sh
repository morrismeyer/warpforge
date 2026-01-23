#!/usr/bin/env bash
set -euo pipefail

# NUC orchestration entrypoint.
#
# Intended to run ON the NUC. It:
# - Updates the local WarpForge repo on the NUC
# - Runs a NUC build + smoke test
# - If green, triggers the NVIDIA box build
# - If green, triggers the AMD box build
#
# Self-update pattern: After git reset, the script re-execs itself
# to ensure it always runs with the latest version from the repo.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${BASH_SOURCE[0]}"

# -------- CONFIG (override via environment variables) --------

# WarpForge repo on the NUC
# Default matches your recent layout: ~/surfworks/warpforge
NUC_REPO_DIR="${NUC_REPO_DIR_OVERRIDE:-$HOME/surfworks/warpforge}"

# Branch from GitHub Actions. Fallback to main for manual runs.
BRANCH="${GITHUB_REF_NAME:-main}"

# Build + smoke test commands on the NUC
NUC_BUILD_CMD="${NUC_BUILD_CMD_OVERRIDE:-./gradlew clean assemble}"
NUC_TEST_CMD="${NUC_TEST_CMD_OVERRIDE:-./gradlew test --no-build-cache}"

# Paths to the NVIDIA and AMD orchestrator scripts.
# Default is to run the sibling scripts shipped with this file.
NVIDIA_ORCH_SCRIPT="${NVIDIA_ORCH_SCRIPT_OVERRIDE:-$SCRIPT_DIR/orchestrate-nvidia-build.sh}"
AMD_ORCH_SCRIPT="${AMD_ORCH_SCRIPT_OVERRIDE:-$SCRIPT_DIR/orchestrate-amd-build.sh}"

LOG_ROOT="${LOG_ROOT_OVERRIDE:-$HOME/build-logs}"
mkdir -p "$LOG_ROOT"

RUN_ID="${GITHUB_RUN_ID:-manual-$(date +%Y%m%d-%H%M%S)}"
LOG_FILE="$LOG_ROOT/nuc-${RUN_ID}.log"

# WarpForge repo URL for cloning if missing
WARP_FORGE_REPO_URL="${WARP_FORGE_REPO_URL_OVERRIDE:-git@github.com:morrismeyer/warpforge.git}"

# -------- END CONFIG --------

log() {
  echo "[$(date)] $*" | tee -a "$LOG_FILE"
}

log "==== NUC ORCHESTRATOR START (run ${RUN_ID}) ===="
log "Branch: ${BRANCH}"
log "NUC repo dir: ${NUC_REPO_DIR}"
log "NUC build cmd: ${NUC_BUILD_CMD}"
log "NUC test  cmd: ${NUC_TEST_CMD}"
log "NVIDIA script: ${NVIDIA_ORCH_SCRIPT}"
log "AMD script:    ${AMD_ORCH_SCRIPT}"
log "Repo URL:      ${WARP_FORGE_REPO_URL}"

# 1) Prepare repo on NUC
log "Preparing WarpForge repo on NUC..."

mkdir -p "$(dirname "$NUC_REPO_DIR")"
if [[ ! -d "$NUC_REPO_DIR/.git" ]]; then
  log "Repo not present, cloning..."
  mkdir -p "$NUC_REPO_DIR"
  cd "$NUC_REPO_DIR"
  git clone "$WARP_FORGE_REPO_URL" . 2>&1 | tee -a "$LOG_FILE"
else
  cd "$NUC_REPO_DIR"
fi

log "Fetching origin..."
git fetch origin 2>&1 | tee -a "$LOG_FILE"

# Robust branch checkout (works even if local branch does not exist yet)
if git show-ref --verify --quiet "refs/remotes/origin/${BRANCH}"; then
  if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    git checkout "${BRANCH}" 2>&1 | tee -a "$LOG_FILE"
  else
    git checkout -b "${BRANCH}" "origin/${BRANCH}" 2>&1 | tee -a "$LOG_FILE"
  fi
  log "Resetting working tree to origin/${BRANCH}..."
  git reset --hard "origin/${BRANCH}" 2>&1 | tee -a "$LOG_FILE"
  log "Cleaning untracked files (excluding venvs, build caches, and GraalPy tools)..."
  git clean -fdx -e '.pytorch-venv' -e '.gradle' -e 'snakegrinder-dist/tools' 2>&1 | tee -a "$LOG_FILE"

  # Diagnostic: Verify clean state
  log "=== Git Status After Reset ==="
  git status 2>&1 | tee -a "$LOG_FILE"
  log "Current HEAD: $(git rev-parse HEAD)"
  log "Expected HEAD (origin/${BRANCH}): $(git rev-parse origin/${BRANCH})"

  # Check for problematic file
  if [[ -f "warpforge-io/src/main/java/io/surfworks/warpforge/io/rdma/impl/UcxRdmaEndpoint.java" ]]; then
    log "UcxRdmaEndpoint.java imports (first 15 lines):"
    head -15 "warpforge-io/src/main/java/io/surfworks/warpforge/io/rdma/impl/UcxRdmaEndpoint.java" | tee -a "$LOG_FILE"
  fi
  log "=== End Git Status ==="

  # Self-update: Re-exec with the newly pulled script version
  # This ensures we always run with the latest script from the repo
  if [[ "${SCRIPT_ALREADY_UPDATED:-}" != "true" ]]; then
    log "Re-executing script with updated version from git..."
    export SCRIPT_ALREADY_UPDATED=true
    exec "$SCRIPT_PATH" "$@"
  fi
else
  log "ERROR: origin/${BRANCH} does not exist. Check the branch name in GITHUB_REF_NAME."
  exit 1
fi

# 1b) Diagnostic: Check UCC/UCX library status before build
log "=== Native Library Diagnostics (Pre-Build) ==="
UCC_LIB_PATH="$NUC_REPO_DIR/../ucc/install/lib/libucc.so"
UCX_LIB_PATH="$NUC_REPO_DIR/../openucx/install/lib/libucp.so"

log "Checking UCC library..."
if [[ -f "$UCC_LIB_PATH" ]]; then
  log "UCC library found at: $UCC_LIB_PATH"
  log "UCC dependencies:"
  ldd "$UCC_LIB_PATH" 2>&1 | tee -a "$LOG_FILE" || log "ldd failed for UCC"
  # Check for missing dependencies
  MISSING=$(ldd "$UCC_LIB_PATH" 2>&1 | grep "not found" || true)
  if [[ -n "$MISSING" ]]; then
    log "WARNING: UCC has missing dependencies:"
    echo "$MISSING" | tee -a "$LOG_FILE"
  else
    log "UCC dependencies OK"
  fi
else
  log "UCC library NOT found at expected path: $UCC_LIB_PATH"
fi

log "Checking UCX library..."
if [[ -f "$UCX_LIB_PATH" ]]; then
  log "UCX library found at: $UCX_LIB_PATH"
  MISSING=$(ldd "$UCX_LIB_PATH" 2>&1 | grep "not found" || true)
  if [[ -n "$MISSING" ]]; then
    log "WARNING: UCX has missing dependencies:"
    echo "$MISSING" | tee -a "$LOG_FILE"
  else
    log "UCX dependencies OK"
  fi
else
  log "UCX library NOT found at expected path: $UCX_LIB_PATH"
fi

log "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<not set>}"
log "=== End Native Library Diagnostics ==="

# 2) Build + smoke test on NUC
log "Running NUC build: ${NUC_BUILD_CMD}"
bash -lc "${NUC_BUILD_CMD}" 2>&1 | tee -a "$LOG_FILE"

log "Running NUC tests: ${NUC_TEST_CMD}"
bash -lc "${NUC_TEST_CMD}" 2>&1 | tee -a "$LOG_FILE"

# Explicitly run snakeburger-core tests and verify they executed
log "Running snakeburger-core tests explicitly..."
bash -lc "./gradlew :snakeburger-core:test --info --no-build-cache 2>&1" | tee -a "$LOG_FILE" || true

# Verify test results exist and have sufficient count
TEST_RESULTS_DIR="snakeburger-core/build/test-results/test"
if [[ -d "$TEST_RESULTS_DIR" ]]; then
  TEST_COUNT=$(find "$TEST_RESULTS_DIR" -name "*.xml" -exec grep -h "tests=" {} \; 2>/dev/null | grep -oE 'tests="[0-9]+"' | grep -oE '[0-9]+' | awk '{sum+=$1} END {print sum}')
  log "snakeburger-core test count: ${TEST_COUNT:-0}"

  if [[ "${TEST_COUNT:-0}" -lt 300 ]]; then
    log "ERROR: Expected at least 300 snakeburger-core tests but found ${TEST_COUNT:-0}"
    log "This suggests tests were skipped. Checking Babylon JDK availability..."
    if [[ -d "$HOME/surfworks/babylon/build" ]]; then
      log "Babylon build directory exists at $HOME/surfworks/babylon/build"
      find "$HOME/surfworks/babylon/build" -name "javac" -type f 2>&1 | head -5 | tee -a "$LOG_FILE" || true
    else
      log "ERROR: Babylon JDK not found at $HOME/surfworks/babylon/build"
      log "snakeburger tests require Babylon JDK. Please build it first:"
      log "  cd ~/surfworks/babylon && bash configure && make images"
    fi
    exit 1
  fi

  log "SUCCESS: snakeburger-core tests executed (${TEST_COUNT} tests)"
else
  log "ERROR: No test results found at $TEST_RESULTS_DIR"
  log "snakeburger-core tests may not have run!"
  log "Checking if Babylon JDK is available..."
  if [[ -d "$HOME/surfworks/babylon/build" ]]; then
    log "Babylon build directory exists"
    find "$HOME/surfworks/babylon/build" -name "javac" -type f 2>&1 | head -5 | tee -a "$LOG_FILE" || true
  else
    log "ERROR: Babylon JDK not found at $HOME/surfworks/babylon/build"
  fi
  exit 1
fi

# Verify warpforge-core tests executed
log "Verifying warpforge-core tests..."
TEST_RESULTS_DIR="warpforge-core/build/test-results/test"
if [[ -d "$TEST_RESULTS_DIR" ]]; then
  TEST_COUNT=$(find "$TEST_RESULTS_DIR" -name "*.xml" -exec grep -h "tests=" {} \; 2>/dev/null | grep -oE 'tests="[0-9]+"' | grep -oE '[0-9]+' | awk '{sum+=$1} END {print sum}')
  log "warpforge-core test count: ${TEST_COUNT:-0}"

  if [[ "${TEST_COUNT:-0}" -lt 100 ]]; then
    log "ERROR: Expected at least 100 warpforge-core tests but found ${TEST_COUNT:-0}"
    exit 1
  fi

  log "SUCCESS: warpforge-core tests executed (${TEST_COUNT} tests)"
else
  log "ERROR: No test results found at $TEST_RESULTS_DIR"
  exit 1
fi

# Verify warpforge-io tests executed (RDMA, Collective, VirtualThreads)
log "Verifying warpforge-io tests..."
TEST_RESULTS_DIR="warpforge-io/build/test-results/test"
if [[ -d "$TEST_RESULTS_DIR" ]]; then
  TEST_COUNT=$(find "$TEST_RESULTS_DIR" -name "*.xml" -exec grep -h "tests=" {} \; 2>/dev/null | grep -oE 'tests="[0-9]+"' | grep -oE '[0-9]+' | awk '{sum+=$1} END {print sum}')
  log "warpforge-io test count: ${TEST_COUNT:-0}"

  if [[ "${TEST_COUNT:-0}" -lt 150 ]]; then
    log "ERROR: Expected at least 150 warpforge-io tests but found ${TEST_COUNT:-0}"
    log "warpforge-io tests include RDMA, Collective, and VirtualThreads tests"
    exit 1
  fi

  log "SUCCESS: warpforge-io tests executed (${TEST_COUNT} tests)"
else
  log "ERROR: No test results found at $TEST_RESULTS_DIR"
  log "warpforge-io tests should run with standard JDK 25"
  exit 1
fi

log "NUC build + tests SUCCESS"

# 2b) SnakeGrinder distribution tests (requires pre-built PyTorch venv)
log "Running snakegrinder-dist tests..."
SNAKEGRINDER_VENV="$NUC_REPO_DIR/snakegrinder-dist/.pytorch-venv"
PRUNE_MARKER="$SNAKEGRINDER_VENV/.prune-marker"

# Check for stale venv (missing prune marker = pre-validation era venv)
if [[ -d "$SNAKEGRINDER_VENV" && ! -f "$PRUNE_MARKER" ]]; then
  log "WARNING: PyTorch venv exists but lacks prune marker (pre-validation era)"
  log "Deleting stale venv to force rebuild with proper validation..."
  rm -rf "$SNAKEGRINDER_VENV"
  log "Stale venv deleted. Will skip tests this run."
fi

if [[ -d "$SNAKEGRINDER_VENV" ]]; then
  log "PyTorch venv found at $SNAKEGRINDER_VENV"
  if [[ -f "$PRUNE_MARKER" ]]; then
    log "Prune marker contents:"
    cat "$PRUNE_MARKER" | tee -a "$LOG_FILE"
  fi
  bash -lc "./gradlew :snakegrinder-dist:testDist --no-configuration-cache --no-build-cache 2>&1" | tee -a "$LOG_FILE"
  log "snakegrinder-dist tests SUCCESS"
else
  log "WARNING: PyTorch venv not found at $SNAKEGRINDER_VENV"
  log "Skipping snakegrinder-dist tests. To enable, run:"
  log "  cd $NUC_REPO_DIR && ./gradlew :snakegrinder-dist:buildPytorchVenv"
  log "  cd $NUC_REPO_DIR && ./gradlew :snakegrinder-dist:prunePytorchVenv"
fi

# 3) NVIDIA tier
if [[ ! -x "$NVIDIA_ORCH_SCRIPT" ]]; then
  log "ERROR: NVIDIA orchestrator script not found or not executable at ${NVIDIA_ORCH_SCRIPT}"
  exit 1
fi

log "Invoking NVIDIA orchestrator: ${NVIDIA_ORCH_SCRIPT}"
set +e
"$NVIDIA_ORCH_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
NVIDIA_STATUS=$?
set -e

log "NVIDIA orchestrator exit status: ${NVIDIA_STATUS}"

if [[ "$NVIDIA_STATUS" -ne 0 ]]; then
  log "Overall CI result: FAILURE (NVIDIA build or script errored)"
  exit "$NVIDIA_STATUS"
fi

# 4) AMD tier
if [[ ! -x "$AMD_ORCH_SCRIPT" ]]; then
  log "ERROR: AMD orchestrator script not found or not executable at ${AMD_ORCH_SCRIPT}"
  exit 1
fi

log "Invoking AMD orchestrator: ${AMD_ORCH_SCRIPT}"
set +e
"$AMD_ORCH_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
AMD_STATUS=$?
set -e

log "AMD orchestrator exit status: ${AMD_STATUS}"

if [[ "$AMD_STATUS" -ne 0 ]]; then
  log "Overall CI result: FAILURE (AMD build or script errored)"
  exit "$AMD_STATUS"
fi

log "Overall CI result: SUCCESS (NUC, NVIDIA, and AMD builds/tests OK)"
exit 0
