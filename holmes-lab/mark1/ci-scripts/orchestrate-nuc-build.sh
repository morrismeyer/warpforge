#!/usr/bin/env bash
set -euo pipefail

# -------- CONFIG --------

# Local repo on nuc
NUC_REPO_DIR="$HOME/projects/warpforge"

# Branch from GitHub, fallback to main for manual runs
BRANCH="${GITHUB_REF_NAME:-main}"

# Build + smoke test commands on nuc (placeholders for now)
NUC_BUILD_CMD="${NUC_BUILD_CMD_OVERRIDE:-./gradlew clean assemble cpuTest}"
NUC_TEST_CMD="${NUC_TEST_CMD_OVERRIDE:-test}"

# Path to orchestrator scripts on NUC
NVIDIA_ORCH_SCRIPT="/home/morris/ci/orchestrate-nvidia-build.sh"
AMD_ORCH_SCRIPT="/home/morris/ci/orchestrate-amd-build.sh"

LOG_ROOT="$HOME/build-logs"
mkdir -p "$LOG_ROOT"

RUN_ID="${GITHUB_RUN_ID:-manual-$(date +%Y%m%d-%H%M%S)}"
LOG_FILE="$LOG_ROOT/nuc-${RUN_ID}.log"

echo "[$(date)] ==== NUC ORCHESTRATOR START (run $RUN_ID) ==== " | tee -a "$LOG_FILE"
echo "[$(date)] Branch: $BRANCH" | tee -a "$LOG_FILE"
echo "[$(date)] NUC repo dir: $NUC_REPO_DIR" | tee -a "$LOG_FILE"
echo "[$(date)] NUC build cmd: $NUC_BUILD_CMD" | tee -a "$LOG_FILE"
echo "[$(date)] NUC test  cmd: $NUC_TEST_CMD" | tee -a "$LOG_FILE"

# 1. Prepare repo on nuc
echo "[$(date)] Preparing repo on nuc..." | tee -a "$LOG_FILE"

mkdir -p "$(dirname "$NUC_REPO_DIR")"
if [ ! -d "$NUC_REPO_DIR/.git" ]; then
  echo "[$(date)] Repo not present, cloning..." | tee -a "$LOG_FILE"
  mkdir -p "$NUC_REPO_DIR"
  cd "$NUC_REPO_DIR"
  git clone git@github.com:morrismeyer/warpforge.git . | tee -a "$LOG_FILE"
else
  cd "$NUC_REPO_DIR"
fi

echo "[$(date)] Fetching latest from origin..." | tee -a "$LOG_FILE"
git fetch origin | tee -a "$LOG_FILE"

echo "[$(date)] Checking out branch $BRANCH..." | tee -a "$LOG_FILE"
git checkout "$BRANCH" | tee -a "$LOG_FILE"

echo "[$(date)] Pulling branch $BRANCH..." | tee -a "$LOG_FILE"
git pull --ff-only origin "$BRANCH" | tee -a "$LOG_FILE"

# 2. Build + smoke test on nuc (placeholders for now)
echo "[$(date)] Running NUC build: $NUC_BUILD_CMD" | tee -a "$LOG_FILE"
bash -lc "$NUC_BUILD_CMD" | tee -a "$LOG_FILE"

echo "[$(date)] Running NUC tests: $NUC_TEST_CMD" | tee -a "$LOG_FILE"
bash -lc "$NUC_TEST_CMD" | tee -a "$LOG_FILE"

echo "[$(date)] NUC build+tests SUCCESS" | tee -a "$LOG_FILE"

# 3. If nuc is clean, run NVIDIA orchestrator
if [ ! -x "$NVIDIA_ORCH_SCRIPT" ]; then
  echo "[$(date)] ERROR: NVIDIA orchestrator script not found or not executable at $NVIDIA_ORCH_SCRIPT" | tee -a "$LOG_FILE"
  exit 1
fi

echo "[$(date)] Invoking NVIDIA orchestrator: $NVIDIA_ORCH_SCRIPT" | tee -a "$LOG_FILE"

set +e
"$NVIDIA_ORCH_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
NVIDIA_STATUS=$?
set -e

echo "[$(date)] NVIDIA orchestrator exit status: $NVIDIA_STATUS" | tee -a "$LOG_FILE"

if [ "$NVIDIA_STATUS" -ne 0 ]; then
  echo "[$(date)] Overall CI result: FAILURE (NVIDIA build/tests failed or NVIDIA script errored)" | tee -a "$LOG_FILE"
  exit "$NVIDIA_STATUS"
fi

# 4. If NVIDIA is clean, run AMD orchestrator
if [ ! -x "$AMD_ORCH_SCRIPT" ]; then
  echo "[$(date)] ERROR: AMD orchestrator script not found or not executable at $AMD_ORCH_SCRIPT" | tee -a "$LOG_FILE"
  exit 1
fi

echo "[$(date)] Invoking AMD orchestrator: $AMD_ORCH_SCRIPT" | tee -a "$LOG_FILE"

set +e
"$AMD_ORCH_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
AMD_STATUS=$?
set -e

echo "[$(date)] AMD orchestrator exit status: $AMD_STATUS" | tee -a "$LOG_FILE"

if [ "$AMD_STATUS" -ne 0 ]; then
  echo "[$(date)] Overall CI result: FAILURE (AMD build/tests failed or AMD script errored)" | tee -a "$LOG_FILE"
  exit "$AMD_STATUS"
fi

# 5. If we got here, all three tiers are green
echo "[$(date)] Overall CI result: SUCCESS (NUC, NVIDIA, and AMD builds/tests OK)" | tee -a "$LOG_FILE"
exit 0
