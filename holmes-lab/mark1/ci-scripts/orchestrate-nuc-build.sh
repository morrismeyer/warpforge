#!/usr/bin/env bash
set -euo pipefail

# NUC orchestration entrypoint.
#
# Intended to run ON the NUC. It:
# - Updates the local WarpForge repo on the NUC
# - Runs a NUC build + smoke test
# - If green, triggers the NVIDIA box build
# - If green, triggers the AMD box build

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# -------- CONFIG (override via environment variables) --------

# WarpForge repo on the NUC
# Default matches your recent layout: ~/surfworks/warpforge
NUC_REPO_DIR="${NUC_REPO_DIR_OVERRIDE:-$HOME/surfworks/warpforge}"

# Branch from GitHub Actions. Fallback to main for manual runs.
BRANCH="${GITHUB_REF_NAME:-main}"

# Build + smoke test commands on the NUC
NUC_BUILD_CMD="${NUC_BUILD_CMD_OVERRIDE:-./gradlew clean assemble}"
NUC_TEST_CMD="${NUC_TEST_CMD_OVERRIDE:-./gradlew test}"

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
else
  log "ERROR: origin/${BRANCH} does not exist. Check the branch name in GITHUB_REF_NAME."
  exit 1
fi

# 2) Build + smoke test on NUC
log "Running NUC build: ${NUC_BUILD_CMD}"
bash -lc "${NUC_BUILD_CMD}" 2>&1 | tee -a "$LOG_FILE"

log "Running NUC tests: ${NUC_TEST_CMD}"
bash -lc "${NUC_TEST_CMD}" 2>&1 | tee -a "$LOG_FILE"

log "NUC build + tests SUCCESS"

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
