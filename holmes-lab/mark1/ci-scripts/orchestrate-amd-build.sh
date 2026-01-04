#!/usr/bin/env bash
set -euo pipefail

# Orchestrated build runner for the AMD box.
#
# Intended to run ON the NUC. It:
# - Wakes the AMD box
# - Waits for SSH
# - Updates the WarpForge repo on the AMD box
# - Runs build + tests on the AMD box
# - Optionally powers the box off if successful

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# -------- CONFIG (override via environment variables) --------

TARGET_HOST="${TARGET_HOST_OVERRIDE:-amd}"
TARGET_MAC="${TARGET_MAC_OVERRIDE:-30:56:0f:20:67:79}"

# Broadcast IP for your LAN (Wake-on-LAN uses broadcast, not the sleeping host IP)
BROADCAST_IP="${BROADCAST_IP_OVERRIDE:-192.168.1.255}"

# Path to the repo on the AMD box
REMOTE_REPO_DIR="${REMOTE_REPO_DIR_OVERRIDE:-/home/morris/surfworks/warpforge}"

# WarpForge repo URL for cloning if missing
WARP_FORGE_REPO_URL="${WARP_FORGE_REPO_URL_OVERRIDE:-git@github.com:morrismeyer/warpforge.git}"

# Branch to build; GitHub Actions sets GITHUB_REF_NAME
BRANCH="${GITHUB_REF_NAME:-main}"

# Build and test commands to run on the AMD box
BUILD_CMD="${BUILD_CMD_OVERRIDE:-./gradlew clean assemble}"
TEST_CMD="${TEST_CMD_OVERRIDE:-./gradlew test}"

# SSH wait behavior
# - Set SKIP_IF_UNREACHABLE=1 to continue the overall orchestrator even if the target box is offline.
# - Set SKIP_IF_UNREACHABLE=0 to fail fast if the target box never becomes reachable.
SSH_CONNECT_TIMEOUT_SECONDS="${SSH_CONNECT_TIMEOUT_SECONDS:-5}"
SSH_POLL_INTERVAL_SECONDS="${SSH_POLL_INTERVAL_SECONDS:-10}"
SSH_MAX_WAIT_SECONDS="${SSH_MAX_WAIT_SECONDS:-60}"
SKIP_IF_UNREACHABLE="${SKIP_IF_UNREACHABLE:-1}"

# If 1, power off AMD only when build + tests succeed
POWER_OFF_ON_SUCCESS="${POWER_OFF_ON_SUCCESS:-1}"

# Where to store logs on the NUC
LOG_ROOT="${LOG_ROOT_OVERRIDE:-$HOME/build-logs}"

# -------- END CONFIG --------

mkdir -p "$LOG_ROOT"
RUN_ID="${GITHUB_RUN_ID:-manual-$(date +%Y%m%d-%H%M%S)}"
LOG_FILE="$LOG_ROOT/amd-${RUN_ID}.log"

log() {
  echo "[$(date)] $*" | tee -a "$LOG_FILE"
}

log "Starting orchestrated build on ${TARGET_HOST}, run ID ${RUN_ID}"
log "Branch: ${BRANCH}"
log "Remote repo dir: ${REMOTE_REPO_DIR}"
log "Repo URL: ${WARP_FORGE_REPO_URL}"
log "Build cmd: ${BUILD_CMD}"
log "Test cmd:  ${TEST_CMD}"
log "Broadcast: ${BROADCAST_IP}"

# 1) Wake the GPU box
log "Waking ${TARGET_HOST} (MAC ${TARGET_MAC})..."

WAKE_SCRIPT="${WAKE_SCRIPT_OVERRIDE:-$SCRIPT_DIR/wake-amd.sh}"
if [[ -x "$WAKE_SCRIPT" ]]; then
  MAC_OVERRIDE="$TARGET_MAC" BROADCAST_IP_OVERRIDE="$BROADCAST_IP" "$WAKE_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
else
  if command -v wakeonlan >/dev/null 2>&1; then
    wakeonlan -i "$BROADCAST_IP" "$TARGET_MAC" 2>&1 | tee -a "$LOG_FILE"
  else
    log "ERROR: wakeonlan not found and wake script missing at ${WAKE_SCRIPT}"
    exit 1
  fi
fi

# 2) Wait for SSH to come up
log "Waiting for SSH on ${TARGET_HOST} (max ${SSH_MAX_WAIT_SECONDS}s)..."
SSH_OK=0
START_TS="$(date +%s)"
ATTEMPT=0
while true; do
  ATTEMPT=$((ATTEMPT + 1))
  if ssh -o ConnectTimeout="${SSH_CONNECT_TIMEOUT_SECONDS}" -o BatchMode=yes "$TARGET_HOST" "echo up" >/dev/null 2>&1; then
    log "${TARGET_HOST} is up (SSH reachable)"
    SSH_OK=1
    break
  fi

  NOW_TS="$(date +%s)"
  ELAPSED=$((NOW_TS - START_TS))
  if (( ELAPSED >= SSH_MAX_WAIT_SECONDS )); then
    break
  fi
  log "...still waiting for SSH (${ATTEMPT}, elapsed ${ELAPSED}s)"
  sleep "${SSH_POLL_INTERVAL_SECONDS}"
done

if [[ "$SSH_OK" -ne 1 ]]; then
  if [[ "$SKIP_IF_UNREACHABLE" -eq 1 ]]; then
    log "WARN: ${TARGET_HOST} did not become reachable within ${SSH_MAX_WAIT_SECONDS}s. Skipping remote build (SKIP_IF_UNREACHABLE=1)."
    exit 0
  fi
  log "ERROR: ${TARGET_HOST} did not come up in time (SKIP_IF_UNREACHABLE=0)"
  exit 1
fi

# 3) Run build + tests on the AMD box
log "Starting build + test on ${TARGET_HOST} in ${REMOTE_REPO_DIR}"

set +e  # capture status rather than exiting immediately on ssh failure

ssh "$TARGET_HOST" bash -lc "
  set -euo pipefail

  echo \"[remote \$(date)] Preparing repo in ${REMOTE_REPO_DIR}\"
  mkdir -p \"${REMOTE_REPO_DIR}\"
  cd \"${REMOTE_REPO_DIR}\"

  if [ ! -d .git ]; then
    echo \"[remote \$(date)] Cloning repo...\"
    git clone \"${WARP_FORGE_REPO_URL}\" .
  fi

  echo \"[remote \$(date)] Fetching origin...\"
  git fetch origin

  if git show-ref --verify --quiet \"refs/remotes/origin/${BRANCH}\"; then
    if git show-ref --verify --quiet \"refs/heads/${BRANCH}\"; then
      git checkout \"${BRANCH}\"
    else
      git checkout -b \"${BRANCH}\" \"origin/${BRANCH}\"
    fi
    git reset --hard \"origin/${BRANCH}\"
  else
    echo \"[remote \$(date)] ERROR: origin/${BRANCH} does not exist\" >&2
    exit 1
  fi

  echo \"[remote \$(date)] Running build: ${BUILD_CMD}\"
  bash -lc \"${BUILD_CMD}\"

  echo \"[remote \$(date)] Running tests: ${TEST_CMD}\"
  bash -lc \"${TEST_CMD}\"

  echo \"[remote \$(date)] Build + tests completed successfully\"
" 2>&1 | tee -a "$LOG_FILE"

BUILD_STATUS=$?
set -e

log "Remote build + test finished with status ${BUILD_STATUS}"

# 4) Decide whether to power off the AMD box
if [[ "$BUILD_STATUS" -eq 0 ]]; then
  log "Build + tests succeeded."
  if [[ "$POWER_OFF_ON_SUCCESS" -eq 1 ]]; then
    log "POWER_OFF_ON_SUCCESS=1 -> powering off ${TARGET_HOST}..."
    ssh -o BatchMode=yes "$TARGET_HOST" "sudo /usr/sbin/shutdown -h now" || true
  else
    log "POWER_OFF_ON_SUCCESS=0 -> leaving ${TARGET_HOST} running"
  fi
else
  log "Build + tests FAILED (status=${BUILD_STATUS}) -> leaving ${TARGET_HOST} running for debugging."
fi

exit "$BUILD_STATUS"
