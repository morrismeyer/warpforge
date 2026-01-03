#!/usr/bin/env bash
set -euo pipefail

# -------- CONFIGURABLE SECTION --------

TARGET_HOST="amd"
TARGET_IP="192.168.1.162"
TARGET_MAC="30:56:0f:20:67:79"       # <-- amd MAC (enp10s0)

# Path to the repo on nvidia
REMOTE_REPO_DIR="/home/morris/projects/warpforge"   # adjust if needed

# Branch to build; GitHub Actions sets GITHUB_REF_NAME,
# fallback to main for manual runs
BRANCH="${GITHUB_REF_NAME:-main}"

# Build and test commands to run on nvidia
BUILD_CMD="${BUILD_CMD_OVERRIDE:-./gradlew clean assemble amdTest}"
TEST_CMD="${TEST_CMD_OVERRIDE:-true}"


# If 1, power off nvidia only when build+tests both succeed
POWER_OFF_ON_SUCCESS="${POWER_OFF_ON_SUCCESS:-1}"

# Where to store logs on nuc
LOG_ROOT="$HOME/build-logs"

# -------- END CONFIG SECTION --------

mkdir -p "$LOG_ROOT"
RUN_ID="${GITHUB_RUN_ID:-manual-$(date +%Y%m%d-%H%M%S)}"
LOG_FILE="$LOG_ROOT/nvidia-${RUN_ID}.log"

echo "[$(date)] Starting orchestrated build on $TARGET_HOST, run ID $RUN_ID" | tee -a "$LOG_FILE"
echo "[$(date)] Branch: $BRANCH" | tee -a "$LOG_FILE"
echo "[$(date)] Remote repo dir: $REMOTE_REPO_DIR" | tee -a "$LOG_FILE"
echo "[$(date)] Build cmd: $BUILD_CMD" | tee -a "$LOG_FILE"
echo "[$(date)] Test cmd:  $TEST_CMD" | tee -a "$LOG_FILE"

# 1. Wake the GPU box
echo "[$(date)] Waking $TARGET_HOST (MAC $TARGET_MAC)..." | tee -a "$LOG_FILE"

if [[ -x "$HOME/ci/wake-nvidia.sh" ]]; then
  "$HOME/ci/wake-nvidia.sh" | tee -a "$LOG_FILE"
else
  if command -v wakeonlan >/dev/null 2>&1; then
    # Use directed broadcast on the LAN
    wakeonlan -i 192.168.1.162 "$TARGET_MAC" | tee -a "$LOG_FILE"
  else
    echo "[$(date)] ERROR: wakeonlan not found and wake-nvidia.sh missing" | tee -a "$LOG_FILE"
    exit 1
  fi
fi

# 2. Wait for SSH to come up
echo "[$(date)] Waiting for SSH on $TARGET_HOST..." | tee -a "$LOG_FILE"
SSH_OK=0
for i in {1..30}; do
  if ssh -o ConnectTimeout=5 -o BatchMode=yes "$TARGET_HOST" "echo up" >/dev/null 2>&1; then
    echo "[$(date)] $TARGET_HOST is up (SSH reachable)" | tee -a "$LOG_FILE"
    SSH_OK=1
    break
  fi
  echo "[$(date)] ...still waiting for SSH ($i)" | tee -a "$LOG_FILE"
  sleep 10
done

if [[ "$SSH_OK" -ne 1 ]]; then
  echo "[$(date)] ERROR: $TARGET_HOST did not come up in time" | tee -a "$LOG_FILE"
  exit 1
fi

# 3. Run build + tests on the GPU box
echo "[$(date)] Starting build + test on $TARGET_HOST in $REMOTE_REPO_DIR" | tee -a "$LOG_FILE"

set +e  # we want to capture status rather than exit immediately on ssh failure

ssh "$TARGET_HOST" bash -lc "
  set -euo pipefail

  echo \"[remote \$(date)] Preparing repo in $REMOTE_REPO_DIR\"
  mkdir -p \"$REMOTE_REPO_DIR\"
  cd \"$REMOTE_REPO_DIR\"

  if [ ! -d .git ]; then
    echo \"[remote \$(date)] Cloning repo...\"
    # TODO: replace this with your actual repo URL
    git clone git@github.com:YOURUSER/YOURREPO.git .
  fi

  echo \"[remote \$(date)] Fetching latest...\"
  git fetch origin

  echo \"[remote \$(date)] Checking out branch $BRANCH\"
  git checkout \"$BRANCH\"

  echo \"[remote \$(date)] Pulling branch $BRANCH\"
  git pull --ff-only origin \"$BRANCH\"

  echo \"[remote \$(date)] Running build: $BUILD_CMD\"
  $BUILD_CMD

  echo \"[remote \$(date)] Running tests: $TEST_CMD\"
  $TEST_CMD

  echo \"[remote \$(date)] Build + tests completed successfully\"
" 2>&1 | tee -a "$LOG_FILE"

BUILD_STATUS=$?
set -e

echo "[$(date)] Remote build+test finished with status $BUILD_STATUS" | tee -a "$LOG_FILE"

# 4. Decide whether to power off the GPU box
if [[ "$BUILD_STATUS" -eq 0 ]]; then
  echo "[$(date)] Build+tests succeeded." | tee -a "$LOG_FILE"

  if [[ "$POWER_OFF_ON_SUCCESS" -eq 1 ]]; then
    # assumes a narrow sudoers rule on nvidia:
    # morris ALL=(ALL) NOPASSWD:/sbin/shutdown
    echo "[$(date)] POWER_OFF_ON_SUCCESS=1 -> powering off $TARGET_HOST..." | tee -a "$LOG_FILE"
    ssh -o BatchMode=yes "$TARGET_HOST" "sudo /usr/sbin/shutdown -h now" || true
  else
    echo "[$(date)] POWER_OFF_ON_SUCCESS=0 -> leaving $TARGET_HOST running" | tee -a "$LOG_FILE"
  fi
else
  echo "[$(date)] Build+tests FAILED (status=$BUILD_STATUS) -> leaving $TARGET_HOST running for debugging." | tee -a "$LOG_FILE"
fi

exit "$BUILD_STATUS"

