# Snakeburger macOS packaging (jpackage)

This folder contains scripts to build a macOS `.app`, plus optional DMG and PKG artifacts.

## Phase A: build the `.app`

```bash
cd ~/surfworks/warpforge

# Optional (recommended)
export SNAKEBURGER_MAIN_CLASS=io.surfworks.snakeburger.cli.SnakeBurgerMain

# Optional: change the macOS app name (default is "Snake Burger")
# export SNAKEBURGER_APP_NAME="Snake Burger"

tools/snakeburger-jpackage/build-snakeburger-cli-jpackage.sh
tools/snakeburger-jpackage/run-snakeburger-cli-jpackage.sh --help
```

Outputs:
- `snakeburger-cli/build/snakeburger-cli-jpackage/<App Name>.app`
- `snakeburger-cli/build/distributions/snakeburger-cli-jpackage-<version>.zip`

## Phase B: build DMG or PKG (no install)

These scripts only **build** the artifacts. They do not mount, copy, or install anything.

```bash
# DMG
tools/snakeburger-jpackage/build-snakeburger-cli-dmg.sh

# PKG
tools/snakeburger-jpackage/build-snakeburger-cli-pkg.sh
```

Outputs:
- `snakeburger-cli/build/distributions/snakeburger-cli-<version>.dmg`
- `snakeburger-cli/build/distributions/snakeburger-cli-<version>.pkg`

## Notes

- If you see macOS spawn helper issues around `codesign`, these scripts set:
  `JAVA_TOOL_OPTIONS="... -Djdk.lang.Process.launchMechanism=FORK"`.
- `tzdb.dat` is ensured inside the staged runtime before running jpackage.
