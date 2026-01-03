# SnakeBurger packaging (Linux)

These scripts build Linux installer artifacts using `jpackage`.

## Goals (Linux)

1. Build a runnable stage image (via `snakeburger-app-image` tooling).
2. Build installer artifacts for Linux: `.deb` and `.rpm` using `jpackage`.
3. (Deferred) Multi-distro, fully portable packaging.

We currently focus on Goals 1 and 2.

## Build a `.deb`

- Script: `tools/snakeburger-jpackage/build-snakeburger-cli-deb.sh`
- Output is copied to: `snakeburger-cli/build/distributions/*.deb`

Prerequisites on Ubuntu/Debian:

- `dpkg-deb` (package: `dpkg-dev`)

## Build an `.rpm`

- Script: `tools/snakeburger-jpackage/build-snakeburger-cli-rpm.sh`
- Output is copied to: `snakeburger-cli/build/distributions/*.rpm`

Prerequisites on Ubuntu/Debian:

- `rpmbuild` (provided by package: `rpm`)

Note: the RPM script builds a temporary `jpackage --type app-image` first (reusing the staged Babylon runtime) to avoid jpackage warnings.


## Configuration

Optional environment variables:

- `SNAKEBURGER_MAIN_CLASS` (default: `io.surfworks.snakeburger.cli.SnakeBurgerMain`)
- `SNAKEBURGER_APP_NAME` (default: `Snake Burger`)
- `SNAKEBURGER_PKG_NAME` (default: `snakeburger`)
- `SNAKEBURGER_APP_VERSION_RAW` (default: derived from jar file name)
- `SNAKEBURGER_JPACKAGE_BIN` (override path to `jpackage`)
