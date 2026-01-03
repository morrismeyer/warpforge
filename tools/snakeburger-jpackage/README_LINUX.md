# SnakeBurger Linux packaging (Goals 1 and 2)

This keeps Linux packaging intentionally simple:

- Produces a **jpackage app-image** (a directory with `bin/` + the bundled runtime).
- Produces a **tar.gz** you can copy to another similar Linux host.
- No deb/rpm, no distro-specific install logic.

## Build

Run the build script:

- `tools/snakeburger-jpackage/build-snakeburger-cli-linux-appimage.sh`

Optional environment variables:

- `SNAKEBURGER_MAIN_CLASS` (defaults to auto-detect from the Gradle launcher)
- `SNAKEBURGER_APP_NAME` (defaults to `snakeburger`)

Outputs:

- Directory: `snakeburger-cli/build/snakeburger-cli-jpackage-linux/snakeburger/`
- Tarball: `snakeburger-cli/build/distributions/snakeburger-cli-linux-appimage-<version>.tar.gz`

## Run

- `tools/snakeburger-jpackage/run-snakeburger-cli-linux-appimage.sh --help`

Or after extracting the tarball:

- `./snakeburger/bin/snakeburger --help`
