# Snakeburger CLI App-Image (zip)

This creates a single zip that contains:
- `app/` (the Gradle `installDist` output for snakeburger-cli)
- `runtime/` (a Babylon JDK runtime, either from `jlink` or a full JDK copy as a fallback)
- `bin/snakeburger` (a small launcher that runs the CLI using the bundled runtime)

## Build

From repo root:

```bash
chmod +x tools/snakeburger-app-image/build-snakeburger-cli-appimage.sh
./tools/snakeburger-app-image/build-snakeburger-cli-appimage.sh
```

The zip is written to:

```
snakeburger-cli/build/distributions/*appimage*.zip
```

## Run

```bash
chmod +x tools/snakeburger-app-image/run-snakeburger-cli-appimage.sh
./tools/snakeburger-app-image/run-snakeburger-cli-appimage.sh --help
```

If you want to keep the unpacked temp dir for inspection:

```bash
SNAKEBURGER_KEEP_TMP=1 ./tools/snakeburger-app-image/run-snakeburger-cli-appimage.sh --help
```
