#!/usr/bin/env bash
set -euo pipefail
./gradlew :snakeburger-cli:jar
./gradlew :babylon-runtime:zipSnakeBurgerRuntimeImage
