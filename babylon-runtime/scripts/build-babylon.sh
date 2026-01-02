#!/usr/bin/env bash
set -euo pipefail
./gradlew :babylon-runtime:syncBabylon
./gradlew :babylon-runtime:configureBabylon
./gradlew :babylon-runtime:buildBabylonImages
./gradlew :babylon-runtime:verifyBabylonJdk
./gradlew :babylon-runtime:writeBabylonEnv
