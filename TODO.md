# TODO

## Build Infrastructure
- [x] Autodetect and install Babylon JDK build dependencies on macOS (Xcode CLI tools, autoconf, etc.)
- [x] Autodetect and install Babylon JDK build dependencies on Linux (gcc, make, autoconf, etc.)
- [ ] Autodetect and install GraalPy PyTorch build dependencies (cmake, ninja, C++ compiler)

## Core Modules
- [ ] warpforge-core
- [ ] warpforge-core GPU JFR events
- [ ] warpforge-core Slurm, Ray, K8S integration
- [ ] warpforge-gpu-backend
- [ ] warpforge-cli
- [ ] warpforge-ucx-transport-ffm

## ML Pipeline
- [ ] snakegrinder-core PyTorch / JAX to StableHLO MLIR (GraalPy + real PyTorch)
- [ ] snakeburger-core StableHLO MLIR to Babylon Code Reflection IR