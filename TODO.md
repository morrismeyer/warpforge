# TODO

## Build Infrastructure
- [x] Autodetect and install Babylon JDK build dependencies on macOS (Xcode CLI tools, autoconf, etc.)
- [x] Autodetect and install Babylon JDK build dependencies on Linux (gcc, make, autoconf, etc.)
- [x] Autodetect and install GraalPy PyTorch build dependencies (cmake, ninja, C++ compiler)

## Core Modules
- [ ] warpforge-core
- [ ] warpforge-core GPU JFR events
- [ ] warpforge-core Slurm, Ray, K8S integration
- [ ] warpforge-gpu-backend
- [ ] warpforge-cli
- [ ] warpforge-ucx-transport-ffm

## ML Pipeline
- [x] snakegrinder-core FX-to-StableHLO converter (torch.fx.symbolic_trace → MLIR)
- [ ] snakegrinder self-contained distribution (native-image + bundled libs, zero config)
- [ ] snakeburger-core StableHLO MLIR to Babylon Code Reflection IR

## Distribution ("It Just Works")
- [ ] Create wrapper script for snakegrinder native-image (sets library path automatically)
- [ ] Gradle task to assemble snakegrinder distribution directory
- [ ] Gradle task to create .tar.gz / .zip distribution archive
- [ ] Test distribution on clean macOS system (no dev tools installed)
- [ ] Test distribution on clean Linux system (no dev tools installed)

## Phase 2: TPU Support / XLA Compilation
Cloud TPU access requires XLA compilation. Since Google does not sell PCIe TPU cards
for local development, this is deferred to Phase 2 after AMD/NVIDIA GPU support is solid.

- [ ] StableHLO MLIR bytecode export (bundle `stablehlo-opt` tool)
- [ ] XLA compilation path for Cloud TPU targeting
- [ ] Integration testing with GCP Cloud TPU VMs
- [ ] Documentation for enterprise TPU deployment workflows

Note: StableHLO text format is sufficient for Phase 1 (Babylon IR path).
Bytecode is only needed for XLA toolchain handoff.