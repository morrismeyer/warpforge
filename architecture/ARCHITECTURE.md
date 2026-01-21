# WarpForge Architecture

This document describes the high-level architecture of WarpForge, a Java-centric ML compiler stack that bridges PyTorch models to GPU execution via Babylon Code Reflection.

## Vision

WarpForge is building a complete ML-to-GPU pipeline where:

1. **SnakeGrinder** captures PyTorch models with full fidelity
2. **SnakeBurger** transforms them into Babylon Code Reflection IR
3. **WarpForge backends** execute on NVIDIA and AMD GPUs

The key insight: by using Babylon's Code Reflection API, we can treat GPU kernels as **code models** that can be analyzed, transformed, and lowered to multiple targets from a single representation.

## System Overview

```
PyTorch Model (nn.Module)
         │
         ▼
┌─────────────────────────────────────┐
│  SnakeGrinder                       │
│  - GraalPy 25 + PyTorch 2.7         │
│  - torch.fx.symbolic_trace          │
│  - FX Graph → StableHLO MLIR        │
│  - Native executable distribution   │
└──────────────┬──────────────────────┘
               │ .mlir file (StableHLO text)
               ▼
┌─────────────────────────────────────┐
│  SnakeBurger                        │
│  - Babylon JDK (Java 26)            │
│  - StableHLO Parser                 │
│  - Babylon Code Reflection IR       │
│  - jlink/jpackage distribution      │
└──────────────┬──────────────────────┘
               │ Code Model
               ▼
┌─────────────────────────────────────┐
│  WarpForge Backend                  │
│  - Op-by-op execution (Phase 1)     │
│  - Fused kernel generation (Phase 2+)│
│  - NVIDIA (CUDA/PTX) + AMD (ROCm)   │
└─────────────────────────────────────┘
```

## Key Architectural Decisions

### StableHLO as the Interchange Format

StableHLO MLIR text format serves as the **stable interface** between Python and Java worlds:

- **Versioned and standardized** - StableHLO has backward compatibility guarantees
- **Text-based** - Human-readable, easy to debug, no binary compatibility issues
- **Complete** - Captures all operations needed for ML workloads

This decouples SnakeGrinder from SnakeBurger evolution. Either can be updated independently as long as they agree on StableHLO.

### Babylon Code Reflection for Kernel Generation

Babylon's Code Reflection API treats **code as data**. This is ideal for:

- **Analysis** - Walk the operation graph, identify patterns
- **Transformation** - Fuse operations, optimize memory access
- **Code generation** - Lower to PTX, HIP, or other targets

See [Backend Phases](backend-phases.md) for the phased approach to kernel generation.

### Java-First Philosophy

When implementation choices are equivalent, prefer Java. This means:

- Fusion analysis and kernel generation happen in Java (not Python)
- The intelligence lives in WarpForge/Babylon, not SnakeGrinder
- SnakeGrinder captures faithfully; WarpForge optimizes

## Module Map

| Module | Purpose | JDK |
|--------|---------|-----|
| `snakegrinder-core` | GraalPy polyglot context | GraalVM 25 |
| `snakegrinder-cli` | CLI for model tracing | GraalVM 25 |
| `snakegrinder-dist` | Native distribution build | GraalVM 25 |
| `snakeburger-core` | StableHLO → Babylon IR | Babylon (Java 26) |
| `snakeburger-cli` | CLI for IR operations | Babylon (Java 26) |
| `warpforge-core` | Core IR and analysis | Java 25 |
| `warpforge-backend-cpu` | CPU reference backend | Java 25 |
| `warpforge-backend-nvidia` | NVIDIA GPU backend | Java 25 |
| `warpforge-backend-amd` | AMD GPU backend | Java 25 |
| `warpforge-codegen-api` | Kernel codegen interfaces | Java 25 |
| `warpforge-launch-core` | Scheduler integration (Ray, Slurm, K8s) | Java 25 |
| `warpforge-launch-cli` | CLI for job submission | Java 25 |

## Related Documents

- [Backend Phases](backend-phases.md) - Phased approach from op-by-op to fused kernels
- [Backend Kernel Instrumentation](BACKEND-KERNEL-INSTRUMENTATION.md) - Salt-based observability strategy
- [WarpForge Launch](WARPFORGE-LAUNCH.md) - Distributed job submission, scheduler integration, GPU binary distribution
- [CLAUDE.md](../CLAUDE.md) - Build commands, development workflow, code style

## Design Principles

### "It Just Works"

End users receive self-contained distributions. No environment variables, no PATH manipulation, no pip install. Extract and run.

### Correctness First, Then Performance

Phase 1 establishes correctness with op-by-op execution. Fusion optimizations come later, built on a foundation of verified behavior.

### Single Source of Truth

- Versions in `snakegrinder-dist/versions.env`
- Architecture in `architecture/`
- Build instructions in `CLAUDE.md`

No duplicated configuration that can drift out of sync.
