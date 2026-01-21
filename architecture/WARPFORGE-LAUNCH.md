# WarpForge Launch Architecture

This document describes the distributed job submission and execution infrastructure for WarpForge, covering scheduler integration, GPU binary distribution, and the path from development to production deployment.

## Overview

WarpForge Launch provides a unified interface for submitting ML compilation and execution jobs to heterogeneous compute infrastructure:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        warpforge-launch-cli                         │
│                                                                     │
│   warpforge-launch submit --source model.py --gpu NVIDIA            │
│   warpforge-launch status <job-id>                                  │
│   warpforge-launch cluster-info                                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       warpforge-launch-core                         │
│                                                                     │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│   │ RayScheduler│  │SlurmScheduler│  │  K8sScheduler│  │  Local   │ │
│   └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
    ┌───────────┐     ┌───────────┐     ┌───────────┐
    │ Ray Cluster│    │ Slurm HPC │    │ Kubernetes │
    │ (SCP bins) │    │(Singularity)│   │ (Docker)   │
    └───────────┘     └───────────┘     └───────────┘
```

## Scheduler Implementations

### Scheduler Interface

All schedulers implement a common interface:

```java
public interface Scheduler extends AutoCloseable {
    String name();
    SchedulerCapabilities capabilities();
    String submit(JobSubmission submission);
    JobStatus status(String jobId);
    JobResult result(String jobId);
    boolean cancel(String jobId);
    List<JobStatus> list(JobQuery query);
    ClusterInfo clusterInfo();
}
```

### Ray Scheduler

**Use case:** Development, CI/CD, small-scale distributed execution

**Distribution method:** Native binaries via SCP to known path

```
NUC (Build Server)                    GPU Boxes (Ray Workers)
─────────────────                     ─────────────────────────
snakegrinder binary  ───SCP───►      /opt/warpforge/bin/snakegrinder
```

**How it works:**
1. Binaries are SCP'd to `/opt/warpforge/bin/` on Ray workers
2. RayScheduler submits jobs via Ray Jobs REST API
3. Entrypoint assumes `snakegrinder` is at known path
4. Results collected via Ray's result API

**Configuration:**
```java
public record RayConfig(
    String dashboardUrl,           // "http://localhost:8265"
    Duration connectionTimeout,
    Duration requestTimeout,
    String snakegrinderPath        // "/opt/warpforge/bin/snakegrinder"
)
```

**Why SCP for Ray:**
- Matches existing Holmes Mark 1 CI pattern (SSH + SCP)
- No container overhead for local development
- Fastest iteration loop
- Ray workers are controlled machines (not ephemeral pods)

### Slurm Scheduler

**Use case:** Academic HPC clusters, NVIDIA DGX SuperPOD, national labs

**Distribution method:** Singularity/Apptainer containers

```
NUC (Build Server)                    Slurm Cluster
─────────────────                     ──────────────
warpforge:nvidia.sif  ───Push───►    Shared storage / registry
                                              │
                                              ▼
                                     singularity run --nv warpforge.sif
```

**How it works:**
1. NUC builds container image, converts to Singularity SIF
2. SlurmScheduler generates SBATCH script with `singularity run`
3. Script uploaded via SSH, submitted with `sbatch`
4. Status polled via `squeue` and `sacct`

**Generated batch script:**
```bash
#!/bin/bash
#SBATCH --job-name=warpforge-${CORRELATION_ID}
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

singularity run --nv /shared/containers/warpforge-nvidia.sif \
    snakegrinder --trace-with-values \
    --source ${MODEL_SOURCE} \
    --out /scratch/output-${SLURM_JOB_ID}
```

**Why containers for Slurm:**
- HPC standard (Singularity is the de facto container runtime)
- Portable across clusters
- No need to install software on shared login nodes
- Reproducible execution environment

### Kubernetes Scheduler

**Use case:** Cloud deployment, enterprise on-prem, production scale-out

**Distribution method:** Docker containers from registry

```
NUC (Build Server)                    Kubernetes Cluster
─────────────────                     ──────────────────
docker build + push  ───►   nuc.local:5000/warpforge:nvidia
                                              │
                                              ▼
                                     Pod with GPU resource request
```

**How it works:**
1. NUC builds container image, pushes to local registry
2. KubernetesScheduler creates V1Job with GPU resource requests
3. Pod pulls image from registry, executes snakegrinder
4. Status tracked via Kubernetes Jobs API

**Pod specification:**
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: warpforge-${CORRELATION_ID}
  labels:
    warpforge.io/job-id: ${JOB_ID}
spec:
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        accelerator: nvidia
      containers:
      - name: warpforge
        image: nuc.local:5000/warpforge:nvidia-${VERSION}
        command: ["/opt/warpforge/bin/snakegrinder", ...]
        resources:
          limits:
            nvidia.com/gpu: 1
```

**Why containers for Kubernetes:**
- Kubernetes-native deployment model
- Immutable artifacts (same image tested = same image deployed)
- Scales to thousands of nodes
- GPU resource scheduling via device plugins

## Distribution Strategy: The Hybrid Model

Different schedulers have different deployment patterns. WarpForge uses a **hybrid model** that matches each scheduler's natural distribution mechanism:

| Scheduler | Distribution | Registry | Rationale |
|-----------|-------------|----------|-----------|
| **Ray** | SCP to known path | None | Fast iteration, controlled workers |
| **Slurm** | Singularity container | Local or shared FS | HPC standard |
| **Kubernetes** | Docker container | Local registry | Cloud-native standard |

### Why Not Cloud Registry?

For Holmes Mark 1 (home lab CI), cloud registries add cost and latency:

- **GitHub Container Registry (ghcr.io):** Free storage, but egress costs money
- **ML container images:** 5-20GB each (PyTorch + CUDA libs)
- **Per CI run:** Pull to multiple GPU boxes = significant bandwidth

**Solution:** Local registry on NUC

```
┌─────────────────────────────────────────────────────────────────┐
│  NUC (192.168.1.x)                                              │
│  ├── Docker Registry on port 5000                               │
│  ├── Builds NVIDIA + AMD container images                       │
│  └── Pushes to nuc.local:5000/warpforge:*                      │
└─────────────────────────────────────────────────────────────────┘
                    │ LAN only (gigabit, free)
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│  NVIDIA Box   │       │   AMD Box     │
│  docker pull  │       │  docker pull  │
│  nuc.local:   │       │  nuc.local:   │
│  5000/...     │       │  5000/...     │
└───────────────┘       └───────────────┘
```

Benefits:
- Zero internet bandwidth cost
- Gigabit local network speed
- Works offline
- One `docker run registry:2` to set up

## Build Artifacts

The NUC build server produces:

| Artifact | Format | Distribution Target |
|----------|--------|---------------------|
| `snakegrinder` | Native binary (Linux x86_64) | Ray workers via SCP |
| `warpforge:nvidia` | Docker image | K8s via local registry |
| `warpforge:amd` | Docker image | K8s via local registry |
| `warpforge-nvidia.sif` | Singularity image | Slurm via shared FS |
| `warpforge-amd.sif` | Singularity image | Slurm via shared FS |

### Build Commands

```bash
# Build native binary (for Ray)
./gradlew :snakegrinder-dist:assembleDist

# Build container images (for Slurm/K8s) - future
./gradlew :snakegrinder-dist:buildContainers

# Distribute to Ray workers
./holmes-lab/mark1/ci-scripts/smoke-test-gpu-binaries.sh
```

## GPU Resource Mapping

The `ResourceRequirements` record specifies hardware needs:

```java
public record ResourceRequirements(
    GpuType gpuType,        // NONE, NVIDIA, AMD, ANY
    int gpuCount,
    long memoryMb,
    int cpuCores,
    String queue,
    int priority,
    Set<String> nodeAffinity
)
```

Each scheduler maps this to platform-specific format:

| Scheduler | NVIDIA GPU Request | AMD GPU Request |
|-----------|-------------------|-----------------|
| **Ray** | `{"GPU": N}` | `{"GPU": N}` |
| **Slurm** | `#SBATCH --gres=gpu:nvidia:N` | `#SBATCH --gres=gpu:amd:N` |
| **Kubernetes** | `nvidia.com/gpu: N` + nodeSelector | `amd.com/gpu: N` + nodeSelector |

## Holmes Mark 1 Integration

The Holmes Mark 1 lab uses the Ray + SCP path for CI:

```
┌─────────────────────────────────────────────────────────────────┐
│  GitHub Actions                                                 │
│  └── Triggers orchestrate-nuc-build.sh on NUC                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  NUC Orchestrator                                               │
│  1. Build WarpForge (./gradlew clean assemble)                 │
│  2. Run tests (./gradlew test)                                 │
│  3. Build snakegrinder-dist                                    │
│  4. Run smoke-test-gpu-binaries.sh ◄── NEW                     │
│     - SCP binaries to GPU boxes                                │
│     - Verify execution on each box                             │
│  5. Trigger NVIDIA/AMD box tests                               │
└─────────────────────────────────────────────────────────────────┘
```

### Smoke Test Script

`holmes-lab/mark1/ci-scripts/smoke-test-gpu-binaries.sh` validates the SCP + Ray path:

**Phases:**
1. Verify local `snakegrinder-dist/build/dist` exists
2. Wake NVIDIA box, SCP binaries, run verification tests
3. Wake AMD box, SCP binaries, run verification tests
4. Report results

**Verification tests per GPU box:**
1. Binary exists at `/opt/warpforge/bin/snakegrinder`
2. `--version` executes successfully
3. `--help` executes successfully
4. `--trace-example` produces valid output
5. Output contains StableHLO markers

**Usage:**
```bash
# After building snakegrinder-dist
./gradlew :snakegrinder-dist:assembleDist
./holmes-lab/mark1/ci-scripts/smoke-test-gpu-binaries.sh

# Skip wake-on-LAN (boxes already running)
WAKE_BOXES=0 ./holmes-lab/mark1/ci-scripts/smoke-test-gpu-binaries.sh

# Fail on unreachable boxes (strict mode)
SKIP_IF_UNREACHABLE=0 ./holmes-lab/mark1/ci-scripts/smoke-test-gpu-binaries.sh
```

**Configuration:**
| Variable | Default | Purpose |
|----------|---------|---------|
| `DIST_DIR_OVERRIDE` | Auto-detected | Path to snakegrinder-dist build |
| `INSTALL_DIR_OVERRIDE` | `/opt/warpforge` | Install location on GPU boxes |
| `NVIDIA_HOST_OVERRIDE` | `nvidia` | SSH hostname for NVIDIA box |
| `AMD_HOST_OVERRIDE` | `amd` | SSH hostname for AMD box |
| `WAKE_BOXES` | `1` | Send Wake-on-LAN before testing |
| `SKIP_IF_UNREACHABLE` | `1` | Skip vs fail on unreachable boxes |

## Job Submission Flow

### Example: Submit PyTorch Model via Ray

```bash
# 1. Ensure binaries are distributed (done by CI or manually)
./holmes-lab/mark1/ci-scripts/smoke-test-gpu-binaries.sh

# 2. Submit job
warpforge-launch submit \
    --source models/resnet18.py \
    --class ResNet18 \
    --inputs '[(1,3,224,224):f32]' \
    --scheduler ray \
    --gpu NVIDIA \
    --gpu-count 1

# 3. Check status
warpforge-launch status ray-abc123

# 4. Get results
warpforge-launch result ray-abc123
```

### What Happens Internally

1. CLI parses arguments into `JobDefinition`
2. `SchedulerRegistry.get("ray")` returns `RayScheduler`
3. `RayScheduler.submit()` builds JSON payload:
   ```json
   {
     "entrypoint": "/opt/warpforge/bin/snakegrinder --trace-with-values --source models/resnet18.py ...",
     "entrypoint_resources": {"CPU": 4, "GPU": 1}
   }
   ```
4. HTTP POST to Ray Jobs API
5. Job runs on Ray worker with GPU
6. Results returned via Ray

## Future: Container Infrastructure

When containerization is needed (Slurm/K8s deployment), the plan is:

### Dockerfiles

```
containers/
├── nvidia/
│   └── Dockerfile      # Multi-stage: build + runtime
├── amd/
│   └── Dockerfile
└── base/
    └── Dockerfile.build
```

### NVIDIA Dockerfile (sketch)

```dockerfile
# Stage 1: Build
FROM nvidia/cuda:12.4-devel-ubuntu22.04 AS builder
RUN apt-get update && apt-get install -y openjdk-25-jdk
COPY . /src
WORKDIR /src
RUN ./gradlew :snakegrinder-dist:assembleDist

# Stage 2: Runtime (minimal)
FROM nvidia/cuda:12.4-runtime-ubuntu22.04
COPY --from=builder /src/snakegrinder-dist/build/dist /opt/warpforge
ENV PATH="/opt/warpforge/bin:$PATH"
ENTRYPOINT ["snakegrinder"]
```

### Local Registry Setup

```bash
# On NUC - one-time setup
docker run -d \
  -p 5000:5000 \
  --restart=always \
  --name registry \
  -v /opt/registry:/var/lib/registry \
  registry:2

# Configure GPU boxes to trust local registry
# /etc/docker/daemon.json on each GPU box:
{
  "insecure-registries": ["nuc.local:5000"]
}
```

## Related Documents

- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system architecture
- [backend-phases.md](backend-phases.md) - GPU backend development phases
- [CLAUDE.md](../CLAUDE.md) - Build commands and development workflow

## Design Principles

### Match the Scheduler's Natural Pattern

Don't force containers where binaries work (Ray). Don't force binaries where containers are expected (Kubernetes).

### Local-First for Development

Holmes Mark 1 is a home lab. Optimize for:
- Zero cloud costs
- Fast local network
- Offline capability

### Production-Ready Path

The container path (Slurm/K8s) mirrors how large datacenters deploy:
- Immutable artifacts
- Registry-based distribution
- No build tools on GPU servers
