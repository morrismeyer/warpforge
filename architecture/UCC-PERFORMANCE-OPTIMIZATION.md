# UCC Collective Performance Optimization Roadmap

This document tracks performance optimization efforts for the UCC collective implementation targeting the Mellanox 100GbE cross-connect between NVIDIA and AMD GPU boxes.

## Completed Optimizations

### 1. Adaptive Polling Strategy (2026-01-22)

**Implementation:** `UccHelper.waitForCompletionWithProgress()`

The polling loop now uses an adaptive strategy instead of calling `ucc_context_progress()` on every iteration:

| Phase | Duration | Poll Interval | Rationale |
|-------|----------|---------------|-----------|
| Fast | First 100 iterations | Every iteration | Low latency for small ops |
| Medium | Up to 10ms | Every 10 status checks | Balance latency/overhead |
| Slow | After 10ms | Every 100 status checks | Minimize FFM overhead |

**Expected Impact:** 15-30% throughput improvement for 1MB+ operations

### 2. Eliminate Unnecessary Buffer Copy (2026-01-22)

**Implementation:** `UccCollectiveImpl.broadcast()`

Previously: All ranks copied input to output buffer before broadcast
Now: Only root rank copies; non-root ranks receive directly into uninitialized buffer

**Expected Impact:** 10-15% improvement for broadcast on non-root ranks

---

## Remaining Optimization Opportunities

### Priority 1: Arena Pooling (Medium Effort, 3-8% Impact)

**Current Issue:**
```java
try (Arena opArena = Arena.ofConfined()) {
    MemorySegment args = ucc_coll_args.allocate(opArena);
    // ... operation ...
}
```

Each operation creates and destroys a new confined arena.

**Proposed Solution:**
Create a pool of pre-allocated arenas at initialization:
```java
private final Arena[] arenaPool = new Arena[4];
private final AtomicInteger arenaIndex = new AtomicInteger();

Arena acquireArena() {
    return arenaPool[arenaIndex.getAndIncrement() % arenaPool.length];
}
```

### Priority 2: Dedicated Progress Thread (High Effort, 20-30% Impact)

**Current Issue:**
All UCC operations block the calling thread during the polling loop due to UCX thread affinity requirements.

**Proposed Architecture:**
```
┌─────────────────────────────────────┐
│  Main Application Thread            │
│  (submits operations, returns CF)   │
├─────────────────────────────────────┤
│  Progress Thread (dedicated)        │
│  - Calls ucc_context_progress()     │
│  - Polls request completions        │
│  - Completes CompletableFutures     │
└─────────────────────────────────────┘
```

Benefits:
- True async operations
- Single thread handles UCX affinity
- Better CPU utilization
- Enables computation/communication overlap

### Priority 3: UCC Algorithm Selection (Low Effort, 5-15% Impact)

**Current Configuration:**
```bash
UCC_TL_UCP_ALLREDUCE_ALG=knomial
```

**Optimization Options:**
```bash
# For large messages (1MB+)
UCC_TL_UCP_ALLREDUCE_ALG=ring

# Or let UCC choose based on message size
UCC_TL_UCP_ALLREDUCE_ALG=auto

# Additional tuning for RDMA
UCX_RNDV_SCHEME=get_zcopy
UCX_AFFINITY=core:0
```

### Priority 4: In-Place Operations (Low Effort, 5-10% Impact)

Where possible, use in-place variants to avoid allocating output tensors:
- `allReduceInPlace` instead of `allReduce` when result can overwrite input
- `broadcastInPlace` instead of `broadcast`

---

## Performance Testing Commands

### Run Performance Test (from NUC)
```bash
cd /path/to/warpforge/holmes-lab/mark1/ci-scripts
./perf-test-ucc-collectives.sh --quick  # Quick test
./perf-test-ucc-collectives.sh          # Full test
```

### Overnight Continuous Testing
```bash
./overnight-perf-iteration.sh --duration 8 --interval 15
```

### Manual Two-Node Test

On mark1nvidia:
```bash
./gradlew :warpforge-io:uccPerfMaster -Psize=16777216 -Piterations=100
```

On mark1amd:
```bash
./gradlew :warpforge-io:uccPerfWorker -Psize=16777216 -Piterations=100
```

---

## Performance Targets

| Metric | Target | Baseline 100GbE |
|--------|--------|-----------------|
| AllReduce 16MB | > 80 Gbps | 100 Gbps |
| Broadcast 16MB | > 85 Gbps | 100 Gbps |
| AllGather 16MB | > 75 Gbps | 100 Gbps |
| Barrier Latency | < 50 us | N/A |

These targets assume achievable throughput accounting for:
- UCC/UCX protocol overhead (~5-10%)
- FFM JNI-like overhead (~5-10%)
- Collective algorithm overhead (varies by operation)

---

## Profiling Tools

### JFR Profiling
```bash
./gradlew :warpforge-io:uccPerfTest \
    --args='--rank 0 ...' \
    -Djdk.jfr.startup=true
```

### UCX Debug Output
```bash
UCX_LOG_LEVEL=debug ./gradlew :warpforge-io:uccPerfTest ...
```

### UCC Debug Output
```bash
UCC_LOG_LEVEL=debug ./gradlew :warpforge-io:uccPerfTest ...
```

---

## Change Log

| Date | Change | Impact |
|------|--------|--------|
| 2026-01-22 | Adaptive polling strategy | 15-30% for 1MB+ |
| 2026-01-22 | Eliminate broadcast buffer copy | 10-15% for broadcast |
| 2026-01-22 | Add performance test infrastructure | - |
