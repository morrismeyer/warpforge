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

### 3. Arena Pooling (2026-01-23)

**Implementation:** `OperationArenaPool.java`

Created a pool of 4 pre-allocated confined arenas that are reused across operations:
- Eliminates per-operation arena allocation overhead
- Falls back to temporary arena if pool exhausted
- Enabled by default (`-Dwarpforge.ucc.arenaPool=true`)

**Expected Impact:** 3-8% throughput improvement

### 4. Dedicated Progress Thread (2026-01-23)

**Implementation:** `UccProgressThread.java`

Single dedicated thread for all UCC progress driving:
- Main thread submits operations and receives CompletableFuture
- Progress thread calls `ucc_context_progress()` and polls completions
- Batched completion polling reduces FFM overhead
- Experimental: enable via `-Dwarpforge.ucc.progressThread=true`

**Expected Impact:** 20-30% when overlapping computation with communication

### 5. UCC Algorithm Selection (2026-01-23)

**Implementation:** `build.gradle` UCC environment configuration

Optimized settings for large message (1MB+) throughput:
```bash
UCC_TL_UCP_ALLREDUCE_ALG=ring      # Ring algorithm for 2-node large messages
UCC_TL_UCP_ALLGATHER_ALG=ring
UCC_TL_UCP_REDUCE_SCATTER_ALG=ring
UCX_RNDV_SCHEME=get_zcopy          # Zero-copy RDMA for large messages
UCX_RNDV_THRESH=8192               # Lower threshold for rendezvous
```

**Expected Impact:** 5-15% for specific operations

### 6. In-Place Operations (Already Implemented)

The following in-place variants are available and should be preferred when possible:
- `allReduceInPlace(tensor, op)` - modifies tensor in place
- `broadcastInPlace(tensor, root)` - receives directly into tensor
- `allReduceRaw(buffer, count, dtype, op)` - raw buffer operation

**Expected Impact:** 5-10% by avoiding output tensor allocation

### 7. Request Segment Cache (2026-01-23)

**Implementation:** `RequestSegmentCache.java`

Caches reinterpreted UCC request segments to avoid FFM overhead:
- Polling loops typically check status 100-10000 times
- Each reinterpret call has ~50-100ns FFM overhead
- Cache eliminates redundant reinterpretation
- Automatically invalidated when request is finalized

**Expected Impact:** 2-5% latency reduction for polling-heavy operations

### 8. Improved Arena Pooling (2026-01-23)

**Implementation:** Enhanced `OperationArenaPool.PooledArena`

Pre-allocates common structures within pooled arenas:
- Pre-allocated `ucc_coll_args` segment (reused across operations)
- Pre-allocated pointer segment for request handles
- Reset mechanism for clean reuse
- Reduces per-operation allocation from 2 to 0 allocations

**Expected Impact:** Additional 1-3% over basic arena pooling

### 9. Persistent Collectives (2026-01-23)

**Implementation:** `PersistentCollective.java`

For repeated operations with same parameters (benchmarks, training loops):
- Initialize collective once, post multiple times
- Eliminates per-operation `ucc_collective_init` overhead (~100-500us)
- Cached request segment for optimized status polling
- Built-in execution statistics

**Usage:**
```java
PersistentCollective allreduce = PersistentCollective.allReduceInPlace(
    team, context, buffer, count, UccConstants.DT_FLOAT32, UccConstants.OP_SUM);

for (int i = 0; i < 1000; i++) {
    allreduce.execute();  // Only post + wait, no init
}

allreduce.close();
```

**Expected Impact:** 30-50% latency reduction for repeated operations

### 10. Batch Collective Operations (2026-01-23)

**Implementation:** `BatchCollective.java`

For executing multiple collectives in sequence:
- Queue multiple operations, execute all in batch
- Single progress loop drives all pending operations
- Reduces context progress call overhead
- Early completion of faster operations while waiting for slower ones

**Usage:**
```java
BatchCollective batch = new BatchCollective(team, context);

// Queue multiple operations
batch.queueAllReduceInPlace(buf1, count, dtype, op);
batch.queueAllReduceInPlace(buf2, count, dtype, op);
batch.queueBroadcast(buf3, count, dtype, root);

// Execute all and wait
batch.executeAll();

batch.close();
```

**Expected Impact:** 10-20% improvement for multi-operation sequences

---

## All Optimizations Complete

Total expected improvement for 1MB+ operations: **50-75%**

| Optimization | Status | Impact |
|--------------|--------|--------|
| Adaptive polling | Done | 15-30% |
| Buffer copy elimination | Done | 10-15% |
| Arena pooling | Done | 3-8% |
| Dedicated progress thread | Done | 20-30% |
| UCC algorithm selection | Done | 5-15% |
| In-place operations | Available | 5-10% |
| Request segment cache | Done | 2-5% |
| Improved arena pooling | Done | 1-3% |
| Persistent collectives | Done | 30-50% (repeated ops) |
| Batch collective operations | Done | 10-20% (multi-op sequences) |

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
| 2026-01-23 | Batch collective operations (BatchCollective) | 10-20% for multi-op |
| 2026-01-23 | Persistent collectives (PersistentCollective) | 30-50% for repeated ops |
| 2026-01-23 | Request segment cache (RequestSegmentCache) | 2-5% latency |
| 2026-01-23 | Enhanced arena pooling (pre-allocated structs) | 1-3% additional |
| 2026-01-23 | Arena pooling (OperationArenaPool) | 3-8% |
| 2026-01-23 | Dedicated progress thread (UccProgressThread) | 20-30% |
| 2026-01-23 | UCC algorithm selection (ring for large messages) | 5-15% |
| 2026-01-23 | UCX/UCC C baseline harness (ucx_max_bandwidth, ucc_collective_benchmark) | - |
| 2026-01-23 | Head-to-head benchmark script (benchmark-head-to-head.sh) | - |
| 2026-01-22 | Adaptive polling strategy | 15-30% for 1MB+ |
| 2026-01-22 | Eliminate broadcast buffer copy | 10-15% for broadcast |
| 2026-01-22 | Add performance test infrastructure | - |
