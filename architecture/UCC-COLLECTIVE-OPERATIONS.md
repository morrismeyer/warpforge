# UCC Collective Operations Implementation

This document describes the implementation of StableHLO/PyTorch collective primitives over the Mellanox ConnectX-5 networking cross-connect using UCC (Unified Collective Communications) via FFM bindings.

## Overview

WarpForge collective operations bridge StableHLO distributed operations to hardware-accelerated RDMA:

```
StableHLO Collective Ops          WarpForge                    Hardware
-------------------               ---------                    --------
stablehlo.all_reduce    -->    CollectiveApi.allReduce()   -->  UCC/UCX
stablehlo.all_gather    -->    CollectiveApi.allGather()   -->  ConnectX-5
stablehlo.all_to_all    -->    CollectiveApi.allToAll()    -->  RDMA
stablehlo.reduce_scatter -->   CollectiveApi.reduceScatter()
stablehlo.collective_broadcast --> CollectiveApi.broadcast()
```

## Hardware Baseline

Performance targets derived from `holmes-lab/mark1/mellanox-perf/CONNECTX5.md`:

|-----------------|---------------|-----------------|
| Metric          | Linux Baseline | Limiting Factor |
|-----------------|---------------|-----------------|
| Max Bandwidth   | 55.97 Gbps    | PCIe 3.0 x8     |
| RDMA Write      | 55.87 Gbps    | PCIe bottleneck |
| RDMA Read       | 55.97 Gbps    | PCIe bottleneck |
| Send            | 55.88 Gbps    | PCIe bottleneck |
| Write Latency   | ~1.0 us       | Excellent       |
| Send Latency    | ~0.92 us      | Excellent       |
|-----------------|---------------|-----------------|

**Note**: The 100 Gbps link is PCIe-limited. Full line rate requires ConnectX-6 Dx (PCIe 4.0) or ConnectX-7 (PCIe 5.0).

## Collective Operations

### StableHLO to UCC Mapping

|----------------------|---------------------|-----------------------------------|
| StableHLO Op         | CollectiveApi       | UCC Type                          |
|----------------------|---------------------|-----------------------------------|
| `all_reduce`         | `allReduce()`       | `UCC_COLL_TYPE_ALLREDUCE` (4)     |
| `all_gather`         | `allGather()`       | `UCC_COLL_TYPE_ALLGATHER` (1)     |
| `all_to_all`         | `allToAll()`        | `UCC_COLL_TYPE_ALLTOALL` (8)      |
| `reduce_scatter`     | `reduceScatter()`   | `UCC_COLL_TYPE_REDUCE_SCATTER`    |
| `collective_broadcast` | `broadcast()`     | `UCC_COLL_TYPE_BCAST` (64)        |
| -                    | `reduce()`          | `UCC_COLL_TYPE_REDUCE` (2048)     |
| -                    | `scatter()`         | `UCC_COLL_TYPE_SCATTER` (16384)   |
| -                    | `gather()`          | `UCC_COLL_TYPE_GATHER` (512)      |
| -                    | `barrier()`         | `UCC_COLL_TYPE_BARRIER` (32)      |
|----------------------|---------------------|-----------------------------------|

### Reduction Operations

|----------|----------|------------------------|
| AllReduceOp | UCC Code | Description         |
|----------|----------|------------------------|
| SUM      | 0        | Element-wise sum       |
| PROD     | 1        | Element-wise product   |
| MIN      | 2        | Element-wise minimum   |
| MAX      | 3        | Element-wise maximum   |
| AVG      | 12       | Element-wise average   |
| BAND     | 7        | Bitwise AND            |
| BOR      | 8        | Bitwise OR             |
| BXOR     | 9        | Bitwise XOR            |
| LAND     | 4        | Logical AND            |
| LOR      | 5        | Logical OR             |
| MINLOC   | 11       | Min with location      |
| MAXLOC   | 10       | Max with location      |
|----------|----------|------------------------|

## Architecture

```
+-----------------------------------------------------------------------------+
|                         Collective Operations Stack                          |
+-----------------------------------------------------------------------------+
|                                                                             |
|  +-----------------------------------------------------------------------+  |
|  |  CollectiveApi Interface                                              |  |
|  |  - allReduce(), allGather(), broadcast(), reduceScatter()             |  |
|  |  - barrier(), reduce(), scatter(), gather()                           |  |
|  |  - send(), recv() (point-to-point)                                    |  |
|  +-----------------------------------------------------------------------+  |
|                    |                              |                         |
|                    v                              v                         |
|  +-------------------------------+  +-------------------------------+       |
|  |  UccCollectiveImpl            |  |  CollectiveMock               |       |
|  |  (Linux + UCC libraries)      |  |  (Any platform, testing)      |       |
|  +-------------------------------+  +-------------------------------+       |
|                    |                                                        |
|                    v                                                        |
|  +-----------------------------------------------------------------------+  |
|  |  FFM Bindings (jextract-generated)                                    |  |
|  |  io.surfworks.warpforge.io.ffi.ucc.*                                  |  |
|  +-----------------------------------------------------------------------+  |
|                    |                                                        |
|                    v                                                        |
|  +-----------------------------------------------------------------------+  |
|  |  UCC Library (libucc.so)                                              |  |
|  |  - Collective algorithms (ring, recursive-halving, etc.)              |  |
|  +-----------------------------------------------------------------------+  |
|                    |                                                        |
|                    v                                                        |
|  +-----------------------------------------------------------------------+  |
|  |  UCX Transport (libucp.so)                                            |  |
|  |  - RDMA operations via ibverbs                                        |  |
|  +-----------------------------------------------------------------------+  |
|                    |                                                        |
|                    v                                                        |
|  +-----------------------------------------------------------------------+  |
|  |  Mellanox ConnectX-5 (mlx5_0)                                         |  |
|  |  - 100 GbE QSFP28 link (PCIe-limited to ~56 Gbps)                     |  |
|  +-----------------------------------------------------------------------+  |
|                                                                             |
+-----------------------------------------------------------------------------+
```

## Implementation Components

### Core Files

|-----------------------------------------------|-------------------------------------|
| File                                          | Purpose                             |
|-----------------------------------------------|-------------------------------------|
| `impl/UccCollectiveImpl.java`                 | UCC-backed collective operations    |
| `impl/UccConstants.java`                      | UCC constants and datatype mapping  |
| `impl/UccHelper.java`                         | FFM utility methods                 |
| `impl/OobCoordinator.java`                    | TCP-based out-of-band coordination  |
| `mock/CollectiveMock.java`                    | Mock for testing without hardware   |
| `Collective.java`                             | Factory with auto-detection         |
|-----------------------------------------------|-------------------------------------|

### FFM Bindings

763 jextract-generated Java files in `warpforge-io/src/generated/java/io/surfworks/warpforge/io/ffi/ucc/`:

|--------------------------|----------------------------------------------|
| Type                     | Purpose                                      |
|--------------------------|----------------------------------------------|
| `Ucc.java`               | Main FFM class with all function signatures  |
| `ucc_lib_params.java`    | Library initialization parameters            |
| `ucc_context_params.java`| Context creation parameters                  |
| `ucc_team_params.java`   | Team creation with OOB callbacks             |
| `ucc_coll_args.java`     | Collective operation arguments               |
| `ucc_coll_buffer_info.java` | Buffer descriptions (32 bytes)            |
| `ucc_oob_coll.java`      | Out-of-band callback structure               |
|--------------------------|----------------------------------------------|

## TODOs

### Virtual Thread Threading Model

**Issue**: UCX requires all operations on a worker to be called from the same thread that created the worker. Virtual threads can migrate between carrier threads, which violates this constraint even when pinned during individual FFM calls.

**Current Solution**: All UCC operations run synchronously on the main thread that created the context. This works but:
- Blocks the calling thread during collective operations
- Doesn't leverage virtual threads for async I/O overlapping

**Better Approach**: Create a dedicated platform thread for all UCC operations:
1. Spawn a single platform thread at initialization
2. Submit UCC operations to a queue processed by that thread
3. Return `CompletableFuture` that completes when the operation finishes
4. All UCX calls happen on the same thread, satisfying thread affinity

**Alternative**: Investigate if UCX can be configured to relax thread affinity checks (e.g., `UCX_MT_MODE=multi` or similar). This may require recompiling UCX with different threading options.

## UCC Initialization Sequence

```java
// 1. Initialize UCC library
MemorySegment libParams = ucc_lib_params.allocate(arena);
ucc_lib_params.mask(libParams, 0L);

MemorySegment libHandlePtr = arena.allocate(ValueLayout.ADDRESS);
int status = Ucc.ucc_init_version(1, 0, libParams, MemorySegment.NULL, libHandlePtr);
checkStatus(status, "ucc_init_version");
uccLib = libHandlePtr.get(ValueLayout.ADDRESS, 0);

// 2. Create OOB coordinator (TCP connections between ranks)
oobCoordinator = new OobCoordinator(config, arena);

// 3. Create UCC context with OOB callbacks
MemorySegment ctxParams = ucc_context_params.allocate(arena);
ucc_context_params.mask(ctxParams, Ucc.UCC_CONTEXT_PARAM_FIELD_OOB());
// ... configure OOB from coordinator ...

MemorySegment ctxHandlePtr = arena.allocate(ValueLayout.ADDRESS);
status = Ucc.ucc_context_create(uccLib, ctxParams, MemorySegment.NULL, ctxHandlePtr);
checkStatus(status, "ucc_context_create");
uccContext = ctxHandlePtr.get(ValueLayout.ADDRESS, 0);

// 4. Create UCC team (async with OOB allgather)
MemorySegment teamParams = ucc_team_params.allocate(arena);
long teamMask = Ucc.UCC_TEAM_PARAM_FIELD_EP()
              | Ucc.UCC_TEAM_PARAM_FIELD_EP_RANGE()
              | Ucc.UCC_TEAM_PARAM_FIELD_TEAM_SIZE()
              | Ucc.UCC_TEAM_PARAM_FIELD_OOB();
ucc_team_params.mask(teamParams, teamMask);
ucc_team_params.ep(teamParams, config.rank());
ucc_team_params.team_size(teamParams, config.worldSize());
// ... configure OOB ...

MemorySegment teamHandlePtr = arena.allocate(ValueLayout.ADDRESS);
status = Ucc.ucc_team_create_post(ctxHandlePtr, 1, teamParams, teamHandlePtr);
checkStatus(status, "ucc_team_create_post");

// Poll until team creation completes
uccTeam = teamHandlePtr.get(ValueLayout.ADDRESS, 0);
while (Ucc.ucc_team_create_test(uccTeam) == Ucc.UCC_INPROGRESS()) {
    Thread.onSpinWait();
}
```

## Collective Operation Pattern

All collective operations follow this pattern:

```java
public CompletableFuture<Tensor> allReduce(Tensor input, AllReduceOp op) {
    checkInitialized();
    checkNotClosed();
    return VirtualThreads.supplyAsync(() -> {
        // 1. Allocate output tensor
        Tensor result = Tensor.zeros(input.dtype(), input.shape());

        // 2. Set up collective arguments
        MemorySegment args = ucc_coll_args.allocate(arena);
        ucc_coll_args.mask(args, 0L);
        ucc_coll_args.coll_type(args, Ucc.UCC_COLL_TYPE_ALLREDUCE());
        ucc_coll_args.op(args, op.uccCode());

        // 3. Configure source buffer
        MemorySegment srcInfo = ucc_coll_args.src(args);
        setupBufferInfo(srcInfo, input);

        // 4. Configure destination buffer
        MemorySegment dstInfo = ucc_coll_args.dst(args);
        setupBufferInfo(dstInfo, result);

        // 5. Initialize and post collective
        MemorySegment requestPtr = arena.allocate(ValueLayout.ADDRESS);
        int status = Ucc.ucc_collective_init_and_post(args, requestPtr, uccTeam);
        UccHelper.checkStatus(status, "allreduce init_and_post");

        // 6. Wait for completion
        MemorySegment request = requestPtr.get(ValueLayout.ADDRESS, 0);
        UccHelper.waitForCompletion(request);

        // 7. Update statistics
        allReduceCount.incrementAndGet();
        totalOps.incrementAndGet();
        totalBytes.addAndGet(input.spec().byteSize());

        return result;
    });
}

private void setupBufferInfo(MemorySegment info, Tensor tensor) {
    ucc_coll_buffer_info.buffer(info, tensor.data());
    ucc_coll_buffer_info.count(info, tensor.spec().numElements());
    ucc_coll_buffer_info.datatype(info, UccConstants.scalarTypeToUccDatatype(tensor.dtype()));
    ucc_coll_buffer_info.mem_type(info, Ucc.UCC_MEMORY_TYPE_HOST());
}
```

## Out-of-Band Coordination

UCC team formation requires out-of-band (OOB) allgather for exchanging endpoint information:

```
+------------------+                           +------------------+
|  Rank 0 (Server) |                           |  Rank 1 (Client) |
+------------------+                           +------------------+
        |                                              |
        |  1. Listen on masterPort                     |
        |<---------------------------------------------|
        |     2. Connect to masterAddress:masterPort   |
        |                                              |
        |  3. OOB allgather (each rank sends data)     |
        |--------------------------------------------->|
        |<---------------------------------------------|
        |                                              |
        |  4. Rank 0 aggregates and broadcasts         |
        |--------------------------------------------->|
        |                                              |
        |  5. UCC team ready                           |
        +----------------------------------------------+
```

The `OobCoordinator` class implements this protocol using TCP sockets and provides upcall stubs for UCC's OOB callbacks.

## Performance Targets

Following `architecture/RDMA-PERF-TESTING.md`:

### Bandwidth

|-----------|-----------------|--------------------|
| Operation | Linux Baseline  | Java Target (95%)  |
|-----------|-----------------|--------------------|
| AllReduce | 55.88 Gbps      | 53.09 Gbps         |
| AllGather | 55.87 Gbps      | 53.08 Gbps         |
| Broadcast | 55.88 Gbps      | 53.09 Gbps         |
|-----------|-----------------|--------------------|

### Latency

|-----------|-----------------|--------------------|
| Operation | Linux Baseline  | Java Target (+20%) |
|-----------|-----------------|--------------------|
| Barrier   | ~0.92 us        | 1.10 us            |
| AllReduce | ~1.00 us (small)| 1.20 us            |
|-----------|-----------------|--------------------|

## Testing

### Unit Tests (Mock-based, no hardware)

```bash
./gradlew :warpforge-io:test --tests "*CollectiveApiTest*"
```

### UCC Integration Tests (Linux with UCC libs)

```bash
./gradlew :warpforge-io:test --tests "*UccIntegrationTest*" -Ptags=ucc
```

### Two-Node Performance Tests (Mark 1 lab)

```bash
# On mark1nvidia (10.0.0.1):
./gradlew :warpforge-io:run --args='collective-server 29500'

# On mark1amd:
./gradlew :warpforge-io:collectivePerfTest \
    -Dcollective.master=10.0.0.1 \
    -Dcollective.port=29500
```

## Prerequisites

1. **Generate FFM stubs** (if not already present):
   ```bash
   ./gradlew :openucx-runtime:generateJextractStubs
   ```

2. **Install UCC libraries** (Linux only):
   ```bash
   ./gradlew :openucx-runtime:ensureOpenUCXReady
   ```

3. **Verify RDMA connectivity**:
   ```bash
   # Check InfiniBand devices
   ibstat

   # Test RDMA write bandwidth
   ib_write_bw -d rocep3s0
   ```

## Related Documents

- [RDMA-PERF-TESTING.md](RDMA-PERF-TESTING.md) - RDMA performance testing framework
- [BACKEND-KERNEL-INSTRUMENTATION.md](BACKEND-KERNEL-INSTRUMENTATION.md) - Kernel observability
- [CONNECTX5.md](../holmes-lab/mark1/mellanox-perf/CONNECTX5.md) - Hardware baseline measurements
