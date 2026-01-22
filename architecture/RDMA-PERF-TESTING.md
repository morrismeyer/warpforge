# RDMA Performance Testing Architecture

This document describes WarpForge's RDMA performance testing framework for validating the Mellanox ConnectX-5 cross-connect between Mark 1 lab GPU nodes.

## Hardware Baseline

The performance targets are derived from actual hardware measurements documented in `holmes-lab/mark1/mellanox-perf/CONNECTX5.md`:

| Metric | Linux Baseline | Limiting Factor |
|--------|---------------|-----------------|
| Max Bandwidth | 55.97 Gbps | PCIe 3.0 x8 (not network) |
| RDMA Write | 55.87 Gbps | PCIe bottleneck |
| RDMA Read | 55.97 Gbps | PCIe bottleneck |
| Send | 55.88 Gbps | PCIe bottleneck |
| Write Latency | ~1.0 μs | Excellent |
| Send Latency | ~0.92 μs | Excellent |
| P99 Latency | ~1.15 μs | Excellent |

### PCIe Bottleneck Analysis

```
┌──────────────────────────────────────────────────────────────────────┐
│                       PCIe Bandwidth Calculation                      │
├──────────────────────────────────────────────────────────────────────┤
│  ConnectX-5: PCIe 3.0 capable                                        │
│  Slot: x8 electrical (motherboard limitation)                        │
│                                                                      │
│  PCIe 3.0 x8 @ 8 GT/s:                                              │
│    8 lanes × 8 GT/s × (128/130 encoding) = 62.5 Gbps theoretical    │
│                                                                      │
│  Achieved: 55.9 Gbps = 89% of theoretical                           │
│  This is excellent utilization for real-world workloads.            │
└──────────────────────────────────────────────────────────────────────┘
```

The 100 Gbps link rate is not achievable with PCIe 3.0 x8. To reach full line rate, upgrade to:
- ConnectX-6 Dx (PCIe 4.0) - 126 Gbps theoretical on x8
- ConnectX-7 (PCIe 5.0) - 252 Gbps theoretical on x8

## Test Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RDMA Performance Test Suite                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  IbPerfBaselineTest (@Tag("rdma-perf"))                             │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  Uses native ib_write_bw, ib_read_bw, etc.                  │   │   │
│  │  │  Establishes maximum achievable throughput                   │   │   │
│  │  │  Runs client-side only (server started manually)             │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  RdmaPerformanceTest (@Tag("rdma-perf"))                            │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  Java RDMA API bandwidth tests (send, write, read)          │   │   │
│  │  │  Java RDMA API latency tests (ping-pong, one-sided)         │   │   │
│  │  │  Compares against Linux baseline                             │   │   │
│  │  │  Target: ≥95% of baseline bandwidth, ≤120% of baseline lat  │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  RdmaZeroCopyTest (@Tag("unit"))                                    │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  Verifies MemorySegment identity (no copies)                │   │   │
│  │  │  Verifies address stability after registration              │   │   │
│  │  │  Verifies bidirectional visibility of modifications         │   │   │
│  │  │  Runs locally with mock RDMA (no hardware required)         │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  RdmaDataIntegrityTest (@Tag("rdma-perf"))                          │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  Pattern tests (0x00, 0xFF, 0xAA, 0x55, etc.)               │   │   │
│  │  │  Sequential tests (monotonic, position encoding)             │   │   │
│  │  │  Random data with CRC32/SHA-256 verification                │   │   │
│  │  │  Stress tests (many small, repeated large)                   │   │   │
│  │  │  Boundary tests (first/last bytes)                          │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Running Tests

### Unit Tests (No Hardware Required)

```bash
# Zero-copy verification (uses mock RDMA)
./gradlew :warpforge-io:test --tests "*RdmaZeroCopyTest*"
```

### Performance Tests (Requires Two Nodes)

```bash
# On server node (e.g., mark1nvidia @ 10.0.0.1):
./gradlew :warpforge-io:run --args='server 18515'

# On client node (e.g., mark1amd):
./gradlew :warpforge-io:rdmaPerfTest \
    -Drdma.server.host=10.0.0.1 \
    -Drdma.server.port=18515
```

### Linux Baseline Tests (Requires Two Nodes)

```bash
# On server node:
ib_write_bw --size=1048576 --iters=5000 --report_gbits -d rocep3s0

# On client node:
./gradlew :warpforge-io:rdmaPerfTest \
    --tests "*IbPerfBaselineTest*" \
    -Drdma.server.host=10.0.0.1
```

### Data Integrity Tests (Requires Two Nodes)

```bash
# On server node:
./gradlew :warpforge-io:run --args='server 18516'

# On client node:
./gradlew :warpforge-io:rdmaPerfTest \
    --tests "*RdmaDataIntegrityTest*" \
    -Drdma.server.host=10.0.0.1 \
    -Drdma.server.port=18516
```

## Performance Targets

### Bandwidth

| Operation | Linux Baseline | Java Target (95%) | Notes |
|-----------|---------------|-------------------|-------|
| Send | 55.88 Gbps | 53.09 Gbps | Two-sided |
| RDMA Write | 55.87 Gbps | 53.08 Gbps | One-sided |
| RDMA Read | 55.97 Gbps | 53.17 Gbps | One-sided |

### Latency

| Operation | Linux Baseline | Java Target (+20%) | Notes |
|-----------|---------------|-------------------|-------|
| Send | 0.92 μs | 1.10 μs | Round-trip/2 |
| RDMA Write | 1.00 μs | 1.20 μs | One-sided |

## Zero-Copy Verification

The zero-copy tests verify that:

1. **Segment Identity**: `RdmaBuffer.segment()` returns the exact same `MemorySegment` object passed to `registerMemory()`

2. **Address Stability**: The memory address does not change after registration

3. **Bidirectional Visibility**: Writes through the original segment are visible through `RdmaBuffer.segment()` and vice versa

4. **No Intermediate Copies**: Registration of large buffers (256MB) completes in <100ms, proving no copy occurs

## Data Integrity Verification

Tests use multiple verification methods:

1. **Pattern Tests**: Known bit patterns detect single-bit errors
2. **Sequential Tests**: Monotonic sequences detect reordering
3. **CRC32**: Fast checksum for large transfers
4. **SHA-256**: Cryptographic verification for critical data
5. **Boundary Tests**: Verify first/last bytes are not corrupted

## Future Enhancements

### Phase 1 (Current)
- Basic bandwidth and latency tests
- Zero-copy verification
- Data integrity tests

### Phase 2 (Planned)
- GPU Direct RDMA tests
- Multi-threaded concurrent transfer tests
- Connection recovery tests

### Phase 3 (Future)
- Continuous performance regression tracking
- Historical trend analysis
- Automated alerting on performance degradation

## Related Documents

- [CONNECTX5.md](../holmes-lab/mark1/mellanox-perf/CONNECTX5.md) - Hardware baseline measurements
- [warpforge-io/README.md](../warpforge-io/README.md) - RDMA API documentation
