# UCC Performance Tuning Guide

This guide documents all tunable parameters for WarpForge UCC collective operations.

## Quick Start

For maximum performance out of the box, use these environment variables:

```bash
# UCC algorithm selection (optimal for 2-node, large message)
export UCC_TL_UCP_ALLREDUCE_ALG=ring
export UCC_TL_UCP_ALLGATHER_ALG=ring
export UCC_TL_UCP_REDUCE_SCATTER_ALG=ring

# UCX zero-copy RDMA
export UCX_RNDV_SCHEME=get_zcopy
export UCX_RNDV_THRESH=8192
```

## Java System Properties

### Arena Pooling (Enabled by Default)

```bash
# Enable/disable arena pooling
-Dwarpforge.ucc.arenaPool=true    # Default: true

# Benefit: 3-8% throughput improvement
# Trade-off: Slight memory overhead from pooled arenas
```

### Progress Thread (Experimental)

```bash
# Enable dedicated progress thread
-Dwarpforge.ucc.progressThread=true    # Default: false

# Benefit: 20-30% when overlapping computation with communication
# Trade-off: Additional thread, higher CPU usage when idle
```

### Tensor Buffer Pool

```bash
# Enable tensor buffer pooling
-Dwarpforge.ucc.tensorPool=true    # Default: false

# Benefit: 5-20us saved per operation
# Trade-off: Memory overhead from pooled tensors
```

## UCC Environment Variables

### Algorithm Selection

For 2-node setups with large messages (>1MB):

```bash
# Use ring algorithm for collectives
UCC_TL_UCP_ALLREDUCE_ALG=ring
UCC_TL_UCP_ALLGATHER_ALG=ring
UCC_TL_UCP_REDUCE_SCATTER_ALG=ring

# For small messages, use recursive doubling
UCC_TL_UCP_ALLREDUCE_ALG=knomial
```

### Transport Selection

```bash
# Use specific transport layer
UCC_TL=ucp          # Default, uses UCX
UCC_TL=nccl         # For NVIDIA GPUs
UCC_TL=cuda         # For CUDA-aware operations
```

### Debug and Logging

```bash
# Enable debug output
UCC_LOG_LEVEL=debug
UCC_LOG_LEVEL=info
UCC_LOG_LEVEL=warn    # Default

# Component-specific logging
UCC_TL_UCP_LOG_LEVEL=debug
```

## UCX Environment Variables

### RDMA Optimization

```bash
# Zero-copy for large messages
UCX_RNDV_SCHEME=get_zcopy    # or put_zcopy

# Lower rendezvous threshold for earlier zero-copy
UCX_RNDV_THRESH=8192         # Default: 8KB

# For very large messages
UCX_RNDV_THRESH=4096
```

### Transport Selection

```bash
# InfiniBand/RoCE optimized
UCX_TLS=rc_mlx5,ud_mlx5

# TCP fallback
UCX_TLS=tcp

# All transports (auto-select)
UCX_TLS=all
```

### Memory Management

```bash
# Memory pool configuration
UCX_MPOOL_FIFO=y     # FIFO memory pool (better for multi-threaded)

# Registration cache
UCX_RCACHE=y         # Enable registration cache
UCX_RCACHE_MAX=128   # Max entries in registration cache
```

### Debug and Logging

```bash
# UCX log levels
UCX_LOG_LEVEL=debug
UCX_LOG_LEVEL=info
UCX_LOG_LEVEL=warn    # Default

# Protocol tracing
UCX_PROTO_TRACE=1
```

## Performance Profiles

### Low Latency (Small Messages)

```bash
# Optimize for latency
export UCX_RNDV_THRESH=16384
export UCC_TL_UCP_ALLREDUCE_ALG=knomial
-Dwarpforge.ucc.progressThread=true
```

### High Throughput (Large Messages)

```bash
# Optimize for throughput
export UCX_RNDV_SCHEME=get_zcopy
export UCX_RNDV_THRESH=4096
export UCC_TL_UCP_ALLREDUCE_ALG=ring
-Dwarpforge.ucc.arenaPool=true
-Dwarpforge.ucc.tensorPool=true
```

### Benchmark Mode

```bash
# Maximum performance for benchmarks
export UCX_RNDV_SCHEME=get_zcopy
export UCX_RNDV_THRESH=4096
export UCC_TL_UCP_ALLREDUCE_ALG=ring
export UCC_TL_UCP_ALLGATHER_ALG=ring
-Dwarpforge.ucc.arenaPool=true
-Dwarpforge.ucc.progressThread=false
```

## Hardware-Specific Tuning

### Mellanox ConnectX-5/6 (100GbE)

```bash
# Optimal settings for ConnectX-5/6
export UCX_TLS=rc_mlx5
export UCX_RNDV_SCHEME=get_zcopy
export UCX_RNDV_THRESH=8192
export UCX_DC_MLX5_TIMEOUT=22
```

### Intel OmniPath

```bash
export UCX_TLS=psm2
export UCX_PSM2_MQ_RNDV_THRESH=65536
```

### AWS EFA

```bash
export UCX_TLS=efa
export FI_EFA_TX_MIN_CREDITS=1024
```

## Troubleshooting

### Low Throughput

1. Check UCX transport: `UCX_LOG_LEVEL=info` should show `rc_mlx5` or similar
2. Verify zero-copy: Look for "rndv" in UCX debug output
3. Check algorithm: `UCC_LOG_LEVEL=debug` shows selected algorithm

### High Latency

1. Enable progress thread: `-Dwarpforge.ucc.progressThread=true`
2. Lower rendezvous threshold: `UCX_RNDV_THRESH=4096`
3. Use adaptive algorithms: `UCC_TL_UCP_ALLREDUCE_ALG=knomial`

### Memory Issues

1. Check arena pool: Disable if memory is tight `-Dwarpforge.ucc.arenaPool=false`
2. Disable tensor pool: `-Dwarpforge.ucc.tensorPool=false`
3. Reduce UCX cache: `UCX_RCACHE_MAX=32`

## Monitoring

### JFR Profiling

```bash
./gradlew :warpforge-io:uccPerfTest \
    -Djdk.jfr.startup=true
```

### UCX Stats

```bash
# Enable UCX statistics
UCX_STATS_DEST=stdout
UCX_STATS_TRIGGER=10s
```

## See Also

- [UCC-PERFORMANCE-OPTIMIZATION.md](UCC-PERFORMANCE-OPTIMIZATION.md) - Optimization roadmap
- [UCX Documentation](https://openucx.org/documentation/)
- [UCC GitHub](https://github.com/openucx/ucc)
