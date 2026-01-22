# ConnectX-5 Performance Baseline - Mark 1 Lab

**Test Date:** 2026-01-22
**Test Engineer:** Automated via validate-rdma-throughput.sh

## Hardware Configuration

### Network Interface Cards

| Node | Device | Model | Firmware | Port State | Link Rate |
|------|--------|-------|----------|------------|-----------|
| mark1nvidia | rocep3s0 | MT27800 (ConnectX-5) | 16.29.2002 | Active/LinkUp | 100 Gbps |
| mark1amd | rocep5s0 | MT27800 (ConnectX-5) | 16.29.2002 | Active/LinkUp | 100 Gbps |

**Card Details:**
- Model: Mellanox ConnectX-5 EN 100GbE single-port QSFP28
- Part Number: MCX515A-CCAT
- PCIe Capability: 3.0 x16 (8 GT/s)
- Link Layer: Ethernet (RoCE)

### PCIe Configuration

| Node | Slot | Capability | Actual Status | Bandwidth |
|------|------|------------|---------------|-----------|
| mark1nvidia | PCIEX16_2 | x16 @ 8GT/s | **x8 @ 8GT/s** | 63 Gbps theoretical |
| mark1amd | PCIEX16_2 | x16 @ 8GT/s | **x8 @ 8GT/s** | 63 Gbps theoretical |

**Commands to verify PCIe status:**
```bash
# Check PCIe link width and speed
sudo lspci -vvv | grep -A 30 Mellanox | grep -E 'LnkSta:|LnkCap:'

# Output on both nodes:
# LnkCap: Port #0, Speed 8GT/s, Width x16, ASPM not supported
# LnkSta: Speed 8GT/s (ok), Width x8 (downgraded)
```

### PCIe Bottleneck Analysis

The ConnectX-5 cards are installed in PCIe x8 slots due to motherboard lane sharing:

**Motherboard:** ASUS ROG Crosshair X670E Hero
- PCIEX16_1: True x16 (occupied by GPU)
- PCIEX16_2: x8 electrical (occupied by Mellanox)

The AMD Ryzen 7000 CPU provides only 16 PCIe 5.0 lanes for expansion slots. When both slots are populated, they share lanes in x8/x8 configuration. This is a hardware limitation, not misconfiguration.

**PCIe Bandwidth Calculation:**
```
PCIe 3.0 x8 @ 8 GT/s:
  8 lanes × 8 GT/s × (128/130 encoding) = 62.5 Gbps theoretical
  Actual achieved: 55.9 Gbps = 89% efficiency
```

### Network Topology

```
┌─────────────────────┐         Direct QSFP28 Cable         ┌─────────────────────┐
│   mark1nvidia       │◄───────────────────────────────────►│     mark1amd        │
│   10.0.0.1/24       │          100 Gbps Link              │   10.0.0.2/24       │
│   enp3s0np0         │                                     │   enp5s0np0         │
│   rocep3s0          │                                     │   rocep5s0          │
└─────────────────────┘                                     └─────────────────────┘
```

## Performance Results

### Bandwidth Tests

All tests run with 1MB message size, 5000 iterations.

| Test | Command | Peak (Gbps) | Average (Gbps) | % of Line Rate | % of PCIe Max |
|------|---------|-------------|----------------|----------------|---------------|
| RDMA Write | `ib_write_bw` | 55.87 | 55.86 | 56% | 89% |
| RDMA Read | `ib_read_bw` | 55.97 | 55.97 | 56% | 90% |
| Send | `ib_send_bw` | 55.88 | 55.88 | 56% | 89% |

**Conclusion:** All bandwidth tests are **PCIe-limited**, not network-limited. The NIC is achieving ~89% of theoretical PCIe 3.0 x8 bandwidth.

### Latency Tests

All tests run with 8-byte messages, 10000 iterations.

| Test | Command | Min (μs) | Typical (μs) | Avg (μs) | P99 (μs) | P99.9 (μs) |
|------|---------|----------|--------------|----------|----------|------------|
| RDMA Write | `ib_write_lat` | 0.93 | 0.99 | 1.00 | 1.15 | 1.40 |
| Send | `ib_send_lat` | 0.89 | 0.92 | 0.92 | 0.95 | 1.40 |

**Conclusion:** Sub-microsecond typical latency. Excellent for distributed ML gradient synchronization.

## Commands Executed

### Prerequisites Check
```bash
# Verify RDMA devices
ibv_devices

# Output:
#     device                 node GUID
#     ------              ----------------
#     rocep3s0            0c42a10300a82c2a   (nvidia)
#     rocep5s0            0c42a10300a82c36   (amd)

# Verify device status
ibstat

# Output shows:
#   CA type: MT4119
#   Firmware version: 16.29.2002
#   Port 1: State: Active, Physical state: LinkUp, Rate: 100
```

### Bandwidth Tests

**RDMA Write Bandwidth (one-sided):**
```bash
# Server (mark1nvidia):
ib_write_bw --size=1048576 --iters=5000 --report_gbits -d rocep3s0

# Client (mark1amd):
ib_write_bw --size=1048576 --iters=5000 --report_gbits -d rocep5s0 10.0.0.1

# Result:
# #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
# 1048576    5000             55.87              55.86               0.006660
```

**RDMA Read Bandwidth (one-sided):**
```bash
# Server (mark1nvidia):
ib_read_bw --size=1048576 --iters=5000 --report_gbits -d rocep3s0

# Client (mark1amd):
ib_read_bw --size=1048576 --iters=5000 --report_gbits -d rocep5s0 10.0.0.1

# Result:
# #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
# 1048576    5000             55.97              55.97               0.006672
```

**Send Bandwidth (two-sided):**
```bash
# Server (mark1nvidia):
ib_send_bw --size=1048576 --iters=5000 --report_gbits -d rocep3s0

# Client (mark1amd):
ib_send_bw --size=1048576 --iters=5000 --report_gbits -d rocep5s0 10.0.0.1

# Result:
# #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
# 1048576    5000             55.88              55.88               0.006661
```

### Latency Tests

**RDMA Write Latency:**
```bash
# Server (mark1nvidia):
ib_write_lat --size=8 --iters=10000 -d rocep3s0

# Client (mark1amd):
ib_write_lat --size=8 --iters=10000 -d rocep5s0 10.0.0.1

# Result:
# #bytes #iterations  t_min[usec]  t_max[usec]  t_typical[usec]  t_avg[usec]  99%[usec]  99.9%[usec]
# 8       10000        0.93         3.52         0.99             1.00         1.15       1.40
```

**Send Latency:**
```bash
# Server (mark1nvidia):
ib_send_lat --size=8 --iters=10000 -d rocep3s0

# Client (mark1amd):
ib_send_lat --size=8 --iters=10000 -d rocep5s0 10.0.0.1

# Result:
# #bytes #iterations  t_min[usec]  t_max[usec]  t_typical[usec]  t_avg[usec]  99%[usec]  99.9%[usec]
# 8       10000        0.89         5.14         0.92             0.92         0.95       1.40
```

## Summary

| Metric | Value | Limiting Factor |
|--------|-------|-----------------|
| Max Bandwidth | 55.97 Gbps | PCIe 3.0 x8 (not network) |
| Typical Latency | ~1 μs | Excellent |
| P99 Latency | ~1.15 μs | Excellent |
| Line Rate Utilization | 56% | PCIe bottleneck |
| PCIe Utilization | 89% | Near optimal for x8 |

## Recommendations

### To Achieve 100 Gbps

**Option 1: Upgrade NIC (Recommended)**
- Replace ConnectX-5 (PCIe 3.0) with ConnectX-6 Dx or ConnectX-7 (PCIe 4.0/5.0)
- PCIe 4.0 x8 = 126 Gbps (sufficient for 100 GbE)
- PCIe 5.0 x8 = 252 Gbps (sufficient for 200 GbE)
- Estimated cost: $150-400 per card on eBay

**Option 2: Swap GPU and NIC slots**
- Move Mellanox to PCIEX16_1 (true x16)
- GPU moves to PCIEX16_2 (x8)
- Network: ~95 Gbps, GPU: reduced bandwidth
- Cost: $0, but impacts GPU performance

**Option 3: Platform upgrade**
- Threadripper (48 lanes) or EPYC (128 lanes)
- Both devices get full x16
- Cost: $2000+ for motherboard + CPU

### Upgrade Path Part Numbers

| Card | PCIe | Speed | eBay Search |
|------|------|-------|-------------|
| ConnectX-6 Dx | 4.0 x16 | 100 GbE | `MCX623106AN-CDAT` |
| ConnectX-7 | 5.0 x16 | 200 GbE | `MCX75310AAS-NEAT` |

## Test Environment

```
Platform: Linux 6.8.0-90-generic (Ubuntu 22.04)
perftest version: 39.0-1
Date: 2026-01-22
```

## Warnings Observed

```
Conflicting CPU frequency values detected: 5529.980000 != 600.000000. CPU Frequency is not max.
```

This warning indicates CPU frequency scaling is active. For production benchmarks, consider:
```bash
# Set CPU governor to performance
sudo cpupower frequency-set -g performance
```
