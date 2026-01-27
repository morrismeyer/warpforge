# Research Validation Audit Results

This document contains the results of auditing all research validation implementations in `ptest/src/main/java/.../research/`.

## Summary

+--------------------------------+-------------+---------------------+
| Validation                     | GPU Work    | Correctness Issue   |
+--------------------------------+-------------+---------------------+
| OverlappingIoValidation        | REAL        | None                |
| PipelineBubbleValidation       | REAL        | None                |
| OccupancyAdmissionValidation   | REAL        | None (FIXED)        |
| SloInferenceValidation         | REAL        | None                |
| ResearchValidationRunner       | N/A         | None                |
+--------------------------------+-------------+---------------------+

**Overall Verdict**: 4/4 PASS

---

## Detailed Analysis

### OverlappingIoValidation.java (Tally - ASPLOS 2025)

**Verdict**: PASS

**Purpose**: Validates compute/memory overlap patterns on multiple streams.

**GPU Work** (real operations):
- `backend.copyToDeviceAsync(host, lease.streamHandle())` - H2D transfer
- `backend.synchronizeStream(lease.streamHandle())` - Stream sync
- `backend.copyToHostAsync(device, lease.streamHandle())` - D2H transfer
- `backend.allocateDevice(spec)` - GPU allocation

**Scenarios**:
1. `validateComputeMemoryOverlap()` - Tests overlap of transfers on different streams
2. `validateMultiStreamOverlap()` - Tests 4 concurrent streams
3. `validateSequentialVsOverlapped()` - Measures speedup from parallelism

**Correctness**: Metrics are derived from real GPU timing, not simulated.

---

### PipelineBubbleValidation.java (PipeFill - MLSys 2025)

**Verdict**: PASS

**Purpose**: Validates pipeline bubble filling with useful work.

**GPU Work** (real operations in `doStageWork()`, lines 273-285):
```java
Tensor device = backend.copyToDeviceAsync(host, streamHandle);
backend.synchronizeStream(streamHandle);
```

**Scenarios**:
1. `validateIdentifyBubbles()` - Measures idle time between pipeline stages
2. `validateFillWithUsefulWork()` - Schedules filler work during bubbles
3. `validateFillEfficiency()` - Ensures filler work adds <10% overhead

**Correctness**: Bubble detection based on real GPU timing. Filler work is lightweight (CPU math), but that's appropriate for bubble filling.

---

### OccupancyAdmissionValidation.java (Orion - EuroSys 2024)

**Verdict**: PASS (FIXED 2026-01-27)

**Purpose**: Validates occupancy-based admission control per Orion paper.

**GPU Work** (real operations in `doInferenceWork()`):
```java
Tensor device = backend.copyToDeviceAsync(host, streamHandle);
backend.synchronizeStream(streamHandle);
Tensor result = backend.copyToHostAsync(device, streamHandle);
backend.synchronizeStream(streamHandle);
```

**Fix Applied**:
1. Created `NvmlRuntime.java` with FFM bindings to libnvidia-ml:
   - `nvmlDeviceGetUtilizationRates()` - GPU/memory utilization
   - `nvmlDeviceGetMemoryInfo()` - memory usage
   - `nvmlDeviceGetTemperature()` - thermal monitoring
   - `nvmlDeviceGetPowerUsage()` - power monitoring

2. Created `GpuMonitoring` interface in warpforge-core:
   - `getGpuUtilization()` - 0-100% GPU busy time
   - `getMemoryUtilization()` - 0-100% memory bandwidth
   - `getMetrics()` - snapshot of all metrics

3. Updated `NvidiaBackend` to implement `GpuMonitoring`:
   - Initializes NVML on construction
   - Queries real utilization from NVML
   - Cleans up NVML on close

4. Updated `OccupancyAdmissionValidation.java`:
   - Uses real NVML metrics when available
   - Falls back to timing-based proxy when NVML unavailable
   - Documents NVML utilization semantics (time-based, not SM-based)

**Important caveat from architecture docs**: NVML's "utilization" measures
"% of time any kernel was running", not "% of compute capacity used".
A kernel using 10% of SMs still shows 100% utilization while running.
We use this as a proxy for GPU busyness.

**Scenarios**:
1. `validateTrackOccupancy()` - Uses REAL NVML utilization queries
2. `validateAdmissionControl()` - Admission decisions based on REAL utilization
3. `validateThroughputGain()` - Real GPU work with throughput measurement

---

### SloInferenceValidation.java (Alibaba Aegaeon - SOSP 2025)

**Verdict**: PASS

**Purpose**: Validates SLO-bounded inference with latency targets.

**GPU Work** (real operations in `doInference()`, lines 259-273):
```java
Tensor device = backend.copyToDeviceAsync(host, stream);
backend.synchronizeStream(stream);
Tensor result = backend.copyToHostAsync(device, stream);
backend.synchronizeStream(stream);
```

**Scenarios**:
1. `validateP99Latency()` - Measures real P99 latency against 100ms SLO
2. `validateGracefulDegradation()` - Tests throughput with reduced quality (smaller tensors)
3. `validateBatchAdaptation()` - Finds optimal batch size for SLO compliance

**Correctness**: All latency measurements from real GPU operations. SLO enforcement uses actual timing.

---

### ResearchValidationRunner.java

**Verdict**: PASS (runner only)

**Purpose**: CLI runner that executes all validation suites.

**Features**:
- Backend auto-detection (NVIDIA/AMD)
- Verbose mode
- Summary table output
- JFR recording support

---

## Recommendations

### Completed (2026-01-27)

1. ~~**Fix OccupancyAdmissionValidation**~~:
   - ~~Add NVML bindings: `nvmlDeviceGetUtilizationRates()`, `nvmlDeviceGetMemoryInfo()`~~
   - ~~Create `NvmlMetrics` class in warpforge-backend-nvidia~~
   - ~~Replace simulated occupancy with real queries~~

   **Done**: Created `NvmlRuntime.java`, `GpuMonitoring.java`, updated `NvidiaBackend` and `OccupancyAdmissionValidation`.

### Completed (2026-01-27)

2. **ROCm SMI for AMD** (Backend Parity):
   - ~~Similar to NVML but for AMD GPUs~~
   - ~~`rocm_smi_lib` provides utilization metrics~~
   - ~~Create `RocmSmiRuntime.java` following `NvmlRuntime.java` pattern~~
   - ~~Update `AmdBackend` to implement `GpuMonitoring`~~

   **Done**: Created `RocmSmiRuntime.java`, updated `AmdBackend` to implement `GpuMonitoring`.

### Medium Priority

3. **Add Compute Kernels to Validations**:
   - Current validations use memory transfers only
   - Add simple compute kernels (vector add, reduction)
   - Verify compute correctness, not just timing

### Low Priority

4. **JFR Event Correlation**:
   - Add JFR event IDs that link to validation scenarios
   - Enable post-hoc analysis of validation results

---

## Action Items

+----+----------------------------------------+----------+-----------+
| #  | Task                                   | Priority | Status    |
+----+----------------------------------------+----------+-----------+
| 1  | Add NVML nvmlDeviceGetUtilization     | HIGH     | DONE      |
|    | bindings to warpforge-backend-nvidia  |          |           |
+----+----------------------------------------+----------+-----------+
| 2  | Implement real OccupancyTracker       | HIGH     | DONE      |
|    | using NVML queries                    |          |           |
+----+----------------------------------------+----------+-----------+
| 3  | Add rocm_smi bindings for AMD         | HIGH     | DONE      |
|    | occupancy metrics (Backend Parity)    |          |           |
+----+----------------------------------------+----------+-----------+
| 4  | Add compute kernel validations        | MEDIUM   | TODO      |
|    | (vector add, reduction)               |          |           |
+----+----------------------------------------+----------+-----------+

---

## Audit Completed

- **Date**: 2026-01-27
- **Auditor**: Claude Code
- **Files Reviewed**: 5 research validation files
- **Status**: 4/4 PASS (All validations use real GPU metrics)
