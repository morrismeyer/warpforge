# Research Validation Audit Results

This document contains the results of auditing all research validation implementations in `ptest/src/main/java/.../research/`.

## Summary

+--------------------------------+-------------+---------------------+
| Validation                     | GPU Work    | Correctness Issue   |
+--------------------------------+-------------+---------------------+
| OverlappingIoValidation        | REAL        | None                |
| PipelineBubbleValidation       | REAL        | None                |
| OccupancyAdmissionValidation   | REAL (work) | SIMULATED OCCUPANCY |
| SloInferenceValidation         | REAL        | None                |
| ResearchValidationRunner       | N/A         | None                |
+--------------------------------+-------------+---------------------+

**Overall Verdict**: 3/4 PASS, 1 NEEDS WORK

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

**Verdict**: NEEDS WORK - Uses simulated occupancy

**Purpose**: Validates occupancy-based admission control per Orion paper.

**GPU Work** (real operations in `doInferenceWork()`, lines 272-281):
```java
Tensor device = backend.copyToDeviceAsync(host, streamHandle);
backend.synchronizeStream(streamHandle);
Tensor result = backend.copyToHostAsync(device, streamHandle);
backend.synchronizeStream(streamHandle);
```

**PROBLEM**: Occupancy tracking is SIMULATED, not real NVML:

```java
// Line 74 - This is SIMULATED occupancy, not real SM utilization!
tracker.recordOccupancy(level * 3); // Simulated SM usage
```

The `OccupancyTracker` class (lines 286-305) just stores integer values - it does NOT call:
- `nvmlDeviceGetUtilizationRates()` for GPU utilization
- `cupti` for SM occupancy metrics
- Any real hardware query

**Architecture Doc Requirement** (from GPU-SCHEDULING.md):
> For Orion-style occupancy-aware scheduling, we need real SM occupancy from NVML

**What's Missing**:
1. NVML bindings for `nvmlDeviceGetUtilizationRates()`
2. Real-time SM utilization queries
3. Actual occupancy-based admission decisions

**Scenarios**:
1. `validateTrackOccupancy()` - Uses simulated occupancy (FAKE)
2. `validateAdmissionControl()` - Admission decisions based on fake occupancy
3. `validateThroughputGain()` - Real GPU work, but doesn't prove Orion's insight

**Fix Required**:
1. Add NVML bindings in warpforge-backend-nvidia
2. Implement `GpuBackend.getSmUtilization()` method
3. Replace `OccupancyTracker` with real NVML queries

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

### High Priority

1. **Fix OccupancyAdmissionValidation**:
   - Add NVML bindings: `nvmlDeviceGetUtilizationRates()`, `nvmlDeviceGetMemoryInfo()`
   - Create `NvmlMetrics` class in warpforge-backend-nvidia
   - Replace simulated occupancy with real queries
   - See architecture/GPU-SCHEDULING.md for NVML API requirements

### Medium Priority

2. **Add Compute Kernels to Validations**:
   - Current validations use memory transfers only
   - Add simple compute kernels (vector add, reduction)
   - Verify compute correctness, not just timing

3. **ROCm SMI for AMD**:
   - Similar to NVML but for AMD GPUs
   - `rocm_smi_lib` provides utilization metrics

### Low Priority

4. **JFR Event Correlation**:
   - Add JFR event IDs that link to validation scenarios
   - Enable post-hoc analysis of validation results

---

## Action Items

+----+----------------------------------------+----------+
| #  | Task                                   | Priority |
+----+----------------------------------------+----------+
| 1  | Add NVML nvmlDeviceGetUtilization     | HIGH     |
|    | bindings to warpforge-backend-nvidia  |          |
+----+----------------------------------------+----------+
| 2  | Implement real OccupancyTracker       | HIGH     |
|    | using NVML queries                    |          |
+----+----------------------------------------+----------+
| 3  | Add rocm_smi bindings for AMD         | MEDIUM   |
|    | occupancy metrics                     |          |
+----+----------------------------------------+----------+
| 4  | Add compute kernel validations        | MEDIUM   |
|    | (vector add, reduction)               |          |
+----+----------------------------------------+----------+

---

## Audit Completed

- **Date**: 2026-01-27
- **Auditor**: Claude Code
- **Files Reviewed**: 5 research validation files
- **Status**: 3/4 PASS, 1 NEEDS WORK (OccupancyAdmissionValidation)
