# Backend Development Phases

This document describes the phased approach to GPU backend development in WarpForge, progressing from simple op-by-op execution to sophisticated fused kernel generation via Babylon Code Reflection.

## Background: How PyTorch Executes Operations

Understanding PyTorch's execution model informs our architecture:

### Eager Mode (Default)

In standard PyTorch eager execution, each operation launches a **separate GPU kernel**:

```python
a = op_a(tensor)   # Kernel 1: read input, compute, write intermediate
b = op_b(a)        # Kernel 2: read intermediate, compute, write intermediate
c = op_c(b)        # Kernel 3: read intermediate, compute, write output
```

Each kernel incurs:
- Kernel launch overhead
- Global memory read for inputs
- Global memory write for outputs

For a chain of elementwise operations, this means unnecessary memory traffic - `op_b` reads from global memory what `op_a` just wrote, even though the intermediate could stay in registers.

### torch.compile (PyTorch 2.0+)

With `torch.compile`, TorchDynamo captures the graph and TorchInductor can **fuse** compatible operations:

```python
@torch.compile
def fused(tensor):
    a = op_a(tensor)
    b = op_b(a)
    c = op_c(b)
    return c
```

The fused kernel does: **read once → compute A → compute B → compute C → write once**.

WarpForge aims to achieve similar fusion, but implemented in Java via Babylon Code Reflection.

## Phase 1: Op-by-Op Execution

**Goal**: Establish end-to-end correctness with the simplest possible implementation.

### Architecture

```
StableHLO Op Graph
       |
       v
+----------------------------------+
|  warpforge-backend-nvidia        |
|  +----------------------------+  |
|  | For each op in graph:      |  |
|  |   - Allocate output tensor |  |
|  |   - Call cuBLAS/cuDNN      |  |
|  |   - Synchronize            |  |
|  +----------------------------+  |
+----------------------------------+
```

### Implementation

Each StableHLO operation maps to a vendor library call:

|----------------|---------------------------|----------------------------|
| StableHLO Op   | NVIDIA (cuBLAS/cuDNN)     | AMD (hipBLAS/MIOpen)       |
|----------------|---------------------------|----------------------------|
| `dot_general`  | `cublasSgemm`             | `hipblasSgemm`             |
| `convolution`  | `cudnnConvolutionForward` | `miopenConvolutionForward` |
| `add`          | Custom CUDA kernel        | Custom HIP kernel          |
| `multiply`     | Custom CUDA kernel        | Custom HIP kernel          |
| `reduce`       | `cublasSasum` / custom    | `hipblasSasum` / custom    |
|----------------|---------------------------|----------------------------|

### Java FFM Bindings

Use Java's Foreign Function & Memory API to call into native libraries:

```java
public class CuBLAS {
    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup CUBLAS = SymbolLookup.libraryLookup("cublas", Arena.global());

    private static final MethodHandle cublasSgemm = LINKER.downcallHandle(
        CUBLAS.find("cublasSgemm_v2").orElseThrow(),
        FunctionDescriptor.of(JAVA_INT, /* ... args ... */)
    );

    public static void sgemm(/* args */) {
        cublasSgemm.invokeExact(/* args */);
    }
}
```

### Deliverables

- [ ] `warpforge-backend-nvidia` with cuBLAS/cuDNN FFM bindings
- [ ] `warpforge-backend-amd` with hipBLAS/MIOpen FFM bindings
- [ ] Op-by-op executor that walks StableHLO graph
- [ ] EndToEnd tests comparing WarpForge output to PyTorch reference

### Success Criteria

- Numerical correctness within tolerance (fp32: 1e-5 relative error)
- All StableHLO ops from traced models supported
- Tests pass on both NVIDIA and AMD hardware

## Phase 2: Elementwise Fusion

**Goal**: Fuse chains of elementwise operations into single kernels.

### Fusion Candidates

Elementwise operations can always fuse when connected:

```
relu → mul → add → tanh
```

Becomes a single kernel that:
1. Reads input once from global memory
2. Computes all operations using registers
3. Writes output once to global memory

### Architecture

```
StableHLO Op Graph
       |
       v
+----------------------------------+
|  Fusion Analysis (Java)          |
|  - Identify elementwise chains   |
|  - Group into FusionClusters     |
+--------------+-------------------+
               |
       +-------+-------+
       v               v
   Unfused Ops    FusionClusters
       |               |
       v               v
   cuBLAS calls   Babylon Codegen
                       |
                       v
               +-------------------+
               | warpforge-backend-|
               | babylon-ptx       |
               |                   |
               | Code Model -> PTX |
               +-------------------+
```

### Babylon Code Generation

The fused kernel is represented as a Babylon Code Model:

```java
// Conceptual representation of fused relu → mul → add
FuncOp fusedKernel = func("fused_relu_mul_add",
    funcType(List.of(ptrType(FLOAT), ptrType(FLOAT), ptrType(FLOAT), INT), VOID))
    .body(b -> {
        var input = b.param(0);
        var bias = b.param(1);
        var output = b.param(2);
        var n = b.param(3);

        b.forLoop(0, n, idx -> {
            var v = b.load(input, idx);
            v = b.max(v, b.constant(0.0f));      // relu
            v = b.mul(v, b.constant(2.0f));      // scale
            v = b.add(v, b.load(bias, idx));     // bias add
            b.store(output, idx, v);
        });
    });
```

This Code Model then lowers to PTX or HIP:

```java
String ptx = PTXLowering.lower(fusedKernel);
String hip = HIPLowering.lower(fusedKernel);
```

### Deliverables

- [ ] `FusionAnalyzer` - identifies fusable operation chains
- [ ] `FusionCluster` - represents a group of fused ops
- [ ] `warpforge-backend-babylon-ptx` - Babylon Code Model → PTX
- [ ] `warpforge-backend-babylon-rocm` - Babylon Code Model → HIP
- [ ] Runtime kernel compilation and caching

### Success Criteria

- Fused kernels produce identical results to unfused execution
- Measurable reduction in kernel launch count
- Memory bandwidth reduction for fused chains

## Phase 3: Reduction Fusion

**Goal**: Fuse reductions with adjacent elementwise operations.

### Patterns

```
# Pattern 1: Elementwise → Reduce
mul → sum

# Pattern 2: Reduce → Elementwise (broadcast)
sum → div (for mean)

# Pattern 3: Softmax pattern
exp → sum → div
```

### Complexity

Reductions change the parallelism pattern:
- Elementwise: one thread per element
- Reduction: tree-based parallel reduction

Fusion requires careful handling of the transition between patterns.

### Deliverables

- [ ] Reduction fusion patterns in `FusionAnalyzer`
- [ ] Parallel reduction Code Model templates
- [ ] Softmax and layer norm as single fused kernels

## Phase 4: MatMul Fusion

**Goal**: Fuse matrix multiplication with bias and activation.

### Patterns

```
# Dense layer
matmul → add (bias) → relu

# Attention QKV
matmul → reshape → transpose
```

### Approach Options

1. **cuBLAS epilogue fusion** - NVIDIA provides fused GEMM+bias+activation
2. **Custom GEMM kernel** - Full control but significant complexity
3. **Hybrid** - Use cuBLAS for GEMM, fuse only the epilogue

Option 3 is pragmatic: cuBLAS GEMM is highly optimized, and we add value by fusing the epilogue operations that cuBLAS doesn't handle.

### Deliverables

- [ ] MatMul epilogue fusion detection
- [ ] Integration with cuBLAS epilogue APIs (where available)
- [ ] Custom epilogue fusion for unsupported patterns

## Three-Tier Kernel Architecture

Orthogonal to the development phases, WarpForge provides three execution **tiers** that balance performance against observability. These tiers apply across all phases.

```
+-----------------------------------------------------------------------+
|  Tier              | Performance | Observability | Implementation     |
+-----------------------------------------------------------------------+
|  PRODUCTION        |    100%     | External only | cuBLAS/rocBLAS     |
|  OPTIMIZED_        |    ~93%     | Salt instr.   | Optimized PTX/HIP  |
|   OBSERVABLE       |             |               |                    |
|  CORRECTNESS       |    ~1%      | Full tracing  | Naive PTX/HIP      |
+-----------------------------------------------------------------------+
```

### Use Case Selection

|---------------------------|----------------------|----------------------------|
| Scenario                  | Tier                 | Rationale                  |
|---------------------------|----------------------|----------------------------|
| Training at scale         | PRODUCTION           | Maximum throughput         |
| "Which op is slow?"       | PRODUCTION + JFR     | External timing sufficient |
| "Why is this GEMM slow?"  | OPTIMIZED_OBSERVABLE | Need kernel internals      |
| "Results don't match"     | CORRECTNESS          | Full numerical trace       |
|---------------------------|----------------------|----------------------------|

The OPTIMIZED_OBSERVABLE tier is inspired by [Simon Boehm's work](https://siboehm.com/articles/22/CUDA-MMM) showing 93.7% of cuBLAS performance is achievable with optimized CUDA using coalescing, shared memory tiling, register blocking, and warptiling. This allows salt instrumentation while maintaining near-production speed.

See [BACKEND-KERNEL-INSTRUMENTATION.md](BACKEND-KERNEL-INSTRUMENTATION.md) for detailed implementation.

## The Two-Tier Backend Structure

After Phase 2+, we have two categories of backends:

### Library Backends (Phase 1)

Direct FFM calls to vendor libraries. Used for:
- Operations that don't benefit from fusion (large standalone matmuls)
- Fallback when fusion isn't applicable
- Operations with highly-optimized library implementations

```
warpforge-backend-nvidia  →  cuBLAS, cuDNN
warpforge-backend-amd     →  hipBLAS, MIOpen
```

### Babylon Codegen Backends (Phase 2+)

Babylon Code Reflection generates kernels. Used for:
- Fused operation chains
- Custom operations not in vendor libraries
- Cross-vendor portable kernels

```
warpforge-backend-babylon-ptx   →  NVIDIA GPUs
warpforge-backend-babylon-rocm  →  AMD GPUs
```

### Unified Execution Model

The executor chooses the appropriate backend per operation:

```java
public class HybridExecutor {

    void execute(OpGraph graph) {
        List<FusionCluster> clusters = fusionAnalyzer.analyze(graph);

        for (FusionCluster cluster : clusters) {
            if (cluster.isSingleOp() && hasOptimizedLibraryImpl(cluster.op())) {
                // Use vendor library
                libraryBackend.execute(cluster.op());
            } else {
                // Generate and execute fused kernel
                FuncOp kernel = codeGenerator.generate(cluster);
                babylonBackend.execute(kernel);
            }
        }
    }
}
```

## HAT Integration Potential

Oracle's HAT (Heterogeneous Accelerator Toolkit) is exploring similar territory - using Babylon for accelerator targeting. Potential integration paths:

1. **Build on HAT** when it stabilizes
2. **Contribute to HAT** from WarpForge learnings
3. **Parallel development** that could merge later

The key is that our architecture (Code Model → target lowering) aligns with HAT's direction.

## Timeline and Dependencies

```
Phase 1 ──────────────────────────────────────────────────────────►
        │ FFM bindings, op-by-op execution, correctness tests
        │
        └─► Phase 2 ──────────────────────────────────────────────►
                    │ Fusion analysis, elementwise fusion, Babylon codegen
                    │
                    └─► Phase 3 ────────────────────────────────────►
                                │ Reduction fusion
                                │
                                └─► Phase 4 ────────────────────────►
                                            │ MatMul fusion
```

Each phase builds on the previous. Phase 1's correctness tests become the regression suite for Phase 2+.

## Metrics and Validation

### Correctness

- Numerical comparison against PyTorch reference outputs
- Bit-exact comparison where applicable (integer ops)
- Tolerance-based comparison for floating point (relative error < 1e-5)

### Performance (Phase 2+)

- Kernel launch count reduction
- Memory bandwidth utilization
- End-to-end model inference time

### Coverage

- Percentage of StableHLO ops supported
- Percentage of operations executed via fused kernels (Phase 2+)
