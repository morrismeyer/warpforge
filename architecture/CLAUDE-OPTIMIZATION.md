# VLIW SIMD Kernel Optimization Documentation

This document captures the optimization journey for Anthropic's VLIW SIMD performance take-home challenge, with analysis of how each optimization pattern maps to:

1. **Babylon Code Reflection** - Standalone implementations in `jdk.incubator.code`
2. **warpforge-optimization-babylon** - Custom WarpForge extensions
3. **Babylon-MCP-Claude Code backchannel** - Optimization hints beyond Babylon's native facilities
4. **warpforge-core-jfr TODOs** - JFR instrumentation needed to detect optimization opportunities

## Executive Summary

| Optimization | Cycles Saved | Babylon Path | JFR Events Needed |
|--------------|--------------|--------------|-------------------|
| SIMD Vectorization | ~147K → ~18K | `CoreOp.vectorize()` | `SlotUtilization` |
| 4-Way Batch Pipelining | ~18K → ~5K | Custom pattern matcher | `PipelineStall` |
| Gather/Compute Overlap | ~5K → ~4.7K | HAT-style memory planning | `MemoryLatency` |
| Cross-Round Gather Overlap | ~4.7K → ~4.4K | Dataflow analysis | `GatherStall` |
| Pipelined Hash | ~4.4K → ~4.3K | Instruction scheduling | `HashPipeline` |
| Staggered Index Completion | ~4.4K → ~4.3K | Live range analysis | `IndexCompute` |

---

## Target Architecture

### VLIW Machine Characteristics

```
SLOT_LIMITS = {
    "alu": 12,      # 12 scalar ALU ops per cycle
    "valu": 6,      # 6 vector ALU ops per cycle (VLEN=8 elements each)
    "load": 2,      # 2 memory loads per cycle
    "store": 2,     # 2 memory stores per cycle
    "flow": 1,      # 1 control flow op per cycle
}
VLEN = 8           # Vector length
N_CORES = 1        # Single core
```

### Key Constraints

1. **Memory latency**: Gather operations require 4 cycles (8 scalar loads at 2/cycle)
2. **Flow bottleneck**: Only 1 vselect operation per cycle
3. **Hash dependency chains**: 6 stages, each with op1→op3 dependency then op2
4. **Index computation**: 6 operations with sequential dependencies

---

## Optimization 1: SIMD Vectorization

### Problem
Scalar processing of 256 batch elements takes ~147K cycles.

### Solution
Use vector operations (vload, vstore, valu) to process VLEN=8 elements per operation.

### Before (Scalar)
```python
for i in range(batch_size):
    val = mem[inp_values_p + i]
    val = myhash(val ^ tree_val)
    mem[inp_values_p + i] = val
```

### After (Vectorized)
```python
for vec in range(batch_size // VLEN):
    emit_combined(load=[("vload", v_val, val_base)])
    emit_combined(valu=[("^", v_val, v_val, v_tree)])
    # ... hash stages ...
    emit_combined(store=[("vstore", val_base, v_val)])
```

### Cycles Impact
**147,734 → ~18,000** (8x improvement from VLEN=8)

### Babylon Implementation Path

**Standalone Babylon**: This is a canonical loop vectorization transformation. `jdk.incubator.code` could support:

```java
// CoreOp extension for vectorization
public static VectorizeOp vectorize(FuncOp func, int vectorWidth) {
    return new VectorizeTransformer(vectorWidth).transform(func);
}
```

The transformation would:
1. Identify loops with iteration count divisible by VLEN
2. Replace scalar ops with vector equivalents
3. Adjust memory access patterns

**warpforge-optimization-babylon**: WarpForge could provide a `VectorizePass` that uses Babylon's code model to:
- Analyze loop bounds statically
- Verify no cross-iteration dependencies
- Generate vectorized code model

**JFR Events Needed**:
- `warpforge.vliw.SlotUtilization` - Detect scalar code using <1/8th of VALU slots
- `warpforge.vliw.VectorizationOpportunity` - Flag loops amenable to vectorization

### warpforge-core-jfr TODO
```java
@Label("VLIW Slot Utilization")
@Category({"WarpForge", "VLIW"})
@Description("Tracks VALU slot utilization per cycle")
public class SlotUtilizationEvent extends Event {
    @Label("Cycle")
    public long cycle;

    @Label("VALU Slots Used")
    public int valuSlotsUsed;

    @Label("Max VALU Slots")
    public int maxValuSlots;

    @Label("Vectorization Candidate")
    public boolean vectorizationCandidate;
}
```

---

## Optimization 2: 4-Way Batch Pipelining (ILP)

### Problem
Processing vectors A, B, C, D sequentially leaves pipeline bubbles during memory operations.

### Solution
Interleave operations on 4 vectors to hide memory latency.

### Before (Sequential)
```python
# Vector A
emit_combined(load=[("vload", A['v_val'], A['val_base'])])
# ... all A operations ...
# Vector B
emit_combined(load=[("vload", B['v_val'], B['val_base'])])
# ... all B operations ...
```

### After (4-Way Interleaved)
```python
# Load all 4 vectors
emit_combined(load=[("vload", A['v_val'], A['val_base']), ("vload", A['v_idx'], A['idx_base'])])
emit_combined(load=[("vload", B['v_val'], B['val_base']), ("vload", B['v_idx'], B['idx_base'])])
# ...

# Process interleaved - while A hashes, B/C/D can overlap
emit_combined(valu=[
    ("^", A['v_val'], A['v_val'], A['v_tree']),
    ("^", B['v_val'], B['v_val'], B['v_tree']),
    ("^", C['v_val'], C['v_val'], C['v_tree']),
    ("^", D['v_val'], D['v_val'], D['v_tree']),
])
```

### Cycles Impact
**~18,000 → ~5,000**

### Babylon Implementation Path

**Standalone Babylon**: This requires inter-iteration analysis across multiple loop iterations. Babylon's `Body` abstraction could be extended:

```java
// Unroll and interleave transformation
public static FuncOp unrollAndInterleave(FuncOp func, String loopVar, int factor) {
    return new UnrollInterleaveTransformer(loopVar, factor).transform(func);
}
```

**warpforge-optimization-babylon**: More likely implementation path:
- Pattern match on independent loop iterations
- Build dependence graph
- Schedule operations to maximize slot utilization

**MCP Backchannel**: Claude Code could suggest interleaving opportunities by analyzing:
- Loop iteration independence
- Memory access patterns
- Current slot utilization

### warpforge-core-jfr TODO
```java
@Label("Pipeline Stall Detection")
@Category({"WarpForge", "VLIW"})
public class PipelineStallEvent extends Event {
    @Label("Stall Cycle")
    public long cycle;

    @Label("Stall Cause")
    public String stallCause; // "memory_latency", "dependency_chain", "resource_conflict"

    @Label("Idle ALU Slots")
    public int idleAluSlots;

    @Label("Idle VALU Slots")
    public int idleValuSlots;
}
```

---

## Optimization 3: Gather/Compute Overlap

### Problem
Gather operations (loading tree values for rounds 3+) take 4 cycles but VALU is idle.

### Solution
Start compute for earlier vectors while later vectors are gathering.

### Before
```python
# D gather (4 cycles, VALU idle)
for i in range(0, VLEN, 2):
    emit_combined(load=[("load", D['v_tree'] + i, D['s_addr'][i]),
                        ("load", D['v_tree'] + i + 1, D['s_addr'][i + 1])])

# A/B/C compute (VALU active, loads idle)
emit_combined(valu=[("^", A['v_val'], A['v_val'], A['v_tree']), ...])
```

### After (Overlapped)
```python
# D gather cycle 0-1 + A/B/C XOR
emit_combined(
    load=[("load", D['v_tree'] + 0, D['s_addr'][0]),
          ("load", D['v_tree'] + 1, D['s_addr'][1])],
    valu=[
        ("^", A['v_val'], A['v_val'], A['v_tree']),
        ("^", B['v_val'], B['v_val'], B['v_tree']),
        ("^", C['v_val'], C['v_val'], C['v_tree']),
    ]
)
# D gather cycle 2-3 + A/B/C hash stage 0
emit_combined(
    load=[("load", D['v_tree'] + 2, D['s_addr'][2]),
          ("load", D['v_tree'] + 3, D['s_addr'][3])],
    valu=[
        (op1_0, A['v_tmp1'], A['v_val'], v_hash_const[0]),
        (op3_0, A['v_tmp2'], A['v_val'], v_hash_shift[0]),
        # ...
    ]
)
```

### Cycles Impact
**~5,000 → ~4,700** (saves ~300 cycles)

### Babylon Implementation Path

**HAT-Style Memory Planning**: This is directly analogous to HAT's approach:

```java
// In HAT: memory planning for GPU operations
// Backend schedules loads to overlap with compute

// WarpForge extension:
public class GatherComputeOverlapper implements CodeTransformer {
    @Override
    public Block.Builder transform(Block.Builder bb, Op op) {
        if (op instanceof GatherOp gather) {
            // Find independent compute ops that can overlap
            List<Op> overlapCandidates = findIndependentOps(gather);
            return scheduleOverlapped(bb, gather, overlapCandidates);
        }
        return bb.op(op);
    }
}
```

**JFR Events Needed**:
- `warpforge.vliw.MemoryLatency` - Track cycles spent waiting for memory
- `warpforge.vliw.GatherStall` - Specifically track gather latency vs available compute

### warpforge-core-jfr TODO
```java
@Label("Memory Latency")
@Category({"WarpForge", "VLIW"})
public class MemoryLatencyEvent extends Event {
    @Label("Load Start Cycle")
    public long startCycle;

    @Label("Load Complete Cycle")
    public long completeCycle;

    @Label("VALU Ops During Wait")
    public int valuOpsDuringWait;

    @Label("Could Have Overlapped")
    public int missedOverlapOps;
}
```

---

## Optimization 4: Cross-Round Gather Overlap

### Problem
At end of each round, we compute indices then start next round's gather. These operations serialize.

### Solution
Start next round's A/B/C gather during current round's D index computation.

### Before
```python
# Round N: D index computation
emit_combined(valu=[("&", D['v_idx'], D['v_idx'], D['v_tmp1'])])

# Round N+1: A addresses and gather
emit_combined(alu=[("+", A['s_addr'][i], ...) for i in range(VLEN)])
for i in range(0, VLEN, 2):
    emit_combined(load=[("load", A['v_tree'] + i, A['s_addr'][i]), ...])
```

### After (Cross-Round Overlap)
```python
# D index in progress, A addresses + B mask
emit_combined(
    alu=[("+", A['s_addr'][i], self.scratch["forest_values_p"], A['v_idx'] + i) for i in range(VLEN)],
    valu=[
        ("&", B['v_idx'], B['v_idx'], B['v_tmp1']),
        ("-", C['v_tmp1'], v_zero, C['v_tmp1']),
    ]
)
# A gather + B addresses + C mask
emit_combined(
    alu=[("+", B['s_addr'][i], self.scratch["forest_values_p"], B['v_idx'] + i) for i in range(VLEN)],
    load=[("load", A['v_tree'] + 0, A['s_addr'][0]), ("load", A['v_tree'] + 1, A['s_addr'][1])],
    valu=[("&", C['v_idx'], C['v_idx'], C['v_tmp1'])]
)
```

### Cycles Impact
**~4,700 → ~4,400** (saves ~300 cycles)

### Babylon Implementation Path

**Dataflow Analysis Extension**: Babylon's code model needs cross-body dataflow:

```java
// Analyze dependencies across loop iterations
public class CrossIterationDataflow {
    // Identify values produced in iteration N needed in N+1
    public List<Value> interIterationLiveValues(Body loopBody) {
        // ...
    }

    // Schedule iteration N+1 loads during N's tail
    public FuncOp overlapIterations(FuncOp func, int overlapDepth) {
        // ...
    }
}
```

**warpforge-optimization-babylon**: This is a key optimization for GPU kernels and should be a first-class pass.

### warpforge-core-jfr TODO
```java
@Label("Cross-Round Gather Stall")
@Category({"WarpForge", "VLIW"})
public class GatherStallEvent extends Event {
    @Label("Round")
    public int round;

    @Label("Cycles Waiting for Gather")
    public int gatherWaitCycles;

    @Label("Parallel Compute Available")
    public boolean parallelComputeAvailable;

    @Label("Next Round Gather Preloadable")
    public boolean nextRoundPreloadable;
}
```

---

## Optimization 5: Pipelined Hash Computation

### Problem
Hash has 6 stages with op1/op3 → op2 dependency. Sequential: 18 cycles for 4 vectors (3 cycles × 6 stages).

### Solution
Pipeline: A/B run 1 stage ahead of C/D, allowing 6 ops/cycle.

### Before (Sequential per stage)
```python
for hi in range(6):
    # Stage hi: all 4 vectors do op1/op3
    emit_combined(valu=[
        (ops[hi][0], A['v_tmp1'], A['v_val'], v_hash_const[hi]),
        (ops[hi][2], A['v_tmp2'], A['v_val'], v_hash_shift[hi]),
        # ... B, C, D same stage
    ])
    # Stage hi: all 4 vectors do op2
    emit_combined(valu=[
        (ops[hi][1], A['v_val'], A['v_tmp1'], A['v_tmp2']),
        # ... B, C, D same stage
    ])
```

### After (Pipelined - A/B lead C/D by 1 stage)
```python
# Cycle 1: A/B/C op1/op3 (stage 0)
emit_combined(valu=[
    (ops[0][0], A['v_tmp1'], A['v_val'], v_hash_const[0]),
    (ops[0][2], A['v_tmp2'], A['v_val'], v_hash_shift[0]),
    (ops[0][0], B['v_tmp1'], B['v_val'], v_hash_const[0]),
    (ops[0][2], B['v_tmp2'], B['v_val'], v_hash_shift[0]),
    (ops[0][0], C['v_tmp1'], C['v_val'], v_hash_const[0]),
    (ops[0][2], C['v_tmp2'], C['v_val'], v_hash_shift[0]),
])
# Cycle 2: D op1/op3 (0) + A/B/C op2 (0)
emit_combined(valu=[
    (ops[0][0], D['v_tmp1'], D['v_val'], v_hash_const[0]),
    (ops[0][2], D['v_tmp2'], D['v_val'], v_hash_shift[0]),
    (ops[0][1], A['v_val'], A['v_tmp1'], A['v_tmp2']),
    (ops[0][1], B['v_val'], B['v_tmp1'], B['v_tmp2']),
    (ops[0][1], C['v_val'], C['v_tmp1'], C['v_tmp2']),
])
# Cycle 3: D op2 (0) + A/B op1/op3 (1)
emit_combined(valu=[
    (ops[0][1], D['v_val'], D['v_tmp1'], D['v_tmp2']),
    (ops[1][0], A['v_tmp1'], A['v_val'], v_hash_const[1]),
    (ops[1][2], A['v_tmp2'], A['v_val'], v_hash_shift[1]),
    (ops[1][0], B['v_tmp1'], B['v_val'], v_hash_const[1]),
    (ops[1][2], B['v_tmp2'], B['v_val'], v_hash_shift[1]),
])
# Cycles 4-12: steady state
for hi in range(1, 6):
    # C/D op1/op3 (hi) + A/B op2 (hi)
    emit_combined(valu=[
        (ops[hi][0], C['v_tmp1'], C['v_val'], v_hash_const[hi]),
        (ops[hi][2], C['v_tmp2'], C['v_val'], v_hash_shift[hi]),
        (ops[hi][0], D['v_tmp1'], D['v_val'], v_hash_const[hi]),
        (ops[hi][2], D['v_tmp2'], D['v_val'], v_hash_shift[hi]),
        (ops[hi][1], A['v_val'], A['v_tmp1'], A['v_tmp2']),
        (ops[hi][1], B['v_val'], B['v_tmp1'], B['v_tmp2']),
    ])
    # ...
```

### Cycles Impact
**18 cycles → 13 cycles** per hash (saves 5 cycles × rounds × vector groups)

### Babylon Implementation Path

**Instruction Scheduling**: This is classic software pipelining:

```java
// Babylon extension for instruction scheduling
public class SoftwarePipeliner implements CodeTransformer {
    private final int slotLimits;

    public FuncOp pipeline(FuncOp func, List<DependencyChain> chains) {
        // Model each chain as a DAG
        // Schedule operations to maximize slot utilization
        // Stagger chains to fill pipeline
    }
}
```

**MCP Backchannel Opportunity**: Claude Code could:
1. Identify hash-like patterns (N stages, each with dependency chain)
2. Suggest pipelining depth based on SLOT_LIMITS
3. Generate pipelined schedule automatically

### warpforge-core-jfr TODO
```java
@Label("Hash Pipeline Efficiency")
@Category({"WarpForge", "VLIW"})
public class HashPipelineEvent extends Event {
    @Label("Hash Stages")
    public int stages;

    @Label("Pipeline Depth")
    public int pipelineDepth;

    @Label("Cycles Per Stage (Sequential)")
    public int cyclesPerStageSequential;

    @Label("Cycles Per Stage (Pipelined)")
    public int cyclesPerStagePipelined;

    @Label("Speedup")
    public float speedup;
}
```

---

## Optimization 6: Staggered Index Completion

### Problem
Index computation has 6 sequential ops. All 4 vectors finish at same time → resource spike.

### Solution
Stagger completion: A finishes first, then B, then C, while D still computing.

### Before (All finish together)
```python
# All 4 vectors do each index step together
emit_combined(valu=[
    ("&", A['v_tmp1'], A['v_val'], v_one),
    ("&", B['v_tmp1'], B['v_val'], v_one),
    ("&", C['v_tmp1'], C['v_val'], v_one),
    ("&", D['v_tmp1'], D['v_val'], v_one),
])
# ... 5 more steps for all 4 ...
```

### After (Staggered)
```python
# A starts early, B follows, C/D trail
emit_combined(valu=[
    ("&", A['v_tmp1'], A['v_val'], v_one),
    ("*", A['v_idx'], A['v_idx'], v_two),
    (op2_5, C['v_val'], C['v_tmp1'], C['v_tmp2']),  # C still in hash
    (op2_3, D['v_val'], D['v_tmp1'], D['v_tmp2']),  # D further behind
])
emit_combined(valu=[
    ("+", A['v_tmp1'], A['v_tmp1'], v_one),         # A step 2
    ("&", B['v_tmp1'], B['v_val'], v_one),          # B step 0
    ("*", B['v_idx'], B['v_idx'], v_two),           # B step 1
    (op1_4, D['v_tmp1'], D['v_val'], v_hash_const[4]),
    # ...
])
```

### Cycles Impact
Enables earlier start of cross-round gather overlap, saves ~100 cycles

### Babylon Implementation Path

**Live Range Analysis**: Schedule operations to minimize register pressure and enable overlaps:

```java
public class StaggeredCompletionScheduler {
    // Analyze when each vector's values are last needed
    // Schedule completions to overlap with next-round setup

    public Schedule staggerCompletions(List<DependencyChain> chains, int overlapTarget) {
        // chains = [A_index, B_index, C_index, D_index]
        // Schedule A to complete early, D to complete late
        // Fill gaps with next-round address computation
    }
}
```

### warpforge-core-jfr TODO
```java
@Label("Index Computation Stagger")
@Category({"WarpForge", "VLIW"})
public class IndexComputeEvent extends Event {
    @Label("Vector ID")
    public String vectorId;  // "A", "B", "C", "D"

    @Label("Completion Cycle")
    public long completionCycle;

    @Label("Stagger Depth")
    public int staggerDepth;

    @Label("Overlap Enabled")
    public boolean overlapEnabled;
}
```

---

## warpforge-backend-nvidia Instrumentation Review

### Current PTX Instrumentation (CudaKernels.java)

The existing salt-based instrumentation is well-designed:

```java
// SALT_NONE (0) - Production kernel
// SALT_TIMING (1) - Cycle counters (~8 instructions overhead)
// SALT_TRACE (2) - Memory access patterns (higher overhead)

// Timing instrumentation pattern:
if (salt >= SALT_TIMING) {
    ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
}
// ... compute ...
if (salt >= SALT_TIMING) {
    ptx.append("    mov.u64         %rd_t1, %globaltimer;\n");
    ptx.append("    sub.u64         %rd_delta, %rd_t1, %rd_t0;\n");
    ptx.append("    atom.global.add.u64 [timing], %rd_delta;\n");
}
```

### warpforge-backend-nvidia TODOs

#### TODO 1: Slot Utilization Instrumentation

Add instrumentation to track how many operations execute per cycle in VLIW-style execution:

```java
// In CudaKernels.java - new salt level
public static final int SALT_SLOT_TRACE = 3;

// Generate instrumentation for slot utilization
if (salt >= SALT_SLOT_TRACE) {
    // Track warp execution mask (which threads active)
    ptx.append("    vote.sync.any.pred %p_any, 1, 0xffffffff;\n");
    // Track instruction mix
    ptx.append("    atom.global.add.u32 [slot_counters + 0], 1;  // ALU count\n");
}
```

#### TODO 2: Memory Latency Instrumentation

Track gather/scatter latency to identify overlap opportunities:

```java
// Before gather
ptx.append("    mov.u64 %rd_gather_start, %globaltimer;\n");

// After gather complete
ptx.append("    mov.u64 %rd_gather_end, %globaltimer;\n");
ptx.append("    sub.u64 %rd_gather_lat, %rd_gather_end, %rd_gather_start;\n");
ptx.append("    atom.global.add.u64 [gather_latency], %rd_gather_lat;\n");
```

#### TODO 3: Dependency Chain Instrumentation

Identify long dependency chains (like hash stages):

```java
// Mark dependency chain boundaries
ptx.append("    // CHAIN_START: hash_stage_0\n");
// ... chain operations ...
ptx.append("    // CHAIN_END: hash_stage_0\n");
```

Post-process PTX to extract chain lengths and suggest pipelining.

---

## warpforge-core-jfr TODO Summary

| Event Class | Purpose | Triggers Optimization |
|-------------|---------|----------------------|
| `SlotUtilizationEvent` | Track VALU/ALU slot fill rate | Vectorization, ILP |
| `PipelineStallEvent` | Detect pipeline bubbles | Interleaving, overlap |
| `MemoryLatencyEvent` | Track load latency | Gather/compute overlap |
| `GatherStallEvent` | Cross-round gather waits | Cross-round overlap |
| `HashPipelineEvent` | Hash pipeline efficiency | Software pipelining |
| `IndexComputeEvent` | Index completion timing | Staggered completion |

### Implementation Priority

1. **High Priority**: `SlotUtilizationEvent`, `PipelineStallEvent` - These identify the most impactful optimizations
2. **Medium Priority**: `MemoryLatencyEvent`, `GatherStallEvent` - Memory-bound workloads
3. **Lower Priority**: `HashPipelineEvent`, `IndexComputeEvent` - Specific patterns

---

## Babylon Code Reflection Integration Points

### jdk.incubator.code Extensions Needed

1. **VectorizeOp**: Transform scalar loops to vector operations
2. **UnrollInterleaveOp**: Unroll and interleave for ILP
3. **OverlapOp**: Schedule compute during memory latency
4. **PipelineOp**: Software pipeline dependency chains

### HAT Integration

HAT's backend abstraction maps directly:

```
HAT Backend          →  VLIW Optimization
─────────────────────────────────────────
OpenCL WorkGroup     →  Vector batch (VLEN)
CUDA Block           →  4-way batch (A/B/C/D)
Memory Coalescing    →  Gather optimization
Kernel Fusion        →  Cross-round overlap
```

The HAT FFI backends (cuda, opencl, hip) provide the template for WarpForge:
- Same algorithm, different execution engine
- Salt-based instrumentation for all paths
- Babylon code model as unified IR

---

## Conclusion

The VLIW SIMD optimization journey reveals patterns that map cleanly to Babylon's code reflection framework:

1. **Loop transformations** (vectorize, unroll, interleave) - Standard compiler passes
2. **Memory scheduling** (overlap, prefetch) - HAT-style backend planning
3. **Instruction scheduling** (pipelining, staggering) - Classic code generation

The key insight is that **JFR instrumentation can detect optimization opportunities at runtime**, enabling:
- Claude Code to suggest transformations via MCP backchannel
- WarpForge to auto-tune based on collected profiles
- Babylon to encode patterns as reusable transformations

The warpforge-backend-nvidia salt-based instrumentation provides the foundation. Extending it with slot utilization and dependency chain tracking would enable automated detection of all optimizations documented here.

---

## Detailed Babylon Code Reflection Mapping

Based on thorough review of `jdk.incubator.code` and HAT, here are the specific extension points:

### Op.java Key Abstractions

The `Op` class provides several interface markers that are directly applicable:

```java
// From Op.java - relevant to VLIW optimization

public interface Pure {
    // Pure operations can be freely reordered
    // Key for: Pipelined hash, cross-round overlap
}

public interface Loop extends Nested {
    Body loopBody();
    // Provides access to loop body for vectorization analysis
}

public interface Lowerable {
    Block.Builder lower(Block.Builder b, CodeTransformer opT);
    // Used to transform high-level ops to VLIW instructions
    // Key for: All optimizations - lowering is the code generation step
}
```

**WarpForge Extension Point**: Create `VLIWLoweringTransformer` that implements `Lowerable` lowering with VLIW slot packing:

```java
public class VLIWLoweringTransformer implements CodeTransformer {
    private final int[] slotLimits; // {alu: 12, valu: 6, load: 2, store: 2, flow: 1}

    @Override
    public FuncOp transform(FuncOp func) {
        // Group ops by execution unit
        // Pack into minimum cycles respecting slot limits
        // Track dependencies to prevent illegal reordering
    }
}
```

### Body.java Key Methods

```java
// From Body.java - dataflow analysis for optimizations

public Map<Block, Block> immediateDominators();
// Used for: SSA conversion, dependency chain analysis

public Map<Block, Set<Block>> dominanceFrontier();
// Used for: Cross-round overlap - identify values that flow across iterations

public Map<Block, Block> immediatePostDominators();
// Used for: Live range analysis for staggered completion
```

**WarpForge Extension Point**: Create `CrossIterationAnalyzer`:

```java
public class CrossIterationAnalyzer {
    // Analyze loop body to find:
    // 1. Values computed in iteration N needed in N+1
    // 2. Memory operations that can be overlapped with compute
    // 3. Independent computation chains that can be interleaved

    public List<Value> findCrossIterationLiveValues(Body loopBody);
    public List<GatherOverlapCandidate> findGatherOverlapOpportunities(Body loopBody);
}
```

### SSA.java - Dependency Analysis Foundation

```java
// From analysis/SSA.java

public static <T extends Op & Op.Nested> T transform(T nestedOp) {
    // Transforms to SSA form - prerequisite for dependency analysis
    // Enables: Identifying true data dependencies vs anti-dependencies
}
```

**WarpForge Extension Point**: Build on SSA for VLIW scheduling:

```java
public class VLIWScheduler {
    // Input: SSA-form FuncOp
    // Output: FuncOp with ops grouped into VLIW bundles

    public FuncOp schedule(FuncOp ssaFunc, SlotLimits limits) {
        // Build def-use chains from SSA
        // Compute earliest possible cycle for each op
        // Pack ops into cycles respecting:
        //   - Data dependencies (def must precede use)
        //   - Resource constraints (slot limits)
        //   - Memory latency (loads have N-cycle latency)
    }
}
```

### HAT Integration Points

HAT's `Backend.java` provides the abstraction for GPU code generation:

```java
// From hat/backend/Backend.java

public abstract void dispatchKernel(KernelCallGraph kernelCallGraph,
                                     KernelContext kernelContext,
                                     Object... args);
```

**WarpForge Extension**: Implement `VLIWBackend extends Backend`:

```java
public class VLIWBackend extends Backend {
    // Target the VLIW emulator instead of GPU
    // Use HAT's buffer tracking for memory planning
    // Generate VLIW instruction bundles instead of PTX/SPIR-V

    @Override
    public void dispatchKernel(KernelCallGraph kcg, KernelContext ctx, Object... args) {
        // 1. Get FuncOp from kernel call graph
        // 2. Transform to SSA
        // 3. Apply vectorization pass
        // 4. Apply VLIW scheduling
        // 5. Emit VLIW instruction stream
    }
}
```

HAT's `HATOpDispatcher.java` shows how to handle HAT-specific operations:

```java
// From hat/codebuilders/HATOpDispatcher.java

T hatVectorLoadOp(HATVectorOp.HATVectorLoadOp hatVectorLoadOp);
T hatVectorStoreOp(HATVectorOp.HATVectorStoreView hatFloat4StoreOp);
T hatBinaryVectorOp(HATVectorOp.HATVectorBinaryOp hatVectorBinaryOp);
```

**WarpForge Extension**: Create `VLIWOpDispatcher`:

```java
public interface VLIWOpDispatcher extends HATOpDispatcher<VLIWCodeBuilder> {
    // Handle VLIW-specific operations

    VLIWCodeBuilder vlwGatherOp(VLIWGatherOp gatherOp);
    // Generates 8 scalar loads packed into 4 cycles

    VLIWCodeBuilder vlwHashOp(VLIWHashOp hashOp);
    // Generates pipelined hash computation
}
```

---

## MCP Backchannel Design

The Babylon-MCP-Claude Code backchannel enables optimization suggestions beyond what static analysis can detect:

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Claude Code (MCP Client)                                           │
│                                                                     │
│  Receives: JFR events from instrumented execution                   │
│  Analyzes: Slot utilization, pipeline stalls, memory latency        │
│  Suggests: Optimization transformations                             │
│                                                                     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ MCP Protocol
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  WarpForge MCP Server                                               │
│                                                                     │
│  Exposes:                                                           │
│  - /optimize/vectorize       - Apply vectorization                  │
│  - /optimize/interleave      - Apply N-way interleaving             │
│  - /optimize/pipeline-hash   - Apply hash pipelining                │
│  - /optimize/overlap-gather  - Apply gather/compute overlap         │
│  - /profile/slot-utilization - Get slot usage data                  │
│  - /profile/dependency-chain - Get chain analysis                   │
│                                                                     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  WarpForge Optimization Engine                                      │
│                                                                     │
│  - VectorizePass                                                    │
│  - InterleavePass                                                   │
│  - PipelinePass                                                     │
│  - GatherOverlapPass                                                │
│  - CrossRoundOverlapPass                                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### MCP Tool Definitions

```json
{
  "tools": [
    {
      "name": "optimize_vectorize",
      "description": "Apply SIMD vectorization to a loop",
      "inputSchema": {
        "type": "object",
        "properties": {
          "funcOp": {"type": "string", "description": "Serialized FuncOp"},
          "loopId": {"type": "string", "description": "Loop to vectorize"},
          "vectorWidth": {"type": "integer", "description": "SIMD width"}
        }
      }
    },
    {
      "name": "profile_slot_utilization",
      "description": "Get slot utilization data from JFR",
      "inputSchema": {
        "type": "object",
        "properties": {
          "funcOp": {"type": "string"},
          "cycleRange": {"type": "array", "items": {"type": "integer"}}
        }
      }
    }
  ]
}
```

### Claude Code Optimization Flow

1. **Profile**: Execute kernel with SALT_TIMING instrumentation
2. **Analyze**: Claude Code receives JFR events showing:
   - Cycles with <50% VALU slot usage → vectorization opportunity
   - Cycles with loads but idle VALU → gather/compute overlap opportunity
   - Long dependency chains → pipelining opportunity
3. **Suggest**: Claude Code calls MCP tools to apply transformations
4. **Verify**: Re-profile to confirm improvement

---

## Current Status and Next Steps

**Current Performance**: 4,283 cycles (down from 147,734 baseline)

**Remaining Gap to Target**: Need to reach <2,164 cycles (~50% reduction needed)

### Potential Further Optimizations

1. **Deeper Hash Pipelining**: Current pipelining is 2-deep (A/B lead C/D). Could try 4-deep where each vector is at different stage.

2. **Loop Fusion**: Fuse the 16 rounds into fewer iterations with deeper software pipelining.

3. **Register Pressure Optimization**: Reduce scratch memory usage by more aggressive register allocation.

4. **Prefetch Optimization**: Start next vector group's loads during current group's final stores.

### warpforge-core-jfr Implementation Priority

| Priority | Event | Purpose |
|----------|-------|---------|
| P0 | `SlotUtilizationEvent` | Detect under-utilized cycles |
| P0 | `PipelineStallEvent` | Detect dependency-caused stalls |
| P1 | `MemoryLatencyEvent` | Track load completion timing |
| P1 | `GatherStallEvent` | Specifically track gather operations |
| P2 | `HashPipelineEvent` | Track hash stage timing |
| P2 | `IndexComputeEvent` | Track index computation timing |

### warpforge-backend-nvidia Implementation Priority

| Priority | Feature | Purpose |
|----------|---------|---------|
| P0 | `SALT_SLOT_TRACE` level | Track slot utilization in PTX |
| P1 | Dependency chain markers | Identify optimization opportunities |
| P2 | Memory access pattern tracking | Enable gather optimization |

---

## Session Log: Optimization Attempts

### Starting State
- **Previous best**: 4,283 cycles (from prior optimization session)
- **Current file**: Reverted to baseline at 147,734 cycles

### Analysis Results

**Instruction Distribution (from 4,283 cycle version)**:
```
alu:   3,399 ops
valu:  13,464 ops (13464 / 6 = 2,244 cycles minimum)
load:  3,460 ops (3460 / 2 = 1,730 cycles minimum)
store: 64 ops
flow:  130 ops (vselects + pauses)
```

**Key Insight**: The theoretical minimum is ~2,244 cycles (VALU-limited). At 4,283 cycles, we're at ~52% VALU slot utilization. The gap to 2,164 cycles target requires near-perfect packing.

### Attempted Optimizations

#### Attempt 1: Round 1 vselect/hash overlap
- **Approach**: Overlap vselect (flow engine) with XOR and hash start (valu engine)
- **Result**: FAILED - incorrect output
- **Root cause**: Hash pipelining dependencies were incorrectly tracked when interleaving with vselect

#### Attempt 2: Round 2 vselect/hash overlap
- **Approach**: More aggressive overlap of 12 vselects with XOR and hash stages
- **Result**: FAILED - incorrect output
- **Root cause**: Complex staggered pipeline with 4 vectors at different stages introduced ordering bugs

### Lessons Learned

1. **Correctness is paramount**: Each optimization must preserve exact output. The reference kernel `reference_kernel2` is the ground truth.

2. **Dependency tracking is critical**: When pipelining across multiple vectors at different stages, tracking which operation depends on which result becomes complex.

3. **vselect is a bottleneck**: 130 flow operations at 1/cycle is significant. But overlapping them with valu requires careful dependency analysis.

4. **Theoretical vs practical limits**: Even with 52% VALU utilization, reaching 100% is extremely difficult due to:
   - Dependency chains forcing serialization
   - Memory latency (gather = 4 cycles)
   - Flow bottleneck (1 vselect/cycle)

### Recommendations for Future Work

1. **Automated verification**: Build incremental tests that validate each cycle's intermediate state against reference.

2. **Dependency graph visualization**: Generate a DAG of all operations and their dependencies to guide manual optimization.

3. **JFR instrumentation**: Implement `SlotUtilizationEvent` and `PipelineStallEvent` to identify exactly where cycles are wasted.

4. **Babylon code model**: Encode the optimization transformations in Babylon IR so they can be:
   - Verified for correctness via symbolic execution
   - Applied automatically with dependency checking
   - Composed to find optimal schedules

### Files Created

- `/Users/morris/surfworks/original_performance_takehome/architecture/CLAUDE-OPTIMIZATION.md` - This document
