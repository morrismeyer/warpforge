# VLIW SIMD Kernel Optimization Documentation

This document captures the optimization journey for Anthropic's VLIW SIMD performance take-home challenge, with analysis of how each optimization pattern maps to:

1. **Babylon Code Reflection** - Standalone implementations in `jdk.incubator.code`
2. **warpforge-optimization-babylon** - Custom WarpForge extensions
3. **Babylon-MCP-Claude Code backchannel** - Optimization hints beyond Babylon's native facilities
4. **warpforge-core-jfr TODOs** - JFR instrumentation needed to detect optimization opportunities

## Executive Summary

| Optimization               | Cycles Saved    | Babylon Path              | JFR Events Needed  |
|----------------------------|-----------------|---------------------------|--------------------|
| SIMD Vectorization         | ~147K → ~18K    | `CoreOp.vectorize()`      | `SlotUtilization`  |
| 4-Way Batch Pipelining     | ~18K → ~5K      | Custom pattern matcher    | `PipelineStall`    |
| Gather/Compute Overlap     | ~5K → ~4.7K     | HAT-style memory planning | `MemoryLatency`    |
| Cross-Round Gather Overlap | ~4.7K → ~4.4K   | Dataflow analysis         | `GatherStall`      |
| Pipelined Hash             | ~4.4K → ~4.3K   | Instruction scheduling    | `HashPipeline`     |
| Staggered Index Completion | ~4.4K → ~4.3K   | Live range analysis       | `IndexCompute`     |

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

| Event Class            | Purpose                       | Triggers Optimization   |
|------------------------|-------------------------------|-------------------------|
| `SlotUtilizationEvent` | Track VALU/ALU slot fill rate | Vectorization, ILP      |
| `PipelineStallEvent`   | Detect pipeline bubbles       | Interleaving, overlap   |
| `MemoryLatencyEvent`   | Track load latency            | Gather/compute overlap  |
| `GatherStallEvent`     | Cross-round gather waits      | Cross-round overlap     |
| `HashPipelineEvent`    | Hash pipeline efficiency      | Software pipelining     |
| `IndexComputeEvent`    | Index completion timing       | Staggered completion    |

### Implementation Priority

1. **High Priority**: `SlotUtilizationEvent`, `PipelineStallEvent` - Most impactful optimizations
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
HAT Backend            VLIW Optimization
────────────────────   ─────────────────────────
OpenCL WorkGroup   ──▶ Vector batch (VLEN)
CUDA Block         ──▶ 4-way batch (A/B/C/D)
Memory Coalescing  ──▶ Gather optimization
Kernel Fusion      ──▶ Cross-round overlap
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
┌─────────────────────────────────────────────────────────────────────────┐
│  Claude Code (MCP Client)                                               │
│                                                                         │
│  Receives: JFR events from instrumented execution                       │
│  Analyzes: Slot utilization, pipeline stalls, memory latency            │
│  Suggests: Optimization transformations                                 │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ MCP Protocol
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  WarpForge MCP Server                                                   │
│                                                                         │
│  Exposes:                                                               │
│    - /optimize/vectorize         Apply vectorization                    │
│    - /optimize/interleave        Apply N-way interleaving               │
│    - /optimize/pipeline-hash     Apply hash pipelining                  │
│    - /optimize/overlap-gather    Apply gather/compute overlap           │
│    - /profile/slot-utilization   Get slot usage data                    │
│    - /profile/dependency-chain   Get chain analysis                     │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  WarpForge Optimization Engine                                          │
│                                                                         │
│    - VectorizePass                                                      │
│    - InterleavePass                                                     │
│    - PipelinePass                                                       │
│    - GatherOverlapPass                                                  │
│    - CrossRoundOverlapPass                                              │
└─────────────────────────────────────────────────────────────────────────┘
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

| Priority | Event                  | Purpose                            |
|----------|------------------------|------------------------------------|
| P0       | `SlotUtilizationEvent` | Detect under-utilized cycles       |
| P0       | `PipelineStallEvent`   | Detect dependency-caused stalls    |
| P1       | `MemoryLatencyEvent`   | Track load completion timing       |
| P1       | `GatherStallEvent`     | Specifically track gather ops      |
| P2       | `HashPipelineEvent`    | Track hash stage timing            |
| P2       | `IndexComputeEvent`    | Track index computation timing     |

### warpforge-backend-nvidia Implementation Priority

| Priority | Feature                        | Purpose                            |
|----------|--------------------------------|------------------------------------|
| P0       | `SALT_SLOT_TRACE` level        | Track slot utilization in PTX      |
| P1       | Dependency chain markers       | Identify optimization opportunities|
| P2       | Memory access pattern tracking | Enable gather optimization         |

---

## Session Log: Optimization Attempts

### Session 2: Reconstruction and Further Optimization (January 2026)

**Starting State**: Baseline at 147,734 cycles (fresh clone)

**Final State**: 6,349 cycles (23.3x speedup)

**Instruction Distribution (current version)**:
```
alu:   5,120 ops
valu:  12,304 ops (12304 / 6 = 2,050 cycles minimum)
load:  5,179 ops (5179 / 2 = 2,589 cycles minimum)  <- BOTTLENECK
store: 1,024 ops (1024 / 2 = 512 cycles minimum)
flow:  514 ops (514 / 1 = 514 cycles minimum)
```

**Key Insight**: The kernel is **load-bound** at 2,589 cycles theoretical minimum, not VALU-bound. Current implementation at 6,349 cycles achieves ~41% of theoretical maximum throughput.

### Optimizations Implemented

1. **SIMD Vectorization** (147K → ~18K cycles)
   - Use vload/vstore/valu for VLEN=8 element operations
   - Single vload replaces 8 scalar loads

2. **4-Way Batch Pipelining** (~18K → ~7K cycles)
   - Process vectors A, B, C, D in parallel
   - Interleave operations to maximize slot utilization

3. **Gather/Compute Overlap** (~7K → ~6.5K cycles)
   - Start hash computation for A while gathering B
   - Continue A/B hash while gathering C
   - Continue all hashes while gathering D

4. **Hash/Index Overlap** (~6.5K → ~6.3K cycles)
   - Start A's index computation while C/D still hashing
   - Pipeline: A index overlaps with C hash stage 4-5, D hash stages 2-5

### Per-Iteration Cycle Breakdown

| Phase     | Description                            | Cycles |
|-----------|----------------------------------------|--------|
| 1         | Address computation + vloads           | 5      |
| 2         | Gather address computation             | 3      |
| 3         | Gather A                               | 4      |
| 4         | Gather B + A XOR/hash 0-1              | 4      |
| 5         | Gather C + A hash 2-5, B XOR/hash 0-3  | 4      |
| 6         | Gather D + hash tail                   | 4      |
| 7         | Finish hash + index computation        | ~15    |
| 8         | vselects                               | 4      |
| 9         | Stores                                 | 4      |
| **Total** |                                        | **~47**|

With 16 rounds × 8 vector groups: 47 × 128 = 6,016 cycles + overhead ≈ 6,349 cycles

### Remaining Optimization Opportunities

1. **Cross-Round Gather Overlap** (~300 cycles savings)
   - Start round N+1's address computation during round N's stores
   - Start round N+1's vloads during round N's final vselects

2. **Deeper Hash Pipelining** (~200 cycles savings)
   - Run A/B two stages ahead of C/D instead of one
   - Requires careful dependency tracking

3. **Store/Load Overlap** (~256 cycles savings)
   - Overlap current stores with next iteration's vloads
   - Requires double-buffering of addresses

### Session 1: Earlier Optimization Attempts (Archived)

**Previous best**: 4,283 cycles (now lost to git revert)

**Attempted but failed optimizations**:

1. **vselect/hash overlap**: Attempted to overlap vselects with XOR and hash start
   - **Result**: FAILED - incorrect output
   - **Root cause**: Hash pipelining dependencies incorrectly tracked

2. **Aggressive 12-vselect overlap**: Attempted to overlap all vselects with hash stages
   - **Result**: FAILED - incorrect output
   - **Root cause**: Complex staggered pipeline introduced ordering bugs

### Lessons Learned

1. **Correctness is paramount**: Each optimization must preserve exact output.

2. **Dependency tracking is critical**: When pipelining across multiple vectors at different stages, tracking dependencies becomes complex.

3. **Load-bound vs compute-bound**: The kernel is load-bound (2,589 cycle minimum) not compute-bound (2,050 cycle minimum). Optimization focus should be on memory access patterns.

4. **vselect bottleneck**: 514 flow operations at 1/cycle (514 cycles) is ~20% of total runtime.

### Files Modified

- `/Users/morris/surfworks/original_performance_takehome/perf_takehome.py` - Optimized kernel (6,349 cycles)
- `/Users/morris/surfworks/warpforge/architecture/CLAUDE-OPTIMIZATION.md` - This document

---

## Real GPU Microarchitecture: NVIDIA and AMD

This section maps the VLIW optimization patterns to real GPU hardware, enabling industrial-strength JFR instrumentation.

### NVIDIA Architecture (Ampere/Ada Lovelace/Hopper)

#### Streaming Multiprocessor (SM) Structure

```
┌──────────────────────────────────────────────────────────────────────────┐
│  NVIDIA Streaming Multiprocessor (SM)                                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────┐  ┌───────────────────┐      (4 schedulers/SM)     │
│  │ Warp Scheduler 0  │  │ Warp Scheduler 1  │                            │
│  │ Dispatch Unit     │  │ Dispatch Unit     │                            │
│  └───────────────────┘  └───────────────────┘                            │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Processing Block (4 per SM)                                       │  │
│  │                                                                    │  │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐          │  │
│  │  │ 16 FP32 Cores │  │ 16 FP32 Cores │  │ Tensor Core   │          │  │
│  │  └───────────────┘  └───────────────┘  └───────────────┘          │  │
│  │                                                                    │  │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐          │  │
│  │  │ 16 INT32 Cores│  │ Load/Store    │  │ SFU (4)       │          │  │
│  │  └───────────────┘  └───────────────┘  └───────────────┘          │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Register File: 65,536 x 32-bit registers per SM                   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Shared Memory / L1 Cache: 128 KB (configurable split)             │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

#### Instruction Latencies (SASS)

| Instruction Class    | Latency (cycles) | Notes                       |
|----------------------|------------------|-----------------------------|
| FP32 arithmetic      | 4                | FFMA, FADD, FMUL            |
| INT32 arithmetic     | 4                | IADD3, IMAD                 |
| FP64 arithmetic      | 8                | DFMA (2x FP32)              |
| Shared memory load   | 23               | Ampere, ~19ns at 1.2GHz     |
| Shared memory store  | 19               |                             |
| Global memory        | 200-400+         | Depends on cache hit        |
| L2 cache hit         | ~100-200         | ~80-160ns                   |
| Tensor Core (IMMA)   | 4-8              | Matrix multiply-accumulate  |
| SFU (sin, cos, etc.) | 16               | Special function unit       |

**Source**: [Demystifying the Nvidia Ampere Architecture](https://arxiv.org/pdf/2208.11174), [Dissecting the NVIDIA Hopper Architecture](https://arxiv.org/pdf/2402.13499)

#### Warp Scheduling

- **Warp**: 32 threads executing in lockstep (SIMT)
- **Schedulers per SM**: 4 (can issue to 4 warps per cycle)
- **Issue rate**: 1-2 instructions per warp per cycle
- **Dual issue**: One ALU + one memory operation can issue together
- **Context switch**: Zero overhead (all warp state in registers)
- **Latency hiding**: Scheduler switches warps while waiting for memory

#### Mapping VLIW Concepts to NVIDIA

| VLIW Concept     | NVIDIA Equivalent                  |
|------------------|------------------------------------|
| VALU (6 slots)   | FP32 cores (128 per SM on Ada)     |
| ALU (12 slots)   | INT32 cores (64 per SM)            |
| Load (2 slots)   | Load/Store units (32 per SM)       |
| Flow (1 slot)    | Warp scheduler predication         |
| VLEN=8           | Warp size 32 (4x wider)            |
| Slot utilization | Warp occupancy / stall reasons     |

### AMD Architecture (GCN/RDNA3)

#### Compute Unit (CU) Structure

```
┌──────────────────────────────────────────────────────────────────────────┐
│  AMD Compute Unit (CU) - RDNA3                                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Scheduler + Branch Unit                                           │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────┐  ┌────────────────────────────┐          │
│  │  SIMD32 (32 ALUs)          │  │  SIMD32 (32 ALUs)          │          │
│  │  + 512 VGPRs               │  │  + 512 VGPRs               │          │
│  └────────────────────────────┘  └────────────────────────────┘          │
│                                   (2 SIMDs per CU)                       │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Scalar ALU (SALU) + 128 SGPRs                                     │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Matrix Core Unit (RDNA3 AI Accelerators)                          │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌─────────────────────────────────┬──────────────────────────────────┐  │
│  │  L0 Vector Cache: 32 KB        │  L0 Scalar Cache: 16 KB          │  │
│  └─────────────────────────────────┴──────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Local Data Share (LDS): 64 KB per CU                              │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

#### Instruction Latencies

| Instruction Class | Latency (cycles) | Notes                  |
|-------------------|------------------|------------------------|
| VALU FP32         | 4-5              | v_add_f32, v_fma_f32   |
| VALU INT32        | 4                | v_add_i32              |
| SALU              | 2-4              | Scalar operations      |
| LDS load          | ~25              | Local data share       |
| L0 vector cache   | ~15-20           | Cache hit              |
| L0 scalar cache   | ~15              | Lower latency path     |
| Global memory     | 400-700+         | HBM2/GDDR6             |

**Source**: [RDNA3 ISA Reference Guide](https://docs.amd.com/v/u/en-US/rdna3-shader-instruction-set-architecture-feb-2023_0), [Microbenchmarking AMD's RDNA 3](https://chipsandcheese.com/2023/01/07/microbenchmarking-amds-rdna-3-graphics-architecture/)

#### Wavefront Scheduling

- **Wavefront (GCN)**: 64 threads
- **Wavefront (RDNA)**: 32 or 64 threads (native 32)
- **SIMDs per CU**: 2 (RDNA3), executing 32 threads per cycle
- **Issue rate**: 1 instruction per SIMD per cycle
- **Context switch**: Zero overhead (contexts resident)
- **Dual issue (VOPD)**: RDNA3 can dual-issue certain VALU ops

#### Mapping VLIW Concepts to AMD

| VLIW Concept     | AMD Equivalent                     |
|------------------|------------------------------------|
| VALU (6 slots)   | VALU (32-wide SIMD x 2)            |
| ALU (12 slots)   | SALU (scalar path)                 |
| Load (2 slots)   | Vector memory + scalar memory      |
| Flow (1 slot)    | Scalar branch unit                 |
| VLEN=8           | Wave32 or Wave64                   |
| Slot utilization | Wave occupancy / stall cycles      |

### Optimization Pattern Mapping

#### Pattern 1: SIMD Vectorization

| VLIW           | NVIDIA                  | AMD                      |
|----------------|-------------------------|--------------------------|
| VLEN=8 vectors | 32-wide warps           | 32/64-wide waves         |
| vload/vstore   | coalesced LD.E.128      | buffer_load_dwordx4      |
| valu ops       | FP32 ALU on all lanes   | v_fma_f32 on wave        |

**Key difference**: Real GPUs have larger SIMD widths (32-64 vs 8), so memory coalescing is even more critical.

#### Pattern 2: Batch Pipelining

| VLIW                    | NVIDIA                       | AMD                          |
|-------------------------|------------------------------|------------------------------|
| 4-way interleave        | Multiple warps in flight     | Multiple waves per CU        |
| ILP within vector group | ILP across warp instructions | ILP across wave instructions |
| Hide memory latency     | Warp scheduler switches      | Wavefront scheduler switches |

**Key insight**: Real GPUs hide latency through massive parallelism (thousands of threads), not explicit software pipelining. The scheduler handles latency hiding automatically.

#### Pattern 3: Gather/Compute Overlap

| VLIW                 | NVIDIA                       | AMD                          |
|----------------------|------------------------------|------------------------------|
| Explicit overlap     | Implicit via warp switching  | Implicit via wave switching  |
| 2 loads/cycle limit  | ~32 loads/cycle (coalesced)  | ~32 loads/cycle (coalesced)  |
| Gather = 4 cycles    | Gather = 1 instruction       | Gather = 1 instruction       |

**Key insight**: Scatter/gather in real GPUs is handled by hardware address generation. The optimization focus shifts from explicit scheduling to memory access pattern optimization (coalescing, bank conflicts).

#### Pattern 4: Hash Pipelining

| VLIW            | NVIDIA                    | AMD                              |
|-----------------|---------------------------|----------------------------------|
| op1/op3 parallel| ILP detected by hardware  | ILP via dual-issue (VOPD)        |
| op2 dependent   | Dependency scoreboard     | Dependency handling in scheduler |
| 6 VALU slots    | 128 FP32 cores            | 64 ALUs per CU                   |

**Key insight**: Modern GPUs have enough compute throughput that hash-like patterns are rarely compute-bound. The bottleneck is usually memory bandwidth.

### Industrial JFR Events for Real GPUs

Based on the microarchitecture analysis, here are revised JFR events applicable to real hardware:

#### NVIDIA Events

```java
@Label("NVIDIA Warp Stall")
@Category({"WarpForge", "NVIDIA", "Performance"})
public class NvidiaWarpStallEvent extends Event {
    @Label("SM ID")
    public int smId;

    @Label("Warp ID")
    public int warpId;

    @Label("Stall Reason")
    public String stallReason;  // "memory_dependency", "execution_dependency", "barrier", "not_selected"

    @Label("Stall Cycles")
    public long stallCycles;

    @Label("Active Warps")
    public int activeWarps;

    @Label("Achieved Occupancy")
    public float achievedOccupancy;
}

@Label("NVIDIA Memory Access Pattern")
@Category({"WarpForge", "NVIDIA", "Memory"})
public class NvidiaMemoryAccessEvent extends Event {
    @Label("Access Type")
    public String accessType;  // "coalesced", "strided", "scattered"

    @Label("Sectors Requested")
    public int sectorsRequested;

    @Label("Sectors Actual")
    public int sectorsActual;

    @Label("L1 Hit Rate")
    public float l1HitRate;

    @Label("L2 Hit Rate")
    public float l2HitRate;

    @Label("Memory Throughput GB/s")
    public float throughputGBps;
}
```

#### AMD Events

```java
@Label("AMD Wave Occupancy")
@Category({"WarpForge", "AMD", "Performance"})
public class AmdWaveOccupancyEvent extends Event {
    @Label("CU ID")
    public int cuId;

    @Label("Active Waves")
    public int activeWaves;

    @Label("Max Waves")
    public int maxWaves;

    @Label("VGPR Usage")
    public int vgprUsage;

    @Label("LDS Usage KB")
    public float ldsUsageKb;

    @Label("Occupancy Limiter")
    public String limiter;  // "vgpr", "sgpr", "lds", "waves_per_simd"
}

@Label("AMD VALU Utilization")
@Category({"WarpForge", "AMD", "Performance"})
public class AmdValuUtilizationEvent extends Event {
    @Label("CU ID")
    public int cuId;

    @Label("VALU Busy Cycles")
    public long valuBusyCycles;

    @Label("SALU Busy Cycles")
    public long saluBusyCycles;

    @Label("LDS Busy Cycles")
    public long ldsBusyCycles;

    @Label("Vector Memory Busy Cycles")
    public long vmemBusyCycles;
}
```

### Key Differences: VLIW Emulator vs Real GPUs

| Aspect              | VLIW Emulator              | Real GPUs                        |
|---------------------|----------------------------|----------------------------------|
| Parallelism         | Explicit slot packing      | Implicit massive parallelism     |
| Latency hiding      | Software pipelining        | Hardware warp/wave scheduling    |
| Memory              | 2 loads/cycle hard limit   | ~32+ loads/cycle (coalesced)     |
| Register allocation | Manual scratch management  | Hardware register file           |
| Control flow        | Explicit select/branch     | Predicated execution, divergence |
| Optimization focus  | Instruction scheduling     | Memory access patterns           |

### Implications for Babylon/WarpForge

1. **Code generation target**: Babylon should generate PTX/GCN assembly that maximizes:
   - Memory coalescing (all threads access consecutive addresses)
   - Register pressure management (to maintain occupancy)
   - Avoiding bank conflicts in shared memory

2. **JFR instrumentation**: Focus on:
   - Warp/wave stall reasons (not slot utilization)
   - Memory access patterns (coalescing efficiency)
   - Occupancy metrics (registers, shared memory)
   - Cache hit rates

3. **MCP optimization hints**: Claude Code should suggest:
   - Memory layout transformations for coalescing
   - Loop tiling for cache locality
   - Occupancy-aware kernel parameters

**Sources**:
- [Demystifying the Nvidia Ampere Architecture](https://arxiv.org/pdf/2208.11174)
- [Dissecting the NVIDIA Hopper Architecture](https://arxiv.org/pdf/2402.13499)
- [RDNA3 ISA Reference Guide](https://docs.amd.com/v/u/en-US/rdna3-shader-instruction-set-architecture-feb-2023_0)
- [Microbenchmarking AMD's RDNA 3](https://chipsandcheese.com/2023/01/07/microbenchmarking-amds-rdna-3-graphics-architecture/)
- [AMD GPU Architecture Programming Documentation](https://gpuopen.com/amd-gpu-architecture-programming-documentation/)
- [ROCm Compute Profiler Documentation](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/compute-unit.html)

---

## Generalized Babylon-MCP Optimization Framework

This section describes the architecture for a production-ready optimization framework that bridges Babylon Code Reflection, Claude Code (via MCP), and real GPU hardware.

### Framework Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      Babylon-MCP Optimization Framework                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Layer 1: Babylon Code Reflection (jdk.incubator.code)                 │ │
│  │                                                                        │ │
│  │  PyTorch Model  ──▶  FX Graph  ──▶  StableHLO  ──▶  Babylon IR        │ │
│  │                                                                        │ │
│  │  Key classes:                                                          │ │
│  │    - Op.java      Operation representation                             │ │
│  │    - Body.java    Control flow blocks with dataflow analysis           │ │
│  │    - SSA.java     Static single assignment transformation              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Layer 2: WarpForge Optimization Passes                                │ │
│  │                                                                        │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐           │ │
│  │  │ MemoryLayout   │  │ LoopTiling     │  │ Occupancy      │           │ │
│  │  │ Pass           │  │ Pass           │  │ Optimizer      │           │ │
│  │  └────────────────┘  └────────────────┘  └────────────────┘           │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐           │ │
│  │  │ Coalescing     │  │ Bank Conflict  │  │ Register       │           │ │
│  │  │ Analyzer       │  │ Resolver       │  │ Allocator      │           │ │
│  │  └────────────────┘  └────────────────┘  └────────────────┘           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Layer 3: Backend Code Generation (HAT-style)                          │ │
│  │                                                                        │ │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │ │
│  │  │  NVIDIA Backend  │  │  AMD Backend     │  │  CPU Backend     │     │ │
│  │  │  (PTX/CUDA)      │  │  (GCN/HIP)       │  │  (Reference)     │     │ │
│  │  │                  │  │                  │  │                  │     │ │
│  │  │  Salt-based      │  │  Salt-based      │  │  No salt needed  │     │ │
│  │  │  instrumentation │  │  instrumentation │  │  (full trace)    │     │ │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Layer 4: JFR Instrumentation & Telemetry                              │ │
│  │                                                                        │ │
│  │  GPU Events:                                                           │ │
│  │    - WarpStallEvent / WaveOccupancyEvent                               │ │
│  │    - MemoryAccessPatternEvent                                          │ │
│  │    - CacheEfficiencyEvent                                              │ │
│  │    - KernelLaunchEvent                                                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Layer 5: Claude Code MCP Integration                                  │ │
│  │                                                                        │ │
│  │  MCP Server exposes:                                                   │ │
│  │    - /profile/kernel/{id}       Get profiling data                     │ │
│  │    - /analyze/memory-pattern    Analyze memory access patterns         │ │
│  │    - /optimize/suggest          Get optimization suggestions           │ │
│  │    - /transform/apply           Apply transformation                   │ │
│  │                                                                        │ │
│  │  Claude Code can:                                                      │ │
│  │    1. Read JFR telemetry                                               │ │
│  │    2. Analyze bottlenecks                                              │ │
│  │    3. Suggest code transformations                                     │ │
│  │    4. Apply optimizations via MCP tools                                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### MCP Tool Definitions

```json
{
  "name": "warpforge-optimizer",
  "version": "1.0.0",
  "tools": [
    {
      "name": "profile_kernel",
      "description": "Get profiling data for a kernel execution",
      "inputSchema": {
        "type": "object",
        "properties": {
          "kernelId": {"type": "string"},
          "metrics": {
            "type": "array",
            "items": {"enum": ["warp_stalls", "memory_pattern", "cache_hits", "occupancy"]}
          }
        },
        "required": ["kernelId"]
      }
    },
    {
      "name": "analyze_bottleneck",
      "description": "Analyze performance bottlenecks in a kernel",
      "inputSchema": {
        "type": "object",
        "properties": {
          "kernelId": {"type": "string"},
          "targetBackend": {"enum": ["nvidia", "amd", "cpu"]}
        }
      }
    },
    {
      "name": "suggest_optimization",
      "description": "Get optimization suggestions based on profiling data",
      "inputSchema": {
        "type": "object",
        "properties": {
          "kernelId": {"type": "string"},
          "category": {
            "enum": ["memory_coalescing", "occupancy", "bank_conflicts", "register_pressure"]
          }
        }
      }
    },
    {
      "name": "apply_transformation",
      "description": "Apply a code transformation to optimize a kernel",
      "inputSchema": {
        "type": "object",
        "properties": {
          "kernelId": {"type": "string"},
          "transformation": {
            "enum": ["tile_loops", "transpose_memory", "unroll", "vectorize", "fuse_kernels"]
          },
          "parameters": {"type": "object"}
        }
      }
    }
  ]
}
```

### Optimization Workflow

```
1. USER: "My matmul kernel is slow"

2. CLAUDE CODE:
   - Calls profile_kernel to get JFR events
   - Calls analyze_bottleneck to identify issues

3. MCP SERVER returns:
   {
     "bottleneck": "memory_coalescing",
     "details": "75% non-coalesced global memory accesses",
     "affected_regions": ["line 42-58: matrix A load"],
     "warp_stalls": {"memory_dependency": 45%, "execution_dependency": 10%}
   }

4. CLAUDE CODE:
   - Calls suggest_optimization(category="memory_coalescing")

5. MCP SERVER returns:
   {
     "suggestions": [
       {
         "transformation": "transpose_memory",
         "expected_improvement": "3.2x",
         "code_changes": "Transpose matrix A in shared memory before compute"
       },
       {
         "transformation": "tile_loops",
         "expected_improvement": "2.8x",
         "parameters": {"tile_size": 32}
       }
     ]
   }

6. CLAUDE CODE:
   - Presents options to user
   - User selects "transpose_memory"
   - Calls apply_transformation

7. MCP SERVER:
   - Applies transformation via Babylon IR
   - Regenerates optimized PTX/GCN
   - Returns new kernel ID

8. CLAUDE CODE:
   - Re-profiles to verify improvement
   - Reports results to user
```

### Hardware-Specific Optimization Passes

#### NVIDIA Optimizations

| Pass             | Input      | Output                           | Target Metric             |
|------------------|------------|----------------------------------|---------------------------|
| CoalescingPass   | Babylon IR | Babylon IR with AoS→SoA xforms   | Memory efficiency         |
| SharedMemTiling  | Babylon IR | Babylon IR with tiled loops      | L1 cache utilization      |
| WarpShufflePass  | Babylon IR | Babylon IR using warp shuffles   | Reduce shared mem pressure|
| TensorCoreMapping| Matrix ops | WMMA intrinsics                  | Tensor core utilization   |

#### AMD Optimizations

| Pass              | Input      | Output                           | Target Metric             |
|-------------------|------------|----------------------------------|---------------------------|
| LDSOptimizer      | Babylon IR | Babylon IR with LDS usage        | LDS bandwidth             |
| WavefrontOccupancy| Babylon IR | Register-pressure-aware IR       | Occupancy                 |
| DualIssueVOPD     | Babylon IR | Paired VALU ops                  | VALU throughput           |
| ScalarPromotion   | Babylon IR | SALU-promoted uniform values     | VGPR savings              |

### Integration with Mark 1 Holmes Lab

The framework is designed to run on the Mark 1 Holmes Lab hardware:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                             Mark 1 Holmes Lab                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                        ┌──────────────────────┐                              │
│                        │  NUC (Orchestrator)  │                              │
│                        │                      │                              │
│                        │  - Claude Code       │                              │
│                        │  - MCP Server        │                              │
│                        │  - JFR Aggregation   │                              │
│                        │  - Test Coordinator  │                              │
│                        └──────────┬───────────┘                              │
│                                   │                                          │
│                     ┌─────────────┴─────────────┐                            │
│                     │                           │                            │
│                     ▼                           ▼                            │
│        ┌────────────────────────┐  ┌────────────────────────┐               │
│        │  NVIDIA Box           │  │  AMD Box               │               │
│        │  (RTX GPU)            │  │  (RDNA3 GPU)           │               │
│        │                       │  │                        │               │
│        │  - warpforge-backend- │  │  - warpforge-backend-  │               │
│        │    nvidia             │  │    amd                 │               │
│        │  - CUDA runtime       │  │  - ROCm runtime        │               │
│        │  - nvdisasm           │  │  - rocprofiler         │               │
│        └────────────────────────┘  └────────────────────────┘               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Next Steps

1. **Implement JFR event generation** in warpforge-backend-nvidia and warpforge-backend-amd
2. **Build MCP server** exposing optimization tools
3. **Create optimization passes** based on Babylon IR transformations
4. **Integrate with CI/CD** for automated performance regression testing
5. **Validate on Mark 1** with real workloads

---

## Summary

This document captured:

1. **VLIW Optimization Journey**: From 147,734 to 6,349 cycles (23x speedup)
2. **Optimization Patterns**: SIMD vectorization, batch pipelining, gather/compute overlap, hash pipelining
3. **Babylon Integration Points**: Op.java, Body.java, SSA transformation, HAT backend model
4. **Real GPU Microarchitecture**: NVIDIA SM/warp scheduling, AMD CU/wavefront scheduling
5. **Industrial JFR Events**: Warp stalls, memory patterns, occupancy metrics
6. **Babylon-MCP Framework**: Architecture for Claude Code-assisted GPU optimization

The key insight is that while the VLIW emulator taught valuable lessons about instruction scheduling and pipelining, real GPU optimization focuses on:

- **Memory access patterns** (coalescing, bank conflicts)
- **Occupancy management** (registers, shared memory)
- **Latency hiding through parallelism** (not explicit software pipelining)

The Babylon-MCP framework bridges these concepts, enabling Claude Code to suggest real-world GPU optimizations based on JFR telemetry.
