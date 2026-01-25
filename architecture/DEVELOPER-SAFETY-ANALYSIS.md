# WarpForge Developer Safety Analysis

This document analyzes categories of runtime errors and surprises that PyTorch developers commonly encounter, and proposes how WarpForge can use Java's stronger type system and compile-time checking to eliminate them entirely.

## Philosophy: Shift Errors Left

The goal is to **shift errors from runtime to compile time**. Every error caught by the compiler is:
- Caught before any compute resources are consumed
- Caught before any training time is wasted
- Caught with precise location information
- Impossible to ignore or work around

PyTorch's dynamic nature means most errors surface at runtime, often after significant compute has been invested. WarpForge should aim for the opposite: if it compiles, it runs correctly.

---

## Category 1: Shape Errors

**Current Coverage**: Partial (runtime validation, new test suite)

### PyTorch Pain Points

```python
# Dies after hours of training when batch size changes
x = torch.randn(32, 784)
w = torch.randn(784, 256)
# Later, someone changes batch size...
x = torch.randn(64, 784)
# Still works, but then:
b = torch.randn(32, 256)  # Oops, hardcoded old batch size
result = x @ w + b  # RuntimeError at 3am
```

### WarpForge Opportunity: Phantom Type Shape Parameters

```java
// Compile-time shape tracking
public interface Tensor<S extends Shape> {
    S shape();
}

// Dimension markers
interface Batch extends Dim {}
interface Features extends Dim {}
interface Hidden extends Dim {}

// Operations enforce shape compatibility at compile time
public <B extends Batch, F extends Features, H extends Hidden>
Tensor<Shape2<B, H>> linear(
    Tensor<Shape2<B, F>> input,
    Tensor<Shape2<F, H>> weight,
    Tensor<Shape1<H>> bias  // Bias shape derived from weight
) {
    return backend.linear(input, weight, bias);
}

// Usage - compiler catches mismatches:
Tensor<Shape2<Batch32, Features784>> x = loadInput();
Tensor<Shape2<Features784, Hidden256>> w = loadWeights();
Tensor<Shape1<Hidden128>> wrongBias = ...;  // COMPILE ERROR: Hidden128 != Hidden256
```

### Additional Shape Checks to Implement

| Check | PyTorch Behavior | WarpForge Target |
|-------|------------------|------------------|
| Matmul inner dimensions | Runtime error | Compile-time via generics |
| Broadcast compatibility | Runtime error | Compile-time via type constraints |
| Reduction dimension bounds | Runtime error | Compile-time via bounded types |
| Reshape element count | Runtime error | Compile-time via dependent types |
| Concat dimension matching | Runtime error | Compile-time via shape algebra |
| Conv kernel/input compatibility | Runtime error | Compile-time via dimension types |

---

## Category 2: DType Errors and Precision Surprises

### PyTorch Pain Points

```python
# Silent precision loss
x = torch.randn(100, dtype=torch.float64)
y = torch.randn(100, dtype=torch.float32)
z = x + y  # Silently converts to float32, losing precision

# Mixed precision training surprises
model = model.half()  # Convert to fp16
# Later...
loss = criterion(output, target)  # target might still be fp32
loss.backward()  # Gradient explosion or NaN

# Integer overflow in indexing
indices = torch.tensor([1, 2, 3], dtype=torch.int8)
indices = indices + 200  # Overflow! No warning.
```

### WarpForge Opportunity: Explicit DType in Type System

```java
// DType is part of the tensor type
public interface Tensor<S extends Shape, D extends DType> {
    S shape();
    D dtype();
}

// Operations specify dtype behavior explicitly
public <S extends Shape>
Tensor<S, F32> add(Tensor<S, F32> a, Tensor<S, F32> b) {
    return backend.add(a, b);
}

// Mixed precision requires explicit conversion
public <S extends Shape>
Tensor<S, F32> addMixed(Tensor<S, F64> a, Tensor<S, F32> b) {
    // COMPILE ERROR: No implicit conversion
    // Must explicitly choose:
    return add(a.toF32(), b);  // Explicit downcast
    // or:
    return add(a, b.toF64()).toF32();  // Compute in higher precision
}

// Precision-preserving operations
@PreservesPrecision
public <S extends Shape, D extends FloatingDType>
Tensor<S, D> relu(Tensor<S, D> x) {
    return backend.relu(x);
}

// Precision-reducing operations require acknowledgment
@ReducesPrecision(from = F32.class, to = F16.class)
public <S extends Shape>
Tensor<S, F16> toHalf(Tensor<S, F32> x) {
    return backend.cast(x, F16.class);
}
```

### DType Checks to Implement

| Check | PyTorch Behavior | WarpForge Target |
|-------|------------------|------------------|
| Mixed dtype operations | Silent promotion | Compile error, require explicit |
| Precision loss in cast | Silent | Require @AcknowledgePrecisionLoss |
| Integer overflow | Silent wraparound | Runtime check + optional compile warning |
| NaN in integer cast | Undefined behavior | Exception with context |
| Loss function dtype match | Runtime error | Compile-time via generics |

---

## Category 3: Device Placement Errors

### PyTorch Pain Points

```python
# The classic device mismatch
model = Model().cuda()
x = torch.randn(32, 784)  # On CPU by default
output = model(x)  # RuntimeError: tensors on different devices

# Forgotten .to(device) in complex pipelines
class MyModel(nn.Module):
    def __init__(self):
        self.buffer = torch.zeros(100)  # Not a parameter!

    def forward(self, x):
        return x + self.buffer  # buffer still on CPU after .cuda()

# Device leaks in data loading
def collate_fn(batch):
    # Accidentally create tensors on GPU in DataLoader worker
    return torch.stack(batch).cuda()  # CUDA error in subprocess
```

### WarpForge Opportunity: Device as Type Parameter

```java
// Device is part of the tensor type
public interface Tensor<S extends Shape, D extends DType, Dev extends Device> {
    S shape();
    D dtype();
    Dev device();
}

// Operations require same device
public <S extends Shape, D extends DType, Dev extends Device>
Tensor<S, D, Dev> add(Tensor<S, D, Dev> a, Tensor<S, D, Dev> b) {
    return backend.add(a, b);
}

// Cross-device operations require explicit transfer
public <S extends Shape, D extends DType>
Tensor<S, D, Cuda0> toCuda(Tensor<S, D, Cpu> cpuTensor) {
    return backend.transfer(cpuTensor, Cuda0.INSTANCE);
}

// Model definition with device-aware buffers
public class MyModel implements Module<Cuda0> {
    // Buffers are device-typed
    private final Tensor<Shape1<_100>, F32, Cuda0> buffer;

    public MyModel(Cuda0 device) {
        this.buffer = Tensor.zeros(shape(100), F32, device);
    }

    public <B extends Batch>
    Tensor<Shape2<B, _100>, F32, Cuda0> forward(
        Tensor<Shape2<B, _100>, F32, Cuda0> x  // Must be on same device
    ) {
        return add(x, buffer.broadcast());
    }
}

// Device context for scoped operations
try (var cuda = DeviceContext.cuda(0)) {
    // All tensors created here are on cuda:0
    var x = Tensor.randn(shape(32, 784));  // Automatically on cuda:0
    var y = model.forward(x);
}
```

### Device Checks to Implement

| Check | PyTorch Behavior | WarpForge Target |
|-------|------------------|------------------|
| Cross-device operation | Runtime error | Compile error |
| Model/input device mismatch | Runtime error | Compile error via generics |
| Buffer device after .to() | Silent bug | Automatic via device-typed fields |
| DataLoader device context | Runtime CUDA error | Compile error (no CUDA in workers) |
| Multi-GPU tensor placement | Runtime error | Compile-time device routing |

---

## Category 4: Gradient and Autograd Errors

### PyTorch Pain Points

```python
# Detached tensor surprise
x = torch.randn(10, requires_grad=True)
y = x.detach()  # No warning
z = y * 2
z.sum().backward()  # No gradient flows to x!

# In-place operation breaks gradient
x = torch.randn(10, requires_grad=True)
y = x * 2
x += 1  # In-place modification
y.sum().backward()  # RuntimeError: in-place operation

# Gradient accumulation surprise
optimizer.zero_grad()  # Easy to forget
loss1.backward()
loss2.backward()  # Gradients accumulate!
optimizer.step()  # Training with 2x gradients

# Leaf tensor modification
x = torch.randn(10, requires_grad=True)
x.data += 1  # Modifies leaf, gradients now wrong
```

### WarpForge Opportunity: Explicit Gradient Tracking in Types

```java
// Gradient requirement is part of the type
public interface Tensor<S extends Shape, D extends DType, G extends GradMode> {}

sealed interface GradMode permits RequiresGrad, NoGrad, Detached {}

// Operations that require gradients
public <S extends Shape>
Tensor<S, F32, RequiresGrad> parameterTensor(float[] data, S shape) {
    return Tensor.of(data, shape, F32, RequiresGrad.INSTANCE);
}

// Detach is explicit and changes the type
public <S extends Shape, D extends DType>
Tensor<S, D, Detached> detach(Tensor<S, D, RequiresGrad> t) {
    // Return type makes it clear: no gradients will flow
    return backend.detach(t);
}

// Backward only works on gradient-enabled tensors
public void backward(Tensor<?, ?, RequiresGrad> loss) {
    // Compile error if you try to backward on Detached or NoGrad
    autograd.backward(loss);
}

// Training step with explicit gradient lifecycle
public void trainStep(Model model, Tensor<?, ?, NoGrad> input, Tensor<?, ?, NoGrad> target) {
    try (var gradScope = GradientScope.open()) {
        // Gradients automatically zeroed at scope start
        var output = model.forward(input.withGrad());
        var loss = criterion.compute(output, target);

        gradScope.backward(loss);
        // Gradients automatically applied at scope end
    }
    // Impossible to forget zero_grad or accumulate accidentally
}

// In-place operations on gradient tensors are compile errors
public <S extends Shape>
void addInPlace(Tensor<S, F32, NoGrad> target, Tensor<S, F32, ?> source) {
    // Only allowed on NoGrad tensors
    backend.addInPlace(target, source);
}
```

### Gradient Checks to Implement

| Check | PyTorch Behavior | WarpForge Target |
|-------|------------------|------------------|
| Backward on detached tensor | Silent no-op | Compile error |
| In-place on grad tensor | Runtime error | Compile error |
| Gradient accumulation | Silent accumulation | Scoped auto-zero |
| Leaf tensor modification | Silent corruption | Immutable by default |
| requires_grad propagation | Implicit rules | Explicit type propagation |
| Double backward without retain | Runtime error | Type-level graph retention |

---

## Category 5: Memory Management Errors

### PyTorch Pain Points

```python
# OOM discovered at runtime after hours
for epoch in range(1000):
    for batch in dataloader:
        output = model(batch)
        all_outputs.append(output)  # Memory leak!
        # OOM after epoch 50

# Gradient memory not freed
losses = []
for batch in dataloader:
    loss = compute_loss(batch)
    losses.append(loss)  # Holds entire computation graph!
loss.backward()  # OOM

# CUDA memory fragmentation
# No way to predict or prevent

# Buffer reuse errors
buffer = torch.empty(1000)
result1 = some_op(input, out=buffer)
result2 = other_op(input, out=buffer)  # Overwrites result1!
```

### WarpForge Opportunity: Resource-Safe Tensor Management

```java
// Tensors are AutoCloseable with explicit ownership
public interface Tensor<S, D, Dev> extends AutoCloseable {
    // Ownership transfer is explicit
    Tensor<S, D, Dev> move();  // Transfers ownership, invalidates source
    Tensor<S, D, Dev> copy();  // Creates independent copy
    Tensor<S, D, Dev> view();  // Shared view, does not own memory
}

// Try-with-resources for automatic cleanup
try (var input = loadBatch();
     var output = model.forward(input)) {
    // Process output
    results.add(output.copy());  // Explicit copy for retention
}
// input and output automatically freed

// Scoped memory arenas for batch processing
try (var arena = MemoryArena.confined()) {
    for (var batch : dataloader) {
        try (var batchArena = arena.child()) {
            // All tensors allocated in batchArena
            var output = model.forward(batch);
            var loss = criterion.compute(output, target);
            loss.backward();
        }
        // Batch memory freed, only parameters retained
    }
}

// Memory budget enforcement
var budget = MemoryBudget.of(8, GB);
try (var ctx = ExecutionContext.withBudget(budget)) {
    // Operations that would exceed budget throw BEFORE allocating
    var huge = Tensor.zeros(shape(1_000_000, 1_000_000));  // Throws immediately
}

// Buffer pools with exclusive access
try (var buffer = bufferPool.acquire(shape(1000), F32)) {
    var result = someOp(input, buffer);
    // buffer is exclusively owned here
}
// buffer returned to pool

// Gradient memory management
try (var gradCtx = GradientContext.open()) {
    var loss = computeLoss(batch);
    gradCtx.backward(loss);
    // Gradient graph freed immediately after backward
}
```

### Memory Checks to Implement

| Check | PyTorch Behavior | WarpForge Target |
|-------|------------------|------------------|
| Tensor memory leak | Silent until OOM | Compile warning for unfreed tensors |
| Gradient graph retention | Silent accumulation | Scoped auto-release |
| Buffer aliasing | Silent data corruption | Ownership tracking |
| OOM during allocation | Runtime crash | Budget pre-check |
| Memory fragmentation | Gradual degradation | Arena-based allocation |
| Cross-scope tensor escape | Silent reference | Borrow checker (future) |

---

## Category 6: Training State Errors

### PyTorch Pain Points

```python
# Forgot to set eval mode
model.train()  # Set during training
# ... later, during inference ...
output = model(test_input)  # Dropout still active!

# BatchNorm running stats not updated
model.eval()
for batch in train_loader:
    output = model(batch)  # Running stats frozen!
    loss.backward()
    optimizer.step()

# Frozen layer still computing gradients
for param in model.encoder.parameters():
    param.requires_grad = False
# ... training loop ...
# Gradients still computed, just not used - wasted compute

# State dict key mismatch
state = torch.load('checkpoint.pt')
model.load_state_dict(state)  # RuntimeError: missing/unexpected keys
```

### WarpForge Opportunity: Mode-Aware Types

```java
// Training mode is part of the model type
public interface Model<M extends Mode> {
    <I, O> O forward(I input);
}

sealed interface Mode permits Training, Inference {}

// Mode-specific behavior
public class BatchNorm<M extends Mode> implements Layer<M> {
    private final RunningStats stats;

    @Override
    public Tensor forward(Tensor input, M mode) {
        return switch (mode) {
            case Training t -> {
                stats.update(input);  // Only in training
                yield normalize(input, stats.running());
            }
            case Inference i -> normalize(input, stats.frozen());
        };
    }
}

// Dropout that's a no-op in inference (compile-time eliminated)
public class Dropout<M extends Mode> implements Layer<M> {
    @Override
    public Tensor forward(Tensor input, M mode) {
        return switch (mode) {
            case Training t -> dropoutImpl(input, rate);
            case Inference i -> input;  // Compiler can optimize this away
        };
    }
}

// Model usage with explicit mode
Model<Training> trainModel = new TransformerModel<>();
Model<Inference> evalModel = trainModel.toInference();

// These are different types - can't accidentally use wrong mode
var trainOutput = trainModel.forward(batch);   // Dropout active
var evalOutput = evalModel.forward(testBatch); // Dropout disabled

// Frozen layers don't compute gradients
public class FrozenEncoder implements Layer<NoGrad> {
    // Forward returns NoGrad tensors - no gradient computation
    public Tensor<?, ?, NoGrad> forward(Tensor<?, ?, ?> input) {
        return encoder.forward(input.detach());
    }
}

// State dict with schema validation
@StateSchema(version = 2)
public record TransformerState(
    @Key("encoder.weight") Tensor<Shape2<_512, _512>, F32> encoderWeight,
    @Key("encoder.bias") Tensor<Shape1<_512>, F32> encoderBias,
    // ... all keys explicitly declared
) {}

// Load with compile-time schema checking
TransformerState state = StateLoader.load("checkpoint.pt", TransformerState.class);
// Missing or extra keys are compile errors if schema doesn't match
```

### Training State Checks to Implement

| Check | PyTorch Behavior | WarpForge Target |
|-------|------------------|------------------|
| Wrong mode (train/eval) | Silent wrong behavior | Compile error via mode types |
| BatchNorm stats in eval | Silent stale stats | Mode-aware update logic |
| Frozen layer gradient waste | Silent compute waste | NoGrad type skips backward |
| State dict key mismatch | Runtime error | Compile-time schema validation |
| Optimizer state mismatch | Runtime error | Type-safe optimizer state |
| Learning rate schedule errors | Silent wrong LR | Validated schedule types |

---

## Category 7: Numerical Stability Errors

### PyTorch Pain Points

```python
# NaN propagation
x = torch.tensor([0.0])
y = torch.log(x)  # -inf
z = y * 0  # NaN
# NaN silently propagates through entire network

# Softmax numerical instability
logits = torch.tensor([1000.0, 1000.0, 1000.0])
probs = torch.softmax(logits, dim=0)  # [nan, nan, nan]

# Gradient explosion/vanishing
# No warning until loss becomes NaN

# Float comparison
x = torch.tensor(0.1 + 0.2)
y = torch.tensor(0.3)
x == y  # False! Due to floating point representation
```

### WarpForge Opportunity: Numerical Safety Annotations

```java
// Operations annotated with numerical properties
@NumericallyStable
public <S extends Shape>
Tensor<S, F32> softmax(Tensor<S, F32> logits, int dim) {
    // Implementation uses log-sum-exp trick internally
    var maxLogit = logits.max(dim, keepdim=true);
    var shifted = logits.sub(maxLogit);
    var expShifted = shifted.exp();
    return expShifted.div(expShifted.sum(dim, keepdim=true));
}

// NaN-checking mode
try (var ctx = NumericalContext.checked()) {
    var output = model.forward(input);
    // Every operation checks for NaN/Inf
    // Throws immediately with full stack trace on first NaN
}

// Gradient clipping built into optimizer
var optimizer = Adam.builder()
    .learningRate(1e-3)
    .gradientClipping(GradientClip.byNorm(1.0))  // Built-in, not afterthought
    .nanGuard(true)  // Refuse to update if gradients contain NaN
    .build();

// Approximate equality for tensors
public boolean approxEqual(Tensor a, Tensor b, Tolerance tol) {
    // Default tolerance based on dtype
    return TensorCompare.allClose(a, b, tol);
}

// Loss functions with numerical stability guarantees
@NumericallyStable
@RequiresPositive("predictions")
public Tensor<Scalar, F32> crossEntropy(
    Tensor<?, F32> predictions,  // Must be probabilities in (0, 1)
    Tensor<?, I64> targets
) {
    // Uses log-softmax internally for stability
}

// Overflow-safe operations
@OverflowChecked
public Tensor<S, I32> addChecked(Tensor<S, I32> a, Tensor<S, I32> b) {
    // Throws on overflow instead of wrapping
}
```

### Numerical Checks to Implement

| Check | PyTorch Behavior | WarpForge Target |
|-------|------------------|------------------|
| NaN propagation | Silent corruption | Optional checked mode, immediate throw |
| Softmax overflow | NaN output | Numerically stable implementation |
| Log of zero | -inf, then NaN | Domain check or safe log(x + eps) |
| Gradient explosion | Silent until NaN | Gradient monitoring hooks |
| Float equality | Wrong results | approxEqual with dtype-aware tolerance |
| Integer overflow | Silent wraparound | Checked arithmetic option |
| Underflow to zero | Silent precision loss | Subnormal detection option |

---

## Category 8: Data Pipeline Errors

### PyTorch Pain Points

```python
# Normalization mismatch
# Training: normalized with dataset stats
# Inference: forgot to normalize, or used wrong stats

# Data type mismatch
dataset = ImageDataset()  # Returns uint8
model = Model()  # Expects float32
output = model(dataset[0])  # RuntimeError or silent wrong results

# Incorrect augmentation during eval
transform = transforms.Compose([
    transforms.RandomCrop(224),  # Random during eval!
    transforms.ToTensor(),
])

# Collate function shape mismatch
def collate(batch):
    # Images might have different sizes
    return torch.stack(batch)  # RuntimeError

# Label encoding mismatch
# Train labels: 0-9
# Test labels: 1-10 (off by one)
```

### WarpForge Opportunity: Type-Safe Data Pipelines

```java
// Dataset with explicit schema
public interface Dataset<T extends DataSchema> {
    T get(int index);
    int size();
}

// Schema defines exact types
@DataSchema
public record ImageClassificationSample(
    @Shape({3, 224, 224}) @DType(F32) @Normalized(ImageNet.STATS) Tensor image,
    @Range(0, 999) int label
) {}

// Transforms are type-safe
public interface Transform<In extends DataSchema, Out extends DataSchema> {
    Out apply(In input);
}

// Normalization is explicit in types
public class ImageNetNormalize implements Transform<RawImage, NormalizedImage> {
    @Override
    public NormalizedImage apply(RawImage raw) {
        // Type system ensures normalized output
        return new NormalizedImage(normalize(raw.tensor(), ImageNet.STATS));
    }
}

// Mode-aware transforms
public sealed interface AugmentationMode permits TrainAugment, EvalAugment {}

public class RandomCrop implements Transform<Image, Image> {
    @Override
    public Image apply(Image input, AugmentationMode mode) {
        return switch (mode) {
            case TrainAugment t -> randomCrop(input);
            case EvalAugment e -> centerCrop(input);  // Deterministic in eval
        };
    }
}

// DataLoader with schema validation
DataLoader<ImageClassificationSample> loader = DataLoader.builder()
    .dataset(dataset)
    .batchSize(32)
    .collate(Collate.stack())  // Type-safe collation
    .validate(true)  // Runtime validation of schema
    .build();

// Batch type includes batch dimension
public record Batch<T extends DataSchema, B extends BatchSize>(
    Tensor<Shape3<B, C, H, W>, F32> images,
    Tensor<Shape1<B>, I64> labels
) {}
```

### Data Pipeline Checks to Implement

| Check | PyTorch Behavior | WarpForge Target |
|-------|------------------|------------------|
| Normalization mismatch | Silent wrong results | Type-level normalization tracking |
| DType mismatch | Runtime error or silent | Compile error via generics |
| Augment mode mismatch | Silent wrong behavior | Mode-aware transform types |
| Variable size collation | Runtime error | Compile error or padding strategy |
| Label range mismatch | Silent wrong training | @Range annotation validation |
| Missing preprocessing | Silent wrong results | Schema completeness checking |

---

## Category 9: Distributed Training Errors

### PyTorch Pain Points

```python
# Gradient synchronization forgotten
loss.backward()
# Forgot all_reduce!
optimizer.step()  # Each GPU has different gradients

# Batch size not adjusted for world size
dataloader = DataLoader(dataset, batch_size=32)  # Same on all GPUs
# Effective batch size is 32 * world_size

# Non-deterministic operations across ranks
x = torch.randn(100)  # Different on each rank!
# Then used in computation

# Deadlock from mismatched collective
if rank == 0:
    dist.all_reduce(tensor)
else:
    pass  # Rank 1+ hangs forever

# State dict saved on rank 0 only
if rank == 0:
    torch.save(model.state_dict(), 'model.pt')
# Other ranks have different state!
```

### WarpForge Opportunity: Type-Safe Distributed Primitives

```java
// Rank is part of the execution context type
public interface DistributedContext<R extends Rank> {
    R rank();
    WorldSize worldSize();
}

// Operations that require synchronization
@RequiresAllRanks
public <S extends Shape, D extends DType>
Tensor<S, D> allReduce(DistributedContext<?> ctx, Tensor<S, D> tensor, ReduceOp op) {
    // Compiler ensures this is called on all ranks
    return collective.allReduce(ctx, tensor, op);
}

// Synchronized training step
public class DistributedTrainer<Ctx extends DistributedContext<?>> {

    @Synchronized  // Compiler verifies all ranks execute this
    public void trainStep(Ctx ctx, Batch batch) {
        var loss = model.forward(batch);
        loss.backward();

        // Type system ensures gradient sync happens
        gradientSynchronizer.allReduce(ctx, model.gradients());

        optimizer.step();
    }
}

// Rank-local operations are explicit
@RankLocal
public Tensor<?, ?> localComputation(DistributedContext<?> ctx, Tensor<?, ?> input) {
    // No synchronization required
    return someLocalOp(input);
}

// Deterministic random with rank-aware seeding
public class DistributedRandom {
    public static Tensor<S, F32> randn(DistributedContext<?> ctx, S shape) {
        // Seed derived from global seed + rank
        // Documented behavior: same across runs, different across ranks
        return Tensor.randn(shape, ctx.deterministicSeed());
    }

    public static Tensor<S, F32> randnSynchronized(DistributedContext<?> ctx, S shape) {
        // Same values on all ranks
        return Tensor.randn(shape, ctx.globalSeed());
    }
}

// Checkpoint with automatic rank handling
@Checkpoint
public void save(DistributedContext<?> ctx, Path path) {
    // Framework handles rank 0 saving, others wait
    // State is verified consistent across ranks before save
    checkpointManager.save(ctx, model, optimizer, path);
}

// Batch size automatically adjusted
DataLoader<T> loader = DistributedDataLoader.builder()
    .dataset(dataset)
    .globalBatchSize(256)  // Divided by world_size automatically
    .build(ctx);
```

### Distributed Checks to Implement

| Check | PyTorch Behavior | WarpForge Target |
|-------|------------------|------------------|
| Missing gradient sync | Silent divergence | @RequiresAllRanks annotation |
| Batch size scaling | Manual calculation | Automatic globalBatchSize |
| Non-deterministic across ranks | Silent divergence | Rank-aware seeding |
| Collective deadlock | Hang | Compile-time balanced calls |
| State dict inconsistency | Silent corruption | Verified distributed save |
| Rank-specific code paths | Silent bugs | Type-level rank tracking |

---

## Category 10: API Misuse and Version Errors

### PyTorch Pain Points

```python
# Argument order confusion
# Is it (input, weight) or (weight, input)?
F.linear(weight, input)  # Wrong order, runtime shape error

# Deprecated API silent breakage
model.cuda()  # Deprecated, use .to(device)
# Works, but behavior might change

# Version incompatibility
# Model saved with torch 1.x
# Loaded with torch 2.x
# Silent behavior change in some ops
```

### WarpForge Opportunity: Explicit, Versioned APIs

```java
// Named parameters via builder pattern
var output = Linear.builder()
    .input(x)        // Can't confuse order
    .weight(w)
    .bias(b)
    .build()
    .forward();

// Or via records with named fields
public record LinearParams(
    Tensor<?, ?, ?> input,
    Tensor<?, ?, ?> weight,
    Optional<Tensor<?, ?, ?>> bias
) {}

// Deprecation with migration path
@Deprecated(since = "2.0", forRemoval = true)
@ReplaceWith("tensor.to(device)")
public Tensor cuda(Tensor t) {
    return t.to(Device.cuda(0));
}

// Versioned serialization
@ModelVersion(2)
public record TransformerV2(...) implements Model {
    // Migration from V1
    public static TransformerV2 migrate(TransformerV1 v1) {
        return new TransformerV2(
            v1.encoder(),
            v1.decoder(),
            newFieldWithDefault()
        );
    }
}

// Automatic version detection and migration
Model model = ModelLoader.load(path);
// Detects version, applies migrations automatically
// Warns if loading old version

// Operation versioning
@Since("1.0")
@Stable
public Tensor relu(Tensor x) { ... }

@Since("2.0")
@Experimental
public Tensor gelu(Tensor x, boolean approximate) { ... }
```

---

## Implementation Priority

Based on impact and feasibility:

### Phase 1: Foundation (Immediate)
1. **Shape type parameters** - Phantom types for compile-time shape checking
2. **DType type parameters** - Explicit precision in type system
3. **Device type parameters** - Eliminate device mismatch errors
4. **Memory arena management** - AutoCloseable tensors with scoped arenas

### Phase 2: Training Safety (Near-term)
5. **Gradient mode types** - RequiresGrad/NoGrad/Detached in type system
6. **Training/Inference mode types** - Compile-time mode enforcement
7. **NaN checking mode** - Optional numerical validation
8. **Gradient scope** - Automatic zero_grad and memory management

### Phase 3: Data Pipeline (Medium-term)
9. **Data schema types** - Type-safe datasets and transforms
10. **Mode-aware augmentation** - Train vs eval transforms
11. **Normalization tracking** - Type-level normalization state

### Phase 4: Distributed (Long-term)
12. **Rank-aware types** - Distributed context in type system
13. **Synchronized operations** - Compile-time collective verification
14. **Distributed checkpoint** - Automatic rank coordination

---

## Success Metrics

WarpForge should aim for:

| Metric | Target |
|--------|--------|
| Compile-time caught errors | >80% of current PyTorch runtime errors |
| "Compiles but wrong" bugs | <5% of PyTorch equivalent |
| Time to first error | Seconds (compile) vs hours (runtime) |
| Error message quality | Location + cause + fix suggestion |
| IDE integration | Full autocomplete for shapes/dtypes |

---

## Conclusion

By encoding ML concepts (shapes, dtypes, devices, gradients, modes) into Java's type system, WarpForge can catch entire categories of errors at compile time. This transforms the developer experience from "debug at 3am when training crashes" to "fix the red squiggle in your IDE before you run anything."

The investment in type system design pays dividends in:
- Reduced debugging time
- Increased confidence in code correctness
- Better IDE tooling and autocomplete
- Self-documenting code (types are documentation)
- Easier refactoring (compiler catches breakage)

This is WarpForge's competitive moat: not just "Java for ML" but "ML that can't fail silently."
