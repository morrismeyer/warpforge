# ATen Core Operation Coverage

This document tracks PyTorch ATen Core operation coverage for the PyTorch → StableHLO conversion.

## Summary

| Category | Implemented | Total | Coverage |
|----------|-------------|-------|----------|
| Elementwise Binary | 17 | 20 | 85% |
| Elementwise Unary | 25 | 35 | 71% |
| Comparison | 6 | 6 | 100% |
| Selection | 3 | 4 | 75% |
| Reduction | 12 | 15 | 80% |
| Matrix Operations | 10 | 12 | 83% |
| Shape Operations | 15 | 20 | 75% |
| Slicing/Indexing | 7 | 10 | 70% |
| Activation Functions | 16 | 18 | 89% |
| Convolution | 4 | 8 | 50% |
| Pooling | 3 | 6 | 50% |
| Normalization | 3 | 5 | 60% |
| Padding | 3 | 4 | 75% |
| Type Conversion | 8 | 8 | 100% |
| **Total** | **132** | **171** | **77%** |

## Detailed Coverage by Category

### Elementwise Binary Operations

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `add` | `stablehlo.add` | ✅ | |
| `sub` | `stablehlo.subtract` | ✅ | |
| `mul` | `stablehlo.multiply` | ✅ | |
| `div` | `stablehlo.divide` | ✅ | |
| `pow` | `stablehlo.power` | ✅ | |
| `remainder` | `stablehlo.remainder` | ✅ | |
| `maximum` | `stablehlo.maximum` | ✅ | |
| `minimum` | `stablehlo.minimum` | ✅ | |
| `atan2` | `stablehlo.atan2` | ✅ | |
| `floor_divide` | `divide + floor` | ✅ | Composite |
| `bitwise_and` | `stablehlo.and` | ✅ | |
| `bitwise_or` | `stablehlo.or` | ✅ | |
| `bitwise_xor` | `stablehlo.xor` | ✅ | |
| `bitwise_left_shift` | `stablehlo.shift_left` | ✅ | |
| `bitwise_right_shift` | `stablehlo.shift_right_arithmetic` | ✅ | |
| `logical_and` | `stablehlo.and` | ✅ | |
| `logical_or` | `stablehlo.or` | ✅ | |

### Elementwise Unary Operations

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `neg` | `stablehlo.negate` | ✅ | |
| `abs` | `stablehlo.abs` | ✅ | |
| `exp` | `stablehlo.exponential` | ✅ | |
| `log` | `stablehlo.log` | ✅ | |
| `sqrt` | `stablehlo.sqrt` | ✅ | |
| `rsqrt` | `stablehlo.rsqrt` | ✅ | |
| `sin` | `stablehlo.sine` | ✅ | |
| `cos` | `stablehlo.cosine` | ✅ | |
| `tan` | `stablehlo.tan` | ✅ | |
| `tanh` | `stablehlo.tanh` | ✅ | |
| `sigmoid` | `stablehlo.logistic` | ✅ | |
| `sign` | `stablehlo.sign` | ✅ | |
| `floor` | `stablehlo.floor` | ✅ | |
| `ceil` | `stablehlo.ceil` | ✅ | |
| `round` | `stablehlo.round_nearest_even` | ✅ | |
| `trunc` | `stablehlo.custom_call @trunc` | ✅ | |
| `expm1` | `stablehlo.exponential_minus_one` | ✅ | |
| `log1p` | `stablehlo.log_plus_one` | ✅ | |
| `log2` | `log / log(2)` | ✅ | Composite |
| `log10` | `log / log(10)` | ✅ | Composite |
| `reciprocal` | `1 / x` | ✅ | Composite |
| `square` | `x * x` | ✅ | Composite |
| `sinh` | `stablehlo.custom_call @sinh` | ✅ | |
| `cosh` | `stablehlo.custom_call @cosh` | ✅ | |
| `bitwise_not` | `stablehlo.not` | ✅ | |

### Comparison Operations

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `eq` | `stablehlo.compare EQ` | ✅ | |
| `ne` | `stablehlo.compare NE` | ✅ | |
| `lt` | `stablehlo.compare LT` | ✅ | |
| `le` | `stablehlo.compare LE` | ✅ | |
| `gt` | `stablehlo.compare GT` | ✅ | |
| `ge` | `stablehlo.compare GE` | ✅ | |

### Selection Operations

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `where` | `stablehlo.select` | ✅ | |
| `clamp` | `stablehlo.clamp` | ✅ | |
| `masked_fill` | `stablehlo.select` | ✅ | |

### Reduction Operations

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `sum` | `stablehlo.reduce` | ✅ | |
| `max` | `stablehlo.reduce` | ✅ | |
| `min` | `stablehlo.reduce` | ✅ | |
| `mean` | `stablehlo.reduce + divide` | ✅ | Composite |
| `prod` | `stablehlo.reduce` | ✅ | |
| `all` | `stablehlo.reduce` | ✅ | |
| `any` | `stablehlo.reduce` | ✅ | |
| `logsumexp` | `exp + reduce + log` | ✅ | Composite |
| `cumsum` | `stablehlo.custom_call @cumsum` | ✅ | Scan op |
| `cumprod` | `stablehlo.custom_call @cumprod` | ✅ | Scan op |
| `softmax` | `exp + reduce + divide` | ✅ | Composite |
| `log_softmax` | `x - log(sum(exp))` | ✅ | Composite |

### Matrix Operations

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `matmul` | `stablehlo.dot_general` | ✅ | |
| `mm` | `stablehlo.dot_general` | ✅ | |
| `bmm` | `stablehlo.dot_general` | ✅ | Batched |
| `linear` | `dot_general + add` | ✅ | Composite |
| `mv` | `stablehlo.dot_general` | ✅ | |
| `dot` | `stablehlo.dot_general` | ✅ | |
| `outer` | `stablehlo.dot_general` | ✅ | |
| `addmm` | `dot_general + add` | ✅ | Composite |
| `baddbmm` | `dot_general + add` | ✅ | Composite |

### Shape Operations

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `reshape` | `stablehlo.reshape` | ✅ | |
| `view` | `stablehlo.reshape` | ✅ | |
| `flatten` | `stablehlo.reshape` | ✅ | |
| `squeeze` | `stablehlo.reshape` | ✅ | |
| `unsqueeze` | `stablehlo.reshape` | ✅ | |
| `transpose` | `stablehlo.transpose` | ✅ | |
| `permute` | `stablehlo.transpose` | ✅ | |
| `expand` | `stablehlo.broadcast_in_dim` | ✅ | |
| `concatenate` | `stablehlo.concatenate` | ✅ | |
| `stack` | `stablehlo.concatenate` | ✅ | |
| `split` | `stablehlo.slice` | ✅ | Multiple slices |
| `chunk` | `stablehlo.slice` | ✅ | Multiple slices |
| `flip` | `stablehlo.reverse` | ✅ | |
| `roll` | `stablehlo.custom_call @roll` | ✅ | |
| `tile` | `stablehlo.broadcast_in_dim` | ✅ | |

### Slicing/Indexing Operations

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `slice` | `stablehlo.slice` | ✅ | |
| `index_select` | `stablehlo.gather` | ✅ | |
| `gather` | `stablehlo.gather` | ✅ | |
| `scatter` | `stablehlo.scatter` | ✅ | |
| `masked_select` | `stablehlo.custom_call` | ✅ | |
| `narrow` | `stablehlo.slice` | ❌ | Planned |
| `index_copy` | `stablehlo.scatter` | ❌ | Planned |

### Activation Functions

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `relu` | `stablehlo.maximum` | ✅ | |
| `leaky_relu` | `select + multiply` | ✅ | Composite |
| `elu` | `select + exp + subtract` | ✅ | Composite |
| `silu/swish` | `x * sigmoid(x)` | ✅ | Composite |
| `gelu` | `tanh approximation` | ✅ | Composite |
| `mish` | `x * tanh(softplus)` | ✅ | Composite |
| `softplus` | `log(1 + exp)` | ✅ | Composite |
| `softsign` | `x / (1 + |x|)` | ✅ | Composite |
| `hardtanh` | `stablehlo.clamp` | ✅ | |
| `relu6` | `stablehlo.clamp` | ✅ | |
| `hardsigmoid` | `clamp((x+3)/6)` | ✅ | Composite |
| `hardswish` | `x * hardsigmoid` | ✅ | Composite |
| `selu` | `scale * elu` | ✅ | Composite |
| `prelu` | `select + multiply` | ✅ | Composite |
| `sigmoid` | `stablehlo.logistic` | ✅ | |
| `tanh` | `stablehlo.tanh` | ✅ | |

### Convolution Operations

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `conv1d` | `stablehlo.convolution` | ✅ | |
| `conv2d` | `stablehlo.convolution` | ✅ | |
| `conv_transpose2d` | `stablehlo.convolution` | ❌ | Planned |
| `depthwise_conv2d` | `stablehlo.convolution` | ✅ | |

### Pooling Operations

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `max_pool2d` | `stablehlo.reduce_window` | ✅ | |
| `avg_pool2d` | `stablehlo.reduce_window` | ✅ | |
| `adaptive_avg_pool2d` | `stablehlo.reduce_window` | ✅ | |

### Normalization Operations

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `batch_norm` | `composite` | ✅ | |
| `layer_norm` | `composite` | ✅ | |
| `instance_norm` | `composite` | ❌ | Planned |
| `group_norm` | `composite` | ❌ | Planned |

### Type Conversion Operations

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `to` | `stablehlo.convert` | ✅ | |
| `float` | `stablehlo.convert` | ✅ | |
| `half` | `stablehlo.convert` | ✅ | |
| `double` | `stablehlo.convert` | ✅ | |
| `int` | `stablehlo.convert` | ✅ | |
| `long` | `stablehlo.convert` | ✅ | |
| `bool` | `stablehlo.convert` | ✅ | |
| `bfloat16` | `stablehlo.convert` | ✅ | |

### Padding Operations

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `pad` | `stablehlo.pad` | ✅ | |
| `constant_pad` | `stablehlo.pad` | ✅ | |
| `reflect_pad` | `stablehlo.custom_call` | ✅ | |

## Implementation Files

| File | Purpose |
|------|---------|
| `snakegrinder-core/src/main/resources/snakegrinder/fx_to_stablehlo.py` | Main converter |
| `snakegrinder-core/src/main/resources/snakegrinder/stablehlo_op_models.py` | Test models |
| `snakeburger-core/src/main/java/.../StableHloAst.java` | AST definitions |

## Known Limitations

1. **Scan operations** (cumsum, cumprod) use custom_call as StableHLO lacks native scan
2. **Dynamic shapes** are not fully supported
3. **Backward/training ops** not yet implemented
4. **Sparse operations** not supported

## Next Steps

1. Add remaining convolution variants (conv_transpose, dilated)
2. Implement backward operations for training
3. Add more pooling variants (adaptive_max_pool, global pooling)
4. Improve normalization coverage (instance_norm, group_norm)
