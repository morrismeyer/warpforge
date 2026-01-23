# ATen Core Operation Coverage

This document tracks PyTorch ATen Core operation coverage for the PyTorch → StableHLO conversion.

## Summary

| Category | Implemented | Total | Coverage |
|----------|-------------|-------|----------|
| Elementwise Binary | 20 | 20 | 100% |
| Elementwise Unary | 35 | 35 | 100% |
| Comparison | 6 | 6 | 100% |
| Selection | 4 | 4 | 100% |
| Reduction | 16 | 16 | 100% |
| Matrix Operations | 12 | 12 | 100% |
| Shape Operations | 20 | 20 | 100% |
| Slicing/Indexing | 10 | 10 | 100% |
| Activation Functions | 18 | 18 | 100% |
| Convolution | 8 | 8 | 100% |
| Pooling | 7 | 7 | 100% |
| Normalization | 5 | 5 | 100% |
| Padding | 4 | 4 | 100% |
| Type Conversion | 8 | 8 | 100% |
| **Total** | **173** | **173** | **100%** |

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
| `fmod` | `stablehlo.remainder` | ✅ | Same as remainder for floats |
| `true_divide` | `stablehlo.divide` | ✅ | |
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
| `logical_xor` | `stablehlo.xor` | ✅ | |

### Elementwise Unary Operations

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `neg` | `stablehlo.negate` | ✅ | |
| `abs` | `stablehlo.abs` | ✅ | |
| `exp` | `stablehlo.exponential` | ✅ | |
| `exp2` | `exp(x * ln(2))` | ✅ | Composite |
| `log` | `stablehlo.log` | ✅ | |
| `sqrt` | `stablehlo.sqrt` | ✅ | |
| `rsqrt` | `stablehlo.rsqrt` | ✅ | |
| `sin` | `stablehlo.sine` | ✅ | |
| `cos` | `stablehlo.cosine` | ✅ | |
| `tan` | `stablehlo.tan` | ✅ | |
| `asin` | `stablehlo.asin` | ✅ | |
| `acos` | `stablehlo.acos` | ✅ | |
| `atan` | `stablehlo.atan` | ✅ | |
| `tanh` | `stablehlo.tanh` | ✅ | |
| `sinh` | `stablehlo.custom_call @sinh` | ✅ | |
| `cosh` | `stablehlo.custom_call @cosh` | ✅ | |
| `asinh` | `log(x + sqrt(x^2 + 1))` | ✅ | Composite |
| `acosh` | `log(x + sqrt(x^2 - 1))` | ✅ | Composite |
| `atanh` | `0.5 * log((1+x)/(1-x))` | ✅ | Composite |
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
| `erf` | `stablehlo.erf` | ✅ | |
| `erfc` | `1 - erf(x)` | ✅ | Composite |
| `lgamma` | `stablehlo.custom_call @lgamma` | ✅ | |
| `digamma` | `stablehlo.custom_call @digamma` | ✅ | |

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
| `select` | `stablehlo.slice` | ✅ | Dimension selection |

### Reduction Operations

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `sum` | `stablehlo.reduce` | ✅ | |
| `max` | `stablehlo.reduce` | ✅ | |
| `min` | `stablehlo.reduce` | ✅ | |
| `mean` | `stablehlo.reduce + divide` | ✅ | Composite |
| `prod` | `stablehlo.reduce` | ✅ | |
| `std` | `sqrt(var(x))` | ✅ | Composite |
| `var` | `mean((x - mean(x))^2)` | ✅ | Composite |
| `all` | `stablehlo.reduce` | ✅ | |
| `any` | `stablehlo.reduce` | ✅ | |
| `argmax` | `stablehlo.custom_call @argmax` | ✅ | |
| `argmin` | `stablehlo.custom_call @argmin` | ✅ | |
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
| `einsum` | `stablehlo.custom_call @einsum` | ✅ | |
| `tensordot` | `stablehlo.custom_call @tensordot` | ✅ | |
| `inner` | `stablehlo.dot_general` | ✅ | |

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
| `movedim` | `stablehlo.transpose` | ✅ | |
| `swapaxes` | `stablehlo.transpose` | ✅ | |
| `expand` | `stablehlo.broadcast_in_dim` | ✅ | |
| `concatenate` | `stablehlo.concatenate` | ✅ | |
| `stack` | `stablehlo.concatenate` | ✅ | |
| `split` | `stablehlo.slice` | ✅ | Multiple slices |
| `chunk` | `stablehlo.slice` | ✅ | Multiple slices |
| `unbind` | `stablehlo.custom_call @unbind` | ✅ | |
| `flip` | `stablehlo.reverse` | ✅ | |
| `roll` | `stablehlo.custom_call @roll` | ✅ | |
| `tile` | `stablehlo.broadcast_in_dim` | ✅ | |
| `narrow` | `stablehlo.slice` | ✅ | |
| `repeat` | `stablehlo.broadcast_in_dim` | ✅ | |

### Slicing/Indexing Operations

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `slice` | `stablehlo.slice` | ✅ | |
| `index_select` | `stablehlo.gather` | ✅ | |
| `gather` | `stablehlo.gather` | ✅ | |
| `scatter` | `stablehlo.scatter` | ✅ | |
| `masked_select` | `stablehlo.custom_call` | ✅ | |
| `index_copy` | `stablehlo.scatter` | ✅ | |
| `index_add` | `stablehlo.scatter` | ✅ | With addition |
| `index_fill` | `stablehlo.scatter` | ✅ | With constant |
| `take` | `stablehlo.gather` | ✅ | |
| `put` | `stablehlo.scatter` | ✅ | |

### Activation Functions

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `relu` | `stablehlo.maximum` | ✅ | |
| `leaky_relu` | `select + multiply` | ✅ | Composite |
| `elu` | `select + exp + subtract` | ✅ | Composite |
| `celu` | `max(0,x) + min(0, alpha*(exp(x/alpha)-1))` | ✅ | Composite |
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
| `logsigmoid` | `log(sigmoid(x))` | ✅ | Composite |
| `sigmoid` | `stablehlo.logistic` | ✅ | |
| `tanh` | `stablehlo.tanh` | ✅ | |

### Convolution Operations

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `conv1d` | `stablehlo.convolution` | ✅ | |
| `conv2d` | `stablehlo.convolution` | ✅ | |
| `conv3d` | `stablehlo.convolution` | ✅ | |
| `conv_transpose1d` | `stablehlo.convolution` | ✅ | Transposed |
| `conv_transpose2d` | `stablehlo.convolution` | ✅ | Transposed |
| `conv_transpose3d` | `stablehlo.convolution` | ✅ | Transposed |
| `depthwise_conv2d` | `stablehlo.convolution` | ✅ | Groups=channels |
| `separable_conv2d` | `stablehlo.convolution` | ✅ | Depthwise + pointwise |

### Pooling Operations

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `max_pool1d` | `stablehlo.reduce_window` | ✅ | |
| `max_pool2d` | `stablehlo.reduce_window` | ✅ | |
| `avg_pool1d` | `stablehlo.reduce_window` | ✅ | |
| `avg_pool2d` | `stablehlo.reduce_window` | ✅ | |
| `adaptive_avg_pool2d` | `stablehlo.reduce_window` | ✅ | |
| `adaptive_max_pool1d` | `stablehlo.reduce_window` | ✅ | |
| `adaptive_max_pool2d` | `stablehlo.reduce_window` | ✅ | |

### Normalization Operations

| PyTorch Op | StableHLO Target | Status | Notes |
|------------|------------------|--------|-------|
| `batch_norm` | `composite` | ✅ | |
| `layer_norm` | `composite` | ✅ | |
| `instance_norm` | `composite` | ✅ | Per-channel normalization |
| `group_norm` | `composite` | ✅ | Groups of channels |
| `local_response_norm` | `composite` | ✅ | LRN |

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
| `circular_pad` | `stablehlo.custom_call @circular_pad` | ✅ | |

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

## Completed Milestones

1. ✅ Full elementwise binary/unary coverage including inverse trig and special functions
2. ✅ All convolution variants (1D/2D/3D, transpose, depthwise)
3. ✅ All pooling variants (1D/2D, adaptive, max/avg)
4. ✅ Full normalization coverage (batch, layer, instance, group)
5. ✅ Complete shape operations including narrow, unbind, movedim, swapaxes
6. ✅ Full indexing operations including index_copy, index_add, index_fill
7. ✅ All activation functions including celu, logsigmoid
8. ✅ Advanced matrix operations (einsum, tensordot)
