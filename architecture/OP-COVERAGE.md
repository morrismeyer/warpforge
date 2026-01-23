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
| Training/Backward | 45 | 45 | 100% |
| **Total** | **218** | **218** | **100%** |

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
2. **FFT operations** not yet implemented (torch.fft module)

## Completed Milestones

1. ✅ Full elementwise binary/unary coverage including inverse trig and special functions
2. ✅ All convolution variants (1D/2D/3D, transpose, depthwise)
3. ✅ All pooling variants (1D/2D, adaptive, max/avg)
4. ✅ Full normalization coverage (batch, layer, instance, group)
5. ✅ Complete shape operations including narrow, unbind, movedim, swapaxes
6. ✅ Full indexing operations including index_copy, index_add, index_fill
7. ✅ All activation functions including celu, logsigmoid
8. ✅ Advanced matrix operations (einsum, tensordot)
9. ✅ Training/backward operations (~45 gradient ops for autograd support)
10. ✅ Dynamic shape support (dynamic_reshape, dynamic_slice, dynamic_broadcast, etc.)
11. ✅ Quantization operations (INT8/INT4) - bridges to Babylon ONNX quantization
12. ✅ Sparse tensor operations (COO, CSR, CSC, BSR, semi-structured)
13. ✅ Complex tensor operations (complex64/complex128) - native StableHLO support

## Dynamic Shape Support

### Current State

The converter currently assumes **static shapes** at trace time. All tensor dimensions are concrete integers:

```mlir
// Current output - all dimensions are concrete
func.func @forward(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32>
```

### Target State

Support **dynamic dimensions** using StableHLO's `?` marker for unknown sizes:

```mlir
// Target output - batch dimension is dynamic
func.func @forward(%arg0: tensor<?x3x224x224xf32>) -> tensor<?x1000xf32>
```

### Implementation Requirements

1. **Shape Tracking**: Track which dimensions are dynamic vs static through the graph
2. **Type Inference**: Propagate dynamic dimensions through operations
3. **Symbolic Shapes**: Use `stablehlo.get_dimension_size` for runtime shape queries
4. **Shape Constraints**: Emit `stablehlo.dynamic_*` variants where needed

### Operations Requiring Dynamic Variants

| Static Op | Dynamic Variant | Notes |
|-----------|-----------------|-------|
| `reshape` | `dynamic_reshape` | Target shape from tensor |
| `slice` | `dynamic_slice` | Bounds from tensors |
| `broadcast_in_dim` | `dynamic_broadcast_in_dim` | Output shape from tensor |
| `pad` | `dynamic_pad` | Padding from tensors |
| `gather` | `dynamic_gather` | Slice sizes from tensor |
| `iota` | `dynamic_iota` | Output shape from tensor |

### PyTorch Dynamic Shape Sources

- `torch.export` with dynamic batch dim: `dynamic_shapes={"x": {0: Dim("batch")}}`
- Variable sequence lengths in NLP models
- Dynamic image sizes in detection models

### Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Dynamic dimension tracking | ✅ | `dynamic_dims` parameter in FXToStableHLO |
| `?` emission in type signatures | ✅ | `tensor<?x8xf32>` format |
| Dynamic dim propagation | ✅ | Through elementwise, reduction, transpose |
| `dynamic_reshape` | ✅ | Emitted when input/output has dynamic dims |
| `dynamic_slice` | ✅ | For narrow/select with dynamic dims |
| `dynamic_broadcast_in_dim` | ✅ | For expand with dynamic dims |
| `dynamic_pad` | ✅ | For pad with dynamic dims |
| `dynamic_gather` | ✅ | For gather/index_select with dynamic dims |
| `get_dimension_size` | ✅ | For runtime shape queries |

### Test Models

| Model | Dynamic Dims | Description |
|-------|--------------|-------------|
| `dynamic_batch_mlp` | batch | MLP with variable batch size |
| `dynamic_batch_conv` | batch | Conv2d with variable batch |
| `dynamic_seq_transformer` | batch, seq | Transformer with variable sequence |
| `dynamic_reshape` | batch | Reshape preserving batch dim |
| `dynamic_matmul` | batch | Batched matmul |
| `dynamic_reduction` | batch | Reduction preserving batch |
| `dynamic_broadcast` | batch | Broadcasting with dynamic batch |
| `dynamic_transpose` | batch, seq | Transpose with dimension remapping |
| `dynamic_slice` | batch | Narrow with dynamic batch |
| `dynamic_select` | batch | Select with dynamic batch |
| `dynamic_pad` | batch | Padding with dynamic batch |
| `dynamic_gather` | batch | Gather with dynamic batch |
| `dynamic_expand` | batch | Expand/broadcast with dynamic batch |
| `dynamic_index_select` | batch | Index select with dynamic batch |

## Quantization Support

### Overview

WarpForge supports INT8/INT4 quantized inference by bridging PyTorch quantization to Babylon's ONNX quantization operators.

```
PyTorch Quantization          StableHLO              Babylon ONNX
─────────────────────────────────────────────────────────────────
quantize_per_tensor    →   uniform_quantize    →   QuantizeLinear
dequantize             →   uniform_dequantize  →   DequantizeLinear
quantized_linear       →   custom_call         →   QLinearMatMul
quantized_conv2d       →   custom_call         →   QLinearConv
fake_quantize_*        →   clamp+round+scale   →   (training only)
```

### Supported Data Types

| Type | Bits | Range | Use Case |
|------|------|-------|----------|
| `qint8` / `i8` | 8 | -128 to 127 | Standard quantization |
| `quint8` / `ui8` | 8 | 0 to 255 | Activation quantization |
| `qint4` / `i4` | 4 | -8 to 7 | LLM weight compression (GPTQ, AWQ) |
| `quint4` / `ui4` | 4 | 0 to 15 | Ultra-low precision |

### Implemented Operations

| PyTorch Op | StableHLO Target | Babylon ONNX | Status |
|------------|------------------|--------------|--------|
| `quantize_per_tensor` | `uniform_quantize` | `QuantizeLinear` | ✅ |
| `dequantize` | `uniform_dequantize` | `DequantizeLinear` | ✅ |
| `fake_quantize_per_tensor_affine` | clamp+round+scale | (QAT) | ✅ |
| `fake_quantize_per_channel_affine` | `custom_call` | (QAT) | ✅ |
| `quantized_linear` | `custom_call @quantized_linear` | `QLinearMatMul` | ✅ |
| `quantized_conv2d` | `custom_call @quantized_conv2d` | `QLinearConv` | ✅ |
| `quantized_batch_norm` | `custom_call` | (composite) | ✅ |
| `quantized_add` | `custom_call @quantized_add` | (elementwise) | ✅ |
| `quantized_mul` | `custom_call @quantized_mul` | (elementwise) | ✅ |
| `quantized_relu` | `maximum` | (same) | ✅ |
| `int_repr` | `bitcast_convert` | (access int values) | ✅ |
| `q_scale` | `custom_call @q_scale` | (get scale) | ✅ |
| `q_zero_point` | `custom_call @q_zero_point` | (get zero point) | ✅ |

### Babylon ONNX Integration

Babylon's ONNX support (in `oracle.code.onnx`) provides the downstream quantization operators:

```java
// Babylon ONNX tensor types (oracle.code.onnx.Tensor.ElementType)
INT4(22, Object.class),   // 4-bit signed
UINT4(21, Object.class),  // 4-bit unsigned
INT8(3, byte.class),      // 8-bit signed
UINT8(2, byte.class),     // 8-bit unsigned

// Babylon ONNX operators (oracle.code.onnx.ir.OnnxOps)
QuantizeLinear      // float → quantized
DequantizeLinear    // quantized → float
DynamicQuantizeLinear // runtime quantization
MatMulInteger       // INT8 matrix multiplication
QLinearMatMul       // quantized matmul with scale/zero_point
QLinearConv         // quantized convolution
```

### Test Models

| Model | Description |
|-------|-------------|
| `fake_quantize_per_tensor` | QAT fake quantization |
| `quantized_linear` | Quantized linear layer |
| `quantized_conv2d` | Quantized convolution |
| `quantized_relu` | Quantized ReLU activation |
| `quantized_add` / `quantized_mul` | Quantized elementwise ops |
| `int8_linear` | Full INT8 inference model |
| `int4_linear` | INT4 weight compression model |

## Sparse Tensor Support

### Overview

WarpForge supports sparse tensor operations by mapping PyTorch sparse tensors to `stablehlo.custom_call` with format metadata. This approach is used because:

1. **StableHLO Sparsity is RFC-level** - Native sparse tensor support in StableHLO is still a proposal (2023 RFC)
2. **MLIR sparse_tensor dialect** - Has mature support, but requires mixed-dialect output
3. **Backend flexibility** - `custom_call` lets backends implement efficient kernels (cuSPARSE, MKL, etc.)

### Supported Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| **COO** | Coordinate format (`indices`, `values`) | General sparse, easy construction |
| **CSR** | Compressed Sparse Row | Efficient row slicing, SpMV |
| **CSC** | Compressed Sparse Column | Efficient column slicing |
| **BSR** | Block Sparse Row | Block-structured sparsity |
| **Semi-structured** | 2:4 sparsity pattern | NVIDIA Ampere/Hopper Sparse Tensor Cores |

### Implemented Operations

| PyTorch Op | StableHLO Target | Description |
|------------|------------------|-------------|
| `sparse_coo_tensor` | `custom_call @sparse_coo_tensor` | Create COO tensor |
| `sparse_csr_tensor` | `custom_call @sparse_csr_tensor` | Create CSR tensor |
| `sparse_csc_tensor` | `custom_call @sparse_csc_tensor` | Create CSC tensor |
| `sparse_bsr_tensor` | `custom_call @sparse_bsr_tensor` | Create BSR tensor |
| `to_sparse` / `to_sparse_coo` | `custom_call @to_sparse_coo` | Dense → COO |
| `to_sparse_csr` | `custom_call @to_sparse_csr` | Dense → CSR |
| `to_sparse_csc` | `custom_call @to_sparse_csc` | Dense → CSC |
| `to_sparse_bsr` | `custom_call @to_sparse_bsr` | Dense → BSR |
| `to_dense` | `custom_call @sparse_to_dense` | Sparse → Dense |
| `coalesce` | `custom_call @sparse_coalesce` | Merge duplicate indices |
| `indices` | `custom_call @sparse_indices` | Get COO indices |
| `values` | `custom_call @sparse_values` | Get non-zero values |
| `crow_indices` | `custom_call @sparse_crow_indices` | Get CSR row pointers |
| `col_indices` | `custom_call @sparse_col_indices` | Get CSR/CSC column indices |
| `nnz` | `custom_call @sparse_nnz` | Count non-zeros |
| `sparse.mm` | `custom_call @sparse_mm` | Sparse @ Dense matmul |
| `sparse.addmm` | `custom_call @sparse_addmm` | β*C + α*(A @ B) |
| `sparse.sampled_addmm` | `custom_call @sparse_sampled_addmm` | Masked sparse attention |
| `sparse.sum` | `custom_call @sparse_sum` | Sparse reduction |
| `sparse.softmax` | `custom_call @sparse_softmax` | Softmax over non-zeros |

### Semi-Structured Sparsity (2:4)

NVIDIA Ampere and Hopper GPUs support **2:4 structured sparsity** in Sparse Tensor Cores, providing up to 2x speedup for matrix operations:

```python
# Mark weights for 2:4 sparsity
sparse_weight = to_sparse_semi_structured(dense_weight)
```

Emits:
```mlir
%sparse = stablehlo.custom_call @to_sparse_semi_structured(%dense)
    {format = "semi_structured", pattern = "2:4"} : (tensor<M x K x f16>) -> tensor<M x K x f16>
```

### Test Models

| Model | Description |
|-------|-------------|
| `sparse_coo_tensor` | Create COO from indices/values |
| `sparse_csr_tensor` | Create CSR from compressed indices |
| `to_sparse_coo` | Convert dense to COO |
| `to_sparse_csr` | Convert dense to CSR |
| `to_sparse_bsr` | Convert dense to BSR with block size |
| `sparse_mm` | Sparse matrix multiplication |
| `sparse_addmm` | Fused sparse matmul with add |
| `sparse_sum` | Sum reduction on sparse |
| `sparse_softmax` | Softmax over non-zeros |
| `sparse_indices` / `sparse_values` | Access sparse components |
| `semi_structured_sparse` | 2:4 sparsity for NVIDIA GPUs |

### Backend Implementation Notes

Backends should implement sparse custom_calls using:
- **NVIDIA**: cuSPARSE for COO/CSR/CSC, cuSPARSELt for semi-structured
- **AMD**: hipSPARSE, rocSPARSE
- **CPU**: Intel MKL Sparse, oneDNN
- **Generic**: MLIR sparse_tensor dialect lowering

## Complex Tensor Support

### Overview

WarpForge supports complex tensor operations using StableHLO's native complex number types (`complex<f32>` for complex64, `complex<f64>` for complex128).

### Implemented Operations

| PyTorch Op | StableHLO Target | Description |
|------------|------------------|-------------|
| `torch.complex(real, imag)` | `stablehlo.complex` | Create complex from parts |
| `torch.real(x)` | `stablehlo.real` | Extract real part |
| `torch.imag(x)` | `stablehlo.imag` | Extract imaginary part |
| `torch.conj(x)` | composite | Complex conjugate |
| `torch.angle(x)` | composite | Phase angle |
| `torch.polar(abs, angle)` | composite | From polar coords |
| `torch.view_as_real(x)` | `custom_call` | View as real tensor |
| `torch.view_as_complex(x)` | `custom_call` | View as complex tensor |

### Test Models

| Model | Description |
|-------|-------------|
| `complex` | Create complex from real/imag |
| `real` / `imag` | Extract components |
| `conj` | Complex conjugate |
| `angle` / `polar` | Polar coordinates |
| `complex_abs` / `complex_mul` / `complex_add` | Arithmetic |
| `complex_exp` / `complex_log` / `complex_sqrt` | Transcendentals |
