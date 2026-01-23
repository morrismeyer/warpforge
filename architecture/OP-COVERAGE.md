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
| RNN/LSTM/GRU | 17 | 17 | 100% |
| Attention | 26 | 26 | 100% |
| Embedding | 23 | 23 | 100% |
| Loss Functions | 32 | 32 | 100% |
| **Total** | **316** | **316** | **100%** |

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

1. **Scan operations** use `custom_call` since StableHLO lacks native scan primitives (backends implement via thrust/CUB on CUDA, parallel_scan on CPU)

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
14. ✅ FFT operations (torch.fft module) - native StableHLO FFT support
15. ✅ Scan/cumulative operations (cumsum, cumprod, cummax, cummin, logcumsumexp, diff)
16. ✅ RNN/LSTM/GRU operations (lstm, gru, rnn, cells, bidirectional, multi-layer)
17. ✅ Attention operations (scaled_dot_product_attention, multi_head_attention, transformer layers)
18. ✅ Embedding operations (embedding, embedding_bag, one_hot, positional embedding)
19. ✅ Loss functions (cross_entropy, mse, l1, huber, kl_div, triplet, ctc, etc.)

## Loss Functions Support

### Overview

WarpForge supports comprehensive loss functions for training neural networks across classification, regression, metric learning, and sequence modeling tasks.

### Supported Operations

| PyTorch Op | StableHLO Target | Description |
|------------|------------------|-------------|
| `nn.CrossEntropyLoss` | `custom_call @cross_entropy_loss` | Multi-class classification |
| `nn.NLLLoss` | `custom_call @cross_entropy_loss` | Negative log likelihood |
| `nn.BCELoss` | `custom_call @binary_cross_entropy` | Binary classification |
| `nn.BCEWithLogitsLoss` | `custom_call @bce_with_logits` | Binary with logits |
| `nn.MSELoss` | `custom_call @mse_loss` | Mean squared error |
| `nn.L1Loss` | `custom_call @l1_loss` | Mean absolute error |
| `nn.SmoothL1Loss` | `custom_call @smooth_l1_loss` | Huber loss variant |
| `nn.HuberLoss` | `custom_call @huber_loss` | Robust regression |
| `nn.KLDivLoss` | `custom_call @kl_div` | Distribution divergence |
| `nn.TripletMarginLoss` | `custom_call @triplet_margin_loss` | Metric learning |
| `nn.CosineEmbeddingLoss` | `custom_call @cosine_embedding_loss` | Cosine similarity |
| `nn.CTCLoss` | `custom_call @ctc_loss` | Sequence-to-sequence |
| `nn.PoissonNLLLoss` | `custom_call @poisson_nll_loss` | Count data |
| `nn.GaussianNLLLoss` | `custom_call @gaussian_nll_loss` | Uncertainty modeling |

### Test Models (32 total)

**Classification Losses (10)**:
- `cross_entropy_loss` - Standard cross-entropy
- `cross_entropy_loss_weighted` - With class weights
- `cross_entropy_loss_ignore_index` - Ignore padding
- `cross_entropy_loss_label_smoothing` - Regularization
- `cross_entropy_loss_sum/none` - Different reductions
- `nll_loss` - Negative log likelihood
- `bce_loss`, `bce_with_logits_loss`, `bce_with_logits_pos_weight` - Binary

**Regression Losses (7)**:
- `mse_loss`, `mse_loss_sum` - Mean squared error
- `l1_loss` - Mean absolute error
- `smooth_l1_loss`, `smooth_l1_loss_beta` - Huber variant
- `huber_loss`, `huber_loss_delta` - Robust regression

**Distribution Losses (2)**:
- `kl_div_loss`, `kl_div_loss_log_target` - KL divergence

**Metric Learning Losses (5)**:
- `hinge_embedding_loss` - Hinge loss
- `margin_ranking_loss` - Ranking
- `triplet_margin_loss`, `triplet_margin_loss_custom` - Triplet
- `cosine_embedding_loss` - Cosine similarity

**Sequence Losses (2)**:
- `ctc_loss`, `ctc_loss_zero_infinity` - CTC

**Other Losses (6)**:
- `poisson_nll_loss` - Count data
- `gaussian_nll_loss` - Uncertainty
- `soft_margin_loss`, `multi_margin_loss` - Margin
- `multilabel_margin_loss`, `multilabel_soft_margin_loss` - Multi-label

## Embedding Operations Support

### Overview

WarpForge supports comprehensive embedding operations for NLP, recommendation systems, and any model requiring discrete index-to-vector lookups.

### Supported Operations

| PyTorch Op | StableHLO Target | Description |
|------------|------------------|-------------|
| `nn.Embedding` | `stablehlo.gather` | Basic embedding lookup |
| `nn.Embedding(padding_idx=...)` | `stablehlo.gather` | Embedding with padding token |
| `nn.Embedding(max_norm=...)` | `custom_call @embedding` | Embedding with norm constraint |
| `nn.Embedding(sparse=True)` | `custom_call @embedding` | Sparse gradient embedding |
| `F.embedding` | `stablehlo.gather` | Functional embedding API |
| `nn.EmbeddingBag(mode='sum')` | `custom_call @embedding_bag` | Sum pooled embeddings |
| `nn.EmbeddingBag(mode='mean')` | `custom_call @embedding_bag` | Mean pooled embeddings |
| `nn.EmbeddingBag(mode='max')` | `custom_call @embedding_bag` | Max pooled embeddings |
| `F.one_hot` | `custom_call @one_hot` | One-hot encoding |

### Key Features

- **Padding support**: `padding_idx` zeroes out specific tokens
- **Norm constraints**: `max_norm` renormalizes vectors exceeding threshold
- **Sparse gradients**: `sparse=True` for efficient large vocabulary training
- **EmbeddingBag modes**: sum, mean, max pooling for variable-length sequences
- **Per-sample weights**: Weighted combination in EmbeddingBag
- **Positional encoding**: Learned position embeddings for transformers

### Test Models (23 total)

**Basic Embedding (7)**:
- `embedding` - Basic nn.Embedding
- `embedding_with_padding` - With padding_idx
- `embedding_with_max_norm` - With max_norm constraint
- `embedding_with_norm_type` - Custom p-norm
- `embedding_scale_grad` - Frequency-scaled gradients
- `embedding_sparse` - Sparse gradients
- `embedding_functional` - F.embedding API

**EmbeddingBag (7)**:
- `embedding_bag_sum` - Sum pooling
- `embedding_bag_mean` - Mean pooling
- `embedding_bag_max` - Max pooling
- `embedding_bag_with_weights` - Per-sample weights
- `embedding_bag_padding` - With padding_idx
- `embedding_bag_sparse` - Sparse gradients
- `embedding_bag_last_offset` - include_last_offset=True

**One-Hot (2)**:
- `one_hot` - Fixed num_classes
- `one_hot_dynamic` - Auto-detect num_classes

**Embedding Patterns (7)**:
- `embedding_with_projection` - Embedding → Linear
- `embedding_with_dropout` - Embedding → Dropout
- `embedding_with_layernorm` - Embedding → LayerNorm
- `positional_embedding` - Token + Position embeddings
- `embedding_sum_2d` - Token + Segment + Position (BERT-style)

## Attention Operations Support

### Overview

WarpForge supports comprehensive attention mechanisms for transformer and sequence models. These operations map to `stablehlo.custom_call` to enable backend-optimized implementations (Flash Attention on NVIDIA, Memory-efficient attention on AMD, etc.).

### Supported Operations

| PyTorch Op | StableHLO Target | Description |
|------------|------------------|-------------|
| `F.scaled_dot_product_attention` | `custom_call @scaled_dot_product_attention` | Core SDPA operation |
| `F.scaled_dot_product_attention(..., is_causal=True)` | `custom_call @scaled_dot_product_attention` | Causal masking for autoregressive |
| `F.scaled_dot_product_attention(..., attn_mask=...)` | `custom_call @scaled_dot_product_attention` | Explicit attention mask |
| `nn.MultiheadAttention` | `custom_call @multi_head_attention` | Multi-head attention layer |
| `nn.TransformerEncoderLayer` | (decomposed) | Attention + FFN + residuals |
| `nn.TransformerDecoderLayer` | (decomposed) | Self-attn + cross-attn + FFN |
| `nn.TransformerEncoder` | (decomposed) | Stacked encoder layers |
| `nn.TransformerDecoder` | (decomposed) | Stacked decoder layers |
| `nn.Transformer` | (decomposed) | Full encoder-decoder model |

### Key Features

- **Causal masking**: `is_causal=True` for autoregressive generation (GPT-style)
- **Custom masking**: Explicit attention masks for padding, bidirectional attention
- **Dropout**: Training-time attention dropout
- **Custom scale**: Override default 1/sqrt(d_k) scaling
- **Batch-first**: Support both (seq, batch, embed) and (batch, seq, embed) layouts
- **Cross-attention**: Query from one source, key/value from another
- **Key padding mask**: Ignore padding tokens in sequences

### Backend Optimizations

The `custom_call` approach allows backends to select optimal implementations:

| Backend | Implementation | Notes |
|---------|---------------|-------|
| NVIDIA | Flash Attention v2 | cuDNN 8.9+ required for fused kernel |
| NVIDIA | Memory-efficient attention | xFormers fallback |
| AMD | Composable Kernel | ROCm 6.0+ |
| CPU | oneDNN | Optimized matmul + softmax |

### Test Models (26 total)

**Scaled Dot-Product Attention (5)**:
- `scaled_dot_product_attention` - Basic SDPA
- `sdpa_causal` - Causal masking
- `sdpa_dropout` - With dropout
- `sdpa_mask` - Explicit attention mask
- `sdpa_scale` - Custom scale factor

**Multi-Head Attention (11)**:
- `multihead_attention` - Basic MHA (seq-first)
- `mha_batch_first` - Batch-first layout
- `mha_key_padding_mask` - Key padding mask
- `mha_attn_mask` - Attention mask
- `mha_dropout` - With dropout
- `mha_no_bias` - Without bias
- `mha_add_bias_kv` - Bias to key/value
- `mha_add_zero_attn` - Zero attention
- `mha_kdim_vdim` - Different key/value dimensions
- `self_attention` - Self-attention pattern
- `cross_attention` - Cross-attention pattern

**Attention Patterns (3)**:
- `attention_with_projection` - Attention + linear
- `attention_with_layernorm` - Pre-norm attention
- `attention_with_residual` - Attention + skip connection

**Transformer Modules (7)**:
- `transformer_encoder_layer` - Single encoder layer
- `transformer_encoder_layer_batch_first` - Batch-first encoder
- `transformer_decoder_layer` - Single decoder layer
- `transformer_encoder` - Stacked encoder
- `transformer_decoder` - Stacked decoder
- `transformer` - Full encoder-decoder model

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

## FFT Operations Support

### Overview

WarpForge supports FFT (Fast Fourier Transform) operations using StableHLO's native `stablehlo.fft` operation. StableHLO FFT has built-in support for:
- `FFT` - Complex-to-complex forward transform
- `IFFT` - Complex-to-complex inverse transform
- `RFFT` - Real-to-complex forward transform
- `IRFFT` - Complex-to-real inverse transform

### Implemented Operations

| PyTorch Op | StableHLO Target | Description |
|------------|------------------|-------------|
| `torch.fft.fft` | `stablehlo.fft FFT` | 1D complex FFT |
| `torch.fft.ifft` | `stablehlo.fft IFFT` | 1D complex inverse FFT |
| `torch.fft.rfft` | `stablehlo.fft RFFT` | 1D real-to-complex FFT |
| `torch.fft.irfft` | `stablehlo.fft IRFFT` | 1D complex-to-real FFT |
| `torch.fft.hfft` | `stablehlo.fft IRFFT` | Hermitian FFT |
| `torch.fft.ihfft` | `stablehlo.fft RFFT` | Inverse Hermitian FFT |
| `torch.fft.fft2` | `stablehlo.fft FFT` | 2D complex FFT |
| `torch.fft.ifft2` | `stablehlo.fft IFFT` | 2D complex inverse FFT |
| `torch.fft.rfft2` | `stablehlo.fft RFFT` | 2D real-to-complex FFT |
| `torch.fft.irfft2` | `stablehlo.fft IRFFT` | 2D complex-to-real FFT |
| `torch.fft.fftn` | `stablehlo.fft FFT` | N-D complex FFT |
| `torch.fft.ifftn` | `stablehlo.fft IFFT` | N-D complex inverse FFT |
| `torch.fft.rfftn` | `stablehlo.fft RFFT` | N-D real-to-complex FFT |
| `torch.fft.irfftn` | `stablehlo.fft IRFFT` | N-D complex-to-real FFT |
| `torch.fft.fftshift` | `custom_call @fftshift` | Shift zero-frequency to center |
| `torch.fft.ifftshift` | `custom_call @ifftshift` | Inverse of fftshift |
| `torch.fft.fftfreq` | `custom_call @fftfreq` | Generate DFT frequency bins |
| `torch.fft.rfftfreq` | `custom_call @rfftfreq` | Generate RFFT frequency bins |

### StableHLO FFT Syntax

```mlir
// 1D FFT
%result = stablehlo.fft %input, type = FFT, length = [16] : tensor<2x16xcomplex<f32>> -> tensor<2x16xcomplex<f32>>

// 2D FFT
%result = stablehlo.fft %input, type = FFT, length = [8, 8] : tensor<2x8x8xcomplex<f32>> -> tensor<2x8x8xcomplex<f32>>

// Real FFT (real input -> complex output)
%result = stablehlo.fft %input, type = RFFT, length = [16] : tensor<2x16xf32> -> tensor<2x9xcomplex<f32>>
```

### Test Models

| Model | Description |
|-------|-------------|
| `fft` / `ifft` | 1D complex FFT/IFFT |
| `rfft` / `irfft` | 1D real FFT/IRFFT |
| `hfft` / `ihfft` | Hermitian FFT |
| `fft2` / `ifft2` | 2D complex FFT/IFFT |
| `rfft2` / `irfft2` | 2D real FFT/IRFFT |
| `fftn` / `ifftn` | N-D complex FFT/IFFT |
| `rfftn` / `irfftn` | N-D real FFT/IRFFT |
| `fftshift` / `ifftshift` | Frequency domain shifting |

### Backend Implementation Notes

StableHLO's native FFT operation maps directly to:
- **NVIDIA**: cuFFT (highly optimized GPU FFT library)
- **AMD**: rocFFT
- **CPU**: FFTW or Intel MKL FFT
- **XLA**: Native XLA FFT lowering

The helper operations (fftshift, ifftshift, fftfreq, rfftfreq) use custom_call and can be implemented using standard array operations (roll/concatenate for shift, iota/divide for frequency).

## Scan/Cumulative Operations Support

### Overview

WarpForge supports scan (cumulative) operations via `stablehlo.custom_call`. StableHLO does not have native scan primitives, so backends must provide their own implementations. This is the standard approach used by JAX/XLA.

### Why custom_call?

Scan operations are inherently sequential in their naive form, but efficient parallel implementations (parallel prefix scan, Brent-Kung algorithm) are hardware-specific:

- **GPU**: thrust::inclusive_scan, CUB DeviceScan (NVIDIA), hipcub (AMD)
- **TPU**: HLO ReduceWindow with causal masking
- **CPU**: std::inclusive_scan (C++17), OpenMP parallel prefix

Using `custom_call` allows each backend to use its optimal implementation.

### Implemented Operations

| PyTorch Op | StableHLO Target | Description |
|------------|------------------|-------------|
| `torch.cumsum` | `custom_call @cumsum` | Cumulative sum |
| `torch.cumprod` | `custom_call @cumprod` | Cumulative product |
| `torch.cummax` | `custom_call @cummax` | Cumulative maximum |
| `torch.cummin` | `custom_call @cummin` | Cumulative minimum |
| `torch.logcumsumexp` | `custom_call @logcumsumexp` | Log of cumulative sum of exp |
| `torch.diff` | `custom_call @diff` | N-th discrete difference |

### Test Models

| Model | Description |
|-------|-------------|
| `cumsum` / `cumprod` | Cumulative sum/product along dim 1 |
| `cummax` / `cummin` | Cumulative max/min (returns values) |
| `logcumsumexp` | Numerically stable log-sum-exp scan |
| `diff` / `diff_n2` | First/second-order discrete difference |

### Backend Implementation Notes

**NVIDIA (CUDA)**:
```cpp
thrust::inclusive_scan(input, input + n, output, op);
// Or CUB for better performance
cub::DeviceScan::InclusiveScan(d_temp, temp_bytes, d_in, d_out, op, n);
```

**CPU**:
```cpp
std::inclusive_scan(input, input + n, output, op);  // C++17
```

## RNN/LSTM/GRU Operations Support

### Overview

WarpForge supports recurrent neural network operations via `stablehlo.custom_call`. StableHLO does not have native RNN operations - backends implement using optimized libraries (cuDNN on NVIDIA, MIOpen on AMD, oneDNN on CPU).

### Implemented Operations

| PyTorch Op | StableHLO Target | Description |
|------------|------------------|-------------|
| `nn.LSTM` | `custom_call @lstm` | Long Short-Term Memory |
| `nn.GRU` | `custom_call @gru` | Gated Recurrent Unit |
| `nn.RNN` | `custom_call @rnn` | Vanilla RNN (tanh/relu) |
| `nn.LSTMCell` | `custom_call @lstm_cell` | Single LSTM step |
| `nn.GRUCell` | `custom_call @gru_cell` | Single GRU step |
| `nn.RNNCell` | `custom_call @rnn_cell` | Single RNN step |
| `pack_padded_sequence` | `custom_call @pack_padded_sequence` | Pack variable-length sequences |
| `pad_packed_sequence` | `custom_call @pad_packed_sequence` | Unpack to padded tensor |

### Supported Variants

- **batch_first**: Input shape (batch, seq, features) vs (seq, batch, features)
- **bidirectional**: Forward and backward passes
- **multi-layer**: Stacked RNN layers (num_layers > 1)
- **with_hidden**: Explicit initial hidden state

### Test Models (17 models)

| Model | Description |
|-------|-------------|
| `lstm` | Basic LSTM (seq_first) |
| `lstm_batch_first` | LSTM with batch_first=True |
| `lstm_bidirectional` | Bidirectional LSTM |
| `lstm_multi_layer` | 3-layer stacked LSTM |
| `lstm_with_hidden` | LSTM with initial (h0, c0) |
| `gru` / `gru_batch_first` / `gru_bidirectional` / `gru_multi_layer` | GRU variants |
| `rnn_tanh` / `rnn_relu` / `rnn_bidirectional` | RNN variants |
| `lstm_cell` / `gru_cell` / `rnn_cell` | Single-step cells |

### Backend Implementation Notes

**NVIDIA (cuDNN)**:
```cpp
cudnnRNNForwardInference(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx,
                          cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy,
                          workspace, workSpaceSize);
```

**AMD (MIOpen)**:
```cpp
miopenRNNForwardInference(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx,
                           cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy,
                           workspace, workSpaceSize);
```

**CPU (oneDNN)**:
```cpp
dnnl::lstm_forward::primitive_desc lstm_pd(engine, prop_kind::forward_inference,
    direction::unidirectional_left2right, src_layer_md, src_iter_md, ...);
```
