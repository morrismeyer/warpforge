# StableHLO Operation Test Models
#
# This file contains minimal PyTorch nn.Module classes that exercise specific
# operations, producing corresponding StableHLO ops when traced.
#
# Each model is designed to be as simple as possible while still producing
# the target StableHLO operation.

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Elementwise Binary Operations
# =============================================================================

class AddOp(nn.Module):
    """Produces: stablehlo.add"""
    def forward(self, x, y):
        return x + y

class SubtractOp(nn.Module):
    """Produces: stablehlo.subtract"""
    def forward(self, x, y):
        return x - y

class MultiplyOp(nn.Module):
    """Produces: stablehlo.multiply"""
    def forward(self, x, y):
        return x * y

class DivideOp(nn.Module):
    """Produces: stablehlo.divide"""
    def forward(self, x, y):
        return x / y

class MaximumOp(nn.Module):
    """Produces: stablehlo.maximum"""
    def forward(self, x, y):
        return torch.maximum(x, y)

class MinimumOp(nn.Module):
    """Produces: stablehlo.minimum"""
    def forward(self, x, y):
        return torch.minimum(x, y)

class PowerOp(nn.Module):
    """Produces: stablehlo.power"""
    def forward(self, x, y):
        return torch.pow(x, y)

class RemainderOp(nn.Module):
    """Produces: stablehlo.remainder"""
    def forward(self, x, y):
        return torch.remainder(x, y)

# =============================================================================
# Elementwise Unary Operations
# =============================================================================

class NegateOp(nn.Module):
    """Produces: stablehlo.negate"""
    def forward(self, x):
        return -x

class AbsOp(nn.Module):
    """Produces: stablehlo.abs"""
    def forward(self, x):
        return torch.abs(x)

class SignOp(nn.Module):
    """Produces: stablehlo.sign"""
    def forward(self, x):
        return torch.sign(x)

class FloorOp(nn.Module):
    """Produces: stablehlo.floor"""
    def forward(self, x):
        return torch.floor(x)

class CeilOp(nn.Module):
    """Produces: stablehlo.ceil"""
    def forward(self, x):
        return torch.ceil(x)

class RoundOp(nn.Module):
    """Produces: stablehlo.round_nearest_even"""
    def forward(self, x):
        return torch.round(x)

# =============================================================================
# Transcendental Operations
# =============================================================================

class ExpOp(nn.Module):
    """Produces: stablehlo.exponential"""
    def forward(self, x):
        return torch.exp(x)

class Expm1Op(nn.Module):
    """Produces: stablehlo.exponential_minus_one"""
    def forward(self, x):
        return torch.expm1(x)

class LogOp(nn.Module):
    """Produces: stablehlo.log"""
    def forward(self, x):
        return torch.log(x)

class Log1pOp(nn.Module):
    """Produces: stablehlo.log_plus_one"""
    def forward(self, x):
        return torch.log1p(x)

class SqrtOp(nn.Module):
    """Produces: stablehlo.sqrt"""
    def forward(self, x):
        return torch.sqrt(x)

class RsqrtOp(nn.Module):
    """Produces: stablehlo.rsqrt"""
    def forward(self, x):
        return torch.rsqrt(x)

class SinOp(nn.Module):
    """Produces: stablehlo.sine"""
    def forward(self, x):
        return torch.sin(x)

class CosOp(nn.Module):
    """Produces: stablehlo.cosine"""
    def forward(self, x):
        return torch.cos(x)

class TanOp(nn.Module):
    """Produces: stablehlo.tan"""
    def forward(self, x):
        return torch.tan(x)

class TanhOp(nn.Module):
    """Produces: stablehlo.tanh"""
    def forward(self, x):
        return torch.tanh(x)

class AtanOp(nn.Module):
    """Produces: stablehlo.atan"""
    def forward(self, x):
        return torch.atan(x)

class AsinOp(nn.Module):
    """Produces: stablehlo.asin"""
    def forward(self, x):
        return torch.asin(x)

class AcosOp(nn.Module):
    """Produces: stablehlo.acos"""
    def forward(self, x):
        return torch.acos(x)

class AsinhOp(nn.Module):
    """Produces: asinh composite (log + sqrt + multiply + add)"""
    def forward(self, x):
        return torch.asinh(x)

class AcoshOp(nn.Module):
    """Produces: acosh composite (log + sqrt + multiply + subtract)"""
    def forward(self, x):
        return torch.acosh(x)

class AtanhOp(nn.Module):
    """Produces: atanh composite (log + divide + multiply)"""
    def forward(self, x):
        return torch.atanh(x)

class ErfOp(nn.Module):
    """Produces: stablehlo.erf"""
    def forward(self, x):
        return torch.erf(x)

class ErfcOp(nn.Module):
    """Produces: stablehlo.erf + subtract"""
    def forward(self, x):
        return torch.erfc(x)

class Exp2Op(nn.Module):
    """Produces: exp(x * ln(2))"""
    def forward(self, x):
        return torch.exp2(x)

class LgammaOp(nn.Module):
    """Produces: stablehlo.custom_call @lgamma"""
    def forward(self, x):
        return torch.lgamma(x)

class DigammaOp(nn.Module):
    """Produces: stablehlo.custom_call @digamma"""
    def forward(self, x):
        return torch.digamma(x)

class FmodOp(nn.Module):
    """Produces: stablehlo.remainder"""
    def forward(self, x, y):
        return torch.fmod(x, y)

class TrueDivideOp(nn.Module):
    """Produces: stablehlo.divide"""
    def forward(self, x, y):
        return torch.true_divide(x, y)

class LogicalXorOp(nn.Module):
    """Produces: stablehlo.xor"""
    def forward(self, x, y):
        return torch.logical_xor(x > 0, y > 0)

class SigmoidOp(nn.Module):
    """Produces: stablehlo.logistic"""
    def forward(self, x):
        return torch.sigmoid(x)

# =============================================================================
# Comparison Operations
# =============================================================================

class CompareEqOp(nn.Module):
    """Produces: stablehlo.compare (EQ)"""
    def forward(self, x, y):
        return (x == y).float()

class CompareNeOp(nn.Module):
    """Produces: stablehlo.compare (NE)"""
    def forward(self, x, y):
        return (x != y).float()

class CompareLtOp(nn.Module):
    """Produces: stablehlo.compare (LT)"""
    def forward(self, x, y):
        return (x < y).float()

class CompareLteOp(nn.Module):
    """Produces: stablehlo.compare (LE)"""
    def forward(self, x, y):
        return (x <= y).float()

class CompareGtOp(nn.Module):
    """Produces: stablehlo.compare (GT)"""
    def forward(self, x, y):
        return (x > y).float()

class CompareGteOp(nn.Module):
    """Produces: stablehlo.compare (GE)"""
    def forward(self, x, y):
        return (x >= y).float()

# =============================================================================
# Selection Operations
# =============================================================================

class SelectOp(nn.Module):
    """Produces: stablehlo.select"""
    def forward(self, cond, x, y):
        return torch.where(cond > 0, x, y)

class ClampOp(nn.Module):
    """Produces: stablehlo.clamp"""
    def forward(self, x):
        return torch.clamp(x, min=-1.0, max=1.0)

# =============================================================================
# Matrix Operations
# =============================================================================

class MatmulOp(nn.Module):
    """Produces: stablehlo.dot_general"""
    def forward(self, x, y):
        return torch.matmul(x, y)

class LinearOp(nn.Module):
    """Produces: stablehlo.dot_general + stablehlo.add"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(x)

class BatchMatmulOp(nn.Module):
    """Produces: stablehlo.dot_general with batch dimensions"""
    def forward(self, x, y):
        return torch.bmm(x, y)

class MmOp(nn.Module):
    """Produces: stablehlo.dot_general (2D matrix multiply)"""
    def forward(self, x, y):
        return torch.mm(x, y)

class MvOp(nn.Module):
    """Produces: stablehlo.dot_general (matrix-vector)"""
    def forward(self, mat, vec):
        return torch.mv(mat, vec)

class DotOp(nn.Module):
    """Produces: stablehlo.dot_general (dot product)"""
    def forward(self, x, y):
        return torch.dot(x, y)

class OuterOp(nn.Module):
    """Produces: stablehlo.dot_general (outer product)"""
    def forward(self, x, y):
        return torch.outer(x, y)

class AddmmOp(nn.Module):
    """Produces: stablehlo.dot_general + stablehlo.add"""
    def forward(self, bias, mat1, mat2):
        return torch.addmm(bias, mat1, mat2)

class BaddbmmOp(nn.Module):
    """Produces: stablehlo.dot_general + stablehlo.add (batched)"""
    def forward(self, input, batch1, batch2):
        return torch.baddbmm(input, batch1, batch2)

# =============================================================================
# Shape Operations
# =============================================================================

class ReshapeOp(nn.Module):
    """Produces: stablehlo.reshape"""
    def forward(self, x):
        return x.reshape(-1, 4)

class TransposeOp(nn.Module):
    """Produces: stablehlo.transpose"""
    def forward(self, x):
        return x.transpose(0, 1)

class PermuteOp(nn.Module):
    """Produces: stablehlo.transpose"""
    def forward(self, x):
        return x.permute(2, 0, 1)

class SqueezeOp(nn.Module):
    """Produces: stablehlo.reshape"""
    def forward(self, x):
        return x.squeeze(1)

class UnsqueezeOp(nn.Module):
    """Produces: stablehlo.reshape"""
    def forward(self, x):
        return x.unsqueeze(1)

class ConcatOp(nn.Module):
    """Produces: stablehlo.concatenate"""
    def forward(self, x, y):
        return torch.cat([x, y], dim=1)

class StackOp(nn.Module):
    """Produces: stablehlo.concatenate (after reshape)"""
    def forward(self, x, y):
        return torch.stack([x, y], dim=0)

class BroadcastOp(nn.Module):
    """Produces: stablehlo.broadcast_in_dim"""
    def forward(self, x):
        return x.expand(4, -1, -1)

class FlattenOp(nn.Module):
    """Produces: stablehlo.reshape"""
    def forward(self, x):
        return x.flatten(1)

class ViewOp(nn.Module):
    """Produces: stablehlo.reshape"""
    def forward(self, x):
        return x.view(-1, 4)

class FlipOp(nn.Module):
    """Produces: stablehlo.reverse"""
    def forward(self, x):
        return torch.flip(x, dims=[1])

class RollOp(nn.Module):
    """Produces: stablehlo.custom_call @roll"""
    def forward(self, x):
        return torch.roll(x, shifts=2, dims=1)

class SplitOp(nn.Module):
    """Produces: stablehlo.slice (multiple)"""
    def forward(self, x):
        parts = torch.split(x, 2, dim=1)
        return parts[0]

class ChunkOp(nn.Module):
    """Produces: stablehlo.slice (multiple)"""
    def forward(self, x):
        chunks = torch.chunk(x, 2, dim=1)
        return chunks[0]

class TileOp(nn.Module):
    """Produces: stablehlo.broadcast_in_dim"""
    def forward(self, x):
        return x.repeat(1, 2)

class NarrowOp(nn.Module):
    """Produces: stablehlo.slice"""
    def forward(self, x):
        return torch.narrow(x, dim=1, start=1, length=3)

class UnbindOp(nn.Module):
    """Produces: stablehlo.custom_call @unbind"""
    def forward(self, x):
        return torch.unbind(x, dim=0)[0]

class SelectDimOp(nn.Module):
    """Produces: stablehlo.slice (for torch.select)"""
    def forward(self, x):
        return torch.select(x, dim=1, index=2)

class MovedimOp(nn.Module):
    """Produces: stablehlo.transpose"""
    def forward(self, x):
        return torch.movedim(x, 0, 2)

class SwapaxesOp(nn.Module):
    """Produces: stablehlo.transpose"""
    def forward(self, x):
        return torch.swapaxes(x, 0, 1)

# =============================================================================
# Slicing and Indexing Operations
# =============================================================================

class SliceOp(nn.Module):
    """Produces: stablehlo.slice"""
    def forward(self, x):
        return x[:, 1:3]

class SliceWithStepOp(nn.Module):
    """Produces: stablehlo.slice with strides"""
    def forward(self, x):
        return x[:, ::2]

class GatherOp(nn.Module):
    """Produces: stablehlo.gather"""
    def forward(self, x, indices):
        return torch.gather(x, dim=1, index=indices)

class IndexSelectOp(nn.Module):
    """Produces: stablehlo.gather"""
    def forward(self, x, indices):
        return torch.index_select(x, dim=1, index=indices)

class ScatterOp(nn.Module):
    """Produces: stablehlo.scatter"""
    def forward(self, x, indices, src):
        return torch.scatter(x, dim=1, index=indices, src=src)

class MaskedFillOp(nn.Module):
    """Produces: stablehlo.select"""
    def forward(self, x, mask):
        return x.masked_fill(mask > 0, 0.0)

class MaskedSelectOp(nn.Module):
    """Produces: stablehlo.custom_call @masked_select"""
    def forward(self, x, mask):
        return torch.masked_select(x, mask > 0)

class IndexCopyOp(nn.Module):
    """Produces: stablehlo.scatter"""
    def forward(self, x, indices, src):
        return x.index_copy(1, indices, src)

class IndexAddOp(nn.Module):
    """Produces: stablehlo.custom_call @index_add"""
    def forward(self, x, indices, src):
        return x.index_add(1, indices, src)

class IndexFillOp(nn.Module):
    """Produces: stablehlo.scatter"""
    def forward(self, x, indices):
        return x.index_fill(1, indices, 0.0)

# =============================================================================
# Reduction Operations
# =============================================================================

class SumReduceOp(nn.Module):
    """Produces: stablehlo.reduce (sum)"""
    def forward(self, x):
        return torch.sum(x, dim=1)

class MeanReduceOp(nn.Module):
    """Produces: stablehlo.reduce (sum) + stablehlo.divide"""
    def forward(self, x):
        return torch.mean(x, dim=1)

class MaxReduceOp(nn.Module):
    """Produces: stablehlo.reduce (max)"""
    def forward(self, x):
        return torch.max(x, dim=1)[0]

class MinReduceOp(nn.Module):
    """Produces: stablehlo.reduce (min)"""
    def forward(self, x):
        return torch.min(x, dim=1)[0]

class ProdReduceOp(nn.Module):
    """Produces: stablehlo.reduce (prod)"""
    def forward(self, x):
        return torch.prod(x, dim=1)

class SumAllReduceOp(nn.Module):
    """Produces: stablehlo.reduce over all dims"""
    def forward(self, x):
        return torch.sum(x)

class StdReduceOp(nn.Module):
    """Produces: stablehlo.custom_call @std"""
    def forward(self, x):
        return torch.std(x, dim=1)

class VarReduceOp(nn.Module):
    """Produces: stablehlo.custom_call @var"""
    def forward(self, x):
        return torch.var(x, dim=1)

class ArgmaxOp(nn.Module):
    """Produces: stablehlo.custom_call @argmax"""
    def forward(self, x):
        return torch.argmax(x, dim=1)

class ArgminOp(nn.Module):
    """Produces: stablehlo.custom_call @argmin"""
    def forward(self, x):
        return torch.argmin(x, dim=1)

# =============================================================================
# Activation Functions (composed operations)
# =============================================================================

class ReLUOp(nn.Module):
    """Produces: stablehlo.maximum (with zero)"""
    def forward(self, x):
        return F.relu(x)

class LeakyReLUOp(nn.Module):
    """Produces: stablehlo.select + stablehlo.multiply"""
    def forward(self, x):
        return F.leaky_relu(x, negative_slope=0.01)

class SoftmaxOp(nn.Module):
    """Produces: stablehlo.exponential + stablehlo.reduce + stablehlo.divide"""
    def forward(self, x):
        return F.softmax(x, dim=-1)

class LogSoftmaxOp(nn.Module):
    """Produces: log(softmax) operations"""
    def forward(self, x):
        return F.log_softmax(x, dim=-1)

class GeluOp(nn.Module):
    """Produces: GELU approximation operations"""
    def forward(self, x):
        return F.gelu(x)

class SiluOp(nn.Module):
    """Produces: stablehlo.logistic + stablehlo.multiply (x * sigmoid(x))"""
    def forward(self, x):
        return F.silu(x)

class EluOp(nn.Module):
    """Produces: stablehlo.select + stablehlo.exponential + stablehlo.subtract"""
    def forward(self, x):
        return F.elu(x, alpha=1.0)

class MishOp(nn.Module):
    """Produces: stablehlo.multiply + stablehlo.tanh + stablehlo.log + stablehlo.add + stablehlo.exponential"""
    def forward(self, x):
        return F.mish(x)

class SoftplusOp(nn.Module):
    """Produces: stablehlo.log + stablehlo.add + stablehlo.exponential"""
    def forward(self, x):
        return F.softplus(x)

class SoftsignOp(nn.Module):
    """Produces: stablehlo.divide + stablehlo.add + stablehlo.abs"""
    def forward(self, x):
        return F.softsign(x)

class HardtanhOp(nn.Module):
    """Produces: stablehlo.clamp"""
    def forward(self, x):
        return F.hardtanh(x, min_val=-1.0, max_val=1.0)

class Relu6Op(nn.Module):
    """Produces: stablehlo.clamp (0 to 6)"""
    def forward(self, x):
        return F.relu6(x)

class HardsigmoidOp(nn.Module):
    """Produces: stablehlo.clamp + stablehlo.add + stablehlo.divide"""
    def forward(self, x):
        return F.hardsigmoid(x)

class HardswishOp(nn.Module):
    """Produces: stablehlo.multiply + stablehlo.clamp + stablehlo.add + stablehlo.divide"""
    def forward(self, x):
        return F.hardswish(x)

class SeluOp(nn.Module):
    """Produces: stablehlo.multiply + stablehlo.select + stablehlo.exponential"""
    def forward(self, x):
        return F.selu(x)

class PreluOp(nn.Module):
    """Produces: stablehlo.select + stablehlo.multiply"""
    def __init__(self):
        super().__init__()
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(x)

class CeluOp(nn.Module):
    """Produces: stablehlo.select + stablehlo.exponential + stablehlo.divide"""
    def forward(self, x):
        return F.celu(x, alpha=1.0)

class LogsigmoidOp(nn.Module):
    """Produces: stablehlo.negate + stablehlo.exponential + stablehlo.log"""
    def forward(self, x):
        return F.logsigmoid(x)

# =============================================================================
# Additional Reduction Operations
# =============================================================================

class AllReduceOp(nn.Module):
    """Produces: stablehlo.reduce (all/and)"""
    def forward(self, x):
        return torch.all(x > 0, dim=1)

class AnyReduceOp(nn.Module):
    """Produces: stablehlo.reduce (any/or)"""
    def forward(self, x):
        return torch.any(x > 0, dim=1)

class LogsumexpOp(nn.Module):
    """Produces: stablehlo.log + stablehlo.reduce + stablehlo.exponential"""
    def forward(self, x):
        return torch.logsumexp(x, dim=1)

class CumsumOp(nn.Module):
    """Produces: stablehlo.custom_call @cumsum"""
    def forward(self, x):
        return torch.cumsum(x, dim=1)

class CumprodOp(nn.Module):
    """Produces: stablehlo.custom_call @cumprod"""
    def forward(self, x):
        return torch.cumprod(x, dim=1)


class CummaxOp(nn.Module):
    """Cumulative maximum along a dimension.
    Produces: stablehlo.custom_call @cummax
    """
    def forward(self, x):
        values, indices = torch.cummax(x, dim=1)
        return values


class CumminOp(nn.Module):
    """Cumulative minimum along a dimension.
    Produces: stablehlo.custom_call @cummin
    """
    def forward(self, x):
        values, indices = torch.cummin(x, dim=1)
        return values


class LogcumsumexpOp(nn.Module):
    """Log of cumulative sum of exponentials (numerically stable).
    Produces: stablehlo.custom_call @logcumsumexp
    """
    def forward(self, x):
        return torch.logcumsumexp(x, dim=1)


class DiffOp(nn.Module):
    """Discrete difference along a dimension.
    Produces: stablehlo.custom_call @diff
    """
    def forward(self, x):
        return torch.diff(x, dim=1)


class DiffN2Op(nn.Module):
    """Second-order discrete difference.
    Produces: stablehlo.custom_call @diff
    """
    def forward(self, x):
        return torch.diff(x, n=2, dim=1)


# =============================================================================
# Type Conversion Operations
# =============================================================================

class ToFloatOp(nn.Module):
    """Produces: stablehlo.convert (to float)"""
    def forward(self, x):
        return x.float()

class ToHalfOp(nn.Module):
    """Produces: stablehlo.convert (to half/float16)"""
    def forward(self, x):
        return x.half()

class ToDoubleOp(nn.Module):
    """Produces: stablehlo.convert (to double/float64)"""
    def forward(self, x):
        return x.double()

class ToIntOp(nn.Module):
    """Produces: stablehlo.convert (to int32)"""
    def forward(self, x):
        return x.int()

class ToBoolOp(nn.Module):
    """Produces: stablehlo.convert (to bool)"""
    def forward(self, x):
        return x.bool()

# =============================================================================
# Convolution Operations
# =============================================================================

class Conv1dOp(nn.Module):
    """Produces: stablehlo.convolution (1D)"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 4, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

class Conv2dOp(nn.Module):
    """Produces: stablehlo.convolution (2D)"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

class Conv2dStridedOp(nn.Module):
    """Produces: stablehlo.convolution with strides"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class DepthwiseConv2dOp(nn.Module):
    """Produces: stablehlo.convolution (depthwise)"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4)

    def forward(self, x):
        return self.conv(x)

class Conv3dOp(nn.Module):
    """Produces: stablehlo.convolution (3D)"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 4, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

class ConvTranspose1dOp(nn.Module):
    """Produces: stablehlo.convolution (transpose 1D)"""
    def __init__(self):
        super().__init__()
        self.conv = nn.ConvTranspose1d(4, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

class ConvTranspose2dOp(nn.Module):
    """Produces: stablehlo.convolution (transpose 2D)"""
    def __init__(self):
        super().__init__()
        self.conv = nn.ConvTranspose2d(16, 3, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

class ConvTranspose3dOp(nn.Module):
    """Produces: stablehlo.convolution (transpose 3D)"""
    def __init__(self):
        super().__init__()
        self.conv = nn.ConvTranspose3d(4, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

# =============================================================================
# Pooling Operations
# =============================================================================

class MaxPool2dOp(nn.Module):
    """Produces: stablehlo.reduce_window (max)"""
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)

class AvgPool2dOp(nn.Module):
    """Produces: stablehlo.reduce_window (sum) + divide"""
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)

class AdaptiveAvgPool2dOp(nn.Module):
    """Produces: adaptive pooling operations"""
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        return self.pool(x)

class MaxPool1dOp(nn.Module):
    """Produces: stablehlo.reduce_window (max) 1D"""
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)

class AvgPool1dOp(nn.Module):
    """Produces: stablehlo.reduce_window (avg) 1D"""
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)

class AdaptiveMaxPool1dOp(nn.Module):
    """Produces: stablehlo.custom_call @adaptive_max_pool1d"""
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool1d(4)

    def forward(self, x):
        return self.pool(x)

class AdaptiveMaxPool2dOp(nn.Module):
    """Produces: stablehlo.custom_call @adaptive_max_pool2d"""
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        return self.pool(x)

# =============================================================================
# Normalization Operations
# =============================================================================

class BatchNorm1dOp(nn.Module):
    """Produces: stablehlo.batch_norm_inference"""
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(8)
        self.bn.eval()  # Use inference mode

    def forward(self, x):
        return self.bn(x)

class BatchNorm2dOp(nn.Module):
    """Produces: stablehlo.batch_norm_inference (2D)"""
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(4)
        self.bn.eval()

    def forward(self, x):
        return self.bn(x)

class LayerNormOp(nn.Module):
    """Produces: layer norm operations"""
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm([8])

    def forward(self, x):
        return self.ln(x)

class InstanceNorm2dOp(nn.Module):
    """Produces: stablehlo.custom_call @instance_norm"""
    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm2d(4)

    def forward(self, x):
        return self.norm(x)

class GroupNormOp(nn.Module):
    """Produces: stablehlo.custom_call @group_norm"""
    def __init__(self):
        super().__init__()
        self.norm = nn.GroupNorm(2, 8)

    def forward(self, x):
        return self.norm(x)

# =============================================================================
# Advanced Matrix Operations
# =============================================================================

class EinsumOp(nn.Module):
    """Produces: stablehlo.custom_call @einsum"""
    def forward(self, x, y):
        return torch.einsum('ij,jk->ik', x, y)

class TensordotOp(nn.Module):
    """Produces: stablehlo.custom_call @tensordot"""
    def forward(self, x, y):
        return torch.tensordot(x, y, dims=1)

# =============================================================================
# Padding Operations
# =============================================================================

class PadConstantOp(nn.Module):
    """Produces: stablehlo.pad"""
    def forward(self, x):
        return F.pad(x, (1, 1, 1, 1), mode='constant', value=0)

class PadReflectOp(nn.Module):
    """Produces: pad with reflection"""
    def forward(self, x):
        return F.pad(x, (1, 1, 1, 1), mode='reflect')

class PadReplicateOp(nn.Module):
    """Produces: pad with edge replication"""
    def forward(self, x):
        return F.pad(x, (1, 1, 1, 1), mode='replicate')

class PadCircularOp(nn.Module):
    """Produces: stablehlo.custom_call @circular_pad"""
    def forward(self, x):
        return F.pad(x, (1, 1, 1, 1), mode='circular')

# =============================================================================
# Type Conversion Operations
# =============================================================================

class ConvertToFloat16Op(nn.Module):
    """Produces: stablehlo.convert (to f16)"""
    def forward(self, x):
        return x.half()

class ConvertToFloat64Op(nn.Module):
    """Produces: stablehlo.convert (to f64)"""
    def forward(self, x):
        return x.double()

class ConvertToIntOp(nn.Module):
    """Produces: stablehlo.convert (to int)"""
    def forward(self, x):
        return x.int()

# =============================================================================
# Constant Operations
# =============================================================================

class ConstantOp(nn.Module):
    """Produces: stablehlo.constant"""
    def forward(self, x):
        ones = torch.ones_like(x)
        return x + ones

class ZerosLikeOp(nn.Module):
    """Produces: stablehlo.constant (zeros)"""
    def forward(self, x):
        return torch.zeros_like(x)

class OnesLikeOp(nn.Module):
    """Produces: stablehlo.constant (ones)"""
    def forward(self, x):
        return torch.ones_like(x)

class FullLikeOp(nn.Module):
    """Produces: stablehlo.constant (filled)"""
    def forward(self, x):
        return torch.full_like(x, 3.14)

# =============================================================================
# Iota (Range) Operations
# =============================================================================

class ArangeOp(nn.Module):
    """Produces: stablehlo.iota"""
    def forward(self, x):
        indices = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
        return x + indices

# =============================================================================
# Backward / Gradient Operations (Training Support)
# =============================================================================

class ReluBackwardOp(nn.Module):
    """relu_backward: grad * (input > 0)"""
    def forward(self, grad, x):
        return grad * (x > 0).float()

class SigmoidBackwardOp(nn.Module):
    """sigmoid_backward: grad * out * (1 - out)"""
    def forward(self, grad, out):
        return grad * out * (1 - out)

class TanhBackwardOp(nn.Module):
    """tanh_backward: grad * (1 - out^2)"""
    def forward(self, grad, out):
        return grad * (1 - out * out)

class ExpBackwardOp(nn.Module):
    """exp_backward: grad * exp_out"""
    def forward(self, grad, out):
        return grad * out

class LogBackwardOp(nn.Module):
    """log_backward: grad / x"""
    def forward(self, grad, x):
        return grad / x

class SqrtBackwardOp(nn.Module):
    """sqrt_backward: grad / (2 * sqrt_out)"""
    def forward(self, grad, out):
        return grad / (2 * out)

class MulBackwardOp(nn.Module):
    """mul_backward: grad * other"""
    def forward(self, grad, other):
        return grad * other

class DivBackwardOp(nn.Module):
    """div_backward: grad / denom"""
    def forward(self, grad, denom):
        return grad / denom

class NegBackwardOp(nn.Module):
    """neg_backward: -grad"""
    def forward(self, grad):
        return -grad

class AbsBackwardOp(nn.Module):
    """abs_backward: grad * sign(x)"""
    def forward(self, grad, x):
        return grad * torch.sign(x)

class MatmulBackwardOp(nn.Module):
    """matmul_backward: grad @ B^T"""
    def forward(self, grad, other):
        return torch.matmul(grad, other.transpose(-2, -1))

class SumBackwardOp(nn.Module):
    """sum_backward: broadcast grad"""
    def forward(self, grad, template):
        return grad.expand_as(template)

class TransposeBackwardOp(nn.Module):
    """transpose_backward: reverse transpose"""
    def forward(self, grad):
        return grad.transpose(-2, -1)

class MSELossBackwardOp(nn.Module):
    """mse_loss_backward: 2 * (pred - target)"""
    def forward(self, grad, pred, target):
        return grad * 2 * (pred - target) / pred.numel()

class L1LossBackwardOp(nn.Module):
    """l1_loss_backward: sign(pred - target)"""
    def forward(self, grad, pred, target):
        return grad * torch.sign(pred - target)

# =============================================================================
# Complex Multi-Op Models (for integration testing)
# =============================================================================

class SimpleMLP(nn.Module):
    """Two-layer MLP with ReLU"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SimpleConvNet(nn.Module):
    """Simple ConvNet for image-like input"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        return self.pool(x)

class AttentionBlock(nn.Module):
    """Simplified attention mechanism"""
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(8, 8)
        self.key = nn.Linear(8, 8)
        self.value = nn.Linear(8, 8)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (8 ** 0.5)
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

class ResidualBlock(nn.Module):
    """Residual connection pattern"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x + residual

# =============================================================================
# Dynamic Shape Test Models
# =============================================================================
# Models for testing dynamic dimension support (batch size, sequence length)

class DynamicBatchMLP(nn.Module):
    """MLP that works with variable batch size.
    Tests: dynamic batch dim propagation through linear layers.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        # x: [batch, 8] -> [batch, 4]
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DynamicBatchConv(nn.Module):
    """Conv2d that works with variable batch size.
    Tests: dynamic batch dim through convolution.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)

    def forward(self, x):
        # x: [batch, 3, H, W] -> [batch, 16, H, W]
        return F.relu(self.conv(x))


class DynamicSeqTransformerBlock(nn.Module):
    """Simplified transformer block with dynamic sequence length.
    Tests: dynamic seq dim through attention and linear layers.
    """
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(8, 8)
        self.ff = nn.Linear(8, 8)

    def forward(self, x):
        # x: [batch, seq, 8] -> [batch, seq, 8]
        attn_out = self.attn(x)
        return F.relu(self.ff(attn_out))


class DynamicReshape(nn.Module):
    """Test dynamic reshape operation.
    Tests: dynamic_reshape emission when batch dim is dynamic.
    """
    def forward(self, x):
        # x: [batch, 4, 4] -> [batch, 16]
        batch = x.shape[0]
        return x.view(batch, -1)


class DynamicMatmul(nn.Module):
    """Test matmul with dynamic batch dimension.
    Tests: batch dim preservation through matmul.
    """
    def forward(self, x, y):
        # x: [batch, 4, 8], y: [batch, 8, 4] -> [batch, 4, 4]
        return torch.matmul(x, y)


class DynamicReduction(nn.Module):
    """Test reduction with dynamic batch dimension.
    Tests: batch dim preservation when reducing other dims.
    """
    def forward(self, x):
        # x: [batch, 4, 8] -> [batch, 4] (reduce last dim)
        return x.sum(dim=-1)


class DynamicBroadcast(nn.Module):
    """Test broadcasting with dynamic batch dimension.
    Tests: broadcast_in_dim with dynamic dims.
    """
    def forward(self, x, scale):
        # x: [batch, 8], scale: [8] -> [batch, 8]
        return x * scale


class DynamicTranspose(nn.Module):
    """Test transpose with dynamic dimensions.
    Tests: dimension index remapping for dynamic dims.
    """
    def forward(self, x):
        # x: [batch, seq, 8] -> [batch, 8, seq]
        return x.transpose(1, 2)


class DynamicSlice(nn.Module):
    """Test slicing with dynamic batch dimension.
    Tests: dynamic_slice emission for narrow operation.
    """
    def forward(self, x):
        # x: [batch, 8, 16] -> [batch, 4, 16] (narrow dim 1)
        return x.narrow(1, 2, 4)


class DynamicSelect(nn.Module):
    """Test index selection with dynamic batch dimension.
    Tests: dynamic_slice emission for select operation.
    """
    def forward(self, x):
        # x: [batch, 8, 16] -> [batch, 16] (select index 3 from dim 1)
        return x.select(1, 3)


class DynamicPad(nn.Module):
    """Test padding with dynamic batch dimension.
    Tests: dynamic_pad emission.
    """
    def forward(self, x):
        # x: [batch, 8] -> [batch, 12] (pad 2 on each side of last dim)
        return F.pad(x, (2, 2))


class DynamicGather(nn.Module):
    """Test gather with dynamic batch dimension.
    Tests: dynamic_gather emission.
    """
    def forward(self, x, indices):
        # x: [batch, 8, 4], indices: [batch, 8, 2] -> [batch, 8, 2]
        return torch.gather(x, 2, indices)


class DynamicExpand(nn.Module):
    """Test expand/broadcast with dynamic batch dimension.
    Tests: dynamic_broadcast_in_dim emission.
    """
    def forward(self, x):
        # x: [batch, 1, 8] -> [batch, 4, 8] (expand dim 1)
        return x.expand(-1, 4, -1)


class DynamicIndexSelect(nn.Module):
    """Test index_select with dynamic batch dimension.
    Tests: dynamic_gather emission for index_select.
    """
    def forward(self, x, indices):
        # x: [batch, 8, 16], indices: [3] -> [batch, 3, 16]
        return torch.index_select(x, 1, indices)


# =============================================================================
# Quantization Test Models
# =============================================================================
# Models for testing INT8/INT4 quantization support
# These map to StableHLO uniform_quantize/dequantize and custom_call ops
# which Babylon can lower to ONNX quantization operators

class QuantizePerTensorOp(nn.Module):
    """Test quantize_per_tensor operation.
    Converts float tensor to quantized INT8 tensor.
    Maps to: stablehlo.uniform_quantize
    Babylon target: ONNX QuantizeLinear
    """
    def forward(self, x):
        # Quantize to INT8 with scale=0.1, zero_point=0
        return torch.quantize_per_tensor(x, scale=0.1, zero_point=0, dtype=torch.qint8)


class DequantizeOp(nn.Module):
    """Test dequantize operation.
    Converts quantized tensor back to float.
    Maps to: stablehlo.uniform_dequantize
    Babylon target: ONNX DequantizeLinear
    """
    def forward(self, x):
        # x is a quantized tensor, dequantize to float
        return x.dequantize()


class FakeQuantizePerTensorOp(nn.Module):
    """Test fake_quantize_per_tensor_affine for QAT.
    Simulates quantization during training.
    Maps to: clamp + round + scale sequence in StableHLO
    """
    def forward(self, x):
        # Fake quantize with 8-bit range
        return torch.fake_quantize_per_tensor_affine(
            x, scale=0.1, zero_point=0, quant_min=-128, quant_max=127
        )


class QuantizedLinearModel(nn.Module):
    """Test quantized linear layer.
    Maps to: stablehlo.custom_call @quantized_linear
    Babylon target: ONNX QLinearMatMul
    """
    def __init__(self):
        super().__init__()
        # Create a float linear layer that we'll quantize
        self.linear = nn.Linear(8, 16)

    def forward(self, x):
        return self.linear(x)


class QuantizedConv2dModel(nn.Module):
    """Test quantized Conv2d layer.
    Maps to: stablehlo.custom_call @quantized_conv2d
    Babylon target: ONNX QLinearConv
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class QuantizedReluOp(nn.Module):
    """Test quantized ReLU operation.
    Maps to: stablehlo.maximum with zero
    """
    def forward(self, x):
        return F.relu(x)


class QuantizedAddOp(nn.Module):
    """Test quantized addition.
    Maps to: stablehlo.custom_call @quantized_add
    """
    def forward(self, x, y):
        return x + y


class QuantizedMulOp(nn.Module):
    """Test quantized multiplication.
    Maps to: stablehlo.custom_call @quantized_mul
    """
    def forward(self, x, y):
        return x * y


class Int8LinearModel(nn.Module):
    """Full INT8 quantized inference model.
    Demonstrates the quantize -> compute -> dequantize pattern.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        # In real quantized inference:
        # 1. Input is quantized
        # 2. Compute in INT8
        # 3. Output is dequantized
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Int4LinearModel(nn.Module):
    """INT4 quantized model for ultra-low precision inference.
    Used for LLM weight compression (e.g., GPTQ, AWQ).
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 16)

    def forward(self, x):
        return self.fc(x)


# =============================================================================
# Sparse Tensor Operations
# =============================================================================

class SparseCooTensorOp(nn.Module):
    """Creates a sparse COO tensor from indices and values.
    Produces: stablehlo.custom_call @sparse_coo_tensor
    """
    def forward(self, indices, values):
        # indices: [2, nnz], values: [nnz]
        return torch.sparse_coo_tensor(indices, values, size=(4, 4))


class SparseCsrTensorOp(nn.Module):
    """Creates a sparse CSR tensor from crow_indices, col_indices, and values.
    Produces: stablehlo.custom_call @sparse_csr_tensor
    """
    def forward(self, crow_indices, col_indices, values):
        return torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(4, 4))


class ToDenseOp(nn.Module):
    """Converts a sparse tensor to dense.
    Produces: stablehlo.custom_call @sparse_to_dense
    """
    def forward(self, x):
        # x is sparse, convert to dense
        return x.to_dense()


class ToSparseCooOp(nn.Module):
    """Converts a dense tensor to sparse COO format.
    Produces: stablehlo.custom_call @to_sparse_coo
    """
    def forward(self, x):
        return x.to_sparse()


class ToSparseCsrOp(nn.Module):
    """Converts a dense tensor to sparse CSR format.
    Produces: stablehlo.custom_call @to_sparse_csr
    """
    def forward(self, x):
        return x.to_sparse_csr()


class SparseCoalesceOp(nn.Module):
    """Coalesces duplicate indices in a sparse COO tensor.
    Produces: stablehlo.custom_call @sparse_coalesce
    """
    def forward(self, x):
        return x.coalesce()


class SparseMmOp(nn.Module):
    """Sparse matrix multiplication (sparse @ dense).
    Produces: stablehlo.custom_call @sparse_mm
    """
    def forward(self, sparse, dense):
        return torch.sparse.mm(sparse, dense)


class SparseAddmmOp(nn.Module):
    """Sparse addmm: beta * input + alpha * (sparse @ dense).
    Produces: stablehlo.custom_call @sparse_addmm
    """
    def forward(self, input_mat, sparse, dense):
        return torch.sparse.addmm(input_mat, sparse, dense, beta=1.0, alpha=1.0)


class SparseSumOp(nn.Module):
    """Sum reduction on sparse tensor.
    Produces: stablehlo.custom_call @sparse_sum
    """
    def forward(self, x):
        return torch.sparse.sum(x)


class SparseSoftmaxOp(nn.Module):
    """Softmax over non-zero elements of sparse tensor.
    Produces: stablehlo.custom_call @sparse_softmax
    """
    def forward(self, x):
        return torch.sparse.softmax(x, dim=-1)


class SparseIndicesOp(nn.Module):
    """Get indices from sparse COO tensor.
    Produces: stablehlo.custom_call @sparse_indices
    """
    def forward(self, x):
        return x.indices()


class SparseValuesOp(nn.Module):
    """Get values from sparse tensor.
    Produces: stablehlo.custom_call @sparse_values
    """
    def forward(self, x):
        return x.values()


class SparseNnzOp(nn.Module):
    """Get number of non-zero elements.
    Produces: stablehlo.custom_call @sparse_nnz
    """
    def forward(self, x):
        return x._nnz()


class ToSparseBsrOp(nn.Module):
    """Converts a dense tensor to sparse BSR format with block size.
    Produces: stablehlo.custom_call @to_sparse_bsr
    """
    def forward(self, x):
        return x.to_sparse_bsr(blocksize=(2, 2))


class SemiStructuredSparseOp(nn.Module):
    """Convert to semi-structured (2:4) sparsity for NVIDIA accelerators.
    Produces: stablehlo.custom_call @to_sparse_semi_structured

    Note: Semi-structured sparsity requires specific shapes (M x K where K % 16 == 0)
    and is primarily used with NVIDIA Ampere/Hopper Sparse Tensor Cores.
    """
    def __init__(self):
        super().__init__()
        # Linear layer with specific dimensions for 2:4 sparsity
        self.fc = nn.Linear(16, 32)

    def forward(self, x):
        # In a real scenario, weight would be pruned to 2:4 pattern
        return self.fc(x)


class SparseMaskedMmOp(nn.Module):
    """Sparse masked matrix multiplication (for attention).
    Produces: stablehlo.custom_call @sparse_sampled_addmm
    """
    def forward(self, mask, mat1, mat2):
        return torch.sparse.sampled_addmm(mask, mat1, mat2)


# =============================================================================
# Complex Tensor Operations
# =============================================================================

class ComplexOp(nn.Module):
    """Creates a complex tensor from real and imaginary parts."""
    def forward(self, real, imag):
        return torch.complex(real, imag)


class RealOp(nn.Module):
    """Extracts the real part of a complex tensor."""
    def forward(self, x):
        return torch.real(x)


class ImagOp(nn.Module):
    """Extracts the imaginary part of a complex tensor."""
    def forward(self, x):
        return torch.imag(x)


class ConjOp(nn.Module):
    """Computes the complex conjugate."""
    def forward(self, x):
        return torch.conj(x)


class ViewAsRealOp(nn.Module):
    """Views complex tensor as real tensor with trailing dim of size 2."""
    def forward(self, x):
        return torch.view_as_real(x)


class ViewAsComplexOp(nn.Module):
    """Views real tensor with trailing dim of size 2 as complex tensor."""
    def forward(self, x):
        return torch.view_as_complex(x)


class AngleOp(nn.Module):
    """Computes the angle (phase) of complex numbers."""
    def forward(self, x):
        return torch.angle(x)


class PolarOp(nn.Module):
    """Creates complex tensor from polar coordinates (abs, angle)."""
    def forward(self, abs_val, angle):
        return torch.polar(abs_val, angle)


class ComplexAbsOp(nn.Module):
    """Computes the absolute value (modulus) of complex numbers."""
    def forward(self, x):
        return torch.abs(x)


class ComplexMulOp(nn.Module):
    """Complex multiplication."""
    def forward(self, x, y):
        return x * y


class ComplexAddOp(nn.Module):
    """Complex addition."""
    def forward(self, x, y):
        return x + y


class ComplexMatmulOp(nn.Module):
    """Complex matrix multiplication."""
    def forward(self, x, y):
        return torch.matmul(x, y)


class ComplexExpOp(nn.Module):
    """Complex exponential."""
    def forward(self, x):
        return torch.exp(x)


class ComplexLogOp(nn.Module):
    """Complex logarithm."""
    def forward(self, x):
        return torch.log(x)


class ComplexSqrtOp(nn.Module):
    """Complex square root."""
    def forward(self, x):
        return torch.sqrt(x)


# =============================================================================
# FFT Operations
# =============================================================================

class FFTOp(nn.Module):
    """1D FFT (complex-to-complex).
    Produces: stablehlo.fft FFT
    """
    def forward(self, x):
        return torch.fft.fft(x)


class IFFTOp(nn.Module):
    """1D inverse FFT (complex-to-complex).
    Produces: stablehlo.fft IFFT
    """
    def forward(self, x):
        return torch.fft.ifft(x)


class RFFTOp(nn.Module):
    """1D real FFT (real-to-complex).
    Produces: stablehlo.fft RFFT
    """
    def forward(self, x):
        return torch.fft.rfft(x)


class IRFFTOp(nn.Module):
    """1D inverse real FFT (complex-to-real).
    Produces: stablehlo.fft IRFFT
    """
    def forward(self, x):
        return torch.fft.irfft(x)


class HFFTOp(nn.Module):
    """Hermitian FFT (for Hermitian-symmetric input).
    Produces: stablehlo.fft IRFFT
    """
    def forward(self, x):
        return torch.fft.hfft(x)


class IHFFTOp(nn.Module):
    """Inverse Hermitian FFT.
    Produces: stablehlo.fft RFFT
    """
    def forward(self, x):
        return torch.fft.ihfft(x)


class FFT2Op(nn.Module):
    """2D FFT (complex-to-complex).
    Produces: stablehlo.fft FFT with 2D lengths
    """
    def forward(self, x):
        return torch.fft.fft2(x)


class IFFT2Op(nn.Module):
    """2D inverse FFT (complex-to-complex).
    Produces: stablehlo.fft IFFT with 2D lengths
    """
    def forward(self, x):
        return torch.fft.ifft2(x)


class RFFT2Op(nn.Module):
    """2D real FFT (real-to-complex).
    Produces: stablehlo.fft RFFT with 2D lengths
    """
    def forward(self, x):
        return torch.fft.rfft2(x)


class IRFFT2Op(nn.Module):
    """2D inverse real FFT (complex-to-real).
    Produces: stablehlo.fft IRFFT with 2D lengths
    """
    def forward(self, x):
        return torch.fft.irfft2(x)


class FFTNOp(nn.Module):
    """N-dimensional FFT.
    Produces: stablehlo.fft FFT with N-D lengths
    """
    def forward(self, x):
        return torch.fft.fftn(x)


class IFFTNOp(nn.Module):
    """N-dimensional inverse FFT.
    Produces: stablehlo.fft IFFT with N-D lengths
    """
    def forward(self, x):
        return torch.fft.ifftn(x)


class RFFTNOp(nn.Module):
    """N-dimensional real FFT.
    Produces: stablehlo.fft RFFT with N-D lengths
    """
    def forward(self, x):
        return torch.fft.rfftn(x)


class IRFFTNOp(nn.Module):
    """N-dimensional inverse real FFT.
    Produces: stablehlo.fft IRFFT with N-D lengths
    """
    def forward(self, x):
        return torch.fft.irfftn(x)


class FFTShiftOp(nn.Module):
    """Shift zero-frequency component to center.
    Produces: stablehlo.custom_call @fftshift
    """
    def forward(self, x):
        return torch.fft.fftshift(x)


class IFFTShiftOp(nn.Module):
    """Inverse of fftshift.
    Produces: stablehlo.custom_call @ifftshift
    """
    def forward(self, x):
        return torch.fft.ifftshift(x)


# =============================================================================
# Operation Registry
# =============================================================================
# Maps operation names to (ModelClass, input_specs) tuples
# input_specs format: list of (shape, dtype) tuples
# For dynamic shape tests, we use a sample batch size of 2 but mark dim 0 as dynamic

OPERATION_REGISTRY = {
    # Elementwise Binary
    'add': (AddOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'subtract': (SubtractOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'multiply': (MultiplyOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'divide': (DivideOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'maximum': (MaximumOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'minimum': (MinimumOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'power': (PowerOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'remainder': (RemainderOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),

    # Elementwise Unary
    'negate': (NegateOp, [([1, 8], 'f32')]),
    'abs': (AbsOp, [([1, 8], 'f32')]),
    'sign': (SignOp, [([1, 8], 'f32')]),
    'floor': (FloorOp, [([1, 8], 'f32')]),
    'ceil': (CeilOp, [([1, 8], 'f32')]),
    'round': (RoundOp, [([1, 8], 'f32')]),

    # Transcendental
    'exp': (ExpOp, [([1, 8], 'f32')]),
    'expm1': (Expm1Op, [([1, 8], 'f32')]),
    'log': (LogOp, [([1, 8], 'f32')]),
    'log1p': (Log1pOp, [([1, 8], 'f32')]),
    'sqrt': (SqrtOp, [([1, 8], 'f32')]),
    'rsqrt': (RsqrtOp, [([1, 8], 'f32')]),
    'sin': (SinOp, [([1, 8], 'f32')]),
    'cos': (CosOp, [([1, 8], 'f32')]),
    'tan': (TanOp, [([1, 8], 'f32')]),
    'tanh': (TanhOp, [([1, 8], 'f32')]),
    'atan': (AtanOp, [([1, 8], 'f32')]),
    'sigmoid': (SigmoidOp, [([1, 8], 'f32')]),

    # Comparison
    'compare_eq': (CompareEqOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'compare_ne': (CompareNeOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'compare_lt': (CompareLtOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'compare_lte': (CompareLteOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'compare_gt': (CompareGtOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'compare_gte': (CompareGteOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),

    # Selection
    'select': (SelectOp, [([1, 8], 'f32'), ([1, 8], 'f32'), ([1, 8], 'f32')]),
    'clamp': (ClampOp, [([1, 8], 'f32')]),

    # Matrix
    'matmul': (MatmulOp, [([1, 4, 8], 'f32'), ([1, 8, 4], 'f32')]),
    'linear': (LinearOp, [([1, 8], 'f32')]),
    'batch_matmul': (BatchMatmulOp, [([2, 4, 8], 'f32'), ([2, 8, 4], 'f32')]),
    'mm': (MmOp, [([4, 8], 'f32'), ([8, 4], 'f32')]),
    'mv': (MvOp, [([4, 8], 'f32'), ([8], 'f32')]),
    'dot': (DotOp, [([8], 'f32'), ([8], 'f32')]),
    'outer': (OuterOp, [([4], 'f32'), ([8], 'f32')]),
    'addmm': (AddmmOp, [([4, 4], 'f32'), ([4, 8], 'f32'), ([8, 4], 'f32')]),
    'baddbmm': (BaddbmmOp, [([2, 4, 4], 'f32'), ([2, 4, 8], 'f32'), ([2, 8, 4], 'f32')]),

    # Shape
    'reshape': (ReshapeOp, [([2, 8], 'f32')]),
    'transpose': (TransposeOp, [([4, 8], 'f32')]),
    'permute': (PermuteOp, [([2, 4, 8], 'f32')]),
    'squeeze': (SqueezeOp, [([2, 1, 8], 'f32')]),
    'unsqueeze': (UnsqueezeOp, [([2, 8], 'f32')]),
    'concat': (ConcatOp, [([1, 4], 'f32'), ([1, 4], 'f32')]),
    'stack': (StackOp, [([4, 8], 'f32'), ([4, 8], 'f32')]),
    'broadcast': (BroadcastOp, [([1, 4, 8], 'f32')]),
    'flatten': (FlattenOp, [([2, 4, 8], 'f32')]),
    'view': (ViewOp, [([2, 8], 'f32')]),
    'flip': (FlipOp, [([2, 8], 'f32')]),
    'roll': (RollOp, [([2, 8], 'f32')]),
    'split': (SplitOp, [([2, 8], 'f32')]),
    'chunk': (ChunkOp, [([2, 8], 'f32')]),
    'tile': (TileOp, [([2, 4], 'f32')]),

    # Slicing
    'slice': (SliceOp, [([1, 8], 'f32')]),
    'slice_step': (SliceWithStepOp, [([1, 8], 'f32')]),
    'gather': (GatherOp, [([2, 8], 'f32'), ([2, 4], 'i64')]),
    'index_select': (IndexSelectOp, [([2, 8], 'f32'), ([4], 'i64')]),
    'scatter': (ScatterOp, [([2, 8], 'f32'), ([2, 4], 'i64'), ([2, 4], 'f32')]),
    'masked_fill': (MaskedFillOp, [([2, 8], 'f32'), ([2, 8], 'f32')]),
    'masked_select': (MaskedSelectOp, [([2, 8], 'f32'), ([2, 8], 'f32')]),

    # Reductions
    'sum_reduce': (SumReduceOp, [([2, 8], 'f32')]),
    'mean_reduce': (MeanReduceOp, [([2, 8], 'f32')]),
    'max_reduce': (MaxReduceOp, [([2, 8], 'f32')]),
    'min_reduce': (MinReduceOp, [([2, 8], 'f32')]),
    'prod_reduce': (ProdReduceOp, [([2, 8], 'f32')]),
    'sum_all_reduce': (SumAllReduceOp, [([2, 8], 'f32')]),

    # Activations
    'relu': (ReLUOp, [([1, 8], 'f32')]),
    'leaky_relu': (LeakyReLUOp, [([1, 8], 'f32')]),
    'softmax': (SoftmaxOp, [([1, 8], 'f32')]),
    'log_softmax': (LogSoftmaxOp, [([1, 8], 'f32')]),
    'gelu': (GeluOp, [([1, 8], 'f32')]),
    'silu': (SiluOp, [([1, 8], 'f32')]),
    'elu': (EluOp, [([1, 8], 'f32')]),
    'mish': (MishOp, [([1, 8], 'f32')]),
    'softplus': (SoftplusOp, [([1, 8], 'f32')]),
    'softsign': (SoftsignOp, [([1, 8], 'f32')]),
    'hardtanh': (HardtanhOp, [([1, 8], 'f32')]),
    'relu6': (Relu6Op, [([1, 8], 'f32')]),
    'hardsigmoid': (HardsigmoidOp, [([1, 8], 'f32')]),
    'hardswish': (HardswishOp, [([1, 8], 'f32')]),
    'selu': (SeluOp, [([1, 8], 'f32')]),
    'prelu': (PreluOp, [([1, 8], 'f32')]),

    # Additional Reductions
    'all_reduce': (AllReduceOp, [([2, 8], 'f32')]),
    'any_reduce': (AnyReduceOp, [([2, 8], 'f32')]),
    'logsumexp': (LogsumexpOp, [([2, 8], 'f32')]),
    'cumsum': (CumsumOp, [([2, 8], 'f32')]),
    'cumprod': (CumprodOp, [([2, 8], 'f32')]),
    'cummax': (CummaxOp, [([2, 8], 'f32')]),
    'cummin': (CumminOp, [([2, 8], 'f32')]),
    'logcumsumexp': (LogcumsumexpOp, [([2, 8], 'f32')]),
    'diff': (DiffOp, [([2, 8], 'f32')]),
    'diff_n2': (DiffN2Op, [([2, 10], 'f32')]),  # Need extra elements for n=2

    # Type Conversion
    'to_float': (ToFloatOp, [([1, 8], 'i32')]),
    'to_half': (ToHalfOp, [([1, 8], 'f32')]),
    'to_double': (ToDoubleOp, [([1, 8], 'f32')]),
    'to_int': (ToIntOp, [([1, 8], 'f32')]),
    'to_bool': (ToBoolOp, [([1, 8], 'f32')]),

    # Convolution
    'conv1d': (Conv1dOp, [([1, 1, 16], 'f32')]),
    'conv2d': (Conv2dOp, [([1, 3, 8, 8], 'f32')]),
    'conv2d_strided': (Conv2dStridedOp, [([1, 3, 8, 8], 'f32')]),
    'depthwise_conv2d': (DepthwiseConv2dOp, [([1, 4, 8, 8], 'f32')]),

    # Pooling
    'max_pool2d': (MaxPool2dOp, [([1, 4, 8, 8], 'f32')]),
    'avg_pool2d': (AvgPool2dOp, [([1, 4, 8, 8], 'f32')]),
    'adaptive_avg_pool2d': (AdaptiveAvgPool2dOp, [([1, 4, 8, 8], 'f32')]),

    # Normalization
    'batch_norm1d': (BatchNorm1dOp, [([2, 8], 'f32')]),
    'batch_norm2d': (BatchNorm2dOp, [([1, 4, 8, 8], 'f32')]),
    'layer_norm': (LayerNormOp, [([2, 8], 'f32')]),

    # Padding
    'pad_constant': (PadConstantOp, [([1, 1, 4, 4], 'f32')]),
    'pad_reflect': (PadReflectOp, [([1, 1, 4, 4], 'f32')]),
    'pad_replicate': (PadReplicateOp, [([1, 1, 4, 4], 'f32')]),

    # Constants
    'constant': (ConstantOp, [([1, 8], 'f32')]),
    'zeros_like': (ZerosLikeOp, [([1, 8], 'f32')]),
    'ones_like': (OnesLikeOp, [([1, 8], 'f32')]),
    'full_like': (FullLikeOp, [([1, 8], 'f32')]),

    # Iota
    'arange': (ArangeOp, [([1, 8], 'f32')]),

    # Inverse Trigonometric
    'asin': (AsinOp, [([1, 8], 'f32')]),
    'acos': (AcosOp, [([1, 8], 'f32')]),
    'asinh': (AsinhOp, [([1, 8], 'f32')]),
    'acosh': (AcoshOp, [([1, 8], 'f32')]),
    'atanh': (AtanhOp, [([1, 8], 'f32')]),

    # Special Functions
    'erf': (ErfOp, [([1, 8], 'f32')]),
    'erfc': (ErfcOp, [([1, 8], 'f32')]),
    'exp2': (Exp2Op, [([1, 8], 'f32')]),
    'lgamma': (LgammaOp, [([1, 8], 'f32')]),
    'digamma': (DigammaOp, [([1, 8], 'f32')]),

    # Additional Binary Operations
    'fmod': (FmodOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'true_divide': (TrueDivideOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'logical_xor': (LogicalXorOp, [([1, 8], 'bool'), ([1, 8], 'bool')]),

    # Additional Shape Operations
    'narrow': (NarrowOp, [([2, 8], 'f32')]),
    'unbind': (UnbindOp, [([2, 8], 'f32')]),
    'select_dim': (SelectDimOp, [([2, 8], 'f32')]),
    'movedim': (MovedimOp, [([2, 4, 8], 'f32')]),
    'swapaxes': (SwapaxesOp, [([2, 4], 'f32')]),

    # Additional Indexing Operations
    'index_copy': (IndexCopyOp, [([2, 8], 'f32'), ([2], 'i64'), ([2, 8], 'f32')]),
    'index_add': (IndexAddOp, [([2, 8], 'f32'), ([2], 'i64'), ([2, 8], 'f32')]),
    'index_fill': (IndexFillOp, [([2, 8], 'f32'), ([2], 'i64')]),

    # Additional Reductions
    'std_reduce': (StdReduceOp, [([2, 8], 'f32')]),
    'var_reduce': (VarReduceOp, [([2, 8], 'f32')]),
    'argmax': (ArgmaxOp, [([2, 8], 'f32')]),
    'argmin': (ArgminOp, [([2, 8], 'f32')]),

    # Additional Activations
    'celu': (CeluOp, [([1, 8], 'f32')]),
    'logsigmoid': (LogsigmoidOp, [([1, 8], 'f32')]),

    # Additional Convolution
    'conv3d': (Conv3dOp, [([1, 3, 8, 8, 8], 'f32')]),
    'conv_transpose1d': (ConvTranspose1dOp, [([1, 4, 16], 'f32')]),
    'conv_transpose2d': (ConvTranspose2dOp, [([1, 4, 8, 8], 'f32')]),
    'conv_transpose3d': (ConvTranspose3dOp, [([1, 4, 4, 4, 4], 'f32')]),

    # Additional Pooling
    'max_pool1d': (MaxPool1dOp, [([1, 4, 16], 'f32')]),
    'avg_pool1d': (AvgPool1dOp, [([1, 4, 16], 'f32')]),
    'adaptive_max_pool1d': (AdaptiveMaxPool1dOp, [([1, 4, 16], 'f32')]),
    'adaptive_max_pool2d': (AdaptiveMaxPool2dOp, [([1, 4, 8, 8], 'f32')]),

    # Additional Normalization
    'instance_norm2d': (InstanceNorm2dOp, [([1, 4, 8, 8], 'f32')]),
    'group_norm': (GroupNormOp, [([1, 8, 4, 4], 'f32')]),

    # Additional Matrix Operations
    'einsum': (EinsumOp, [([2, 4, 8], 'f32'), ([2, 8, 4], 'f32')]),
    'tensordot': (TensordotOp, [([4, 8], 'f32'), ([8, 4], 'f32')]),

    # Additional Padding
    'pad_circular': (PadCircularOp, [([1, 1, 4, 4], 'f32')]),

    # Backward / Gradient Operations (Training)
    'relu_backward': (ReluBackwardOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'sigmoid_backward': (SigmoidBackwardOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'tanh_backward': (TanhBackwardOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'exp_backward': (ExpBackwardOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'log_backward': (LogBackwardOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'sqrt_backward': (SqrtBackwardOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'mul_backward': (MulBackwardOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'div_backward': (DivBackwardOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'neg_backward': (NegBackwardOp, [([1, 8], 'f32')]),
    'abs_backward': (AbsBackwardOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'matmul_backward': (MatmulBackwardOp, [([1, 4, 8], 'f32'), ([1, 8, 4], 'f32')]),
    'sum_backward': (SumBackwardOp, [([1], 'f32'), ([1, 8], 'f32')]),
    'transpose_backward': (TransposeBackwardOp, [([4, 8], 'f32')]),
    'mse_loss_backward': (MSELossBackwardOp, [([1], 'f32'), ([1, 8], 'f32'), ([1, 8], 'f32')]),
    'l1_loss_backward': (L1LossBackwardOp, [([1], 'f32'), ([1, 8], 'f32'), ([1, 8], 'f32')]),

    # Complex models
    'simple_mlp': (SimpleMLP, [([1, 8], 'f32')]),
    'simple_convnet': (SimpleConvNet, [([1, 3, 16, 16], 'f32')]),
    'attention_block': (AttentionBlock, [([1, 4, 8], 'f32')]),
    'residual_block': (ResidualBlock, [([1, 8], 'f32')]),

    # Dynamic shape models (use sample batch=2, but batch dim is dynamic)
    # Note: These require dynamic_dims parameter when converting
    'dynamic_batch_mlp': (DynamicBatchMLP, [([2, 8], 'f32')]),
    'dynamic_batch_conv': (DynamicBatchConv, [([2, 3, 16, 16], 'f32')]),
    'dynamic_seq_transformer': (DynamicSeqTransformerBlock, [([2, 4, 8], 'f32')]),
    'dynamic_reshape': (DynamicReshape, [([2, 4, 4], 'f32')]),
    'dynamic_matmul': (DynamicMatmul, [([2, 4, 8], 'f32'), ([2, 8, 4], 'f32')]),
    'dynamic_reduction': (DynamicReduction, [([2, 4, 8], 'f32')]),
    'dynamic_broadcast': (DynamicBroadcast, [([2, 8], 'f32'), ([8], 'f32')]),
    'dynamic_transpose': (DynamicTranspose, [([2, 4, 8], 'f32')]),
    'dynamic_slice': (DynamicSlice, [([2, 8, 16], 'f32')]),
    'dynamic_select': (DynamicSelect, [([2, 8, 16], 'f32')]),
    'dynamic_pad': (DynamicPad, [([2, 8], 'f32')]),
    'dynamic_gather': (DynamicGather, [([2, 8, 4], 'f32'), ([2, 8, 2], 'i64')]),
    'dynamic_expand': (DynamicExpand, [([2, 1, 8], 'f32')]),
    'dynamic_index_select': (DynamicIndexSelect, [([2, 8, 16], 'f32'), ([3], 'i64')]),

    # Quantization models
    # Note: Some of these use fake_quantize for tracing since real quantized tensors
    # require special handling in torch.fx
    'fake_quantize_per_tensor': (FakeQuantizePerTensorOp, [([1, 8], 'f32')]),
    'quantized_linear': (QuantizedLinearModel, [([1, 8], 'f32')]),
    'quantized_conv2d': (QuantizedConv2dModel, [([1, 3, 16, 16], 'f32')]),
    'quantized_relu': (QuantizedReluOp, [([1, 8], 'f32')]),
    'quantized_add': (QuantizedAddOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'quantized_mul': (QuantizedMulOp, [([1, 8], 'f32'), ([1, 8], 'f32')]),
    'int8_linear': (Int8LinearModel, [([1, 8], 'f32')]),
    'int4_linear': (Int4LinearModel, [([1, 8], 'f32')]),

    # Sparse tensor operations
    # Note: Sparse tensors use special input formats (indices + values for COO, etc.)
    'sparse_coo_tensor': (SparseCooTensorOp, [([2, 4], 'i64'), ([4], 'f32')]),
    'sparse_csr_tensor': (SparseCsrTensorOp, [([5], 'i64'), ([4], 'i64'), ([4], 'f32')]),
    'to_sparse_coo': (ToSparseCooOp, [([4, 4], 'f32')]),
    'to_sparse_csr': (ToSparseCsrOp, [([4, 4], 'f32')]),
    'to_sparse_bsr': (ToSparseBsrOp, [([8, 8], 'f32')]),
    'sparse_coalesce': (SparseCoalesceOp, [([4, 4], 'f32')]),
    'sparse_mm': (SparseMmOp, [([4, 4], 'f32'), ([4, 4], 'f32')]),
    'sparse_addmm': (SparseAddmmOp, [([4, 4], 'f32'), ([4, 4], 'f32'), ([4, 4], 'f32')]),
    'sparse_sum': (SparseSumOp, [([4, 4], 'f32')]),
    'sparse_softmax': (SparseSoftmaxOp, [([4, 4], 'f32')]),
    'sparse_indices': (SparseIndicesOp, [([4, 4], 'f32')]),
    'sparse_values': (SparseValuesOp, [([4, 4], 'f32')]),
    'sparse_nnz': (SparseNnzOp, [([4, 4], 'f32')]),
    'sparse_to_dense': (ToDenseOp, [([4, 4], 'f32')]),
    'semi_structured_sparse': (SemiStructuredSparseOp, [([1, 16], 'f32')]),
    'sparse_masked_mm': (SparseMaskedMmOp, [([4, 4], 'f32'), ([4, 4], 'f32'), ([4, 4], 'f32')]),

    # Complex tensor operations
    'complex': (ComplexOp, [([2, 4], 'f32'), ([2, 4], 'f32')]),
    'real': (RealOp, [([2, 4], 'c64')]),
    'imag': (ImagOp, [([2, 4], 'c64')]),
    'conj': (ConjOp, [([2, 4], 'c64')]),
    'view_as_real': (ViewAsRealOp, [([2, 4], 'c64')]),
    'view_as_complex': (ViewAsComplexOp, [([2, 4, 2], 'f32')]),
    'angle': (AngleOp, [([2, 4], 'c64')]),
    'polar': (PolarOp, [([2, 4], 'f32'), ([2, 4], 'f32')]),
    'complex_abs': (ComplexAbsOp, [([2, 4], 'c64')]),
    'complex_mul': (ComplexMulOp, [([2, 4], 'c64'), ([2, 4], 'c64')]),
    'complex_add': (ComplexAddOp, [([2, 4], 'c64'), ([2, 4], 'c64')]),
    'complex_matmul': (ComplexMatmulOp, [([2, 4, 8], 'c64'), ([2, 8, 4], 'c64')]),
    'complex_exp': (ComplexExpOp, [([2, 4], 'c64')]),
    'complex_log': (ComplexLogOp, [([2, 4], 'c64')]),
    'complex_sqrt': (ComplexSqrtOp, [([2, 4], 'c64')]),

    # FFT operations
    # Note: FFT operations typically require complex or real inputs of specific shapes
    # 1D FFT operations use last dimension as transform dimension
    'fft': (FFTOp, [([2, 16], 'c64')]),
    'ifft': (IFFTOp, [([2, 16], 'c64')]),
    'rfft': (RFFTOp, [([2, 16], 'f32')]),
    'irfft': (IRFFTOp, [([2, 9], 'c64')]),  # n//2 + 1 complex values -> n real values
    'hfft': (HFFTOp, [([2, 9], 'c64')]),
    'ihfft': (IHFFTOp, [([2, 16], 'f32')]),
    # 2D FFT operations
    'fft2': (FFT2Op, [([2, 8, 8], 'c64')]),
    'ifft2': (IFFT2Op, [([2, 8, 8], 'c64')]),
    'rfft2': (RFFT2Op, [([2, 8, 8], 'f32')]),
    'irfft2': (IRFFT2Op, [([2, 8, 5], 'c64')]),  # (h, w//2 + 1) complex -> (h, w) real
    # N-D FFT operations
    'fftn': (FFTNOp, [([2, 4, 4, 4], 'c64')]),
    'ifftn': (IFFTNOp, [([2, 4, 4, 4], 'c64')]),
    'rfftn': (RFFTNOp, [([2, 4, 4, 4], 'f32')]),
    'irfftn': (IRFFTNOp, [([2, 4, 4, 3], 'c64')]),
    # FFT helper operations
    'fftshift': (FFTShiftOp, [([2, 16], 'c64')]),
    'ifftshift': (IFFTShiftOp, [([2, 16], 'c64')]),
}

# Registry of dynamic dimensions for models that need them
# Maps op_name -> dict mapping input_index -> set of dynamic dim indices
DYNAMIC_DIMS_REGISTRY = {
    'dynamic_batch_mlp': {0: {0}},           # input 0, dim 0 is dynamic
    'dynamic_batch_conv': {0: {0}},          # batch dim is dynamic
    'dynamic_seq_transformer': {0: {0, 1}},  # batch and seq dims are dynamic
    'dynamic_reshape': {0: {0}},             # batch dim is dynamic
    'dynamic_matmul': {0: {0}, 1: {0}},      # batch dim is dynamic for both inputs
    'dynamic_reduction': {0: {0}},           # batch dim is dynamic
    'dynamic_broadcast': {0: {0}},           # batch dim is dynamic (scale is static)
    'dynamic_transpose': {0: {0, 1}},        # batch and seq dims are dynamic
    'dynamic_slice': {0: {0}},               # batch dim is dynamic
    'dynamic_select': {0: {0}},              # batch dim is dynamic
    'dynamic_pad': {0: {0}},                 # batch dim is dynamic
    'dynamic_gather': {0: {0}, 1: {0}},      # batch dim is dynamic for both inputs
    'dynamic_expand': {0: {0}},              # batch dim is dynamic
    'dynamic_index_select': {0: {0}},        # batch dim is dynamic (indices are static)
}

def get_model_source(op_name):
    """Get the Python source code for a specific operation model."""
    if op_name not in OPERATION_REGISTRY:
        raise ValueError(f"Unknown operation: {op_name}")

    model_class, _ = OPERATION_REGISTRY[op_name]
    class_name = model_class.__name__

    # Return minimal source that imports and uses the class
    return f'''
import torch
import torch.nn as nn
import torch.nn.functional as F

{_get_class_source(model_class)}
'''

def _get_class_source(cls):
    """Extract class source (simplified - in practice would use inspect)."""
    import inspect
    return inspect.getsource(cls)

def list_operations():
    """List all available operations."""
    return list(OPERATION_REGISTRY.keys())

def get_dynamic_dims(op_name):
    """Get dynamic dimensions for a model, if any.

    Returns:
        dict mapping input_index -> set of dynamic dim indices, or None if static
    """
    return DYNAMIC_DIMS_REGISTRY.get(op_name, None)

def list_dynamic_operations():
    """List operations that have dynamic dimension support."""
    return list(DYNAMIC_DIMS_REGISTRY.keys())
