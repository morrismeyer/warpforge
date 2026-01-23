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
# RNN/LSTM/GRU Operations
# =============================================================================

class LSTMOp(nn.Module):
    """Basic LSTM layer.
    Produces: stablehlo.custom_call @lstm
    Input: (seq_len, batch, input_size)
    Output: (output, (h_n, c_n))
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=8, hidden_size=16, num_layers=1, batch_first=False)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return output


class LSTMBatchFirstOp(nn.Module):
    """LSTM with batch_first=True.
    Input: (batch, seq_len, input_size)
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=8, hidden_size=16, num_layers=1, batch_first=True)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return output


class LSTMBidirectionalOp(nn.Module):
    """Bidirectional LSTM.
    Output hidden size is 2 * hidden_size.
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=8, hidden_size=16, num_layers=1, bidirectional=True)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return output


class LSTMMultiLayerOp(nn.Module):
    """Multi-layer LSTM (stacked).
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=8, hidden_size=16, num_layers=3)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return output


class LSTMWithHiddenOp(nn.Module):
    """LSTM with initial hidden state provided.
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=8, hidden_size=16, num_layers=1)

    def forward(self, x, h0, c0):
        output, (h_n, c_n) = self.lstm(x, (h0, c0))
        return output


class GRUOp(nn.Module):
    """Basic GRU layer.
    Produces: stablehlo.custom_call @gru
    """
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=8, hidden_size=16, num_layers=1)

    def forward(self, x):
        output, h_n = self.gru(x)
        return output


class GRUBatchFirstOp(nn.Module):
    """GRU with batch_first=True.
    """
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=8, hidden_size=16, num_layers=1, batch_first=True)

    def forward(self, x):
        output, h_n = self.gru(x)
        return output


class GRUBidirectionalOp(nn.Module):
    """Bidirectional GRU.
    """
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=8, hidden_size=16, num_layers=1, bidirectional=True)

    def forward(self, x):
        output, h_n = self.gru(x)
        return output


class GRUMultiLayerOp(nn.Module):
    """Multi-layer GRU.
    """
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=8, hidden_size=16, num_layers=3)

    def forward(self, x):
        output, h_n = self.gru(x)
        return output


class RNNTanhOp(nn.Module):
    """Basic RNN with tanh nonlinearity.
    Produces: stablehlo.custom_call @rnn
    """
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=8, hidden_size=16, num_layers=1, nonlinearity='tanh')

    def forward(self, x):
        output, h_n = self.rnn(x)
        return output


class RNNReluOp(nn.Module):
    """RNN with ReLU nonlinearity.
    """
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=8, hidden_size=16, num_layers=1, nonlinearity='relu')

    def forward(self, x):
        output, h_n = self.rnn(x)
        return output


class RNNBidirectionalOp(nn.Module):
    """Bidirectional RNN.
    """
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=8, hidden_size=16, num_layers=1, bidirectional=True)

    def forward(self, x):
        output, h_n = self.rnn(x)
        return output


class LSTMCellOp(nn.Module):
    """LSTM cell (single step).
    Produces: stablehlo.custom_call @lstm_cell
    """
    def __init__(self):
        super().__init__()
        self.cell = nn.LSTMCell(input_size=8, hidden_size=16)

    def forward(self, x, hx, cx):
        h_n, c_n = self.cell(x, (hx, cx))
        return h_n


class GRUCellOp(nn.Module):
    """GRU cell (single step).
    Produces: stablehlo.custom_call @gru_cell
    """
    def __init__(self):
        super().__init__()
        self.cell = nn.GRUCell(input_size=8, hidden_size=16)

    def forward(self, x, hx):
        h_n = self.cell(x, hx)
        return h_n


class RNNCellOp(nn.Module):
    """RNN cell (single step).
    """
    def __init__(self):
        super().__init__()
        self.cell = nn.RNNCell(input_size=8, hidden_size=16)

    def forward(self, x, hx):
        h_n = self.cell(x, hx)
        return h_n


# =============================================================================
# Attention Operations
# =============================================================================

class ScaledDotProductAttentionOp(nn.Module):
    """Basic scaled dot-product attention (F.scaled_dot_product_attention).
    Produces: stablehlo.custom_call @scaled_dot_product_attention
    Input: query, key, value with shapes (batch, num_heads, seq_len, head_dim)
    """
    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(query, key, value)


class ScaledDotProductAttentionCausalOp(nn.Module):
    """Scaled dot-product attention with causal mask.
    is_causal=True applies a causal mask preventing attention to future tokens.
    """
    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(query, key, value, is_causal=True)


class ScaledDotProductAttentionDropoutOp(nn.Module):
    """Scaled dot-product attention with dropout.
    dropout_p specifies dropout probability during training.
    """
    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(query, key, value, dropout_p=0.1)


class ScaledDotProductAttentionMaskOp(nn.Module):
    """Scaled dot-product attention with explicit attention mask.
    Mask is added to attention scores before softmax.
    """
    def forward(self, query, key, value, attn_mask):
        return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)


class ScaledDotProductAttentionScaleOp(nn.Module):
    """Scaled dot-product attention with custom scale factor.
    Default scale is 1/sqrt(head_dim), can be overridden.
    """
    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(query, key, value, scale=0.1)


class MultiheadAttentionOp(nn.Module):
    """Multi-head attention layer (nn.MultiheadAttention).
    Produces: stablehlo.custom_call @multi_head_attention
    Input: query, key, value with shape (seq_len, batch, embed_dim)
    """
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=False)

    def forward(self, query, key, value):
        output, attn_weights = self.mha(query, key, value)
        return output


class MultiheadAttentionBatchFirstOp(nn.Module):
    """Multi-head attention with batch_first=True.
    Input: query, key, value with shape (batch, seq_len, embed_dim)
    """
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)

    def forward(self, query, key, value):
        output, attn_weights = self.mha(query, key, value)
        return output


class MultiheadAttentionWithMaskOp(nn.Module):
    """Multi-head attention with key padding mask.
    key_padding_mask: (batch, seq_len) bool tensor, True = masked position
    """
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=8)

    def forward(self, query, key, value, key_padding_mask):
        output, attn_weights = self.mha(query, key, value, key_padding_mask=key_padding_mask)
        return output


class MultiheadAttentionAttnMaskOp(nn.Module):
    """Multi-head attention with attention mask.
    attn_mask: (seq_len, seq_len) or (batch*num_heads, seq_len, seq_len)
    """
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=8)

    def forward(self, query, key, value, attn_mask):
        output, attn_weights = self.mha(query, key, value, attn_mask=attn_mask)
        return output


class MultiheadAttentionDropoutOp(nn.Module):
    """Multi-head attention with dropout.
    """
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=8, dropout=0.1)

    def forward(self, query, key, value):
        output, attn_weights = self.mha(query, key, value)
        return output


class MultiheadAttentionNoBiasOp(nn.Module):
    """Multi-head attention without bias.
    """
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=8, bias=False)

    def forward(self, query, key, value):
        output, attn_weights = self.mha(query, key, value)
        return output


class MultiheadAttentionAddBiasKVOp(nn.Module):
    """Multi-head attention with add_bias_kv=True.
    Adds learnable bias to key and value sequences.
    """
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=8, add_bias_kv=True)

    def forward(self, query, key, value):
        output, attn_weights = self.mha(query, key, value)
        return output


class MultiheadAttentionAddZeroAttnOp(nn.Module):
    """Multi-head attention with add_zero_attn=True.
    Adds a new batch of zeros to key and value sequences.
    """
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=8, add_zero_attn=True)

    def forward(self, query, key, value):
        output, attn_weights = self.mha(query, key, value)
        return output


class MultiheadAttentionKDimVDimOp(nn.Module):
    """Multi-head attention with different key/value dimensions.
    kdim and vdim specify dimensions for key and value projections.
    """
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=8, kdim=32, vdim=32)

    def forward(self, query, key, value):
        output, attn_weights = self.mha(query, key, value)
        return output


class SelfAttentionOp(nn.Module):
    """Self-attention (query = key = value).
    Common in transformer encoders.
    """
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=8)

    def forward(self, x):
        output, attn_weights = self.mha(x, x, x)
        return output


class CrossAttentionOp(nn.Module):
    """Cross-attention (query from one source, key/value from another).
    Common in transformer decoders.
    """
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=8)

    def forward(self, query, memory):
        output, attn_weights = self.mha(query, memory, memory)
        return output


class AttentionWithProjectionOp(nn.Module):
    """Attention followed by output projection.
    Tests attention -> linear pattern common in transformers.
    """
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        self.proj = nn.Linear(64, 64)

    def forward(self, query, key, value):
        output, _ = self.mha(query, key, value)
        return self.proj(output)


class AttentionWithLayerNormOp(nn.Module):
    """Attention with layer normalization (pre-norm style).
    Tests pattern: LayerNorm -> Attention
    """
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(64)
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=8)

    def forward(self, x):
        normed = self.norm(x)
        output, _ = self.mha(normed, normed, normed)
        return output


class AttentionWithResidualOp(nn.Module):
    """Attention with residual connection.
    Tests pattern: x + Attention(x)
    """
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=8)

    def forward(self, x):
        output, _ = self.mha(x, x, x)
        return x + output


class TransformerEncoderLayerOp(nn.Module):
    """Full transformer encoder layer.
    Tests complete: attention + feedforward + residuals + norms
    """
    def __init__(self):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=False)

    def forward(self, x):
        return self.layer(x)


class TransformerEncoderLayerBatchFirstOp(nn.Module):
    """Transformer encoder layer with batch_first=True.
    """
    def __init__(self):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True)

    def forward(self, x):
        return self.layer(x)


class TransformerDecoderLayerOp(nn.Module):
    """Full transformer decoder layer.
    Tests self-attention + cross-attention + feedforward
    """
    def __init__(self):
        super().__init__()
        self.layer = nn.TransformerDecoderLayer(d_model=64, nhead=8, batch_first=False)

    def forward(self, tgt, memory):
        return self.layer(tgt, memory)


class TransformerEncoderOp(nn.Module):
    """Stacked transformer encoder.
    Tests multiple encoder layers in sequence.
    """
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        return self.encoder(x)


class TransformerDecoderOp(nn.Module):
    """Stacked transformer decoder.
    Tests multiple decoder layers in sequence.
    """
    def __init__(self):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=64, nhead=8, batch_first=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

    def forward(self, tgt, memory):
        return self.decoder(tgt, memory)


class TransformerOp(nn.Module):
    """Full transformer model (encoder + decoder).
    """
    def __init__(self):
        super().__init__()
        self.transformer = nn.Transformer(d_model=64, nhead=8, num_encoder_layers=2, num_decoder_layers=2, batch_first=False)

    def forward(self, src, tgt):
        return self.transformer(src, tgt)


# =============================================================================
# Embedding Operations
# =============================================================================

class EmbeddingOp(nn.Module):
    """Basic embedding layer (nn.Embedding).
    Produces: stablehlo.gather
    Input: indices tensor, Output: embedded vectors
    """
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=1000, embedding_dim=64)

    def forward(self, x):
        return self.embedding(x)


class EmbeddingWithPaddingOp(nn.Module):
    """Embedding with padding_idx.
    Vectors at padding_idx are always zero.
    """
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=1000, embedding_dim=64, padding_idx=0)

    def forward(self, x):
        return self.embedding(x)


class EmbeddingWithMaxNormOp(nn.Module):
    """Embedding with max_norm constraint.
    Embedding vectors are renormalized if they exceed max_norm.
    """
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=1000, embedding_dim=64, max_norm=1.0)

    def forward(self, x):
        return self.embedding(x)


class EmbeddingWithNormTypeOp(nn.Module):
    """Embedding with custom norm type for max_norm.
    norm_type controls the p-norm used (default is L2).
    """
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=1000, embedding_dim=64, max_norm=1.0, norm_type=1.0)

    def forward(self, x):
        return self.embedding(x)


class EmbeddingScaleGradOp(nn.Module):
    """Embedding with scale_grad_by_freq=True.
    Gradients scaled by word frequency in batch.
    """
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=1000, embedding_dim=64, scale_grad_by_freq=True)

    def forward(self, x):
        return self.embedding(x)


class EmbeddingSparseOp(nn.Module):
    """Sparse embedding (sparse=True).
    Gradients are sparse tensors for efficiency.
    """
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=1000, embedding_dim=64, sparse=True)

    def forward(self, x):
        return self.embedding(x)


class EmbeddingFunctionalOp(nn.Module):
    """Embedding using F.embedding functional API.
    """
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1000, 64))

    def forward(self, x):
        return F.embedding(x, self.weight)


class EmbeddingBagSumOp(nn.Module):
    """EmbeddingBag with mode='sum'.
    Sums embeddings for each bag/sample.
    """
    def __init__(self):
        super().__init__()
        self.embedding_bag = nn.EmbeddingBag(num_embeddings=1000, embedding_dim=64, mode='sum')

    def forward(self, x, offsets):
        return self.embedding_bag(x, offsets)


class EmbeddingBagMeanOp(nn.Module):
    """EmbeddingBag with mode='mean'.
    Averages embeddings for each bag/sample.
    """
    def __init__(self):
        super().__init__()
        self.embedding_bag = nn.EmbeddingBag(num_embeddings=1000, embedding_dim=64, mode='mean')

    def forward(self, x, offsets):
        return self.embedding_bag(x, offsets)


class EmbeddingBagMaxOp(nn.Module):
    """EmbeddingBag with mode='max'.
    Takes max embedding for each bag/sample.
    """
    def __init__(self):
        super().__init__()
        self.embedding_bag = nn.EmbeddingBag(num_embeddings=1000, embedding_dim=64, mode='max')

    def forward(self, x, offsets):
        return self.embedding_bag(x, offsets)


class EmbeddingBagWithWeightsOp(nn.Module):
    """EmbeddingBag with per_sample_weights.
    Weighted combination of embeddings.
    """
    def __init__(self):
        super().__init__()
        self.embedding_bag = nn.EmbeddingBag(num_embeddings=1000, embedding_dim=64, mode='sum')

    def forward(self, x, offsets, per_sample_weights):
        return self.embedding_bag(x, offsets, per_sample_weights)


class EmbeddingBagPaddingOp(nn.Module):
    """EmbeddingBag with padding_idx.
    """
    def __init__(self):
        super().__init__()
        self.embedding_bag = nn.EmbeddingBag(num_embeddings=1000, embedding_dim=64, mode='sum', padding_idx=0)

    def forward(self, x, offsets):
        return self.embedding_bag(x, offsets)


class EmbeddingBagSparseOp(nn.Module):
    """Sparse EmbeddingBag.
    """
    def __init__(self):
        super().__init__()
        self.embedding_bag = nn.EmbeddingBag(num_embeddings=1000, embedding_dim=64, mode='sum', sparse=True)

    def forward(self, x, offsets):
        return self.embedding_bag(x, offsets)


class EmbeddingBagLastOffsetOp(nn.Module):
    """EmbeddingBag with include_last_offset=True.
    """
    def __init__(self):
        super().__init__()
        self.embedding_bag = nn.EmbeddingBag(num_embeddings=1000, embedding_dim=64, mode='sum', include_last_offset=True)

    def forward(self, x, offsets):
        return self.embedding_bag(x, offsets)


class OneHotOp(nn.Module):
    """One-hot encoding (F.one_hot).
    """
    def forward(self, x):
        return F.one_hot(x, num_classes=10)


class OneHotDynamicOp(nn.Module):
    """One-hot encoding with num_classes=-1 (auto-detect).
    """
    def forward(self, x):
        return F.one_hot(x)


class EmbeddingWithProjectionOp(nn.Module):
    """Embedding followed by linear projection.
    Common pattern in transformers.
    """
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=1000, embedding_dim=64)
        self.proj = nn.Linear(64, 128)

    def forward(self, x):
        embedded = self.embedding(x)
        return self.proj(embedded)


class EmbeddingWithDropoutOp(nn.Module):
    """Embedding followed by dropout.
    Common pattern for regularization.
    """
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=1000, embedding_dim=64)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        embedded = self.embedding(x)
        return self.dropout(embedded)


class EmbeddingWithLayerNormOp(nn.Module):
    """Embedding followed by layer normalization.
    Common in transformer architectures.
    """
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=1000, embedding_dim=64)
        self.layer_norm = nn.LayerNorm(64)

    def forward(self, x):
        embedded = self.embedding(x)
        return self.layer_norm(embedded)


class PositionalEmbeddingOp(nn.Module):
    """Learned positional embedding.
    Separate embedding for position indices.
    """
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=1000, embedding_dim=64)
        self.position_embedding = nn.Embedding(num_embeddings=512, embedding_dim=64)

    def forward(self, x):
        # x: (batch, seq_len)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        token_embed = self.token_embedding(x)
        pos_embed = self.position_embedding(positions)
        return token_embed + pos_embed


class EmbeddingSum2dOp(nn.Module):
    """Sum of multiple embeddings (e.g., token + segment + position).
    BERT-style combined embedding.
    """
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=30000, embedding_dim=64)
        self.segment_embedding = nn.Embedding(num_embeddings=2, embedding_dim=64)
        self.position_embedding = nn.Embedding(num_embeddings=512, embedding_dim=64)

    def forward(self, token_ids, segment_ids, position_ids):
        token_embed = self.token_embedding(token_ids)
        segment_embed = self.segment_embedding(segment_ids)
        position_embed = self.position_embedding(position_ids)
        return token_embed + segment_embed + position_embed


# =============================================================================
# Loss Function Operations
# =============================================================================

class CrossEntropyLossOp(nn.Module):
    """Cross-entropy loss (nn.CrossEntropyLoss).
    Standard classification loss.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        return self.loss(input, target)


class CrossEntropyLossWeightedOp(nn.Module):
    """Cross-entropy loss with class weights.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=torch.ones(10))

    def forward(self, input, target):
        return self.loss(input, target)


class CrossEntropyLossIgnoreIndexOp(nn.Module):
    """Cross-entropy loss with ignore_index.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input, target):
        return self.loss(input, target)


class CrossEntropyLossLabelSmoothingOp(nn.Module):
    """Cross-entropy loss with label smoothing.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, input, target):
        return self.loss(input, target)


class CrossEntropyLossSumOp(nn.Module):
    """Cross-entropy loss with reduction='sum'.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, input, target):
        return self.loss(input, target)


class CrossEntropyLossNoneOp(nn.Module):
    """Cross-entropy loss with reduction='none'.
    Returns per-sample loss.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        return self.loss(input, target)


class NLLLossOp(nn.Module):
    """Negative log likelihood loss (nn.NLLLoss).
    Expects log-probabilities input.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.NLLLoss()

    def forward(self, input, target):
        return self.loss(input, target)


class BCELossOp(nn.Module):
    """Binary cross-entropy loss (nn.BCELoss).
    Expects probabilities input.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, input, target):
        return self.loss(input, target)


class BCEWithLogitsLossOp(nn.Module):
    """BCE with logits (nn.BCEWithLogitsLoss).
    More numerically stable than BCELoss.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        return self.loss(input, target)


class BCEWithLogitsPosWeightOp(nn.Module):
    """BCE with logits and pos_weight.
    For imbalanced binary classification.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.ones(10) * 2)

    def forward(self, input, target):
        return self.loss(input, target)


class MSELossOp(nn.Module):
    """Mean squared error loss (nn.MSELoss).
    Standard regression loss.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        return self.loss(input, target)


class MSELossSumOp(nn.Module):
    """MSE loss with reduction='sum'.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, input, target):
        return self.loss(input, target)


class L1LossOp(nn.Module):
    """L1/MAE loss (nn.L1Loss).
    Robust to outliers.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, input, target):
        return self.loss(input, target)


class SmoothL1LossOp(nn.Module):
    """Smooth L1 / Huber loss (nn.SmoothL1Loss).
    Combines L1 and L2.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.SmoothL1Loss()

    def forward(self, input, target):
        return self.loss(input, target)


class SmoothL1LossBetaOp(nn.Module):
    """Smooth L1 loss with custom beta.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.SmoothL1Loss(beta=0.5)

    def forward(self, input, target):
        return self.loss(input, target)


class HuberLossOp(nn.Module):
    """Huber loss (nn.HuberLoss).
    Similar to SmoothL1 with delta parameter.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.HuberLoss()

    def forward(self, input, target):
        return self.loss(input, target)


class HuberLossDeltaOp(nn.Module):
    """Huber loss with custom delta.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.HuberLoss(delta=0.5)

    def forward(self, input, target):
        return self.loss(input, target)


class KLDivLossOp(nn.Module):
    """KL divergence loss (nn.KLDivLoss).
    Measures distribution difference.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, input, target):
        return self.loss(F.log_softmax(input, dim=1), F.softmax(target, dim=1))


class KLDivLossLogTargetOp(nn.Module):
    """KL divergence with log_target=True.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, input, target):
        return self.loss(F.log_softmax(input, dim=1), F.log_softmax(target, dim=1))


class HingeEmbeddingLossOp(nn.Module):
    """Hinge embedding loss (nn.HingeEmbeddingLoss).
    For learning embeddings.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.HingeEmbeddingLoss()

    def forward(self, input, target):
        return self.loss(input, target)


class MarginRankingLossOp(nn.Module):
    """Margin ranking loss (nn.MarginRankingLoss).
    For learning to rank.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.MarginRankingLoss()

    def forward(self, input1, input2, target):
        return self.loss(input1, input2, target)


class TripletMarginLossOp(nn.Module):
    """Triplet margin loss (nn.TripletMarginLoss).
    For metric learning.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.TripletMarginLoss()

    def forward(self, anchor, positive, negative):
        return self.loss(anchor, positive, negative)


class TripletMarginLossCustomOp(nn.Module):
    """Triplet margin loss with custom margin and p.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.TripletMarginLoss(margin=2.0, p=1)

    def forward(self, anchor, positive, negative):
        return self.loss(anchor, positive, negative)


class CosineEmbeddingLossOp(nn.Module):
    """Cosine embedding loss (nn.CosineEmbeddingLoss).
    For learning cosine similarity.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.CosineEmbeddingLoss()

    def forward(self, input1, input2, target):
        return self.loss(input1, input2, target)


class CTCLossOp(nn.Module):
    """CTC loss (nn.CTCLoss).
    For sequence-to-sequence without alignment.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.CTCLoss()

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        return self.loss(log_probs, targets, input_lengths, target_lengths)


class CTCLossZeroInfinityOp(nn.Module):
    """CTC loss with zero_infinity=True.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.CTCLoss(zero_infinity=True)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        return self.loss(log_probs, targets, input_lengths, target_lengths)


class PoissonNLLLossOp(nn.Module):
    """Poisson NLL loss (nn.PoissonNLLLoss).
    For count data.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.PoissonNLLLoss()

    def forward(self, input, target):
        return self.loss(input, target)


class GaussianNLLLossOp(nn.Module):
    """Gaussian NLL loss (nn.GaussianNLLLoss).
    For regression with uncertainty.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.GaussianNLLLoss()

    def forward(self, input, target, var):
        return self.loss(input, target, var)


class SoftMarginLossOp(nn.Module):
    """Soft margin loss (nn.SoftMarginLoss).
    Two-class classification.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.SoftMarginLoss()

    def forward(self, input, target):
        return self.loss(input, target)


class MultiMarginLossOp(nn.Module):
    """Multi-margin loss (nn.MultiMarginLoss).
    Multi-class hinge loss.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.MultiMarginLoss()

    def forward(self, input, target):
        return self.loss(input, target)


class MultiLabelMarginLossOp(nn.Module):
    """Multi-label margin loss (nn.MultiLabelMarginLoss).
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.MultiLabelMarginLoss()

    def forward(self, input, target):
        return self.loss(input, target)


class MultiLabelSoftMarginLossOp(nn.Module):
    """Multi-label soft margin loss (nn.MultiLabelSoftMarginLoss).
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.MultiLabelSoftMarginLoss()

    def forward(self, input, target):
        return self.loss(input, target)


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
# Dropout Operations
# =============================================================================

class DropoutOp(nn.Module):
    """Standard dropout layer.
    Produces: stablehlo.custom_call @dropout
    """
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        return self.dropout(x)


class DropoutP03Op(nn.Module):
    """Dropout with p=0.3.
    Produces: stablehlo.custom_call @dropout
    """
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        return self.dropout(x)


class Dropout1dOp(nn.Module):
    """1D channel dropout (drops entire channels).
    Produces: stablehlo.custom_call @dropout1d
    """
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout1d(p=0.5)

    def forward(self, x):
        return self.dropout(x)


class Dropout2dOp(nn.Module):
    """2D channel dropout (drops entire channels).
    Produces: stablehlo.custom_call @dropout2d
    """
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        return self.dropout(x)


class Dropout3dOp(nn.Module):
    """3D channel dropout (drops entire channels).
    Produces: stablehlo.custom_call @dropout3d
    """
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout3d(p=0.5)

    def forward(self, x):
        return self.dropout(x)


class AlphaDropoutOp(nn.Module):
    """Alpha dropout for self-normalizing networks (SELU).
    Produces: stablehlo.custom_call @alpha_dropout
    """
    def __init__(self):
        super().__init__()
        self.dropout = nn.AlphaDropout(p=0.5)

    def forward(self, x):
        return self.dropout(x)


class DropoutFunctionalOp(nn.Module):
    """Functional dropout API.
    Produces: stablehlo.custom_call @dropout
    """
    def forward(self, x):
        return F.dropout(x, p=0.5, training=True)


class FeatureAlphaDropoutOp(nn.Module):
    """Feature-wise alpha dropout.
    Produces: stablehlo.custom_call @feature_alpha_dropout
    """
    def forward(self, x):
        return F.feature_alpha_dropout(x, p=0.5, training=True)


# =============================================================================
# Upsampling Operations
# =============================================================================

class UpsampleNearest2dOp(nn.Module):
    """Nearest neighbor upsampling (2D).
    Produces: stablehlo.custom_call @upsample_nearest
    """
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.upsample(x)


class UpsampleNearest3dOp(nn.Module):
    """Nearest neighbor upsampling (3D).
    Produces: stablehlo.custom_call @upsample_nearest
    """
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.upsample(x)


class UpsampleBilinear2dOp(nn.Module):
    """Bilinear upsampling (2D).
    Produces: stablehlo.custom_call @upsample_bilinear
    """
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        return self.upsample(x)


class UpsampleBicubic2dOp(nn.Module):
    """Bicubic upsampling (2D).
    Produces: stablehlo.custom_call @upsample_bicubic
    """
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

    def forward(self, x):
        return self.upsample(x)


class UpsampleTrilinear3dOp(nn.Module):
    """Trilinear upsampling (3D).
    Produces: stablehlo.custom_call @upsample_trilinear
    """
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x):
        return self.upsample(x)


class UpsampleBilinearAlignCornersOp(nn.Module):
    """Bilinear upsampling with align_corners=True.
    Produces: stablehlo.custom_call @upsample_bilinear
    """
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        return self.upsample(x)


class InterpolateNearestOp(nn.Module):
    """F.interpolate with nearest mode.
    Produces: stablehlo.custom_call @interpolate
    """
    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')


class InterpolateBilinearOp(nn.Module):
    """F.interpolate with bilinear mode.
    Produces: stablehlo.custom_call @interpolate
    """
    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


class InterpolateBicubicOp(nn.Module):
    """F.interpolate with bicubic mode.
    Produces: stablehlo.custom_call @interpolate
    """
    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)


class InterpolateSizeOp(nn.Module):
    """F.interpolate with explicit size.
    Produces: stablehlo.custom_call @interpolate
    """
    def forward(self, x):
        return F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)


class InterpolateScaleFactorOp(nn.Module):
    """F.interpolate with scale_factor.
    Produces: stablehlo.custom_call @interpolate
    """
    def forward(self, x):
        return F.interpolate(x, scale_factor=3, mode='nearest')


class PixelShuffleOp(nn.Module):
    """Pixel shuffle for sub-pixel convolution upsampling.
    Produces: stablehlo.custom_call @pixel_shuffle
    """
    def __init__(self):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.pixel_shuffle(x)


class PixelUnshuffleOp(nn.Module):
    """Pixel unshuffle - inverse of pixel shuffle.
    Produces: stablehlo.custom_call @pixel_unshuffle
    """
    def __init__(self):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(2)

    def forward(self, x):
        return self.pixel_unshuffle(x)


class PixelShuffle3xOp(nn.Module):
    """Pixel shuffle with upscale_factor=3.
    Produces: stablehlo.custom_call @pixel_shuffle
    """
    def __init__(self):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(3)

    def forward(self, x):
        return self.pixel_shuffle(x)


# =============================================================================
# Grid Sampling Operations
# =============================================================================

class GridSampleBilinearOp(nn.Module):
    """Grid sample with bilinear interpolation.
    Produces: stablehlo.custom_call @grid_sample
    """
    def forward(self, input, grid):
        return F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


class GridSampleNearestOp(nn.Module):
    """Grid sample with nearest neighbor interpolation.
    Produces: stablehlo.custom_call @grid_sample
    """
    def forward(self, input, grid):
        return F.grid_sample(input, grid, mode='nearest', padding_mode='zeros', align_corners=False)


class GridSampleBicubicOp(nn.Module):
    """Grid sample with bicubic interpolation.
    Produces: stablehlo.custom_call @grid_sample
    """
    def forward(self, input, grid):
        return F.grid_sample(input, grid, mode='bicubic', padding_mode='zeros', align_corners=False)


class GridSampleZerosOp(nn.Module):
    """Grid sample with zeros padding.
    Produces: stablehlo.custom_call @grid_sample
    """
    def forward(self, input, grid):
        return F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


class GridSampleBorderOp(nn.Module):
    """Grid sample with border padding.
    Produces: stablehlo.custom_call @grid_sample
    """
    def forward(self, input, grid):
        return F.grid_sample(input, grid, mode='bilinear', padding_mode='border', align_corners=False)


class GridSampleReflectionOp(nn.Module):
    """Grid sample with reflection padding.
    Produces: stablehlo.custom_call @grid_sample
    """
    def forward(self, input, grid):
        return F.grid_sample(input, grid, mode='bilinear', padding_mode='reflection', align_corners=False)


class GridSampleAlignCornersOp(nn.Module):
    """Grid sample with align_corners=True.
    Produces: stablehlo.custom_call @grid_sample
    """
    def forward(self, input, grid):
        return F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True)


class AffineGridOp(nn.Module):
    """Affine grid generation.
    Produces: stablehlo.custom_call @affine_grid
    """
    def forward(self, theta):
        # Generate grid for 8x8 output
        return F.affine_grid(theta, (theta.size(0), 1, 8, 8), align_corners=False)


class SpatialTransformerOp(nn.Module):
    """Full spatial transformer: affine_grid + grid_sample.
    Produces: stablehlo.custom_call @affine_grid, @grid_sample
    """
    def forward(self, input, theta):
        grid = F.affine_grid(theta, input.size(), align_corners=False)
        return F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


# =============================================================================
# Image Operations
# =============================================================================

class HorizontalFlipOp(nn.Module):
    """Horizontal flip.
    Produces: stablehlo.reverse
    """
    def forward(self, x):
        return torch.flip(x, dims=[-1])


class VerticalFlipOp(nn.Module):
    """Vertical flip.
    Produces: stablehlo.reverse
    """
    def forward(self, x):
        return torch.flip(x, dims=[-2])


class AdjustBrightnessOp(nn.Module):
    """Adjust brightness by multiplying.
    Produces: stablehlo.multiply
    """
    def forward(self, x):
        return x * 1.5


class AdjustContrastOp(nn.Module):
    """Adjust contrast.
    Produces: composite ops
    """
    def forward(self, x):
        mean = x.mean(dim=(-2, -1), keepdim=True)
        return (x - mean) * 1.5 + mean


class NormalizeImageOp(nn.Module):
    """Image normalization with mean/std.
    Produces: stablehlo.subtract, stablehlo.divide
    """
    def forward(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (x - mean) / std


# =============================================================================
# Sorting Operations
# =============================================================================

class SortOp(nn.Module):
    """Sort tensor along dimension.
    Produces: stablehlo.custom_call @sort
    """
    def forward(self, x):
        return torch.sort(x, dim=-1)[0]


class SortDescendingOp(nn.Module):
    """Sort tensor in descending order.
    Produces: stablehlo.custom_call @sort
    """
    def forward(self, x):
        return torch.sort(x, dim=-1, descending=True)[0]


class ArgsortOp(nn.Module):
    """Get indices that would sort tensor.
    Produces: stablehlo.custom_call @argsort
    """
    def forward(self, x):
        return torch.argsort(x, dim=-1)


class TopkOp(nn.Module):
    """Get top-k values and indices.
    Produces: stablehlo.custom_call @topk
    """
    def forward(self, x):
        return torch.topk(x, k=3, dim=-1)[0]


class TopkIndicesOp(nn.Module):
    """Get top-k indices only.
    Produces: stablehlo.custom_call @topk
    """
    def forward(self, x):
        return torch.topk(x, k=3, dim=-1)[1]


class KthvalueOp(nn.Module):
    """Get k-th smallest value.
    Produces: stablehlo.custom_call @kthvalue
    """
    def forward(self, x):
        return torch.kthvalue(x, k=2, dim=-1)[0]


class MsortOp(nn.Module):
    """Sort along first dimension.
    Produces: stablehlo.custom_call @sort
    """
    def forward(self, x):
        return torch.msort(x)


# =============================================================================
# Unique/Set Operations
# =============================================================================

class UniqueOp(nn.Module):
    """Get unique elements.
    Produces: stablehlo.custom_call @unique
    """
    def forward(self, x):
        return torch.unique(x)


class UniqueSortedOp(nn.Module):
    """Get unique elements, sorted.
    Produces: stablehlo.custom_call @unique
    """
    def forward(self, x):
        return torch.unique(x, sorted=True)


class UniqueConsecutiveOp(nn.Module):
    """Get unique consecutive elements.
    Produces: stablehlo.custom_call @unique_consecutive
    """
    def forward(self, x):
        return torch.unique_consecutive(x)


class BincountOp(nn.Module):
    """Count occurrences of each value.
    Produces: stablehlo.custom_call @bincount
    """
    def forward(self, x):
        return torch.bincount(x.flatten().to(torch.int64), minlength=10)


class HistcOp(nn.Module):
    """Compute histogram.
    Produces: stablehlo.custom_call @histc
    """
    def forward(self, x):
        return torch.histc(x, bins=10, min=0, max=1)


# =============================================================================
# Linear Algebra Operations
# =============================================================================

class MatrixInverseOp(nn.Module):
    """Matrix inverse.
    Produces: stablehlo.custom_call @inv
    """
    def forward(self, x):
        return torch.linalg.inv(x)


class MatrixDetOp(nn.Module):
    """Matrix determinant.
    Produces: stablehlo.custom_call @det
    """
    def forward(self, x):
        return torch.linalg.det(x)


class SVDOp(nn.Module):
    """Singular value decomposition.
    Produces: stablehlo.custom_call @svd
    """
    def forward(self, x):
        U, S, Vh = torch.linalg.svd(x)
        return S  # Return singular values


class QROp(nn.Module):
    """QR decomposition.
    Produces: stablehlo.custom_call @qr
    """
    def forward(self, x):
        Q, R = torch.linalg.qr(x)
        return R


class CholeskyOp(nn.Module):
    """Cholesky decomposition.
    Produces: stablehlo.custom_call @cholesky
    """
    def forward(self, x):
        # Make input positive definite: x @ x.T + I
        xxt = torch.matmul(x, x.transpose(-2, -1))
        eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
        return torch.linalg.cholesky(xxt + eye)


class EighOp(nn.Module):
    """Eigenvalue decomposition for symmetric matrices.
    Produces: stablehlo.custom_call @eigh
    """
    def forward(self, x):
        # Make symmetric
        sym = (x + x.transpose(-2, -1)) / 2
        eigenvalues, eigenvectors = torch.linalg.eigh(sym)
        return eigenvalues


class MatrixNormOp(nn.Module):
    """Matrix Frobenius norm.
    Produces: stablehlo.custom_call @norm
    """
    def forward(self, x):
        return torch.linalg.norm(x, ord='fro')


class VectorNormOp(nn.Module):
    """Vector L2 norm.
    Produces: stablehlo.custom_call @norm
    """
    def forward(self, x):
        return torch.linalg.norm(x, dim=-1)


class SolveOp(nn.Module):
    """Solve linear system Ax = B.
    Produces: stablehlo.custom_call @solve
    """
    def forward(self, A, B):
        return torch.linalg.solve(A, B)


class LstSqOp(nn.Module):
    """Least squares solution.
    Produces: stablehlo.custom_call @lstsq
    """
    def forward(self, A, B):
        return torch.linalg.lstsq(A, B).solution


class CrossProductOp(nn.Module):
    """Cross product of 3D vectors.
    Produces: stablehlo.custom_call @cross
    """
    def forward(self, a, b):
        return torch.linalg.cross(a, b)


class TraceOp(nn.Module):
    """Matrix trace (sum of diagonal).
    Produces: stablehlo.custom_call @trace
    """
    def forward(self, x):
        return torch.trace(x)


class DiagonalOp(nn.Module):
    """Extract diagonal from matrix.
    Produces: stablehlo.custom_call @diagonal
    """
    def forward(self, x):
        return torch.diagonal(x, dim1=-2, dim2=-1)


class DiagEmbedOp(nn.Module):
    """Embed vector as diagonal matrix.
    Produces: stablehlo.custom_call @diag_embed
    """
    def forward(self, x):
        return torch.diag_embed(x)


class TrilOp(nn.Module):
    """Lower triangular part.
    Produces: stablehlo.custom_call @tril
    """
    def forward(self, x):
        return torch.tril(x)


class TriuOp(nn.Module):
    """Upper triangular part.
    Produces: stablehlo.custom_call @triu
    """
    def forward(self, x):
        return torch.triu(x)


class PinvOp(nn.Module):
    """Pseudo-inverse.
    Produces: stablehlo.custom_call @pinv
    """
    def forward(self, x):
        return torch.linalg.pinv(x)


class MatrixRankOp(nn.Module):
    """Matrix rank.
    Produces: stablehlo.custom_call @matrix_rank
    """
    def forward(self, x):
        return torch.linalg.matrix_rank(x)


# =============================================================================
# Tensor Creation Operations
# =============================================================================

class ZerosOp(nn.Module):
    """Create tensor of zeros.
    Produces: stablehlo.constant
    """
    def forward(self, x):
        return torch.zeros_like(x)


class OnesOp(nn.Module):
    """Create tensor of ones.
    Produces: stablehlo.constant
    """
    def forward(self, x):
        return torch.ones_like(x)


class FullOp(nn.Module):
    """Create tensor filled with value.
    Produces: stablehlo.constant
    """
    def forward(self, x):
        return torch.full_like(x, 3.14)


class EmptyLikeOp(nn.Module):
    """Create uninitialized tensor (zeros in practice).
    Produces: stablehlo.constant
    """
    def forward(self, x):
        return torch.empty_like(x)


class ArangeOp(nn.Module):
    """Create 1D tensor with evenly spaced values.
    Produces: stablehlo.custom_call @arange
    """
    def forward(self, x):
        # Return arange based on input size
        n = x.shape[-1]
        return torch.arange(n, dtype=x.dtype, device=x.device)


class LinspaceOp(nn.Module):
    """Create 1D tensor with linearly spaced values.
    Produces: stablehlo.custom_call @linspace
    """
    def forward(self, x):
        return torch.linspace(0, 1, steps=x.shape[-1], dtype=x.dtype, device=x.device)


class LogspaceOp(nn.Module):
    """Create 1D tensor with logarithmically spaced values.
    Produces: stablehlo.custom_call @logspace
    """
    def forward(self, x):
        return torch.logspace(0, 2, steps=x.shape[-1], dtype=x.dtype, device=x.device)


class EyeOp(nn.Module):
    """Create identity matrix.
    Produces: stablehlo.custom_call @eye
    """
    def forward(self, x):
        n = x.shape[-1]
        return torch.eye(n, dtype=x.dtype, device=x.device)


class EyeRectOp(nn.Module):
    """Create rectangular identity matrix.
    Produces: stablehlo.custom_call @eye
    """
    def forward(self, x):
        return torch.eye(x.shape[-2], x.shape[-1], dtype=x.dtype, device=x.device)


# =============================================================================
# Random Operations
# =============================================================================

class RandOp(nn.Module):
    """Uniform random [0, 1).
    Produces: stablehlo.custom_call @rand
    """
    def forward(self, x):
        return torch.rand_like(x)


class RandnOp(nn.Module):
    """Standard normal random.
    Produces: stablehlo.custom_call @randn
    """
    def forward(self, x):
        return torch.randn_like(x)


class RandintOp(nn.Module):
    """Random integers in range.
    Produces: stablehlo.custom_call @randint
    """
    def forward(self, x):
        return torch.randint_like(x.to(torch.int64), low=0, high=10)


class RandpermOp(nn.Module):
    """Random permutation.
    Produces: stablehlo.custom_call @randperm
    """
    def forward(self, x):
        n = x.shape[-1]
        return torch.randperm(n, device=x.device)


class BernoulliOp(nn.Module):
    """Bernoulli random (0 or 1).
    Produces: stablehlo.custom_call @bernoulli
    """
    def forward(self, x):
        # x is probability of 1
        probs = torch.sigmoid(x)  # ensure [0, 1]
        return torch.bernoulli(probs)


class MultinomialOp(nn.Module):
    """Sample from multinomial distribution.
    Produces: stablehlo.custom_call @multinomial
    """
    def forward(self, x):
        # x is unnormalized log probabilities
        probs = torch.softmax(x, dim=-1)
        return torch.multinomial(probs, num_samples=3, replacement=True)


class NormalOp(nn.Module):
    """Normal distribution sampling.
    Produces: stablehlo.custom_call @normal
    """
    def forward(self, mean, std):
        return torch.normal(mean, std.abs() + 0.1)


class UniformOp(nn.Module):
    """Uniform random in range.
    Produces: stablehlo.custom_call @uniform
    """
    def forward(self, x):
        result = torch.empty_like(x)
        return result.uniform_(-1, 1)


class ExponentialOp(nn.Module):
    """Exponential distribution.
    Produces: stablehlo.custom_call @exponential
    """
    def forward(self, x):
        result = torch.empty_like(x)
        return result.exponential_(lambd=1.0)


class PoissonOp(nn.Module):
    """Poisson distribution.
    Produces: stablehlo.custom_call @poisson
    """
    def forward(self, x):
        # x is the rate parameter (must be positive)
        rates = x.abs() + 0.1
        return torch.poisson(rates)


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

    # RNN/LSTM/GRU Operations
    # Input shapes: (seq_len, batch, input_size) for seq-first, (batch, seq_len, input_size) for batch-first
    'lstm': (LSTMOp, [([10, 2, 8], 'f32')]),  # seq_len=10, batch=2, input=8
    'lstm_batch_first': (LSTMBatchFirstOp, [([2, 10, 8], 'f32')]),  # batch=2, seq_len=10
    'lstm_bidirectional': (LSTMBidirectionalOp, [([10, 2, 8], 'f32')]),
    'lstm_multi_layer': (LSTMMultiLayerOp, [([10, 2, 8], 'f32')]),
    'lstm_with_hidden': (LSTMWithHiddenOp, [([10, 2, 8], 'f32'), ([1, 2, 16], 'f32'), ([1, 2, 16], 'f32')]),
    'gru': (GRUOp, [([10, 2, 8], 'f32')]),
    'gru_batch_first': (GRUBatchFirstOp, [([2, 10, 8], 'f32')]),
    'gru_bidirectional': (GRUBidirectionalOp, [([10, 2, 8], 'f32')]),
    'gru_multi_layer': (GRUMultiLayerOp, [([10, 2, 8], 'f32')]),
    'rnn_tanh': (RNNTanhOp, [([10, 2, 8], 'f32')]),
    'rnn_relu': (RNNReluOp, [([10, 2, 8], 'f32')]),
    'rnn_bidirectional': (RNNBidirectionalOp, [([10, 2, 8], 'f32')]),
    'lstm_cell': (LSTMCellOp, [([2, 8], 'f32'), ([2, 16], 'f32'), ([2, 16], 'f32')]),  # batch=2
    'gru_cell': (GRUCellOp, [([2, 8], 'f32'), ([2, 16], 'f32')]),
    'rnn_cell': (RNNCellOp, [([2, 8], 'f32'), ([2, 16], 'f32')]),

    # Attention Operations
    # Scaled dot-product attention shapes: (batch, num_heads, seq_len, head_dim)
    'scaled_dot_product_attention': (ScaledDotProductAttentionOp, [([2, 8, 10, 16], 'f32'), ([2, 8, 10, 16], 'f32'), ([2, 8, 10, 16], 'f32')]),  # batch=2, heads=8, seq=10, head_dim=16
    'sdpa_causal': (ScaledDotProductAttentionCausalOp, [([2, 8, 10, 16], 'f32'), ([2, 8, 10, 16], 'f32'), ([2, 8, 10, 16], 'f32')]),
    'sdpa_dropout': (ScaledDotProductAttentionDropoutOp, [([2, 8, 10, 16], 'f32'), ([2, 8, 10, 16], 'f32'), ([2, 8, 10, 16], 'f32')]),
    'sdpa_mask': (ScaledDotProductAttentionMaskOp, [([2, 8, 10, 16], 'f32'), ([2, 8, 10, 16], 'f32'), ([2, 8, 10, 16], 'f32'), ([2, 8, 10, 10], 'f32')]),  # mask: (batch, heads, seq, seq)
    'sdpa_scale': (ScaledDotProductAttentionScaleOp, [([2, 8, 10, 16], 'f32'), ([2, 8, 10, 16], 'f32'), ([2, 8, 10, 16], 'f32')]),
    # Multi-head attention shapes: (seq_len, batch, embed_dim) for seq-first
    'multihead_attention': (MultiheadAttentionOp, [([10, 2, 64], 'f32'), ([10, 2, 64], 'f32'), ([10, 2, 64], 'f32')]),  # seq=10, batch=2, embed=64
    'mha_batch_first': (MultiheadAttentionBatchFirstOp, [([2, 10, 64], 'f32'), ([2, 10, 64], 'f32'), ([2, 10, 64], 'f32')]),
    'mha_key_padding_mask': (MultiheadAttentionWithMaskOp, [([10, 2, 64], 'f32'), ([10, 2, 64], 'f32'), ([10, 2, 64], 'f32'), ([2, 10], 'bool')]),  # mask: (batch, seq)
    'mha_attn_mask': (MultiheadAttentionAttnMaskOp, [([10, 2, 64], 'f32'), ([10, 2, 64], 'f32'), ([10, 2, 64], 'f32'), ([10, 10], 'f32')]),  # mask: (seq, seq)
    'mha_dropout': (MultiheadAttentionDropoutOp, [([10, 2, 64], 'f32'), ([10, 2, 64], 'f32'), ([10, 2, 64], 'f32')]),
    'mha_no_bias': (MultiheadAttentionNoBiasOp, [([10, 2, 64], 'f32'), ([10, 2, 64], 'f32'), ([10, 2, 64], 'f32')]),
    'mha_add_bias_kv': (MultiheadAttentionAddBiasKVOp, [([10, 2, 64], 'f32'), ([10, 2, 64], 'f32'), ([10, 2, 64], 'f32')]),
    'mha_add_zero_attn': (MultiheadAttentionAddZeroAttnOp, [([10, 2, 64], 'f32'), ([10, 2, 64], 'f32'), ([10, 2, 64], 'f32')]),
    'mha_kdim_vdim': (MultiheadAttentionKDimVDimOp, [([10, 2, 64], 'f32'), ([10, 2, 32], 'f32'), ([10, 2, 32], 'f32')]),  # key/value have different dim
    # Attention patterns
    'self_attention': (SelfAttentionOp, [([10, 2, 64], 'f32')]),  # single input for self-attn
    'cross_attention': (CrossAttentionOp, [([10, 2, 64], 'f32'), ([20, 2, 64], 'f32')]),  # query and memory can differ in seq_len
    'attention_with_projection': (AttentionWithProjectionOp, [([10, 2, 64], 'f32'), ([10, 2, 64], 'f32'), ([10, 2, 64], 'f32')]),
    'attention_with_layernorm': (AttentionWithLayerNormOp, [([10, 2, 64], 'f32')]),
    'attention_with_residual': (AttentionWithResidualOp, [([10, 2, 64], 'f32')]),
    # Transformer layers and models
    'transformer_encoder_layer': (TransformerEncoderLayerOp, [([10, 2, 64], 'f32')]),
    'transformer_encoder_layer_batch_first': (TransformerEncoderLayerBatchFirstOp, [([2, 10, 64], 'f32')]),
    'transformer_decoder_layer': (TransformerDecoderLayerOp, [([10, 2, 64], 'f32'), ([20, 2, 64], 'f32')]),  # tgt and memory
    'transformer_encoder': (TransformerEncoderOp, [([10, 2, 64], 'f32')]),
    'transformer_decoder': (TransformerDecoderOp, [([10, 2, 64], 'f32'), ([20, 2, 64], 'f32')]),
    'transformer': (TransformerOp, [([20, 2, 64], 'f32'), ([10, 2, 64], 'f32')]),  # src and tgt

    # Embedding Operations
    # Input: indices tensor (i64), Output: embedded vectors (f32)
    'embedding': (EmbeddingOp, [([2, 10], 'i64')]),  # batch=2, seq=10
    'embedding_with_padding': (EmbeddingWithPaddingOp, [([2, 10], 'i64')]),
    'embedding_with_max_norm': (EmbeddingWithMaxNormOp, [([2, 10], 'i64')]),
    'embedding_with_norm_type': (EmbeddingWithNormTypeOp, [([2, 10], 'i64')]),
    'embedding_scale_grad': (EmbeddingScaleGradOp, [([2, 10], 'i64')]),
    'embedding_sparse': (EmbeddingSparseOp, [([2, 10], 'i64')]),
    'embedding_functional': (EmbeddingFunctionalOp, [([2, 10], 'i64')]),
    # EmbeddingBag: indices (1D), offsets (1D)
    'embedding_bag_sum': (EmbeddingBagSumOp, [([20], 'i64'), ([4], 'i64')]),  # 20 indices, 4 bags
    'embedding_bag_mean': (EmbeddingBagMeanOp, [([20], 'i64'), ([4], 'i64')]),
    'embedding_bag_max': (EmbeddingBagMaxOp, [([20], 'i64'), ([4], 'i64')]),
    'embedding_bag_with_weights': (EmbeddingBagWithWeightsOp, [([20], 'i64'), ([4], 'i64'), ([20], 'f32')]),  # per_sample_weights
    'embedding_bag_padding': (EmbeddingBagPaddingOp, [([20], 'i64'), ([4], 'i64')]),
    'embedding_bag_sparse': (EmbeddingBagSparseOp, [([20], 'i64'), ([4], 'i64')]),
    'embedding_bag_last_offset': (EmbeddingBagLastOffsetOp, [([20], 'i64'), ([5], 'i64')]),  # 5 offsets with include_last_offset
    # One-hot encoding
    'one_hot': (OneHotOp, [([2, 5], 'i64')]),  # batch=2, seq=5
    'one_hot_dynamic': (OneHotDynamicOp, [([2, 5], 'i64')]),
    # Embedding patterns
    'embedding_with_projection': (EmbeddingWithProjectionOp, [([2, 10], 'i64')]),
    'embedding_with_dropout': (EmbeddingWithDropoutOp, [([2, 10], 'i64')]),
    'embedding_with_layernorm': (EmbeddingWithLayerNormOp, [([2, 10], 'i64')]),
    'positional_embedding': (PositionalEmbeddingOp, [([2, 10], 'i64')]),
    'embedding_sum_2d': (EmbeddingSum2dOp, [([2, 10], 'i64'), ([2, 10], 'i64'), ([2, 10], 'i64')]),  # token, segment, position

    # Loss Functions
    # Classification losses: input (batch, num_classes), target (batch,) for class indices
    'cross_entropy_loss': (CrossEntropyLossOp, [([4, 10], 'f32'), ([4], 'i64')]),  # batch=4, classes=10
    'cross_entropy_loss_weighted': (CrossEntropyLossWeightedOp, [([4, 10], 'f32'), ([4], 'i64')]),
    'cross_entropy_loss_ignore_index': (CrossEntropyLossIgnoreIndexOp, [([4, 10], 'f32'), ([4], 'i64')]),
    'cross_entropy_loss_label_smoothing': (CrossEntropyLossLabelSmoothingOp, [([4, 10], 'f32'), ([4], 'i64')]),
    'cross_entropy_loss_sum': (CrossEntropyLossSumOp, [([4, 10], 'f32'), ([4], 'i64')]),
    'cross_entropy_loss_none': (CrossEntropyLossNoneOp, [([4, 10], 'f32'), ([4], 'i64')]),
    'nll_loss': (NLLLossOp, [([4, 10], 'f32'), ([4], 'i64')]),
    # Binary classification losses
    'bce_loss': (BCELossOp, [([4, 10], 'f32'), ([4, 10], 'f32')]),
    'bce_with_logits_loss': (BCEWithLogitsLossOp, [([4, 10], 'f32'), ([4, 10], 'f32')]),
    'bce_with_logits_pos_weight': (BCEWithLogitsPosWeightOp, [([4, 10], 'f32'), ([4, 10], 'f32')]),
    # Regression losses
    'mse_loss': (MSELossOp, [([4, 10], 'f32'), ([4, 10], 'f32')]),
    'mse_loss_sum': (MSELossSumOp, [([4, 10], 'f32'), ([4, 10], 'f32')]),
    'l1_loss': (L1LossOp, [([4, 10], 'f32'), ([4, 10], 'f32')]),
    'smooth_l1_loss': (SmoothL1LossOp, [([4, 10], 'f32'), ([4, 10], 'f32')]),
    'smooth_l1_loss_beta': (SmoothL1LossBetaOp, [([4, 10], 'f32'), ([4, 10], 'f32')]),
    'huber_loss': (HuberLossOp, [([4, 10], 'f32'), ([4, 10], 'f32')]),
    'huber_loss_delta': (HuberLossDeltaOp, [([4, 10], 'f32'), ([4, 10], 'f32')]),
    # Distribution losses
    'kl_div_loss': (KLDivLossOp, [([4, 10], 'f32'), ([4, 10], 'f32')]),
    'kl_div_loss_log_target': (KLDivLossLogTargetOp, [([4, 10], 'f32'), ([4, 10], 'f32')]),
    # Embedding/metric learning losses
    'hinge_embedding_loss': (HingeEmbeddingLossOp, [([4, 10], 'f32'), ([4, 10], 'f32')]),
    'margin_ranking_loss': (MarginRankingLossOp, [([4], 'f32'), ([4], 'f32'), ([4], 'f32')]),
    'triplet_margin_loss': (TripletMarginLossOp, [([4, 64], 'f32'), ([4, 64], 'f32'), ([4, 64], 'f32')]),  # anchor, positive, negative
    'triplet_margin_loss_custom': (TripletMarginLossCustomOp, [([4, 64], 'f32'), ([4, 64], 'f32'), ([4, 64], 'f32')]),
    'cosine_embedding_loss': (CosineEmbeddingLossOp, [([4, 64], 'f32'), ([4, 64], 'f32'), ([4], 'f32')]),
    # Sequence losses
    'ctc_loss': (CTCLossOp, [([50, 4, 20], 'f32'), ([4, 10], 'i64'), ([4], 'i64'), ([4], 'i64')]),  # (T, N, C), targets, input_lengths, target_lengths
    'ctc_loss_zero_infinity': (CTCLossZeroInfinityOp, [([50, 4, 20], 'f32'), ([4, 10], 'i64'), ([4], 'i64'), ([4], 'i64')]),
    # Statistical losses
    'poisson_nll_loss': (PoissonNLLLossOp, [([4, 10], 'f32'), ([4, 10], 'f32')]),
    'gaussian_nll_loss': (GaussianNLLLossOp, [([4, 10], 'f32'), ([4, 10], 'f32'), ([4, 10], 'f32')]),  # input, target, var
    # Margin losses
    'soft_margin_loss': (SoftMarginLossOp, [([4, 10], 'f32'), ([4, 10], 'f32')]),
    'multi_margin_loss': (MultiMarginLossOp, [([4, 10], 'f32'), ([4], 'i64')]),
    'multilabel_margin_loss': (MultiLabelMarginLossOp, [([4, 10], 'f32'), ([4, 10], 'i64')]),
    'multilabel_soft_margin_loss': (MultiLabelSoftMarginLossOp, [([4, 10], 'f32'), ([4, 10], 'f32')]),

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

    # Dropout operations
    'dropout': (DropoutOp, [([2, 8, 16], 'f32')]),
    'dropout_p03': (DropoutP03Op, [([2, 8, 16], 'f32')]),
    'dropout1d': (Dropout1dOp, [([2, 8, 16], 'f32')]),
    'dropout2d': (Dropout2dOp, [([2, 8, 8, 8], 'f32')]),
    'dropout3d': (Dropout3dOp, [([2, 8, 4, 4, 4], 'f32')]),
    'alpha_dropout': (AlphaDropoutOp, [([2, 8, 16], 'f32')]),
    'dropout_functional': (DropoutFunctionalOp, [([2, 8, 16], 'f32')]),
    'feature_alpha_dropout': (FeatureAlphaDropoutOp, [([2, 8, 16], 'f32')]),

    # Upsampling operations
    'upsample_nearest_2d': (UpsampleNearest2dOp, [([1, 3, 8, 8], 'f32')]),
    'upsample_nearest_3d': (UpsampleNearest3dOp, [([1, 3, 4, 4, 4], 'f32')]),
    'upsample_bilinear_2d': (UpsampleBilinear2dOp, [([1, 3, 8, 8], 'f32')]),
    'upsample_bicubic_2d': (UpsampleBicubic2dOp, [([1, 3, 8, 8], 'f32')]),
    'upsample_trilinear_3d': (UpsampleTrilinear3dOp, [([1, 3, 4, 4, 4], 'f32')]),
    'upsample_bilinear_align_corners': (UpsampleBilinearAlignCornersOp, [([1, 3, 8, 8], 'f32')]),
    'interpolate_nearest': (InterpolateNearestOp, [([1, 3, 8, 8], 'f32')]),
    'interpolate_bilinear': (InterpolateBilinearOp, [([1, 3, 8, 8], 'f32')]),
    'interpolate_bicubic': (InterpolateBicubicOp, [([1, 3, 8, 8], 'f32')]),
    'interpolate_size': (InterpolateSizeOp, [([1, 3, 8, 8], 'f32')]),
    'interpolate_scale_factor': (InterpolateScaleFactorOp, [([1, 3, 8, 8], 'f32')]),
    'pixel_shuffle': (PixelShuffleOp, [([1, 12, 8, 8], 'f32')]),  # 12 = 3 * 2^2
    'pixel_unshuffle': (PixelUnshuffleOp, [([1, 3, 16, 16], 'f32')]),
    'pixel_shuffle_3x': (PixelShuffle3xOp, [([1, 27, 4, 4], 'f32')]),  # 27 = 3 * 3^2

    # Grid sampling operations
    'grid_sample_bilinear': (GridSampleBilinearOp, [([2, 3, 8, 8], 'f32'), ([2, 8, 8, 2], 'f32')]),
    'grid_sample_nearest': (GridSampleNearestOp, [([2, 3, 8, 8], 'f32'), ([2, 8, 8, 2], 'f32')]),
    'grid_sample_bicubic': (GridSampleBicubicOp, [([2, 3, 8, 8], 'f32'), ([2, 8, 8, 2], 'f32')]),
    'grid_sample_zeros': (GridSampleZerosOp, [([2, 3, 8, 8], 'f32'), ([2, 8, 8, 2], 'f32')]),
    'grid_sample_border': (GridSampleBorderOp, [([2, 3, 8, 8], 'f32'), ([2, 8, 8, 2], 'f32')]),
    'grid_sample_reflection': (GridSampleReflectionOp, [([2, 3, 8, 8], 'f32'), ([2, 8, 8, 2], 'f32')]),
    'grid_sample_align_corners': (GridSampleAlignCornersOp, [([2, 3, 8, 8], 'f32'), ([2, 8, 8, 2], 'f32')]),
    'affine_grid': (AffineGridOp, [([2, 2, 3], 'f32')]),  # theta: (N, 2, 3)
    'spatial_transformer': (SpatialTransformerOp, [([2, 3, 8, 8], 'f32'), ([2, 2, 3], 'f32')]),

    # Image operations
    'horizontal_flip': (HorizontalFlipOp, [([1, 3, 8, 8], 'f32')]),
    'vertical_flip': (VerticalFlipOp, [([1, 3, 8, 8], 'f32')]),
    'adjust_brightness': (AdjustBrightnessOp, [([1, 3, 8, 8], 'f32')]),
    'adjust_contrast': (AdjustContrastOp, [([1, 3, 8, 8], 'f32')]),
    'normalize_image': (NormalizeImageOp, [([1, 3, 8, 8], 'f32')]),

    # Sorting operations
    'sort': (SortOp, [([2, 16], 'f32')]),
    'sort_descending': (SortDescendingOp, [([2, 16], 'f32')]),
    'argsort': (ArgsortOp, [([2, 16], 'f32')]),
    'topk': (TopkOp, [([2, 16], 'f32')]),
    'topk_indices': (TopkIndicesOp, [([2, 16], 'f32')]),
    'kthvalue': (KthvalueOp, [([2, 16], 'f32')]),
    'msort': (MsortOp, [([8, 8], 'f32')]),

    # Unique/set operations
    'unique': (UniqueOp, [([64], 'f32')]),
    'unique_sorted': (UniqueSortedOp, [([64], 'f32')]),
    'unique_consecutive': (UniqueConsecutiveOp, [([64], 'f32')]),
    'bincount': (BincountOp, [([64], 'f32')]),
    'histc': (HistcOp, [([64], 'f32')]),

    # Linear algebra operations
    'matrix_inverse': (MatrixInverseOp, [([2, 4, 4], 'f32')]),
    'matrix_det': (MatrixDetOp, [([2, 4, 4], 'f32')]),
    'svd': (SVDOp, [([2, 4, 4], 'f32')]),
    'qr': (QROp, [([2, 4, 4], 'f32')]),
    'cholesky': (CholeskyOp, [([2, 4, 4], 'f32')]),
    'eigh': (EighOp, [([2, 4, 4], 'f32')]),
    'matrix_norm': (MatrixNormOp, [([4, 4], 'f32')]),
    'vector_norm': (VectorNormOp, [([2, 8], 'f32')]),
    'solve': (SolveOp, [([2, 4, 4], 'f32'), ([2, 4, 2], 'f32')]),
    'lstsq': (LstSqOp, [([2, 4, 4], 'f32'), ([2, 4, 2], 'f32')]),
    'cross_product': (CrossProductOp, [([2, 3], 'f32'), ([2, 3], 'f32')]),
    'trace': (TraceOp, [([4, 4], 'f32')]),
    'diagonal': (DiagonalOp, [([2, 4, 4], 'f32')]),
    'diag_embed': (DiagEmbedOp, [([2, 4], 'f32')]),
    'tril': (TrilOp, [([4, 4], 'f32')]),
    'triu': (TriuOp, [([4, 4], 'f32')]),
    'pinv': (PinvOp, [([2, 4, 4], 'f32')]),
    'matrix_rank': (MatrixRankOp, [([2, 4, 4], 'f32')]),

    # ===== TENSOR CREATION OPERATIONS =====
    'zeros': (ZerosOp, [([2, 8], 'f32')]),
    'ones': (OnesOp, [([2, 8], 'f32')]),
    'full': (FullOp, [([2, 8], 'f32')]),
    'zeros_like': (ZerosLikeOp, [([2, 8], 'f32')]),
    'ones_like': (OnesLikeOp, [([2, 8], 'f32')]),
    'full_like': (FullLikeOp, [([2, 8], 'f32')]),
    'empty_like': (EmptyLikeOp, [([2, 8], 'f32')]),
    'arange': (ArangeOp, [([16], 'f32')]),
    'linspace': (LinspaceOp, [([10], 'f32')]),
    'logspace': (LogspaceOp, [([10], 'f32')]),
    'eye': (EyeOp, [([4, 4], 'f32')]),
    'eye_rect': (EyeRectOp, [([4, 6], 'f32')]),

    # ===== RANDOM OPERATIONS =====
    'rand': (RandOp, [([2, 8], 'f32')]),
    'randn': (RandnOp, [([2, 8], 'f32')]),
    'rand_like': (RandLikeOp, [([2, 8], 'f32')]),
    'randn_like': (RandnLikeOp, [([2, 8], 'f32')]),
    'randint': (RandintOp, [([2, 8], 'i64')]),
    'randint_like': (RandintLikeOp, [([2, 8], 'i64')]),
    'randperm': (RandpermOp, [([10], 'i64')]),
    'bernoulli': (BernoulliOp, [([2, 8], 'f32')]),
    'multinomial': (MultinomialOp, [([2, 8], 'f32')]),
    'normal': (NormalOp, [([2, 8], 'f32')]),
    'uniform': (UniformOp, [([2, 8], 'f32')]),
    'exponential': (ExponentialOp, [([2, 8], 'f32')]),
    'poisson': (PoissonOp, [([2, 8], 'f32')]),
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
