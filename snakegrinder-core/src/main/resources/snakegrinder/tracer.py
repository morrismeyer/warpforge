"""
Core tracing infrastructure for mock PyTorch/JAX.

TracedTensor is a proxy that records operations to a computation graph
instead of performing actual tensor computations.
"""

from __future__ import annotations
from typing import Tuple, Optional, List, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from graph_ir import ComputationGraph, GraphNode

# Simple counter for unique IDs
_id_counter = 0


def _next_id(prefix: str = "t") -> str:
    global _id_counter
    _id_counter += 1
    return f"{prefix}_{_id_counter}"


class TensorMeta:
    """Metadata about a tensor without actual data."""

    __slots__ = ('shape', 'dtype')

    def __init__(self, shape: Tuple[int, ...], dtype: str = 'f32'):
        self.shape = tuple(shape)
        self.dtype = dtype

    def element_count(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    def __repr__(self):
        shape_str = 'x'.join(str(d) for d in self.shape)
        return f"TensorMeta({shape_str}x{self.dtype})"

    def __eq__(self, other):
        if isinstance(other, TensorMeta):
            return self.shape == other.shape and self.dtype == other.dtype
        return False


class TracedTensor:
    """
    A proxy tensor that records operations instead of computing.
    Each TracedTensor corresponds to a node in the computation graph.
    """

    # Class-level reference to the current graph being traced
    _current_graph: Optional[ComputationGraph] = None

    __slots__ = ('meta', 'node_id', 'producing_op')

    def __init__(self,
                 meta: TensorMeta,
                 node_id: Optional[str] = None,
                 producing_op: Optional[GraphNode] = None):
        self.meta = meta
        self.node_id = node_id or _next_id("t")
        self.producing_op = producing_op

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.meta.shape

    @property
    def dtype(self):
        return MockDtype(self.meta.dtype)

    @property
    def ndim(self) -> int:
        return len(self.meta.shape)

    # --- Operator overloads that record to graph ---

    def __matmul__(self, other: TracedTensor) -> TracedTensor:
        return self._record_binary_op('dot_general', self, other, infer_matmul_shape)

    def __rmatmul__(self, other: TracedTensor) -> TracedTensor:
        return self._record_binary_op('dot_general', other, self, infer_matmul_shape)

    def __add__(self, other: Union[TracedTensor, int, float]) -> TracedTensor:
        if isinstance(other, (int, float)):
            other = self._scalar_to_tensor(other)
        return self._record_binary_op('add', self, other, infer_broadcast_shape)

    def __radd__(self, other: Union[TracedTensor, int, float]) -> TracedTensor:
        return self.__add__(other)

    def __mul__(self, other: Union[TracedTensor, int, float]) -> TracedTensor:
        if isinstance(other, (int, float)):
            other = self._scalar_to_tensor(other)
        return self._record_binary_op('multiply', self, other, infer_broadcast_shape)

    def __rmul__(self, other: Union[TracedTensor, int, float]) -> TracedTensor:
        return self.__mul__(other)

    def __neg__(self) -> TracedTensor:
        return self._record_unary_op('negate', self)

    def __sub__(self, other: Union[TracedTensor, int, float]) -> TracedTensor:
        if isinstance(other, (int, float)):
            other = self._scalar_to_tensor(other)
        return self + (-other)

    def __rsub__(self, other: Union[TracedTensor, int, float]) -> TracedTensor:
        if isinstance(other, (int, float)):
            other = self._scalar_to_tensor(other)
        return other + (-self)

    def __truediv__(self, other: Union[TracedTensor, int, float]) -> TracedTensor:
        if isinstance(other, (int, float)):
            other = self._scalar_to_tensor(other)
        return self._record_binary_op('divide', self, other, infer_broadcast_shape)

    def reshape(self, *new_shape) -> TracedTensor:
        if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
            new_shape = tuple(new_shape[0])
        return self._record_reshape(self, new_shape)

    def transpose(self, *permutation) -> TracedTensor:
        if len(permutation) == 0:
            # Default: reverse all dimensions
            permutation = tuple(reversed(range(len(self.shape))))
        elif len(permutation) == 1 and isinstance(permutation[0], (list, tuple)):
            permutation = tuple(permutation[0])
        return self._record_transpose(self, permutation)

    @property
    def T(self) -> TracedTensor:
        """Transpose property for 2D tensors."""
        return self.transpose()

    def sum(self, axis=None, keepdims=False) -> TracedTensor:
        """Reduction sum."""
        return self._record_reduce('reduce_sum', self, axis, keepdims)

    def mean(self, axis=None, keepdims=False) -> TracedTensor:
        """Reduction mean."""
        return self._record_reduce('reduce_mean', self, axis, keepdims)

    def __repr__(self):
        return f"TracedTensor({self.meta}, id={self.node_id})"

    # --- Static helpers for recording ops ---

    def _scalar_to_tensor(self, scalar: Union[int, float]) -> TracedTensor:
        """Convert a scalar to a broadcast-compatible constant tensor."""
        graph = TracedTensor._current_graph
        if graph is None:
            raise RuntimeError("No active trace context")

        dtype = 'f32' if isinstance(scalar, float) else 'i32'
        # Create a scalar tensor that will broadcast
        meta = TensorMeta(shape=(), dtype=dtype)
        node = graph.add_op('constant', [], meta, attrs={'value': scalar})
        result = TracedTensor(meta, producing_op=node)
        node.outputs = [result]
        return result

    @staticmethod
    def _record_binary_op(op_name: str,
                          lhs: TracedTensor,
                          rhs: TracedTensor,
                          shape_fn) -> TracedTensor:
        graph = TracedTensor._current_graph
        if graph is None:
            raise RuntimeError("No active trace context")

        result_shape = shape_fn(lhs.meta, rhs.meta)
        result_dtype = lhs.meta.dtype  # Simplified: same as lhs
        result_meta = TensorMeta(result_shape, result_dtype)

        node = graph.add_op(op_name, [lhs, rhs], result_meta)
        result = TracedTensor(result_meta, producing_op=node)
        node.outputs = [result]
        return result

    @staticmethod
    def _record_unary_op(op_name: str, operand: TracedTensor) -> TracedTensor:
        graph = TracedTensor._current_graph
        if graph is None:
            raise RuntimeError("No active trace context")

        result_meta = TensorMeta(operand.shape, operand.meta.dtype)
        node = graph.add_op(op_name, [operand], result_meta)
        result = TracedTensor(result_meta, producing_op=node)
        node.outputs = [result]
        return result

    @staticmethod
    def _record_reshape(operand: TracedTensor, new_shape: Tuple[int, ...]) -> TracedTensor:
        graph = TracedTensor._current_graph
        if graph is None:
            raise RuntimeError("No active trace context")

        # Handle -1 in shape (infer dimension)
        new_shape = _resolve_shape_with_inference(operand.meta, new_shape)

        result_meta = TensorMeta(new_shape, operand.meta.dtype)
        node = graph.add_op('reshape', [operand], result_meta,
                            attrs={'new_shape': new_shape})
        result = TracedTensor(result_meta, producing_op=node)
        node.outputs = [result]
        return result

    @staticmethod
    def _record_transpose(operand: TracedTensor, permutation: Tuple[int, ...]) -> TracedTensor:
        graph = TracedTensor._current_graph
        if graph is None:
            raise RuntimeError("No active trace context")

        new_shape = tuple(operand.shape[p] for p in permutation)
        result_meta = TensorMeta(new_shape, operand.meta.dtype)
        node = graph.add_op('transpose', [operand], result_meta,
                            attrs={'permutation': permutation})
        result = TracedTensor(result_meta, producing_op=node)
        node.outputs = [result]
        return result

    @staticmethod
    def _record_reduce(op_name: str,
                       operand: TracedTensor,
                       axis,
                       keepdims: bool) -> TracedTensor:
        graph = TracedTensor._current_graph
        if graph is None:
            raise RuntimeError("No active trace context")

        # Compute output shape
        if axis is None:
            # Reduce all dimensions
            if keepdims:
                new_shape = tuple(1 for _ in operand.shape)
            else:
                new_shape = ()
            reduce_dims = tuple(range(len(operand.shape)))
        else:
            if isinstance(axis, int):
                axis = (axis,)
            reduce_dims = tuple(a if a >= 0 else len(operand.shape) + a for a in axis)
            if keepdims:
                new_shape = tuple(1 if i in reduce_dims else d
                                  for i, d in enumerate(operand.shape))
            else:
                new_shape = tuple(d for i, d in enumerate(operand.shape)
                                  if i not in reduce_dims)

        result_meta = TensorMeta(new_shape, operand.meta.dtype)
        node = graph.add_op(op_name, [operand], result_meta,
                            attrs={'dimensions': reduce_dims, 'keepdims': keepdims})
        result = TracedTensor(result_meta, producing_op=node)
        node.outputs = [result]
        return result


class MockDtype:
    """Mock dtype object for API compatibility."""

    __slots__ = ('name',)

    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        if isinstance(other, MockDtype):
            return self.name == other.name
        return False

    def __repr__(self):
        return f"dtype({self.name})"


# --- Shape inference helpers ---

def infer_matmul_shape(lhs: TensorMeta, rhs: TensorMeta) -> Tuple[int, ...]:
    """Infer output shape for matrix multiplication."""
    # Handle 2D matmul: (M, K) @ (K, N) -> (M, N)
    if len(lhs.shape) == 2 and len(rhs.shape) == 2:
        m, k1 = lhs.shape
        k2, n = rhs.shape
        if k1 != k2:
            raise ValueError(f"Matmul dimension mismatch: {lhs.shape} @ {rhs.shape}")
        return (m, n)

    # Handle batched matmul: (..., M, K) @ (..., K, N) -> (..., M, N)
    if len(lhs.shape) >= 2 and len(rhs.shape) >= 2:
        batch_lhs = lhs.shape[:-2]
        batch_rhs = rhs.shape[:-2]
        m, k1 = lhs.shape[-2:]
        k2, n = rhs.shape[-2:]

        if k1 != k2:
            raise ValueError(f"Matmul dimension mismatch: {lhs.shape} @ {rhs.shape}")

        # Broadcast batch dimensions
        batch = _broadcast_shapes(batch_lhs, batch_rhs)
        return batch + (m, n)

    raise ValueError(f"Matmul requires at least 2D tensors: {lhs.shape} @ {rhs.shape}")


def infer_broadcast_shape(lhs: TensorMeta, rhs: TensorMeta) -> Tuple[int, ...]:
    """Infer broadcast shape for elementwise ops."""
    return _broadcast_shapes(lhs.shape, rhs.shape)


def _broadcast_shapes(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute broadcast shape for two shapes."""
    # Pad shorter shape with 1s on the left
    ndim = max(len(shape1), len(shape2))
    s1 = (1,) * (ndim - len(shape1)) + shape1
    s2 = (1,) * (ndim - len(shape2)) + shape2

    result = []
    for d1, d2 in zip(s1, s2):
        if d1 == d2:
            result.append(d1)
        elif d1 == 1:
            result.append(d2)
        elif d2 == 1:
            result.append(d1)
        else:
            raise ValueError(f"Cannot broadcast shapes {shape1} and {shape2}")
    return tuple(result)


def _resolve_shape_with_inference(meta: TensorMeta, new_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Resolve -1 dimension in reshape."""
    if -1 not in new_shape:
        return new_shape

    # Count elements
    total = meta.element_count()
    known = 1
    unknown_idx = -1
    for i, d in enumerate(new_shape):
        if d == -1:
            if unknown_idx != -1:
                raise ValueError("Only one -1 allowed in reshape")
            unknown_idx = i
        else:
            known *= d

    if total % known != 0:
        raise ValueError(f"Cannot reshape {meta.shape} to {new_shape}")

    inferred = total // known
    return tuple(inferred if i == unknown_idx else d for i, d in enumerate(new_shape))
