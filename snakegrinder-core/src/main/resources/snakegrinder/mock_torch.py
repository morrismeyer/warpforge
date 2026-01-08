"""
Mock PyTorch module for tracing.

Implements just enough of the torch API to capture computations
without any actual tensor math or native code dependencies.
"""

from tracer import TracedTensor, TensorMeta

__version__ = "mock-2.0.0"

# --- Dtype constants ---

class dtype:
    """Mock dtype class."""
    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return f"torch.{self._name}"

    def __eq__(self, other):
        if isinstance(other, dtype):
            return self._name == other._name
        return False


float32 = dtype('float32')
float64 = dtype('float64')
float16 = dtype('float16')
bfloat16 = dtype('bfloat16')
int32 = dtype('int32')
int64 = dtype('int64')
int16 = dtype('int16')
int8 = dtype('int8')
bool = dtype('bool')

# Aliases
float = float32
double = float64
int = int32
long = int64


def _resolve_dtype(dt) -> str:
    """Convert dtype to internal string representation."""
    if dt is None:
        return 'f32'
    if isinstance(dt, dtype):
        name = dt._name
        mapping = {
            'float32': 'f32', 'float64': 'f64', 'float16': 'f16',
            'bfloat16': 'bf16', 'int32': 'i32', 'int64': 'i64',
            'int16': 'i16', 'int8': 'i8', 'bool': 'i1'
        }
        return mapping.get(name, 'f32')
    return 'f32'


# --- Tensor creation functions ---

def tensor(data, dtype=None, device=None, requires_grad=False):
    """Create a tensor from data. In tracing mode, this creates an input."""
    # For tracing, we need shape info. Try to infer from data.
    if isinstance(data, (list, tuple)):
        shape = _infer_shape(data)
    else:
        shape = ()

    dt = _resolve_dtype(dtype)
    meta = TensorMeta(shape=shape, dtype=dt)

    graph = TracedTensor._current_graph
    if graph is None:
        raise RuntimeError("No active trace context. Call trace() first.")

    return graph.add_input(meta)


def _infer_shape(data):
    """Infer shape from nested lists/tuples."""
    if not isinstance(data, (list, tuple)):
        return ()
    if len(data) == 0:
        return (0,)
    inner = _infer_shape(data[0])
    return (len(data),) + inner


def zeros(*size, dtype=None, device=None, requires_grad=False):
    """Create a zero tensor placeholder."""
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = tuple(size[0])
    return _create_input(size, dtype)


def ones(*size, dtype=None, device=None, requires_grad=False):
    """Create a ones tensor placeholder."""
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = tuple(size[0])
    return _create_input(size, dtype)


def randn(*size, dtype=None, device=None, requires_grad=False):
    """Create a random tensor placeholder."""
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = tuple(size[0])
    return _create_input(size, dtype)


def rand(*size, dtype=None, device=None, requires_grad=False):
    """Create a random tensor placeholder."""
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = tuple(size[0])
    return _create_input(size, dtype)


def empty(*size, dtype=None, device=None, requires_grad=False):
    """Create an empty tensor placeholder."""
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = tuple(size[0])
    return _create_input(size, dtype)


def _create_input(shape, dtype):
    """Create a traced input tensor."""
    dt = _resolve_dtype(dtype)
    meta = TensorMeta(shape=shape, dtype=dt)
    graph = TracedTensor._current_graph
    if graph is None:
        raise RuntimeError("No active trace context")
    return graph.add_input(meta)


# --- Functional operations ---

def matmul(input, other):
    """Matrix multiplication."""
    return input @ other


def mm(input, mat2):
    """Matrix-matrix multiplication."""
    return input @ mat2


def bmm(input, mat2):
    """Batched matrix-matrix multiplication."""
    return input @ mat2


def add(input, other, alpha=1):
    """Add tensors."""
    if alpha != 1:
        other = other * alpha
    return input + other


def mul(input, other):
    """Multiply tensors."""
    return input * other


def sub(input, other, alpha=1):
    """Subtract tensors."""
    if alpha != 1:
        other = other * alpha
    return input - other


def div(input, other):
    """Divide tensors."""
    return input / other


def neg(input):
    """Negate tensor."""
    return -input


def reshape(input, shape):
    """Reshape tensor."""
    return input.reshape(shape)


def transpose(input, dim0, dim1):
    """Transpose two dimensions."""
    ndim = len(input.shape)
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return input.transpose(perm)


def permute(input, dims):
    """Permute tensor dimensions."""
    return input.transpose(dims)


def squeeze(input, dim=None):
    """Remove size-1 dimensions."""
    if dim is None:
        new_shape = tuple(d for d in input.shape if d != 1)
    else:
        new_shape = tuple(d for i, d in enumerate(input.shape) if i != dim or d != 1)
    return input.reshape(new_shape)


def unsqueeze(input, dim):
    """Add a size-1 dimension."""
    shape = list(input.shape)
    shape.insert(dim, 1)
    return input.reshape(tuple(shape))


def cat(tensors, dim=0):
    """Concatenate tensors. Simplified: records as custom op."""
    graph = TracedTensor._current_graph
    if graph is None:
        raise RuntimeError("No active trace context")

    # Compute output shape
    shapes = [t.shape for t in tensors]
    new_shape = list(shapes[0])
    new_shape[dim] = sum(s[dim] for s in shapes)
    result_meta = TensorMeta(tuple(new_shape), tensors[0].meta.dtype)

    node = graph.add_op('concatenate', list(tensors), result_meta,
                        attrs={'dimension': dim})
    result = TracedTensor(result_meta, producing_op=node)
    node.outputs = [result]
    return result


def stack(tensors, dim=0):
    """Stack tensors along new dimension."""
    # First unsqueeze each tensor
    unsqueezed = [unsqueeze(t, dim) for t in tensors]
    return cat(unsqueezed, dim)


def sum(input, dim=None, keepdim=False):
    """Sum reduction."""
    return input.sum(axis=dim, keepdims=keepdim)


def mean(input, dim=None, keepdim=False):
    """Mean reduction."""
    return input.mean(axis=dim, keepdims=keepdim)


# --- Neural network module ---

class nn:
    """Mock torch.nn namespace."""

    class Module:
        """Base class for neural network modules."""
        def __init__(self):
            pass

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, *args, **kwargs):
            raise NotImplementedError

    class functional:
        """Mock torch.nn.functional namespace."""

        @staticmethod
        def relu(input, inplace=False):
            """ReLU activation."""
            graph = TracedTensor._current_graph
            if graph is None:
                raise RuntimeError("No active trace context")

            result_meta = TensorMeta(input.shape, input.meta.dtype)
            node = graph.add_op('relu', [input], result_meta)
            result = TracedTensor(result_meta, producing_op=node)
            node.outputs = [result]
            return result

        @staticmethod
        def gelu(input, approximate='none'):
            """GELU activation (recorded as custom op)."""
            graph = TracedTensor._current_graph
            if graph is None:
                raise RuntimeError("No active trace context")

            result_meta = TensorMeta(input.shape, input.meta.dtype)
            node = graph.add_op('gelu', [input], result_meta,
                                attrs={'approximate': approximate})
            result = TracedTensor(result_meta, producing_op=node)
            node.outputs = [result]
            return result

        @staticmethod
        def sigmoid(input):
            """Sigmoid activation."""
            graph = TracedTensor._current_graph
            if graph is None:
                raise RuntimeError("No active trace context")

            result_meta = TensorMeta(input.shape, input.meta.dtype)
            node = graph.add_op('logistic', [input], result_meta)
            result = TracedTensor(result_meta, producing_op=node)
            node.outputs = [result]
            return result

        @staticmethod
        def tanh(input):
            """Tanh activation."""
            graph = TracedTensor._current_graph
            if graph is None:
                raise RuntimeError("No active trace context")

            result_meta = TensorMeta(input.shape, input.meta.dtype)
            node = graph.add_op('tanh', [input], result_meta)
            result = TracedTensor(result_meta, producing_op=node)
            node.outputs = [result]
            return result

        @staticmethod
        def softmax(input, dim=-1):
            """Softmax."""
            graph = TracedTensor._current_graph
            if graph is None:
                raise RuntimeError("No active trace context")

            result_meta = TensorMeta(input.shape, input.meta.dtype)
            node = graph.add_op('softmax', [input], result_meta,
                                attrs={'dimension': dim})
            result = TracedTensor(result_meta, producing_op=node)
            node.outputs = [result]
            return result

        @staticmethod
        def linear(input, weight, bias=None):
            """Linear transformation: y = x @ W.T + b"""
            # Weight is (out_features, in_features), need to transpose
            result = input @ weight.T
            if bias is not None:
                result = result + bias
            return result

        @staticmethod
        def dropout(input, p=0.5, training=True, inplace=False):
            """Dropout - identity during tracing."""
            return input

    # Alias for convenience
    F = functional


# --- Random module ---

class _random:
    _seed = 0

    @staticmethod
    def manual_seed(seed):
        _random._seed = seed


def manual_seed(seed):
    """Set random seed (no-op in tracing)."""
    _random.manual_seed(seed)


# --- Device handling (no-op) ---

class device:
    def __init__(self, type, index=None):
        self.type = type
        self.index = index


def cuda_is_available():
    """CUDA availability check."""
    return False


class cuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        return 0


# --- Autograd (no-op stubs) ---

def no_grad():
    """Context manager for disabling gradients."""
    class NoGrad:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    return NoGrad()


def enable_grad():
    """Context manager for enabling gradients."""
    class EnableGrad:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    return EnableGrad()


# --- Tensor type (alias to TracedTensor for isinstance checks) ---
Tensor = TracedTensor
