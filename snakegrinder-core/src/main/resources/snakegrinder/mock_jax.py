"""
Mock JAX module for tracing.

Implements JAX-style API that records operations to the computation graph.
No actual computation or native dependencies.
"""

from tracer import TracedTensor, TensorMeta

__version__ = "mock-0.4.0"


def _resolve_dtype(dt) -> str:
    """Convert dtype to internal string representation."""
    if dt is None:
        return 'f32'
    if isinstance(dt, str):
        mapping = {
            'float32': 'f32', 'float64': 'f64', 'float16': 'f16',
            'bfloat16': 'bf16', 'int32': 'i32', 'int64': 'i64',
            'int16': 'i16', 'int8': 'i8', 'bool': 'i1'
        }
        return mapping.get(dt, dt)
    return 'f32'


def _create_input(shape, dtype):
    """Create a traced input tensor."""
    if isinstance(shape, int):
        shape = (shape,)
    dt = _resolve_dtype(dtype)
    meta = TensorMeta(shape=tuple(shape), dtype=dt)
    graph = TracedTensor._current_graph
    if graph is None:
        raise RuntimeError("No active trace context")
    return graph.add_input(meta)


# --- jax.numpy module ---

class numpy:
    """Mock jax.numpy namespace."""

    # Dtype constants
    float32 = 'float32'
    float64 = 'float64'
    float16 = 'float16'
    bfloat16 = 'bfloat16'
    int32 = 'int32'
    int64 = 'int64'
    int16 = 'int16'
    int8 = 'int8'
    bool_ = 'bool'

    @staticmethod
    def array(data, dtype=None):
        """Create array from data."""
        if isinstance(data, TracedTensor):
            return data
        if isinstance(data, (list, tuple)):
            shape = _infer_shape(data)
        else:
            shape = ()
        return _create_input(shape, dtype)

    @staticmethod
    def zeros(shape, dtype=None):
        """Create zeros array."""
        return _create_input(shape, dtype)

    @staticmethod
    def ones(shape, dtype=None):
        """Create ones array."""
        return _create_input(shape, dtype)

    @staticmethod
    def empty(shape, dtype=None):
        """Create empty array."""
        return _create_input(shape, dtype)

    @staticmethod
    def matmul(a, b):
        """Matrix multiplication."""
        return a @ b

    @staticmethod
    def dot(a, b):
        """Dot product / matrix multiplication."""
        return a @ b

    @staticmethod
    def add(a, b):
        """Elementwise addition."""
        return a + b

    @staticmethod
    def subtract(a, b):
        """Elementwise subtraction."""
        return a - b

    @staticmethod
    def multiply(a, b):
        """Elementwise multiplication."""
        return a * b

    @staticmethod
    def divide(a, b):
        """Elementwise division."""
        return a / b

    @staticmethod
    def negative(a):
        """Elementwise negation."""
        return -a

    @staticmethod
    def reshape(a, newshape):
        """Reshape array."""
        return a.reshape(newshape)

    @staticmethod
    def transpose(a, axes=None):
        """Transpose array."""
        if axes is None:
            axes = tuple(reversed(range(len(a.shape))))
        return a.transpose(axes)

    @staticmethod
    def swapaxes(a, axis1, axis2):
        """Swap two axes."""
        ndim = len(a.shape)
        axes = list(range(ndim))
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
        return a.transpose(tuple(axes))

    @staticmethod
    def sum(a, axis=None, keepdims=False):
        """Sum reduction."""
        return a.sum(axis=axis, keepdims=keepdims)

    @staticmethod
    def mean(a, axis=None, keepdims=False):
        """Mean reduction."""
        return a.mean(axis=axis, keepdims=keepdims)

    @staticmethod
    def squeeze(a, axis=None):
        """Remove size-1 dimensions."""
        if axis is None:
            new_shape = tuple(d for d in a.shape if d != 1)
        else:
            new_shape = tuple(d for i, d in enumerate(a.shape) if i != axis or d != 1)
        return a.reshape(new_shape)

    @staticmethod
    def expand_dims(a, axis):
        """Add a dimension."""
        shape = list(a.shape)
        shape.insert(axis, 1)
        return a.reshape(tuple(shape))

    @staticmethod
    def concatenate(arrays, axis=0):
        """Concatenate arrays."""
        graph = TracedTensor._current_graph
        if graph is None:
            raise RuntimeError("No active trace context")

        shapes = [t.shape for t in arrays]
        new_shape = list(shapes[0])
        new_shape[axis] = sum(s[axis] for s in shapes)
        result_meta = TensorMeta(tuple(new_shape), arrays[0].meta.dtype)

        node = graph.add_op('concatenate', list(arrays), result_meta,
                            attrs={'dimension': axis})
        result = TracedTensor(result_meta, producing_op=node)
        node.outputs = [result]
        return result

    @staticmethod
    def stack(arrays, axis=0):
        """Stack arrays along new axis."""
        expanded = [numpy.expand_dims(a, axis) for a in arrays]
        return numpy.concatenate(expanded, axis)


# Alias
jnp = numpy


def _infer_shape(data):
    """Infer shape from nested lists/tuples."""
    if not isinstance(data, (list, tuple)):
        return ()
    if len(data) == 0:
        return (0,)
    inner = _infer_shape(data[0])
    return (len(data),) + inner


# --- jax.random module ---

class random:
    """Mock jax.random namespace."""

    @staticmethod
    def PRNGKey(seed):
        """Create a PRNG key."""
        return _MockPRNGKey(seed)

    @staticmethod
    def split(key, num=2):
        """Split a PRNG key."""
        return [_MockPRNGKey(i) for i in range(num)]

    @staticmethod
    def normal(key, shape, dtype=None):
        """Generate random normal array."""
        return _create_input(shape, dtype)

    @staticmethod
    def uniform(key, shape=(), dtype=None, minval=0.0, maxval=1.0):
        """Generate random uniform array."""
        return _create_input(shape, dtype)

    @staticmethod
    def randint(key, shape, minval, maxval, dtype='int32'):
        """Generate random integers."""
        return _create_input(shape, dtype)


class _MockPRNGKey:
    """Mock PRNG key."""
    def __init__(self, seed):
        self.seed = seed


# --- jax.nn module ---

class nn:
    """Mock jax.nn namespace."""

    @staticmethod
    def relu(x):
        """ReLU activation."""
        graph = TracedTensor._current_graph
        if graph is None:
            raise RuntimeError("No active trace context")

        result_meta = TensorMeta(x.shape, x.meta.dtype)
        node = graph.add_op('relu', [x], result_meta)
        result = TracedTensor(result_meta, producing_op=node)
        node.outputs = [result]
        return result

    @staticmethod
    def gelu(x, approximate=True):
        """GELU activation."""
        graph = TracedTensor._current_graph
        if graph is None:
            raise RuntimeError("No active trace context")

        result_meta = TensorMeta(x.shape, x.meta.dtype)
        node = graph.add_op('gelu', [x], result_meta,
                            attrs={'approximate': approximate})
        result = TracedTensor(result_meta, producing_op=node)
        node.outputs = [result]
        return result

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation."""
        graph = TracedTensor._current_graph
        if graph is None:
            raise RuntimeError("No active trace context")

        result_meta = TensorMeta(x.shape, x.meta.dtype)
        node = graph.add_op('logistic', [x], result_meta)
        result = TracedTensor(result_meta, producing_op=node)
        node.outputs = [result]
        return result

    @staticmethod
    def tanh(x):
        """Tanh activation."""
        graph = TracedTensor._current_graph
        if graph is None:
            raise RuntimeError("No active trace context")

        result_meta = TensorMeta(x.shape, x.meta.dtype)
        node = graph.add_op('tanh', [x], result_meta)
        result = TracedTensor(result_meta, producing_op=node)
        node.outputs = [result]
        return result

    @staticmethod
    def softmax(x, axis=-1):
        """Softmax."""
        graph = TracedTensor._current_graph
        if graph is None:
            raise RuntimeError("No active trace context")

        result_meta = TensorMeta(x.shape, x.meta.dtype)
        node = graph.add_op('softmax', [x], result_meta,
                            attrs={'dimension': axis})
        result = TracedTensor(result_meta, producing_op=node)
        node.outputs = [result]
        return result


# --- JAX transformations (stubs) ---

def jit(fun, static_argnums=None, static_argnames=None, donate_argnums=None):
    """Mock jit - returns function unchanged for tracing."""
    return fun


def grad(fun, argnums=0, has_aux=False):
    """Mock grad - not supported in tracing."""
    raise NotImplementedError("grad() not supported in mock tracer")


def vmap(fun, in_axes=0, out_axes=0):
    """Mock vmap - not supported in tracing."""
    raise NotImplementedError("vmap() not supported in mock tracer")


def pmap(fun, axis_name=None, in_axes=0, out_axes=0):
    """Mock pmap - not supported in tracing."""
    raise NotImplementedError("pmap() not supported in mock tracer")


# --- lax module (low-level ops) ---

class lax:
    """Mock jax.lax namespace."""

    @staticmethod
    def dot_general(lhs, rhs, dimension_numbers, precision=None):
        """General dot product."""
        # For simple cases, delegate to matmul
        return lhs @ rhs

    @staticmethod
    def conv_general_dilated(lhs, rhs, window_strides, padding,
                              lhs_dilation=None, rhs_dilation=None,
                              dimension_numbers=None, feature_group_count=1,
                              batch_group_count=1, precision=None):
        """General convolution - recorded as custom op."""
        graph = TracedTensor._current_graph
        if graph is None:
            raise RuntimeError("No active trace context")

        # Simplified output shape computation
        # In practice this would need proper convolution shape inference
        result_meta = TensorMeta(lhs.shape, lhs.meta.dtype)  # Placeholder
        node = graph.add_op('convolution', [lhs, rhs], result_meta,
                            attrs={'window_strides': window_strides, 'padding': padding})
        result = TracedTensor(result_meta, producing_op=node)
        node.outputs = [result]
        return result

    @staticmethod
    def reduce(operand, init_value, computation, dimensions):
        """General reduction."""
        return operand.sum(axis=dimensions)

    @staticmethod
    def broadcast_in_dim(operand, shape, broadcast_dimensions):
        """Broadcast to shape."""
        return operand.reshape(shape)


# --- Device handling ---

def devices(backend=None):
    """List available devices."""
    return [_MockDevice('cpu', 0)]


def device_count(backend=None):
    """Count available devices."""
    return 1


class _MockDevice:
    """Mock device."""
    def __init__(self, platform, id):
        self.platform = platform
        self.id = id

    def __repr__(self):
        return f"MockDevice({self.platform}:{self.id})"


# Alias for Array type
Array = TracedTensor
