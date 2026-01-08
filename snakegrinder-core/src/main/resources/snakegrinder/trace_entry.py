"""
Entry point for mock tracing.

This module provides the main interface called from Java to trace
Python source code and emit StableHLO MLIR.
"""

import sys
from tracer import TracedTensor, TensorMeta
from graph_ir import ComputationGraph
from stablehlo_emitter import StableHLOEmitter


def setup_mock_modules(framework='torch'):
    """
    Install mock modules into sys.modules so user imports work.

    Args:
        framework: 'torch' or 'jax' - primary framework to mock
    """
    import mock_torch
    import mock_jax

    # Always install both, user code might import either
    sys.modules['torch'] = mock_torch
    sys.modules['torch.nn'] = mock_torch.nn
    sys.modules['torch.nn.functional'] = mock_torch.nn.functional

    sys.modules['jax'] = mock_jax
    sys.modules['jax.numpy'] = mock_jax.numpy
    sys.modules['jnp'] = mock_jax.numpy
    sys.modules['jax.random'] = mock_jax.random
    sys.modules['jax.nn'] = mock_jax.nn
    sys.modules['jax.lax'] = mock_jax.lax


def trace(fn, input_specs, name=None):
    """
    Trace a Python function and capture its computation graph.

    Args:
        fn: A callable that takes tensor arguments
        input_specs: List of (shape, dtype) tuples describing inputs
        name: Optional name for the traced function

    Returns:
        ComputationGraph representing the traced computation

    Example:
        def my_model(x, w):
            return x @ w

        graph = trace(my_model, [
            ((2, 3), 'f32'),
            ((3, 4), 'f32')
        ])
    """
    func_name = name or (fn.__name__ if hasattr(fn, '__name__') else 'traced')
    graph = ComputationGraph(name=func_name)

    # Set up tracing context
    old_graph = TracedTensor._current_graph
    TracedTensor._current_graph = graph

    try:
        # Create input placeholders
        inputs = []
        for i, spec in enumerate(input_specs):
            if isinstance(spec, tuple) and len(spec) == 2:
                shape, dtype = spec
            else:
                # Assume just shape, default to f32
                shape = spec
                dtype = 'f32'

            if isinstance(shape, int):
                shape = (shape,)

            meta = TensorMeta(shape=tuple(shape), dtype=dtype)
            tensor = graph.add_input(meta, name=f"arg{i}")
            inputs.append(tensor)

        # Execute the function with traced tensors
        result = fn(*inputs)

        # Handle multiple outputs
        if isinstance(result, tuple):
            outputs = list(result)
        elif isinstance(result, list):
            outputs = result
        else:
            outputs = [result]

        graph.set_outputs(outputs)
        return graph

    finally:
        TracedTensor._current_graph = old_graph


def trace_to_stablehlo(fn, input_specs, name=None) -> str:
    """
    Trace a function and directly emit StableHLO MLIR.

    Args:
        fn: A callable that takes tensor arguments
        input_specs: List of (shape, dtype) tuples or just shapes
        name: Optional function name

    Returns:
        StableHLO MLIR text
    """
    graph = trace(fn, input_specs, name)
    emitter = StableHLOEmitter(graph)
    return emitter.emit()


def trace_source_code(source_code, function_name, input_specs, framework='torch'):
    """
    Trace Python source code and return StableHLO.

    This is the main entry point called from Java.

    Args:
        source_code: Python source as a string
        function_name: Name of the function to trace
        input_specs: List of (shape, dtype) tuples
        framework: 'torch' or 'jax' - which mock module style

    Returns:
        dict with 'status', 'mlir', 'error', 'warnings' keys
    """
    result = {
        'status': 'ok',
        'mlir': None,
        'error': None,
        'warnings': [],
        'function_name': function_name,
        'framework': framework,
        'input_count': len(input_specs),
    }

    try:
        # Set up mock modules
        setup_mock_modules(framework)

        # Build namespace with mock modules
        namespace = {'__builtins__': __builtins__}

        if framework == 'torch':
            import mock_torch
            namespace['torch'] = mock_torch
            namespace['F'] = mock_torch.nn.functional
        else:
            import mock_jax
            namespace['jax'] = mock_jax
            namespace['jnp'] = mock_jax.numpy

        # Execute the source to define the function
        exec(source_code, namespace)

        if function_name not in namespace:
            result['status'] = 'error'
            result['error'] = f"Function '{function_name}' not found in source code"
            return result

        fn = namespace[function_name]

        # Parse input_specs if needed
        parsed_specs = _parse_input_specs(input_specs)

        # Trace and emit
        mlir = trace_to_stablehlo(fn, parsed_specs, name=function_name)
        result['mlir'] = mlir

        return result

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        return result


def _parse_input_specs(input_specs):
    """
    Parse input specifications into (shape, dtype) tuples.

    Accepts:
    - List of (shape_tuple, dtype_str)
    - List of shape_tuples (defaults to f32)
    - List of lists/tuples that are shapes
    """
    parsed = []
    for spec in input_specs:
        if isinstance(spec, dict):
            # {'shape': (2, 3), 'dtype': 'f32'}
            shape = tuple(spec.get('shape', ()))
            dtype = spec.get('dtype', 'f32')
            parsed.append((shape, dtype))
        elif isinstance(spec, (list, tuple)):
            if len(spec) == 2 and isinstance(spec[0], (list, tuple)) and isinstance(spec[1], str):
                # ((2, 3), 'f32')
                parsed.append((tuple(spec[0]), spec[1]))
            else:
                # Just a shape like (2, 3)
                parsed.append((tuple(spec), 'f32'))
        else:
            raise ValueError(f"Invalid input spec: {spec}")
    return parsed


# --- Convenience functions for direct testing ---

def trace_matmul_example():
    """
    Trace a simple matrix multiplication for testing.
    Returns StableHLO MLIR.
    """
    setup_mock_modules('torch')

    def matmul_fn(a, b):
        return a @ b

    return trace_to_stablehlo(matmul_fn, [
        ((2, 3), 'f32'),
        ((3, 4), 'f32')
    ], name='matmul_fn')


def trace_mlp_example():
    """
    Trace a simple MLP layer for testing.
    Returns StableHLO MLIR.
    """
    setup_mock_modules('torch')
    import mock_torch as torch

    def mlp_layer(x, w1, w2):
        h = x @ w1
        h = torch.nn.functional.relu(h)
        return h @ w2

    return trace_to_stablehlo(mlp_layer, [
        ((4, 8), 'f32'),    # x: batch=4, features=8
        ((8, 16), 'f32'),   # w1: 8->16
        ((16, 2), 'f32'),   # w2: 16->2
    ], name='mlp_layer')


# For direct execution testing
if __name__ == '__main__':
    print("=== Matmul Example ===")
    print(trace_matmul_example())
    print()
    print("=== MLP Example ===")
    print(trace_mlp_example())
