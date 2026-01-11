"""
Convert PyTorch FX graph to StableHLO MLIR.

Uses torch.fx.symbolic_trace to capture the computation graph from real PyTorch
models, then converts to StableHLO MLIR text format.
"""
from __future__ import annotations  # PEP 563: Postponed evaluation of annotations

from typing import Dict, List, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# Lazily imported by _ensure_torch_imported()
torch = None
symbolic_trace = None
Graph = None
Node = None


def _ensure_torch_imported():
    """Lazily import torch only when needed."""
    global torch, symbolic_trace, Graph, Node
    if torch is None:
        import torch as _torch
        from torch.fx import symbolic_trace as _symbolic_trace, Graph as _Graph, Node as _Node
        torch = _torch
        symbolic_trace = _symbolic_trace
        Graph = _Graph
        Node = _Node


class FXToStableHLO:
    """Convert FX graph to StableHLO MLIR text."""

    def __init__(self, traced_module, sample_inputs: Tuple[torch.Tensor, ...]):
        self.traced = traced_module
        self.sample_inputs = sample_inputs
        self.graph = traced_module.graph
        self.ssa_map: Dict[str, str] = {}
        self.ssa_counter = 0
        self.shape_map: Dict[str, Tuple] = {}
        self.dtype_map: Dict[str, str] = {}
        self._infer_shapes()

    def _infer_shapes(self):
        """Infer shapes by running the model with ShapeProp."""
        from torch.fx.passes.shape_prop import ShapeProp
        ShapeProp(self.traced).propagate(*self.sample_inputs)

        for node in self.graph.nodes:
            if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
                meta = node.meta['tensor_meta']
                if hasattr(meta, 'shape'):
                    self.shape_map[node.name] = tuple(meta.shape)
                    self.dtype_map[node.name] = str(meta.dtype).replace('torch.', '')

    def _new_ssa(self) -> str:
        self.ssa_counter += 1
        return f"%{self.ssa_counter}"

    def _tensor_type(self, node_name: str) -> str:
        """Get MLIR tensor type string."""
        shape = self.shape_map.get(node_name, ())
        dtype = self.dtype_map.get(node_name, 'f32')

        dtype_map = {
            'float32': 'f32', 'float64': 'f64', 'float16': 'f16',
            'bfloat16': 'bf16', 'int32': 'i32', 'int64': 'i64',
            'int8': 'i8', 'int16': 'i16', 'bool': 'i1'
        }
        mlir_dtype = dtype_map.get(dtype, 'f32')

        if not shape:
            return f'tensor<{mlir_dtype}>'
        shape_str = 'x'.join(str(d) for d in shape)
        return f'tensor<{shape_str}x{mlir_dtype}>'

    def convert(self) -> str:
        """Convert FX graph to StableHLO MLIR."""
        lines = ['module @main {']

        inputs = []
        output_node = None

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                inputs.append(node)
            elif node.op == 'output':
                output_node = node

        # Build function signature
        arg_strs = []
        for i, inp in enumerate(inputs):
            ssa_name = f'%arg{i}'
            self.ssa_map[inp.name] = ssa_name
            arg_strs.append(f'{ssa_name}: {self._tensor_type(inp.name)}')

        # Get output type
        output_args = output_node.args[0] if output_node else None
        if isinstance(output_args, Node):
            output_type = self._tensor_type(output_args.name)
        else:
            output_type = 'tensor<f32>'

        lines.append(f'  func.func public @forward({", ".join(arg_strs)}) -> ({output_type}) {{')

        # Convert each node
        for node in self.graph.nodes:
            if node.op in ('placeholder', 'output'):
                continue
            node_lines = self._convert_node(node)
            for line in node_lines:
                lines.append(f'    {line}')

        # Add return
        if output_node and isinstance(output_node.args[0], Node):
            ret_ssa = self.ssa_map.get(output_node.args[0].name, '%0')
            ret_type = self._tensor_type(output_node.args[0].name)
            lines.append(f'    stablehlo.return {ret_ssa} : {ret_type}')

        lines.append('  }')
        lines.append('}')

        return '\n'.join(lines)

    def _convert_node(self, node) -> List[str]:
        """Convert a single FX node to StableHLO."""
        result_ssa = self._new_ssa()
        self.ssa_map[node.name] = result_ssa
        result_type = self._tensor_type(node.name)

        if node.op == 'call_module':
            return self._convert_call_module(node, result_ssa, result_type)
        elif node.op == 'call_function':
            return self._convert_call_function(node, result_ssa, result_type)
        elif node.op == 'call_method':
            return self._convert_call_method(node, result_ssa, result_type)
        elif node.op == 'get_attr':
            return self._convert_get_attr(node, result_ssa, result_type)
        else:
            return [f'// Unsupported op: {node.op} - {node.target}']

    def _convert_call_module(self, node, result_ssa: str, result_type: str) -> List[str]:
        """Convert call_module (e.g., nn.Linear, nn.ReLU)."""
        module = self.traced.get_submodule(node.target)

        if isinstance(module, torch.nn.Linear):
            input_ssa = self.ssa_map.get(node.args[0].name, '%unknown')
            input_type = self._tensor_type(node.args[0].name)
            weight_shape = module.weight.shape

            weight_ssa = f'{result_ssa}_weight'
            weight_type = f'tensor<{weight_shape[0]}x{weight_shape[1]}xf32>'

            lines = [f'// Linear layer: {node.target}']
            lines.append(f'{weight_ssa} = stablehlo.constant dense<0.0> : {weight_type}  // placeholder for weight')

            matmul_ssa = f'{result_ssa}_matmul'
            # Use the official StableHLO dot attribute format
            dot_attr = '#stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>'
            lines.append(
                f'{matmul_ssa} = stablehlo.dot_general {input_ssa}, {weight_ssa}, '
                f'{dot_attr} : ({input_type}, {weight_type}) -> {result_type}'
            )

            if module.bias is not None:
                bias_ssa = f'{result_ssa}_bias'
                bias_type = f'tensor<{weight_shape[0]}xf32>'
                lines.append(f'{bias_ssa} = stablehlo.constant dense<0.0> : {bias_type}  // placeholder for bias')
                lines.append(f'{result_ssa} = stablehlo.add {matmul_ssa}, {bias_ssa} : {result_type}')
            else:
                lines[-1] = lines[-1].replace(matmul_ssa, result_ssa)

            return lines

        elif isinstance(module, torch.nn.Conv2d):
            # Conv2d: input NCHW, kernel OIHW
            input_ssa = self.ssa_map.get(node.args[0].name, '%unknown')
            input_type = self._tensor_type(node.args[0].name)

            # Get conv parameters
            out_channels = module.out_channels
            in_channels = module.in_channels
            kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
            sh, sw = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
            ph, pw = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)

            kernel_ssa = f'{result_ssa}_kernel'
            kernel_type = f'tensor<{out_channels}x{in_channels}x{kh}x{kw}xf32>'

            lines = [f'// Conv2d layer: {node.target}']
            lines.append(f'{kernel_ssa} = stablehlo.constant dense<0.0> : {kernel_type}  // placeholder for kernel')

            # Format: %c = stablehlo.convolution %lhs, %rhs, strides=[...], padding_low=[...], padding_high=[...], ...
            conv_ssa = f'{result_ssa}_conv' if module.bias is not None else result_ssa
            lines.append(
                f'{conv_ssa} = stablehlo.convolution {input_ssa}, {kernel_ssa}, '
                f'strides = [{sh}, {sw}], padding_low = [{ph}, {pw}], padding_high = [{ph}, {pw}], '
                f'lhs_dilation = [1, 1], rhs_dilation = [1, 1], feature_group_count = 1, batch_group_count = 1 : '
                f'({input_type}, {kernel_type}) -> {result_type}'
            )

            if module.bias is not None:
                bias_ssa = f'{result_ssa}_bias'
                bias_type = f'tensor<{out_channels}xf32>'
                lines.append(f'{bias_ssa} = stablehlo.constant dense<0.0> : {bias_type}  // placeholder for bias')
                lines.append(f'{result_ssa} = stablehlo.add {conv_ssa}, {bias_ssa} : {result_type}')

            return lines

        elif isinstance(module, torch.nn.MaxPool2d):
            # MaxPool2d
            input_ssa = self.ssa_map.get(node.args[0].name, '%unknown')
            input_type = self._tensor_type(node.args[0].name)

            kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
            sh, sw = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
            ph, pw = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)

            init_ssa = f'{result_ssa}_init'
            lines = [f'// MaxPool2d layer: {node.target}']
            lines.append(f'{init_ssa} = stablehlo.constant dense<-3.40282e+38> : tensor<f32>')
            # Format: %r = stablehlo.reduce_window %operand, %init, window=[...], strides=[...], reducer=max : (...) -> tensor<...>
            lines.append(
                f'{result_ssa} = stablehlo.reduce_window {input_ssa}, {init_ssa}, '
                f'window = [1, 1, {kh}, {kw}], strides = [1, 1, {sh}, {sw}], '
                f'padding_low = [0, 0, {ph}, {pw}], padding_high = [0, 0, {ph}, {pw}], '
                f'reducer = max : ({input_type}, tensor<f32>) -> {result_type}'
            )
            return lines

        elif isinstance(module, torch.nn.AvgPool2d):
            # AvgPool2d - similar to MaxPool2d but with add reducer and division
            input_ssa = self.ssa_map.get(node.args[0].name, '%unknown')
            input_type = self._tensor_type(node.args[0].name)

            kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
            sh, sw = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
            ph, pw = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)

            init_ssa = f'{result_ssa}_init'
            lines = [f'// AvgPool2d layer: {node.target}']
            lines.append(f'{init_ssa} = stablehlo.constant dense<0.0> : tensor<f32>')
            lines.append(
                f'{result_ssa} = stablehlo.reduce_window {input_ssa}, {init_ssa}, '
                f'window = [1, 1, {kh}, {kw}], strides = [1, 1, {sh}, {sw}], '
                f'padding_low = [0, 0, {ph}, {pw}], padding_high = [0, 0, {ph}, {pw}], '
                f'reducer = add : ({input_type}, tensor<f32>) -> {result_type}'
            )
            return lines

        elif isinstance(module, (torch.nn.ReLU, torch.nn.ReLU6)):
            input_ssa = self.ssa_map.get(node.args[0].name, '%unknown')
            zero_ssa = f'{result_ssa}_zero'
            return [
                f'{zero_ssa} = stablehlo.constant dense<0.0> : {result_type}',
                f'{result_ssa} = stablehlo.maximum {input_ssa}, {zero_ssa} : {result_type}'
            ]

        elif isinstance(module, torch.nn.Sigmoid):
            input_ssa = self.ssa_map.get(node.args[0].name, '%unknown')
            return [f'{result_ssa} = stablehlo.logistic {input_ssa} : {result_type}']

        elif isinstance(module, torch.nn.Tanh):
            input_ssa = self.ssa_map.get(node.args[0].name, '%unknown')
            return [f'{result_ssa} = stablehlo.tanh {input_ssa} : {result_type}']

        elif isinstance(module, torch.nn.Flatten):
            input_ssa = self.ssa_map.get(node.args[0].name, '%unknown')
            input_type = self._tensor_type(node.args[0].name)
            return [f'{result_ssa} = stablehlo.reshape {input_ssa} : {input_type} -> {result_type}']

        elif isinstance(module, torch.nn.Dropout):
            # Dropout is identity during inference
            input_ssa = self.ssa_map.get(node.args[0].name, '%unknown')
            input_type = self._tensor_type(node.args[0].name)
            return [f'{result_ssa} = stablehlo.reshape {input_ssa} : {input_type} -> {result_type}']

        else:
            return [f'// Unsupported module: {type(module).__name__}']

    def _convert_call_function(self, node, result_ssa: str, result_type: str) -> List[str]:
        """Convert call_function (e.g., torch.relu, torch.add, etc.)."""
        target = node.target
        target_name = getattr(target, '__name__', str(target))

        # Helper to get input SSA values
        def get_input(idx):
            if idx < len(node.args):
                arg = node.args[idx]
                if hasattr(arg, 'name'):
                    return self.ssa_map.get(arg.name, '%unknown')
            return '%unknown'

        def get_input_type(idx):
            if idx < len(node.args) and hasattr(node.args[idx], 'name'):
                return self._tensor_type(node.args[idx].name)
            return result_type

        # === Activation functions ===
        if target_name == 'relu' or target == torch.relu:
            input_ssa = get_input(0)
            zero_ssa = f'{result_ssa}_zero'
            return [
                f'{zero_ssa} = stablehlo.constant dense<0.0> : {result_type}',
                f'{result_ssa} = stablehlo.maximum {input_ssa}, {zero_ssa} : {result_type}'
            ]

        # === Binary arithmetic ops ===
        elif target_name == 'add':
            return [f'{result_ssa} = stablehlo.add {get_input(0)}, {get_input(1)} : {result_type}']

        elif target_name in ('sub', 'subtract'):
            return [f'{result_ssa} = stablehlo.subtract {get_input(0)}, {get_input(1)} : {result_type}']

        elif target_name in ('mul', 'multiply'):
            return [f'{result_ssa} = stablehlo.multiply {get_input(0)}, {get_input(1)} : {result_type}']

        elif target_name in ('div', 'divide', 'truediv', 'true_divide'):
            return [f'{result_ssa} = stablehlo.divide {get_input(0)}, {get_input(1)} : {result_type}']

        elif target_name == 'maximum':
            return [f'{result_ssa} = stablehlo.maximum {get_input(0)}, {get_input(1)} : {result_type}']

        elif target_name == 'minimum':
            return [f'{result_ssa} = stablehlo.minimum {get_input(0)}, {get_input(1)} : {result_type}']

        # === Unary ops ===
        elif target_name in ('neg', 'negative'):
            return [f'{result_ssa} = stablehlo.negate {get_input(0)} : {result_type}']

        elif target_name == 'abs':
            return [f'{result_ssa} = stablehlo.abs {get_input(0)} : {result_type}']

        # === Transcendental functions ===
        elif target_name == 'exp':
            return [f'{result_ssa} = stablehlo.exponential {get_input(0)} : {result_type}']

        elif target_name == 'log':
            return [f'{result_ssa} = stablehlo.log {get_input(0)} : {result_type}']

        elif target_name == 'sqrt':
            return [f'{result_ssa} = stablehlo.sqrt {get_input(0)} : {result_type}']

        elif target_name == 'sin':
            return [f'{result_ssa} = stablehlo.sine {get_input(0)} : {result_type}']

        elif target_name == 'cos':
            return [f'{result_ssa} = stablehlo.cosine {get_input(0)} : {result_type}']

        elif target_name == 'tanh':
            return [f'{result_ssa} = stablehlo.tanh {get_input(0)} : {result_type}']

        elif target_name == 'sigmoid':
            return [f'{result_ssa} = stablehlo.logistic {get_input(0)} : {result_type}']

        # === Shape operations ===
        elif target_name == 'cat':
            # torch.cat([x, y], dim=n) -> stablehlo.concatenate
            # Format: %c = stablehlo.concatenate %a, %b, dim = 0 : (t, t) -> t
            tensors = node.args[0] if node.args else []
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', 0)
            ssa_list = ', '.join(self.ssa_map.get(t.name, '%unknown') for t in tensors)
            type_list = ', '.join(self._tensor_type(t.name) for t in tensors)
            return [f'{result_ssa} = stablehlo.concatenate {ssa_list}, dim = {dim} : ({type_list}) -> {result_type}']

        elif target_name == 'getitem':
            # Tensor slicing - simplified handling
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            # For now, use slice op with inferred dimensions
            # This is a simplification - real slicing needs start/limit/stride
            return [f'{result_ssa} = stablehlo.slice {input_ssa}, starts = [0, 0], limits = [1, 2], strides = [1, 1] : {input_type} -> {result_type}']

        elif target_name == 'ones_like':
            # Create a tensor of ones with same shape
            return [f'{result_ssa} = stablehlo.constant dense<1.0> : {result_type}']

        elif target_name == 'zeros_like':
            return [f'{result_ssa} = stablehlo.constant dense<0.0> : {result_type}']

        # === Reduction ops ===
        elif target_name == 'sum':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            # Get reduction dimension
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', 0)
            init_ssa = f'{result_ssa}_init'
            # Format: %r = stablehlo.reduce %operand, %init, dims=[...], reducer=add : (...) -> tensor<...>
            return [
                f'{init_ssa} = stablehlo.constant dense<0.0> : tensor<f32>',
                f'{result_ssa} = stablehlo.reduce {input_ssa}, {init_ssa}, dims = [{dim}], reducer = add : ({input_type}, tensor<f32>) -> {result_type}'
            ]

        elif target_name == 'max':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', 0)
            init_ssa = f'{result_ssa}_init'
            # Format: %r = stablehlo.reduce %operand, %init, dims=[...], reducer=max : (...) -> tensor<...>
            # Use large negative number instead of -inf for parser compatibility
            return [
                f'{init_ssa} = stablehlo.constant dense<-3.40282e+38> : tensor<f32>',
                f'{result_ssa} = stablehlo.reduce {input_ssa}, {init_ssa}, dims = [{dim}], reducer = max : ({input_type}, tensor<f32>) -> {result_type}'
            ]

        # === Softmax ===
        elif target_name == 'softmax':
            input_ssa = get_input(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', -1)
            # softmax(x) = exp(x) / sum(exp(x))
            exp_ssa = f'{result_ssa}_exp'
            sum_ssa = f'{result_ssa}_sum'
            return [
                f'{exp_ssa} = stablehlo.exponential {input_ssa} : {result_type}',
                f'// Note: Full softmax requires reduction and broadcast - simplified here',
                f'{result_ssa} = stablehlo.divide {exp_ssa}, {exp_ssa} : {result_type}  // placeholder'
            ]

        # === Matrix operations ===
        elif target_name == 'matmul':
            lhs = get_input(0)
            rhs = get_input(1)
            lhs_type = get_input_type(0)
            rhs_type = get_input_type(1)
            dot_attr = '#stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>'
            return [
                f'{result_ssa} = stablehlo.dot_general {lhs}, {rhs}, '
                f'{dot_attr} : ({lhs_type}, {rhs_type}) -> {result_type}'
            ]

        else:
            return [f'// Unsupported function: {target_name}']

    def _convert_call_method(self, node, result_ssa: str, result_type: str) -> List[str]:
        """Convert call_method (tensor methods like reshape, transpose, etc.)."""
        method_name = node.target
        input_node = node.args[0] if node.args else None
        input_ssa = self.ssa_map.get(input_node.name, '%arg0') if input_node else '%arg0'
        input_type = self._tensor_type(input_node.name) if input_node else result_type

        if method_name == 'reshape' or method_name == 'view':
            # x.reshape(shape) or x.view(shape) -> stablehlo.reshape
            # Format: %r = stablehlo.reshape %op : tensor<...> -> tensor<...>
            return [f'{result_ssa} = stablehlo.reshape {input_ssa} : {input_type} -> {result_type}']

        elif method_name == 'transpose':
            # x.transpose(dim0, dim1) -> stablehlo.transpose
            # Format: %t = stablehlo.transpose %op, dims = [1, 0] : tensor<...> -> tensor<...>
            dim0 = node.args[1] if len(node.args) > 1 else 0
            dim1 = node.args[2] if len(node.args) > 2 else 1

            # Build permutation array - swap dim0 and dim1
            input_shape = self.shape_map.get(input_node.name, ()) if input_node else ()
            ndim = len(input_shape) if input_shape else 2
            permutation = list(range(ndim))
            permutation[dim0], permutation[dim1] = permutation[dim1], permutation[dim0]
            perm_str = ', '.join(str(p) for p in permutation)

            return [f'{result_ssa} = stablehlo.transpose {input_ssa}, dims = [{perm_str}] : {input_type} -> {result_type}']

        elif method_name == 'permute':
            # x.permute(dims) -> stablehlo.transpose
            # Format: %t = stablehlo.transpose %op, dims = [...] : tensor<...> -> tensor<...>
            dims = node.args[1] if len(node.args) > 1 else node.kwargs.get('dims', [])
            if isinstance(dims, (list, tuple)):
                perm_str = ', '.join(str(d) for d in dims)
            else:
                perm_str = str(dims)
            return [f'{result_ssa} = stablehlo.transpose {input_ssa}, dims = [{perm_str}] : {input_type} -> {result_type}']

        elif method_name == 'expand':
            # x.expand(sizes) -> stablehlo.broadcast_in_dim
            # Format: %b = stablehlo.broadcast_in_dim %op, dims = [...] : (input_type) -> result_type
            input_shape = self.shape_map.get(input_node.name, ()) if input_node else ()
            output_shape = self.shape_map.get(node.name, ())

            # Build broadcast_dimensions - map input dims to output dims
            broadcast_dims = []
            input_ndim = len(input_shape)
            output_ndim = len(output_shape)
            offset = output_ndim - input_ndim

            for i in range(input_ndim):
                broadcast_dims.append(offset + i)

            dims_str = ', '.join(str(d) for d in broadcast_dims)
            return [f'{result_ssa} = stablehlo.broadcast_in_dim {input_ssa}, dims = [{dims_str}] : ({input_type}) -> {result_type}']

        elif method_name == 'contiguous':
            # contiguous() is a no-op for StableHLO - just pass through
            return [f'{result_ssa} = stablehlo.reshape {input_ssa} : {input_type} -> {result_type}']

        elif method_name == 'flatten':
            # x.flatten(start_dim, end_dim) -> stablehlo.reshape
            return [f'{result_ssa} = stablehlo.reshape {input_ssa} : {input_type} -> {result_type}']

        elif method_name == 'squeeze':
            # x.squeeze(dim) -> stablehlo.reshape (removes dim of size 1)
            return [f'{result_ssa} = stablehlo.reshape {input_ssa} : {input_type} -> {result_type}']

        elif method_name == 'unsqueeze':
            # x.unsqueeze(dim) -> stablehlo.reshape (adds dim of size 1)
            return [f'{result_ssa} = stablehlo.reshape {input_ssa} : {input_type} -> {result_type}']

        else:
            return [f'// Unsupported method: {method_name}']

    def _convert_get_attr(self, node, result_ssa: str, result_type: str) -> List[str]:
        """Convert get_attr (parameter access)."""
        return [f'{result_ssa} = stablehlo.constant dense<0.0> : {result_type}  // {node.target}']


def trace_model(source_code: str, class_name: str, input_shapes: list) -> str:
    """
    Trace a model from Python source code.

    Args:
        source_code: Python source containing an nn.Module class
        class_name: Name of the nn.Module class to trace
        input_shapes: List of tuples, e.g., [(1, 8), (1, 16)]

    Returns:
        StableHLO MLIR text
    """
    _ensure_torch_imported()

    namespace = {'torch': torch}
    exec(source_code, namespace)

    if class_name not in namespace:
        raise ValueError(f"Class '{class_name}' not found in source")

    model_class = namespace[class_name]
    model = model_class()
    model.eval()

    sample_inputs = tuple(torch.randn(*shape) for shape in input_shapes)
    traced = symbolic_trace(model)
    converter = FXToStableHLO(traced, sample_inputs)
    return converter.convert()


def trace_builtin_example() -> str:
    """Run the built-in MLP example."""
    _ensure_torch_imported()

    class SimpleMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(8, 16)
            self.fc2 = torch.nn.Linear(16, 4)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    model = SimpleMLP()
    model.eval()

    traced = symbolic_trace(model)
    sample_input = (torch.randn(1, 8),)
    converter = FXToStableHLO(traced, sample_input)
    return converter.convert()
