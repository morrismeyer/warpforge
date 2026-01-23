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
            # Handle tuple extraction from multi-return ops (max, min, topk)
            input_node = node.args[0] if node.args else None
            index = node.args[1] if len(node.args) > 1 else 0

            if input_node and hasattr(input_node, 'target'):
                input_target_name = getattr(input_node.target, '__name__', str(input_node.target))

                # Check if extracting from multi-return reduction ops
                if input_target_name in ('max', 'min', 'topk'):
                    if index == 0:
                        # Index 0 = values tensor - the reduce already computed this
                        # Just pass through the SSA from the reduce
                        input_ssa = self.ssa_map.get(input_node.name, '%unknown')
                        self.ssa_map[node.name] = input_ssa  # Alias to the same SSA
                        return [f'// getitem[0] on {input_target_name} - using reduce result directly']
                    elif index == 1:
                        # Index 1 = indices tensor - would need argmax/argmin
                        return [f'// TODO: getitem[1] on {input_target_name} needs argmax/argmin support']

            # General tensor slicing - simplified handling
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

        elif target_name == 'max' or target_name == 'amax':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', 0)

            # For torch.max with dim (returns tuple), compute result type from input shape
            # ShapeProp may not set tensor_meta properly for tuple-returning ops
            input_node = node.args[0] if node.args else None
            input_shape = self.shape_map.get(input_node.name, ()) if input_node and hasattr(input_node, 'name') else ()
            input_dtype = self.dtype_map.get(input_node.name, 'f32') if input_node and hasattr(input_node, 'name') else 'f32'

            # Compute reduced shape (remove the dimension being reduced)
            if input_shape:
                reduced_shape = list(input_shape)
                if isinstance(dim, int) and 0 <= dim < len(reduced_shape):
                    del reduced_shape[dim]
                elif isinstance(dim, int) and dim < 0 and -dim <= len(reduced_shape):
                    del reduced_shape[dim]

                # Build result type from reduced shape
                dtype_map = {'float32': 'f32', 'float64': 'f64', 'float16': 'f16',
                             'bfloat16': 'bf16', 'int32': 'i32', 'int64': 'i64',
                             'int8': 'i8', 'int16': 'i16', 'bool': 'i1'}
                mlir_dtype = dtype_map.get(input_dtype, 'f32')
                if reduced_shape:
                    shape_str = 'x'.join(str(d) for d in reduced_shape)
                    computed_result_type = f'tensor<{shape_str}x{mlir_dtype}>'
                else:
                    computed_result_type = f'tensor<{mlir_dtype}>'
            else:
                # Fallback to what we have
                computed_result_type = result_type

            init_ssa = f'{result_ssa}_init'
            # Format: %r = stablehlo.reduce %operand, %init, dims=[...], reducer=max : (...) -> tensor<...>
            # Use large negative number instead of -inf for parser compatibility
            return [
                f'{init_ssa} = stablehlo.constant dense<-3.40282e+38> : tensor<f32>',
                f'{result_ssa} = stablehlo.reduce {input_ssa}, {init_ssa}, dims = [{dim}], reducer = max : ({input_type}, tensor<f32>) -> {computed_result_type}'
            ]

        elif target_name == 'amin' or target_name == 'min':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', 0)

            # For torch.min with dim (returns tuple), compute result type from input shape
            # ShapeProp may not set tensor_meta properly for tuple-returning ops
            input_node = node.args[0] if node.args else None
            input_shape = self.shape_map.get(input_node.name, ()) if input_node and hasattr(input_node, 'name') else ()
            input_dtype = self.dtype_map.get(input_node.name, 'f32') if input_node and hasattr(input_node, 'name') else 'f32'

            # Compute reduced shape (remove the dimension being reduced)
            if input_shape:
                reduced_shape = list(input_shape)
                if isinstance(dim, int) and 0 <= dim < len(reduced_shape):
                    del reduced_shape[dim]
                elif isinstance(dim, int) and dim < 0 and -dim <= len(reduced_shape):
                    del reduced_shape[dim]

                # Build result type from reduced shape
                dtype_map = {'float32': 'f32', 'float64': 'f64', 'float16': 'f16',
                             'bfloat16': 'bf16', 'int32': 'i32', 'int64': 'i64',
                             'int8': 'i8', 'int16': 'i16', 'bool': 'i1'}
                mlir_dtype = dtype_map.get(input_dtype, 'f32')
                if reduced_shape:
                    shape_str = 'x'.join(str(d) for d in reduced_shape)
                    computed_result_type = f'tensor<{shape_str}x{mlir_dtype}>'
                else:
                    computed_result_type = f'tensor<{mlir_dtype}>'
            else:
                # Fallback to what we have
                computed_result_type = result_type

            init_ssa = f'{result_ssa}_init'
            # Format: %r = stablehlo.reduce %operand, %init, dims=[...], reducer=min : (...) -> tensor<...>
            # Use large positive number instead of +inf for parser compatibility
            return [
                f'{init_ssa} = stablehlo.constant dense<3.40282e+38> : tensor<f32>',
                f'{result_ssa} = stablehlo.reduce {input_ssa}, {init_ssa}, dims = [{dim}], reducer = min : ({input_type}, tensor<f32>) -> {computed_result_type}'
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

        elif target_name == 'log_softmax':
            input_ssa = get_input(0)
            # log_softmax(x) = x - log(sum(exp(x)))
            exp_ssa = f'{result_ssa}_exp'
            log_ssa = f'{result_ssa}_log'
            return [
                f'{exp_ssa} = stablehlo.exponential {input_ssa} : {result_type}',
                f'// Note: Full log_softmax requires reduction - simplified here',
                f'{log_ssa} = stablehlo.log {exp_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.subtract {input_ssa}, {log_ssa} : {result_type}  // placeholder'
            ]

        # === Additional Reduction Operations ===
        elif target_name == 'mean':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', None)
            keepdim = node.kwargs.get('keepdim', False)
            if dim is None:
                dim = 0
            init_ssa = f'{result_ssa}_init'
            sum_ssa = f'{result_ssa}_sum'
            return [
                f'{init_ssa} = stablehlo.constant dense<0.0> : tensor<f32>',
                f'{sum_ssa} = stablehlo.reduce {input_ssa}, {init_ssa}, dims = [{dim}], reducer = add : ({input_type}, tensor<f32>) -> {result_type}',
                f'// Note: mean requires dividing by count - simplified here',
                f'{result_ssa} = {sum_ssa}  // placeholder for mean'
            ]

        elif target_name == 'prod':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', None)
            if dim is None:
                dim = 0
            init_ssa = f'{result_ssa}_init'
            return [
                f'{init_ssa} = stablehlo.constant dense<1.0> : tensor<f32>',
                f'{result_ssa} = stablehlo.reduce {input_ssa}, {init_ssa}, dims = [{dim}], reducer = multiply : ({input_type}, tensor<f32>) -> {result_type}'
            ]

        elif target_name == 'all':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', None)
            if dim is None:
                dim = 0
            init_ssa = f'{result_ssa}_init'
            return [
                f'{init_ssa} = stablehlo.constant dense<true> : tensor<i1>',
                f'{result_ssa} = stablehlo.reduce {input_ssa}, {init_ssa}, dims = [{dim}], reducer = and : ({input_type}, tensor<i1>) -> {result_type}'
            ]

        elif target_name == 'any':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', None)
            if dim is None:
                dim = 0
            init_ssa = f'{result_ssa}_init'
            return [
                f'{init_ssa} = stablehlo.constant dense<false> : tensor<i1>',
                f'{result_ssa} = stablehlo.reduce {input_ssa}, {init_ssa}, dims = [{dim}], reducer = or : ({input_type}, tensor<i1>) -> {result_type}'
            ]

        elif target_name == 'logsumexp':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', 0)
            # logsumexp(x) = log(sum(exp(x)))
            exp_ssa = f'{result_ssa}_exp'
            sum_init = f'{result_ssa}_init'
            sum_ssa = f'{result_ssa}_sum'
            return [
                f'{exp_ssa} = stablehlo.exponential {input_ssa} : {input_type}',
                f'{sum_init} = stablehlo.constant dense<0.0> : tensor<f32>',
                f'{sum_ssa} = stablehlo.reduce {exp_ssa}, {sum_init}, dims = [{dim}], reducer = add : ({input_type}, tensor<f32>) -> {result_type}',
                f'{result_ssa} = stablehlo.log {sum_ssa} : {result_type}'
            ]

        elif target_name == 'cumsum':
            input_ssa = get_input(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', 0)
            return [f'// cumsum requires scan - not directly supported in StableHLO',
                    f'{result_ssa} = stablehlo.custom_call @cumsum({input_ssa}) {{dim = {dim}}} : ({get_input_type(0)}) -> {result_type}']

        elif target_name == 'cumprod':
            input_ssa = get_input(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', 0)
            return [f'// cumprod requires scan - not directly supported in StableHLO',
                    f'{result_ssa} = stablehlo.custom_call @cumprod({input_ssa}) {{dim = {dim}}} : ({get_input_type(0)}) -> {result_type}']

        # === Activation Functions ===
        elif target_name == 'leaky_relu':
            input_ssa = get_input(0)
            negative_slope = node.args[1] if len(node.args) > 1 else node.kwargs.get('negative_slope', 0.01)
            zero_ssa = f'{result_ssa}_zero'
            scale_ssa = f'{result_ssa}_scale'
            scaled_ssa = f'{result_ssa}_scaled'
            cmp_ssa = f'{result_ssa}_cmp'
            return [
                f'{zero_ssa} = stablehlo.constant dense<0.0> : {result_type}',
                f'{scale_ssa} = stablehlo.constant dense<{negative_slope}> : {result_type}',
                f'{scaled_ssa} = stablehlo.multiply {input_ssa}, {scale_ssa} : {result_type}',
                f'{cmp_ssa} = stablehlo.compare GE, {input_ssa}, {zero_ssa} : ({result_type}, {result_type}) -> tensor<i1>',
                f'{result_ssa} = stablehlo.select {cmp_ssa}, {input_ssa}, {scaled_ssa} : (tensor<i1>, {result_type}, {result_type}) -> {result_type}'
            ]

        elif target_name == 'elu':
            input_ssa = get_input(0)
            alpha = node.args[1] if len(node.args) > 1 else node.kwargs.get('alpha', 1.0)
            zero_ssa = f'{result_ssa}_zero'
            alpha_ssa = f'{result_ssa}_alpha'
            exp_ssa = f'{result_ssa}_exp'
            sub_ssa = f'{result_ssa}_sub'
            scaled_ssa = f'{result_ssa}_scaled'
            cmp_ssa = f'{result_ssa}_cmp'
            return [
                f'{zero_ssa} = stablehlo.constant dense<0.0> : {result_type}',
                f'{alpha_ssa} = stablehlo.constant dense<{alpha}> : {result_type}',
                f'{exp_ssa} = stablehlo.exponential {input_ssa} : {result_type}',
                f'{sub_ssa} = stablehlo.subtract {exp_ssa}, {alpha_ssa} : {result_type}',
                f'{scaled_ssa} = stablehlo.multiply {alpha_ssa}, {sub_ssa} : {result_type}',
                f'{cmp_ssa} = stablehlo.compare GE, {input_ssa}, {zero_ssa} : ({result_type}, {result_type}) -> tensor<i1>',
                f'{result_ssa} = stablehlo.select {cmp_ssa}, {input_ssa}, {scaled_ssa} : (tensor<i1>, {result_type}, {result_type}) -> {result_type}'
            ]

        elif target_name in ('silu', 'swish'):
            # silu(x) = x * sigmoid(x)
            input_ssa = get_input(0)
            sigmoid_ssa = f'{result_ssa}_sigmoid'
            return [
                f'{sigmoid_ssa} = stablehlo.logistic {input_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.multiply {input_ssa}, {sigmoid_ssa} : {result_type}'
            ]

        elif target_name == 'gelu':
            # gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            input_ssa = get_input(0)
            half_ssa = f'{result_ssa}_half'
            coef_ssa = f'{result_ssa}_coef'
            cube_coef_ssa = f'{result_ssa}_cube_coef'
            sq_ssa = f'{result_ssa}_sq'
            cube_ssa = f'{result_ssa}_cube'
            scaled_cube_ssa = f'{result_ssa}_scaled_cube'
            sum_ssa = f'{result_ssa}_sum'
            inner_ssa = f'{result_ssa}_inner'
            tanh_ssa = f'{result_ssa}_tanh'
            one_ssa = f'{result_ssa}_one'
            add_ssa = f'{result_ssa}_add'
            mul1_ssa = f'{result_ssa}_mul1'
            return [
                f'{half_ssa} = stablehlo.constant dense<0.5> : {result_type}',
                f'{coef_ssa} = stablehlo.constant dense<0.7978845608> : {result_type}',  # sqrt(2/π)
                f'{cube_coef_ssa} = stablehlo.constant dense<0.044715> : {result_type}',
                f'{one_ssa} = stablehlo.constant dense<1.0> : {result_type}',
                f'{sq_ssa} = stablehlo.multiply {input_ssa}, {input_ssa} : {result_type}',
                f'{cube_ssa} = stablehlo.multiply {sq_ssa}, {input_ssa} : {result_type}',
                f'{scaled_cube_ssa} = stablehlo.multiply {cube_ssa}, {cube_coef_ssa} : {result_type}',
                f'{sum_ssa} = stablehlo.add {input_ssa}, {scaled_cube_ssa} : {result_type}',
                f'{inner_ssa} = stablehlo.multiply {sum_ssa}, {coef_ssa} : {result_type}',
                f'{tanh_ssa} = stablehlo.tanh {inner_ssa} : {result_type}',
                f'{add_ssa} = stablehlo.add {one_ssa}, {tanh_ssa} : {result_type}',
                f'{mul1_ssa} = stablehlo.multiply {input_ssa}, {add_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.multiply {half_ssa}, {mul1_ssa} : {result_type}'
            ]

        elif target_name == 'mish':
            # mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
            input_ssa = get_input(0)
            exp_ssa = f'{result_ssa}_exp'
            one_ssa = f'{result_ssa}_one'
            add_ssa = f'{result_ssa}_add'
            log_ssa = f'{result_ssa}_log'
            tanh_ssa = f'{result_ssa}_tanh'
            return [
                f'{one_ssa} = stablehlo.constant dense<1.0> : {result_type}',
                f'{exp_ssa} = stablehlo.exponential {input_ssa} : {result_type}',
                f'{add_ssa} = stablehlo.add {one_ssa}, {exp_ssa} : {result_type}',
                f'{log_ssa} = stablehlo.log {add_ssa} : {result_type}',
                f'{tanh_ssa} = stablehlo.tanh {log_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.multiply {input_ssa}, {tanh_ssa} : {result_type}'
            ]

        elif target_name == 'softplus':
            # softplus(x) = ln(1 + exp(x))
            input_ssa = get_input(0)
            exp_ssa = f'{result_ssa}_exp'
            one_ssa = f'{result_ssa}_one'
            add_ssa = f'{result_ssa}_add'
            return [
                f'{one_ssa} = stablehlo.constant dense<1.0> : {result_type}',
                f'{exp_ssa} = stablehlo.exponential {input_ssa} : {result_type}',
                f'{add_ssa} = stablehlo.add {one_ssa}, {exp_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.log {add_ssa} : {result_type}'
            ]

        elif target_name == 'softsign':
            # softsign(x) = x / (1 + |x|)
            input_ssa = get_input(0)
            one_ssa = f'{result_ssa}_one'
            abs_ssa = f'{result_ssa}_abs'
            denom_ssa = f'{result_ssa}_denom'
            return [
                f'{one_ssa} = stablehlo.constant dense<1.0> : {result_type}',
                f'{abs_ssa} = stablehlo.abs {input_ssa} : {result_type}',
                f'{denom_ssa} = stablehlo.add {one_ssa}, {abs_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.divide {input_ssa}, {denom_ssa} : {result_type}'
            ]

        elif target_name == 'hardtanh':
            input_ssa = get_input(0)
            min_val = node.args[1] if len(node.args) > 1 else node.kwargs.get('min_val', -1.0)
            max_val = node.args[2] if len(node.args) > 2 else node.kwargs.get('max_val', 1.0)
            min_ssa = f'{result_ssa}_min'
            max_ssa = f'{result_ssa}_max'
            return [
                f'{min_ssa} = stablehlo.constant dense<{min_val}> : {result_type}',
                f'{max_ssa} = stablehlo.constant dense<{max_val}> : {result_type}',
                f'{result_ssa} = stablehlo.clamp {min_ssa}, {input_ssa}, {max_ssa} : {result_type}'
            ]

        elif target_name == 'relu6':
            input_ssa = get_input(0)
            zero_ssa = f'{result_ssa}_zero'
            six_ssa = f'{result_ssa}_six'
            return [
                f'{zero_ssa} = stablehlo.constant dense<0.0> : {result_type}',
                f'{six_ssa} = stablehlo.constant dense<6.0> : {result_type}',
                f'{result_ssa} = stablehlo.clamp {zero_ssa}, {input_ssa}, {six_ssa} : {result_type}'
            ]

        elif target_name == 'hardsigmoid':
            # hardsigmoid(x) = clamp((x + 3) / 6, 0, 1)
            input_ssa = get_input(0)
            three_ssa = f'{result_ssa}_three'
            six_ssa = f'{result_ssa}_six'
            add_ssa = f'{result_ssa}_add'
            div_ssa = f'{result_ssa}_div'
            zero_ssa = f'{result_ssa}_zero'
            one_ssa = f'{result_ssa}_one'
            return [
                f'{three_ssa} = stablehlo.constant dense<3.0> : {result_type}',
                f'{six_ssa} = stablehlo.constant dense<6.0> : {result_type}',
                f'{zero_ssa} = stablehlo.constant dense<0.0> : {result_type}',
                f'{one_ssa} = stablehlo.constant dense<1.0> : {result_type}',
                f'{add_ssa} = stablehlo.add {input_ssa}, {three_ssa} : {result_type}',
                f'{div_ssa} = stablehlo.divide {add_ssa}, {six_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.clamp {zero_ssa}, {div_ssa}, {one_ssa} : {result_type}'
            ]

        elif target_name == 'hardswish':
            # hardswish(x) = x * hardsigmoid(x) = x * clamp((x + 3) / 6, 0, 1)
            input_ssa = get_input(0)
            three_ssa = f'{result_ssa}_three'
            six_ssa = f'{result_ssa}_six'
            add_ssa = f'{result_ssa}_add'
            div_ssa = f'{result_ssa}_div'
            zero_ssa = f'{result_ssa}_zero'
            one_ssa = f'{result_ssa}_one'
            clamped_ssa = f'{result_ssa}_clamped'
            return [
                f'{three_ssa} = stablehlo.constant dense<3.0> : {result_type}',
                f'{six_ssa} = stablehlo.constant dense<6.0> : {result_type}',
                f'{zero_ssa} = stablehlo.constant dense<0.0> : {result_type}',
                f'{one_ssa} = stablehlo.constant dense<1.0> : {result_type}',
                f'{add_ssa} = stablehlo.add {input_ssa}, {three_ssa} : {result_type}',
                f'{div_ssa} = stablehlo.divide {add_ssa}, {six_ssa} : {result_type}',
                f'{clamped_ssa} = stablehlo.clamp {zero_ssa}, {div_ssa}, {one_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.multiply {input_ssa}, {clamped_ssa} : {result_type}'
            ]

        elif target_name == 'selu':
            # selu(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
            # scale ≈ 1.0507, alpha ≈ 1.6733
            input_ssa = get_input(0)
            scale_ssa = f'{result_ssa}_scale'
            alpha_ssa = f'{result_ssa}_alpha'
            zero_ssa = f'{result_ssa}_zero'
            one_ssa = f'{result_ssa}_one'
            exp_ssa = f'{result_ssa}_exp'
            sub_ssa = f'{result_ssa}_sub'
            scaled_neg_ssa = f'{result_ssa}_scaled_neg'
            cmp_ssa = f'{result_ssa}_cmp'
            selected_ssa = f'{result_ssa}_selected'
            return [
                f'{scale_ssa} = stablehlo.constant dense<1.0507009873554805> : {result_type}',
                f'{alpha_ssa} = stablehlo.constant dense<1.6732632423543772> : {result_type}',
                f'{zero_ssa} = stablehlo.constant dense<0.0> : {result_type}',
                f'{one_ssa} = stablehlo.constant dense<1.0> : {result_type}',
                f'{exp_ssa} = stablehlo.exponential {input_ssa} : {result_type}',
                f'{sub_ssa} = stablehlo.subtract {exp_ssa}, {one_ssa} : {result_type}',
                f'{scaled_neg_ssa} = stablehlo.multiply {alpha_ssa}, {sub_ssa} : {result_type}',
                f'{cmp_ssa} = stablehlo.compare GE, {input_ssa}, {zero_ssa} : ({result_type}, {result_type}) -> tensor<i1>',
                f'{selected_ssa} = stablehlo.select {cmp_ssa}, {input_ssa}, {scaled_neg_ssa} : (tensor<i1>, {result_type}, {result_type}) -> {result_type}',
                f'{result_ssa} = stablehlo.multiply {scale_ssa}, {selected_ssa} : {result_type}'
            ]

        elif target_name == 'prelu':
            # prelu(x, weight) = max(0, x) + weight * min(0, x)
            input_ssa = get_input(0)
            weight_ssa = get_input(1) if len(node.args) > 1 else f'{result_ssa}_weight'
            zero_ssa = f'{result_ssa}_zero'
            cmp_ssa = f'{result_ssa}_cmp'
            neg_ssa = f'{result_ssa}_neg'
            scaled_ssa = f'{result_ssa}_scaled'
            return [
                f'{zero_ssa} = stablehlo.constant dense<0.0> : {result_type}',
                f'{cmp_ssa} = stablehlo.compare GE, {input_ssa}, {zero_ssa} : ({result_type}, {result_type}) -> tensor<i1>',
                f'{scaled_ssa} = stablehlo.multiply {input_ssa}, {weight_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.select {cmp_ssa}, {input_ssa}, {scaled_ssa} : (tensor<i1>, {result_type}, {result_type}) -> {result_type}'
            ]

        # === Type Conversion ===
        elif target_name in ('to', 'type', 'float', 'half', 'double', 'int', 'long', 'bool', 'bfloat16'):
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.convert {input_ssa} : {input_type} -> {result_type}']

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

        elif target_name == 'mm':
            # 2D matrix multiply: (M, K) x (K, N) -> (M, N)
            lhs = get_input(0)
            rhs = get_input(1)
            lhs_type = get_input_type(0)
            rhs_type = get_input_type(1)
            dot_attr = '#stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>'
            return [
                f'{result_ssa} = stablehlo.dot_general {lhs}, {rhs}, '
                f'{dot_attr} : ({lhs_type}, {rhs_type}) -> {result_type}'
            ]

        elif target_name == 'bmm':
            # Batched matrix multiply: (B, M, K) x (B, K, N) -> (B, M, N)
            lhs = get_input(0)
            rhs = get_input(1)
            lhs_type = get_input_type(0)
            rhs_type = get_input_type(1)
            dot_attr = '#stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>'
            return [
                f'{result_ssa} = stablehlo.dot_general {lhs}, {rhs}, '
                f'{dot_attr} : ({lhs_type}, {rhs_type}) -> {result_type}'
            ]

        elif target_name == 'linear':
            # Linear: x @ weight.T + bias
            input_ssa = get_input(0)
            weight_ssa = get_input(1)
            input_type = get_input_type(0)
            weight_type = get_input_type(1)
            has_bias = len(node.args) > 2 or 'bias' in node.kwargs
            dot_attr = '#stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>'
            if has_bias:
                bias_ssa = get_input(2) if len(node.args) > 2 else node.kwargs.get('bias', '%bias')
                mm_ssa = f'{result_ssa}_mm'
                return [
                    f'{mm_ssa} = stablehlo.dot_general {input_ssa}, {weight_ssa}, '
                    f'{dot_attr} : ({input_type}, {weight_type}) -> {result_type}',
                    f'{result_ssa} = stablehlo.add {mm_ssa}, {bias_ssa} : {result_type}'
                ]
            else:
                return [
                    f'{result_ssa} = stablehlo.dot_general {input_ssa}, {weight_ssa}, '
                    f'{dot_attr} : ({input_type}, {weight_type}) -> {result_type}'
                ]

        elif target_name == 'mv':
            # Matrix-vector multiply: (M, N) x (N,) -> (M,)
            lhs = get_input(0)
            rhs = get_input(1)
            lhs_type = get_input_type(0)
            rhs_type = get_input_type(1)
            dot_attr = '#stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>'
            return [
                f'{result_ssa} = stablehlo.dot_general {lhs}, {rhs}, '
                f'{dot_attr} : ({lhs_type}, {rhs_type}) -> {result_type}'
            ]

        elif target_name == 'dot':
            # Dot product: (N,) x (N,) -> scalar
            lhs = get_input(0)
            rhs = get_input(1)
            lhs_type = get_input_type(0)
            rhs_type = get_input_type(1)
            dot_attr = '#stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>'
            return [
                f'{result_ssa} = stablehlo.dot_general {lhs}, {rhs}, '
                f'{dot_attr} : ({lhs_type}, {rhs_type}) -> {result_type}'
            ]

        elif target_name == 'outer':
            # Outer product: (M,) x (N,) -> (M, N)
            lhs = get_input(0)
            rhs = get_input(1)
            lhs_type = get_input_type(0)
            rhs_type = get_input_type(1)
            dot_attr = '#stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [], rhs_contracting_dimensions = []>'
            return [
                f'{result_ssa} = stablehlo.dot_general {lhs}, {rhs}, '
                f'{dot_attr} : ({lhs_type}, {rhs_type}) -> {result_type}'
            ]

        elif target_name == 'addmm':
            # addmm(bias, mat1, mat2) = bias + mat1 @ mat2
            bias_ssa = get_input(0)
            lhs = get_input(1)
            rhs = get_input(2)
            lhs_type = get_input_type(1)
            rhs_type = get_input_type(2)
            mm_ssa = f'{result_ssa}_mm'
            dot_attr = '#stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>'
            return [
                f'{mm_ssa} = stablehlo.dot_general {lhs}, {rhs}, '
                f'{dot_attr} : ({lhs_type}, {rhs_type}) -> {result_type}',
                f'{result_ssa} = stablehlo.add {mm_ssa}, {bias_ssa} : {result_type}'
            ]

        elif target_name == 'baddbmm':
            # baddbmm(input, batch1, batch2) = input + batch1 @ batch2
            input_ssa = get_input(0)
            lhs = get_input(1)
            rhs = get_input(2)
            lhs_type = get_input_type(1)
            rhs_type = get_input_type(2)
            mm_ssa = f'{result_ssa}_bmm'
            dot_attr = '#stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>'
            return [
                f'{mm_ssa} = stablehlo.dot_general {lhs}, {rhs}, '
                f'{dot_attr} : ({lhs_type}, {rhs_type}) -> {result_type}',
                f'{result_ssa} = stablehlo.add {mm_ssa}, {input_ssa} : {result_type}'
            ]

        # === Shape operations ===
        elif target_name == 'reshape':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.reshape {input_ssa} : {input_type} -> {result_type}']

        elif target_name == 'view':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.reshape {input_ssa} : {input_type} -> {result_type}']

        elif target_name == 'flatten':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.reshape {input_ssa} : {input_type} -> {result_type}']

        elif target_name == 'squeeze':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.reshape {input_ssa} : {input_type} -> {result_type}']

        elif target_name == 'unsqueeze':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.reshape {input_ssa} : {input_type} -> {result_type}']

        elif target_name == 'transpose':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim0 = node.args[1] if len(node.args) > 1 else 0
            dim1 = node.args[2] if len(node.args) > 2 else 1
            input_shape = self.shape_map.get(node.args[0].name if hasattr(node.args[0], 'name') else '', ())
            ndim = len(input_shape) if input_shape else 2
            permutation = list(range(ndim))
            if dim0 < ndim and dim1 < ndim:
                permutation[dim0], permutation[dim1] = permutation[dim1], permutation[dim0]
            perm_str = ', '.join(str(p) for p in permutation)
            return [f'{result_ssa} = stablehlo.transpose {input_ssa}, dims = [{perm_str}] : {input_type} -> {result_type}']

        elif target_name == 'permute':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dims = node.args[1] if len(node.args) > 1 else node.kwargs.get('dims', [])
            if isinstance(dims, (list, tuple)):
                perm_str = ', '.join(str(d) for d in dims)
            else:
                perm_str = str(dims)
            return [f'{result_ssa} = stablehlo.transpose {input_ssa}, dims = [{perm_str}] : {input_type} -> {result_type}']

        elif target_name == 'split':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            # split returns tuple - simplified to custom_call
            return [f'{result_ssa} = stablehlo.custom_call @split({input_ssa}) : ({input_type}) -> {result_type}']

        elif target_name == 'chunk':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            chunks = node.args[1] if len(node.args) > 1 else node.kwargs.get('chunks', 2)
            return [f'{result_ssa} = stablehlo.custom_call @chunk({input_ssa}) {{chunks = {chunks}}} : ({input_type}) -> {result_type}']

        elif target_name == 'tile':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @tile({input_ssa}) : ({input_type}) -> {result_type}']

        elif target_name == 'repeat':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @repeat({input_ssa}) : ({input_type}) -> {result_type}']

        elif target_name == 'roll':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            shifts = node.args[1] if len(node.args) > 1 else 0
            dims = node.args[2] if len(node.args) > 2 else 0
            return [f'{result_ssa} = stablehlo.custom_call @roll({input_ssa}) {{shifts = {shifts}, dims = {dims}}} : ({input_type}) -> {result_type}']

        elif target_name == 'flip':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dims = node.args[1] if len(node.args) > 1 else [0]
            return [f'{result_ssa} = stablehlo.reverse {input_ssa}, dims = {list(dims)} : {input_type}']

        # === Indexing/Slicing ===
        elif target_name == 'index_select':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else 0
            indices_ssa = get_input(2) if len(node.args) > 2 else '%indices'
            return [f'{result_ssa} = stablehlo.gather {input_ssa}[{indices_ssa}], dim = {dim} : ({input_type}) -> {result_type}']

        elif target_name == 'gather':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else 0
            indices_ssa = get_input(2) if len(node.args) > 2 else '%indices'
            return [f'{result_ssa} = stablehlo.gather {input_ssa}[{indices_ssa}], dim = {dim} : ({input_type}) -> {result_type}']

        elif target_name == 'scatter':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else 0
            indices_ssa = get_input(2) if len(node.args) > 2 else '%indices'
            src_ssa = get_input(3) if len(node.args) > 3 else '%src'
            return [f'{result_ssa} = stablehlo.scatter {input_ssa}, {indices_ssa}, {src_ssa}, dim = {dim} : ({input_type}) -> {result_type}']

        elif target_name == 'masked_fill':
            input_ssa = get_input(0)
            mask_ssa = get_input(1) if len(node.args) > 1 else '%mask'
            value = node.args[2] if len(node.args) > 2 else 0.0
            value_ssa = f'{result_ssa}_value'
            return [
                f'{value_ssa} = stablehlo.constant dense<{value}> : {result_type}',
                f'{result_ssa} = stablehlo.select {mask_ssa}, {value_ssa}, {input_ssa} : (tensor<i1>, {result_type}, {result_type}) -> {result_type}'
            ]

        elif target_name == 'masked_select':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            mask_ssa = get_input(1) if len(node.args) > 1 else '%mask'
            return [f'{result_ssa} = stablehlo.custom_call @masked_select({input_ssa}, {mask_ssa}) : ({input_type}, tensor<i1>) -> {result_type}']

        # === Padding operations ===
        elif target_name == 'pad':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            pad = node.args[1] if len(node.args) > 1 else []
            value = node.args[2] if len(node.args) > 2 else 0.0
            value_ssa = f'{result_ssa}_value'
            # Convert PyTorch pad format (pairs from last dim) to StableHLO format
            return [
                f'{value_ssa} = stablehlo.constant dense<{value}> : tensor<f32>',
                f'{result_ssa} = stablehlo.pad {input_ssa}, {value_ssa}, low = [], high = [], interior = [] : ({input_type}, tensor<f32>) -> {result_type}'
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


def trace_with_values(source_code: str, class_name: str, input_shapes: list, seed: int = 42) -> dict:
    """
    Trace a model and capture actual tensor values for E2E verification.

    This function runs the model forward pass with deterministic inputs
    and captures both the graph (MLIR) and the actual tensor values.

    Args:
        source_code: Python source containing an nn.Module class
        class_name: Name of the nn.Module class to trace
        input_shapes: List of tuples, e.g., [(1, 8), (1, 16)]
        seed: Random seed for reproducible inputs (default: 42)

    Returns:
        Dictionary with:
            - 'mlir': StableHLO MLIR text
            - 'inputs': List of numpy arrays (serializable)
            - 'outputs': List of numpy arrays (serializable)
            - 'seed': Random seed used
            - 'input_shapes': Input shape tuples
            - 'output_shapes': Output shape tuples
    """
    _ensure_torch_imported()

    # Set deterministic seed for reproducibility
    torch.manual_seed(seed)

    # Parse and instantiate model
    namespace = {'torch': torch}
    exec(source_code, namespace)

    if class_name not in namespace:
        raise ValueError(f"Class '{class_name}' not found in source")

    model_class = namespace[class_name]
    model = model_class()
    model.eval()

    # Create deterministic inputs
    sample_inputs = tuple(torch.randn(*shape) for shape in input_shapes)

    # Run forward pass to get actual outputs
    with torch.no_grad():
        outputs = model(*sample_inputs)

    # Normalize outputs to tuple
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    elif not isinstance(outputs, tuple):
        outputs = tuple(outputs)

    # Trace for MLIR (reusing same inputs for consistency)
    traced = symbolic_trace(model)
    converter = FXToStableHLO(traced, sample_inputs)
    mlir = converter.convert()

    # Convert tensors to numpy arrays for serialization
    input_arrays = [inp.detach().cpu().numpy() for inp in sample_inputs]
    output_arrays = [out.detach().cpu().numpy() for out in outputs]

    return {
        'mlir': mlir,
        'inputs': input_arrays,
        'outputs': output_arrays,
        'seed': seed,
        'input_shapes': [tuple(arr.shape) for arr in input_arrays],
        'output_shapes': [tuple(arr.shape) for arr in output_arrays],
    }


def serialize_npy(arr) -> bytes:
    """Serialize a numpy array to .npy format bytes."""
    import io
    import numpy as np
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    return buffer.getvalue()


def trace_with_values_npy(source_code: str, class_name: str, input_shapes: list, seed: int = 42) -> dict:
    """
    Like trace_with_values but returns tensors as .npy bytes for direct file writing.

    Returns:
        Dictionary with:
            - 'mlir': StableHLO MLIR text
            - 'input_npy': List of bytes (.npy format)
            - 'output_npy': List of bytes (.npy format)
            - 'seed': Random seed used
            - 'input_shapes': Input shape tuples
            - 'output_shapes': Output shape tuples
    """
    result = trace_with_values(source_code, class_name, input_shapes, seed)

    # Convert numpy arrays to .npy bytes
    input_npy = [serialize_npy(arr) for arr in result['inputs']]
    output_npy = [serialize_npy(arr) for arr in result['outputs']]

    return {
        'mlir': result['mlir'],
        'input_npy': input_npy,
        'output_npy': output_npy,
        'seed': result['seed'],
        'input_shapes': result['input_shapes'],
        'output_shapes': result['output_shapes'],
    }


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
