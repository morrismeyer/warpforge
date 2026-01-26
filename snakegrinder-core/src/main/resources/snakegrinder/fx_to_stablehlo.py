"""
Convert PyTorch FX graph to StableHLO MLIR.

Uses torch.fx.symbolic_trace to capture the computation graph from real PyTorch
models, then converts to StableHLO MLIR text format.
"""
from __future__ import annotations  # PEP 563: Postponed evaluation of annotations

from typing import Dict, List, Tuple, Any, Optional, Set, TYPE_CHECKING

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
    """Convert FX graph to StableHLO MLIR text.

    Args:
        traced_module: The torch.fx traced module
        sample_inputs: Sample input tensors for shape inference
        dynamic_dims: Optional dict mapping input index to set of dynamic dimension indices.
                      Example: {0: {0}} means input 0 has dynamic batch dimension (dim 0).
                      Example: {0: {0, 1}} means input 0 has dynamic dims 0 and 1.
    """

    def __init__(self, traced_module, sample_inputs: Tuple[torch.Tensor, ...],
                 dynamic_dims: Optional[Dict[int, Set[int]]] = None,
                 capture_weights: bool = False):
        self.traced = traced_module
        self.sample_inputs = sample_inputs
        self.graph = traced_module.graph
        self.ssa_map: Dict[str, str] = {}
        self.ssa_counter = 0
        self.shape_map: Dict[str, Tuple] = {}
        self.dtype_map: Dict[str, str] = {}
        # Track which (node_name, dim_index) pairs are dynamic
        self.dynamic_dim_map: Dict[str, Set[int]] = {}
        self.input_dynamic_dims = dynamic_dims or {}
        # Weight capture support
        self.capture_weights = capture_weights
        self.weight_args: List[Dict[str, Any]] = []  # List of {name, tensor, ssa, type}
        self.weight_ssa_map: Dict[str, str] = {}  # module_target -> ssa for weight
        self.bias_ssa_map: Dict[str, str] = {}    # module_target -> ssa for bias
        self._infer_shapes()
        if capture_weights:
            self._collect_weights()

    def _collect_weights(self):
        """Collect weights from modules that need them (Linear, Conv2d, etc.).

        This scans the graph for call_module nodes and extracts their weight/bias tensors,
        adding them as function arguments instead of emitting dense<0.0> placeholders.
        """
        # Count placeholders to know where weight args start
        num_inputs = sum(1 for node in self.graph.nodes if node.op == 'placeholder')
        arg_idx = num_inputs

        for node in self.graph.nodes:
            if node.op != 'call_module':
                continue

            module = self.traced.get_submodule(node.target)

            # Linear layer weights
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.detach()
                weight_ssa = f'%arg{arg_idx}'
                weight_type = f'tensor<{weight.shape[0]}x{weight.shape[1]}xf32>'
                self.weight_args.append({
                    'name': f'{node.target}.weight',
                    'tensor': weight,
                    'ssa': weight_ssa,
                    'type': weight_type,
                })
                self.weight_ssa_map[node.target] = weight_ssa
                arg_idx += 1

                if module.bias is not None:
                    bias = module.bias.detach()
                    bias_ssa = f'%arg{arg_idx}'
                    bias_type = f'tensor<{bias.shape[0]}xf32>'
                    self.weight_args.append({
                        'name': f'{node.target}.bias',
                        'tensor': bias,
                        'ssa': bias_ssa,
                        'type': bias_type,
                    })
                    self.bias_ssa_map[node.target] = bias_ssa
                    arg_idx += 1

            # Conv2d layer weights
            elif isinstance(module, torch.nn.Conv2d):
                weight = module.weight.detach()
                out_c, in_c, kh, kw = weight.shape
                weight_ssa = f'%arg{arg_idx}'
                weight_type = f'tensor<{out_c}x{in_c}x{kh}x{kw}xf32>'
                self.weight_args.append({
                    'name': f'{node.target}.weight',
                    'tensor': weight,
                    'ssa': weight_ssa,
                    'type': weight_type,
                })
                self.weight_ssa_map[node.target] = weight_ssa
                arg_idx += 1

                if module.bias is not None:
                    bias = module.bias.detach()
                    bias_ssa = f'%arg{arg_idx}'
                    bias_type = f'tensor<{bias.shape[0]}xf32>'
                    self.weight_args.append({
                        'name': f'{node.target}.bias',
                        'tensor': bias,
                        'ssa': bias_ssa,
                        'type': bias_type,
                    })
                    self.bias_ssa_map[node.target] = bias_ssa
                    arg_idx += 1

            # LayerNorm weights (gamma/beta)
            elif isinstance(module, torch.nn.LayerNorm):
                if module.weight is not None:
                    weight = module.weight.detach()
                    weight_ssa = f'%arg{arg_idx}'
                    # LayerNorm normalized_shape can be multi-dimensional
                    shape_str = 'x'.join(str(d) for d in weight.shape)
                    weight_type = f'tensor<{shape_str}xf32>'
                    self.weight_args.append({
                        'name': f'{node.target}.weight',
                        'tensor': weight,
                        'ssa': weight_ssa,
                        'type': weight_type,
                    })
                    self.weight_ssa_map[node.target] = weight_ssa
                    arg_idx += 1

                if module.bias is not None:
                    bias = module.bias.detach()
                    bias_ssa = f'%arg{arg_idx}'
                    shape_str = 'x'.join(str(d) for d in bias.shape)
                    bias_type = f'tensor<{shape_str}xf32>'
                    self.weight_args.append({
                        'name': f'{node.target}.bias',
                        'tensor': bias,
                        'ssa': bias_ssa,
                        'type': bias_type,
                    })
                    self.bias_ssa_map[node.target] = bias_ssa
                    arg_idx += 1

            # Embedding layer weights
            elif isinstance(module, torch.nn.Embedding):
                weight = module.weight.detach()
                weight_ssa = f'%arg{arg_idx}'
                weight_type = f'tensor<{weight.shape[0]}x{weight.shape[1]}xf32>'
                self.weight_args.append({
                    'name': f'{node.target}.weight',
                    'tensor': weight,
                    'ssa': weight_ssa,
                    'type': weight_type,
                })
                self.weight_ssa_map[node.target] = weight_ssa
                arg_idx += 1

    def _infer_shapes(self):
        """Infer shapes by running the model with ShapeProp and track dynamic dimensions."""
        from torch.fx.passes.shape_prop import ShapeProp
        ShapeProp(self.traced).propagate(*self.sample_inputs)

        # First pass: collect shapes and initialize dynamic dims for inputs
        input_idx = 0
        for node in self.graph.nodes:
            if node.op == 'placeholder':
                # Mark input dynamic dimensions
                if input_idx in self.input_dynamic_dims:
                    self.dynamic_dim_map[node.name] = self.input_dynamic_dims[input_idx]
                input_idx += 1

            if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
                meta = node.meta['tensor_meta']
                if hasattr(meta, 'shape'):
                    self.shape_map[node.name] = tuple(meta.shape)
                    self.dtype_map[node.name] = str(meta.dtype).replace('torch.', '')

        # Second pass: propagate dynamic dimensions through the graph
        self._propagate_dynamic_dims()

    def _propagate_dynamic_dims(self):
        """Propagate dynamic dimensions through the graph.

        Rules for dimension propagation:
        - Elementwise ops: inherit dynamic dims from inputs (must match)
        - Reduction ops: remove the reduced dimension from dynamic set
        - Reshape/view: dynamic dims are lost (need explicit tracking)
        - Transpose/permute: remap dimension indices
        - Broadcast: new dimensions are static, existing dims keep dynamism
        - Matmul/dot: contracting dims are removed, batch dims preserved
        """
        for node in self.graph.nodes:
            if node.op in ('placeholder', 'output'):
                continue

            # Get input dynamic dims
            input_dynamic_dims: List[Set[int]] = []
            for arg in node.args:
                if hasattr(arg, 'name') and arg.name in self.dynamic_dim_map:
                    input_dynamic_dims.append(self.dynamic_dim_map[arg.name])
                elif hasattr(arg, 'name'):
                    input_dynamic_dims.append(set())

            if not input_dynamic_dims:
                continue

            target_name = self._get_target_name(node)
            output_dynamic = set()

            # Elementwise operations: inherit dynamic dims from first input
            if target_name in ('add', 'sub', 'mul', 'div', 'pow', 'maximum', 'minimum',
                               'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'where',
                               'neg', 'abs', 'exp', 'log', 'sqrt', 'sin', 'cos', 'tanh',
                               'sigmoid', 'relu', 'gelu', 'silu'):
                if input_dynamic_dims:
                    output_dynamic = input_dynamic_dims[0].copy()

            # Reduction ops: remove reduced dimension
            elif target_name in ('sum', 'mean', 'max', 'min', 'prod'):
                if input_dynamic_dims:
                    output_dynamic = input_dynamic_dims[0].copy()
                    # Get reduction dim from kwargs or args
                    dim = node.kwargs.get('dim', None)
                    if dim is None and len(node.args) > 1:
                        dim = node.args[1]
                    if isinstance(dim, int) and dim in output_dynamic:
                        output_dynamic.remove(dim)
                    # Adjust indices for dims after the reduced one
                    if isinstance(dim, int):
                        output_dynamic = {d - 1 if d > dim else d for d in output_dynamic}

            # Transpose/permute: remap dimension indices
            elif target_name in ('transpose', 'permute', 't'):
                if input_dynamic_dims:
                    input_shape = self.shape_map.get(node.args[0].name if hasattr(node.args[0], 'name') else '', ())
                    if target_name == 't' and len(input_shape) == 2:
                        # Simple 2D transpose: swap dims 0 and 1
                        output_dynamic = {1 if d == 0 else 0 if d == 1 else d
                                          for d in input_dynamic_dims[0]}
                    elif target_name == 'transpose' and len(node.args) >= 3:
                        dim0, dim1 = node.args[1], node.args[2]
                        output_dynamic = set()
                        for d in input_dynamic_dims[0]:
                            if d == dim0:
                                output_dynamic.add(dim1)
                            elif d == dim1:
                                output_dynamic.add(dim0)
                            else:
                                output_dynamic.add(d)
                    elif target_name == 'permute' and len(node.args) >= 2:
                        perm = node.args[1]
                        if isinstance(perm, (list, tuple)):
                            output_dynamic = set()
                            for new_idx, old_idx in enumerate(perm):
                                if old_idx in input_dynamic_dims[0]:
                                    output_dynamic.add(new_idx)

            # Reshape/view: preserve dynamic dims that map 1:1
            elif target_name in ('reshape', 'view', 'flatten'):
                # Conservative: if any input dim is dynamic, mark corresponding output dims
                if input_dynamic_dims and 0 in input_dynamic_dims[0]:
                    # Batch dimension typically stays at position 0
                    output_dynamic.add(0)

            # Matmul: batch dims preserved, contracting dims removed
            elif target_name in ('matmul', 'mm', 'bmm', 'linear'):
                if input_dynamic_dims:
                    # Batch dimensions (all except last 2) are preserved
                    output_shape = self.shape_map.get(node.name, ())
                    for d in input_dynamic_dims[0]:
                        if d < len(output_shape):
                            output_dynamic.add(d)

            # Default: inherit from first input
            elif input_dynamic_dims:
                output_dynamic = input_dynamic_dims[0].copy()

            if output_dynamic:
                self.dynamic_dim_map[node.name] = output_dynamic

    def _get_target_name(self, node) -> str:
        """Extract target name from node for operation identification."""
        if node.op == 'call_function':
            target = node.target
            if hasattr(target, '__name__'):
                return target.__name__
            return str(target).split('.')[-1]
        elif node.op == 'call_method':
            return str(node.target)
        elif node.op == 'call_module':
            return node.target
        return ''

    def _new_ssa(self) -> str:
        self.ssa_counter += 1
        return f"%{self.ssa_counter}"

    def _tensor_type(self, node_name: str) -> str:
        """Get MLIR tensor type string, using '?' for dynamic dimensions."""
        shape = self.shape_map.get(node_name, ())
        dtype = self.dtype_map.get(node_name, 'f32')
        dynamic_dims = self.dynamic_dim_map.get(node_name, set())

        dtype_map = {
            'float32': 'f32', 'float64': 'f64', 'float16': 'f16',
            'bfloat16': 'bf16', 'int32': 'i32', 'int64': 'i64',
            'int8': 'i8', 'int16': 'i16', 'bool': 'i1'
        }
        mlir_dtype = dtype_map.get(dtype, 'f32')

        if not shape:
            return f'tensor<{mlir_dtype}>'

        # Build shape string, using '?' for dynamic dimensions
        shape_parts = []
        for i, d in enumerate(shape):
            if i in dynamic_dims:
                shape_parts.append('?')
            else:
                shape_parts.append(str(d))
        shape_str = 'x'.join(shape_parts)
        return f'tensor<{shape_str}x{mlir_dtype}>'

    def _is_dynamic_shape(self, node_name: str) -> bool:
        """Check if a node has any dynamic dimensions."""
        return bool(self.dynamic_dim_map.get(node_name, set()))

    def _get_shape_tensor(self, node_name: str, result_ssa: str) -> List[str]:
        """Generate stablehlo.get_dimension_size ops for dynamic dimensions."""
        lines = []
        shape = self.shape_map.get(node_name, ())
        dynamic_dims = self.dynamic_dim_map.get(node_name, set())
        input_ssa = self.ssa_map.get(node_name, '%unknown')

        dim_ssas = []
        for i, d in enumerate(shape):
            if i in dynamic_dims:
                dim_ssa = f'{result_ssa}_dim{i}'
                lines.append(f'{dim_ssa} = stablehlo.get_dimension_size {input_ssa}, dim = {i} : '
                             f'({self._tensor_type(node_name)}) -> tensor<i32>')
                dim_ssas.append(dim_ssa)
            else:
                dim_ssa = f'{result_ssa}_dim{i}'
                lines.append(f'{dim_ssa} = stablehlo.constant dense<{d}> : tensor<i32>')
                dim_ssas.append(dim_ssa)

        # Concatenate into shape tensor
        if dim_ssas:
            concat_ssa = f'{result_ssa}_shape'
            lines.append(f'{concat_ssa} = stablehlo.concatenate {", ".join(dim_ssas)}, '
                         f'dim = 0 : ({", ".join(["tensor<i32>"] * len(dim_ssas))}) -> tensor<{len(dim_ssas)}xi32>')

        return lines

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

        # Build function signature - regular inputs first
        arg_strs = []
        for i, inp in enumerate(inputs):
            ssa_name = f'%arg{i}'
            self.ssa_map[inp.name] = ssa_name
            arg_strs.append(f'{ssa_name}: {self._tensor_type(inp.name)}')

        # Add weight arguments if capture_weights is enabled
        if self.capture_weights and self.weight_args:
            for weight_info in self.weight_args:
                arg_strs.append(f'{weight_info["ssa"]}: {weight_info["type"]}')

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
            input_shape = self.shape_map.get(node.args[0].name, ())
            weight_shape = module.weight.shape
            weight_type = f'tensor<{weight_shape[0]}x{weight_shape[1]}xf32>'

            lines = [f'// Linear layer: {node.target}']

            # Use weight argument if capture_weights is enabled, otherwise emit placeholder
            if self.capture_weights and node.target in self.weight_ssa_map:
                weight_ssa = self.weight_ssa_map[node.target]
                lines.append(f'// Using weight from function argument {weight_ssa}')
            else:
                weight_ssa = f'{result_ssa}_weight'
                lines.append(f'{weight_ssa} = stablehlo.constant dense<0.0> : {weight_type}  // placeholder for weight')

            matmul_ssa = f'{result_ssa}_matmul'
            # Linear layer: contract the last dimension of input with last dimension of weight
            # For 2D input [batch, features], lhs_contract = 1
            # For 3D input [batch, seq, features], lhs_contract = 2
            input_rank = len(input_shape)
            lhs_contract_dim = input_rank - 1
            dot_attr = f'#stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [{lhs_contract_dim}], rhs_contracting_dimensions = [1]>'
            lines.append(
                f'{matmul_ssa} = stablehlo.dot_general {input_ssa}, {weight_ssa}, '
                f'{dot_attr} : ({input_type}, {weight_type}) -> {result_type}'
            )

            if module.bias is not None:
                bias_type = f'tensor<{weight_shape[0]}xf32>'
                # Use bias argument if capture_weights is enabled
                if self.capture_weights and node.target in self.bias_ssa_map:
                    bias_ssa = self.bias_ssa_map[node.target]
                    lines.append(f'// Using bias from function argument {bias_ssa}')
                else:
                    bias_ssa = f'{result_ssa}_bias'
                    lines.append(f'{bias_ssa} = stablehlo.constant dense<0.0> : {bias_type}  // placeholder for bias')
                # Broadcast bias from (out_features,) to output shape (batch, out_features)
                # The bias dimension maps to the last dimension of the output
                output_shape = self.shape_map.get(node.name, ())
                output_rank = len(output_shape)
                bias_broadcast_ssa = f'{result_ssa}_bias_broadcast'
                lines.append(
                    f'{bias_broadcast_ssa} = stablehlo.broadcast_in_dim {bias_ssa}, dims = [{output_rank - 1}] : '
                    f'({bias_type}) -> {result_type}'
                )
                lines.append(f'{result_ssa} = stablehlo.add {matmul_ssa}, {bias_broadcast_ssa} : {result_type}')
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

            kernel_type = f'tensor<{out_channels}x{in_channels}x{kh}x{kw}xf32>'

            lines = [f'// Conv2d layer: {node.target}']

            # Use kernel argument if capture_weights is enabled
            if self.capture_weights and node.target in self.weight_ssa_map:
                kernel_ssa = self.weight_ssa_map[node.target]
                lines.append(f'// Using kernel from function argument {kernel_ssa}')
            else:
                kernel_ssa = f'{result_ssa}_kernel'
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
                bias_type = f'tensor<{out_channels}xf32>'
                # Use bias argument if capture_weights is enabled
                if self.capture_weights and node.target in self.bias_ssa_map:
                    bias_ssa = self.bias_ssa_map[node.target]
                    lines.append(f'// Using bias from function argument {bias_ssa}')
                else:
                    bias_ssa = f'{result_ssa}_bias'
                    lines.append(f'{bias_ssa} = stablehlo.constant dense<0.0> : {bias_type}  // placeholder for bias')
                # Broadcast bias from (out_channels,) to output shape (N, out_channels, H, W)
                # The bias dimension maps to dimension 1 (channel dimension in NCHW)
                bias_broadcast_ssa = f'{result_ssa}_bias_broadcast'
                lines.append(
                    f'{bias_broadcast_ssa} = stablehlo.broadcast_in_dim {bias_ssa}, dims = [1] : '
                    f'({bias_type}) -> {result_type}'
                )
                lines.append(f'{result_ssa} = stablehlo.add {conv_ssa}, {bias_broadcast_ssa} : {result_type}')

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

        elif isinstance(module, torch.nn.LayerNorm):
            # LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
            input_ssa = self.ssa_map.get(node.args[0].name, '%unknown')
            input_type = self._tensor_type(node.args[0].name)

            normalized_shape = list(module.normalized_shape)
            eps = module.eps
            elementwise_affine = module.elementwise_affine

            lines = [f'// LayerNorm layer: {node.target} (normalized_shape={normalized_shape}, eps={eps})']

            # Determine reduction axes (last N dimensions where N = len(normalized_shape))
            input_shape = self.shape_map.get(node.args[0].name, ())
            if input_shape:
                rank = len(input_shape)
                reduce_axes = list(range(rank - len(normalized_shape), rank))
            else:
                # Fallback: assume last dimension
                reduce_axes = [-1]

            # Use custom_call for LayerNorm since StableHLO doesn't have native support
            # This will be handled by SnakeBurger/WarpForge backend
            if elementwise_affine:
                weight_shape_str = 'x'.join(str(d) for d in normalized_shape)
                weight_type = f'tensor<{weight_shape_str}xf32>'

                # Use weight/bias arguments if capture_weights is enabled
                if self.capture_weights and node.target in self.weight_ssa_map:
                    weight_ssa = self.weight_ssa_map[node.target]
                    lines.append(f'// Using gamma from function argument {weight_ssa}')
                else:
                    weight_ssa = f'{result_ssa}_weight'
                    lines.append(f'{weight_ssa} = stablehlo.constant dense<1.0> : {weight_type}  // placeholder for gamma')

                if self.capture_weights and node.target in self.bias_ssa_map:
                    bias_ssa = self.bias_ssa_map[node.target]
                    lines.append(f'// Using beta from function argument {bias_ssa}')
                else:
                    bias_ssa = f'{result_ssa}_bias'
                    lines.append(f'{bias_ssa} = stablehlo.constant dense<0.0> : {weight_type}  // placeholder for beta')

                lines.append(
                    f'{result_ssa} = stablehlo.custom_call @layer_norm({input_ssa}, {weight_ssa}, {bias_ssa}) : '
                    f'({input_type}, {weight_type}, {weight_type}) -> {result_type}'
                )
            else:
                lines.append(
                    f'{result_ssa} = stablehlo.custom_call @layer_norm({input_ssa}) : '
                    f'({input_type}) -> {result_type}'
                )

            return lines

        elif isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
            # BatchNorm: use custom_call since it requires running stats
            input_ssa = self.ssa_map.get(node.args[0].name, '%unknown')
            input_type = self._tensor_type(node.args[0].name)

            num_features = module.num_features
            eps = module.eps
            momentum = module.momentum

            lines = [f'// BatchNorm layer: {node.target} (num_features={num_features}, eps={eps})']

            weight_ssa = f'{result_ssa}_weight'
            bias_ssa = f'{result_ssa}_bias'
            mean_ssa = f'{result_ssa}_mean'
            var_ssa = f'{result_ssa}_var'
            param_type = f'tensor<{num_features}xf32>'

            lines.append(f'{weight_ssa} = stablehlo.constant dense<1.0> : {param_type}  // placeholder for gamma')
            lines.append(f'{bias_ssa} = stablehlo.constant dense<0.0> : {param_type}  // placeholder for beta')
            lines.append(f'{mean_ssa} = stablehlo.constant dense<0.0> : {param_type}  // placeholder for running_mean')
            lines.append(f'{var_ssa} = stablehlo.constant dense<1.0> : {param_type}  // placeholder for running_var')
            lines.append(
                f'{result_ssa} = stablehlo.custom_call @batch_norm({input_ssa}, {weight_ssa}, {bias_ssa}, {mean_ssa}, {var_ssa}) : '
                f'({input_type}, {param_type}, {param_type}, {param_type}, {param_type}) -> {result_type}'
            )

            return lines

        elif isinstance(module, torch.nn.GELU):
            # GELU activation
            input_ssa = self.ssa_map.get(node.args[0].name, '%unknown')
            approximate = getattr(module, 'approximate', 'none')

            lines = [f'// GELU layer: {node.target} (approximate={approximate})']
            lines.append(
                f'{result_ssa} = stablehlo.custom_call @gelu({input_ssa}) : ({self._tensor_type(node.args[0].name)}) -> {result_type}'
            )
            return lines

        elif isinstance(module, torch.nn.SiLU):
            # SiLU / Swish activation: x * sigmoid(x)
            input_ssa = self.ssa_map.get(node.args[0].name, '%unknown')
            input_type = self._tensor_type(node.args[0].name)

            sigmoid_ssa = f'{result_ssa}_sigmoid'
            lines = [f'// SiLU layer: {node.target}']
            lines.append(f'{sigmoid_ssa} = stablehlo.logistic {input_ssa} : {input_type}')
            lines.append(f'{result_ssa} = stablehlo.multiply {input_ssa}, {sigmoid_ssa} : {result_type}')
            return lines

        elif isinstance(module, torch.nn.Softmax):
            # Softmax along specified dimension
            input_ssa = self.ssa_map.get(node.args[0].name, '%unknown')
            dim = module.dim if module.dim is not None else -1

            lines = [f'// Softmax layer: {node.target} (dim={dim})']
            lines.append(
                f'{result_ssa} = stablehlo.custom_call @softmax({input_ssa}) : ({self._tensor_type(node.args[0].name)}) -> {result_type}'
            )
            return lines

        elif isinstance(module, torch.nn.Embedding):
            # Embedding lookup
            input_ssa = self.ssa_map.get(node.args[0].name, '%unknown')
            input_type = self._tensor_type(node.args[0].name)

            num_embeddings = module.num_embeddings
            embedding_dim = module.embedding_dim
            weight_type = f'tensor<{num_embeddings}x{embedding_dim}xf32>'

            lines = [f'// Embedding layer: {node.target} (num_embeddings={num_embeddings}, embedding_dim={embedding_dim})']

            # Use weight argument if capture_weights is enabled, otherwise emit placeholder
            if self.capture_weights and node.target in self.weight_ssa_map:
                weight_ssa = self.weight_ssa_map[node.target]
                lines.append(f'// Using embedding weights from function argument {weight_ssa}')
            else:
                weight_ssa = f'{result_ssa}_weight'
                lines.append(f'{weight_ssa} = stablehlo.constant dense<0.0> : {weight_type}  // placeholder for embedding weights')

            lines.append(
                f'{result_ssa} = stablehlo.custom_call @embedding({input_ssa}, {weight_ssa}) '
                f'{{num_embeddings = {num_embeddings}, embedding_dim = {embedding_dim}}} : '
                f'({input_type}, {weight_type}) -> {result_type}'
            )
            return lines

        else:
            return [f'// Unsupported module: {type(module).__name__}']

    def _convert_call_function(self, node, result_ssa: str, result_type: str) -> List[str]:
        """Convert call_function (e.g., torch.relu, torch.add, etc.)."""
        target = node.target
        target_name = getattr(target, '__name__', str(target))

        # Helper to get input SSA values
        # Returns (ssa_name, constant_lines) where constant_lines is a list of
        # constant definitions needed if the arg is a scalar
        def get_input_with_const(idx):
            if idx < len(node.args):
                arg = node.args[idx]
                if hasattr(arg, 'name'):
                    return self.ssa_map.get(arg.name, '%unknown'), []
                elif isinstance(arg, (int, float)):
                    # Scalar constant - create a broadcast constant
                    const_ssa = f'{result_ssa}_const{idx}'
                    const_val = float(arg)
                    const_line = f'{const_ssa} = stablehlo.constant dense<{const_val}> : {result_type}'
                    return const_ssa, [const_line]
            return '%unknown', []

        def get_input(idx):
            ssa, _ = get_input_with_const(idx)
            return ssa

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
        # These ops support scalar constants as either operand
        elif target_name == 'add':
            lhs, lhs_const = get_input_with_const(0)
            rhs, rhs_const = get_input_with_const(1)
            return lhs_const + rhs_const + [f'{result_ssa} = stablehlo.add {lhs}, {rhs} : {result_type}']

        elif target_name in ('sub', 'subtract'):
            lhs, lhs_const = get_input_with_const(0)
            rhs, rhs_const = get_input_with_const(1)
            return lhs_const + rhs_const + [f'{result_ssa} = stablehlo.subtract {lhs}, {rhs} : {result_type}']

        elif target_name in ('mul', 'multiply'):
            lhs, lhs_const = get_input_with_const(0)
            rhs, rhs_const = get_input_with_const(1)
            return lhs_const + rhs_const + [f'{result_ssa} = stablehlo.multiply {lhs}, {rhs} : {result_type}']

        elif target_name in ('div', 'divide', 'truediv', 'true_divide'):
            lhs, lhs_const = get_input_with_const(0)
            rhs, rhs_const = get_input_with_const(1)
            return lhs_const + rhs_const + [f'{result_ssa} = stablehlo.divide {lhs}, {rhs} : {result_type}']

        elif target_name == 'maximum':
            lhs, lhs_const = get_input_with_const(0)
            rhs, rhs_const = get_input_with_const(1)
            return lhs_const + rhs_const + [f'{result_ssa} = stablehlo.maximum {lhs}, {rhs} : {result_type}']

        elif target_name == 'minimum':
            lhs, lhs_const = get_input_with_const(0)
            rhs, rhs_const = get_input_with_const(1)
            return lhs_const + rhs_const + [f'{result_ssa} = stablehlo.minimum {lhs}, {rhs} : {result_type}']

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

        # === Inverse trigonometric functions ===
        elif target_name == 'asin':
            return [f'{result_ssa} = stablehlo.asin {get_input(0)} : {result_type}']

        elif target_name == 'acos':
            return [f'{result_ssa} = stablehlo.acos {get_input(0)} : {result_type}']

        elif target_name == 'atan':
            return [f'{result_ssa} = stablehlo.atan {get_input(0)} : {result_type}']

        # === Hyperbolic inverse functions ===
        elif target_name == 'asinh':
            # asinh(x) = log(x + sqrt(x^2 + 1))
            input_ssa = get_input(0)
            sq_ssa = f'{result_ssa}_sq'
            one_ssa = f'{result_ssa}_one'
            add1_ssa = f'{result_ssa}_add1'
            sqrt_ssa = f'{result_ssa}_sqrt'
            add2_ssa = f'{result_ssa}_add2'
            return [
                f'{sq_ssa} = stablehlo.multiply {input_ssa}, {input_ssa} : {result_type}',
                f'{one_ssa} = stablehlo.constant dense<1.0> : {result_type}',
                f'{add1_ssa} = stablehlo.add {sq_ssa}, {one_ssa} : {result_type}',
                f'{sqrt_ssa} = stablehlo.sqrt {add1_ssa} : {result_type}',
                f'{add2_ssa} = stablehlo.add {input_ssa}, {sqrt_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.log {add2_ssa} : {result_type}'
            ]

        elif target_name == 'acosh':
            # acosh(x) = log(x + sqrt(x^2 - 1))
            input_ssa = get_input(0)
            sq_ssa = f'{result_ssa}_sq'
            one_ssa = f'{result_ssa}_one'
            sub_ssa = f'{result_ssa}_sub'
            sqrt_ssa = f'{result_ssa}_sqrt'
            add_ssa = f'{result_ssa}_add'
            return [
                f'{sq_ssa} = stablehlo.multiply {input_ssa}, {input_ssa} : {result_type}',
                f'{one_ssa} = stablehlo.constant dense<1.0> : {result_type}',
                f'{sub_ssa} = stablehlo.subtract {sq_ssa}, {one_ssa} : {result_type}',
                f'{sqrt_ssa} = stablehlo.sqrt {sub_ssa} : {result_type}',
                f'{add_ssa} = stablehlo.add {input_ssa}, {sqrt_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.log {add_ssa} : {result_type}'
            ]

        elif target_name == 'atanh':
            # atanh(x) = 0.5 * log((1 + x) / (1 - x))
            input_ssa = get_input(0)
            one_ssa = f'{result_ssa}_one'
            half_ssa = f'{result_ssa}_half'
            add_ssa = f'{result_ssa}_add'
            sub_ssa = f'{result_ssa}_sub'
            div_ssa = f'{result_ssa}_div'
            log_ssa = f'{result_ssa}_log'
            return [
                f'{one_ssa} = stablehlo.constant dense<1.0> : {result_type}',
                f'{half_ssa} = stablehlo.constant dense<0.5> : {result_type}',
                f'{add_ssa} = stablehlo.add {one_ssa}, {input_ssa} : {result_type}',
                f'{sub_ssa} = stablehlo.subtract {one_ssa}, {input_ssa} : {result_type}',
                f'{div_ssa} = stablehlo.divide {add_ssa}, {sub_ssa} : {result_type}',
                f'{log_ssa} = stablehlo.log {div_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.multiply {half_ssa}, {log_ssa} : {result_type}'
            ]

        # === Special functions ===
        elif target_name == 'erf':
            return [f'{result_ssa} = stablehlo.erf {get_input(0)} : {result_type}']

        elif target_name == 'erfc':
            # erfc(x) = 1 - erf(x)
            input_ssa = get_input(0)
            erf_ssa = f'{result_ssa}_erf'
            one_ssa = f'{result_ssa}_one'
            return [
                f'{erf_ssa} = stablehlo.erf {input_ssa} : {result_type}',
                f'{one_ssa} = stablehlo.constant dense<1.0> : {result_type}',
                f'{result_ssa} = stablehlo.subtract {one_ssa}, {erf_ssa} : {result_type}'
            ]

        elif target_name == 'exp2':
            # exp2(x) = 2^x = exp(x * ln(2))
            input_ssa = get_input(0)
            ln2_ssa = f'{result_ssa}_ln2'
            mul_ssa = f'{result_ssa}_mul'
            return [
                f'{ln2_ssa} = stablehlo.constant dense<0.6931471805599453> : {result_type}',
                f'{mul_ssa} = stablehlo.multiply {input_ssa}, {ln2_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.exponential {mul_ssa} : {result_type}'
            ]

        elif target_name == 'lgamma':
            # Log-gamma function - use custom_call
            return [f'{result_ssa} = stablehlo.custom_call @lgamma({get_input(0)}) : ({get_input_type(0)}) -> {result_type}']

        elif target_name == 'digamma':
            # Digamma function (psi) - use custom_call
            return [f'{result_ssa} = stablehlo.custom_call @digamma({get_input(0)}) : ({get_input_type(0)}) -> {result_type}']

        # === Additional binary operations ===
        elif target_name == 'fmod':
            # fmod is the same as remainder for floats
            return [f'{result_ssa} = stablehlo.remainder {get_input(0)}, {get_input(1)} : {result_type}']

        elif target_name == 'true_divide':
            return [f'{result_ssa} = stablehlo.divide {get_input(0)}, {get_input(1)} : {result_type}']

        elif target_name == 'logical_xor':
            return [f'{result_ssa} = stablehlo.xor {get_input(0)}, {get_input(1)} : {result_type}']

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
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', -1)

            # Get shape info for reduction
            input_shape = self.shape_map.get(node.args[0].name, ()) if hasattr(node.args[0], 'name') else ()
            input_rank = len(input_shape)
            if dim < 0:
                dim = input_rank + dim

            # Build reduced shape type (for max and sum) - keeps dim with size 1
            reduced_shape = list(input_shape)
            reduced_shape[dim] = 1
            reduced_shape_str = 'x'.join(str(d) for d in reduced_shape)
            reduced_type = f'tensor<{reduced_shape_str}xf32>'

            # Build broadcast dims - all dims except the reduced one map to themselves
            broadcast_dims = [i for i in range(input_rank)]

            # Softmax = exp(x - max(x)) / sum(exp(x - max(x)))
            # For numerical stability, subtract max before exp
            max_ssa = f'{result_ssa}_max'
            max_bc_ssa = f'{result_ssa}_max_bc'
            shifted_ssa = f'{result_ssa}_shifted'
            exp_ssa = f'{result_ssa}_exp'
            sum_ssa = f'{result_ssa}_sum'
            sum_bc_ssa = f'{result_ssa}_sum_bc'
            init_max = f'{result_ssa}_init_max'
            init_sum = f'{result_ssa}_init_sum'

            lines = [
                # Step 1: Find max along dim (for numerical stability)
                f'{init_max} = stablehlo.constant dense<-3.40282e+38> : tensor<f32>',
                f'{max_ssa} = stablehlo.reduce {input_ssa}, {init_max}, dims = [{dim}], reducer = max : ({input_type}, tensor<f32>) -> {reduced_type}',

                # Step 2: Broadcast max back to original shape and subtract
                f'{max_bc_ssa} = stablehlo.broadcast_in_dim {max_ssa}, dims = [{", ".join(str(d) for d in broadcast_dims)}] : ({reduced_type}) -> {result_type}',
                f'{shifted_ssa} = stablehlo.subtract {input_ssa}, {max_bc_ssa} : {result_type}',

                # Step 3: Compute exp(x - max)
                f'{exp_ssa} = stablehlo.exponential {shifted_ssa} : {result_type}',

                # Step 4: Sum exp along dim
                f'{init_sum} = stablehlo.constant dense<0.0> : tensor<f32>',
                f'{sum_ssa} = stablehlo.reduce {exp_ssa}, {init_sum}, dims = [{dim}], reducer = add : ({result_type}, tensor<f32>) -> {reduced_type}',

                # Step 5: Broadcast sum and divide
                f'{sum_bc_ssa} = stablehlo.broadcast_in_dim {sum_ssa}, dims = [{", ".join(str(d) for d in broadcast_dims)}] : ({reduced_type}) -> {result_type}',
                f'{result_ssa} = stablehlo.divide {exp_ssa}, {sum_bc_ssa} : {result_type}'
            ]
            return lines

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

        # ===== SCAN/CUMULATIVE OPERATIONS =====
        # StableHLO lacks native scan operations. We use custom_call which backends
        # can implement using their native scan primitives (e.g., thrust::inclusive_scan
        # on CUDA, parallel_scan on CPU). Alternative: decompose using stablehlo.while.

        elif target_name == 'cumsum':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', 0)
            dtype = node.kwargs.get('dtype', None)
            dtype_attr = f', dtype = "{dtype}"' if dtype else ''
            return [f'{result_ssa} = stablehlo.custom_call @cumsum({input_ssa}) '
                    f'{{dim = {dim}{dtype_attr}}} : ({input_type}) -> {result_type}']

        elif target_name == 'cumprod':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', 0)
            dtype = node.kwargs.get('dtype', None)
            dtype_attr = f', dtype = "{dtype}"' if dtype else ''
            return [f'{result_ssa} = stablehlo.custom_call @cumprod({input_ssa}) '
                    f'{{dim = {dim}{dtype_attr}}} : ({input_type}) -> {result_type}']

        elif target_name == 'cummax':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', 0)
            return [f'{result_ssa} = stablehlo.custom_call @cummax({input_ssa}) '
                    f'{{dim = {dim}}} : ({input_type}) -> {result_type}']

        elif target_name == 'cummin':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', 0)
            return [f'{result_ssa} = stablehlo.custom_call @cummin({input_ssa}) '
                    f'{{dim = {dim}}} : ({input_type}) -> {result_type}']

        elif target_name == 'logcumsumexp':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', 0)
            return [f'{result_ssa} = stablehlo.custom_call @logcumsumexp({input_ssa}) '
                    f'{{dim = {dim}}} : ({input_type}) -> {result_type}']

        elif target_name == 'diff':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            n = node.args[1] if len(node.args) > 1 else node.kwargs.get('n', 1)
            dim = node.args[2] if len(node.args) > 2 else node.kwargs.get('dim', -1)
            return [f'{result_ssa} = stablehlo.custom_call @diff({input_ssa}) '
                    f'{{n = {n}, dim = {dim}}} : ({input_type}) -> {result_type}']

        # ===== RNN/LSTM/GRU OPERATIONS =====
        # StableHLO has no native RNN ops. Use custom_call for backend-optimized
        # implementations (cuDNN on NVIDIA, MIOpen on AMD, oneDNN on CPU).

        elif target_name in ('lstm', 'lstm_cell'):
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            hx = get_input(1) if len(node.args) > 1 else None
            hidden_size = node.kwargs.get('hidden_size', 256)
            num_layers = node.kwargs.get('num_layers', 1)
            bias = node.kwargs.get('bias', True)
            batch_first = node.kwargs.get('batch_first', False)
            dropout = node.kwargs.get('dropout', 0.0)
            bidirectional = node.kwargs.get('bidirectional', False)
            attrs = f'hidden_size = {hidden_size}, num_layers = {num_layers}, bias = {str(bias).lower()}, '
            attrs += f'batch_first = {str(batch_first).lower()}, dropout = {dropout}, bidirectional = {str(bidirectional).lower()}'
            if hx:
                hx_type = get_input_type(1)
                return [f'{result_ssa} = stablehlo.custom_call @lstm({input_ssa}, {hx}) '
                        f'{{{attrs}}} : ({input_type}, {hx_type}) -> {result_type}']
            return [f'{result_ssa} = stablehlo.custom_call @lstm({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('gru', 'gru_cell'):
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            hx = get_input(1) if len(node.args) > 1 else None
            hidden_size = node.kwargs.get('hidden_size', 256)
            num_layers = node.kwargs.get('num_layers', 1)
            bias = node.kwargs.get('bias', True)
            batch_first = node.kwargs.get('batch_first', False)
            dropout = node.kwargs.get('dropout', 0.0)
            bidirectional = node.kwargs.get('bidirectional', False)
            attrs = f'hidden_size = {hidden_size}, num_layers = {num_layers}, bias = {str(bias).lower()}, '
            attrs += f'batch_first = {str(batch_first).lower()}, dropout = {dropout}, bidirectional = {str(bidirectional).lower()}'
            if hx:
                hx_type = get_input_type(1)
                return [f'{result_ssa} = stablehlo.custom_call @gru({input_ssa}, {hx}) '
                        f'{{{attrs}}} : ({input_type}, {hx_type}) -> {result_type}']
            return [f'{result_ssa} = stablehlo.custom_call @gru({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('rnn_tanh', 'rnn_relu', 'rnn'):
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            hx = get_input(1) if len(node.args) > 1 else None
            hidden_size = node.kwargs.get('hidden_size', 256)
            num_layers = node.kwargs.get('num_layers', 1)
            nonlinearity = node.kwargs.get('nonlinearity', 'tanh')
            bias = node.kwargs.get('bias', True)
            batch_first = node.kwargs.get('batch_first', False)
            dropout = node.kwargs.get('dropout', 0.0)
            bidirectional = node.kwargs.get('bidirectional', False)
            attrs = f'hidden_size = {hidden_size}, num_layers = {num_layers}, nonlinearity = "{nonlinearity}", '
            attrs += f'bias = {str(bias).lower()}, batch_first = {str(batch_first).lower()}, '
            attrs += f'dropout = {dropout}, bidirectional = {str(bidirectional).lower()}'
            if hx:
                hx_type = get_input_type(1)
                return [f'{result_ssa} = stablehlo.custom_call @rnn({input_ssa}, {hx}) '
                        f'{{{attrs}}} : ({input_type}, {hx_type}) -> {result_type}']
            return [f'{result_ssa} = stablehlo.custom_call @rnn({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        # ===== ATTENTION OPERATIONS =====
        # Attention mechanisms for transformers and sequence models.
        # Uses custom_call for backend-optimized implementations (Flash Attention, etc.)

        elif target_name in ('scaled_dot_product_attention', '_scaled_dot_product_attention'):
            # F.scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal)
            query_ssa = get_input(0)
            key_ssa = get_input(1)
            value_ssa = get_input(2)
            query_type = get_input_type(0)
            key_type = get_input_type(1)
            value_type = get_input_type(2)
            attn_mask = get_input(3) if len(node.args) > 3 else None
            dropout_p = node.args[4] if len(node.args) > 4 else node.kwargs.get('dropout_p', 0.0)
            is_causal = node.args[5] if len(node.args) > 5 else node.kwargs.get('is_causal', False)
            scale = node.kwargs.get('scale', None)
            attrs = f'dropout_p = {dropout_p}, is_causal = {str(is_causal).lower()}'
            if scale is not None:
                attrs += f', scale = {scale}'
            if attn_mask:
                mask_type = get_input_type(3)
                return [f'{result_ssa} = stablehlo.custom_call @scaled_dot_product_attention'
                        f'({query_ssa}, {key_ssa}, {value_ssa}, {attn_mask}) '
                        f'{{{attrs}}} : ({query_type}, {key_type}, {value_type}, {mask_type}) -> {result_type}']
            return [f'{result_ssa} = stablehlo.custom_call @scaled_dot_product_attention'
                    f'({query_ssa}, {key_ssa}, {value_ssa}) '
                    f'{{{attrs}}} : ({query_type}, {key_type}, {value_type}) -> {result_type}']

        elif target_name in ('multi_head_attention_forward', 'multihead_attention'):
            # nn.MultiheadAttention forward
            query_ssa = get_input(0)
            key_ssa = get_input(1)
            value_ssa = get_input(2)
            query_type = get_input_type(0)
            key_type = get_input_type(1)
            value_type = get_input_type(2)
            embed_dim = node.kwargs.get('embed_dim', 512)
            num_heads = node.kwargs.get('num_heads', 8)
            dropout = node.kwargs.get('dropout', 0.0)
            bias = node.kwargs.get('bias', True)
            add_bias_kv = node.kwargs.get('add_bias_kv', False)
            add_zero_attn = node.kwargs.get('add_zero_attn', False)
            batch_first = node.kwargs.get('batch_first', False)
            attrs = f'embed_dim = {embed_dim}, num_heads = {num_heads}, dropout = {dropout}, '
            attrs += f'bias = {str(bias).lower()}, add_bias_kv = {str(add_bias_kv).lower()}, '
            attrs += f'add_zero_attn = {str(add_zero_attn).lower()}, batch_first = {str(batch_first).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @multi_head_attention'
                    f'({query_ssa}, {key_ssa}, {value_ssa}) '
                    f'{{{attrs}}} : ({query_type}, {key_type}, {value_type}) -> {result_type}']

        elif target_name == 'softmax':
            # Already handled above, but add attention-specific variant
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', -1)
            return [f'{result_ssa} = stablehlo.custom_call @softmax({input_ssa}) : ({input_type}) -> {result_type}']

        # ===== EMBEDDING OPERATIONS =====
        # Embedding lookup operations for NLP and recommendation systems.
        # Uses stablehlo.gather for basic embedding, custom_call for advanced modes.

        elif target_name in ('embedding', 'embedding_lookup'):
            # F.embedding(input, weight) or nn.Embedding forward
            input_ssa = get_input(0)
            weight_ssa = get_input(1)
            input_type = get_input_type(0)
            weight_type = get_input_type(1)
            padding_idx = node.kwargs.get('padding_idx', None)
            max_norm = node.kwargs.get('max_norm', None)
            norm_type = node.kwargs.get('norm_type', 2.0)
            scale_grad_by_freq = node.kwargs.get('scale_grad_by_freq', False)
            sparse = node.kwargs.get('sparse', False)
            if max_norm is not None or sparse:
                attrs = f'padding_idx = {padding_idx}, max_norm = {max_norm}, norm_type = {norm_type}, '
                attrs += f'scale_grad_by_freq = {str(scale_grad_by_freq).lower()}, sparse = {str(sparse).lower()}'
                return [f'{result_ssa} = stablehlo.custom_call @embedding({input_ssa}, {weight_ssa}) '
                        f'{{{attrs}}} : ({input_type}, {weight_type}) -> {result_type}']
            else:
                return [f'{result_ssa} = stablehlo.gather {weight_ssa}[{input_ssa}] : ({weight_type}, {input_type}) -> {result_type}']

        elif target_name in ('embedding_bag', '_embedding_bag'):
            # F.embedding_bag - embedding lookup with reduction (sum, mean, max)
            input_ssa = get_input(0)
            weight_ssa = get_input(1)
            input_type = get_input_type(0)
            weight_type = get_input_type(1)
            offsets = get_input(2) if len(node.args) > 2 else None
            mode = node.kwargs.get('mode', 'mean')
            sparse = node.kwargs.get('sparse', False)
            include_last_offset = node.kwargs.get('include_last_offset', False)
            padding_idx = node.kwargs.get('padding_idx', None)
            per_sample_weights = get_input(3) if len(node.args) > 3 else None
            attrs = f'mode = "{mode}", sparse = {str(sparse).lower()}, '
            attrs += f'include_last_offset = {str(include_last_offset).lower()}'
            if padding_idx is not None:
                attrs += f', padding_idx = {padding_idx}'
            if offsets and per_sample_weights:
                offsets_type = get_input_type(2)
                psw_type = get_input_type(3)
                return [f'{result_ssa} = stablehlo.custom_call @embedding_bag'
                        f'({input_ssa}, {weight_ssa}, {offsets}, {per_sample_weights}) '
                        f'{{{attrs}}} : ({input_type}, {weight_type}, {offsets_type}, {psw_type}) -> {result_type}']
            elif offsets:
                offsets_type = get_input_type(2)
                return [f'{result_ssa} = stablehlo.custom_call @embedding_bag'
                        f'({input_ssa}, {weight_ssa}, {offsets}) '
                        f'{{{attrs}}} : ({input_type}, {weight_type}, {offsets_type}) -> {result_type}']
            return [f'{result_ssa} = stablehlo.custom_call @embedding_bag({input_ssa}, {weight_ssa}) '
                    f'{{{attrs}}} : ({input_type}, {weight_type}) -> {result_type}']

        elif target_name == 'one_hot':
            # F.one_hot(tensor, num_classes)
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            num_classes = node.args[1] if len(node.args) > 1 else node.kwargs.get('num_classes', -1)
            return [f'{result_ssa} = stablehlo.custom_call @one_hot({input_ssa}) '
                    f'{{num_classes = {num_classes}}} : ({input_type}) -> {result_type}']

        # ===== LOSS FUNCTION OPERATIONS =====
        # Common loss functions for training neural networks.
        # Uses custom_call for complex multi-step loss computations.

        elif target_name in ('cross_entropy_loss', 'cross_entropy', 'nll_loss'):
            # F.cross_entropy / nn.CrossEntropyLoss / F.nll_loss
            input_ssa = get_input(0)
            target_ssa = get_input(1)
            input_type = get_input_type(0)
            target_type = get_input_type(1)
            weight = get_input(2) if len(node.args) > 2 else None
            reduction = node.kwargs.get('reduction', 'mean')
            ignore_index = node.kwargs.get('ignore_index', -100)
            label_smoothing = node.kwargs.get('label_smoothing', 0.0)
            attrs = f'reduction = "{reduction}", ignore_index = {ignore_index}, label_smoothing = {label_smoothing}'
            if weight:
                weight_type = get_input_type(2)
                return [f'{result_ssa} = stablehlo.custom_call @cross_entropy_loss'
                        f'({input_ssa}, {target_ssa}, {weight}) '
                        f'{{{attrs}}} : ({input_type}, {target_type}, {weight_type}) -> {result_type}']
            return [f'{result_ssa} = stablehlo.custom_call @cross_entropy_loss({input_ssa}, {target_ssa}) '
                    f'{{{attrs}}} : ({input_type}, {target_type}) -> {result_type}']

        elif target_name in ('binary_cross_entropy', 'bce_loss'):
            # F.binary_cross_entropy / nn.BCELoss
            input_ssa = get_input(0)
            target_ssa = get_input(1)
            input_type = get_input_type(0)
            target_type = get_input_type(1)
            weight = get_input(2) if len(node.args) > 2 else None
            reduction = node.kwargs.get('reduction', 'mean')
            attrs = f'reduction = "{reduction}"'
            if weight:
                weight_type = get_input_type(2)
                return [f'{result_ssa} = stablehlo.custom_call @binary_cross_entropy'
                        f'({input_ssa}, {target_ssa}, {weight}) '
                        f'{{{attrs}}} : ({input_type}, {target_type}, {weight_type}) -> {result_type}']
            return [f'{result_ssa} = stablehlo.custom_call @binary_cross_entropy({input_ssa}, {target_ssa}) '
                    f'{{{attrs}}} : ({input_type}, {target_type}) -> {result_type}']

        elif target_name in ('binary_cross_entropy_with_logits', 'bce_with_logits_loss'):
            # F.binary_cross_entropy_with_logits / nn.BCEWithLogitsLoss
            input_ssa = get_input(0)
            target_ssa = get_input(1)
            input_type = get_input_type(0)
            target_type = get_input_type(1)
            weight = node.kwargs.get('weight', None)
            pos_weight = node.kwargs.get('pos_weight', None)
            reduction = node.kwargs.get('reduction', 'mean')
            attrs = f'reduction = "{reduction}"'
            return [f'{result_ssa} = stablehlo.custom_call @bce_with_logits({input_ssa}, {target_ssa}) '
                    f'{{{attrs}}} : ({input_type}, {target_type}) -> {result_type}']

        elif target_name in ('mse_loss', 'l2_loss'):
            # F.mse_loss / nn.MSELoss
            input_ssa = get_input(0)
            target_ssa = get_input(1)
            input_type = get_input_type(0)
            target_type = get_input_type(1)
            reduction = node.kwargs.get('reduction', 'mean')
            attrs = f'reduction = "{reduction}"'
            return [f'{result_ssa} = stablehlo.custom_call @mse_loss({input_ssa}, {target_ssa}) '
                    f'{{{attrs}}} : ({input_type}, {target_type}) -> {result_type}']

        elif target_name in ('l1_loss', 'mae_loss'):
            # F.l1_loss / nn.L1Loss
            input_ssa = get_input(0)
            target_ssa = get_input(1)
            input_type = get_input_type(0)
            target_type = get_input_type(1)
            reduction = node.kwargs.get('reduction', 'mean')
            attrs = f'reduction = "{reduction}"'
            return [f'{result_ssa} = stablehlo.custom_call @l1_loss({input_ssa}, {target_ssa}) '
                    f'{{{attrs}}} : ({input_type}, {target_type}) -> {result_type}']

        elif target_name == 'smooth_l1_loss':
            # F.smooth_l1_loss / nn.SmoothL1Loss (Huber loss)
            input_ssa = get_input(0)
            target_ssa = get_input(1)
            input_type = get_input_type(0)
            target_type = get_input_type(1)
            reduction = node.kwargs.get('reduction', 'mean')
            beta = node.kwargs.get('beta', 1.0)
            attrs = f'reduction = "{reduction}", beta = {beta}'
            return [f'{result_ssa} = stablehlo.custom_call @smooth_l1_loss({input_ssa}, {target_ssa}) '
                    f'{{{attrs}}} : ({input_type}, {target_type}) -> {result_type}']

        elif target_name == 'huber_loss':
            # F.huber_loss / nn.HuberLoss
            input_ssa = get_input(0)
            target_ssa = get_input(1)
            input_type = get_input_type(0)
            target_type = get_input_type(1)
            reduction = node.kwargs.get('reduction', 'mean')
            delta = node.kwargs.get('delta', 1.0)
            attrs = f'reduction = "{reduction}", delta = {delta}'
            return [f'{result_ssa} = stablehlo.custom_call @huber_loss({input_ssa}, {target_ssa}) '
                    f'{{{attrs}}} : ({input_type}, {target_type}) -> {result_type}']

        elif target_name == 'kl_div':
            # F.kl_div / nn.KLDivLoss
            input_ssa = get_input(0)
            target_ssa = get_input(1)
            input_type = get_input_type(0)
            target_type = get_input_type(1)
            reduction = node.kwargs.get('reduction', 'mean')
            log_target = node.kwargs.get('log_target', False)
            attrs = f'reduction = "{reduction}", log_target = {str(log_target).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @kl_div({input_ssa}, {target_ssa}) '
                    f'{{{attrs}}} : ({input_type}, {target_type}) -> {result_type}']

        elif target_name in ('hinge_embedding_loss', 'hinge_loss'):
            # F.hinge_embedding_loss / nn.HingeEmbeddingLoss
            input_ssa = get_input(0)
            target_ssa = get_input(1)
            input_type = get_input_type(0)
            target_type = get_input_type(1)
            margin = node.kwargs.get('margin', 1.0)
            reduction = node.kwargs.get('reduction', 'mean')
            attrs = f'margin = {margin}, reduction = "{reduction}"'
            return [f'{result_ssa} = stablehlo.custom_call @hinge_embedding_loss({input_ssa}, {target_ssa}) '
                    f'{{{attrs}}} : ({input_type}, {target_type}) -> {result_type}']

        elif target_name in ('margin_ranking_loss',):
            # F.margin_ranking_loss / nn.MarginRankingLoss
            input1_ssa = get_input(0)
            input2_ssa = get_input(1)
            target_ssa = get_input(2)
            input1_type = get_input_type(0)
            input2_type = get_input_type(1)
            target_type = get_input_type(2)
            margin = node.kwargs.get('margin', 0.0)
            reduction = node.kwargs.get('reduction', 'mean')
            attrs = f'margin = {margin}, reduction = "{reduction}"'
            return [f'{result_ssa} = stablehlo.custom_call @margin_ranking_loss'
                    f'({input1_ssa}, {input2_ssa}, {target_ssa}) '
                    f'{{{attrs}}} : ({input1_type}, {input2_type}, {target_type}) -> {result_type}']

        elif target_name == 'triplet_margin_loss':
            # F.triplet_margin_loss / nn.TripletMarginLoss
            anchor_ssa = get_input(0)
            positive_ssa = get_input(1)
            negative_ssa = get_input(2)
            anchor_type = get_input_type(0)
            positive_type = get_input_type(1)
            negative_type = get_input_type(2)
            margin = node.kwargs.get('margin', 1.0)
            p = node.kwargs.get('p', 2)
            reduction = node.kwargs.get('reduction', 'mean')
            attrs = f'margin = {margin}, p = {p}, reduction = "{reduction}"'
            return [f'{result_ssa} = stablehlo.custom_call @triplet_margin_loss'
                    f'({anchor_ssa}, {positive_ssa}, {negative_ssa}) '
                    f'{{{attrs}}} : ({anchor_type}, {positive_type}, {negative_type}) -> {result_type}']

        elif target_name == 'cosine_embedding_loss':
            # F.cosine_embedding_loss / nn.CosineEmbeddingLoss
            input1_ssa = get_input(0)
            input2_ssa = get_input(1)
            target_ssa = get_input(2)
            input1_type = get_input_type(0)
            input2_type = get_input_type(1)
            target_type = get_input_type(2)
            margin = node.kwargs.get('margin', 0.0)
            reduction = node.kwargs.get('reduction', 'mean')
            attrs = f'margin = {margin}, reduction = "{reduction}"'
            return [f'{result_ssa} = stablehlo.custom_call @cosine_embedding_loss'
                    f'({input1_ssa}, {input2_ssa}, {target_ssa}) '
                    f'{{{attrs}}} : ({input1_type}, {input2_type}, {target_type}) -> {result_type}']

        elif target_name in ('ctc_loss',):
            # F.ctc_loss / nn.CTCLoss
            log_probs_ssa = get_input(0)
            targets_ssa = get_input(1)
            log_probs_type = get_input_type(0)
            targets_type = get_input_type(1)
            input_lengths = get_input(2) if len(node.args) > 2 else None
            target_lengths = get_input(3) if len(node.args) > 3 else None
            blank = node.kwargs.get('blank', 0)
            reduction = node.kwargs.get('reduction', 'mean')
            zero_infinity = node.kwargs.get('zero_infinity', False)
            attrs = f'blank = {blank}, reduction = "{reduction}", zero_infinity = {str(zero_infinity).lower()}'
            if input_lengths and target_lengths:
                il_type = get_input_type(2)
                tl_type = get_input_type(3)
                return [f'{result_ssa} = stablehlo.custom_call @ctc_loss'
                        f'({log_probs_ssa}, {targets_ssa}, {input_lengths}, {target_lengths}) '
                        f'{{{attrs}}} : ({log_probs_type}, {targets_type}, {il_type}, {tl_type}) -> {result_type}']
            return [f'{result_ssa} = stablehlo.custom_call @ctc_loss({log_probs_ssa}, {targets_ssa}) '
                    f'{{{attrs}}} : ({log_probs_type}, {targets_type}) -> {result_type}']

        elif target_name in ('poisson_nll_loss',):
            # F.poisson_nll_loss / nn.PoissonNLLLoss
            input_ssa = get_input(0)
            target_ssa = get_input(1)
            input_type = get_input_type(0)
            target_type = get_input_type(1)
            log_input = node.kwargs.get('log_input', True)
            full = node.kwargs.get('full', False)
            eps = node.kwargs.get('eps', 1e-8)
            reduction = node.kwargs.get('reduction', 'mean')
            attrs = f'log_input = {str(log_input).lower()}, full = {str(full).lower()}, eps = {eps}, reduction = "{reduction}"'
            return [f'{result_ssa} = stablehlo.custom_call @poisson_nll_loss({input_ssa}, {target_ssa}) '
                    f'{{{attrs}}} : ({input_type}, {target_type}) -> {result_type}']

        elif target_name in ('gaussian_nll_loss',):
            # F.gaussian_nll_loss / nn.GaussianNLLLoss
            input_ssa = get_input(0)
            target_ssa = get_input(1)
            var_ssa = get_input(2)
            input_type = get_input_type(0)
            target_type = get_input_type(1)
            var_type = get_input_type(2)
            full = node.kwargs.get('full', False)
            eps = node.kwargs.get('eps', 1e-6)
            reduction = node.kwargs.get('reduction', 'mean')
            attrs = f'full = {str(full).lower()}, eps = {eps}, reduction = "{reduction}"'
            return [f'{result_ssa} = stablehlo.custom_call @gaussian_nll_loss'
                    f'({input_ssa}, {target_ssa}, {var_ssa}) '
                    f'{{{attrs}}} : ({input_type}, {target_type}, {var_type}) -> {result_type}']

        elif target_name in ('soft_margin_loss',):
            # F.soft_margin_loss / nn.SoftMarginLoss
            input_ssa = get_input(0)
            target_ssa = get_input(1)
            input_type = get_input_type(0)
            target_type = get_input_type(1)
            reduction = node.kwargs.get('reduction', 'mean')
            attrs = f'reduction = "{reduction}"'
            return [f'{result_ssa} = stablehlo.custom_call @soft_margin_loss({input_ssa}, {target_ssa}) '
                    f'{{{attrs}}} : ({input_type}, {target_type}) -> {result_type}']

        elif target_name in ('multi_margin_loss',):
            # F.multi_margin_loss / nn.MultiMarginLoss
            input_ssa = get_input(0)
            target_ssa = get_input(1)
            input_type = get_input_type(0)
            target_type = get_input_type(1)
            p = node.kwargs.get('p', 1)
            margin = node.kwargs.get('margin', 1.0)
            reduction = node.kwargs.get('reduction', 'mean')
            attrs = f'p = {p}, margin = {margin}, reduction = "{reduction}"'
            return [f'{result_ssa} = stablehlo.custom_call @multi_margin_loss({input_ssa}, {target_ssa}) '
                    f'{{{attrs}}} : ({input_type}, {target_type}) -> {result_type}']

        elif target_name in ('multilabel_margin_loss',):
            # F.multilabel_margin_loss / nn.MultiLabelMarginLoss
            input_ssa = get_input(0)
            target_ssa = get_input(1)
            input_type = get_input_type(0)
            target_type = get_input_type(1)
            reduction = node.kwargs.get('reduction', 'mean')
            attrs = f'reduction = "{reduction}"'
            return [f'{result_ssa} = stablehlo.custom_call @multilabel_margin_loss({input_ssa}, {target_ssa}) '
                    f'{{{attrs}}} : ({input_type}, {target_type}) -> {result_type}']

        elif target_name in ('multilabel_soft_margin_loss',):
            # F.multilabel_soft_margin_loss / nn.MultiLabelSoftMarginLoss
            input_ssa = get_input(0)
            target_ssa = get_input(1)
            input_type = get_input_type(0)
            target_type = get_input_type(1)
            reduction = node.kwargs.get('reduction', 'mean')
            attrs = f'reduction = "{reduction}"'
            return [f'{result_ssa} = stablehlo.custom_call @multilabel_soft_margin_loss({input_ssa}, {target_ssa}) '
                    f'{{{attrs}}} : ({input_type}, {target_type}) -> {result_type}']

        elif target_name == 'std':
            # std(x) = sqrt(var(x)) = sqrt(mean((x - mean(x))^2))
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', None)
            if dim is None:
                dim = 0
            # Simplified: use custom_call for complex multi-step reduction
            return [f'{result_ssa} = stablehlo.custom_call @std({input_ssa}) {{dim = {dim}}} : ({input_type}) -> {result_type}']

        elif target_name == 'var':
            # var(x) = mean((x - mean(x))^2)
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', None)
            if dim is None:
                dim = 0
            return [f'{result_ssa} = stablehlo.custom_call @var({input_ssa}) {{dim = {dim}}} : ({input_type}) -> {result_type}']

        elif target_name == 'argmax':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', None)
            if dim is None:
                dim = 0
            return [f'{result_ssa} = stablehlo.custom_call @argmax({input_ssa}) {{dim = {dim}}} : ({input_type}) -> {result_type}']

        elif target_name == 'argmin':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', None)
            if dim is None:
                dim = 0
            return [f'{result_ssa} = stablehlo.custom_call @argmin({input_ssa}) {{dim = {dim}}} : ({input_type}) -> {result_type}']

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

        elif target_name == 'celu':
            # celu(x, alpha) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
            input_ssa = get_input(0)
            alpha = node.args[1] if len(node.args) > 1 else node.kwargs.get('alpha', 1.0)
            zero_ssa = f'{result_ssa}_zero'
            alpha_ssa = f'{result_ssa}_alpha'
            div_ssa = f'{result_ssa}_div'
            exp_ssa = f'{result_ssa}_exp'
            one_ssa = f'{result_ssa}_one'
            sub_ssa = f'{result_ssa}_sub'
            scaled_ssa = f'{result_ssa}_scaled'
            cmp_ssa = f'{result_ssa}_cmp'
            return [
                f'{zero_ssa} = stablehlo.constant dense<0.0> : {result_type}',
                f'{alpha_ssa} = stablehlo.constant dense<{alpha}> : {result_type}',
                f'{one_ssa} = stablehlo.constant dense<1.0> : {result_type}',
                f'{div_ssa} = stablehlo.divide {input_ssa}, {alpha_ssa} : {result_type}',
                f'{exp_ssa} = stablehlo.exponential {div_ssa} : {result_type}',
                f'{sub_ssa} = stablehlo.subtract {exp_ssa}, {one_ssa} : {result_type}',
                f'{scaled_ssa} = stablehlo.multiply {alpha_ssa}, {sub_ssa} : {result_type}',
                f'{cmp_ssa} = stablehlo.compare GE, {input_ssa}, {zero_ssa} : ({result_type}, {result_type}) -> tensor<i1>',
                f'{result_ssa} = stablehlo.select {cmp_ssa}, {input_ssa}, {scaled_ssa} : (tensor<i1>, {result_type}, {result_type}) -> {result_type}'
            ]

        elif target_name == 'logsigmoid':
            # logsigmoid(x) = log(sigmoid(x)) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
            input_ssa = get_input(0)
            neg_ssa = f'{result_ssa}_neg'
            exp_ssa = f'{result_ssa}_exp'
            one_ssa = f'{result_ssa}_one'
            add_ssa = f'{result_ssa}_add'
            log_ssa = f'{result_ssa}_log'
            return [
                f'{neg_ssa} = stablehlo.negate {input_ssa} : {result_type}',
                f'{exp_ssa} = stablehlo.exponential {neg_ssa} : {result_type}',
                f'{one_ssa} = stablehlo.constant dense<1.0> : {result_type}',
                f'{add_ssa} = stablehlo.add {one_ssa}, {exp_ssa} : {result_type}',
                f'{log_ssa} = stablehlo.log {add_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.negate {log_ssa} : {result_type}'
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

            # Determine dimensions based on input ranks
            lhs_shape = self.shape_map.get(node.args[0].name, ()) if hasattr(node.args[0], 'name') else ()
            rhs_shape = self.shape_map.get(node.args[1].name, ()) if hasattr(node.args[1], 'name') else ()
            lhs_rank = len(lhs_shape)
            rhs_rank = len(rhs_shape)

            if lhs_rank >= 3 and rhs_rank >= 3:
                # Batched matmul: (..., M, K) x (..., K, N) -> (..., M, N)
                # Batch dims are all but the last two
                batch_dims = list(range(lhs_rank - 2))
                lhs_contract = lhs_rank - 1  # Last dim of LHS (K)
                rhs_contract = rhs_rank - 2  # Second-to-last dim of RHS (K)
                batch_dims_str = ', '.join(str(d) for d in batch_dims)
                dot_attr = f'#stablehlo.dot<lhs_batching_dimensions = [{batch_dims_str}], rhs_batching_dimensions = [{batch_dims_str}], lhs_contracting_dimensions = [{lhs_contract}], rhs_contracting_dimensions = [{rhs_contract}]>'
            elif lhs_rank == 2 and rhs_rank == 2:
                # 2D matmul: (M, K) x (K, N) -> (M, N)
                dot_attr = '#stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>'
            else:
                # Default: contract last dim of LHS with first dim of RHS
                lhs_contract = max(0, lhs_rank - 1)
                rhs_contract = max(0, rhs_rank - 2) if rhs_rank >= 2 else 0
                dot_attr = f'#stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [{lhs_contract}], rhs_contracting_dimensions = [{rhs_contract}]>'

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
            input_node = node.args[0] if hasattr(node.args[0], 'name') else None
            input_name = input_node.name if input_node else ''

            # Use dynamic_reshape if input or output has dynamic dimensions
            if self._is_dynamic_shape(node.name) or self._is_dynamic_shape(input_name):
                lines = self._get_shape_tensor(node.name, result_ssa)
                shape_ssa = f'{result_ssa}_shape'
                output_shape = self.shape_map.get(node.name, ())
                lines.append(f'{result_ssa} = stablehlo.dynamic_reshape {input_ssa}, {shape_ssa} : '
                             f'({input_type}, tensor<{len(output_shape)}xi32>) -> {result_type}')
                return lines
            return [f'{result_ssa} = stablehlo.reshape {input_ssa} : {input_type} -> {result_type}']

        elif target_name == 'view':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            input_node = node.args[0] if hasattr(node.args[0], 'name') else None
            input_name = input_node.name if input_node else ''

            # Use dynamic_reshape if input or output has dynamic dimensions
            if self._is_dynamic_shape(node.name) or self._is_dynamic_shape(input_name):
                lines = self._get_shape_tensor(node.name, result_ssa)
                shape_ssa = f'{result_ssa}_shape'
                output_shape = self.shape_map.get(node.name, ())
                lines.append(f'{result_ssa} = stablehlo.dynamic_reshape {input_ssa}, {shape_ssa} : '
                             f'({input_type}, tensor<{len(output_shape)}xi32>) -> {result_type}')
                return lines
            return [f'{result_ssa} = stablehlo.reshape {input_ssa} : {input_type} -> {result_type}']

        elif target_name == 'flatten':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            input_node = node.args[0] if hasattr(node.args[0], 'name') else None
            input_name = input_node.name if input_node else ''

            # Use dynamic_reshape if input or output has dynamic dimensions
            if self._is_dynamic_shape(node.name) or self._is_dynamic_shape(input_name):
                lines = self._get_shape_tensor(node.name, result_ssa)
                shape_ssa = f'{result_ssa}_shape'
                output_shape = self.shape_map.get(node.name, ())
                lines.append(f'{result_ssa} = stablehlo.dynamic_reshape {input_ssa}, {shape_ssa} : '
                             f'({input_type}, tensor<{len(output_shape)}xi32>) -> {result_type}')
                return lines
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

        elif target_name == 'narrow':
            # narrow(input, dim, start, length) -> slice
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            input_node = node.args[0] if hasattr(node.args[0], 'name') else None
            input_name = input_node.name if input_node else ''
            dim = node.args[1] if len(node.args) > 1 else 0
            start = node.args[2] if len(node.args) > 2 else 0
            length = node.args[3] if len(node.args) > 3 else 1

            # Use dynamic_slice if input or output has dynamic dimensions
            if self._is_dynamic_shape(node.name) or self._is_dynamic_shape(input_name):
                input_shape = self.shape_map.get(input_name, ())
                ndim = len(input_shape)
                # Build start_indices and slice_sizes tensors
                start_indices = [0] * ndim
                start_indices[dim] = start
                slice_sizes = list(input_shape)
                slice_sizes[dim] = length
                start_ssa = f'{result_ssa}_start'
                sizes_ssa = f'{result_ssa}_sizes'
                start_str = ', '.join(str(s) for s in start_indices)
                sizes_str = ', '.join(str(s) if not (i in self.dynamic_dim_map.get(node.name, set())) else '?'
                                      for i, s in enumerate(slice_sizes))
                return [
                    f'{start_ssa} = stablehlo.constant dense<[{start_str}]> : tensor<{ndim}xi64>',
                    f'{result_ssa} = stablehlo.dynamic_slice {input_ssa}, {start_ssa}, '
                    f'slice_sizes = [{sizes_str}] : ({input_type}, tensor<{ndim}xi64>) -> {result_type}'
                ]
            return [f'{result_ssa} = stablehlo.slice {input_ssa}, dim = {dim}, start = {start}, limit = {start + length} : {input_type} -> {result_type}']

        elif target_name == 'unbind':
            # unbind returns tuple of tensors - use custom_call
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else 0
            return [f'{result_ssa} = stablehlo.custom_call @unbind({input_ssa}) {{dim = {dim}}} : ({input_type}) -> {result_type}']

        elif target_name == 'select':
            # select(input, dim, index) -> reduce dimension
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            input_node = node.args[0] if hasattr(node.args[0], 'name') else None
            input_name = input_node.name if input_node else ''
            dim = node.args[1] if len(node.args) > 1 else 0
            index = node.args[2] if len(node.args) > 2 else 0

            # Use dynamic_slice if input has dynamic dimensions
            if self._is_dynamic_shape(input_name):
                input_shape = self.shape_map.get(input_name, ())
                ndim = len(input_shape)
                start_indices = [0] * ndim
                start_indices[dim] = index
                slice_sizes = list(input_shape)
                slice_sizes[dim] = 1
                start_ssa = f'{result_ssa}_start'
                start_str = ', '.join(str(s) for s in start_indices)
                sizes_str = ', '.join(str(s) if not (i in self.dynamic_dim_map.get(input_name, set())) else '?'
                                      for i, s in enumerate(slice_sizes))
                return [
                    f'{start_ssa} = stablehlo.constant dense<[{start_str}]> : tensor<{ndim}xi64>',
                    f'{result_ssa} = stablehlo.dynamic_slice {input_ssa}, {start_ssa}, '
                    f'slice_sizes = [{sizes_str}] : ({input_type}, tensor<{ndim}xi64>) -> {result_type}'
                ]
            return [f'{result_ssa} = stablehlo.slice {input_ssa}, dim = {dim}, start = {index}, limit = {index + 1} : {input_type} -> {result_type}']

        elif target_name in ('movedim', 'moveaxis'):
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            source = node.args[1] if len(node.args) > 1 else 0
            dest = node.args[2] if len(node.args) > 2 else 0
            # Build permutation
            input_shape = self.shape_map.get(node.args[0].name if hasattr(node.args[0], 'name') else '', ())
            ndim = len(input_shape) if input_shape else 4
            dims = list(range(ndim))
            dims.remove(source)
            dims.insert(dest, source)
            perm_str = ', '.join(str(d) for d in dims)
            return [f'{result_ssa} = stablehlo.transpose {input_ssa}, dims = [{perm_str}] : {input_type} -> {result_type}']

        elif target_name == 'swapaxes':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            axis0 = node.args[1] if len(node.args) > 1 else 0
            axis1 = node.args[2] if len(node.args) > 2 else 1
            input_shape = self.shape_map.get(node.args[0].name if hasattr(node.args[0], 'name') else '', ())
            ndim = len(input_shape) if input_shape else 4
            dims = list(range(ndim))
            dims[axis0], dims[axis1] = dims[axis1], dims[axis0]
            perm_str = ', '.join(str(d) for d in dims)
            return [f'{result_ssa} = stablehlo.transpose {input_ssa}, dims = [{perm_str}] : {input_type} -> {result_type}']

        # === Indexing/Slicing ===
        elif target_name == 'index_select':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            input_node = node.args[0] if hasattr(node.args[0], 'name') else None
            input_name = input_node.name if input_node else ''
            dim = node.args[1] if len(node.args) > 1 else 0
            indices_ssa = get_input(2) if len(node.args) > 2 else '%indices'

            # Use dynamic_gather if input or output has dynamic dimensions
            if self._is_dynamic_shape(node.name) or self._is_dynamic_shape(input_name):
                output_shape = self.shape_map.get(node.name, ())
                slice_sizes = [1 if i == dim else s for i, s in enumerate(output_shape)]
                slice_sizes_str = ', '.join(str(s) if not (i in self.dynamic_dim_map.get(node.name, set())) else '?'
                                            for i, s in enumerate(slice_sizes))
                return [f'{result_ssa} = stablehlo.dynamic_gather {input_ssa}, {indices_ssa}, '
                        f'slice_sizes = [{slice_sizes_str}], dim = {dim} : ({input_type}) -> {result_type}']
            return [f'{result_ssa} = stablehlo.gather {input_ssa}[{indices_ssa}], dim = {dim} : ({input_type}) -> {result_type}']

        elif target_name == 'gather':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            input_node = node.args[0] if hasattr(node.args[0], 'name') else None
            input_name = input_node.name if input_node else ''
            dim = node.args[1] if len(node.args) > 1 else 0
            indices_ssa = get_input(2) if len(node.args) > 2 else '%indices'

            # Use dynamic_gather if input or output has dynamic dimensions
            if self._is_dynamic_shape(node.name) or self._is_dynamic_shape(input_name):
                output_shape = self.shape_map.get(node.name, ())
                slice_sizes = [1 if i == dim else s for i, s in enumerate(output_shape)]
                slice_sizes_str = ', '.join(str(s) if not (i in self.dynamic_dim_map.get(node.name, set())) else '?'
                                            for i, s in enumerate(slice_sizes))
                return [f'{result_ssa} = stablehlo.dynamic_gather {input_ssa}, {indices_ssa}, '
                        f'slice_sizes = [{slice_sizes_str}], dim = {dim} : ({input_type}) -> {result_type}']
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

        elif target_name == 'index_copy':
            # index_copy_(dim, index, source) -> scatter
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else 0
            indices_ssa = get_input(2) if len(node.args) > 2 else '%indices'
            src_ssa = get_input(3) if len(node.args) > 3 else '%src'
            return [f'{result_ssa} = stablehlo.scatter {input_ssa}, {indices_ssa}, {src_ssa}, dim = {dim} : ({input_type}) -> {result_type}']

        elif target_name == 'index_add':
            # index_add_(dim, index, source) -> gather + add + scatter
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else 0
            indices_ssa = get_input(2) if len(node.args) > 2 else '%indices'
            src_ssa = get_input(3) if len(node.args) > 3 else '%src'
            return [f'{result_ssa} = stablehlo.custom_call @index_add({input_ssa}, {indices_ssa}, {src_ssa}) {{dim = {dim}}} : ({input_type}) -> {result_type}']

        elif target_name == 'index_fill':
            # index_fill_(dim, index, value)
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else 0
            indices_ssa = get_input(2) if len(node.args) > 2 else '%indices'
            value = node.args[3] if len(node.args) > 3 else 0.0
            value_ssa = f'{result_ssa}_value'
            return [
                f'{value_ssa} = stablehlo.constant dense<{value}> : {result_type}',
                f'{result_ssa} = stablehlo.scatter {input_ssa}, {indices_ssa}, {value_ssa}, dim = {dim} : ({input_type}) -> {result_type}'
            ]

        # === Padding operations ===
        elif target_name == 'pad':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            input_node = node.args[0] if hasattr(node.args[0], 'name') else None
            input_name = input_node.name if input_node else ''
            pad = node.args[1] if len(node.args) > 1 else []
            value = node.args[2] if len(node.args) > 2 else 0.0
            value_ssa = f'{result_ssa}_value'

            # Use dynamic_pad if input or output has dynamic dimensions
            if self._is_dynamic_shape(node.name) or self._is_dynamic_shape(input_name):
                input_shape = self.shape_map.get(input_name, ())
                ndim = len(input_shape)
                # Build edge_padding_low and edge_padding_high tensors
                low_ssa = f'{result_ssa}_low'
                high_ssa = f'{result_ssa}_high'
                interior_ssa = f'{result_ssa}_interior'
                lines = [
                    f'{value_ssa} = stablehlo.constant dense<{value}> : tensor<f32>',
                    f'{low_ssa} = stablehlo.constant dense<0> : tensor<{ndim}xi64>',
                    f'{high_ssa} = stablehlo.constant dense<0> : tensor<{ndim}xi64>',
                    f'{interior_ssa} = stablehlo.constant dense<0> : tensor<{ndim}xi64>',
                    f'{result_ssa} = stablehlo.dynamic_pad {input_ssa}, {value_ssa}, {low_ssa}, {high_ssa}, {interior_ssa} : '
                    f'({input_type}, tensor<f32>, tensor<{ndim}xi64>, tensor<{ndim}xi64>, tensor<{ndim}xi64>) -> {result_type}'
                ]
                return lines
            # Convert PyTorch pad format (pairs from last dim) to StableHLO format
            return [
                f'{value_ssa} = stablehlo.constant dense<{value}> : tensor<f32>',
                f'{result_ssa} = stablehlo.pad {input_ssa}, {value_ssa}, low = [], high = [], interior = [] : ({input_type}, tensor<f32>) -> {result_type}'
            ]

        elif target_name == 'circular_pad':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            pad = node.args[1] if len(node.args) > 1 else []
            return [f'{result_ssa} = stablehlo.custom_call @circular_pad({input_ssa}) {{pad = {list(pad)}}} : ({input_type}) -> {result_type}']

        # === Convolution variants ===
        elif target_name == 'conv3d':
            input_ssa = get_input(0)
            weight_ssa = get_input(1) if len(node.args) > 1 else '%weight'
            input_type = get_input_type(0)
            weight_type = get_input_type(1) if len(node.args) > 1 else 'tensor<?x?x?x?x?xf32>'
            return [f'{result_ssa} = stablehlo.convolution {input_ssa}, {weight_ssa} : ({input_type}, {weight_type}) -> {result_type}']

        elif target_name == 'conv_transpose1d':
            input_ssa = get_input(0)
            weight_ssa = get_input(1) if len(node.args) > 1 else '%weight'
            input_type = get_input_type(0)
            weight_type = get_input_type(1) if len(node.args) > 1 else 'tensor<?x?x?xf32>'
            return [f'{result_ssa} = stablehlo.convolution {input_ssa}, {weight_ssa}, transpose = true : ({input_type}, {weight_type}) -> {result_type}']

        elif target_name == 'conv_transpose2d':
            input_ssa = get_input(0)
            weight_ssa = get_input(1) if len(node.args) > 1 else '%weight'
            input_type = get_input_type(0)
            weight_type = get_input_type(1) if len(node.args) > 1 else 'tensor<?x?x?x?xf32>'
            return [f'{result_ssa} = stablehlo.convolution {input_ssa}, {weight_ssa}, transpose = true : ({input_type}, {weight_type}) -> {result_type}']

        elif target_name == 'conv_transpose3d':
            input_ssa = get_input(0)
            weight_ssa = get_input(1) if len(node.args) > 1 else '%weight'
            input_type = get_input_type(0)
            weight_type = get_input_type(1) if len(node.args) > 1 else 'tensor<?x?x?x?x?xf32>'
            return [f'{result_ssa} = stablehlo.convolution {input_ssa}, {weight_ssa}, transpose = true : ({input_type}, {weight_type}) -> {result_type}']

        # === Pooling variants ===
        elif target_name == 'max_pool1d':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            kernel_size = node.args[1] if len(node.args) > 1 else 2
            return [f'{result_ssa} = stablehlo.reduce_window {input_ssa}, kernel = [{kernel_size}], reducer = max : {input_type} -> {result_type}']

        elif target_name == 'avg_pool1d':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            kernel_size = node.args[1] if len(node.args) > 1 else 2
            return [f'{result_ssa} = stablehlo.reduce_window {input_ssa}, kernel = [{kernel_size}], reducer = add : {input_type} -> {result_type}']

        elif target_name == 'adaptive_max_pool1d':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            output_size = node.args[1] if len(node.args) > 1 else 1
            return [f'{result_ssa} = stablehlo.custom_call @adaptive_max_pool1d({input_ssa}) {{output_size = {output_size}}} : ({input_type}) -> {result_type}']

        elif target_name == 'adaptive_max_pool2d':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            output_size = node.args[1] if len(node.args) > 1 else (1, 1)
            return [f'{result_ssa} = stablehlo.custom_call @adaptive_max_pool2d({input_ssa}) {{output_size = {list(output_size)}}} : ({input_type}) -> {result_type}']

        # === Normalization variants ===
        elif target_name == 'instance_norm':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @instance_norm({input_ssa}) : ({input_type}) -> {result_type}']

        elif target_name == 'group_norm':
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            num_groups = node.args[1] if len(node.args) > 1 else 1
            return [f'{result_ssa} = stablehlo.custom_call @group_norm({input_ssa}) {{num_groups = {num_groups}}} : ({input_type}) -> {result_type}']

        elif target_name == 'layer_norm':
            # F.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5)
            input_ssa = get_input(0)
            input_type = get_input_type(0)

            # normalized_shape is the second argument
            normalized_shape = node.args[1] if len(node.args) > 1 else []
            if isinstance(normalized_shape, int):
                normalized_shape = [normalized_shape]
            elif hasattr(normalized_shape, '__iter__'):
                normalized_shape = list(normalized_shape)

            # weight and bias are optional (args 2 and 3)
            has_weight = len(node.args) > 2 and node.args[2] is not None
            has_bias = len(node.args) > 3 and node.args[3] is not None
            eps = node.args[4] if len(node.args) > 4 else node.kwargs.get('eps', 1e-5)

            lines = [f'// F.layer_norm (normalized_shape={normalized_shape}, eps={eps})']

            if has_weight and has_bias:
                weight_ssa = get_input(2)
                bias_ssa = get_input(3)
                weight_type = get_input_type(2)
                lines.append(
                    f'{result_ssa} = stablehlo.custom_call @layer_norm({input_ssa}, {weight_ssa}, {bias_ssa}) : '
                    f'({input_type}, {weight_type}, {weight_type}) -> {result_type}'
                )
            elif has_weight:
                weight_ssa = get_input(2)
                weight_type = get_input_type(2)
                lines.append(
                    f'{result_ssa} = stablehlo.custom_call @layer_norm({input_ssa}, {weight_ssa}) : '
                    f'({input_type}, {weight_type}) -> {result_type}'
                )
            else:
                lines.append(
                    f'{result_ssa} = stablehlo.custom_call @layer_norm({input_ssa}) : '
                    f'({input_type}) -> {result_type}'
                )

            return lines

        elif target_name == 'batch_norm':
            # F.batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5)
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            eps = node.kwargs.get('eps', 1e-5)
            momentum = node.kwargs.get('momentum', 0.1)

            lines = [f'// F.batch_norm (eps={eps}, momentum={momentum})']

            # Get parameters if available
            if len(node.args) > 1:
                mean_ssa = get_input(1)
                var_ssa = get_input(2) if len(node.args) > 2 else mean_ssa
                weight_ssa = get_input(3) if len(node.args) > 3 else mean_ssa
                bias_ssa = get_input(4) if len(node.args) > 4 else mean_ssa
                mean_type = get_input_type(1)

                lines.append(
                    f'{result_ssa} = stablehlo.custom_call @batch_norm({input_ssa}, {weight_ssa}, {bias_ssa}, {mean_ssa}, {var_ssa}) : '
                    f'({input_type}, {mean_type}, {mean_type}, {mean_type}, {mean_type}) -> {result_type}'
                )
            else:
                lines.append(
                    f'{result_ssa} = stablehlo.custom_call @batch_norm({input_ssa}) : ({input_type}) -> {result_type}'
                )

            return lines

        elif target_name == 'rms_norm':
            # RMSNorm (used in LLaMA, Gemma, etc.)
            input_ssa = get_input(0)
            input_type = get_input_type(0)

            normalized_shape = node.args[1] if len(node.args) > 1 else []
            if isinstance(normalized_shape, int):
                normalized_shape = [normalized_shape]
            eps = node.kwargs.get('eps', 1e-5)

            has_weight = len(node.args) > 2 and node.args[2] is not None

            lines = [f'// rms_norm (normalized_shape={normalized_shape}, eps={eps})']

            if has_weight:
                weight_ssa = get_input(2)
                weight_type = get_input_type(2)
                lines.append(
                    f'{result_ssa} = stablehlo.custom_call @rms_norm({input_ssa}, {weight_ssa}) : '
                    f'({input_type}, {weight_type}) -> {result_type}'
                )
            else:
                lines.append(
                    f'{result_ssa} = stablehlo.custom_call @rms_norm({input_ssa}) : '
                    f'({input_type}) -> {result_type}'
                )

            return lines

        # === Matrix operations (advanced) ===
        elif target_name == 'einsum':
            # einsum is complex - use custom_call
            equation = node.args[0] if node.args else ''
            operands = node.args[1:] if len(node.args) > 1 else []
            operand_ssas = ', '.join(get_input(i+1) for i in range(len(operands)))
            operand_types = ', '.join(get_input_type(i+1) for i in range(len(operands)))
            return [f'{result_ssa} = stablehlo.custom_call @einsum({operand_ssas}) {{equation = "{equation}"}} : ({operand_types}) -> {result_type}']

        elif target_name == 'tensordot':
            lhs = get_input(0)
            rhs = get_input(1)
            lhs_type = get_input_type(0)
            rhs_type = get_input_type(1)
            dims = node.args[2] if len(node.args) > 2 else 2
            return [f'{result_ssa} = stablehlo.custom_call @tensordot({lhs}, {rhs}) {{dims = {dims}}} : ({lhs_type}, {rhs_type}) -> {result_type}']

        # === BACKWARD / GRADIENT OPERATIONS (Training Support) ===
        elif target_name == 'relu_backward':
            grad_ssa = get_input(0)
            input_ssa = get_input(1)
            zero_ssa = f'{result_ssa}_zero'
            cond_ssa = f'{result_ssa}_cond'
            return [
                f'{zero_ssa} = stablehlo.constant dense<0.0> : {result_type}',
                f'{cond_ssa} = stablehlo.compare GT, {input_ssa}, {zero_ssa} : ({result_type}, {result_type}) -> tensor<*xi1>',
                f'{result_ssa} = stablehlo.select {cond_ssa}, {grad_ssa}, {zero_ssa} : (tensor<*xi1>, {result_type}, {result_type}) -> {result_type}'
            ]

        elif target_name == 'sigmoid_backward':
            grad_ssa = get_input(0)
            output_ssa = get_input(1)
            one_ssa = f'{result_ssa}_one'
            sub_ssa = f'{result_ssa}_sub'
            mul1_ssa = f'{result_ssa}_mul1'
            return [
                f'{one_ssa} = stablehlo.constant dense<1.0> : {result_type}',
                f'{sub_ssa} = stablehlo.subtract {one_ssa}, {output_ssa} : {result_type}',
                f'{mul1_ssa} = stablehlo.multiply {output_ssa}, {sub_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.multiply {grad_ssa}, {mul1_ssa} : {result_type}'
            ]

        elif target_name == 'tanh_backward':
            grad_ssa = get_input(0)
            output_ssa = get_input(1)
            one_ssa = f'{result_ssa}_one'
            sq_ssa = f'{result_ssa}_sq'
            sub_ssa = f'{result_ssa}_sub'
            return [
                f'{one_ssa} = stablehlo.constant dense<1.0> : {result_type}',
                f'{sq_ssa} = stablehlo.multiply {output_ssa}, {output_ssa} : {result_type}',
                f'{sub_ssa} = stablehlo.subtract {one_ssa}, {sq_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.multiply {grad_ssa}, {sub_ssa} : {result_type}'
            ]

        elif target_name in ('gelu_backward', 'softmax_backward', 'log_softmax_backward',
                             'leaky_relu_backward', 'elu_backward', 'selu_backward'):
            grad_ssa = get_input(0)
            input_ssa = get_input(1)
            grad_type = get_input_type(0)
            input_type = get_input_type(1)
            return [f'{result_ssa} = stablehlo.custom_call @{target_name}({grad_ssa}, {input_ssa}) : ({grad_type}, {input_type}) -> {result_type}']

        elif target_name == 'exp_backward':
            grad_ssa = get_input(0)
            output_ssa = get_input(1)
            return [f'{result_ssa} = stablehlo.multiply {grad_ssa}, {output_ssa} : {result_type}']

        elif target_name == 'log_backward':
            grad_ssa = get_input(0)
            input_ssa = get_input(1)
            return [f'{result_ssa} = stablehlo.divide {grad_ssa}, {input_ssa} : {result_type}']

        elif target_name == 'sqrt_backward':
            grad_ssa = get_input(0)
            output_ssa = get_input(1)
            two_ssa = f'{result_ssa}_two'
            denom_ssa = f'{result_ssa}_denom'
            return [
                f'{two_ssa} = stablehlo.constant dense<2.0> : {result_type}',
                f'{denom_ssa} = stablehlo.multiply {two_ssa}, {output_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.divide {grad_ssa}, {denom_ssa} : {result_type}'
            ]

        elif target_name == 'sin_backward':
            grad_ssa = get_input(0)
            input_ssa = get_input(1)
            cos_ssa = f'{result_ssa}_cos'
            return [
                f'{cos_ssa} = stablehlo.cosine {input_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.multiply {grad_ssa}, {cos_ssa} : {result_type}'
            ]

        elif target_name == 'cos_backward':
            grad_ssa = get_input(0)
            input_ssa = get_input(1)
            sin_ssa = f'{result_ssa}_sin'
            neg_ssa = f'{result_ssa}_neg'
            return [
                f'{sin_ssa} = stablehlo.sine {input_ssa} : {result_type}',
                f'{neg_ssa} = stablehlo.negate {sin_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.multiply {grad_ssa}, {neg_ssa} : {result_type}'
            ]

        elif target_name == 'mul_backward':
            grad_ssa = get_input(0)
            other_ssa = get_input(1)
            return [f'{result_ssa} = stablehlo.multiply {grad_ssa}, {other_ssa} : {result_type}']

        elif target_name == 'div_backward':
            grad_ssa = get_input(0)
            denom_ssa = get_input(1)
            return [f'{result_ssa} = stablehlo.divide {grad_ssa}, {denom_ssa} : {result_type}']

        elif target_name == 'neg_backward':
            grad_ssa = get_input(0)
            return [f'{result_ssa} = stablehlo.negate {grad_ssa} : {result_type}']

        elif target_name == 'abs_backward':
            grad_ssa = get_input(0)
            input_ssa = get_input(1)
            sign_ssa = f'{result_ssa}_sign'
            return [
                f'{sign_ssa} = stablehlo.sign {input_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.multiply {grad_ssa}, {sign_ssa} : {result_type}'
            ]

        elif target_name in ('matmul_backward', 'mm_backward', 'bmm_backward', 'linear_backward',
                             'conv2d_backward', 'conv1d_backward', 'conv3d_backward',
                             'max_pool2d_backward', 'avg_pool2d_backward',
                             'batch_norm_backward', 'layer_norm_backward'):
            grad_ssa = get_input(0)
            input_ssa = get_input(1) if len(node.args) > 1 else '%input'
            grad_type = get_input_type(0)
            input_type = get_input_type(1) if len(node.args) > 1 else result_type
            return [f'{result_ssa} = stablehlo.custom_call @{target_name}({grad_ssa}, {input_ssa}) : ({grad_type}, {input_type}) -> {result_type}']

        elif target_name == 'sum_backward':
            grad_ssa = get_input(0)
            grad_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.broadcast_in_dim {grad_ssa}, dims = [] : ({grad_type}) -> {result_type}']

        elif target_name in ('mean_backward', 'max_backward', 'min_backward', 'expand_backward',
                             'gather_backward', 'scatter_backward', 'index_select_backward',
                             'mse_loss_backward', 'cross_entropy_backward', 'nll_loss_backward',
                             'l1_loss_backward', 'smooth_l1_loss_backward', 'binary_cross_entropy_backward',
                             'embedding_backward', 'dropout_backward', 'pow_backward'):
            grad_ssa = get_input(0)
            grad_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @{target_name}({grad_ssa}) : ({grad_type}) -> {result_type}']

        elif target_name in ('reshape_backward', 'squeeze_backward', 'unsqueeze_backward'):
            grad_ssa = get_input(0)
            grad_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.reshape {grad_ssa} : {grad_type} -> {result_type}']

        elif target_name == 'transpose_backward':
            grad_ssa = get_input(0)
            grad_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.transpose {grad_ssa}, dims = [1, 0] : {grad_type} -> {result_type}']

        elif target_name in ('add_backward', 'sub_backward'):
            grad_ssa = get_input(0)
            return [f'{result_ssa} = stablehlo.reshape {grad_ssa} : {get_input_type(0)} -> {result_type}']

        # === QUANTIZATION OPERATIONS ===
        # These map PyTorch quantization ops to StableHLO uniform_quantize/dequantize
        # which can then be lowered to Babylon ONNX quantization ops

        elif target_name == 'quantize_per_tensor':
            # torch.quantize_per_tensor(input, scale, zero_point, dtype)
            # -> stablehlo.uniform_quantize
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            scale = node.args[1] if len(node.args) > 1 else 1.0
            zero_point = node.args[2] if len(node.args) > 2 else 0
            # Determine quantized dtype from node.args[3] or default to int8
            qdtype = 'i8'
            if len(node.args) > 3:
                dtype_arg = str(node.args[3])
                if 'qint8' in dtype_arg or 'int8' in dtype_arg:
                    qdtype = 'i8'
                elif 'quint8' in dtype_arg or 'uint8' in dtype_arg:
                    qdtype = 'ui8'
                elif 'qint4' in dtype_arg or 'int4' in dtype_arg:
                    qdtype = 'i4'
                elif 'quint4' in dtype_arg or 'uint4' in dtype_arg:
                    qdtype = 'ui4'
            # Build quantized tensor type
            input_shape = self.shape_map.get(node.args[0].name if hasattr(node.args[0], 'name') else '', ())
            if input_shape:
                shape_str = 'x'.join(str(d) for d in input_shape)
                quant_type = f'tensor<{shape_str}x!quant.uniform<{qdtype}:f32, {scale}:{zero_point}>>'
            else:
                quant_type = f'tensor<!quant.uniform<{qdtype}:f32, {scale}:{zero_point}>>'
            return [f'{result_ssa} = stablehlo.uniform_quantize {input_ssa} : {input_type} -> {quant_type}']

        elif target_name == 'dequantize':
            # tensor.dequantize() or torch.dequantize(tensor)
            # -> stablehlo.uniform_dequantize
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.uniform_dequantize {input_ssa} : {input_type} -> {result_type}']

        elif target_name == 'fake_quantize_per_tensor_affine':
            # Used in Quantization-Aware Training (QAT)
            # fake_quantize_per_tensor_affine(input, scale, zero_point, quant_min, quant_max)
            # Simulates quantization during training - clamp + round + scale
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            scale = node.args[1] if len(node.args) > 1 else 1.0
            zero_point = node.args[2] if len(node.args) > 2 else 0
            quant_min = node.args[3] if len(node.args) > 3 else -128
            quant_max = node.args[4] if len(node.args) > 4 else 127

            # fake_quant = clamp(round(x / scale) + zero_point, min, max)
            # output = (fake_quant - zero_point) * scale
            scale_ssa = f'{result_ssa}_scale'
            zp_ssa = f'{result_ssa}_zp'
            scaled_ssa = f'{result_ssa}_scaled'
            shifted_ssa = f'{result_ssa}_shifted'
            rounded_ssa = f'{result_ssa}_rounded'
            clamped_ssa = f'{result_ssa}_clamped'
            unshifted_ssa = f'{result_ssa}_unshifted'

            return [
                f'{scale_ssa} = stablehlo.constant dense<{scale}> : {result_type}',
                f'{zp_ssa} = stablehlo.constant dense<{float(zero_point)}> : {result_type}',
                f'{scaled_ssa} = stablehlo.divide {input_ssa}, {scale_ssa} : {result_type}',
                f'{shifted_ssa} = stablehlo.add {scaled_ssa}, {zp_ssa} : {result_type}',
                f'{rounded_ssa} = stablehlo.round_nearest_even {shifted_ssa} : {result_type}',
                f'{clamped_ssa} = stablehlo.clamp dense<{float(quant_min)}>, {rounded_ssa}, dense<{float(quant_max)}> : {result_type}',
                f'{unshifted_ssa} = stablehlo.subtract {clamped_ssa}, {zp_ssa} : {result_type}',
                f'{result_ssa} = stablehlo.multiply {unshifted_ssa}, {scale_ssa} : {result_type}'
            ]

        elif target_name == 'fake_quantize_per_channel_affine':
            # Per-channel fake quantization for weights
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            # Use custom_call for per-channel since it requires broadcasting scales
            scale_ssa = get_input(1) if len(node.args) > 1 else '%scale'
            zp_ssa = get_input(2) if len(node.args) > 2 else '%zero_point'
            axis = node.args[3] if len(node.args) > 3 else 0
            quant_min = node.args[4] if len(node.args) > 4 else -128
            quant_max = node.args[5] if len(node.args) > 5 else 127
            return [f'{result_ssa} = stablehlo.custom_call @fake_quantize_per_channel_affine({input_ssa}, {scale_ssa}, {zp_ssa}) '
                    f'{{axis = {axis}, quant_min = {quant_min}, quant_max = {quant_max}}} : ({input_type}, tensor<f32>, tensor<i32>) -> {result_type}']

        elif target_name in ('quantized_linear', 'linear_relu_dynamic', 'quantized_linear_dynamic'):
            # Quantized linear layer
            # Maps to stablehlo ops that Babylon can lower to ONNX QLinearMatMul
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            weight_ssa = get_input(1) if len(node.args) > 1 else '%weight'
            weight_type = get_input_type(1) if len(node.args) > 1 else 'tensor<?x?xi8>'
            bias_ssa = get_input(2) if len(node.args) > 2 else None

            # For quantized linear, we use custom_call that maps to ONNX QLinearMatMul
            if bias_ssa:
                return [f'{result_ssa} = stablehlo.custom_call @quantized_linear({input_ssa}, {weight_ssa}, {bias_ssa}) : '
                        f'({input_type}, {weight_type}, tensor<f32>) -> {result_type}']
            else:
                return [f'{result_ssa} = stablehlo.custom_call @quantized_linear({input_ssa}, {weight_ssa}) : '
                        f'({input_type}, {weight_type}) -> {result_type}']

        elif target_name in ('quantized_conv2d', 'quantized_conv2d_relu'):
            # Quantized 2D convolution
            # Maps to ONNX QLinearConv via Babylon
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            weight_ssa = get_input(1) if len(node.args) > 1 else '%weight'
            weight_type = get_input_type(1) if len(node.args) > 1 else 'tensor<?x?x?x?xi8>'
            bias_ssa = get_input(2) if len(node.args) > 2 else None
            stride = node.kwargs.get('stride', (1, 1))
            padding = node.kwargs.get('padding', (0, 0))

            stride_str = f'{stride[0]}, {stride[1]}' if isinstance(stride, tuple) else f'{stride}, {stride}'
            pad_str = f'{padding[0]}, {padding[1]}' if isinstance(padding, tuple) else f'{padding}, {padding}'

            if bias_ssa:
                return [f'{result_ssa} = stablehlo.custom_call @quantized_conv2d({input_ssa}, {weight_ssa}, {bias_ssa}) '
                        f'{{stride = [{stride_str}], padding = [{pad_str}]}} : ({input_type}, {weight_type}, tensor<f32>) -> {result_type}']
            else:
                return [f'{result_ssa} = stablehlo.custom_call @quantized_conv2d({input_ssa}, {weight_ssa}) '
                        f'{{stride = [{stride_str}], padding = [{pad_str}]}} : ({input_type}, {weight_type}) -> {result_type}']

        elif target_name == 'quantized_batch_norm':
            # Quantized batch normalization
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @quantized_batch_norm({input_ssa}) : ({input_type}) -> {result_type}']

        elif target_name in ('quantized_add', 'quantized_mul'):
            # Quantized elementwise operations
            lhs = get_input(0)
            rhs = get_input(1) if len(node.args) > 1 else '%rhs'
            lhs_type = get_input_type(0)
            rhs_type = get_input_type(1) if len(node.args) > 1 else lhs_type
            op_name = 'add' if 'add' in target_name else 'multiply'
            return [f'{result_ssa} = stablehlo.custom_call @quantized_{op_name}({lhs}, {rhs}) : ({lhs_type}, {rhs_type}) -> {result_type}']

        elif target_name == 'quantized_relu':
            # Quantized ReLU - maximum with zero
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            zero_ssa = f'{result_ssa}_zero'
            return [
                f'{zero_ssa} = stablehlo.constant dense<0> : {result_type}',
                f'{result_ssa} = stablehlo.maximum {input_ssa}, {zero_ssa} : {result_type}'
            ]

        elif target_name in ('int_repr', 'q_per_tensor_affine'):
            # Get the underlying integer representation of a quantized tensor
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.bitcast_convert {input_ssa} : {input_type} -> {result_type}']

        elif target_name == 'q_scale':
            # Get the scale of a quantized tensor (returns scalar)
            input_ssa = get_input(0)
            return [f'{result_ssa} = stablehlo.custom_call @q_scale({input_ssa}) : (tensor<?xi8>) -> tensor<f32>']

        elif target_name == 'q_zero_point':
            # Get the zero point of a quantized tensor (returns scalar)
            input_ssa = get_input(0)
            return [f'{result_ssa} = stablehlo.custom_call @q_zero_point({input_ssa}) : (tensor<?xi8>) -> tensor<i32>']

        # ===== SPARSE TENSOR OPERATIONS =====
        # Sparse tensors use custom_call with format metadata since StableHLO sparsity is RFC-level.
        # Backends can implement efficient sparse kernels (cuSPARSE, MKL, etc.)

        elif target_name == 'sparse_coo_tensor':
            # torch.sparse_coo_tensor(indices, values, size) -> sparse COO tensor
            # indices: 2D tensor of shape (sparse_dim, nnz)
            # values: 1D tensor of shape (nnz,)
            indices_ssa = get_input(0)
            values_ssa = get_input(1) if len(node.args) > 1 else '%values'
            indices_type = get_input_type(0)
            values_type = get_input_type(1) if len(node.args) > 1 else 'tensor<?xf32>'

            # Extract size from args or kwargs
            size = node.args[2] if len(node.args) > 2 else node.kwargs.get('size', None)
            size_str = str(list(size)) if size else '[]'

            return [f'{result_ssa} = stablehlo.custom_call @sparse_coo_tensor({indices_ssa}, {values_ssa}) '
                    f'{{format = "coo", size = {size_str}}} : ({indices_type}, {values_type}) -> {result_type}']

        elif target_name == 'sparse_csr_tensor':
            # torch.sparse_csr_tensor(crow_indices, col_indices, values, size) -> sparse CSR tensor
            crow_ssa = get_input(0)
            col_ssa = get_input(1) if len(node.args) > 1 else '%col_indices'
            values_ssa = get_input(2) if len(node.args) > 2 else '%values'
            crow_type = get_input_type(0)
            col_type = get_input_type(1) if len(node.args) > 1 else 'tensor<?xi64>'
            values_type = get_input_type(2) if len(node.args) > 2 else 'tensor<?xf32>'

            size = node.args[3] if len(node.args) > 3 else node.kwargs.get('size', None)
            size_str = str(list(size)) if size else '[]'

            return [f'{result_ssa} = stablehlo.custom_call @sparse_csr_tensor({crow_ssa}, {col_ssa}, {values_ssa}) '
                    f'{{format = "csr", size = {size_str}}} : ({crow_type}, {col_type}, {values_type}) -> {result_type}']

        elif target_name == 'sparse_csc_tensor':
            # torch.sparse_csc_tensor(ccol_indices, row_indices, values, size) -> sparse CSC tensor
            ccol_ssa = get_input(0)
            row_ssa = get_input(1) if len(node.args) > 1 else '%row_indices'
            values_ssa = get_input(2) if len(node.args) > 2 else '%values'
            ccol_type = get_input_type(0)
            row_type = get_input_type(1) if len(node.args) > 1 else 'tensor<?xi64>'
            values_type = get_input_type(2) if len(node.args) > 2 else 'tensor<?xf32>'

            size = node.args[3] if len(node.args) > 3 else node.kwargs.get('size', None)
            size_str = str(list(size)) if size else '[]'

            return [f'{result_ssa} = stablehlo.custom_call @sparse_csc_tensor({ccol_ssa}, {row_ssa}, {values_ssa}) '
                    f'{{format = "csc", size = {size_str}}} : ({ccol_type}, {row_type}, {values_type}) -> {result_type}']

        elif target_name == 'sparse_bsr_tensor':
            # torch.sparse_bsr_tensor(crow_indices, col_indices, values, size) -> sparse BSR tensor
            crow_ssa = get_input(0)
            col_ssa = get_input(1) if len(node.args) > 1 else '%col_indices'
            values_ssa = get_input(2) if len(node.args) > 2 else '%values'
            crow_type = get_input_type(0)
            col_type = get_input_type(1) if len(node.args) > 1 else 'tensor<?xi64>'
            values_type = get_input_type(2) if len(node.args) > 2 else 'tensor<?x?x?xf32>'

            size = node.args[3] if len(node.args) > 3 else node.kwargs.get('size', None)
            size_str = str(list(size)) if size else '[]'

            return [f'{result_ssa} = stablehlo.custom_call @sparse_bsr_tensor({crow_ssa}, {col_ssa}, {values_ssa}) '
                    f'{{format = "bsr", size = {size_str}}} : ({crow_type}, {col_type}, {values_type}) -> {result_type}']

        elif target_name in ('to_sparse', 'to_sparse_coo'):
            # x.to_sparse() or torch.to_sparse(x) -> convert dense to sparse COO
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            sparse_dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('sparse_dim', 2)
            return [f'{result_ssa} = stablehlo.custom_call @to_sparse_coo({input_ssa}) '
                    f'{{format = "coo", sparse_dim = {sparse_dim}}} : ({input_type}) -> {result_type}']

        elif target_name == 'to_sparse_csr':
            # x.to_sparse_csr() -> convert dense to sparse CSR
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @to_sparse_csr({input_ssa}) '
                    f'{{format = "csr"}} : ({input_type}) -> {result_type}']

        elif target_name == 'to_sparse_csc':
            # x.to_sparse_csc() -> convert dense to sparse CSC
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @to_sparse_csc({input_ssa}) '
                    f'{{format = "csc"}} : ({input_type}) -> {result_type}']

        elif target_name == 'to_sparse_bsr':
            # x.to_sparse_bsr(blocksize) -> convert dense to sparse BSR
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            blocksize = node.args[1] if len(node.args) > 1 else node.kwargs.get('blocksize', (16, 16))
            bs_str = f'{blocksize[0]}, {blocksize[1]}' if isinstance(blocksize, tuple) else f'{blocksize}, {blocksize}'
            return [f'{result_ssa} = stablehlo.custom_call @to_sparse_bsr({input_ssa}) '
                    f'{{format = "bsr", blocksize = [{bs_str}]}} : ({input_type}) -> {result_type}']

        elif target_name == 'to_dense':
            # sparse.to_dense() -> convert sparse to dense
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @sparse_to_dense({input_ssa}) : ({input_type}) -> {result_type}']

        elif target_name == 'coalesce':
            # sparse.coalesce() -> coalesce duplicate indices in COO tensor
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @sparse_coalesce({input_ssa}) : ({input_type}) -> {result_type}']

        elif target_name == 'is_coalesced':
            # sparse.is_coalesced() -> check if COO tensor is coalesced
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @sparse_is_coalesced({input_ssa}) : ({input_type}) -> tensor<i1>']

        elif target_name in ('sparse_indices', 'indices'):
            # sparse.indices() -> get indices tensor (for COO format)
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @sparse_indices({input_ssa}) '
                    f'{{format = "coo"}} : ({input_type}) -> {result_type}']

        elif target_name in ('sparse_values', 'values'):
            # sparse.values() -> get values tensor
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @sparse_values({input_ssa}) : ({input_type}) -> {result_type}']

        elif target_name in ('crow_indices', 'ccol_indices'):
            # sparse.crow_indices() / sparse.ccol_indices() -> get compressed row/col indices
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            format_type = 'csr' if 'crow' in target_name else 'csc'
            return [f'{result_ssa} = stablehlo.custom_call @sparse_{target_name}({input_ssa}) '
                    f'{{format = "{format_type}"}} : ({input_type}) -> {result_type}']

        elif target_name in ('col_indices', 'row_indices'):
            # sparse.col_indices() / sparse.row_indices() -> get col/row indices for CSR/CSC
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @sparse_{target_name}({input_ssa}) : ({input_type}) -> {result_type}']

        elif target_name == 'sparse_dim':
            # sparse.sparse_dim() -> number of sparse dimensions
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @sparse_dim({input_ssa}) : ({input_type}) -> tensor<i64>']

        elif target_name == 'dense_dim':
            # sparse.dense_dim() -> number of dense dimensions
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @dense_dim({input_ssa}) : ({input_type}) -> tensor<i64>']

        elif target_name == 'nnz':
            # sparse.nnz() or sparse._nnz() -> number of non-zero elements
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @sparse_nnz({input_ssa}) : ({input_type}) -> tensor<i64>']

        elif target_name in ('is_sparse', 'is_sparse_csr'):
            # Check if tensor is sparse
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            format_check = 'csr' if 'csr' in target_name else 'any'
            return [f'{result_ssa} = stablehlo.custom_call @is_sparse({input_ssa}) '
                    f'{{format = "{format_check}"}} : ({input_type}) -> tensor<i1>']

        elif target_name in ('sparse_mm', 'spmm'):
            # Sparse matrix multiplication: sparse @ dense or sparse @ sparse
            lhs = get_input(0)
            rhs = get_input(1) if len(node.args) > 1 else '%rhs'
            lhs_type = get_input_type(0)
            rhs_type = get_input_type(1) if len(node.args) > 1 else result_type
            return [f'{result_ssa} = stablehlo.custom_call @sparse_mm({lhs}, {rhs}) : ({lhs_type}, {rhs_type}) -> {result_type}']

        elif target_name == 'sparse_addmm':
            # Sparse addmm: beta * input + alpha * (sparse @ dense)
            input_ssa = get_input(0)
            sparse_ssa = get_input(1) if len(node.args) > 1 else '%sparse'
            dense_ssa = get_input(2) if len(node.args) > 2 else '%dense'
            input_type = get_input_type(0)
            sparse_type = get_input_type(1) if len(node.args) > 1 else 'tensor<?x?xf32>'
            dense_type = get_input_type(2) if len(node.args) > 2 else 'tensor<?x?xf32>'
            beta = node.kwargs.get('beta', 1.0)
            alpha = node.kwargs.get('alpha', 1.0)
            return [f'{result_ssa} = stablehlo.custom_call @sparse_addmm({input_ssa}, {sparse_ssa}, {dense_ssa}) '
                    f'{{beta = {beta} : f32, alpha = {alpha} : f32}} : ({input_type}, {sparse_type}, {dense_type}) -> {result_type}']

        elif target_name == 'sparse_sampled_addmm':
            # Sparse sampled addmm (used in attention sparsity)
            input_ssa = get_input(0)
            mat1_ssa = get_input(1) if len(node.args) > 1 else '%mat1'
            mat2_ssa = get_input(2) if len(node.args) > 2 else '%mat2'
            input_type = get_input_type(0)
            mat1_type = get_input_type(1) if len(node.args) > 1 else 'tensor<?x?xf32>'
            mat2_type = get_input_type(2) if len(node.args) > 2 else 'tensor<?x?xf32>'
            beta = node.kwargs.get('beta', 1.0)
            alpha = node.kwargs.get('alpha', 1.0)
            return [f'{result_ssa} = stablehlo.custom_call @sparse_sampled_addmm({input_ssa}, {mat1_ssa}, {mat2_ssa}) '
                    f'{{beta = {beta} : f32, alpha = {alpha} : f32}} : ({input_type}, {mat1_type}, {mat2_type}) -> {result_type}']

        elif target_name == 'sparse_sum':
            # Reduction sum on sparse tensor
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', None)
            if dim is not None:
                dim_str = str(list(dim)) if isinstance(dim, (list, tuple)) else f'[{dim}]'
                return [f'{result_ssa} = stablehlo.custom_call @sparse_sum({input_ssa}) '
                        f'{{dim = {dim_str}}} : ({input_type}) -> {result_type}']
            return [f'{result_ssa} = stablehlo.custom_call @sparse_sum({input_ssa}) : ({input_type}) -> {result_type}']

        elif target_name == 'sparse_softmax':
            # Softmax on sparse tensor (only over non-zero elements)
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', -1)
            return [f'{result_ssa} = stablehlo.custom_call @sparse_softmax({input_ssa}) '
                    f'{{dim = {dim}}} : ({input_type}) -> {result_type}']

        elif target_name == 'sparse_log_softmax':
            # Log softmax on sparse tensor
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', -1)
            return [f'{result_ssa} = stablehlo.custom_call @sparse_log_softmax({input_ssa}) '
                    f'{{dim = {dim}}} : ({input_type}) -> {result_type}']

        elif target_name == 'sparse_resize_':
            # Resize sparse tensor (in-place, but we emit as functional)
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            size = node.args[1] if len(node.args) > 1 else node.kwargs.get('size', [])
            sparse_dim = node.args[2] if len(node.args) > 2 else node.kwargs.get('sparse_dim', 2)
            dense_dim = node.args[3] if len(node.args) > 3 else node.kwargs.get('dense_dim', 0)
            return [f'{result_ssa} = stablehlo.custom_call @sparse_resize({input_ssa}) '
                    f'{{size = {list(size)}, sparse_dim = {sparse_dim}, dense_dim = {dense_dim}}} : ({input_type}) -> {result_type}']

        elif target_name == 'sparse_mask':
            # Apply sparse mask to dense tensor
            input_ssa = get_input(0)
            mask_ssa = get_input(1) if len(node.args) > 1 else '%mask'
            input_type = get_input_type(0)
            mask_type = get_input_type(1) if len(node.args) > 1 else 'tensor<?x?xf32>'
            return [f'{result_ssa} = stablehlo.custom_call @sparse_mask({input_ssa}, {mask_ssa}) : ({input_type}, {mask_type}) -> {result_type}']

        elif target_name == 'to_sparse_semi_structured':
            # Convert to semi-structured (2:4) sparsity for NVIDIA Ampere/Hopper
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @to_sparse_semi_structured({input_ssa}) '
                    f'{{format = "semi_structured", pattern = "2:4"}} : ({input_type}) -> {result_type}']

        # ===== FFT OPERATIONS =====
        # StableHLO has native FFT support with fft_type: FFT, IFFT, RFFT, IRFFT

        elif target_name in ('fft', 'fft_fft'):
            # torch.fft.fft(x, n, dim, norm) -> stablehlo.fft FFT
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            n = node.args[1] if len(node.args) > 1 else node.kwargs.get('n', None)
            dim = node.args[2] if len(node.args) > 2 else node.kwargs.get('dim', -1)
            fft_length = n if n else self._get_dim_size(node.args[0], dim)
            return [f'{result_ssa} = stablehlo.fft {input_ssa}, type = FFT, length = [{fft_length}] : {input_type} -> {result_type}']

        elif target_name in ('ifft', 'fft_ifft'):
            # torch.fft.ifft -> stablehlo.fft IFFT
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            n = node.args[1] if len(node.args) > 1 else node.kwargs.get('n', None)
            dim = node.args[2] if len(node.args) > 2 else node.kwargs.get('dim', -1)
            fft_length = n if n else self._get_dim_size(node.args[0], dim)
            return [f'{result_ssa} = stablehlo.fft {input_ssa}, type = IFFT, length = [{fft_length}] : {input_type} -> {result_type}']

        elif target_name in ('rfft', 'fft_rfft'):
            # torch.fft.rfft -> stablehlo.fft RFFT (real-to-complex)
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            n = node.args[1] if len(node.args) > 1 else node.kwargs.get('n', None)
            dim = node.args[2] if len(node.args) > 2 else node.kwargs.get('dim', -1)
            fft_length = n if n else self._get_dim_size(node.args[0], dim)
            return [f'{result_ssa} = stablehlo.fft {input_ssa}, type = RFFT, length = [{fft_length}] : {input_type} -> {result_type}']

        elif target_name in ('irfft', 'fft_irfft'):
            # torch.fft.irfft -> stablehlo.fft IRFFT (complex-to-real)
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            n = node.args[1] if len(node.args) > 1 else node.kwargs.get('n', None)
            dim = node.args[2] if len(node.args) > 2 else node.kwargs.get('dim', -1)
            fft_length = n if n else (self._get_dim_size(node.args[0], dim) - 1) * 2
            return [f'{result_ssa} = stablehlo.fft {input_ssa}, type = IRFFT, length = [{fft_length}] : {input_type} -> {result_type}']

        elif target_name in ('hfft', 'fft_hfft'):
            # torch.fft.hfft (Hermitian FFT) -> stablehlo.fft IRFFT
            # hfft is the inverse of rfft for Hermitian-symmetric inputs
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            n = node.args[1] if len(node.args) > 1 else node.kwargs.get('n', None)
            dim = node.args[2] if len(node.args) > 2 else node.kwargs.get('dim', -1)
            fft_length = n if n else (self._get_dim_size(node.args[0], dim) - 1) * 2
            return [f'{result_ssa} = stablehlo.fft {input_ssa}, type = IRFFT, length = [{fft_length}] : {input_type} -> {result_type}']

        elif target_name in ('ihfft', 'fft_ihfft'):
            # torch.fft.ihfft (inverse Hermitian FFT) -> stablehlo.fft RFFT
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            n = node.args[1] if len(node.args) > 1 else node.kwargs.get('n', None)
            dim = node.args[2] if len(node.args) > 2 else node.kwargs.get('dim', -1)
            fft_length = n if n else self._get_dim_size(node.args[0], dim)
            return [f'{result_ssa} = stablehlo.fft {input_ssa}, type = RFFT, length = [{fft_length}] : {input_type} -> {result_type}']

        elif target_name in ('fft2', 'fft_fft2'):
            # torch.fft.fft2 -> stablehlo.fft FFT with 2D lengths
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            s = node.args[1] if len(node.args) > 1 else node.kwargs.get('s', None)
            dim = node.args[2] if len(node.args) > 2 else node.kwargs.get('dim', (-2, -1))
            if s:
                lengths = ', '.join(str(x) for x in s)
            else:
                lengths = f'{self._get_dim_size(node.args[0], dim[0])}, {self._get_dim_size(node.args[0], dim[1])}'
            return [f'{result_ssa} = stablehlo.fft {input_ssa}, type = FFT, length = [{lengths}] : {input_type} -> {result_type}']

        elif target_name in ('ifft2', 'fft_ifft2'):
            # torch.fft.ifft2 -> stablehlo.fft IFFT with 2D lengths
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            s = node.args[1] if len(node.args) > 1 else node.kwargs.get('s', None)
            dim = node.args[2] if len(node.args) > 2 else node.kwargs.get('dim', (-2, -1))
            if s:
                lengths = ', '.join(str(x) for x in s)
            else:
                lengths = f'{self._get_dim_size(node.args[0], dim[0])}, {self._get_dim_size(node.args[0], dim[1])}'
            return [f'{result_ssa} = stablehlo.fft {input_ssa}, type = IFFT, length = [{lengths}] : {input_type} -> {result_type}']

        elif target_name in ('rfft2', 'fft_rfft2'):
            # torch.fft.rfft2 -> stablehlo.fft RFFT with 2D lengths
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            s = node.args[1] if len(node.args) > 1 else node.kwargs.get('s', None)
            dim = node.args[2] if len(node.args) > 2 else node.kwargs.get('dim', (-2, -1))
            if s:
                lengths = ', '.join(str(x) for x in s)
            else:
                lengths = f'{self._get_dim_size(node.args[0], dim[0])}, {self._get_dim_size(node.args[0], dim[1])}'
            return [f'{result_ssa} = stablehlo.fft {input_ssa}, type = RFFT, length = [{lengths}] : {input_type} -> {result_type}']

        elif target_name in ('irfft2', 'fft_irfft2'):
            # torch.fft.irfft2 -> stablehlo.fft IRFFT with 2D lengths
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            s = node.args[1] if len(node.args) > 1 else node.kwargs.get('s', None)
            dim = node.args[2] if len(node.args) > 2 else node.kwargs.get('dim', (-2, -1))
            if s:
                lengths = ', '.join(str(x) for x in s)
            else:
                d0 = self._get_dim_size(node.args[0], dim[0])
                d1 = (self._get_dim_size(node.args[0], dim[1]) - 1) * 2
                lengths = f'{d0}, {d1}'
            return [f'{result_ssa} = stablehlo.fft {input_ssa}, type = IRFFT, length = [{lengths}] : {input_type} -> {result_type}']

        elif target_name in ('fftn', 'fft_fftn'):
            # torch.fft.fftn -> stablehlo.fft FFT with N-D lengths
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            s = node.args[1] if len(node.args) > 1 else node.kwargs.get('s', None)
            dim = node.args[2] if len(node.args) > 2 else node.kwargs.get('dim', None)
            if s:
                lengths = ', '.join(str(x) for x in s)
            elif dim:
                lengths = ', '.join(str(self._get_dim_size(node.args[0], d)) for d in dim)
            else:
                # Default: all dimensions
                shape = self.shape_map.get(node.args[0].name, ())
                lengths = ', '.join(str(d) for d in shape) if shape else '?'
            return [f'{result_ssa} = stablehlo.fft {input_ssa}, type = FFT, length = [{lengths}] : {input_type} -> {result_type}']

        elif target_name in ('ifftn', 'fft_ifftn'):
            # torch.fft.ifftn -> stablehlo.fft IFFT with N-D lengths
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            s = node.args[1] if len(node.args) > 1 else node.kwargs.get('s', None)
            dim = node.args[2] if len(node.args) > 2 else node.kwargs.get('dim', None)
            if s:
                lengths = ', '.join(str(x) for x in s)
            elif dim:
                lengths = ', '.join(str(self._get_dim_size(node.args[0], d)) for d in dim)
            else:
                shape = self.shape_map.get(node.args[0].name, ())
                lengths = ', '.join(str(d) for d in shape) if shape else '?'
            return [f'{result_ssa} = stablehlo.fft {input_ssa}, type = IFFT, length = [{lengths}] : {input_type} -> {result_type}']

        elif target_name in ('rfftn', 'fft_rfftn'):
            # torch.fft.rfftn -> stablehlo.fft RFFT with N-D lengths
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            s = node.args[1] if len(node.args) > 1 else node.kwargs.get('s', None)
            dim = node.args[2] if len(node.args) > 2 else node.kwargs.get('dim', None)
            if s:
                lengths = ', '.join(str(x) for x in s)
            elif dim:
                lengths = ', '.join(str(self._get_dim_size(node.args[0], d)) for d in dim)
            else:
                shape = self.shape_map.get(node.args[0].name, ())
                lengths = ', '.join(str(d) for d in shape) if shape else '?'
            return [f'{result_ssa} = stablehlo.fft {input_ssa}, type = RFFT, length = [{lengths}] : {input_type} -> {result_type}']

        elif target_name in ('irfftn', 'fft_irfftn'):
            # torch.fft.irfftn -> stablehlo.fft IRFFT with N-D lengths
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            s = node.args[1] if len(node.args) > 1 else node.kwargs.get('s', None)
            dim = node.args[2] if len(node.args) > 2 else node.kwargs.get('dim', None)
            if s:
                lengths = ', '.join(str(x) for x in s)
            elif dim:
                # Last dimension in IRFFT output is (input_dim - 1) * 2
                dim_sizes = []
                for i, d in enumerate(dim):
                    size = self._get_dim_size(node.args[0], d)
                    if i == len(dim) - 1:
                        size = (size - 1) * 2
                    dim_sizes.append(str(size))
                lengths = ', '.join(dim_sizes)
            else:
                shape = self.shape_map.get(node.args[0].name, ())
                if shape:
                    dim_sizes = list(shape[:-1]) + [(shape[-1] - 1) * 2]
                    lengths = ', '.join(str(d) for d in dim_sizes)
                else:
                    lengths = '?'
            return [f'{result_ssa} = stablehlo.fft {input_ssa}, type = IRFFT, length = [{lengths}] : {input_type} -> {result_type}']

        elif target_name in ('fftshift', 'fft_fftshift'):
            # torch.fft.fftshift -> roll by n//2 for each FFT dimension
            # This is a helper function, implemented via stablehlo operations
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', None)
            # fftshift is a circular shift (roll) by half the dimension size
            return [f'{result_ssa} = stablehlo.custom_call @fftshift({input_ssa}) '
                    f'{{dim = {dim if dim else "all"}}} : ({input_type}) -> {result_type}']

        elif target_name in ('ifftshift', 'fft_ifftshift'):
            # torch.fft.ifftshift -> inverse of fftshift (roll by -n//2)
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', None)
            return [f'{result_ssa} = stablehlo.custom_call @ifftshift({input_ssa}) '
                    f'{{dim = {dim if dim else "all"}}} : ({input_type}) -> {result_type}']

        elif target_name in ('fftfreq', 'fft_fftfreq'):
            # torch.fft.fftfreq -> generate frequency bins
            # This returns a 1D tensor of frequencies, not a transform
            n = node.args[0] if node.args else node.kwargs.get('n', 1)
            d = node.args[1] if len(node.args) > 1 else node.kwargs.get('d', 1.0)
            return [f'{result_ssa} = stablehlo.custom_call @fftfreq() '
                    f'{{n = {n}, d = {d}}} : () -> {result_type}']

        elif target_name in ('rfftfreq', 'fft_rfftfreq'):
            # torch.fft.rfftfreq -> frequency bins for real FFT (non-negative only)
            n = node.args[0] if node.args else node.kwargs.get('n', 1)
            d = node.args[1] if len(node.args) > 1 else node.kwargs.get('d', 1.0)
            return [f'{result_ssa} = stablehlo.custom_call @rfftfreq() '
                    f'{{n = {n}, d = {d}}} : () -> {result_type}']

        # ===== DROPOUT OPERATIONS =====
        elif target_name in ('dropout', 'dropout1d', 'dropout2d', 'dropout3d'):
            # nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            p = node.args[1] if len(node.args) > 1 else node.kwargs.get('p', 0.5)
            training = node.kwargs.get('training', True)
            inplace = node.kwargs.get('inplace', False)
            attrs = f'p = {p}, training = {str(training).lower()}, inplace = {str(inplace).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @{target_name}({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('alpha_dropout',):
            # nn.AlphaDropout - maintains self-normalizing property for SELU
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            p = node.args[1] if len(node.args) > 1 else node.kwargs.get('p', 0.5)
            training = node.kwargs.get('training', True)
            inplace = node.kwargs.get('inplace', False)
            attrs = f'p = {p}, training = {str(training).lower()}, inplace = {str(inplace).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @alpha_dropout({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('feature_alpha_dropout',):
            # F.feature_alpha_dropout - channel-wise alpha dropout
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            p = node.args[1] if len(node.args) > 1 else node.kwargs.get('p', 0.5)
            training = node.kwargs.get('training', True)
            inplace = node.kwargs.get('inplace', False)
            attrs = f'p = {p}, training = {str(training).lower()}, inplace = {str(inplace).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @feature_alpha_dropout({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        # ===== UPSAMPLING OPERATIONS =====
        elif target_name in ('upsample', 'upsample_nearest', 'upsample_nearest1d', 'upsample_nearest2d', 'upsample_nearest3d'):
            # nn.Upsample with mode='nearest' or nn.UpsamplingNearest2d
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            size = node.kwargs.get('size', None)
            scale_factor = node.kwargs.get('scale_factor', None)
            size_str = str(list(size)) if size else 'null'
            scale_str = str(list(scale_factor)) if isinstance(scale_factor, (list, tuple)) else str(scale_factor) if scale_factor else 'null'
            attrs = f'size = {size_str}, scale_factor = {scale_str}, mode = "nearest"'
            return [f'{result_ssa} = stablehlo.custom_call @upsample_nearest({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('upsample_bilinear', 'upsample_bilinear2d'):
            # nn.Upsample with mode='bilinear' or nn.UpsamplingBilinear2d
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            size = node.kwargs.get('size', None)
            scale_factor = node.kwargs.get('scale_factor', None)
            align_corners = node.kwargs.get('align_corners', False)
            size_str = str(list(size)) if size else 'null'
            scale_str = str(list(scale_factor)) if isinstance(scale_factor, (list, tuple)) else str(scale_factor) if scale_factor else 'null'
            attrs = f'size = {size_str}, scale_factor = {scale_str}, mode = "bilinear", align_corners = {str(align_corners).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @upsample_bilinear({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('upsample_bicubic', 'upsample_bicubic2d'):
            # nn.Upsample with mode='bicubic'
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            size = node.kwargs.get('size', None)
            scale_factor = node.kwargs.get('scale_factor', None)
            align_corners = node.kwargs.get('align_corners', False)
            size_str = str(list(size)) if size else 'null'
            scale_str = str(list(scale_factor)) if isinstance(scale_factor, (list, tuple)) else str(scale_factor) if scale_factor else 'null'
            attrs = f'size = {size_str}, scale_factor = {scale_str}, mode = "bicubic", align_corners = {str(align_corners).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @upsample_bicubic({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('upsample_trilinear', 'upsample_trilinear3d'):
            # nn.Upsample with mode='trilinear'
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            size = node.kwargs.get('size', None)
            scale_factor = node.kwargs.get('scale_factor', None)
            align_corners = node.kwargs.get('align_corners', False)
            size_str = str(list(size)) if size else 'null'
            scale_str = str(list(scale_factor)) if isinstance(scale_factor, (list, tuple)) else str(scale_factor) if scale_factor else 'null'
            attrs = f'size = {size_str}, scale_factor = {scale_str}, mode = "trilinear", align_corners = {str(align_corners).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @upsample_trilinear({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('interpolate',):
            # F.interpolate - general interpolation function
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            size = node.kwargs.get('size', None)
            scale_factor = node.kwargs.get('scale_factor', None)
            mode = node.kwargs.get('mode', 'nearest')
            align_corners = node.kwargs.get('align_corners', None)
            recompute_scale_factor = node.kwargs.get('recompute_scale_factor', None)
            antialias = node.kwargs.get('antialias', False)
            size_str = str(list(size)) if size else 'null'
            scale_str = str(list(scale_factor)) if isinstance(scale_factor, (list, tuple)) else str(scale_factor) if scale_factor else 'null'
            align_str = str(align_corners).lower() if align_corners is not None else 'null'
            attrs = f'size = {size_str}, scale_factor = {scale_str}, mode = "{mode}", align_corners = {align_str}, antialias = {str(antialias).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @interpolate({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('pixel_shuffle',):
            # nn.PixelShuffle / F.pixel_shuffle - rearranges from (C*r^2, H, W) to (C, H*r, W*r)
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            upscale_factor = node.args[1] if len(node.args) > 1 else node.kwargs.get('upscale_factor', 2)
            attrs = f'upscale_factor = {upscale_factor}'
            return [f'{result_ssa} = stablehlo.custom_call @pixel_shuffle({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('pixel_unshuffle',):
            # nn.PixelUnshuffle / F.pixel_unshuffle - inverse of pixel_shuffle
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            downscale_factor = node.args[1] if len(node.args) > 1 else node.kwargs.get('downscale_factor', 2)
            attrs = f'downscale_factor = {downscale_factor}'
            return [f'{result_ssa} = stablehlo.custom_call @pixel_unshuffle({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        # ===== GRID SAMPLING OPERATIONS =====
        elif target_name in ('grid_sample',):
            # F.grid_sample - sample input using grid coordinates
            input_ssa = get_input(0)
            grid_ssa = get_input(1)
            input_type = get_input_type(0)
            grid_type = get_input_type(1)
            mode = node.kwargs.get('mode', 'bilinear')
            padding_mode = node.kwargs.get('padding_mode', 'zeros')
            align_corners = node.kwargs.get('align_corners', False)
            attrs = f'mode = "{mode}", padding_mode = "{padding_mode}", align_corners = {str(align_corners).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @grid_sample({input_ssa}, {grid_ssa}) '
                    f'{{{attrs}}} : ({input_type}, {grid_type}) -> {result_type}']

        elif target_name in ('affine_grid',):
            # F.affine_grid - generates 2D or 3D flow field (sampling grid)
            theta_ssa = get_input(0)
            theta_type = get_input_type(0)
            size = node.args[1] if len(node.args) > 1 else node.kwargs.get('size')
            align_corners = node.kwargs.get('align_corners', False)
            size_str = str(list(size)) if size else 'null'
            attrs = f'size = {size_str}, align_corners = {str(align_corners).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @affine_grid({theta_ssa}) '
                    f'{{{attrs}}} : ({theta_type}) -> {result_type}']

        # ===== IMAGE OPERATIONS =====
        elif target_name in ('rgb_to_grayscale',):
            # torchvision.transforms.functional.rgb_to_grayscale
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            num_output_channels = node.kwargs.get('num_output_channels', 1)
            attrs = f'num_output_channels = {num_output_channels}'
            return [f'{result_ssa} = stablehlo.custom_call @rgb_to_grayscale({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('adjust_brightness',):
            # torchvision.transforms.functional.adjust_brightness
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            brightness_factor = node.args[1] if len(node.args) > 1 else node.kwargs.get('brightness_factor', 1.0)
            attrs = f'brightness_factor = {brightness_factor}'
            return [f'{result_ssa} = stablehlo.custom_call @adjust_brightness({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('adjust_contrast',):
            # torchvision.transforms.functional.adjust_contrast
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            contrast_factor = node.args[1] if len(node.args) > 1 else node.kwargs.get('contrast_factor', 1.0)
            attrs = f'contrast_factor = {contrast_factor}'
            return [f'{result_ssa} = stablehlo.custom_call @adjust_contrast({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('adjust_saturation',):
            # torchvision.transforms.functional.adjust_saturation
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            saturation_factor = node.args[1] if len(node.args) > 1 else node.kwargs.get('saturation_factor', 1.0)
            attrs = f'saturation_factor = {saturation_factor}'
            return [f'{result_ssa} = stablehlo.custom_call @adjust_saturation({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('adjust_hue',):
            # torchvision.transforms.functional.adjust_hue
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            hue_factor = node.args[1] if len(node.args) > 1 else node.kwargs.get('hue_factor', 0.0)
            attrs = f'hue_factor = {hue_factor}'
            return [f'{result_ssa} = stablehlo.custom_call @adjust_hue({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('rotate',):
            # torchvision.transforms.functional.rotate
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            angle = node.args[1] if len(node.args) > 1 else node.kwargs.get('angle', 0.0)
            interpolation = node.kwargs.get('interpolation', 'nearest')
            expand = node.kwargs.get('expand', False)
            fill = node.kwargs.get('fill', 0)
            attrs = f'angle = {angle}, interpolation = "{interpolation}", expand = {str(expand).lower()}, fill = {fill}'
            return [f'{result_ssa} = stablehlo.custom_call @rotate({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('hflip', 'horizontal_flip'):
            # torchvision.transforms.functional.hflip
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.reverse {input_ssa}, dims = [-1] : {input_type} -> {result_type}']

        elif target_name in ('vflip', 'vertical_flip'):
            # torchvision.transforms.functional.vflip
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.reverse {input_ssa}, dims = [-2] : {input_type} -> {result_type}']

        elif target_name in ('center_crop',):
            # torchvision.transforms.functional.center_crop
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            output_size = node.args[1] if len(node.args) > 1 else node.kwargs.get('output_size')
            size_str = str(list(output_size)) if isinstance(output_size, (list, tuple)) else str([output_size, output_size])
            attrs = f'output_size = {size_str}'
            return [f'{result_ssa} = stablehlo.custom_call @center_crop({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('resize',):
            # torchvision.transforms.functional.resize
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            size = node.args[1] if len(node.args) > 1 else node.kwargs.get('size')
            interpolation = node.kwargs.get('interpolation', 'bilinear')
            antialias = node.kwargs.get('antialias', True)
            size_str = str(list(size)) if isinstance(size, (list, tuple)) else str([size])
            attrs = f'size = {size_str}, interpolation = "{interpolation}", antialias = {str(antialias).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @resize({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('normalize',):
            # torchvision.transforms.functional.normalize (image normalization)
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            mean = node.args[1] if len(node.args) > 1 else node.kwargs.get('mean')
            std = node.args[2] if len(node.args) > 2 else node.kwargs.get('std')
            mean_str = str(list(mean)) if mean else '[0.485, 0.456, 0.406]'
            std_str = str(list(std)) if std else '[0.229, 0.224, 0.225]'
            attrs = f'mean = {mean_str}, std = {std_str}'
            return [f'{result_ssa} = stablehlo.custom_call @normalize({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        # ===== SORTING OPERATIONS =====
        elif target_name in ('sort',):
            # torch.sort -> returns (sorted_values, sorted_indices)
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', -1)
            descending = node.kwargs.get('descending', False)
            stable = node.kwargs.get('stable', False)
            attrs = f'dim = {dim}, descending = {str(descending).lower()}, stable = {str(stable).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @sort({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('argsort',):
            # torch.argsort -> returns indices that would sort the tensor
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', -1)
            descending = node.kwargs.get('descending', False)
            stable = node.kwargs.get('stable', False)
            attrs = f'dim = {dim}, descending = {str(descending).lower()}, stable = {str(stable).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @argsort({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('topk',):
            # torch.topk -> returns (values, indices) of k largest elements
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            k = node.args[1] if len(node.args) > 1 else node.kwargs.get('k')
            dim = node.args[2] if len(node.args) > 2 else node.kwargs.get('dim', -1)
            largest = node.kwargs.get('largest', True)
            sorted_flag = node.kwargs.get('sorted', True)
            attrs = f'k = {k}, dim = {dim}, largest = {str(largest).lower()}, sorted = {str(sorted_flag).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @topk({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('kthvalue',):
            # torch.kthvalue -> returns (value, index) of k-th smallest element
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            k = node.args[1] if len(node.args) > 1 else node.kwargs.get('k')
            dim = node.args[2] if len(node.args) > 2 else node.kwargs.get('dim', -1)
            keepdim = node.kwargs.get('keepdim', False)
            attrs = f'k = {k}, dim = {dim}, keepdim = {str(keepdim).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @kthvalue({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('msort',):
            # torch.msort -> sort along first dimension
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @sort({input_ssa}) '
                    f'{{dim = 0, descending = false, stable = false}} : ({input_type}) -> {result_type}']

        elif target_name in ('searchsorted',):
            # torch.searchsorted -> find indices where elements should be inserted
            sorted_sequence_ssa = get_input(0)
            values_ssa = get_input(1)
            sorted_type = get_input_type(0)
            values_type = get_input_type(1)
            out_int32 = node.kwargs.get('out_int32', False)
            right = node.kwargs.get('right', False)
            side = node.kwargs.get('side', 'left')
            attrs = f'out_int32 = {str(out_int32).lower()}, right = {str(right).lower()}, side = "{side}"'
            return [f'{result_ssa} = stablehlo.custom_call @searchsorted({sorted_sequence_ssa}, {values_ssa}) '
                    f'{{{attrs}}} : ({sorted_type}, {values_type}) -> {result_type}']

        # ===== UNIQUE/SET OPERATIONS =====
        elif target_name in ('unique',):
            # torch.unique -> returns unique elements
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            sorted_flag = node.kwargs.get('sorted', True)
            return_inverse = node.kwargs.get('return_inverse', False)
            return_counts = node.kwargs.get('return_counts', False)
            dim = node.kwargs.get('dim', None)
            dim_str = str(dim) if dim is not None else 'null'
            attrs = f'sorted = {str(sorted_flag).lower()}, return_inverse = {str(return_inverse).lower()}, return_counts = {str(return_counts).lower()}, dim = {dim_str}'
            return [f'{result_ssa} = stablehlo.custom_call @unique({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('unique_consecutive',):
            # torch.unique_consecutive -> unique consecutive elements
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return_inverse = node.kwargs.get('return_inverse', False)
            return_counts = node.kwargs.get('return_counts', False)
            dim = node.kwargs.get('dim', None)
            dim_str = str(dim) if dim is not None else 'null'
            attrs = f'return_inverse = {str(return_inverse).lower()}, return_counts = {str(return_counts).lower()}, dim = {dim_str}'
            return [f'{result_ssa} = stablehlo.custom_call @unique_consecutive({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('bincount',):
            # torch.bincount -> count occurrences of each value
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            weights_ssa = get_input(1) if len(node.args) > 1 else None
            minlength = node.kwargs.get('minlength', 0)
            if weights_ssa:
                weights_type = get_input_type(1)
                return [f'{result_ssa} = stablehlo.custom_call @bincount({input_ssa}, {weights_ssa}) '
                        f'{{minlength = {minlength}}} : ({input_type}, {weights_type}) -> {result_type}']
            return [f'{result_ssa} = stablehlo.custom_call @bincount({input_ssa}) '
                    f'{{minlength = {minlength}}} : ({input_type}) -> {result_type}']

        elif target_name in ('histc',):
            # torch.histc -> histogram of tensor
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            bins = node.args[1] if len(node.args) > 1 else node.kwargs.get('bins', 100)
            min_val = node.args[2] if len(node.args) > 2 else node.kwargs.get('min', 0)
            max_val = node.args[3] if len(node.args) > 3 else node.kwargs.get('max', 0)
            attrs = f'bins = {bins}, min = {min_val}, max = {max_val}'
            return [f'{result_ssa} = stablehlo.custom_call @histc({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('histogram',):
            # torch.histogram -> compute histogram with bin edges
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            bins = node.args[1] if len(node.args) > 1 else node.kwargs.get('bins', 100)
            density = node.kwargs.get('density', False)
            attrs = f'bins = {bins}, density = {str(density).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @histogram({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        # ===== LINEAR ALGEBRA OPERATIONS =====
        elif target_name in ('svd', 'linalg_svd'):
            # torch.linalg.svd -> singular value decomposition
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            full_matrices = node.kwargs.get('full_matrices', True)
            attrs = f'full_matrices = {str(full_matrices).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @svd({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('eig', 'linalg_eig'):
            # torch.linalg.eig -> eigenvalue decomposition
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @eig({input_ssa}) : ({input_type}) -> {result_type}']

        elif target_name in ('eigh', 'linalg_eigh'):
            # torch.linalg.eigh -> eigenvalue decomposition for symmetric/hermitian matrices
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            UPLO = node.kwargs.get('UPLO', 'L')
            attrs = f'UPLO = "{UPLO}"'
            return [f'{result_ssa} = stablehlo.custom_call @eigh({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('qr', 'linalg_qr'):
            # torch.linalg.qr -> QR decomposition
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            mode = node.kwargs.get('mode', 'reduced')
            attrs = f'mode = "{mode}"'
            return [f'{result_ssa} = stablehlo.custom_call @qr({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('cholesky', 'linalg_cholesky'):
            # torch.linalg.cholesky -> Cholesky decomposition
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            upper = node.kwargs.get('upper', False)
            attrs = f'upper = {str(upper).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @cholesky({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('lu', 'linalg_lu'):
            # torch.linalg.lu -> LU decomposition
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            pivot = node.kwargs.get('pivot', True)
            attrs = f'pivot = {str(pivot).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @lu({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('lu_factor', 'linalg_lu_factor'):
            # torch.linalg.lu_factor -> LU factorization
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            pivot = node.kwargs.get('pivot', True)
            attrs = f'pivot = {str(pivot).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @lu_factor({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('inv', 'linalg_inv', 'inverse'):
            # torch.linalg.inv -> matrix inverse
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @inv({input_ssa}) : ({input_type}) -> {result_type}']

        elif target_name in ('pinv', 'linalg_pinv'):
            # torch.linalg.pinv -> pseudo-inverse
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            rcond = node.kwargs.get('rcond', 1e-15)
            hermitian = node.kwargs.get('hermitian', False)
            attrs = f'rcond = {rcond}, hermitian = {str(hermitian).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @pinv({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('det', 'linalg_det'):
            # torch.linalg.det -> matrix determinant
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @det({input_ssa}) : ({input_type}) -> {result_type}']

        elif target_name in ('slogdet', 'linalg_slogdet'):
            # torch.linalg.slogdet -> sign and log of absolute determinant
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @slogdet({input_ssa}) : ({input_type}) -> {result_type}']

        elif target_name in ('matrix_rank', 'linalg_matrix_rank'):
            # torch.linalg.matrix_rank -> rank of matrix
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            atol = node.kwargs.get('atol', None)
            rtol = node.kwargs.get('rtol', None)
            hermitian = node.kwargs.get('hermitian', False)
            atol_str = str(atol) if atol is not None else 'null'
            rtol_str = str(rtol) if rtol is not None else 'null'
            attrs = f'atol = {atol_str}, rtol = {rtol_str}, hermitian = {str(hermitian).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @matrix_rank({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('norm', 'linalg_norm', 'matrix_norm', 'vector_norm'):
            # torch.linalg.norm -> matrix/vector norm
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            ord_val = node.args[1] if len(node.args) > 1 else node.kwargs.get('ord', None)
            dim = node.kwargs.get('dim', None)
            keepdim = node.kwargs.get('keepdim', False)
            ord_str = str(ord_val) if ord_val is not None else '"fro"'
            dim_str = str(dim) if dim is not None else 'null'
            attrs = f'ord = {ord_str}, dim = {dim_str}, keepdim = {str(keepdim).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @norm({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('solve', 'linalg_solve'):
            # torch.linalg.solve -> solve linear system Ax = B
            a_ssa = get_input(0)
            b_ssa = get_input(1)
            a_type = get_input_type(0)
            b_type = get_input_type(1)
            left = node.kwargs.get('left', True)
            attrs = f'left = {str(left).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @solve({a_ssa}, {b_ssa}) '
                    f'{{{attrs}}} : ({a_type}, {b_type}) -> {result_type}']

        elif target_name in ('lstsq', 'linalg_lstsq'):
            # torch.linalg.lstsq -> least squares solution
            a_ssa = get_input(0)
            b_ssa = get_input(1)
            a_type = get_input_type(0)
            b_type = get_input_type(1)
            driver = node.kwargs.get('driver', None)
            driver_str = f'"{driver}"' if driver else 'null'
            attrs = f'driver = {driver_str}'
            return [f'{result_ssa} = stablehlo.custom_call @lstsq({a_ssa}, {b_ssa}) '
                    f'{{{attrs}}} : ({a_type}, {b_type}) -> {result_type}']

        elif target_name in ('cross', 'linalg_cross'):
            # torch.linalg.cross -> cross product
            a_ssa = get_input(0)
            b_ssa = get_input(1)
            a_type = get_input_type(0)
            b_type = get_input_type(1)
            dim = node.kwargs.get('dim', -1)
            attrs = f'dim = {dim}'
            return [f'{result_ssa} = stablehlo.custom_call @cross({a_ssa}, {b_ssa}) '
                    f'{{{attrs}}} : ({a_type}, {b_type}) -> {result_type}']

        elif target_name in ('trace',):
            # torch.trace -> sum of diagonal elements
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @trace({input_ssa}) : ({input_type}) -> {result_type}']

        elif target_name in ('diagonal', 'diag'):
            # torch.diagonal / torch.diag
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            offset = node.args[1] if len(node.args) > 1 else node.kwargs.get('offset', 0)
            dim1 = node.kwargs.get('dim1', 0)
            dim2 = node.kwargs.get('dim2', 1)
            attrs = f'offset = {offset}, dim1 = {dim1}, dim2 = {dim2}'
            return [f'{result_ssa} = stablehlo.custom_call @diagonal({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('diag_embed',):
            # torch.diag_embed -> embed diagonal into matrix
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            offset = node.kwargs.get('offset', 0)
            dim1 = node.kwargs.get('dim1', -2)
            dim2 = node.kwargs.get('dim2', -1)
            attrs = f'offset = {offset}, dim1 = {dim1}, dim2 = {dim2}'
            return [f'{result_ssa} = stablehlo.custom_call @diag_embed({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('tril',):
            # torch.tril -> lower triangular part
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            diagonal = node.args[1] if len(node.args) > 1 else node.kwargs.get('diagonal', 0)
            attrs = f'diagonal = {diagonal}'
            return [f'{result_ssa} = stablehlo.custom_call @tril({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('triu',):
            # torch.triu -> upper triangular part
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            diagonal = node.args[1] if len(node.args) > 1 else node.kwargs.get('diagonal', 0)
            attrs = f'diagonal = {diagonal}'
            return [f'{result_ssa} = stablehlo.custom_call @triu({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        # ===== TENSOR CREATION OPERATIONS =====
        elif target_name in ('zeros',):
            # torch.zeros -> create tensor of zeros
            size = node.args[0] if node.args else node.kwargs.get('size')
            size_str = str(list(size)) if isinstance(size, (list, tuple)) else str([size])
            dtype = node.kwargs.get('dtype', 'f32')
            return [f'{result_ssa} = stablehlo.constant dense<0.0> : {result_type}']

        elif target_name in ('zeros_like',):
            # torch.zeros_like -> create tensor of zeros with same shape as input
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.constant dense<0.0> : {result_type}']

        elif target_name in ('ones',):
            # torch.ones -> create tensor of ones
            size = node.args[0] if node.args else node.kwargs.get('size')
            return [f'{result_ssa} = stablehlo.constant dense<1.0> : {result_type}']

        elif target_name in ('ones_like',):
            # torch.ones_like -> create tensor of ones with same shape as input
            input_ssa = get_input(0)
            return [f'{result_ssa} = stablehlo.constant dense<1.0> : {result_type}']

        elif target_name in ('full',):
            # torch.full -> create tensor filled with specific value
            size = node.args[0] if node.args else node.kwargs.get('size')
            fill_value = node.args[1] if len(node.args) > 1 else node.kwargs.get('fill_value', 0)
            return [f'{result_ssa} = stablehlo.constant dense<{fill_value}> : {result_type}']

        elif target_name in ('full_like',):
            # torch.full_like -> create tensor filled with value, same shape as input
            input_ssa = get_input(0)
            fill_value = node.args[1] if len(node.args) > 1 else node.kwargs.get('fill_value', 0)
            return [f'{result_ssa} = stablehlo.constant dense<{fill_value}> : {result_type}']

        elif target_name in ('empty', 'empty_like'):
            # torch.empty -> uninitialized tensor (treated as zeros in StableHLO)
            return [f'{result_ssa} = stablehlo.constant dense<0.0> : {result_type}']

        elif target_name in ('arange',):
            # torch.arange -> 1D tensor with evenly spaced values
            start = node.args[0] if node.args else node.kwargs.get('start', 0)
            end = node.args[1] if len(node.args) > 1 else node.kwargs.get('end', start)
            step = node.args[2] if len(node.args) > 2 else node.kwargs.get('step', 1)
            if len(node.args) == 1:
                # arange(end) -> arange(0, end, 1)
                end = start
                start = 0
            attrs = f'start = {start}, end = {end}, step = {step}'
            return [f'{result_ssa} = stablehlo.custom_call @arange() {{{attrs}}} : () -> {result_type}']

        elif target_name in ('linspace',):
            # torch.linspace -> 1D tensor with linearly spaced values
            start = node.args[0] if node.args else node.kwargs.get('start', 0)
            end = node.args[1] if len(node.args) > 1 else node.kwargs.get('end', 1)
            steps = node.args[2] if len(node.args) > 2 else node.kwargs.get('steps', 100)
            attrs = f'start = {start}, end = {end}, steps = {steps}'
            return [f'{result_ssa} = stablehlo.custom_call @linspace() {{{attrs}}} : () -> {result_type}']

        elif target_name in ('logspace',):
            # torch.logspace -> 1D tensor with logarithmically spaced values
            start = node.args[0] if node.args else node.kwargs.get('start', 0)
            end = node.args[1] if len(node.args) > 1 else node.kwargs.get('end', 1)
            steps = node.args[2] if len(node.args) > 2 else node.kwargs.get('steps', 100)
            base = node.kwargs.get('base', 10.0)
            attrs = f'start = {start}, end = {end}, steps = {steps}, base = {base}'
            return [f'{result_ssa} = stablehlo.custom_call @logspace() {{{attrs}}} : () -> {result_type}']

        elif target_name in ('eye',):
            # torch.eye -> 2D identity matrix
            n = node.args[0] if node.args else node.kwargs.get('n')
            m = node.args[1] if len(node.args) > 1 else node.kwargs.get('m', n)
            attrs = f'n = {n}, m = {m}'
            return [f'{result_ssa} = stablehlo.custom_call @eye() {{{attrs}}} : () -> {result_type}']

        elif target_name in ('iota',):
            # stablehlo.iota -> generate indices along a dimension
            iota_dim = node.kwargs.get('iota_dimension', 0)
            return [f'{result_ssa} = stablehlo.iota dim = {iota_dim} : {result_type}']

        # ===== RANDOM OPERATIONS =====
        elif target_name in ('rand',):
            # torch.rand -> uniform random [0, 1)
            size = node.args[0] if node.args else node.kwargs.get('size')
            size_str = str(list(size)) if isinstance(size, (list, tuple)) else str([size])
            return [f'{result_ssa} = stablehlo.custom_call @rand() {{size = {size_str}}} : () -> {result_type}']

        elif target_name in ('rand_like',):
            # torch.rand_like -> uniform random with same shape as input
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @rand_like({input_ssa}) : ({input_type}) -> {result_type}']

        elif target_name in ('randn',):
            # torch.randn -> standard normal random
            size = node.args[0] if node.args else node.kwargs.get('size')
            size_str = str(list(size)) if isinstance(size, (list, tuple)) else str([size])
            return [f'{result_ssa} = stablehlo.custom_call @randn() {{size = {size_str}}} : () -> {result_type}']

        elif target_name in ('randn_like',):
            # torch.randn_like -> standard normal with same shape as input
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @randn_like({input_ssa}) : ({input_type}) -> {result_type}']

        elif target_name in ('randint',):
            # torch.randint -> random integers in [low, high)
            low = node.args[0] if node.args else node.kwargs.get('low', 0)
            high = node.args[1] if len(node.args) > 1 else node.kwargs.get('high')
            size = node.args[2] if len(node.args) > 2 else node.kwargs.get('size')
            if len(node.args) == 2 and isinstance(node.args[1], (list, tuple)):
                # randint(high, size) form
                high = low
                low = 0
                size = node.args[1]
            size_str = str(list(size)) if isinstance(size, (list, tuple)) else str([size])
            attrs = f'low = {low}, high = {high}, size = {size_str}'
            return [f'{result_ssa} = stablehlo.custom_call @randint() {{{attrs}}} : () -> {result_type}']

        elif target_name in ('randint_like',):
            # torch.randint_like -> random integers with same shape as input
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            low = node.args[1] if len(node.args) > 1 else node.kwargs.get('low', 0)
            high = node.args[2] if len(node.args) > 2 else node.kwargs.get('high')
            attrs = f'low = {low}, high = {high}'
            return [f'{result_ssa} = stablehlo.custom_call @randint_like({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('randperm',):
            # torch.randperm -> random permutation of integers [0, n)
            n = node.args[0] if node.args else node.kwargs.get('n')
            return [f'{result_ssa} = stablehlo.custom_call @randperm() {{n = {n}}} : () -> {result_type}']

        elif target_name in ('bernoulli',):
            # torch.bernoulli -> Bernoulli distributed random (0 or 1)
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @bernoulli({input_ssa}) : ({input_type}) -> {result_type}']

        elif target_name in ('multinomial',):
            # torch.multinomial -> sample from multinomial distribution
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            num_samples = node.args[1] if len(node.args) > 1 else node.kwargs.get('num_samples', 1)
            replacement = node.kwargs.get('replacement', False)
            attrs = f'num_samples = {num_samples}, replacement = {str(replacement).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @multinomial({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('normal',):
            # torch.normal -> normal distributed random
            mean = node.args[0] if node.args else node.kwargs.get('mean', 0.0)
            std = node.args[1] if len(node.args) > 1 else node.kwargs.get('std', 1.0)
            size = node.args[2] if len(node.args) > 2 else node.kwargs.get('size')
            if isinstance(mean, (int, float)) and isinstance(std, (int, float)):
                size_str = str(list(size)) if size else 'null'
                attrs = f'mean = {mean}, std = {std}, size = {size_str}'
                return [f'{result_ssa} = stablehlo.custom_call @normal() {{{attrs}}} : () -> {result_type}']
            else:
                # mean and std are tensors
                mean_ssa = get_input(0)
                std_ssa = get_input(1)
                mean_type = get_input_type(0)
                std_type = get_input_type(1)
                return [f'{result_ssa} = stablehlo.custom_call @normal({mean_ssa}, {std_ssa}) : '
                        f'({mean_type}, {std_type}) -> {result_type}']

        elif target_name in ('uniform', 'uniform_'):
            # torch.Tensor.uniform_ -> fill with uniform random
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            low = node.args[1] if len(node.args) > 1 else node.kwargs.get('from', 0.0)
            high = node.args[2] if len(node.args) > 2 else node.kwargs.get('to', 1.0)
            attrs = f'low = {low}, high = {high}'
            return [f'{result_ssa} = stablehlo.custom_call @uniform({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('exponential', 'exponential_'):
            # torch.Tensor.exponential_ -> fill with exponential random
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            lambd = node.args[1] if len(node.args) > 1 else node.kwargs.get('lambd', 1.0)
            attrs = f'lambd = {lambd}'
            return [f'{result_ssa} = stablehlo.custom_call @exponential({input_ssa}) '
                    f'{{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('poisson',):
            # torch.poisson -> Poisson distributed random
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            return [f'{result_ssa} = stablehlo.custom_call @poisson({input_ssa}) : ({input_type}) -> {result_type}']

        # ===== WINDOW FUNCTIONS =====
        elif target_name in ('hann_window',):
            # torch.hann_window -> periodic Hann window
            window_length = node.args[0] if node.args else node.kwargs.get('window_length', 10)
            periodic = node.kwargs.get('periodic', True)
            attrs = f'window_length = {window_length}, periodic = {str(periodic).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @hann_window() {{{attrs}}} : () -> {result_type}']

        elif target_name in ('hamming_window',):
            # torch.hamming_window -> Hamming window
            window_length = node.args[0] if node.args else node.kwargs.get('window_length', 10)
            periodic = node.kwargs.get('periodic', True)
            alpha = node.kwargs.get('alpha', 0.54)
            beta = node.kwargs.get('beta', 0.46)
            attrs = f'window_length = {window_length}, periodic = {str(periodic).lower()}, alpha = {alpha}, beta = {beta}'
            return [f'{result_ssa} = stablehlo.custom_call @hamming_window() {{{attrs}}} : () -> {result_type}']

        elif target_name in ('blackman_window',):
            # torch.blackman_window -> Blackman window
            window_length = node.args[0] if node.args else node.kwargs.get('window_length', 10)
            periodic = node.kwargs.get('periodic', True)
            attrs = f'window_length = {window_length}, periodic = {str(periodic).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @blackman_window() {{{attrs}}} : () -> {result_type}']

        elif target_name in ('bartlett_window',):
            # torch.bartlett_window -> Bartlett (triangular) window
            window_length = node.args[0] if node.args else node.kwargs.get('window_length', 10)
            periodic = node.kwargs.get('periodic', True)
            attrs = f'window_length = {window_length}, periodic = {str(periodic).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @bartlett_window() {{{attrs}}} : () -> {result_type}']

        elif target_name in ('kaiser_window',):
            # torch.kaiser_window -> Kaiser window with beta parameter
            window_length = node.args[0] if node.args else node.kwargs.get('window_length', 10)
            periodic = node.kwargs.get('periodic', True)
            beta = node.args[1] if len(node.args) > 1 else node.kwargs.get('beta', 12.0)
            attrs = f'window_length = {window_length}, periodic = {str(periodic).lower()}, beta = {beta}'
            return [f'{result_ssa} = stablehlo.custom_call @kaiser_window() {{{attrs}}} : () -> {result_type}']

        elif target_name in ('gaussian_window',):
            # Gaussian window function
            window_length = node.args[0] if node.args else node.kwargs.get('window_length', 10)
            std = node.args[1] if len(node.args) > 1 else node.kwargs.get('std', 1.0)
            attrs = f'window_length = {window_length}, std = {std}'
            return [f'{result_ssa} = stablehlo.custom_call @gaussian_window() {{{attrs}}} : () -> {result_type}']

        elif target_name in ('cosine_window',):
            # Cosine window function (raised cosine)
            window_length = node.args[0] if node.args else node.kwargs.get('window_length', 10)
            attrs = f'window_length = {window_length}'
            return [f'{result_ssa} = stablehlo.custom_call @cosine_window() {{{attrs}}} : () -> {result_type}']

        elif target_name in ('exponential_window',):
            # Exponential (Poisson) window function
            window_length = node.args[0] if node.args else node.kwargs.get('window_length', 10)
            tau = node.args[1] if len(node.args) > 1 else node.kwargs.get('tau', 1.0)
            attrs = f'window_length = {window_length}, tau = {tau}'
            return [f'{result_ssa} = stablehlo.custom_call @exponential_window() {{{attrs}}} : () -> {result_type}']

        # ===== DISTANCE FUNCTIONS =====
        elif target_name in ('cdist',):
            # torch.cdist -> pairwise distance between two sets of vectors
            x1_ssa = get_input(0)
            x2_ssa = get_input(1)
            x1_type = get_input_type(0)
            x2_type = get_input_type(1)
            p = node.args[2] if len(node.args) > 2 else node.kwargs.get('p', 2.0)
            compute_mode = node.kwargs.get('compute_mode', 'use_mm_for_euclid_dist_if_necessary')
            attrs = f'p = {p}, compute_mode = "{compute_mode}"'
            return [f'{result_ssa} = stablehlo.custom_call @cdist({x1_ssa}, {x2_ssa}) {{{attrs}}} : '
                    f'({x1_type}, {x2_type}) -> {result_type}']

        elif target_name in ('pdist',):
            # torch.pdist -> pairwise distance within a single set of vectors
            input_ssa = get_input(0)
            input_type = get_input_type(0)
            p = node.args[1] if len(node.args) > 1 else node.kwargs.get('p', 2.0)
            attrs = f'p = {p}'
            return [f'{result_ssa} = stablehlo.custom_call @pdist({input_ssa}) {{{attrs}}} : ({input_type}) -> {result_type}']

        elif target_name in ('pairwise_distance',):
            # torch.nn.functional.pairwise_distance -> element-wise distance
            x1_ssa = get_input(0)
            x2_ssa = get_input(1)
            x1_type = get_input_type(0)
            x2_type = get_input_type(1)
            p = node.args[2] if len(node.args) > 2 else node.kwargs.get('p', 2.0)
            eps = node.kwargs.get('eps', 1e-6)
            keepdim = node.kwargs.get('keepdim', False)
            attrs = f'p = {p}, eps = {eps}, keepdim = {str(keepdim).lower()}'
            return [f'{result_ssa} = stablehlo.custom_call @pairwise_distance({x1_ssa}, {x2_ssa}) {{{attrs}}} : '
                    f'({x1_type}, {x2_type}) -> {result_type}']

        elif target_name in ('cosine_similarity',):
            # torch.nn.functional.cosine_similarity -> cosine similarity
            x1_ssa = get_input(0)
            x2_ssa = get_input(1)
            x1_type = get_input_type(0)
            x2_type = get_input_type(1)
            dim = node.args[2] if len(node.args) > 2 else node.kwargs.get('dim', 1)
            eps = node.kwargs.get('eps', 1e-8)
            attrs = f'dim = {dim}, eps = {eps}'
            return [f'{result_ssa} = stablehlo.custom_call @cosine_similarity({x1_ssa}, {x2_ssa}) {{{attrs}}} : '
                    f'({x1_type}, {x2_type}) -> {result_type}']

        elif target_name in ('euclidean_distance', 'l2_distance'):
            # Euclidean (L2) distance between vectors
            x1_ssa = get_input(0)
            x2_ssa = get_input(1)
            x1_type = get_input_type(0)
            x2_type = get_input_type(1)
            return [f'{result_ssa} = stablehlo.custom_call @euclidean_distance({x1_ssa}, {x2_ssa}) : '
                    f'({x1_type}, {x2_type}) -> {result_type}']

        elif target_name in ('manhattan_distance', 'l1_distance'):
            # Manhattan (L1) distance between vectors
            x1_ssa = get_input(0)
            x2_ssa = get_input(1)
            x1_type = get_input_type(0)
            x2_type = get_input_type(1)
            return [f'{result_ssa} = stablehlo.custom_call @manhattan_distance({x1_ssa}, {x2_ssa}) : '
                    f'({x1_type}, {x2_type}) -> {result_type}']

        elif target_name in ('chebyshev_distance', 'linf_distance'):
            # Chebyshev (L-infinity) distance between vectors
            x1_ssa = get_input(0)
            x2_ssa = get_input(1)
            x1_type = get_input_type(0)
            x2_type = get_input_type(1)
            return [f'{result_ssa} = stablehlo.custom_call @chebyshev_distance({x1_ssa}, {x2_ssa}) : '
                    f'({x1_type}, {x2_type}) -> {result_type}']

        elif target_name in ('minkowski_distance',):
            # Minkowski distance with configurable p
            x1_ssa = get_input(0)
            x2_ssa = get_input(1)
            x1_type = get_input_type(0)
            x2_type = get_input_type(1)
            p = node.args[2] if len(node.args) > 2 else node.kwargs.get('p', 2.0)
            attrs = f'p = {p}'
            return [f'{result_ssa} = stablehlo.custom_call @minkowski_distance({x1_ssa}, {x2_ssa}) {{{attrs}}} : '
                    f'({x1_type}, {x2_type}) -> {result_type}']

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
            input_name = input_node.name if input_node else ''
            input_shape = self.shape_map.get(input_name, ()) if input_node else ()
            output_shape = self.shape_map.get(node.name, ())

            # Build broadcast_dimensions - map input dims to output dims
            broadcast_dims = []
            input_ndim = len(input_shape)
            output_ndim = len(output_shape)
            offset = output_ndim - input_ndim

            for i in range(input_ndim):
                broadcast_dims.append(offset + i)

            dims_str = ', '.join(str(d) for d in broadcast_dims)

            # Use dynamic_broadcast_in_dim if input or output has dynamic dimensions
            if self._is_dynamic_shape(node.name) or self._is_dynamic_shape(input_name):
                # Generate output shape tensor for dynamic broadcast
                lines = self._get_shape_tensor(node.name, result_ssa)
                shape_ssa = f'{result_ssa}_shape'
                lines.append(f'{result_ssa} = stablehlo.dynamic_broadcast_in_dim {input_ssa}, {shape_ssa}, '
                             f'dims = [{dims_str}] : ({input_type}, tensor<{output_ndim}xi32>) -> {result_type}')
                return lines
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
    Model weights are captured and exported as additional function arguments.

    Args:
        source_code: Python source containing an nn.Module class
        class_name: Name of the nn.Module class to trace
        input_shapes: List of tuples, e.g., [(1, 8), (1, 16)]
        seed: Random seed for reproducible inputs (default: 42)

    Returns:
        Dictionary with:
            - 'mlir': StableHLO MLIR text
            - 'inputs': List of numpy arrays (actual inputs, not weights)
            - 'weights': List of numpy arrays (model weights as function args)
            - 'weight_names': List of weight parameter names
            - 'outputs': List of numpy arrays (serializable)
            - 'seed': Random seed used
            - 'input_shapes': Input shape tuples
            - 'output_shapes': Output shape tuples
            - 'input_count': Number of actual inputs (before weights)
            - 'weight_count': Number of weight arguments
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

    # Create deterministic inputs with proper dtypes
    def create_input(spec):
        """Create a random tensor from shape/dtype specification."""
        if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[-1], str):
            # Format: (dim1, dim2, ..., 'dtype')
            shape = spec[:-1]
            dtype_str = spec[-1]
        else:
            # Format: (dim1, dim2, ...) - assume float32
            shape = spec
            dtype_str = 'f32'

        # Map dtype string to torch dtype
        dtype_map = {
            'f32': torch.float32,
            'f64': torch.float64,
            'f16': torch.float16,
            'bf16': torch.bfloat16,
            'i32': torch.int32,
            'i64': torch.int64,
            'i16': torch.int16,
            'i8': torch.int8,
            'bool': torch.bool,
        }
        dtype = dtype_map.get(dtype_str, torch.float32)

        # Generate random tensor with appropriate method
        if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            # For embeddings, use reasonable vocab range (0 to 99)
            return torch.randint(0, 100, shape, dtype=dtype)
        elif dtype == torch.bool:
            return torch.rand(*shape) > 0.5
        else:
            return torch.randn(*shape, dtype=dtype)

    sample_inputs = tuple(create_input(spec) for spec in input_shapes)

    # Run forward pass to get actual outputs
    with torch.no_grad():
        outputs = model(*sample_inputs)

    # Normalize outputs to tuple
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    elif not isinstance(outputs, tuple):
        outputs = tuple(outputs)

    # Trace for MLIR with weight capture enabled
    traced = symbolic_trace(model)
    converter = FXToStableHLO(traced, sample_inputs, capture_weights=True)
    mlir = converter.convert()

    # Convert tensors to numpy arrays for serialization
    input_arrays = [inp.detach().cpu().numpy() for inp in sample_inputs]
    output_arrays = [out.detach().cpu().numpy() for out in outputs]

    # Extract weight tensors from the converter
    weight_arrays = [w['tensor'].cpu().numpy() for w in converter.weight_args]
    weight_names = [w['name'] for w in converter.weight_args]

    return {
        'mlir': mlir,
        'inputs': input_arrays,
        'weights': weight_arrays,
        'weight_names': weight_names,
        'outputs': output_arrays,
        'seed': seed,
        'input_shapes': [tuple(arr.shape) for arr in input_arrays],
        'output_shapes': [tuple(arr.shape) for arr in output_arrays],
        'input_count': len(input_arrays),
        'weight_count': len(weight_arrays),
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
            - 'input_npy': List of bytes (.npy format) - actual inputs only
            - 'weight_npy': List of bytes (.npy format) - model weights
            - 'weight_names': List of weight parameter names
            - 'output_npy': List of bytes (.npy format)
            - 'seed': Random seed used
            - 'input_shapes': Input shape tuples
            - 'output_shapes': Output shape tuples
            - 'input_count': Number of actual inputs
            - 'weight_count': Number of weight arguments
    """
    result = trace_with_values(source_code, class_name, input_shapes, seed)

    # Convert numpy arrays to .npy bytes
    input_npy = [serialize_npy(arr) for arr in result['inputs']]
    weight_npy = [serialize_npy(arr) for arr in result['weights']]
    output_npy = [serialize_npy(arr) for arr in result['outputs']]

    return {
        'mlir': result['mlir'],
        'input_npy': input_npy,
        'weight_npy': weight_npy,
        'weight_names': result['weight_names'],
        'output_npy': output_npy,
        'seed': result['seed'],
        'input_shapes': result['input_shapes'],
        'output_shapes': result['output_shapes'],
        'input_count': result['input_count'],
        'weight_count': result['weight_count'],
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
