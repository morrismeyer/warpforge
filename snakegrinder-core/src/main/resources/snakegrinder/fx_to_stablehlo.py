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
            lines.append(
                f'{matmul_ssa} = stablehlo.dot_general {input_ssa}, {weight_ssa}, '
                f'contracting_dims = [1] x [1] : ({input_type}, {weight_type}) -> {result_type}'
            )

            if module.bias is not None:
                bias_ssa = f'{result_ssa}_bias'
                bias_type = f'tensor<{weight_shape[0]}xf32>'
                lines.append(f'{bias_ssa} = stablehlo.constant dense<0.0> : {bias_type}  // placeholder for bias')
                lines.append(f'{result_ssa} = stablehlo.add {matmul_ssa}, {bias_ssa} : {result_type}')
            else:
                lines[-1] = lines[-1].replace(matmul_ssa, result_ssa)

            return lines
        else:
            return [f'// Unsupported module: {type(module).__name__}']

    def _convert_call_function(self, node, result_ssa: str, result_type: str) -> List[str]:
        """Convert call_function (e.g., torch.relu)."""
        target = node.target
        target_name = getattr(target, '__name__', str(target))

        if target_name == 'relu' or target == torch.relu:
            input_ssa = self.ssa_map.get(node.args[0].name, '%unknown')
            zero_ssa = f'{result_ssa}_zero'
            return [
                f'{zero_ssa} = stablehlo.constant dense<0.0> : {result_type}',
                f'{result_ssa} = stablehlo.maximum {input_ssa}, {zero_ssa} : {result_type}'
            ]
        elif target_name == 'add':
            lhs = self.ssa_map.get(node.args[0].name, '%unknown')
            rhs = self.ssa_map.get(node.args[1].name, '%unknown')
            return [f'{result_ssa} = stablehlo.add {lhs}, {rhs} : {result_type}']
        elif target_name == 'matmul':
            lhs = self.ssa_map.get(node.args[0].name, '%unknown')
            rhs = self.ssa_map.get(node.args[1].name, '%unknown')
            lhs_type = self._tensor_type(node.args[0].name)
            rhs_type = self._tensor_type(node.args[1].name)
            return [
                f'{result_ssa} = stablehlo.dot_general {lhs}, {rhs}, '
                f'contracting_dims = [1] x [0] : ({lhs_type}, {rhs_type}) -> {result_type}'
            ]
        else:
            return [f'// Unsupported function: {target_name}']

    def _convert_call_method(self, node, result_ssa: str, result_type: str) -> List[str]:
        """Convert call_method."""
        return [f'// call_method: {node.target}']

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
