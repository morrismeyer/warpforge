"""
StableHLO MLIR Emitter.

Converts a ComputationGraph to StableHLO MLIR text format.
See: https://openxla.org/stablehlo/spec
"""

from __future__ import annotations
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from graph_ir import ComputationGraph, GraphNode
    from tracer import TensorMeta, TracedTensor


class StableHLOEmitter:
    """
    Converts a ComputationGraph to StableHLO MLIR text format.
    """

    # Map from internal dtype to MLIR type string
    DTYPE_MAP = {
        'f32': 'f32',
        'f64': 'f64',
        'f16': 'f16',
        'bf16': 'bf16',
        'i32': 'i32',
        'i64': 'i64',
        'i16': 'i16',
        'i8': 'i8',
        'i1': 'i1',
        'bool': 'i1',
    }

    def __init__(self, graph: ComputationGraph):
        self.graph = graph
        self.ssa_names: Dict[str, str] = {}  # tensor node_id -> %name
        self._ssa_counter = 0

    def emit(self) -> str:
        """Generate complete StableHLO MLIR module."""
        lines = []
        lines.append('module @main {')
        lines.append(self._emit_function())
        lines.append('}')
        return '\n'.join(lines)

    def _emit_function(self) -> str:
        """Generate the main function."""
        sig = self.graph.get_signature()

        # Build function signature
        input_types = [self._tensor_type(m) for m in sig.input_metas]
        output_types = [self._tensor_type(m) for m in sig.output_metas]

        # Create SSA names for inputs
        arg_strs = []
        for i, (name, itype) in enumerate(zip(sig.input_names, input_types)):
            ssa_name = f"%arg{i}"
            self.ssa_names[name] = ssa_name
            arg_strs.append(f"{ssa_name}: {itype}")

        args_str = ', '.join(arg_strs)
        returns_str = ', '.join(output_types)

        lines = []
        lines.append(f'  func.func public @{sig.name}({args_str}) -> ({returns_str}) {{')

        # Emit operations in topological order
        for node in self.graph.topological_sort():
            if node.op_type == 'input':
                continue  # Already handled as function args

            op_lines = self._emit_op(node)
            for line in op_lines:
                lines.append(f'    {line}')

        # Emit return
        output_names = [self._get_ssa_name(t) for t in self.graph.outputs]
        if len(output_types) == 1:
            lines.append(f'    stablehlo.return {output_names[0]} : {output_types[0]}')
        else:
            output_types_str = ', '.join(output_types)
            lines.append(f'    stablehlo.return {", ".join(output_names)} : {output_types_str}')

        lines.append('  }')
        return '\n'.join(lines)

    def _emit_op(self, node: GraphNode) -> List[str]:
        """Emit a single operation. Returns list of MLIR lines."""
        result_name = self._new_ssa_name(node)
        result_type = self._tensor_type(node.result_meta)

        op_type = node.op_type

        if op_type == 'add':
            return self._emit_binary('stablehlo.add', node, result_name, result_type)
        elif op_type == 'multiply':
            return self._emit_binary('stablehlo.multiply', node, result_name, result_type)
        elif op_type == 'divide':
            return self._emit_binary('stablehlo.divide', node, result_name, result_type)
        elif op_type == 'negate':
            return self._emit_unary('stablehlo.negate', node, result_name, result_type)
        elif op_type == 'dot_general':
            return self._emit_dot_general(node, result_name, result_type)
        elif op_type == 'reshape':
            return self._emit_reshape(node, result_name, result_type)
        elif op_type == 'transpose':
            return self._emit_transpose(node, result_name, result_type)
        elif op_type == 'relu':
            return self._emit_relu(node, result_name, result_type)
        elif op_type == 'constant':
            return self._emit_constant(node, result_name, result_type)
        elif op_type == 'reduce_sum':
            return self._emit_reduce('stablehlo.reduce', node, result_name, result_type, 'add')
        elif op_type == 'reduce_mean':
            return self._emit_reduce_mean(node, result_name, result_type)
        else:
            # Unknown op - emit as a comment for debugging
            return [f'// Unknown op: {op_type}',
                    f'{result_name} = stablehlo.custom_call @{op_type}() : () -> {result_type}']

    def _emit_binary(self, op: str, node: GraphNode,
                     result: str, result_type: str) -> List[str]:
        lhs = self._get_ssa_name(node.inputs[0])
        rhs = self._get_ssa_name(node.inputs[1])

        # Check if we need broadcasting
        lhs_meta = node.inputs[0].meta
        rhs_meta = node.inputs[1].meta

        lines = []

        # Handle broadcast if shapes differ
        if lhs_meta.shape != node.result_meta.shape:
            broadcast_lhs = f"{result}_lhs_bc"
            dims = self._compute_broadcast_dims(lhs_meta.shape, node.result_meta.shape)
            lines.append(f'{broadcast_lhs} = stablehlo.broadcast_in_dim {lhs}, dims = [{dims}] : '
                         f'({self._tensor_type(lhs_meta)}) -> {result_type}')
            lhs = broadcast_lhs

        if rhs_meta.shape != node.result_meta.shape:
            broadcast_rhs = f"{result}_rhs_bc"
            dims = self._compute_broadcast_dims(rhs_meta.shape, node.result_meta.shape)
            lines.append(f'{broadcast_rhs} = stablehlo.broadcast_in_dim {rhs}, dims = [{dims}] : '
                         f'({self._tensor_type(rhs_meta)}) -> {result_type}')
            rhs = broadcast_rhs

        lines.append(f'{result} = {op} {lhs}, {rhs} : {result_type}')
        return lines

    def _emit_unary(self, op: str, node: GraphNode,
                    result: str, result_type: str) -> List[str]:
        operand = self._get_ssa_name(node.inputs[0])
        return [f'{result} = {op} {operand} : {result_type}']

    def _emit_dot_general(self, node: GraphNode,
                          result: str, result_type: str) -> List[str]:
        """
        Emit stablehlo.dot_general for matrix multiplication.
        For simple 2D matmul (M,K) @ (K,N):
          - lhs_contracting_dimensions = [1]
          - rhs_contracting_dimensions = [0]
        """
        lhs = self._get_ssa_name(node.inputs[0])
        rhs = self._get_ssa_name(node.inputs[1])
        lhs_type = self._tensor_type(node.inputs[0].meta)
        rhs_type = self._tensor_type(node.inputs[1].meta)

        lhs_shape = node.inputs[0].meta.shape
        rhs_shape = node.inputs[1].meta.shape

        # Determine contracting dimensions
        # For 2D: lhs contracts on dim 1, rhs contracts on dim 0
        # For batched: same but with batch dims
        lhs_contract = len(lhs_shape) - 1
        rhs_contract = len(rhs_shape) - 2 if len(rhs_shape) > 2 else 0

        # Batching dimensions (all dims except last 2)
        lhs_batch = list(range(len(lhs_shape) - 2)) if len(lhs_shape) > 2 else []
        rhs_batch = list(range(len(rhs_shape) - 2)) if len(rhs_shape) > 2 else []

        dot_dims = (
            f'#stablehlo.dot<'
            f'lhs_batching_dimensions = [{", ".join(map(str, lhs_batch))}], '
            f'rhs_batching_dimensions = [{", ".join(map(str, rhs_batch))}], '
            f'lhs_contracting_dimensions = [{lhs_contract}], '
            f'rhs_contracting_dimensions = [{rhs_contract}]'
            f'>'
        )

        return [
            f'{result} = stablehlo.dot_general {lhs}, {rhs}, {dot_dims} : '
            f'({lhs_type}, {rhs_type}) -> {result_type}'
        ]

    def _emit_reshape(self, node: GraphNode,
                      result: str, result_type: str) -> List[str]:
        operand = self._get_ssa_name(node.inputs[0])
        operand_type = self._tensor_type(node.inputs[0].meta)
        return [f'{result} = stablehlo.reshape {operand} : {operand_type} -> {result_type}']

    def _emit_transpose(self, node: GraphNode,
                        result: str, result_type: str) -> List[str]:
        operand = self._get_ssa_name(node.inputs[0])
        operand_type = self._tensor_type(node.inputs[0].meta)
        perm = node.attrs['permutation']
        perm_str = ', '.join(str(p) for p in perm)
        return [
            f'{result} = stablehlo.transpose {operand}, dims = [{perm_str}] : '
            f'{operand_type} -> {result_type}'
        ]

    def _emit_relu(self, node: GraphNode,
                   result: str, result_type: str) -> List[str]:
        """
        ReLU is implemented as stablehlo.maximum(x, 0).
        """
        operand = self._get_ssa_name(node.inputs[0])
        dtype = node.inputs[0].meta.dtype

        # Create zero constant
        zero_name = f"{result}_zero"
        zero_val = "0.0" if dtype.startswith('f') else "0"

        return [
            f'{zero_name} = stablehlo.constant dense<{zero_val}> : {result_type}',
            f'{result} = stablehlo.maximum {operand}, {zero_name} : {result_type}'
        ]

    def _emit_constant(self, node: GraphNode,
                       result: str, result_type: str) -> List[str]:
        """Emit a constant value."""
        value = node.attrs.get('value', 0)
        dtype = node.result_meta.dtype

        if dtype.startswith('f'):
            val_str = f"{float(value)}"
        else:
            val_str = f"{int(value)}"

        return [f'{result} = stablehlo.constant dense<{val_str}> : {result_type}']

    def _emit_reduce(self, op: str, node: GraphNode,
                     result: str, result_type: str, reducer: str) -> List[str]:
        """Emit a reduction operation."""
        operand = self._get_ssa_name(node.inputs[0])
        operand_type = self._tensor_type(node.inputs[0].meta)
        dims = node.attrs.get('dimensions', ())
        dims_str = ', '.join(str(d) for d in dims)
        dtype = node.inputs[0].meta.dtype
        scalar_type = self.DTYPE_MAP.get(dtype, 'f32')

        # Initial value for reduction
        if reducer == 'add':
            init_val = "0.0" if dtype.startswith('f') else "0"
        else:
            init_val = "0.0"

        init_name = f"{result}_init"

        return [
            f'{init_name} = stablehlo.constant dense<{init_val}> : tensor<{scalar_type}>',
            f'{result} = stablehlo.reduce({operand} init: {init_name}) across dimensions = [{dims_str}] : '
            f'({operand_type}, tensor<{scalar_type}>) -> {result_type}',
            f'  reducer(%arg0: tensor<{scalar_type}>, %arg1: tensor<{scalar_type}>)  {{',
            f'    %sum = stablehlo.add %arg0, %arg1 : tensor<{scalar_type}>',
            f'    stablehlo.return %sum : tensor<{scalar_type}>',
            f'  }}'
        ]

    def _emit_reduce_mean(self, node: GraphNode,
                          result: str, result_type: str) -> List[str]:
        """Emit mean as sum / count."""
        # First emit sum
        sum_name = f"{result}_sum"
        sum_lines = self._emit_reduce('stablehlo.reduce', node, sum_name, result_type, 'add')

        # Compute count
        dims = node.attrs.get('dimensions', ())
        count = 1
        for d in dims:
            count *= node.inputs[0].meta.shape[d]

        dtype = node.inputs[0].meta.dtype
        count_val = f"{float(count)}" if dtype.startswith('f') else f"{count}"
        count_name = f"{result}_count"

        return sum_lines + [
            f'{count_name} = stablehlo.constant dense<{count_val}> : {result_type}',
            f'{result} = stablehlo.divide {sum_name}, {count_name} : {result_type}'
        ]

    def _tensor_type(self, meta: TensorMeta) -> str:
        """Convert TensorMeta to MLIR tensor type string."""
        if not meta.shape:  # Scalar
            dtype_str = self.DTYPE_MAP.get(meta.dtype, 'f32')
            return f'tensor<{dtype_str}>'

        shape_str = 'x'.join(str(d) for d in meta.shape)
        dtype_str = self.DTYPE_MAP.get(meta.dtype, 'f32')
        return f'tensor<{shape_str}x{dtype_str}>'

    def _get_ssa_name(self, tensor: TracedTensor) -> str:
        """Get the SSA name for a tensor."""
        if tensor.node_id in self.ssa_names:
            return self.ssa_names[tensor.node_id]
        raise ValueError(f"Unknown tensor: {tensor.node_id}")

    def _new_ssa_name(self, node: GraphNode) -> str:
        """Create a new SSA name for an operation result."""
        self._ssa_counter += 1
        name = f"%{self._ssa_counter}"
        if node.outputs:
            for output in node.outputs:
                self.ssa_names[output.node_id] = name
        return name

    def _compute_broadcast_dims(self, from_shape: tuple, to_shape: tuple) -> str:
        """Compute broadcast dimensions mapping."""
        # Map each dimension of from_shape to corresponding dimension in to_shape
        # Starting from the right (broadcasting semantics)
        offset = len(to_shape) - len(from_shape)
        dims = [offset + i for i in range(len(from_shape))]
        return ', '.join(str(d) for d in dims)
