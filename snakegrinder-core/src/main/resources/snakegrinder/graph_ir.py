"""
Computation Graph IR for mock tracer.

This module defines the internal representation of a traced computation
as a directed acyclic graph (DAG) of operations.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tracer import TracedTensor, TensorMeta


class GraphNode:
    """A single operation in the computation graph."""

    __slots__ = ('node_id', 'op_type', 'inputs', 'outputs', 'attrs', 'result_meta')

    def __init__(self,
                 node_id: str,
                 op_type: str,
                 inputs: List[TracedTensor],
                 result_meta: TensorMeta,
                 attrs: Optional[Dict[str, Any]] = None):
        self.node_id = node_id
        self.op_type = op_type  # 'add', 'multiply', 'dot_general', 'reshape', etc.
        self.inputs = inputs
        self.outputs: List[TracedTensor] = []
        self.attrs = attrs or {}
        self.result_meta = result_meta

    def __repr__(self):
        return f"GraphNode({self.op_type}, id={self.node_id})"


class FunctionSignature:
    """Describes the signature of a traced function."""

    __slots__ = ('name', 'input_metas', 'output_metas', 'input_names', 'output_names')

    def __init__(self,
                 name: str,
                 input_metas: List[TensorMeta],
                 output_metas: List[TensorMeta],
                 input_names: Optional[List[str]] = None,
                 output_names: Optional[List[str]] = None):
        self.name = name
        self.input_metas = input_metas
        self.output_metas = output_metas
        self.input_names = input_names or [f"arg{i}" for i in range(len(input_metas))]
        self.output_names = output_names or [f"out{i}" for i in range(len(output_metas))]


class ComputationGraph:
    """
    The internal IR representing a traced computation.
    This is a DAG of operations with explicit inputs/outputs.
    """

    def __init__(self, name: str = "traced_function"):
        self.name = name
        self.nodes: List[GraphNode] = []
        self.inputs: List[TracedTensor] = []
        self.outputs: List[TracedTensor] = []
        self._node_counter = 0

    def add_input(self, meta: TensorMeta, name: Optional[str] = None) -> TracedTensor:
        """Create a function input placeholder."""
        # Import here to avoid circular dependency
        from tracer import TracedTensor

        idx = len(self.inputs)
        tensor_name = name or f"arg{idx}"

        node = GraphNode(
            node_id=f"input_{idx}",
            op_type='input',
            inputs=[],
            result_meta=meta,
            attrs={'arg_index': idx, 'arg_name': tensor_name}
        )
        self.nodes.append(node)

        tensor = TracedTensor(meta, node_id=tensor_name, producing_op=node)
        node.outputs = [tensor]
        self.inputs.append(tensor)
        return tensor

    def add_op(self,
               op_type: str,
               inputs: List[TracedTensor],
               result_meta: TensorMeta,
               attrs: Optional[Dict[str, Any]] = None) -> GraphNode:
        """Add an operation node to the graph."""
        self._node_counter += 1
        node = GraphNode(
            node_id=f"{op_type}_{self._node_counter}",
            op_type=op_type,
            inputs=inputs,
            result_meta=result_meta,
            attrs=attrs
        )
        self.nodes.append(node)
        return node

    def set_outputs(self, outputs: List[TracedTensor]):
        """Mark which tensors are function outputs."""
        self.outputs = outputs

    def get_signature(self) -> FunctionSignature:
        """Extract the function signature from the graph."""
        return FunctionSignature(
            name=self.name,
            input_metas=[t.meta for t in self.inputs],
            output_metas=[t.meta for t in self.outputs],
            input_names=[t.node_id for t in self.inputs],
            output_names=[t.node_id for t in self.outputs]
        )

    def topological_sort(self) -> List[GraphNode]:
        """Return nodes in topological order."""
        # Nodes are already in creation order which is topological
        return list(self.nodes)

    def validate(self) -> List[str]:
        """Validate the graph structure. Returns list of errors."""
        errors = []
        if not self.inputs:
            errors.append("Graph has no inputs")
        if not self.outputs:
            errors.append("Graph has no outputs")
        return errors

    def __repr__(self):
        return f"ComputationGraph({self.name}, {len(self.nodes)} nodes)"
