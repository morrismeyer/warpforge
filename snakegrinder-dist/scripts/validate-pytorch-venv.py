#!/usr/bin/env python3
"""
Validate PyTorch venv for SnakeGrinder.

This script tests that PyTorch is correctly installed and functional
for snakegrinder's core use case: torch.fx.symbolic_trace.

Exit codes:
  0 - All tests passed
  1 - Test failure
"""

import sys

def test_imports():
    """Test that required modules can be imported."""
    print("Testing imports...")

    import torch
    print(f"  torch: OK (version {torch.__version__})")

    import torch.nn as nn
    print("  torch.nn: OK")

    from torch.fx import symbolic_trace
    print("  torch.fx.symbolic_trace: OK")

    return torch.__version__

def test_version(torch_version, expected_version):
    """Test that PyTorch version matches expected."""
    print(f"Testing version (expected {expected_version})...")

    # Compare major.minor.patch
    actual_parts = torch_version.split('.')[:3]
    expected_parts = expected_version.split('.')[:3]

    if actual_parts != expected_parts:
        print(f"  FAIL: got {torch_version}, expected {expected_version}")
        return False

    print(f"  Version match: OK")
    return True

def test_tensor_operations():
    """Test basic tensor operations."""
    print("Testing tensor operations...")

    import torch

    # Create tensors
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])

    # Addition
    z = x + y
    expected = [5.0, 7.0, 9.0]
    assert z.tolist() == expected, f"Addition failed: {z.tolist()} != {expected}"
    print("  Addition: OK")

    # Multiplication
    z = x * y
    expected = [4.0, 10.0, 18.0]
    assert z.tolist() == expected, f"Multiplication failed: {z.tolist()} != {expected}"
    print("  Multiplication: OK")

    # Matrix multiplication
    a = torch.randn(2, 3)
    b = torch.randn(3, 4)
    c = torch.matmul(a, b)
    assert c.shape == (2, 4), f"Matmul shape wrong: {c.shape}"
    print("  Matrix multiplication: OK")

def test_neural_network():
    """Test basic neural network layers."""
    print("Testing neural network layers...")

    import torch
    import torch.nn as nn

    # Linear layer
    linear = nn.Linear(10, 5)
    x = torch.randn(1, 10)
    y = linear(x)
    assert y.shape == (1, 5), f"Linear output shape wrong: {y.shape}"
    print("  nn.Linear: OK")

    # ReLU
    relu = nn.ReLU()
    x = torch.tensor([-1.0, 0.0, 1.0])
    y = relu(x)
    assert y.tolist() == [0.0, 0.0, 1.0], f"ReLU output wrong: {y.tolist()}"
    print("  nn.ReLU: OK")

    # Sequential
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4)
    )
    x = torch.randn(1, 8)
    y = model(x)
    assert y.shape == (1, 4), f"Sequential output shape wrong: {y.shape}"
    print("  nn.Sequential: OK")

def test_fx_symbolic_trace():
    """Test torch.fx.symbolic_trace - THE KEY TEST for snakegrinder."""
    print("Testing torch.fx.symbolic_trace (critical for snakegrinder)...")

    import torch
    import torch.nn as nn
    from torch.fx import symbolic_trace

    # Define a simple model
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, 4)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # Create and trace the model
    model = SimpleMLP()
    traced = symbolic_trace(model)

    # Verify we got a valid graph
    assert traced.graph is not None, "Traced graph is None"
    print("  Graph created: OK")

    nodes = list(traced.graph.nodes)
    assert len(nodes) > 0, "Graph has no nodes"
    print(f"  Graph has {len(nodes)} nodes: OK")

    # Verify the traced model runs
    x = torch.randn(1, 8)
    y_original = model(x)
    y_traced = traced(x)

    # Results should match
    assert y_original.shape == y_traced.shape, "Shape mismatch"
    assert torch.allclose(y_original, y_traced), "Output mismatch"
    print("  Traced model execution: OK")

    # Print the graph for debugging
    print("  FX Graph:")
    for node in nodes:
        print(f"    {node.op}: {node.name}")

def main():
    """Run all validation tests."""
    # Get expected version from environment or use default
    import os
    expected_version = os.environ.get('PYTORCH_VERSION', '2.7.0')

    print("=" * 60)
    print("SnakeGrinder PyTorch Venv Validation")
    print("=" * 60)
    print()

    try:
        # Test 1: Imports
        torch_version = test_imports()
        print()

        # Test 2: Version
        if not test_version(torch_version, expected_version):
            print("\nWARNING: Version mismatch (continuing anyway)")
        print()

        # Test 3: Tensor operations
        test_tensor_operations()
        print()

        # Test 4: Neural network layers
        test_neural_network()
        print()

        # Test 5: FX symbolic trace (CRITICAL)
        test_fx_symbolic_trace()
        print()

        print("=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
