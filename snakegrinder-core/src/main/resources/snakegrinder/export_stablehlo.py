# SnakeGrinder StableHLO export script.
# Attempts to export a minimal PyTorch computation to StableHLO MLIR.

import sys
import json


def minimal_pytorch_computation():
    """
    The minimal PyTorch computation we are validating.
    This is the "tiny example" from the MVP spec.
    """
    import torch
    torch.manual_seed(0)  # Determinism
    x = torch.randn(2, 3)
    w = torch.randn(3, 4)
    y = x @ w
    return x, w, y


def export_via_torch_xla(output_path):
    """
    Export using torch_xla to StableHLO.
    Returns (success: bool, mlir_content: str or None, error: str or None)
    """
    try:
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm

        torch.manual_seed(0)

        # Get XLA device
        device = xm.xla_device()

        # Create tensors on XLA device
        x = torch.randn(2, 3, device=device)
        w = torch.randn(3, 4, device=device)

        # Define the computation
        def matmul_fn(x, w):
            return x @ w

        # Try to get StableHLO representation
        # torch_xla 2.x provides stablehlo export via torch_xla.stablehlo
        try:
            from torch_xla.stablehlo import exported_program_to_stablehlo
            import torch.export

            # Export the function
            class MatMulModule(torch.nn.Module):
                def forward(self, x, w):
                    return x @ w

            model = MatMulModule()
            example_x = torch.randn(2, 3)
            example_w = torch.randn(3, 4)

            exported = torch.export.export(model, (example_x, example_w))
            stablehlo = exported_program_to_stablehlo(exported)
            mlir_text = stablehlo.get_stablehlo_text()

            return True, mlir_text, None

        except ImportError:
            # Fallback: try older torch_xla API
            try:
                # Get HLO text (may be HLO rather than StableHLO)
                y = matmul_fn(x, w)
                xm.mark_step()

                # Try to get computation graph
                hlo_text = torch_xla._XLAC._get_xla_tensors_hlo([y])

                # HLO is close to StableHLO but not identical
                # For MVP, we'll accept HLO if StableHLO is unavailable
                if "stablehlo." in hlo_text.lower() or "hlo." in hlo_text.lower():
                    return True, hlo_text, None
                else:
                    return True, hlo_text, "Warning: Output is HLO, not StableHLO"

            except Exception as e:
                return False, None, f"torch_xla export failed: {e}"

    except ImportError as e:
        return False, None, f"torch_xla not available: {e}"
    except Exception as e:
        return False, None, f"Unexpected error in torch_xla export: {e}"


def export_via_jax(output_path):
    """
    Export using JAX to StableHLO.
    This is a fallback if torch_xla is unavailable.
    Returns (success: bool, mlir_content: str or None, error: str or None)
    """
    try:
        import jax
        import jax.numpy as jnp
        from jax import make_jaxpr

        # Set deterministic key
        key = jax.random.PRNGKey(0)

        # Equivalent computation in JAX
        def matmul_fn(x, w):
            return jnp.matmul(x, w)

        # Create example inputs
        key1, key2 = jax.random.split(key)
        x = jax.random.normal(key1, (2, 3))
        w = jax.random.normal(key2, (3, 4))

        # Try to get StableHLO
        try:
            # JAX 0.4.1+ has native StableHLO export
            from jax._src.interpreters import mlir as jax_mlir
            from jax.experimental import export as jax_export

            # Export to StableHLO
            lowered = jax.jit(matmul_fn).lower(x, w)
            stablehlo_module = lowered.compiler_ir(dialect='stablehlo')
            mlir_text = str(stablehlo_module)

            return True, mlir_text, None

        except (ImportError, AttributeError):
            # Fallback: try basic MHLO/HLO export
            try:
                lowered = jax.jit(matmul_fn).lower(x, w)
                # Get the HLO text
                hlo_text = lowered.as_text()

                if "stablehlo." in hlo_text or "mhlo." in hlo_text:
                    return True, hlo_text, None
                else:
                    return True, hlo_text, "Warning: Output may be HLO rather than StableHLO"

            except Exception as e:
                return False, None, f"JAX lowering failed: {e}"

    except ImportError as e:
        return False, None, f"JAX not available: {e}"
    except Exception as e:
        return False, None, f"Unexpected error in JAX export: {e}"


def run_pytorch_smoke_test():
    """
    Run the minimal PyTorch computation to verify it works.
    Returns (success: bool, output_shape: str or None, error: str or None)
    """
    try:
        x, w, y = minimal_pytorch_computation()
        shape = str(y.shape)
        print(f"PyTorch smoke test passed. Output shape: {shape}")
        return True, shape, None
    except ImportError as e:
        return False, None, f"PyTorch not available: {e}"
    except Exception as e:
        return False, None, f"PyTorch computation failed: {e}"


def export_stablehlo(output_dir):
    """
    Main export function. Tries available export paths in priority order.
    Returns a result dict for manifest generation.
    """
    import os

    result = {
        "status": "pending",
        "smoke_test_passed": False,
        "smoke_test_shape": None,
        "export_path_used": None,
        "mlir_file": None,
        "error": None,
        "warnings": [],
    }

    # Step 1: Run PyTorch smoke test
    smoke_ok, shape, smoke_err = run_pytorch_smoke_test()
    result["smoke_test_passed"] = smoke_ok
    result["smoke_test_shape"] = shape

    if not smoke_ok:
        result["status"] = "failed"
        result["error"] = f"PyTorch smoke test failed: {smoke_err}"
        return result

    # Step 2: Try export paths in priority order

    # Priority 1: torch_xla
    success, mlir_content, error = export_via_torch_xla(output_dir)
    if success:
        result["export_path_used"] = "torch_xla"
        if error:
            result["warnings"].append(error)
    else:
        # Priority 2: JAX fallback
        result["warnings"].append(f"torch_xla unavailable: {error}")

        success, mlir_content, error = export_via_jax(output_dir)
        if success:
            result["export_path_used"] = "jax"
            if error:
                result["warnings"].append(error)
        else:
            result["warnings"].append(f"JAX unavailable: {error}")

    # Step 3: Write output if we have MLIR content
    if success and mlir_content:
        mlir_path = os.path.join(output_dir, "model.mlir")
        with open(mlir_path, 'w') as f:
            f.write(mlir_content)
        result["mlir_file"] = "model.mlir"
        result["status"] = "ok"

        # Validate that we have stablehlo ops
        if "stablehlo." not in mlir_content.lower():
            if "mhlo." in mlir_content.lower() or "hlo." in mlir_content.lower():
                result["warnings"].append("Output contains HLO/MHLO ops but not StableHLO ops")
            else:
                result["warnings"].append("Output may not contain StableHLO ops")
    else:
        result["status"] = "failed"
        if not result["error"]:
            result["error"] = "No export path available. Install torch_xla or JAX to enable StableHLO export."
            result["error"] += "\n\nTo install torch_xla: pip install torch_xla"
            result["error"] += "\nTo install JAX: pip install jax jaxlib"

    return result


if __name__ == "__main__":
    import os

    # Default output directory
    output_dir = os.environ.get("SNAKEGRINDER_OUTPUT_DIR", ".")

    if len(sys.argv) > 1:
        output_dir = sys.argv[1]

    os.makedirs(output_dir, exist_ok=True)

    result = export_stablehlo(output_dir)
    print(json.dumps(result, indent=2))
