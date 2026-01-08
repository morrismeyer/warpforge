# SnakeGrinder capability probe script.
# Detects available ML frameworks and export paths.
# Outputs JSON to stdout. Never raises on import failures.

import sys
import platform
import json


def probe():
    """
    Probe the Python environment for ML framework capabilities.
    Returns a dict with structured information about available tools.
    """
    result = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_import_ok": False,
        "torch_version": None,
        "torch_cuda_available": False,
        "torch_mps_available": False,
        "torch_error": None,
        "torch_xla_import_ok": False,
        "torch_xla_version": None,
        "torch_xla_error": None,
        "jax_import_ok": False,
        "jax_version": None,
        "jax_error": None,
        "export_path": None,
        "export_path_reason": None,
    }

    # Probe PyTorch
    try:
        import torch
        result["torch_import_ok"] = True
        result["torch_version"] = torch.__version__
        result["torch_cuda_available"] = torch.cuda.is_available()
        # MPS is macOS Metal Performance Shaders
        result["torch_mps_available"] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    except ImportError as e:
        result["torch_error"] = str(e)
    except Exception as e:
        result["torch_error"] = f"Unexpected error: {e}"

    # Probe torch_xla (for StableHLO export)
    try:
        import torch_xla
        result["torch_xla_import_ok"] = True
        result["torch_xla_version"] = getattr(torch_xla, '__version__', 'unknown')
    except ImportError as e:
        result["torch_xla_error"] = str(e)
    except Exception as e:
        result["torch_xla_error"] = f"Unexpected error: {e}"

    # Probe JAX (fallback export path)
    try:
        import jax
        result["jax_import_ok"] = True
        result["jax_version"] = jax.__version__
    except ImportError as e:
        result["jax_error"] = str(e)
    except Exception as e:
        result["jax_error"] = f"Unexpected error: {e}"

    # Determine export path
    if result["torch_xla_import_ok"]:
        result["export_path"] = "torch_xla"
        result["export_path_reason"] = "torch_xla available, can export directly to StableHLO"
    elif result["jax_import_ok"]:
        result["export_path"] = "jax"
        result["export_path_reason"] = "JAX available as fallback for StableHLO export"
    elif result["torch_import_ok"]:
        result["export_path"] = "torch_only"
        result["export_path_reason"] = "PyTorch available but no StableHLO export path (missing torch_xla or JAX)"
    else:
        result["export_path"] = "none"
        result["export_path_reason"] = "No ML frameworks available"

    return result


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2))
