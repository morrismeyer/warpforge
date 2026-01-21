# save as: list_mps_ops.py
import json
import torch

def aten_overload_names():
    aten = torch.ops.aten
    for packet_name in dir(aten):
        packet = getattr(aten, packet_name)
        if not hasattr(packet, "overloads"):
            continue
        try:
            overloads = packet.overloads()
        except Exception:
            continue
        for ov in overloads:
            try:
                op = getattr(packet, ov)
            except Exception:
                continue
            # OpOverload has name()
            if hasattr(op, "name"):
                try:
                    yield op.name()
                except Exception:
                    pass

def has_kernel(op_name: str, dispatch_key: str) -> bool:
    # Using the low-level dispatcher query is the most direct way.
    # Some builds may restrict this; treat exceptions as "unknown/false".
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(op_name, dispatch_key)
    except Exception:
        return False

def main():
    supported = []
    missing = []

    for op_name in sorted(set(aten_overload_names())):
        # Many forward kernels register under "MPS".
        # Backward kernels can register under "AutogradMPS".
        mps = has_kernel(op_name, "MPS")
        autograd_mps = has_kernel(op_name, "AutogradMPS")

        if mps or autograd_mps:
            supported.append({"op": op_name, "MPS": mps, "AutogradMPS": autograd_mps})
        else:
            missing.append(op_name)

    out = {
        "torch_version": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "supported": supported,
        "missing": missing,
        "counts": {"supported": len(supported), "missing": len(missing)},
    }

    with open("mps_ops_report.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote mps_ops_report.json")
    print(f"Supported: {len(supported)}  Missing: {len(missing)}")

if __name__ == "__main__":
    main()
