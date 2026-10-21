"""Explicit runtime inspection and seeding; optional ML imports occur only on request."""

import platform
import random
import sys
from dataclasses import dataclass
from importlib import metadata, util


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    torch_installed: bool


def detect_device(requested: str = "auto") -> DeviceInfo:
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    installed = util.find_spec("torch") is not None
    if requested == "cpu":
        return DeviceInfo("cpu", installed)
    if not installed:
        if requested != "auto":
            raise ValueError(f"{requested} requested but PyTorch is not installed")
        return DeviceInfo("cpu", False)
    import torch

    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    available = {"cpu": True, "cuda": cuda, "mps": mps}
    if requested != "auto" and not available[requested]:
        raise ValueError(f"requested device {requested} is unavailable")
    return DeviceInfo(
        ("cuda" if cuda else "mps" if mps else "cpu") if requested == "auto" else requested, True
    )


def set_random_seeds(
    seed: int, *, include_optional: bool = False, deterministic: bool = True
) -> dict:
    """Seed legacy global RNGs explicitly; imports alone never alter global RNG state."""
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    random.seed(seed)
    applied = ["python.random"]
    if include_optional and util.find_spec("numpy") is not None:
        import numpy as np

        np.random.seed(seed)
        applied.append("numpy.random (legacy)")
    if include_optional and util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = not deterministic
        applied.append("torch")
    return {
        "seed": seed,
        "applied_to": applied,
        "deterministic_algorithms_requested": deterministic,
        "limitations": (
            "GPU operations and different hardware/library versions may be nondeterministic. "
            "Seed NumPy Generator instances separately; set PYTHONHASHSEED before process startup."
        ),
    }


def check_environment(device: str = "auto") -> dict:
    required = ("pydantic", "PyYAML")
    optional = (
        "numpy",
        "pandas",
        "scikit-learn",
        "networkx",
        "torch",
        "torch-geometric",
        "transformers",
        "sentence-transformers",
        "bertopic",
        "neo4j",
        "fastapi",
        "pytest",
        "ruff",
    )
    versions = {}
    for name in required + optional:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    selected = detect_device(device)
    supported = sys.version_info[:2] == (3, 11)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "python_supported": supported,
        "ready": supported and all(versions[name] is not None for name in required),
        "device": selected.device,
        "torch_installed": selected.torch_installed,
        "packages": versions,
        "note": "Optional packages are not required for Stage 01. No GPU determinism guarantee.",
    }
