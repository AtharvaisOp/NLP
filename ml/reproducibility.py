"""Reproducibility and runtime provenance helpers."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import shutil


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def runtime_info() -> dict[str, object]:
    packages = {}
    for display_name, distribution_name in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("sklearn", "scikit-learn"),
        ("numpy", "numpy"),
    ):
        try:
            packages[display_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            packages[display_name] = None
    device = "cpu"
    cuda = None
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            cuda = {
                "runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "total_vram_bytes": total_bytes,
                "available_vram_bytes": free_bytes,
            }
    except ImportError:
        pass
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "packages": packages,
        "cuda": cuda,
        "free_disk_bytes": shutil.disk_usage(os.getcwd()).free,
    }
