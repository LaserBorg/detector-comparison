"""Small helpers used by the webcam inference path."""

from __future__ import annotations

import json

try:
    import torch
except Exception:  # pragma: no cover - allows syntax-only checks without torch
    torch = None

from . import config


def load_meta(name: str) -> dict | None:
    path = config.model_meta(name)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def class_names(nc: int) -> list[str]:
    if nc == 80:
        return config.COCO80
    return [f"class-{index}" for index in range(nc)]


def cuda_device():
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return torch.device("cuda:0")
