"""Detector architectures.

Each ``Architecture`` owns its family's pre/post and artifact resolution, which
keeps ``runtimes/`` and the ``Detector`` wrapper architecture-agnostic. Add a new
detector family (e.g. RF-DETR) by subclassing ``Architecture`` and registering it
here — nothing else changes.
"""

from .base import Architecture
from .rfdetr import RfDetrArchitecture
from .yolo import YoloArchitecture

_REGISTRY = {
    "yolo": YoloArchitecture,
    "rfdetr": RfDetrArchitecture,
}


def create_arch(name: str, model: str) -> Architecture:
    key = name.lower()
    cls = _REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown architecture: {name!r} (supported: {sorted(_REGISTRY)})"
        )
    return cls(model)


def supported_architectures() -> list[str]:
    return list(_REGISTRY)


__all__ = [
    "Architecture",
    "YoloArchitecture",
    "RfDetrArchitecture",
    "create_arch",
    "supported_architectures",
]