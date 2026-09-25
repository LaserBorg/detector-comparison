"""Abstract base for a benchmarked runtime executor.

A ``RuntimeExecutor`` does exactly one thing: run a raw forward pass.

  * ``load(artifact)`` where ``artifact`` is either a file path to the model
    (``.onnx`` / ``.engine``) or a ``callable(blob) -> raw`` (a PyTorch runner
    built by the architecture). The executor knows nothing about detector
    families or pre/post — that lives in ``archs/``.
  * ``infer(blob) -> raw``  (blob = ``(1, 3, H, W)`` fp32 CHW; raw = the model's
    pre-NMS output tensor, converted to a numpy array of fp32).

Keeping the executor family-agnostic is what makes it reusable for RF-DETR and
any future architecture with a different pre/post contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class RuntimeExecutor(ABC):
    """Common interface implemented by every backend."""

    name: str = "base"
    on_gpu: bool = False

    def __init__(self, precision: str):
        self.precision = precision
        self._artifact_mb = 0.0

    @abstractmethod
    def load(self, artifact) -> None:
        """Load a model artifact.

        Args:
            artifact: ``Path`` to a model file, or a ``callable(blob) -> raw``
                (PyTorch runner). Implementations inspect the type.
        """

    @abstractmethod
    def infer(self, blob: np.ndarray) -> np.ndarray:
        """Run one forward pass, returning the model's raw output as fp32 numpy."""

    @property
    def artifact_mb(self) -> float:
        """Size on disk / host of the loaded artifact, in megabytes."""
        return self._artifact_mb

    def _track_artifact(self, artifact) -> None:
        """Record the artifact's on-disk size when it's a file path."""
        if isinstance(artifact, (str, Path)):
            p = Path(artifact)
            if p.exists():
                self._artifact_mb = p.stat().st_size / 1e6

    def warmup(self, blob: np.ndarray, n: int = 10) -> None:
        """Discard the first ``n`` inferences to prime allocators / engines."""
        for _ in range(n):
            self.infer(blob)

    def release(self) -> None:
        """Free device resources (override in subclasses)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


# Re-export under the old name for any external caller referencing it.
RuntimeAdapter = RuntimeExecutor


__all__ = ["RuntimeExecutor", "RuntimeAdapter"]