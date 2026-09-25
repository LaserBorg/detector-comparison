"""PyTorch runtime executor.

Runs a checkpoint through PyTorch. Expected artifact is a ``callable(blob) -> raw``
built by the architecture (``Architecture.torch_runner``) — the executor itself
is family-agnostic.
"""

from __future__ import annotations

import numpy as np

from .base import RuntimeExecutor


class PyTorchExecutor(RuntimeExecutor):
    name = "pytorch"
    on_gpu = True

    def __init__(self, precision: str):
        super().__init__(precision)
        self._runner = None

    def load(self, artifact) -> None:
        if not callable(artifact):
            raise TypeError(
                "PyTorchExecutor expects a callable(blob)->raw from "
                "Architecture.torch_runner(), not a file path."
            )
        self._runner = artifact
        self._artifact_mb = 0.0  # checkpoint size is reported by the Detector

    def infer(self, blob: np.ndarray) -> np.ndarray:
        return self._runner(blob)

    def release(self) -> None:
        self._runner = None