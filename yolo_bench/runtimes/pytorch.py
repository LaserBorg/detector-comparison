"""PyTorch runtime executor.

Runs a checkpoint through PyTorch. Expected artifact is a ``callable(blob) -> raw``
built by the architecture (``Architecture.torch_runner``) — the executor itself
is family-agnostic.

Two runtime kinds are registered: ``pytorch`` (GPU) and ``pytorch_cpu`` (the same
runner forced onto the host CPU) so the comparison can include a pure-CPU
baseline. The CPU variant is a real baseline here: this environment's ``torch``
build is CUDA-enabled but always retains its CPU kernels.
"""

from __future__ import annotations

import numpy as np

from .base import RuntimeExecutor


class PyTorchExecutor(RuntimeExecutor):
    name = "pytorch"
    on_gpu = True

    def __init__(self, precision: str, force_cpu: bool = False):
        # Set capabilities BEFORE the base __init__ validates the precision.
        self.force_cpu = force_cpu
        # `name` drives artifact/device resolution in the Detector, so the CPU
        # variant MUST report a distinct name — otherwise the wrapper matches the
        # GPU branch and silently runs on CUDA.
        self.name = "pytorch_cpu" if force_cpu else "pytorch"
        if force_cpu:
            self.on_gpu = False
            # CPU has no FP16 kernels; net.half() on CPU either errors on the BN
            # layers or runs slower than FP32. Not a meaningful measurement.
            self.supported_precisions = ("fp32",)
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

    @property
    def runtime_version(self) -> str | None:
        try:
            import torch

            return torch.__version__
        except Exception:
            return None

    def release(self) -> None:
        self._runner = None