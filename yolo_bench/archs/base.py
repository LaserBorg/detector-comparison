"""Architecture base: per-model-family pre/post + artifact resolution.

An ``Architecture`` is the only place that knows a detector family's contract:

  * how to ``prepare`` a BGR frame into the input blob (+ a decode context),
  * how to ``decode`` the raw model output into ``(M, 6) [x1,y1,x2,y2,conf,cls]``,
  * where its artifacts live (``checkpoint_path`` / ``onnx_path`` / ``engine_path``),
  * how to build a PyTorch front-end ``torch_runner`` from the checkpoint.

Keeping this separate from ``runtimes/`` is what lets the ``Detector`` wrapper
swap runtimes transparently **and** lets us add new architectures (e.g. RF-DETR)
with completely different pre/post without touching any runtime or the wrapper.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .. import config
from ..utils import load_meta


class Architecture(ABC):
    """Contract every detector family implements."""

    name: str = "base"
    default_nc: int = 0
    imgsz: int = 640

    def __init__(self, model: str):
        self.model = model
        self.nc = self._resolve_nc(model)

    def _resolve_nc(self, model: str) -> int:
        meta = load_meta(model)
        if meta and "nc" in meta:
            return int(meta["nc"])
        return self.default_nc

    # -- pre / post ----------------------------------------------------------

    @abstractmethod
    def prepare(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, object]:
        """Preprocess a BGR frame -> ``(blob, ctx)``; ctx is fed back to decode."""

    @abstractmethod
    def decode(
        self, raw, ctx, conf: float, iou: float, max_det: int
    ) -> np.ndarray:
        """Raw model output -> ``(M, 6) [x1,y1,x2,y2,conf,cls]`` in image space."""

    # -- artifacts -----------------------------------------------------------

    def checkpoint_path(self, model: str) -> Path:
        return config.model_pt(model)

    def onnx_path(self, model: str) -> Path:
        return config.model_onnx(model)

    def engine_path(self, model: str, precision: str) -> Path:
        return config.model_engine(model, precision)

    # -- PyTorch front-end ---------------------------------------------------

    def torch_runner(self, model: str, device, precision: str):
        """Return a ``callable(blob) -> raw`` running the checkpoint in PyTorch.

        Raises ``NotImplementedError`` if the architecture has no PyTorch
        front-end. The model build lives here (not in the runtime) because
        loading a checkpoint differs per architecture.
        """
        raise NotImplementedError(
            f"Architecture '{self.name}' does not provide a PyTorch runner."
        )