"""Base contract for YOLO preprocessing and postprocessing.

An ``Architecture`` is the only place that knows a detector family's contract:

  * how to ``prepare`` a BGR frame into the input blob (+ a decode context),
  * how to ``decode`` the raw model output into ``(M, 6) [x1,y1,x2,y2,conf,cls]``,
    * where its TensorRT engine lives.
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

    def engine_path(self, model: str, precision: str) -> Path:
        return config.model_engine(model, precision)