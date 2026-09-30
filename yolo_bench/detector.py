"""YOLO detector wrapper for a native TensorRT engine.

This is the piece meant to be imported by upcoming projects. It exposes a
single ``detect(frame)`` call over a prebuilt YOLO TensorRT engine.

Output is ``(M, 6)`` float32 ``[x1, y1, x2, y2, conf, cls]`` in original pixels.

Usage
-----
    from yolo_bench import Detector

    det = Detector("yolo11s", "fp16")
    det.load()
    boxes = det(frame_bgr)
"""

from __future__ import annotations

import time
import numpy as np

from . import config
from .archs import create_arch
from .runtimes import create_runtime


class Detector:
    def __init__(
        self,
        model: str = config.DEFAULT_MODEL,
        precision: str = config.DEFAULT_PRECISION,
        *,
        conf: float = config.CONF,
        iou: float = config.IOU,
        max_det: int = config.MAX_DET,
    ):
        self._arch = create_arch("yolo", model)
        self._rt = create_runtime(precision)
        self.precision = precision
        self.model = model
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self._weights_mb = 0.0
        self._loaded = False

    # -- identity ------------------------------------------------------------

    @property
    def architecture(self) -> str:
        return self._arch.name

    @property
    def runtime(self) -> str:
        return self._rt.name

    @property
    def nc(self) -> int:
        return self._arch.nc

    @property
    def imgsz(self) -> int:
        return self._arch.imgsz

    @property
    def weights_mb(self) -> float:
        return self._weights_mb

    @property
    def runtime_version(self) -> str | None:
        """Version of the library performing the forward pass (per-runtime)."""
        return self._rt.runtime_version

    # -- lifecycle -----------------------------------------------------------

    def load(self) -> None:
        """Resolve the architecture-appropriate artifact and hand it to the runtime.

        The registry spec says *which kind* of artifact this backend needs; the
        architecture says *where* that artifact lives. Joining the two is all the
        wrapper has to do, so there is no per-runtime branching here.
        """
        artifact = self._arch.engine_path(self.model, self.precision)
        self._rt.load(artifact)
        if artifact.exists():
            self._weights_mb = artifact.stat().st_size / 1e6
        self._loaded = True

    def warmup(self, frame_bgr: np.ndarray, n: int = 10) -> None:
        blob, _ = self._arch.prepare(frame_bgr)
        for _ in range(n):
            self._rt.infer(blob)

    def run(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
        """Full detect + split timings. Returns ``(dets, (pre_s, infer_s, post_s))``."""
        t0 = time.perf_counter()
        blob, ctx = self._arch.prepare(frame_bgr)
        t1 = time.perf_counter()
        raw = self._rt.infer(blob)
        t2 = time.perf_counter()
        dets = self._arch.decode(raw, ctx, self.conf, self.iou, self.max_det)
        t3 = time.perf_counter()
        return dets, (t1 - t0, t2 - t1, t3 - t2)

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        dets, _ = self.run(frame_bgr)
        return dets

    def release(self) -> None:
        self._rt.release()
        self._loaded = False

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *exc):
        self.release()
        return False