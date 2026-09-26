"""Reusable detector wrapper: transparent runtime swap + architecture plugin.

This is the piece meant to be imported by *upcoming projects*. It ties a detector
family (an ``Architecture``) to a backend (a ``RuntimeExecutor``) and exposes a
single ``detect(frame)`` call. Swapping backends is a constructor argument;
swapping architectures (e.g. YOLO -> RF-DETR) needs only the new ``Architecture``
— no runtime or wrapper changes.

Standardized output for every architecture/runtime combination:
``(M, 6)`` float32 ``[x1, y1, x2, y2, conf, cls]`` in ORIGINAL image pixel space.

Usage
-----
    from yolo_bench import Detector

    det = Detector("yolo", "tensorrt", "fp16", "yolo11s")
    det.load()
    boxes = det(frame_bgr)                              # full pre->forward->post
    boxes, (pre, infer, post) = det.run(frame_bgr)      # + per-stage timings
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import config
from .archs import create_arch
from .runtimes import (
    ARTIFACT_ENGINE,
    ARTIFACT_ONNX,
    ARTIFACT_TORCH_RUNNER,
    create_runtime,
    spec_for,
)
from .utils import torch_device


class Detector:
    def __init__(
        self,
        architecture: str,
        runtime: str,
        precision: str,
        model: str,
        *,
        device=None,
        conf: float = config.CONF,
        iou: float = config.IOU,
        max_det: int = config.MAX_DET,
    ):
        self._arch = create_arch(architecture, model)
        self._rt = create_runtime(runtime, precision)
        self.precision = precision
        self.model = model
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self._device = device
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
        if self._device is None:
            self._device = torch_device()

        kind = self._rt.name
        spec = spec_for(kind)

        if spec.artifact == ARTIFACT_TORCH_RUNNER:
            # CPU backends must get a CPU device, or the runner would move the
            # tensors to CUDA while reporting itself as a CPU baseline.
            import torch  # lazy: keeps this module importable without PyTorch

            device = self._device if spec.on_gpu else torch.device("cpu")
            artifact = self._arch.torch_runner(self.model, device, self.precision)
            weights_path = self._arch.checkpoint_path(self.model)
        elif spec.artifact == ARTIFACT_ONNX:
            artifact = self._arch.onnx_path(self.model)
            weights_path = artifact
        elif spec.artifact == ARTIFACT_ENGINE:
            artifact = self._arch.engine_path(self.model, self.precision)
            weights_path = artifact
        else:  # pragma: no cover - registry guarantees the set
            raise ValueError(
                f"Runtime '{kind}' declares unknown artifact kind {spec.artifact!r}"
            )

        self._rt.load(artifact)
        if isinstance(weights_path, Path) and weights_path.exists():
            self._weights_mb = weights_path.stat().st_size / 1e6
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