"""Reusable detector inference stack + benchmark harness.

Two layers, deliberately separate:

**Inference (import this in other projects)** — no CLI, no orchestration::

    from yolo_bench import Detector

    det = Detector("yolo", "tensorrt", "fp16", "yolo11s")
    det.load()
    boxes = det(frame_bgr)               # (M, 6) [x1,y1,x2,y2,conf,cls]
    det.release()

``Detector`` ties an *architecture* (detector family: pre/post + artifact
resolution) to a *runtime* (raw forward pass). Both are swappable, and the
runtime registry in ``yolo_bench.runtimes`` is the single source of truth for
which backends exist and which precisions they support.

**Harness (orchestration)** — a separate namespace so importing the detector does
not drag it in::

    from yolo_bench import orchestrate

    rows = orchestrate.run(runtimes=["tensorrt", "ort_cuda"],
                           precisions=["fp32", "fp16"], frames=100)

Every configuration runs in its own subprocess there, because peak memory is
meaningless when many configs share one interpreter.
"""

__version__ = "0.3.0"

from . import config  # noqa: F401
from . import orchestrate  # noqa: F401
from .detector import Detector  # noqa: F401
from .runtimes import ALL_KINDS, CUDA_KINDS, CPU_KINDS, create_runtime  # noqa: F401

__all__ = [
    "Detector",
    "config",
    "orchestrate",
    "create_runtime",
    "ALL_KINDS",
    "CUDA_KINDS",
    "CPU_KINDS",
]