"""YOLO11/26 -> ONNX -> TensorRT FP16/FP32 benchmark harness.

The reusable surface is:

    from yolo_bench import Detector
    det = Detector("yolo", "tensorrt", "fp16", "yolo11s")
    det.load()
    boxes = det(frame_bgr)

``Detector`` ties an architecture (detector family) to a runtime backend so
future projects can swap either transparently.
"""

__version__ = "0.2.0"

from . import config  # noqa: F401
from .detector import Detector  # noqa: F401

__all__ = ["Detector", "config"]