"""YOLO11 TensorRT inference."""

__version__ = "1.0.0"

from . import config  # noqa: F401
from .detector import Detector

__all__ = [
    "Detector",
    "config",
]