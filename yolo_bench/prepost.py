"""Compatibility shim for the old ``prepost`` module location.

The shared pre/post has moved to ``yolo_bench/archs/yolo.py`` (so each detector
family owns its own pre/post). This module re-exports the YOLO helpers so any
code or import that still references ``yolo_bench.prepost`` keeps working.
"""

from __future__ import annotations

from .archs.yolo import (
    LetterboxInfo,
    blob_numpy,
    blob_torch,
    letterbox,
    postprocess,
)

__all__ = ["LetterboxInfo", "letterbox", "blob_numpy", "blob_torch", "postprocess"]