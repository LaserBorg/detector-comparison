"""Runtime executors: PyTorch / ONNX Runtime / TensorRT / OpenVINO.

The set of backends and their capabilities live in :mod:`.registry` — this module
just re-exports it so callers have one obvious import point:

    from yolo_bench.runtimes import create_runtime, spec_for, ALL_KINDS

Executors are imported lazily by the registry's factories, so importing this
package does not drag in torch / tensorrt / openvino and stays cheap on a machine
that only has a subset installed.
"""

from .base import RuntimeAdapter, RuntimeExecutor
from .registry import (
    ALL_KINDS,
    ARTIFACT_ENGINE,
    ARTIFACT_ONNX,
    ARTIFACT_TORCH_RUNNER,
    CPU_KINDS,
    CUDA_KINDS,
    REGISTRY,
    RUNTIME_SPECS,
    RuntimeSpec,
    UnknownRuntime,
    UnsupportedPrecision,
    available_kinds,
    create_runtime,
    describe,
    importable,
    plan,
    spec_for,
    unavailable_kinds,
)

__all__ = [
    "RuntimeAdapter", "RuntimeExecutor",
    "RuntimeSpec", "RUNTIME_SPECS", "REGISTRY",
    "ALL_KINDS", "CUDA_KINDS", "CPU_KINDS",
    "ARTIFACT_TORCH_RUNNER", "ARTIFACT_ONNX", "ARTIFACT_ENGINE",
    "UnknownRuntime", "UnsupportedPrecision",
    "create_runtime", "spec_for", "importable", "available_kinds",
    "unavailable_kinds", "plan", "describe",
]