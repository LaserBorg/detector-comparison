"""Runtime executors: PyTorch / ONNX Runtime (CUDA + TensorRT) / native TensorRT.

Import lazily so the package stays importable on a machine with a subset of the
backends installed (e.g. no TensorRT on the GTX 960M, no PyTorch on a minimal
Jetson image).
"""

from .base import RuntimeAdapter, RuntimeExecutor


def create_runtime(kind: str, precision: str) -> RuntimeExecutor:
    """Factory for a runtime executor.

    Args:
        kind: one of 'pytorch' | 'ort_cuda' | 'ort_trt' | 'tensorrt'
        precision: 'fp32' | 'fp16'
    """
    kind = kind.lower()
    try:
        if kind == "pytorch":
            from .pytorch import PyTorchExecutor
            return PyTorchExecutor(precision)
        if kind == "ort_cuda":
            from .ort import OnnxRuntimeCuda
            return OnnxRuntimeCuda(precision)
        if kind == "ort_trt":
            from .ort import OnnxRuntimeTensorrt
            return OnnxRuntimeTensorrt(precision)
        if kind == "tensorrt":
            from .tensorrt import TensorRTExecutor
            return TensorRTExecutor(precision)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"Runtime '{kind}' requires an optional dependency that is not "
            f"installed in this environment: {exc}"
        ) from exc
    raise ValueError(f"Unknown runtime kind: {kind!r} (expected one of "
                     f"pytorch, ort_cuda, ort_trt, tensorrt)")


__all__ = ["RuntimeExecutor", "RuntimeAdapter", "create_runtime"]