"""Native TensorRT runtime."""

from .tensorrt import TensorRTExecutor

def create_runtime(precision: str) -> TensorRTExecutor:
    return TensorRTExecutor(precision)


__all__ = ["TensorRTExecutor", "create_runtime"]