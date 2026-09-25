"""ONNX Runtime executors: CUDA execution provider, and TensorRT execution provider.

Both load a ``.onnx`` file path and run a raw forward pass. Family-agnostic.
"""

from __future__ import annotations

import numpy as np

from .. import config
from .base import RuntimeExecutor


class _OrtBase(RuntimeExecutor):
    """Shared ORT session logic."""

    providers: list[str] = []
    provider_options: list[dict] = []

    def __init__(self, precision: str):
        super().__init__(precision)
        self._sess = None
        self._input_name = None

    def _make_session(self, onnx_path):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(
            onnx_path,
            sess_options=so,
            providers=self.providers,
            provider_options=self.provider_options,
        )
        self._input_name = sess.get_inputs()[0].name
        return sess

    def load(self, artifact) -> None:
        self._track_artifact(artifact)
        self._sess = self._make_session(str(artifact))

    def infer(self, blob: np.ndarray) -> np.ndarray:
        out = self._sess.run(None, {self._input_name: blob})
        return np.asarray(out[0])

    def release(self) -> None:
        self._sess = None


class OnnxRuntimeCuda(_OrtBase):
    """Plain CUDA EP (FP32)."""

    name = "ort_cuda"
    on_gpu = True
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    provider_options = [{"cudnn_conv_algo_search": "EXHAUSTIVE"}, {}]

    def __init__(self, precision: str):
        super().__init__(precision)


class OnnxRuntimeTensorrt(_OrtBase):
    """TensorRT EP. FP16 is enabled via trt_fp16_enable; FP32 otherwise.

    The first warmup run triggers the TRT engine build (slow); enable engine
    caching so subsequent sessions on the same machine load the cached engine.
    """

    name = "ort_trt"
    on_gpu = True
    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider",
                 "CPUExecutionProvider"]

    def __init__(self, precision: str):
        super().__init__(precision)
        fp16 = precision == "fp16"
        self.provider_options = [
            {
                "trt_fp16_enable": 1 if fp16 else 0,
                "trt_engine_cache_enable": 1,
                "trt_engine_cache_path": str(config.MODELS_DIR / "ort_trt_cache"),
                "device_id": 0,
            },
            {},
            {},
        ]