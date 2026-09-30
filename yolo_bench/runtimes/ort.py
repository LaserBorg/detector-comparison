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

        # ORT's pip wheel does not automatically expose its bundled NVIDIA
        # libraries to the dynamic linker (notably in conda/WSL2 environments).
        preload_dlls = getattr(ort, "preload_dlls", None)
        if preload_dlls is not None:
            preload_dlls()

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(
            onnx_path,
            sess_options=so,
            providers=self.providers,
            provider_options=self.provider_options,
        )
        self._input_name = sess.get_inputs()[0].name
        # ORT silently falls back to CPU when an EP fails to load (e.g. missing
        # libcudnn/libcublas, or an EP built for a different CUDA major). That
        # would make this row a CPU measurement masquerading as a GPU one, so
        # fail loudly instead.
        self._assert_gpu_provider(sess)
        return sess

    def _assert_gpu_provider(self, sess) -> None:
        """Raise if the intended execution provider did not actually load.

        ORT falls back to the next provider in the list when one fails to load.
        That silently changes what the row measures — a CPU fallback turns a GPU
        row into a CPU one, and an `ort_trt` row into a plain-CUDA one — so fail
        loudly instead of reporting a mislabelled number.
        """
        if not self.on_gpu:
            return
        active = sess.get_providers()
        if self.providers and self.providers[0] not in active:
            fallback = active[0] if active else "none"
            raise RuntimeError(
                f"{self.name}: requested execution provider "
                f"{self.providers[0]!r} did not load and ORT fell back to "
                f"{fallback!r}, so this row would not measure what it claims. "
                "Common causes: (1) for ort_trt, the onnxruntime-gpu build of the "
                "TensorRT EP links against a different TensorRT major than the "
                "one installed (check for a missing libnvinfer.so.N); (2) a "
                "CUDA-mismatched onnxruntime-gpu wheel (e.g. a CUDA 12 build on a "
                "CUDA 13 host); (3) missing libcudnn/libcublas on the loader "
                "path. See the notes in requirements.txt."
            )

    def load(self, artifact) -> None:
        self._track_artifact(artifact)
        self._sess = self._make_session(str(artifact))

    def infer(self, blob: np.ndarray) -> np.ndarray:
        out = self._sess.run(None, {self._input_name: blob})
        return np.asarray(out[0])

    @property
    def runtime_version(self) -> str | None:
        try:
            import onnxruntime as ort

            return ort.__version__
        except Exception:
            return None

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


class OnnxRuntimeCpu(_OrtBase):
    """CPU EP only — the host-RAM baseline.

    Always available (no GPU libraries involved), which also makes it the
    reference point for "what does this cost without any accelerator".
    """

    name = "ort_cpu"
    on_gpu = False
    # ONNX Runtime's CPU EP has no FP16 kernels; an 'fp16' run would execute the
    # FP32 graph and be reported as FP16.
    supported_precisions = ("fp32",)
    providers = ["CPUExecutionProvider"]
    provider_options = [{}]

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