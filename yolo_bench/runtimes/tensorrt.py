"""Native TensorRT runtime executor (FP32 / FP16 engines).

Loads a prebuilt ``.engine`` file path, deserializes it with the TensorRT
runtime, and executes it directly. Device memory is managed with PyTorch's CUDA
tensors (avoids a hard dependency on pycuda / cuda-python) — we point the
execution context at ``data_ptr()`` addresses and copy the input/output with
torch. This gives the raw, un-wrapped TensorRT path (no ONNX Runtime, no
DeepStream) so it is directly comparable to the ORT-TRT and PyTorch executors.

Uses the classic binding API (``get_binding_index`` / ``execute_v2``) because it is
consistent across TensorRT 8.x through 10.x. IO tensors stay FP32 even for FP16
engines (trtexec keeps IO at FP32 unless told otherwise); only the internal
compute is half precision.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils import cuda_device
from .base import RuntimeExecutor


class TensorRTExecutor(RuntimeExecutor):
    name = "tensorrt"
    on_gpu = True

    def __init__(self, precision: str):
        super().__init__(precision)
        self._engine = None
        self._context = None
        self._device = None
        self._input_t = None
        self._output_t = None
        self._bindings = []
        self._torch = None

    def load(self, artifact) -> None:
        import tensorrt as trt
        import torch  # lazy: only the native TRT adapter needs torch tensors

        self._torch = torch
        engine_path = str(artifact)
        self._track_artifact(artifact)
        if not Path(engine_path).exists():
            raise FileNotFoundError(
                f"{engine_path} missing; build it first with "
                f"`python -m yolo_bench.export --onnx-only --precisions "
                f"{self.precision}`."
            )

        self._device = cuda_device()

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()

        # Discover input/output binding indices.
        in_idx = out_idx = None
        for i in range(self._engine.num_bindings):
            if self._engine.binding_is_input(i):
                in_idx = i
                in_shape = tuple(self._engine.get_binding_shape(i))
            else:
                out_idx = i
                out_shape = tuple(self._engine.get_binding_shape(i))

        if in_idx is None or out_idx is None:
            raise RuntimeError("Engine must have exactly one input and one output binding.")

        # Allocate device buffers via torch (residing on the active CUDA device).
        self._input_t = torch.empty(in_shape, dtype=torch.float32, device=self._device)
        self._output_t = torch.empty(out_shape, dtype=torch.float32, device=self._device)

        # Bindings must be ordered by binding index.
        self._bindings = [0] * self._engine.num_bindings
        self._bindings[in_idx] = self._input_t.data_ptr()
        self._bindings[out_idx] = self._output_t.data_ptr()

    def infer(self, blob: np.ndarray) -> np.ndarray:
        torch = self._torch
        self._input_t.copy_(torch.from_numpy(blob))
        ok = self._context.execute_v2(self._bindings)
        if not ok:
            raise RuntimeError("TensorRT execute_v2 failed (engine may not match input shape).")
        # Synchronize so the output tensor is ready before we copy it out.
        torch.cuda.synchronize(self._device)
        return self._output_t.cpu().numpy()

    def release(self) -> None:
        torch = self._torch
        self._context = None
        self._engine = None
        self._input_t = self._output_t = None
        self._bindings = []
        self._torch = None
        if torch is not None and self._device is not None:
            torch.cuda.empty_cache()