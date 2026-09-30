"""Native TensorRT runtime executor (FP32 / FP16 engines).

Loads a prebuilt ``.engine`` file path, deserializes it with the TensorRT
runtime, and executes it directly. Device memory is managed with PyTorch's CUDA
tensors (avoids a hard dependency on pycuda / cuda-python) — we point the
execution context at ``data_ptr()`` addresses and copy the input/output with
torch. This gives the raw, un-wrapped TensorRT path (no ONNX Runtime, no
DeepStream) so it is directly comparable to the ORT-TRT and PyTorch executors.

Uses the **tensor API** (``num_io_tensors`` / ``set_tensor_address`` /
``execute_async_v3``) on TensorRT 10+, falling back to the legacy binding API
(``get_binding_index`` / ``execute_v2``) on TensorRT 8/9 — the binding API was
removed in TensorRT 10, so the tensor path is what runs on modern installs.
Inference is enqueued on a dedicated non-default CUDA stream: on the default
stream TensorRT inserts its own ``cudaStreamSynchronize`` calls, which would
distort the timings this harness measures.

IO tensors are FLOAT for our engines in both precisions (export keeps the I/O
types in FP32; only the internal compute is half precision), so the raw output is
directly comparable across runtimes. FP16 engines are cast back to fp32 to honour
the ``RuntimeExecutor`` contract.
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
        # A dedicated non-default stream: execute_async_v3 on the default stream
        # makes TensorRT add its own cudaStreamSynchronize calls, which would
        # inflate the timings this harness exists to measure.
        self._stream = None
        # Engine tensor metadata (tensor API) or binding indices (legacy API).
        self._tensor_api = False
        self._in_name = None
        self._out_name = None
        self._in_idx = None
        self._out_idx = None
        self._in_shape = None

    def load(self, artifact) -> None:
        import tensorrt as trt
        import torch  # lazy: only the native TRT adapter needs torch tensors

        self._torch = torch
        engine_path = str(artifact)
        self._track_artifact(artifact)
        if not Path(engine_path).exists():
            raise FileNotFoundError(
                f"{engine_path} missing; build it first with trtexec."
            )

        self._device = cuda_device()
        self._stream = torch.cuda.Stream(device=self._device)

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            engine_bytes = _unwrap_ultralytics_engine(f.read())
        self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            model_name = Path(engine_path).name.split(".", 1)[0]
            raise RuntimeError(
                f"Could not deserialize {engine_path}. The engine was built "
                "with an incompatible TensorRT version, GPU, or operating "
                "system. Rebuild it on this machine with:\n"
                f"  python -m yolo_bench.export --model {model_name} "
                f"--precision {self.precision} --force\n"
                f"Installed TensorRT: {trt.__version__}"
            )

        self._context = self._engine.create_execution_context()

        # TensorRT 10 removed the binding API (num_bindings / get_binding_shape /
        # execute_v2); TRT 11 removed the FP16/INT8 flags too. Detect which API
        # this engine exposes so the same code works on TRT 8-11.
        self._tensor_api = hasattr(self._engine, "num_io_tensors")

        dtype_map = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}

        if self._tensor_api:
            names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
            inputs = [n for n in names
                      if self._engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            outputs = [n for n in names
                       if self._engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
            if len(inputs) != 1 or len(outputs) != 1:
                raise RuntimeError(
                    f"Engine must have exactly one input and one output; "
                    f"got inputs={inputs} outputs={outputs}"
                )
            self._in_name, self._out_name = inputs[0], outputs[0]
            self._in_shape = tuple(self._engine.get_tensor_shape(self._in_name))
            out_shape = tuple(self._engine.get_tensor_shape(self._out_name))
            in_dtype = dtype_map.get(self._engine.get_tensor_dtype(self._in_name), torch.float32)
            out_dtype = dtype_map.get(self._engine.get_tensor_dtype(self._out_name), torch.float32)
        else:  # TensorRT <= 9 legacy bindings
            for i in range(self._engine.num_bindings):
                if self._engine.binding_is_input(i):
                    self._in_idx = i
                    self._in_shape = tuple(self._engine.get_binding_shape(i))
                    in_dtype = torch.float32
                else:
                    self._out_idx = i
                    out_shape = tuple(self._engine.get_binding_shape(i))
                    out_dtype = torch.float32
            if self._in_idx is None or self._out_idx is None:
                raise RuntimeError("Engine must have exactly one input and one output binding.")

        # Allocate device buffers via torch (residing on the active CUDA device).
        self._input_t = torch.empty(self._in_shape, dtype=in_dtype, device=self._device)
        self._output_t = torch.empty(out_shape, dtype=out_dtype, device=self._device)
        self._bind_addresses()

    def _bind_addresses(self) -> None:
        """Point the execution context at the allocated device buffers."""
        if self._tensor_api:
            # Static-shape engines need no profile; assert the shape for safety so
            # an unexpected blob is reported clearly rather than silently misfiring.
            self._context.set_input_shape(self._in_name, self._in_shape)
            self._context.set_tensor_address(self._in_name, self._input_t.data_ptr())
            self._context.set_tensor_address(self._out_name, self._output_t.data_ptr())
        else:
            self._bindings = [0] * self._engine.num_bindings
            self._bindings[self._in_idx] = self._input_t.data_ptr()
            self._bindings[self._out_idx] = self._output_t.data_ptr()

    def infer(self, blob: np.ndarray) -> np.ndarray:
        torch = self._torch
        blob = np.ascontiguousarray(blob, dtype=np.float32)

        # Our engines are static (1, 3, IMGSZ, IMGSZ); rebuild buffers only if a
        # caller feeds a different shape.
        if tuple(blob.shape) != tuple(self._in_shape):
            if not self._tensor_api:
                raise RuntimeError(
                    f"Engine expects input {self._in_shape}, got {tuple(blob.shape)} "
                    f"(legacy bindings cannot be reshaped)."
                )
            self._in_shape = tuple(blob.shape)
            self._input_t = torch.empty(self._in_shape, dtype=self._input_t.dtype,
                                        device=self._device)
            self._context.set_input_shape(self._in_name, self._in_shape)
            out_shape = tuple(self._context.get_tensor_shape(self._out_name))
            self._output_t = torch.empty(out_shape, dtype=self._output_t.dtype,
                                         device=self._device)
            self._bind_addresses()

        # copy_ casts to the engine's own IO dtype, so an FP16 engine built with
        # keep_io_types=True (FLOAT) and one with HALF IO both work.
        self._input_t.copy_(torch.from_numpy(blob))

        if self._tensor_api:
            ok = self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)
            self._stream.synchronize()
        else:
            ok = self._context.execute_v2(self._bindings)
        if not ok:
            raise RuntimeError("TensorRT execution failed (engine may not match input shape).")
        # The wrapper's contract is fp32 raw output regardless of engine precision.
        return self._output_t.cpu().numpy().astype(np.float32, copy=False)

    def release(self) -> None:
        torch = self._torch
        self._context = None
        self._engine = None
        self._input_t = self._output_t = None
        self._bindings = []
        self._stream = None
        self._torch = None
        if torch is not None and self._device is not None:
            torch.cuda.empty_cache()

    @property
    def runtime_version(self) -> str | None:
        try:
            import tensorrt as trt

            return trt.__version__
        except Exception:
            return None


def _unwrap_ultralytics_engine(data: bytes) -> bytes:
    """Remove Ultralytics' length-prefixed JSON metadata from an engine.

    Ultralytics writes ``uint32 metadata_length + JSON + raw TensorRT engine``
    to its ``.engine`` files. Native TensorRT expects only the final payload.
    Raw engines pass through unchanged.
    """
    if len(data) < 8 or data[4:5] != b"{":
        return data
    metadata_length = int.from_bytes(data[:4], "little")
    payload_start = 4 + metadata_length
    if data[payload_start:payload_start + 4] == b"ftrt":
        return data[payload_start:]
    return data