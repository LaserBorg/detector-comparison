"""OpenVINO runtime executor (Intel's inference runtime).

OpenVINO imports ONNX natively, so the artifact is the same ``{name}.onnx`` the
ORT runtimes use — no extra export step. The ONNX -> IR graph conversion happens
inside ``load()``, i.e. outside the timed loop, so the timings stay comparable to
the other runtimes.

Target device comes from the ``device`` argument (the registry passes
``YOLO_BENCH_OPENVINO_DEVICE``, default ``GPU``). Two runtime kinds are registered
so both can appear in one comparison: ``openvino`` (GPU) and ``openvino_cpu``
(CPU).

``precision`` maps onto OpenVINO's ``INFERENCE_PRECISION_HINT`` (``fp32`` ->
``f32``, ``fp16`` -> ``f16``).

Caveat: OpenVINO's *NVIDIA* GPU plugin is a relatively recent addition. It runs
this workload correctly on Ampere, but far slower than the CUDA-backed runtimes
(measured ~28 ms vs TensorRT's ~2.7 ms infer for ``yolo11s``), because it does not
go through the TensorRT kernels. Treat it as an extra data point showing what a
vendor-neutral runtime achieves on NVIDIA hardware, not as a TensorRT competitor.
The CPU plugin is also exposed because CPU-only inference is the real baseline for
edge devices (e.g. a Jetson running without its GPU).
"""

from __future__ import annotations

import os

import numpy as np

from .base import RuntimeExecutor

# One Core per process. Creating a second ``ov.Core()`` and compiling on it
# segfaults in OpenVINO 2026.4 (it happens on the *second* executor regardless of
# device), which breaks a multi-(runtime) run such as ``compare``. A single
# shared Core compiles fine on 'GPU' and 'CPU' in any order.
_CORE = None


def _get_core():
    global _CORE
    if _CORE is None:
        import openvino as ov

        _CORE = ov.Core()
    return _CORE


class OpenVINOExecutor(RuntimeExecutor):
    """OpenVINO compiled-model executor."""

    name = "openvino"

    def __init__(self, precision: str, device: str | None = None):
        # Set capabilities BEFORE the base __init__ validates the precision.
        # Device comes from the caller (registry) or the env, because the right
        # device name is a property of the host, not of the code.
        self.device_name = (
            device or os.environ.get("YOLO_BENCH_OPENVINO_DEVICE", "GPU")
        ).upper()
        self.on_gpu = not self.device_name.startswith("CPU")
        if not self.on_gpu:
            # The CPU plugin has no native FP16; an f16 request would just run
            # FP32 and be mislabelled.
            self.supported_precisions = ("fp32",)
            # Distinct name so errors/rows say 'openvino_cpu', not 'openvino'.
            self.name = "openvino_cpu"
        super().__init__(precision)
        self._compiled = None
        self._output = None

    def load(self, artifact) -> None:
        self._track_artifact(artifact)

        core = _get_core()

        # NOTE: do NOT call core.set_property({'LOG_LEVEL': ...}) here. Setting a
        # core-level log level suppresses the GPU plugin's initialisation, so
        # available_devices silently drops from ['CPU', 'GPU'] to ['CPU'] and the
        # GPU build fails. The ONNX frontend's "N warnings generated" chatter is
        # cosmetic; leave it alone rather than lose the device.
        available = list(core.available_devices)
        if self.device_name not in available:
            raise RuntimeError(
                f"OpenVINO device {self.device_name!r} is not available in this "
                f"install; it exposes {available}. Set the OpenVINO device via "
                f"YOLO_BENCH_OPENVINO_DEVICE (or pass device=) to one of those."
            )

        model = core.read_model(str(artifact))

        # Only ask for an f16 hint on accelerator-style devices: the CPU plugin
        # has no native fp16, so forcing the hint there is wrong.
        cfg: dict = {}
        if self.on_gpu:
            cfg["INFERENCE_PRECISION_HINT"] = "f16" if self.precision == "fp16" else "f32"

        self._compiled = core.compile_model(model, self.device_name, cfg)
        self._output = self._compiled.output(0)

    def infer(self, blob: np.ndarray) -> np.ndarray:
        res = self._compiled(blob)
        # Copy: OpenVINO may return a view into a reusable output buffer, while
        # ONNX Runtime allocates fresh arrays per run. Copy so neither runtime
        # gets an unfair advantage from aliasing the previous result.
        return np.array(res[self._output], dtype=np.float32, copy=True)

    @property
    def runtime_version(self) -> str | None:
        try:
            import openvino as ov

            return ov.__version__
        except Exception:
            return None

    def release(self) -> None:
        self._compiled = None
        self._output = None