"""Runtime registry: the single source of truth for what backends exist.

Why this module exists
----------------------
Runtime knowledge used to be spread over four places: the ``RUNTIMES`` list in
``config.py``, the factory's ``if/elif`` chain in ``runtimes/__init__.py``, an
``if/elif`` chain on artifact type in ``detector.py``, and per-class
``supported_precisions`` attributes. Adding a backend meant touching all four,
and the artifact/device rules were duplicated in two of them.

Now every backend is one ``RuntimeSpec`` record:

  * ``artifact``  — which artifact the architecture must hand it
                    (``"torch_runner"`` = a live PyTorch callable, ``"onnx"``,
                    ``"engine"``). The Architecture owns *where* artifacts live;
                    the spec owns *which* one a backend needs. The Detector just
                    joins the two, so it has no per-runtime branching left.
  * ``precisions`` — the precisions that are genuinely executable. This is the
                    only place precision support is declared.
  * ``group``     — ``"cuda"`` / ``"cpu"`` / ``"other"``, used for grouping rows
                    in reports and for filtering a matrix.
  * ``requires``  — importable module names needed, so the orchestrator can skip
                    unavailable backends instead of crashing a whole run.

The spec does not import the executor classes at module scope (they pull in
torch / tensorrt / openvino); ``factory`` imports lazily on construction.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from typing import Callable

# Artifact kinds a RuntimeSpec can ask an Architecture for.
ARTIFACT_TORCH_RUNNER = "torch_runner"
ARTIFACT_ONNX = "onnx"
ARTIFACT_ENGINE = "engine"


@dataclass(frozen=True)
class RuntimeSpec:
    """Everything the harness needs to know about one backend."""

    kind: str                       # registry key, e.g. "ort_trt"
    label: str                      # human label for reports
    artifact: str                   # ARTIFACT_* constant
    precisions: tuple[str, ...]     # genuinely executable precisions
    group: str                      # "cuda" | "cpu" | "other"
    requires: tuple[str, ...]       # importable module names
    factory: Callable[..., object]  # (precision) -> RuntimeExecutor
    note: str = ""                  # shown in reports / --list-runtimes
    default_device: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def on_gpu(self) -> bool:
        return self.group == "cuda"


# --- lazy factories ---------------------------------------------------------
# Defined as thin functions so importing this module never imports torch/trt.

def _pytorch(precision: str):
    from .pytorch import PyTorchExecutor
    return PyTorchExecutor(precision)


def _pytorch_cpu(precision: str):
    from .pytorch import PyTorchExecutor
    return PyTorchExecutor(precision, force_cpu=True)


def _ort_cuda(precision: str):
    from .ort import OnnxRuntimeCuda
    return OnnxRuntimeCuda(precision)


def _ort_cpu(precision: str):
    from .ort import OnnxRuntimeCpu
    return OnnxRuntimeCpu(precision)


def _ort_trt(precision: str):
    from .ort import OnnxRuntimeTensorrt
    return OnnxRuntimeTensorrt(precision)


def _tensorrt(precision: str):
    from .tensorrt import TensorRTExecutor
    return TensorRTExecutor(precision)


def _openvino_device() -> str:
    """OpenVINO device for the ``openvino`` spec.

    Overridable per-process because OpenVINO exposes different device names on
    different hosts ('GPU', 'GPU.0', 'AUTO', 'MULTI:GPU,CPU', ...) and the right
    choice is a property of the machine, not of the code.
    """
    return os.environ.get("YOLO_BENCH_OPENVINO_DEVICE", "GPU")


def _openvino(precision: str):
    from .openvino import OpenVINOExecutor
    return OpenVINOExecutor(precision, device=_openvino_device())


def _openvino_cpu(precision: str):
    from .openvino import OpenVINOExecutor
    return OpenVINOExecutor(precision, device="CPU")


# --- the registry -----------------------------------------------------------
# Ordered CUDA-first so the interesting rows come first in a report.

RUNTIME_SPECS: tuple[RuntimeSpec, ...] = (
    RuntimeSpec(
        kind="tensorrt",
        label="TensorRT (native)",
        artifact=ARTIFACT_ENGINE,
        precisions=("fp32", "fp16"),
        group="cuda",
        requires=("tensorrt", "torch"),
        factory=_tensorrt,
        note="deserializes a .engine; where the Ampere tensor-core win shows",
    ),
    RuntimeSpec(
        kind="ort_trt",
        label="ONNX Runtime (TensorRT EP)",
        artifact=ARTIFACT_ONNX,
        precisions=("fp32", "fp16"),
        group="cuda",
        requires=("onnxruntime",),
        factory=_ort_trt,
        note="needs a TensorRT major matching the ORT build (see docs/known-issues.md)",
    ),
    RuntimeSpec(
        kind="ort_cuda",
        label="ONNX Runtime (CUDA EP)",
        artifact=ARTIFACT_ONNX,
        precisions=("fp32", "fp16"),
        group="cuda",
        requires=("onnxruntime",),
        factory=_ort_cuda,
        note="fp16 is accepted but gains nothing; read the fp32 row",
    ),
    RuntimeSpec(
        kind="pytorch",
        label="PyTorch (CUDA)",
        artifact=ARTIFACT_TORCH_RUNNER,
        precisions=("fp32", "fp16"),
        group="cuda",
        requires=("torch",),
        factory=_pytorch,
        note="eager; fp16 is *slower* here (latency-bound, see docs/known-issues.md)",
    ),
    RuntimeSpec(
        kind="openvino",
        label="OpenVINO (GPU device)",
        artifact=ARTIFACT_ONNX,
        precisions=("fp32", "fp16"),
        group="cuda",
        requires=("openvino",),
        factory=_openvino,
        default_device=_openvino_device(),
        note="does NOT use TensorRT kernels; ~5x slower on NVIDIA",
    ),
    RuntimeSpec(
        kind="openvino_cpu",
        label="OpenVINO (CPU)",
        artifact=ARTIFACT_ONNX,
        precisions=("fp32",),
        group="cpu",
        requires=("openvino",),
        factory=_openvino_cpu,
        default_device="CPU",
        note="fastest CPU backend measured here",
    ),
    RuntimeSpec(
        kind="ort_cpu",
        label="ONNX Runtime (CPU EP)",
        artifact=ARTIFACT_ONNX,
        precisions=("fp32",),
        group="cpu",
        requires=("onnxruntime",),
        factory=_ort_cpu,
        note="CPU EPs have no FP16 kernels",
    ),
    RuntimeSpec(
        kind="pytorch_cpu",
        label="PyTorch (CPU)",
        artifact=ARTIFACT_TORCH_RUNNER,
        precisions=("fp32",),
        group="cpu",
        requires=("torch",),
        factory=_pytorch_cpu,
        note="slowest CPU backend measured here",
    ),
)

REGISTRY: dict[str, RuntimeSpec] = {s.kind: s for s in RUNTIME_SPECS}

ALL_KINDS: tuple[str, ...] = tuple(REGISTRY)
CUDA_KINDS: tuple[str, ...] = tuple(s.kind for s in RUNTIME_SPECS if s.group == "cuda")
CPU_KINDS: tuple[str, ...] = tuple(s.kind for s in RUNTIME_SPECS if s.group == "cpu")


class UnknownRuntime(ValueError):
    """Raised for a runtime kind that is not registered."""


class UnsupportedPrecision(ValueError):
    """Raised when a (runtime, precision) pair is not executable.

    A distinct type so the orchestrator can treat this as a *configuration* fact
    (skip the combination) rather than as a benchmark failure.
    """


def spec_for(kind: str) -> RuntimeSpec:
    try:
        return REGISTRY[kind.lower()]
    except KeyError:
        raise UnknownRuntime(
            f"Unknown runtime kind: {kind!r} (registered: {', '.join(ALL_KINDS)})"
        ) from None


def importable(kind: str) -> bool:
    """Whether this backend's dependencies are present in the current env.

    Uses ``find_spec`` so it does not pay the import cost (torch alone is
    ~1.5 s) just to answer the question.
    """
    spec = spec_for(kind)
    for mod in spec.requires:
        try:
            if importlib.util.find_spec(mod) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def available_kinds(kinds: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Registered kinds whose dependencies are importable right now."""
    return tuple(k for k in (kinds or ALL_KINDS) if importable(k))


def unavailable_kinds(kinds: tuple[str, ...] | None = None) -> dict[str, str]:
    """``{kind: reason}`` for registered-but-unusable backends.

    Note that dependencies being importable is necessary but not sufficient —
    e.g. ``ort_trt`` imports fine yet its TensorRT EP can still fail to load at
    session creation. The executor asserts that case loudly at load time.
    """
    out: dict[str, str] = {}
    for kind in kinds or ALL_KINDS:
        spec = spec_for(kind)
        missing = [
            m for m in spec.requires
            if importlib.util.find_spec(m) is None
        ]
        if missing:
            out[kind] = f"missing module(s): {', '.join(missing)}"
    return out


def create_runtime(kind: str, precision: str):
    """Instantiate a backend, validating the (runtime, precision) pair.

    This is the only sanctioned construction path, so precision support is
    enforced in exactly one place.
    """
    spec = spec_for(kind)
    if precision not in spec.precisions:
        raise UnsupportedPrecision(
            f"Runtime '{spec.kind}' does not support precision '{precision}' "
            f"(supported: {', '.join(spec.precisions)})."
        )
    return spec.factory(precision)


def plan(
    models: list[str],
    kinds: list[str],
    precisions: list[str],
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str, str]]]:
    """Split a requested matrix into (runnable, skipped-with-reason) combos.

    Returns ``(runnable, skipped)`` where runnable items are
    ``(runtime, precision, model)`` and skipped items add a reason string.
    Lets the orchestrator be explicit about what it is NOT measuring.
    """
    runnable: list[tuple[str, str, str]] = []
    skipped: list[tuple[str, str, str, str]] = []
    for model in models:
        for precision in precisions:
            for kind in kinds:
                try:
                    spec = spec_for(kind)
                except UnknownRuntime as exc:
                    skipped.append((kind, precision, model, str(exc)))
                    continue
                if precision not in spec.precisions:
                    skipped.append((
                        kind, precision, model,
                        f"{spec.kind} supports {', '.join(spec.precisions)} only",
                    ))
                    continue
                if not importable(kind):
                    missing = [
                        m for m in spec.requires
                        if importlib.util.find_spec(m) is None
                    ]
                    skipped.append((
                        kind, precision, model,
                        f"missing module(s): {', '.join(missing)}",
                    ))
                    continue
                runnable.append((kind, precision, model))
    return runnable, skipped


def describe() -> str:
    """Human-readable registry listing for ``--list-runtimes``."""
    lines = [f"{'kind':14s} {'group':5s} {'precisions':12s} {'available':9s} label"]
    for spec in RUNTIME_SPECS:
        ok = "yes" if importable(spec.kind) else "NO"
        lines.append(
            f"{spec.kind:14s} {spec.group:5s} "
            f"{'/'.join(spec.precisions):12s} {ok:9s} {spec.label}"
        )
    lines.append("")
    for spec in RUNTIME_SPECS:
        if spec.note:
            lines.append(f"  {spec.kind:14s} {spec.note}")
    return "\n".join(lines)