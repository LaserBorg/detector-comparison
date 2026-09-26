"""Model conversion utility: Ultralytics .pt -> ONNX -> TensorRT engines.

Pipeline
--------
  * ``.pt``  -> ``.onnx``   via ``ultralytics`` export (imgsz 640, opset, simplify)
  * ``.onnx`` -> ``.engine`` via the **TensorRT Python API** (``trt.Builder``),
    falling back to ``trtexec`` only if the bindings are unavailable
  * metadata (``nc``, input shape, arch, TRT version) written to ``.meta.json``

Why the Python API instead of ``trtexec``?
------------------------------------------
The ``tensorrt`` wheel on PyPI ships **no binaries** — there is no ``trtexec`` in
it — so on a plain pip install (as on the RTX 3090 host) ``trtexec`` simply does
not exist. ``trtexec`` is only present where a full TensorRT install provides it
(e.g. JetPack on Jetson, or the NVIDIA tarball). The Python builder is always
available and is what ``ultralytics`` itself uses.

FP16 on TensorRT >= 11
----------------------
TensorRT 11 is **strongly-typed only**: it removed ``BuilderFlag.FP16`` /
``BuilderFlag.INT8`` and the ``IInt8Calibrator`` interface, and ``ITensor``
precision is read-only, so there is no API to set per-layer precision. Reduced
precision must therefore be baked into the **ONNX graph** before parsing. We use
NVIDIA **ModelOpt AutoCast** (mixed precision, FP32 I/O) for FP16, which is
exactly the mechanism ``ultralytics`` uses. On TensorRT 7-10 the classic
``BuilderFlag.FP16`` route is still used.

Usage
-----
  python -m yolo_bench.export --models yolo11s yolo11l yolo26s yolo26l \
      --precisions fp32 fp16 [--tf32-off] [--cross-check]

  # convert only ONNX->engine (skip the .pt->.onnx step, reuse existing onnx):
  python -m yolo_bench.export --onnx-only --models yolo11s --precisions fp16
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import config
from .utils import detect_arch, ensure_dirs, nvidia_gpu_name

# Complexity captured during the .pt -> .onnx step, keyed by model name. Held in
# module state so `write_meta` can fold it into the metadata without re-loading
# the checkpoint (which needs ultralytics, a [convert]-tier dependency).
GFLOPs_PARAMS: dict[str, dict] = {}


def _measure_complexity(pt_path: Path, imgsz: int) -> dict:
    """(params, GFLOPs) for a checkpoint. Returns {} if it cannot be measured.

    GFLOPs is measured by tracing a 640x640 input through the checkpoint, which
    is what makes an "accuracy vs compute cost" plot possible. It is deliberately
    best-effort: an export should not fail because a FLOPs counter is missing.
    """
    try:
        import torch
        from ultralytics import YOLO

        net = YOLO(str(pt_path)).model
        params = sum(p.numel() for p in net.parameters())
        gflops = None
        try:
            from ultralytics.utils.torch_utils import get_flops

            gflops = float(get_flops(net, imgsz=imgsz))
        except Exception:
            # Fallback: count multiply-accumulates with a thop-style pass.
            try:
                with torch.no_grad():
                    from ultralytics.utils.torch_utils import profile

                    _, gflops = profile(net, imgsz=imgsz)
            except Exception:
                gflops = None
        return {"params_m": round(params / 1e6, 2),
                "gflops": None if gflops is None else round(float(gflops), 2)}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# .pt -> .onnx
# ---------------------------------------------------------------------------

def export_onnx(name: str, tf32_off: bool = False) -> Path:
    """Export ``models/{name}.pt`` to ``models/{name}.onnx``.

    Returns the onnx path. Raises if export fails (e.g. no PyTorch/GPU).
    """
    dst = config.model_onnx(name)
    src = config.model_pt(name)
    if not src.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {src}. Place it in {config.MODELS_DIR} first."
        )

    import torch  # local import: keeps the module importable without torch
    from ultralytics import YOLO

    # Strict FP32 baseline: TF32 silently changes FP32 math on Ampere; disable
    # it for the .pt->.onnx graph so FP32 means genuinely 32-bit.
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    if tf32_off and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Record model complexity while the checkpoint is loaded. Needed to plot
    # performance against model size/cost ("accuracy vs GFLOPs"), and it can only
    # be measured from the .pt, not from the compiled graph. Best-effort: a
    # failure here must not abort an otherwise good export.
    GFLOPs_PARAMS[name] = _measure_complexity(src, config.IMGSZ)

    try:
        model = YOLO(str(src))
        model.export(
            format="onnx",
            imgsz=config.IMGSZ,
            opset=config.OPSET,
            dynamic=False,
            simplify=True,
        )
    finally:
        if tf32_off and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    # Ultralytics writes next to the source checkpoint; move into models/.
    produced = src.with_suffix(".onnx")
    if produced != dst:
        shutil.move(str(produced), str(dst))
    return dst


def _read_nc_from_onnx(onnx_path: Path) -> tuple[int, list[int]]:
    """Return (nc, input_shape) for a detected model's ONNX graph.

    Input shape is (N, 3, H, W); output is (N, 4+nc, num_pred).
    """
    import onnx

    model = onnx.load(str(onnx_path))
    graph = model.graph

    out = graph.output[0]
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    if len(shape) != 3 or shape[0] != 1:
        raise ValueError(f"Unexpected output shape {shape}; expected (1, 4+nc, N)")

    in_shape = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    nc = shape[1] - 4
    return nc, in_shape


def write_meta(name: str, nc: int, input_shape: list[int], **extra) -> Path:
    meta = {
        "name": name,
        "nc": nc,
        "imgsz": config.IMGSZ,
        "input_shape": input_shape,
        "arch": detect_arch(),
        "gpu": nvidia_gpu_name(),
        # Complexity from this run if we just exported; otherwise preserve
        # whatever a previous export (or `meta` command) recorded, so re-running
        # only the ONNX->engine step does not wipe it.
        **GFLOPs_PARAMS.get(name, _existing_complexity(name)),
        **extra,
    }
    path = config.model_meta(name)
    path.write_text(json.dumps(meta, indent=2))
    return path


def _existing_complexity(name: str) -> dict:
    """Params/GFLOPs already recorded in a model's metadata, if any."""
    path = config.model_meta(name)
    if not path.exists():
        return {}
    try:
        old = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    keep = {k: old[k] for k in ("params_m", "gflops") if k in old}
    return keep


# ---------------------------------------------------------------------------
# .onnx -> .engine
# ---------------------------------------------------------------------------

def _trt_major() -> int | None:
    """Major version of the installed TensorRT, or None if not importable."""
    try:
        import tensorrt as trt
        return int(str(trt.__version__).split(".", 1)[0])
    except Exception:
        return None


def _find_trtexec() -> str | None:
    """Path to ``trtexec`` if one is available, else None.

    The pip ``tensorrt`` wheel does NOT ship ``trtexec``; it only exists with a
    full TensorRT install (JetPack on Jetson, or the NVIDIA tarball). Returning
    None makes the caller use the TensorRT Python API instead.
    """
    exe = shutil.which("trtexec")
    if exe:
        return exe
    for cand in (
        "/usr/src/tensorrt/bin/trtexec",   # JetPack / Jetson
        "/opt/tensorrt/bin/trtexec",
    ):
        if Path(cand).exists():
            return cand
    return None


def _trt_version() -> str | None:
    try:
        import tensorrt as trt
        return trt.__version__
    except Exception:
        return None


def _fp16_onnx(onnx_path: Path) -> Path:
    """Bake FP16 mixed precision into ``onnx_path`` and return the new path.

    Required for TensorRT >= 11, which is strongly-typed only (no FP16 builder
    flag). ``keep_io_types=True`` keeps the engine's input/output tensors in
    FP32 so the runtime feeds the same float32 blob for fp32 and fp16 — that is
    what makes the two precisions comparable.

    Uses NVIDIA ModelOpt AutoCast (same as ultralytics), falling back to
    ``onnxconverter-common`` if ModelOpt is not installed.
    """
    import onnx

    dst = onnx_path.with_suffix(".fp16.onnx")
    if dst.exists():
        return dst

    input_name = onnx.load(str(onnx_path), load_external_data=False).graph.input[0].name

    # Calibrate AutoCast on a real image: it keeps a node in FP32 when the
    # observed activation range is large, and noise would strand the early convs.
    im = None
    for cand in (config.DATA_DIR / "bus.jpg", config.DATA_DIR / "sample.jpg"):
        if cand.exists():
            im = cv2.imread(str(cand))
            if im is not None:
                break
    if im is None:
        raise RuntimeError(
            "FP16 AutoCast needs a calibration image; put data/bus.jpg in place "
            "(curl -L -o data/bus.jpg https://ultralytics.com/images/bus.jpg)."
        )
    im = cv2.resize(im, (config.IMGSZ, config.IMGSZ))
    im = im[..., ::-1].transpose(2, 0, 1)          # BGR HWC -> RGB CHW
    im = np.ascontiguousarray(im, dtype=np.float32) / 255.0
    im = im[None, ...]                              # (1, 3, H, W)

    try:
        from modelopt.onnx import autocast

        print("[export] converting ONNX to FP16 mixed precision (ModelOpt AutoCast)"
              " — this keeps the graph's I/O in FP32", flush=True)
        converted = autocast.convert_to_mixed_precision(
            str(onnx_path),
            low_precision_type="fp16",
            keep_io_types=True,
            calibration_data={input_name: im},
        )
        onnx.save(converted, str(dst))
        return dst
    except ImportError:
        print("[export] ModelOpt not found; falling back to onnxconverter-common "
              "float16 conversion", flush=True)

    from onnxconverter_common import float16

    model = onnx.load(str(onnx_path))
    # Ops that are numerically unsafe in FP16 (and the size/DFL head) stay FP32.
    model = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=[
            "Resize", "NonMaxSuppression", "TopK", "ReduceMax", "ReduceMin",
        ],
    )
    onnx.save(model, str(dst))
    return dst


def _log_engine_io(engine_blob: bytes) -> None:
    """Print input/output tensor names, shapes and dtypes of a built engine."""
    try:
        import tensorrt as trt

        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        engine = runtime.deserialize_cuda_engine(engine_blob)
        if engine is None:
            return
        for i in range(engine.num_io_tensors):
            nm = engine.get_tensor_name(i)
            print(f"[export]   {engine.get_tensor_mode(nm).name:6s} "
                  f"{nm}: {engine.get_tensor_dtype(nm).name} "
                  f"{tuple(engine.get_tensor_shape(nm))}")
    except Exception:  # pragma: no cover - logging only
        pass


def _build_engine_api(
    name: str,
    precision: str,
    onnx_path: Path,
    engine_path: Path,
    workspace_gb: float | None,
    tf32_off: bool,
) -> Path:
    """Build an engine from ONNX via the TensorRT Python API."""
    import tensorrt as trt

    major = _trt_major() or 0
    is_trt10_plus = major >= 10
    is_trt11_plus = major >= 11

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config_b = builder.create_builder_config()

    if workspace_gb is not None and hasattr(config_b, "set_memory_pool_limit"):
        config_b.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
        )

    # EXPLICIT_BATCH is the only mode from TRT 10 on; the flag was removed there.
    flags = 0 if is_trt10_plus else (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flags)

    # FP16 on TRT >= 11 must be baked into the graph (see module docstring).
    build_onnx = onnx_path
    if precision == "fp16" and is_trt11_plus:
        build_onnx = _fp16_onnx(onnx_path)

    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(build_onnx)):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT failed to parse {build_onnx}:\n" + "\n".join(errs))

    # TensorRT 7-10 still expose the precision flags.
    if precision == "fp16" and not is_trt11_plus:
        config_b.set_flag(trt.BuilderFlag.FP16)

    # TF32 is a 32-bit tensor-core mode enabled by default on Ampere+; clear it
    # for a strict FP32 baseline (the Python-API equivalent of trtexec --noTF32).
    if tf32_off and precision == "fp32" and hasattr(trt.BuilderFlag, "TF32"):
        config_b.clear_flag(trt.BuilderFlag.TF32)

    if is_trt11_plus:
        print(f"[export] TensorRT {trt.__version__} is strongly-typed; "
              f"fp16 is carried by the ONNX graph, not a builder flag", flush=True)

    t0 = time.time()
    engine = builder.build_serialized_network(network, config_b)
    if engine is None:
        raise RuntimeError("TensorRT build returned no engine; check the log above")
    blob = bytes(engine)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(blob)

    print(f"[export] built {engine_path.name} in {time.time() - t0:.1f}s "
          f"({len(blob) / 1e6:.1f} MB)", flush=True)
    _log_engine_io(blob)
    return engine_path


def _build_engine_trtexec(
    name: str,
    precision: str,
    exe: str,
    onnx_path: Path,
    engine_path: Path,
    workspace_gb: float | None,
    tf32_off: bool,
) -> Path:
    """Build an engine by shelling out to ``trtexec`` (Jetson / full TRT install)."""
    build_onnx = onnx_path
    # TRT >= 11 removed --fp16; the graph must carry FP16 (same as the API path).
    if precision == "fp16" and (_trt_major() or 0) >= 11:
        build_onnx = _fp16_onnx(onnx_path)

    cmd = [exe, f"--onnx={build_onnx}", f"--saveEngine={engine_path}"]
    if precision == "fp16" and (_trt_major() or 0) < 11:
        cmd += ["--fp16", "--explicitBatch"]
    if workspace_gb is not None:
        cmd.append(f"--workspace={int(workspace_gb * 1024)}")  # trtexec wants MiB
    if tf32_off and precision == "fp32":
        cmd.append("--noTF32")

    print(f"[export] building {precision} engine: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not engine_path.exists():
        print((proc.stderr + proc.stdout)[-4000:], file=sys.stderr)
        raise RuntimeError(f"trtexec failed for {name}.{precision} (rc={proc.returncode})")

    size_mb = engine_path.stat().st_size / 1e6
    print(f"[export] wrote {engine_path.name} ({size_mb:.1f} MB)", flush=True)
    return engine_path


def build_engine(
    name: str,
    precision: str,
    workspace_gb: float | None = config.TRT_WORKSPACE_GB,
    tf32_off: bool = False,
    via: str = "auto",
) -> Path:
    """Build ``{name}.{precision}.engine`` from ``{name}.onnx``.

    ``precision`` is 'fp32' or 'fp16'. Engines are NOT portable across GPU
    architectures or TensorRT versions, so this must run on each target device.

    ``via`` selects the builder: ``'auto'`` prefers the TensorRT Python API
    (always available, works with pip installs) and falls back to ``trtexec``;
    ``'python'``/``'trtexec'`` force one.
    """
    onnx_path = config.model_onnx(name)
    if not onnx_path.exists():
        raise FileNotFoundError(f"{onnx_path} missing; run export_onnx first.")
    engine_path = config.model_engine(name, precision)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    if via in ("auto", "python"):
        try:
            import tensorrt  # noqa: F401
            return _build_engine_api(
                name, precision, onnx_path, engine_path, workspace_gb, tf32_off
            )
        except ImportError:
            if via == "python":
                raise
            print("[export] TensorRT bindings unavailable; using trtexec", flush=True)

    exe = _find_trtexec()
    if exe is None:
        raise RuntimeError(
            "No way to build an engine: TensorRT Python bindings are not installed "
            "and trtexec was not found. Install TensorRT with 'pip install tensorrt' "
            "(bindings only; the wheel ships no trtexec) or put trtexec on PATH."
        )
    return _build_engine_trtexec(
        name, precision, exe, onnx_path, engine_path, workspace_gb, tf32_off
    )


# ---------------------------------------------------------------------------
# Cross-check: Ultralytics' own format='engine' path
# ---------------------------------------------------------------------------

def ultralytics_engine(name: str, precision: str) -> Path:
    """Cross-check build using Ultralytics' TensorRT exporter.

    Ultralytics writes ``{name}.engine`` inline; we move it to a tagged name so
    it doesn't collide with the Python-API-built engines. Requires the TensorRT
    Python bindings on top of the pip TensorRT package.
    """
    src = config.model_pt(name)
    if not src.exists():
        raise FileNotFoundError(f"{src} missing.")
    from ultralytics import YOLO

    model = YOLO(str(src))
    quantize = 16 if precision == "fp16" else 32
    model.export(format="engine", imgsz=config.IMGSZ, dynamic=False, quantize=quantize)

    produced = src.with_suffix(".engine")
    dst = config.model_engine(name, precision + ".ultralytics")
    if produced != dst:
        shutil.move(str(produced), str(dst))
    print(f"[export] ultralytics cross-check engine: {dst.name}")
    return dst


# ---------------------------------------------------------------------------
# Complexity metadata backfill
# ---------------------------------------------------------------------------

def annotate_complexity(models: list[str] | None = None) -> dict[str, dict]:
    """(Re)write ``params_m`` / ``gflops`` into each model's metadata.

    Separate from `export` because complexity comes from the ``.pt`` (needs
    ultralytics, a [convert]-tier dependency) while engines/ONNX may already
    exist. Lets a machine that only received checkpoints record complexity without
    re-exporting anything, and adds it to older metadata written before this
    field existed.
    """
    out: dict[str, dict] = {}
    for name in (models or config.MODELS):
        src = config.model_pt(name)
        if not src.exists():
            print(f"[meta] {name}: checkpoint {src.name} missing, skipped")
            continue
        complexity = _measure_complexity(src, config.IMGSZ)
        out[name] = complexity
        if not complexity:
            print(f"[meta] {name}: could not measure complexity")
            continue
        path = config.model_meta(name)
        existing: dict = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                existing = {}
        existing.update(complexity)
        path.write_text(json.dumps(existing, indent=2))
        print(f"[meta] {name}: params={complexity.get('params_m')}M "
              f"GFLOPs={complexity.get('gflops')} -> {path.name}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Convert YOLO .pt -> ONNX -> TensorRT")
    p.add_argument("--models", nargs="+", default=None,
                   help="models to convert (default: all in config.MODELS)")
    p.add_argument("--precisions", nargs="+", default=None,
                   help="fp32 and/or fp16 (default: both)")
    p.add_argument("--meta-only", action="store_true",
                   help="only (re)write params/GFLOPs into .meta.json; no conversion")
    p.add_argument("--onnx-only", action="store_true",
                   help="only do ONNX -> engine (skip .pt -> .onnx)")
    p.add_argument("--tf32-off", action="store_true",
                   help="disable TF32 for a strict FP32 baseline")
    p.add_argument("--cross-check", action="store_true",
                   help="also build engines via ultralytics format='engine'")
    p.add_argument("--workspace", type=float, default=None,
                   help="TensorRT workspace in GiB (default: auto)")
    p.add_argument("--via", choices=("auto", "python", "trtexec"), default="auto",
                   help="engine builder: the TensorRT Python API (default; always "
                        "available), trtexec, or auto")
    args = p.parse_args(argv)

    models = args.models or config.MODELS
    precisions = args.precisions or config.PRECISIONS
    ensure_dirs()

    if args.meta_only:
        annotate_complexity(models)
        return 0

    for name in models:
        try:
            if not args.onnx_only:
                onnx_path = export_onnx(name, tf32_off=args.tf32_off)
                nc, in_shape = _read_nc_from_onnx(onnx_path)
                write_meta(name, nc, in_shape, trt_version=_trt_version())
                print(f"[export] {name}.onnx: nc={nc} input={in_shape}")

            for precision in precisions:
                if precision not in ("fp32", "fp16"):
                    print(f"[export] skipping unknown precision {precision!r}")
                    continue
                build_engine(
                    name, precision,
                    workspace_gb=args.workspace,
                    tf32_off=args.tf32_off,
                    via=args.via,
                )
                if args.cross_check and precision == "fp16":
                    # Only cross-check FP16 (the interesting path) to save time.
                    ultralytics_engine(name, precision)

            # refresh metadata with final arch/TRT info
            if args.onnx_only:
                meta = config.model_meta(name)
                if meta.exists():
                    write_meta(name, json.loads(meta.read_text())["nc"],
                               input_shape=None,
                               trt_version=_trt_version())
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[export] FAILED {name}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())