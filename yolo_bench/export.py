"""Model conversion utility: Ultralytics .pt -> ONNX -> TensorRT engines.

Pipeline
--------
  * ``.pt``  -> ``.onnx``   via ``ultralytics`` export (imgsz 640, opset, simplify)
  * ``.onnx`` -> ``.engine`` via ``trtexec`` (FP16 with ``--fp16``, FP32 otherwise)
  * metadata (``nc``, input shape, arch, TRT version) written to ``.meta.json``

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
from pathlib import Path

from . import config
from .utils import detect_arch, ensure_dirs, nvidia_gpu_name


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
        **extra,
    }
    path = config.model_meta(name)
    path.write_text(json.dumps(meta, indent=2))
    return path


# ---------------------------------------------------------------------------
# .onnx -> .engine (trtexec)
# ---------------------------------------------------------------------------

def _find_trtexec() -> str:
    exe = shutil.which("trtexec")
    if exe:
        return exe
    # JetPack / pip tensorrt may not put trtexec on PATH; check common spots.
    for cand in (
        "/usr/src/tensorrt/bin/trtexec",
        "/opt/tensorrt/bin/trtexec",
    ):
        if Path(cand).exists():
            return cand
    raise FileNotFoundError(
        "trtexec not found. Install TensorRT (pip install tensorrt) or put "
        "trtexec on PATH. On Jetson it ships with JetPack."
    )


def _trt_version() -> str | None:
    try:
        import tensorrt as trt
        return trt.__version__
    except Exception:
        return None


def build_engine(
    name: str,
    precision: str,
    workspace_gb: float | None = config.TRT_WORKSPACE_GB,
    tf32_off: bool = False,
) -> Path:
    """Build ``{name}.{precision}.engine`` from ``{name}.onnx`` via trtexec.

    ``precision`` is 'fp32' or 'fp16'. Engines are NOT portable across GPU
    architectures or TensorRT versions, so this must run on each target device.
    """
    onnx_path = config.model_onnx(name)
    if not onnx_path.exists():
        raise FileNotFoundError(f"{onnx_path} missing; run export_onnx first.")
    engine_path = config.model_engine(name, precision)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        _find_trtexec(),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
    ]
    if precision == "fp16":
        cmd.append("--fp16")
        cmd.append("--explicitBatch")
    if workspace_gb is not None:
        # trtexec expects workspace in MiB
        cmd.append(f"--workspace={int(workspace_gb * 1024)}")
    if tf32_off:
        # Strict FP32 engine; only meaningful for the fp32 path.
        cmd.append("--noTF32")

    print(f"[export] building {precision} engine: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)

    tail = proc.stderr + proc.stdout
    if proc.returncode != 0 or not engine_path.exists():
        print(tail[-4000:], file=sys.stderr)
        raise RuntimeError(f"trtexec failed for {name}.{precision} (rc={proc.returncode})")

    size_mb = engine_path.stat().st_size / 1e6
    print(f"[export] wrote {engine_path.name} ({size_mb:.1f} MB)", flush=True)
    return engine_path


# ---------------------------------------------------------------------------
# Cross-check: Ultralytics' own format='engine' path
# ---------------------------------------------------------------------------

def ultralytics_engine(name: str, precision: str) -> Path:
    """Cross-check build using Ultralytics' TensorRT exporter.

    Ultralytics writes ``{name}.engine`` inline; we move it to a tagged name so
    it doesn't collide with the trtexec-built engines. Requires the TensorRT
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
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Convert YOLO .pt -> ONNX -> TensorRT")
    p.add_argument("--models", nargs="+", default=None,
                   help="models to convert (default: all in config.MODELS)")
    p.add_argument("--precisions", nargs="+", default=None,
                   help="fp32 and/or fp16 (default: both)")
    p.add_argument("--onnx-only", action="store_true",
                   help="only do ONNX -> engine (skip .pt -> .onnx)")
    p.add_argument("--tf32-off", action="store_true",
                   help="disable TF32 for a strict FP32 baseline")
    p.add_argument("--cross-check", action="store_true",
                   help="also build engines via ultralytics format='engine'")
    p.add_argument("--workspace", type=float, default=None,
                   help="TensorRT workspace in GiB (default: auto)")
    args = p.parse_args(argv)

    models = args.models or config.MODELS
    precisions = args.precisions or config.PRECISIONS
    ensure_dirs()

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