"""Download a COCO YOLO checkpoint and export a device-local TensorRT engine."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from . import config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Download a YOLO checkpoint and export a TensorRT engine"
    )
    parser.add_argument("--model", default=config.DEFAULT_MODEL,
                        help="Ultralytics model name, for example yolo11s")
    parser.add_argument("--precision", choices=config.PRECISIONS,
                        default=config.DEFAULT_PRECISION)
    parser.add_argument("--imgsz", type=int, default=config.IMGSZ)
    parser.add_argument("--device", default="0",
                        help="CUDA device passed to Ultralytics (default: 0)")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing checkpoint and engine")
    args = parser.parse_args(argv)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = config.MODELS_DIR / f"{args.model}.pt"
    engine = config.model_engine(args.model, args.precision)
    if engine.exists() and not args.force:
        raise FileExistsError(f"{engine} already exists; use --force to rebuild it")

    from ultralytics import YOLO

    source = str(checkpoint) if checkpoint.exists() else f"{args.model}.pt"
    print(f"Loading COCO checkpoint: {source}", flush=True)
    model = YOLO(source)
    exported = Path(str(model.export(
        format="engine",
        imgsz=args.imgsz,
        quantize=16 if args.precision == "fp16" else 32,
        device=args.device,
        dynamic=False,
    )))

    _move_onnx_artifacts(args.model)

    downloaded = Path(f"{args.model}.pt")
    if not checkpoint.exists() and downloaded.exists():
        shutil.move(str(downloaded), str(checkpoint))

    if not exported.exists():
        candidates = [
            Path(f"{args.model}.engine"),
            checkpoint.with_suffix(".engine"),
        ]
        exported = next((candidate for candidate in candidates if candidate.exists()),
                        exported)
    if not exported.exists():
        raise FileNotFoundError(f"Ultralytics did not produce an engine: {exported}")

    if engine.exists():
        engine.unlink()
    shutil.move(str(exported), str(engine))
    print(f"Checkpoint: {checkpoint}")
    print(f"TensorRT engine: {engine}")
    print("Build this engine separately on each target GPU.")
    return 0


def _move_onnx_artifacts(model: str) -> None:
    """Keep Ultralytics' intermediate ONNX files inside the models directory."""
    for source in (Path(f"{model}.onnx"), Path(f"{model}.fp16.onnx")):
        if not source.exists():
            continue
        destination = config.MODELS_DIR / source.name
        if destination.exists():
            destination.unlink()
        shutil.move(str(source), str(destination))


if __name__ == "__main__":
    raise SystemExit(main())
