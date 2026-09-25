"""Comparison driver: run the full matrix and emit CSV + Markdown.

Runs every (runtime x precision x model) combination against the same mp4 and
writes a combined results table grouping the three metrics (framerate, latency,
memory). Run on the target GPU (the RTX 3090) for meaningful numbers.

Usage
-----
  python -m yolo_bench.compare --video data/sample.mp4 --frames 100 \
      --models yolo11s yolo11l yolo26s yolo26l \
      --runtimes pytorch ort_cuda ort_trt tensorrt --precisions fp32 fp16

  # limits (subset for a quick smoke test):
  python -m yolo_bench.compare --video data/sample.mp4 --frames 30 \
      --models yolo11s --runtimes pytorch --precisions fp32
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from . import config
from .bench import run_benchmark

_FIELDS = [
    "architecture", "runtime", "precision", "model", "frames",
    "e2e_ms", "e2e_med_ms", "e2e_p95_ms",
    "infer_ms", "infer_med_ms", "infer_p95_ms",
    "pre_ms", "post_ms", "fps",
    "peak_vram_gb", "host_rss_mb", "weights_mb",
    "gpu_name", "arch", "torch", "torch_backend", "tf32_on",
]


def _combos(models, runtimes, precisions):
    out = []
    # Stable, predictable order: model-major, then precision, then runtime.
    for model in models:
        for precision in precisions:
            for runtime in runtimes:
                out.append((runtime, precision, model))
    return out


def _nan_row(runtime, precision, model):
    return {
        "architecture": config.arch_for(model), "runtime": runtime,
        "precision": precision, "model": model, "frames": 0,
        "e2e_ms": float("nan"), "e2e_med_ms": float("nan"),
        "e2e_p95_ms": float("nan"), "infer_ms": float("nan"),
        "infer_med_ms": float("nan"), "infer_p95_ms": float("nan"),
        "pre_ms": float("nan"), "post_ms": float("nan"),
        "fps": float("nan"), "peak_vram_gb": float("nan"),
        "host_rss_mb": float("nan"), "weights_mb": float("nan"),
        "gpu_name": None, "arch": None, "torch": None,
        "torch_backend": None, "tf32_on": None,
    }


def _run_one(runtime, precision, model, video, frames, warmup, conf, iou):
    try:
        return run_benchmark(
            runtime, precision, model, video,
            frames=frames, warmup=warmup, conf=conf, iou=iou,
        )
    except Exception as exc:  # noqa: BLE001 - keep going, record the failure
        print(f"[compare] FAILED {model} {runtime} {precision}: {exc}",
              file=sys.stderr)
        return _nan_row(runtime, precision, model)


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _ms(v) -> str:
    if v is None or v != v:  # NaN
        return "-"
    return f"{v:7.2f}"


def _write_md(rows: list[dict], path: Path) -> None:
    """Markdown: one table per model, rows = runtime x precision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# YOLO inference benchmark",
        "",
    ]
    if rows:
        header.append(f"- GPU: `{rows[0]['gpu_name']}` (arch {rows[0]['arch']})")
        header.append(f"- Frames: {rows[0]['frames']}")
    header += [
        "- `fps` = throughput (sustained); higher is better",
        "- `e2e_ms` / `e2e_p95_ms` = full-frame latency (mean / p95); lower is better",
        "- `infer_ms` = raw forward pass only",
        "- `vram_gb` / `rss_mb` / `weights_mb` = memory footprint",
        "",
    ]
    lines = [l for l in header if l != ""]

    models = sorted({r["model"] for r in rows})
    runtimes = config.RUNTIMES
    precisions = config.PRECISIONS

    for model in models:
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| runtime | precision | fps | e2e_ms | e2e_p95 | infer_ms | vram_gb | rss_mb | weights_mb |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for runtime in runtimes:
            for precision in precisions:
                r = next(
                    (x for x in rows
                     if x["model"] == model and x["runtime"] == runtime
                     and x["precision"] == precision), None)
                if r is None:
                    continue
                lines.append(
                    f"| {runtime} | {precision} | {r['fps']:7.1f} | "
                    f"{_ms(r['e2e_ms'])} | {_ms(r['e2e_p95_ms'])} | "
                    f"{_ms(r['infer_ms'])} | {r['peak_vram_gb']:7.2f} | "
                    f"{r['host_rss_mb']:7.0f} | {r['weights_mb']:7.1f} |"
                )
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run the full YOLO benchmark matrix")
    p.add_argument("--video", required=True)
    p.add_argument("--models", nargs="+", default=config.MODELS)
    p.add_argument("--runtimes", nargs="+", default=config.RUNTIMES)
    p.add_argument("--precisions", nargs="+", default=config.PRECISIONS)
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--conf", type=float, default=config.CONF)
    p.add_argument("--iou", type=float, default=config.IOU)
    p.add_argument("--out", type=Path, default=None,
                   help="base name for results (default: results/compare)")
    args = p.parse_args(argv)

    out_base = args.out or (config.RESULTS_DIR / "compare")
    csv_path = out_base.with_suffix(".csv")
    md_path = out_base.with_suffix(".md")

    combos = _combos(args.models, args.runtimes, args.precisions)
    rows = []
    total = len(combos)
    for i, (runtime, precision, model) in enumerate(combos, 1):
        print(f"[compare] ({i}/{total}) {model} {runtime} {precision}", flush=True)
        rows.append(_run_one(runtime, precision, model, Path(args.video),
                             args.frames, args.warmup, args.conf, args.iou))

    _write_csv(rows, csv_path)
    if rows and rows[0].get("gpu_name") is not None:
        _write_md(rows, md_path)

    print(f"[compare] wrote {csv_path}" + (f" and {md_path}" if md_path.exists() else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())