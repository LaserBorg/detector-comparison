"""CLI for a full matrix run: ``python -m yolo_bench compare ...``.

Thin wrapper over :mod:`yolo_bench.orchestrate` — the orchestration logic lives
there so the notebook and this CLI cannot drift apart. Use ``--experiment`` to
run a named setup instead of spelling out the matrix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config, orchestrate
from .runtimes import ALL_KINDS, CPU_KINDS, CUDA_KINDS


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Run a detector benchmark matrix → CSV + Markdown",
        epilog="Runtimes: " + ", ".join(ALL_KINDS),
    )
    p.add_argument("--video", default=None,
                   help="mp4 to read (default: data/sample_1080p_h264.mp4)")
    p.add_argument("--models", nargs="+", default=None,
                   help="models to test (default: all in config.MODELS)")
    p.add_argument("--runtimes", nargs="+", default=None,
                   help="runtimes to test (default: all registered)")
    p.add_argument("--precisions", nargs="+", default=None,
                   help="fp32 and/or fp16 (default: both; runtime support varies)")
    p.add_argument("--frames", type=int, default=None, help="cap frames per config")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--conf", type=float, default=config.CONF)
    p.add_argument("--iou", type=float, default=config.IOU)
    p.add_argument("--out", type=str, default=None,
                   help="output prefix (default: results/compare)")
    p.add_argument("--experiment", choices=orchestrate.experiment_names(),
                   default=None,
                   help="run a named setup from orchestrate.EXPERIMENTS")
    p.add_argument("--group", choices=("all", "cuda", "cpu"), default="all",
                   help="restrict to a runtime group (ignored with --experiment)")
    p.add_argument("--no-isolate", action="store_true",
                   help="run every config in ONE process (faster, but host RSS "
                        "accumulates and memory numbers become meaningless)")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel configs; keep 1 for GPU matrices")
    p.add_argument("--list-runtimes", action="store_true",
                   help="show the runtime registry and exit")
    p.add_argument("--list-experiments", action="store_true",
                   help="show the named experiments and exit")
    args = p.parse_args(argv)

    if args.list_runtimes:
        from .runtimes import describe

        print(describe())
        return 0
    if args.list_experiments:
        for name in orchestrate.experiment_names():
            exp = orchestrate.EXPERIMENTS[name]
            print(f"{name}")
            print(f"    {exp.description}")
            print(f"    models={list(exp.models)}")
            print(f"    runtimes={list(exp.runtimes)} "
                  f"precisions={list(exp.precisions)} frames={exp.frames}")
        return 0

    runtimes = args.runtimes
    if runtimes is None:
        runtimes = {"all": ALL_KINDS, "cuda": CUDA_KINDS, "cpu": CPU_KINDS}[args.group]
        runtimes = list(runtimes)

    out_base = args.out or str(config.RESULTS_DIR / "compare")

    common = dict(
        video=args.video, frames=args.frames, warmup=args.warmup,
        conf=args.conf, iou=args.iou,
        isolate=not args.no_isolate, max_workers=args.workers,
    )
    if args.experiment:
        job = orchestrate.submit_experiment(args.experiment, **common)
    else:
        job = orchestrate.submit(
            models=args.models or list(config.MODELS),
            runtimes=runtimes,
            precisions=args.precisions or list(config.PRECISIONS),
            **common,
        )

    orchestrate._print_progress(job)
    if not job.rows:
        print("[compare] nothing ran", file=sys.stderr)
        return 2

    csv_path = orchestrate.write_csv(job.rows, out_base + ".csv")
    md_path = _write_md(job.rows, out_base + ".md")
    print(f"[compare] wrote {csv_path} and {md_path}")

    failed = [r for r in job.rows if r["status"] == "failed"]
    if failed:
        print(f"[compare] {len(failed)} configuration(s) failed:", file=sys.stderr)
        for r in failed:
            print(f"  {r['model']} {r['runtime']} {r['precision']}: {r['error']}",
                  file=sys.stderr)
    return 0


def _fmt(value, width=7, places=2) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return " " * width
    if f != f:  # NaN
        return " " * width
    return f"{f:{width}.{places}f}"


def _write_md(rows: list[dict], path: str) -> str:
    """Markdown report: one table per model, plus a skipped-configuration list."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    out: list[str] = ["# Detector benchmark report", ""]
    env = next((r for r in rows if r["status"] == "ok"), None)
    if env:
        # Machine identity first: these results get merged across an RTX 3090,
        # an RTX 3070 and a Jetson Orin Nano, so a row's hardware must be
        # visible without cross-referencing the README.
        cores = env.get("cpu_cores")
        threads = env.get("cpu_threads")
        core_txt = f"{cores}c/{threads}t" if cores else f"{threads}t"
        out += [
            f"- **host**: `{env.get('hostname')}` [{env.get('machine')}] — "
            f"`{env.get('os_name')}` (kernel {env.get('kernel')})",
            f"- **cpu**: {env.get('cpu')} [{core_txt}, "
            f"{env.get('host_ram_gb')} GB RAM]",
            f"- **gpu**: {env.get('gpu_name')} ({env.get('gpu_arch')}, "
            f"SM {env.get('gpu_cc')}, {env.get('gpu_vram_gb')} GB), "
            f"driver {env.get('gpu_driver')}",
            f"- **stack**: CUDA {env.get('cuda')} / cuDNN {env.get('cudnn')}, "
            f"TensorRT {env.get('tensorrt')}, ONNX Runtime "
            f"{env.get('onnxruntime')}, torch {env.get('torch')} "
            f"({env.get('torch_backend')})",
        ]
        if env.get("is_jetson"):
            out.append(
                f"- **jetson**: L4T {env.get('l4t')}, "
                f"power mode `{env.get('nvpmodel')}`"
            )
        out += [
            f"- run: `{env.get('run_id')}` at {env.get('timestamp')}",
            "",
        ]
    out += [
        "- `fps` = throughput (higher is better); `e2e_ms` = full frame loop; "
        "`infer_ms` = raw forward pass only",
        "- `vram_gb` = device-wide peak; `weights_mb` = artifact on disk",
        "",
    ]

    models = sorted({r["model"] for r in rows}, key=str)
    for model in models:
        sub = [r for r in rows if r["model"] == model]
        skipped = [r for r in sub if r["status"] == "skipped"]
        out += [f"## {model}", ""]
        out += [
            "| runtime | precision | fps | e2e_ms | e2e_p95 | infer_ms | "
            "vram_gb | weights_mb |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in sub:
            if r["status"] == "ok":
                out.append(
                    f"| {r['runtime']} | {r['precision']} | {_fmt(r['fps'])} | "
                    f"{_fmt(r['e2e_ms'])} | {_fmt(r['e2e_p95_ms'])} | "
                    f"{_fmt(r['infer_ms'])} | {_fmt(r['peak_vram_gb'], 7)} | "
                    f"{_fmt(r['weights_mb'], 8, 1)} |"
                )
            else:
                out.append(
                    f"| {r['runtime']} | {r['precision']} | — | — | — | — | — | — |"
                )
        out.append("")
        if skipped:
            out += ["Skipped in this matrix:", ""]
            for r in skipped:
                out.append(f"- `{r['runtime']}` {r['precision']}: {r['error']}")
            out.append("")

    p.write_text("\n".join(out) + "\n")
    return str(p)


if __name__ == "__main__":
    raise SystemExit(main())