"""Test orchestration: run a matrix of configurations as isolated processes.

This is the *harness* layer, separate from the reusable inference stack
(``Detector`` + ``archs`` + ``runtimes``). It exists because of one measurement
truth: **you cannot get honest peak-memory numbers by running many configurations
in one process.**

Why one process per configuration
---------------------------------
Running 56 configurations in a single interpreter produced `host_rss_mb` values
that grew monotonically (1.2 GB → 4.5 GB) purely because process RSS never gives
memory back — the later rows looked like they used 4× the RAM of the earlier ones
when the difference was just accumulation. Device memory has the same problem:
allocator caches and TensorRT/ORT arenas outlive ``release()``.

So each configuration runs in a fresh subprocess (``run_one`` is invoked by
``python -m yolo_bench.orchestrate._child``), and the parent only aggregates
results. Costs a process spawn per config; buys measurements that mean something.

Use from a notebook
-------------------
    from yolo_bench import orchestrate

    rows = orchestrate.run(
        models=["yolo11s", "yolo26s"],
        runtimes=["tensorrt", "ort_cuda"],
        precisions=["fp32", "fp16"],
        video="data/sample_1080p_h264.mp4",
        frames=100,
        workers=1,          # GPU runs should stay serialized
    )

    # or a named experiment for repeatability
    rows = orchestrate.run_experiment("headline", video=..., frames=100)

Blocking and non-blocking
------------------------
``run()`` is blocking and returns rows. For progress UI in a notebook, use
``submit()`` to get a handle and poll it::

    job = orchestrate.submit(...)
    while not job.done:
        print(job.completed, "/", job.total)

Parent/child handshake
----------------------
The child prints one JSON object per run on a sentinel-prefixed line, because
native libraries (TensorRT, OpenVINO) write freely to stdout/stderr and would
otherwise corrupt the payload. Anything else the child emits is treated as logs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..env import FINGERPRINT_FIELDS
from ..runtimes import CPU_KINDS, CUDA_KINDS, plan, spec_for

# Sentinel so log noise from native libs can't be mistaken for a result.
RESULT_PREFIX = "@@YOLO_BENCH_ROW@@"

# Per-run fields: the measurement, then the cost/complexity of the model, then the
# per-row runtime provenance. The environment fingerprint is appended below.
RUN_FIELDS: tuple[str, ...] = (
    "architecture", "runtime", "precision", "model", "frames",
    "e2e_ms", "e2e_med_ms", "e2e_p95_ms",
    "infer_ms", "infer_med_ms", "infer_p95_ms",
    "pre_ms", "post_ms", "fps",
    "peak_vram_gb", "base_vram_gb", "host_rss_mb", "weights_mb",
    # Model cost, from .meta.json — needed to plot performance vs model size and
    # to split the comparison into small (n/s) and large (m/l/x) groups.
    "params_m", "gflops", "model_size",
    "runtime_version", "group", "status", "error",
)

# Row schema, kept in one place so CSV/DataFrame/report agree.
FIELDS: tuple[str, ...] = RUN_FIELDS + FINGERPRINT_FIELDS

# The set of runtimes a default run covers: CUDA backends first, then CPU.
DEFAULT_RUNTIMES: tuple[str, ...] = tuple(CUDA_KINDS) + tuple(CPU_KINDS)


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Experiment:
    """A named, repeatable test setup.

    Recipes are plain data so a notebook cell can name a setup instead of
    re-typing a long command line, and two cells can't silently drift apart.
    """

    name: str
    description: str
    models: tuple[str, ...] = tuple(config.MODELS)
    runtimes: tuple[str, ...] = DEFAULT_RUNTIMES
    precisions: tuple[str, ...] = ("fp32", "fp16")
    frames: int = 100
    warmup: int = 50
    video: str = "data/sample_1080p_h264.mp4"


EXPERIMENTS: dict[str, Experiment] = {
    "smoke": Experiment(
        name="smoke",
        description="One model, one runtime — proves the stack runs end to end.",
        models=("yolo11s",),
        runtimes=("tensorrt",),
        precisions=("fp32", "fp16"),
        frames=20,
        warmup=5,
    ),
    "gpu": Experiment(
        name="gpu",
        description="All CUDA backends x both precisions, all models.",
        runtimes=tuple(CUDA_KINDS),
    ),
    "cpu": Experiment(
        name="cpu",
        description="CPU baselines only (fp32: CPU EPs have no FP16 kernels).",
        runtimes=tuple(CPU_KINDS),
        precisions=("fp32",),
    ),
    "headline": Experiment(
        name="headline",
        description="Everything: all runtimes, both precisions, all models.",
        runtimes=DEFAULT_RUNTIMES,
    ),
    "precision": Experiment(
        name="precision",
        description="FP16 question only: native TRT vs ORT-CUDA vs eager PyTorch.",
        runtimes=("tensorrt", "ort_cuda", "pytorch"),
        precisions=("fp32", "fp16"),
    ),
    "small-vs-large": Experiment(
        name="small-vs-large",
        description="Same-architecture size scaling at the best-known config.",
        models=("yolo11s", "yolo11l"),
        runtimes=("tensorrt",),
        precisions=("fp16",),
    ),
    "yolo11-vs-yolo26": Experiment(
        name="yolo11-vs-yolo26",
        description="Head-rewrite question: yolo11 vs yolo26 at matched sizes.",
        runtimes=("tensorrt", "ort_cuda"),
        precisions=("fp16",),
    ),
}


def experiment_names() -> list[str]:
    return list(EXPERIMENTS)


# ---------------------------------------------------------------------------
# Child process entry point
# ---------------------------------------------------------------------------

def run_one(
    runtime: str,
    precision: str,
    model: str,
    video: str,
    *,
    frames: int | None = None,
    warmup: int = 50,
    conf: float = config.CONF,
    iou: float = config.IOU,
    architecture: str | None = None,
    annotate: str | None = None,
) -> dict:
    """Run a single configuration in THIS process and return a result row.

    Imported and called by the child entry point; also usable directly when
    process isolation is not needed (e.g. a notebook cell measuring one config
    with its own progress reporting).
    """
    from ..bench import run_benchmark

    started = time.time()
    try:
        row = run_benchmark(
            runtime, precision, model, Path(video),
            architecture=architecture, frames=frames, warmup=warmup,
            conf=conf, iou=iou, annotate=Path(annotate) if annotate else None,
        )
        row["status"] = "ok"
        row["error"] = ""
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        row = _empty_row(runtime, precision, model, architecture)
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["group"] = spec_for(runtime).group if _known(runtime) else "unknown"
    row["wall_s"] = round(time.time() - started, 2)
    return row


def _known(kind: str) -> bool:
    from ..runtimes import REGISTRY

    return kind in REGISTRY


def _empty_row(runtime, precision, model, architecture=None) -> dict:
    """A fully-populated NaN row so every record has the same shape."""
    from ..env import fingerprint

    return {
        "architecture": architecture or config.arch_for(model),
        "runtime": runtime, "precision": precision, "model": model,
        "frames": 0,
        **{k: float("nan") for k in (
            "e2e_ms", "e2e_med_ms", "e2e_p95_ms",
            "infer_ms", "infer_med_ms", "infer_p95_ms",
            "pre_ms", "post_ms", "fps",
            "peak_vram_gb", "base_vram_gb", "host_rss_mb", "weights_mb",
        )},
        "runtime_version": None,
        "group": "unknown", "status": "failed", "error": "",
        # Still record the fingerprint: a failed row's host is useful context,
        # and it keeps the CSV columns identical to a successful row.
        **{k: None for k in FINGERPRINT_FIELDS} | fingerprint(),
    }


def _child_main(argv: list[str]) -> int:
    """Entry point for ``python -m yolo_bench.orchestrate._child``."""
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--runtime", required=True)
    p.add_argument("--precision", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--conf", type=float, default=config.CONF)
    p.add_argument("--iou", type=float, default=config.IOU)
    p.add_argument("--architecture", default=None)
    p.add_argument("--annotate", default=None)
    args = p.parse_args(argv)

    row = run_one(
        args.runtime, args.precision, args.model, args.video,
        frames=args.frames, warmup=args.warmup, conf=args.conf, iou=args.iou,
        architecture=args.architecture, annotate=args.annotate,
    )
    # Sentinel-prefixed single line; native libs can print anything else.
    print(RESULT_PREFIX + json.dumps(row, default=str), flush=True)
    return 0


# ---------------------------------------------------------------------------
# Parent-side runner
# ---------------------------------------------------------------------------

@dataclass
class Job:
    """Handle for a matrix run, pollable for notebook progress feedback."""

    total: int
    rows: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    completed: int = 0
    done: bool = False
    started_at: float = field(default_factory=time.time)

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started_at

    @property
    def progress(self) -> float:
        return self.completed / self.total if self.total else 1.0

    def summary(self) -> str:
        ok = sum(1 for r in self.rows if r["status"] == "ok")
        bad = sum(1 for r in self.rows if r["status"] != "ok")
        return (f"{self.completed}/{self.total} done in {self.elapsed_s:.0f}s "
                f"({ok} ok, {bad} failed/skipped)")

    def to_dataframe(self):
        """Rows as a pandas DataFrame (built in dependency order)."""
        import pandas as pd

        return pd.DataFrame(self.rows)


def submit(
    models: list[str] | None = None,
    runtimes: list[str] | None = None,
    precisions: list[str] | None = None,
    *,
    video: str | None = None,
    frames: int | None = None,
    warmup: int = 50,
    conf: float = config.CONF,
    iou: float = config.IOU,
    env: dict[str, str] | None = None,
    isolate: bool = True,
    max_workers: int = 1,
) -> Job:
    """Start a matrix run and return a pollable :class:`Job`.

    ``isolate=True`` (default) runs each configuration in a fresh subprocess so
    memory numbers are trustworthy. ``max_workers`` defaults to 1: on a single
    GPU, concurrent runs contend for the device and distort both latency and
    framerate. Raise it only for CPU-only matrices.
    """
    job = Job(total=0)

    models = list(models or config.MODELS)
    runtimes = list(runtimes or DEFAULT_RUNTIMES)
    precisions = list(precisions or config.PRECISIONS)
    video = video or str(config.DATA_DIR / "sample_1080p_h264.mp4")

    runnable, skipped = plan(models, runtimes, precisions)
    job.total = len(runnable) + len(skipped)
    job.skipped = [
        {"runtime": k, "precision": p, "model": m, "status": "skipped", "error": why}
        for k, p, m, why in skipped
    ]

    def _one(combo) -> dict:
        kind, precision, model = combo
        if isolate:
            return _run_isolated(kind, precision, model, video, frames, warmup,
                                 conf, iou, env)
        return run_one(kind, precision, model, video, frames=frames,
                       warmup=warmup, conf=conf, iou=iou)

    combos = runnable
    if max_workers <= 1:
        for combo in combos:
            job.rows.append(_one(combo))
            job.completed += 1
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for row in pool.map(_one, combos):
                job.rows.append(row)
                job.completed += 1

    job.rows.extend(_as_rows(job.skipped))
    job.done = True
    return job


def _as_rows(skipped: list[dict]) -> list[dict]:
    return [_empty_row(s["runtime"], s["precision"], s["model"]) | {
        "status": "skipped", "error": s["error"],
        "group": spec_for(s["runtime"]).group if _known(s["runtime"]) else "unknown",
    } for s in skipped]


def _run_isolated(kind, precision, model, video, frames, warmup, conf, iou,
                  env) -> dict:
    """Spawn ``_child`` for one configuration and parse its single result line."""
    cmd = [
        sys.executable, "-m", "yolo_bench.orchestrate._child",
        "--runtime", kind, "--precision", precision, "--model", model,
        "--video", video, "--warmup", str(warmup),
        "--conf", str(conf), "--iou", str(iou),
    ]
    if frames is not None:
        cmd += ["--frames", str(frames)]

    child_env = dict(os.environ)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    if env:
        child_env.update(env)

    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=child_env,
        cwd=str(config.ROOT),
    )
    row = _parse_child_output(proc)
    if row is None:
        row = _empty_row(kind, precision, model)
        row["status"] = "failed"
        row["error"] = (
            f"child produced no result (rc={proc.returncode}); "
            f"tail: {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'none'}"
        )
    row["wall_s"] = row.get("wall_s", 0)
    return row


def _parse_child_output(proc) -> dict | None:
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            try:
                return json.loads(line[len(RESULT_PREFIX):])
            except json.JSONDecodeError:
                return None
    return None


def run(
    models: list[str] | None = None,
    runtimes: list[str] | None = None,
    precisions: list[str] | None = None,
    *,
    video: str | None = None,
    frames: int | None = None,
    warmup: int = 50,
    conf: float = config.CONF,
    iou: float = config.IOU,
    env: dict[str, str] | None = None,
    isolate: bool = True,
    max_workers: int = 1,
    progress: bool = True,
) -> list[dict]:
    """Blocking matrix run; returns the list of result rows."""
    job = submit(models, runtimes, precisions, video=video, frames=frames,
                 warmup=warmup, conf=conf, iou=iou, env=env, isolate=isolate,
                 max_workers=max_workers)
    if progress:
        _print_progress(job)
    return job.rows


def run_experiment(name: str, **overrides) -> list[dict]:
    """Run a named :class:`Experiment`, with optional field overrides."""
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}; have {experiment_names()}")
    exp = EXPERIMENTS[name]
    kwargs = dict(
        models=list(exp.models), runtimes=list(exp.runtimes),
        precisions=list(exp.precisions), video=exp.video,
        frames=exp.frames, warmup=exp.warmup,
    )
    kwargs.update(overrides)
    return run(**kwargs)


def submit_experiment(name: str, **overrides) -> Job:
    """Non-blocking variant of :func:`run_experiment` for notebook progress."""
    exp = EXPERIMENTS[name]
    kwargs = dict(
        models=list(exp.models), runtimes=list(exp.runtimes),
        precisions=list(exp.precisions), video=exp.video,
        frames=exp.frames, warmup=exp.warmup,
    )
    kwargs.update(overrides)
    return submit(**kwargs)


def _print_progress(job: Job) -> None:
    for row in job.rows:
        if row["status"] == "ok":
            print(f"[orchestrate] {row['model']:9s} {row['runtime']:13s} "
                  f"{row['precision']:5s} fps={row['fps']:7.1f} "
                  f"infer={row['infer_ms']:7.2f}ms vram={row['peak_vram_gb']:.2f}GB "
                  f"({row['wall_s']}s)")
        elif row["status"] == "skipped":
            print(f"[orchestrate] SKIP {row['model']:9s} {row['runtime']:13s} "
                  f"{row['precision']:5s} {row['error']}")
        else:
            print(f"[orchestrate] FAIL {row['model']:9s} {row['runtime']:13s} "
                  f"{row['precision']:5s} {row['error']}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: str | Path) -> Path:
    """Write rows to CSV with a stable column order.

    Audits for fields present in the rows but missing from :data:`FIELDS`. A
    mismatch used to be silent (`csv.DictWriter(extrasaction="ignore")` drops
    unknown columns), which cost a full re-run to notice: the complexity fields
    were computed correctly but never written. Now it warns loudly.
    """
    import csv

    if rows:
        unknown = sorted(set().union(*[set(r) for r in rows]) - set(FIELDS))
        if unknown:
            print("[orchestrate] WARNING: dropping fields not in FIELDS: "
                  + ", ".join(unknown), file=sys.stderr)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def read_csv(path: str | Path):
    """Read a results CSV into a DataFrame (float columns coerced)."""
    import pandas as pd

    df = pd.read_csv(path)
    for col in df.columns:
        if col in ("frames",):
            continue
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
    return df


if __name__ == "__main__":
    raise SystemExit(_child_main(sys.argv[1:]))