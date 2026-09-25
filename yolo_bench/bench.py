"""Benchmark driver: run one (architecture, runtime, precision, model) config.

Uses the ``Detector`` wrapper so the same driver works for any architecture
(YOLO now, RF-DETR later) and any runtime. Reports the three metrics this
project exists to compare:

  1. **Framerate**  — ``fps`` (1 / end-to-end time, the sustainable throughput).
  2. **Latency**    — mean/median/p95 of the full frame loop AND the raw forward
     pass. Median is the honest "time a frame spends in the pipeline" number;
     the difference between ``e2e`` and ``infer`` is the pre/post overhead.
  3. **Memory**     — peak device VRAM (GB) + process host RAM (MiB) + artifact
     size on disk (MB).

Usage
-----
  python -m yolo_bench.bench --runtime tensorrt --precision fp16 \
      --model yolo11s --video data/sample.mp4 --frames 100
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from . import config
from .detector import Detector
from .utils import (
    class_names, current_host_rss_mb, ensure_dirs, env_summary,
    peek_vram_gb, reset_vram, summarize,
)


def _read_frames(video: Path, max_frames: int | None):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")
    if max_frames is not None and max_frames <= 0:
        max_frames = None
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video}")
    return frames


def _annotate(frames: list[np.ndarray], all_dets: list[np.ndarray], nc: int,
              out_path: Path) -> None:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    names = class_names(nc)
    try:
        for frame, dets in zip(frames, all_dets):
            out = frame.copy()
            for box in dets:
                x1, y1, x2, y2, c, cls = box
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{names[int(cls)]} {c:.2f}"
                cv2.putText(out, label, (x1, max(y1 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            writer.write(out)
    finally:
        writer.release()


def run_benchmark(
    runtime_kind: str,
    precision: str,
    model: str,
    video: Path,
    *,
    architecture: str | None = None,
    frames: int | None = None,
    warmup: int = 50,
    conf: float = config.CONF,
    iou: float = config.IOU,
    annotate: Path | None = None,
) -> dict:
    """Run the benchmark and return a dict of results."""
    ensure_dirs()
    arch = architecture or config.arch_for(model)

    det = Detector(arch, runtime_kind, precision, model, conf=conf, iou=iou)
    det.load()

    frame_list = _read_frames(video, frames)
    nc = det.nc

    pre_times: list[float] = []
    inf_times: list[float] = []
    post_times: list[float] = []
    e2e_times: list[float] = []

    # Warmup primes allocators; builds the TRT engine on the ORT-TRT path.
    det.warmup(frame_list[0], n=warmup)

    reset_vram(track=True)
    dets_out: list[np.ndarray] = []

    for frame in frame_list:
        dets, (pre_s, inf_s, post_s) = det.run(frame)
        pre_times.append(pre_s)
        inf_times.append(inf_s)
        post_times.append(post_s)
        e2e_times.append(pre_s + inf_s + post_s)
        dets_out.append(dets)

    if annotate is not None:
        _annotate(frame_list, dets_out, nc, annotate)

    peak_gb = peek_vram_gb()
    host_mb = current_host_rss_mb()
    weights_mb = det.weights_mb
    det.release()

    return {
        "architecture": arch,
        "runtime": runtime_kind,
        "precision": precision,
        "model": model,
        "frames": len(frame_list),
        # latency (full loop)
        "e2e_ms": summarize(e2e_times)["mean"] * 1e3,
        "e2e_med_ms": summarize(e2e_times)["median"] * 1e3,
        "e2e_p95_ms": summarize(e2e_times)["p95"] * 1e3,
        # latency (raw forward pass only)
        "infer_ms": summarize(inf_times)["mean"] * 1e3,
        "infer_med_ms": summarize(inf_times)["median"] * 1e3,
        "infer_p95_ms": summarize(inf_times)["p95"] * 1e3,
        # latency (pre/post overhead)
        "pre_ms": summarize(pre_times)["mean"] * 1e3,
        "post_ms": summarize(post_times)["mean"] * 1e3,
        # framerate
        "fps": 1.0 / summarize(e2e_times)["mean"] if e2e_times else 0.0,
        # memory footprint
        "peak_vram_gb": peak_gb,
        "host_rss_mb": host_mb,
        "weights_mb": weights_mb,
        **env_summary(),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Benchmark one detector config on an mp4")
    p.add_argument("--runtime", required=True, choices=config.RUNTIMES)
    p.add_argument("--precision", required=True, choices=config.PRECISIONS)
    p.add_argument("--model", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--architecture", default=None,
                   help="detector family (default: inferred from model name)")
    p.add_argument("--frames", type=int, default=None, help="cap number of frames")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--conf", type=float, default=config.CONF)
    p.add_argument("--iou", type=float, default=config.IOU)
    p.add_argument("--annotate", type=Path, default=None, help="write annotated mp4")
    p.add_argument("--csv", type=Path, default=None, help="append one row to a TSV/CSV")
    args = p.parse_args(argv)

    result = run_benchmark(
        args.runtime, args.precision, args.model, Path(args.video),
        architecture=args.architecture, frames=args.frames, warmup=args.warmup,
        conf=args.conf, iou=args.iou, annotate=args.annotate,
    )

    print(f"[bench] {args.model} {args.runtime} {args.precision}: "
          f"fps={result['fps']:.1f}  e2e={result['e2e_ms']:.2f} ms "
          f"(p95 {result['e2e_p95_ms']:.2f} ms)  "
          f"infer={result['infer_ms']:.2f} ms  "
          f"vram={result['peak_vram_gb']:.2f} GB  rss={result['host_rss_mb']:.0f} MB  "
          f"weights={result['weights_mb']:.1f} MB")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "architecture", "runtime", "precision", "model", "frames",
            "e2e_ms", "e2e_med_ms", "e2e_p95_ms",
            "infer_ms", "infer_med_ms", "infer_p95_ms",
            "pre_ms", "post_ms", "fps",
            "peak_vram_gb", "host_rss_mb", "weights_mb",
            "gpu_name", "arch", "torch", "torch_backend", "tf32_on",
        ]
        write_header = not args.csv.exists()
        with open(args.csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())