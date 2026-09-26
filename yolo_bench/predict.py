"""Inference-only driver: run one config over a video, draw detections, save mp4.

Unlike ``bench`` (which times a capped frame loop and reports metrics), this is
the plain "run the detector on my video" path: it streams the whole clip
frame by frame (no full decode into RAM), draws boxes + labels on each frame,
and writes an annotated mp4. With ``--show`` it also displays the current
frame in a window while processing (press ``q`` to stop early) — useful on a
machine with a display; harmless to omit in headless/SSH sessions.

Encoder
-------
The output video is written through an ``ffmpeg`` subprocess when available
(default ``--encoder auto``):

- ``auto``   try ``h264_nvenc`` (GPU), fall back to ``libx264`` (CPU), then
             OpenCV ``mp4v`` (pure software, no ffmpeg needed).
- ``nvenc``  force ``h264_nvenc``; error if unavailable.
- ``x264``   force ``libx264``.
- ``mp4v``   force the OpenCV built-in writer (no ffmpeg subprocess).

NVENC runs in the GPU's dedicated encoder hardware and coexists with
inference; it needs a small CUDA context (~few hundred MB free VRAM). If the
GPU is fully occupied (e.g. by another process), NVENC init fails and ``auto``
falls back to CPU encoding.

Usage
-----
  python -m yolo_bench predict --runtime tensorrt --precision fp16 \
      --model yolo11s --video data/sample_1080p_h264.mp4 \
      --out out/yolo11s_trt_fp16.mp4 [--show] [--encoder auto]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from . import config
from .detector import Detector
from .utils import class_names


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw(frame: np.ndarray, dets: np.ndarray, names: list[str]) -> None:
    """Draw (M,6) [x1,y1,x2,y2,conf,cls] boxes in place on the frame."""
    for x1, y1, x2, y2, c, cls in dets:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{names[int(cls)]} {c:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


# ---------------------------------------------------------------------------
# Video writers
# ---------------------------------------------------------------------------

class _FfmpegWriter:
    """Write BGR frames to an mp4 via an ffmpeg subprocess (rawvideo pipe).

    Supports any encoder ffmpeg knows about (h264_nvenc, libx264, ...).
    """

    def __init__(self, path: Path, fps: float, w: int, h: int,
                 encoder: str, extra_args: list[str] | None = None) -> None:
        self.path = path
        self._proc: subprocess.Popen | None = None
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", encoder,
            *(extra_args or []),
            "-an", str(path),
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        # Read a bit of stderr to surface init errors early.
        # (We can't block on it — the pipe may stay open for the whole run.)
        self._stderr = self._proc.stderr
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("writer already closed")
        self._proc.stdin.write(frame.tobytes())

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc is not None:
            if self._proc.stdin:
                self._proc.stdin.close()
            err = b""
            if self._stderr:
                err = self._stderr.read()
                self._stderr.close()
            rc = self._proc.wait()
            if rc != 0:
                msg = err.decode(errors="replace").strip()
                raise RuntimeError(
                    f"ffmpeg exited with code {rc}:\n{msg[-2000:]}")
            self._proc = None


class _Cv2Writer:
    """Fallback writer using OpenCV's built-in mp4v codec (no ffmpeg needed)."""

    def __init__(self, path: Path, fps: float, w: int, h: int) -> None:
        self._wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                   fps, (w, h))
        if not self._wr.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open {path}")

    def write(self, frame: np.ndarray) -> None:
        self._wr.write(frame)

    def release(self) -> None:
        self._wr.release()


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _nvenc_available() -> bool:
    """Quick probe: can ffmpeg open h264_nvenc? (needs a free CUDA context)"""
    if not _ffmpeg_available():
        return False
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc=size=64x64:rate=1:duration=0.1",
             "-c:v", "h264_nvenc", "-an", "/dev/null"],
            capture_output=True, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


def _make_writer(out: Path, fps: float, w: int, h: int,
                 encoder: str):
    """Pick a writer based on the --encoder flag. Returns (writer, label)."""
    if encoder == "mp4v":
        return _Cv2Writer(out, fps, w, h), "mp4v (OpenCV)"

    if not _ffmpeg_available():
        if encoder in ("nvenc", "x264"):
            raise RuntimeError(
                f"--encoder {encoder} requires ffmpeg on PATH; "
                "use --encoder mp4v or install ffmpeg")
        # auto with no ffmpeg → mp4v
        return _Cv2Writer(out, fps, w, h), "mp4v (OpenCV, no ffmpeg)"

    if encoder == "nvenc":
        return _FfmpegWriter(out, fps, w, h, "h264_nvenc",
                             ["-preset", "p5"]), "h264_nvenc"

    if encoder == "x264":
        return _FfmpegWriter(out, fps, w, h, "libx264",
                             ["-preset", "veryfast"]), "libx264 (CPU)"

    # auto: try nvenc → x264 → mp4v
    if _nvenc_available():
        return _FfmpegWriter(out, fps, w, h, "h264_nvenc",
                             ["-preset", "p5"]), "h264_nvenc"
    return _FfmpegWriter(out, fps, w, h, "libx264",
                         ["-preset", "veryfast"]), "libx264 (CPU)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Run inference on a video, draw detections, save annotated mp4")
    p.add_argument("--runtime", required=True, choices=config.RUNTIMES)
    p.add_argument("--precision", required=True, choices=config.PRECISIONS)
    p.add_argument("--model", required=True)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="annotated mp4 to write")
    p.add_argument("--architecture", default=None,
                   help="detector family (default: inferred from model name)")
    p.add_argument("--frames", type=int, default=None, help="cap number of frames")
    p.add_argument("--warmup", type=int, default=10,
                   help="un-timed warmup frames to prime the runtime (default 10)")
    p.add_argument("--conf", type=float, default=config.CONF)
    p.add_argument("--iou", type=float, default=config.IOU)
    p.add_argument("--show", action="store_true",
                   help="display the current frame while processing (press q to stop)")
    p.add_argument("--encoder", default="auto",
                   choices=["auto", "nvenc", "x264", "mp4v"],
                   help="video encoder: auto (nvenc→x264→mp4v), nvenc, x264, mp4v")
    args = p.parse_args(argv)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer, enc_label = _make_writer(args.out, fps, w, h, args.encoder)

    det = Detector(args.architecture or config.arch_for(args.model),
                   args.runtime, args.precision, args.model,
                   conf=args.conf, iou=args.iou)
    det.load()
    names = class_names(det.nc)

    # Warmup on the first frame so the first written frame is at steady state.
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"No frames decoded from {args.video}")
    for _ in range(max(0, args.warmup)):
        det(first)

    t0 = time.perf_counter()
    det_time = 0.0
    n = 0
    frame = first
    while True:
        t_det = time.perf_counter()
        dets, _ = det.run(frame)
        det_time += time.perf_counter() - t_det

        _draw(frame, dets, names)
        writer.write(frame)
        n += 1
        if args.show:
            cv2.imshow("yolo_bench predict (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if args.frames is not None and n >= args.frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

    writer.release()
    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    det.release()

    wall = time.perf_counter() - t0
    det_fps = n / det_time if det_time > 0 else float("inf")
    print(f"[predict] {args.model} {args.runtime} {args.precision} "
          f"({enc_label}): {n} frames in {wall:.1f}s "
          f"(detection {det_fps:.1f} fps, end-to-end {n / wall:.1f} fps) "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
