"""Run YOLO TensorRT inference from a webcam."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from . import config
from .detector import Detector
from .utils import class_names


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="YOLO TensorRT webcam inference")
    parser.add_argument("--model", default=config.DEFAULT_MODEL)
    parser.add_argument("--precision", choices=config.PRECISIONS,
                        default=config.DEFAULT_PRECISION)
    parser.add_argument("--camera", type=int, default=config.DEFAULT_CAMERA)
    parser.add_argument(
        "--source",
        help="camera index, video file, or stream URL; overrides --camera",
    )
    parser.add_argument("--conf", type=float, default=config.CONF)
    parser.add_argument("--iou", type=float, default=config.IOU)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args(argv)

    source = _source_value(args.source, args.camera)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        if isinstance(source, int):
            raise RuntimeError(
                f"Cannot open camera {source}. WSL2 usually cannot access a "
                "Windows webcam as /dev/video0. Use --source with a video file "
                "or stream URL, or run this command in Windows/Linux with the "
                "camera attached."
            )
        raise RuntimeError(f"Cannot open source: {source}")

    detector = Detector(args.model, args.precision, conf=args.conf, iou=args.iou)
    names = class_names(detector.nc)
    try:
        detector.load()
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Camera opened but returned no frames")
        for _ in range(args.warmup):
            detector(frame)

        last_display = time.perf_counter()
        display_fps = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            started = time.perf_counter()
            detections, (pre_s, infer_s, post_s) = detector.run(frame)
            latency_ms = (time.perf_counter() - started) * 1000
            _draw(frame, detections, names)
            cv2.putText(frame, f"pre {pre_s * 1000:.1f} ms | "
                        f"TRT+copy {infer_s * 1000:.1f} ms | "
                        f"post {post_s * 1000:.1f} ms",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (0, 255, 255), 2)
            cv2.putText(frame, f"total {latency_ms:.1f} ms | display {display_fps:.1f} FPS",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (0, 255, 255), 2)
            cv2.imshow("YOLO TensorRT (q to quit)", frame)
            key = cv2.waitKey(1)
            now = time.perf_counter()
            display_fps = 1.0 / max(now - last_display, 1e-9)
            last_display = now
            if key & 0xFF == ord("q"):
                break
    finally:
        detector.release()
        cap.release()
        cv2.destroyAllWindows()
    return 0


def _source_value(source: str | None, camera: int) -> int | str:
    if source is None:
        return camera
    try:
        return int(source)
    except ValueError:
        path = Path(source)
        return str(path) if path.exists() else source


def _draw(frame, detections, names) -> None:
    for x1, y1, x2, y2, confidence, class_id in detections:
        first = int(x1), int(y1)
        second = int(x2), int(y2)
        cv2.rectangle(frame, first, second, (0, 255, 0), 2)
        label = f"{names[int(class_id)]} {confidence:.2f}"
        cv2.putText(frame, label, (first[0], max(first[1] - 6, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


if __name__ == "__main__":
    raise SystemExit(main())
