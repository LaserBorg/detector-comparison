"""Correctness gate: each backend must match `ultralytics .predict()`.

For a reference image, this compares every runtime's final detections against
the reference PyTorch `.predict()` output. Matching boxes (by IoU >= threshold
with the same class) are counted; the gate passes when recall is high, i.e. the
hand-rolled pre/post + decode reproduces the framework's own results.

Usage
-----
  python -m yolo_bench.check --image data/bus.jpg --model yolo11s \
      --runtime tensorrt --precision fp16

  # check every available backend:
  python -m yolo_bench.check --image data/bus.jpg --model yolo11s --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from . import config
from .detector import Detector


def _reference_dets(image: np.ndarray, model_name: str, arch: str) -> np.ndarray:
    """Reference detections from Ultralytics' own .predict().

    Returns (M, 6) [x1, y1, x2, y2, conf, cls] in ORIGINAL image space.
    """
    from .archs.yolo import YoloArchitecture

    # Reference predictions currently only exist for the YOLO family (Ultralytics).
    # RF-DETR and other permissive-license detectors would supply their own.
    if arch != "yolo":
        raise NotImplementedError(
            f"No reference predictor for architecture {arch!r}; the correctness "
            "gate currently compares against Ultralytics .predict()."
        )

    a = YoloArchitecture(model_name)
    from ultralytics import YOLO

    model = YOLO(str(a.checkpoint_path(model_name)))
    results = model.predict(source=image, imgsz=config.IMGSZ, conf=config.CONF,
                            iou=config.IOU, verbose=False)
    out = []
    for box in results[0].boxes:
        out.append([
            float(box.xyxy[0][0]), float(box.xyxy[0][1]),
            float(box.xyxy[0][2]), float(box.xyxy[0][3]),
            float(box.conf[0]), float(box.cls[0]),
        ])
    return np.asarray(out, dtype=np.float32).reshape(-1, 6)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-16)


def _match_recall(pred: np.ndarray, ref: np.ndarray, iou_thres: float = 0.5) -> float:
    """Fraction of reference boxes matched by a predicted box (IoU + same class)."""
    if ref.shape[0] == 0:
        return 1.0
    matched = 0
    used = set()
    for i, r in enumerate(ref):
        best_iou = 0.0
        best_j = -1
        for j, p in enumerate(pred):
            if j in used or int(p[5]) != int(r[5]):
                continue
            iou = _iou(p, r)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thres:
            matched += 1
            used.add(best_j)
    return matched / ref.shape[0]


def check_runtime(runtime_kind: str, precision: str, model: str, image: np.ndarray,
                  arch: str, iou_thres: float = 0.5) -> dict:
    ref = _reference_dets(image, model, arch)

    det = Detector(arch, runtime_kind, precision, model)
    det.load()
    det.warmup(image, n=5)
    dets = det(image)
    det.release()

    recall = _match_recall(dets, ref, iou_thres)
    return {
        "runtime": runtime_kind, "precision": precision, "model": model,
        "architecture": arch, "ref_dets": ref.shape[0],
        "pred_dets": dets.shape[0], "recall": recall,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compare backends against ultralytics .predict()")
    p.add_argument("--image", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--architecture", default=None,
                   help="detector family (default: inferred from model name)")
    p.add_argument("--runtime", default=None, choices=config.RUNTIMES)
    p.add_argument("--precision", default=None, choices=config.PRECISIONS)
    p.add_argument("--all", action="store_true", help="try every backend/precision")
    p.add_argument("--iou-thres", type=float, default=0.5)
    args = p.parse_args(argv)

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"cannot read image {args.image}", file=sys.stderr)
        return 1

    arch = args.architecture or config.arch_for(args.model)

    if args.all:
        combos = [(rt, pr) for rt in config.RUNTIMES for pr in config.PRECISIONS]
    else:
        rt = args.runtime or "pytorch"
        pr = args.precision or "fp32"
        combos = [(rt, pr)]

    ok = True
    for runtime, precision in combos:
        try:
            r = check_runtime(runtime, precision, args.model, image, arch,
                              args.iou_thres)
            status = "PASS" if r["recall"] >= 0.9 else "FAIL"
            if r["recall"] < 0.9:
                ok = False
            print(f"[check] {status} {r['model']} {r['runtime']} {r['precision']}: "
                  f"recall={r['recall']:.3f} ({r['pred_dets']} pred / {r['ref_dets']} ref)")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"[check] ERROR {args.model} {runtime} {precision}: {exc}",
                  file=sys.stderr)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())