"""Accuracy measurement — the "quality" axis of the benchmark.

Why this exists
---------------
Framerate alone cannot answer "is FP16 worth it?", because FP16 changes the
*results*, not just the speed. Two different quality questions need two different
tools:

1. **Fidelity vs the framework reference** (``check``): does our re-implemented
   pre/post + the compiled graph reproduce ``ultralytics .predict()``? A pass/fail
   gate, not a quality number.
2. **Quantization cost** (this module): how much does FP16 change the detections
   *in the same runtime*, compared to that runtime's own FP32 output? This needs no
   labels at all, because FP32 is the reference — which is exactly why it is usable
   here. It answers "what did I pay for the FP16 speedup?".

The metric is deliberately conservative and simple to reason about:

  * ``recall``      — fraction of FP32 detections that FP16 also found (IoU>=thr,
                      same class). The headline "did fp16 lose objects?" number.
  * ``mean_iou``    — mean matched IoU over the shared detections. Localisation
                      drift: a box that moved slightly still matches, but a lower
                      mean IoU says the coordinates degraded.
  * ``mean_dconf``  — mean confidence difference (FP16 - FP32) over matches. A
                      systematic negative shift means FP16 scores are lower, which
                      matters when a downstream threshold is fixed.

A true mAP needs a labelled dataset and is not attempted here (see
``coco_map`` for the documented path). Note that agreement against a reference is
*not* accuracy: if FP32 were itself wrong, both would be wrong together. It
measures the quantization delta, which is the question this benchmark asks.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import config
from .detector import Detector


def _load_images(images: list[str | Path]) -> list[tuple[str, np.ndarray]]:
    import cv2

    out = []
    for item in images:
        p = Path(item)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    img = cv2.imread(str(f))
                    if img is not None:
                        out.append((f.name, img))
            continue
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"cannot read image: {p}")
        out.append((p.name, img))
    if not out:
        raise ValueError(f"no images found in {images!r}")
    return out


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in ``a`` against every box in ``b`` (N, M)."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-9)


def _agreement(pred: np.ndarray, ref: np.ndarray, iou_thres: float) -> dict:
    """Match ``pred`` against ``ref`` greedily by IoU + class.

    Greedy highest-IoU-first (not Hungarian): cheap, deterministic, and adequate
    for a delta measurement where both sets come from the same model.
    """
    if ref.shape[0] == 0:
        # Nothing to disagree with; recall is vacuously 1 but extra detections
        # are still worth surfacing via false_positives.
        return {"recall": 1.0, "mean_iou": float("nan"),
                "mean_dconf": float("nan"), "matched": 0,
                "ref_count": 0, "pred_count": int(pred.shape[0]),
                "false_positives": int(pred.shape[0])}
    if pred.shape[0] == 0:
        return {"recall": 0.0, "mean_iou": float("nan"),
                "mean_dconf": float("nan"), "matched": 0,
                "ref_count": int(ref.shape[0]), "pred_count": 0,
                "false_positives": 0}

    ious = _iou_matrix(pred, ref)
    same_class = pred[:, 5:6] == ref[None, :, 5]
    ious = np.where(same_class, ious, 0.0)

    matched_pred: set[int] = set()
    matched_ref: set[int] = set()
    pairs: list[tuple[float, int, int]] = []
    # Flatten and walk highest-IoU first.
    order = np.argsort(ious, axis=None)[::-1]
    for flat in order:
        i, j = divmod(int(flat), ious.shape[1])
        if ious[i, j] < iou_thres:
            break
        if i in matched_pred or j in matched_ref:
            continue
        matched_pred.add(i)
        matched_ref.add(j)
        pairs.append((float(ious[i, j]), i, j))

    matched = len(pairs)
    return {
        "recall": matched / ref.shape[0],
        "mean_iou": float(np.mean([p[0] for p in pairs])) if pairs else float("nan"),
        "mean_dconf": (float(np.mean([pred[i, 4] - ref[j, 4] for _, i, j in pairs]))
                       if pairs else float("nan")),
        "matched": matched,
        "ref_count": int(ref.shape[0]),
        "pred_count": int(pred.shape[0]),
        "false_positives": int(pred.shape[0]) - matched,
    }


def precision_agreement(
    model: str,
    runtime: str,
    images: list[str | Path],
    *,
    baseline_precision: str = "fp32",
    compare_precision: str = "fp16",
    conf: float = config.CONF,
    iou: float = config.IOU,
    architecture: str | None = None,
    iou_thres: float = 0.5,
) -> dict:
    """Quantization cost: compare ``compare_precision`` against ``baseline_precision``.

    Both runs use the **same runtime**, so the only difference is the numeric
    precision — that isolates the quantization effect from any runtime/kernel
    differences. Per-image metrics are aggregated with a detection-weighted mean so
    a busy image does not count the same as an empty one.
    """
    items = _load_images(images)
    arch = architecture or config.arch_for(model)

    per_image: list[dict] = []
    for base_prec, cmp_prec in ((baseline_precision, compare_precision),):
        base = Detector(arch, runtime, base_prec, model, conf=conf, iou=iou)
        cmp_ = Detector(arch, runtime, cmp_prec, model, conf=conf, iou=iou)
        try:
            base.load()
            cmp_.load()
        except Exception:
            # Always release the half-loaded pair before propagating.
            base.release()
            cmp_.release()
            raise
        try:
            for name, frame in items:
                ref = base(frame)
                pred = cmp_(frame)
                stats = _agreement(pred, ref, iou_thres)
                stats["image"] = name
                per_image.append(stats)
        finally:
            base.release()
            cmp_.release()

    ref_total = sum(s["ref_count"] for s in per_image)
    matched_total = sum(s["matched"] for s in per_image)
    weights = np.array([max(1, s["ref_count"]) for s in per_image], dtype=np.float64)

    def _wmean(key: str) -> float:
        vals = np.array([s[key] for s in per_image], dtype=np.float64)
        ok = ~np.isnan(vals)
        if not ok.any():
            return float("nan")
        return float(np.average(vals[ok], weights=weights[ok]))

    return {
        "architecture": arch,
        "model": model,
        "runtime": runtime,
        "baseline_precision": baseline_precision,
        "compare_precision": compare_precision,
        "images": len(per_image),
        "iou_thres": iou_thres,
        # Detection-weighted, so images with more objects dominate the average.
        "recall": matched_total / ref_total if ref_total else 1.0,
        "mean_iou": _wmean("mean_iou"),
        "mean_dconf": _wmean("mean_dconf"),
        "false_positives_per_image": float(
            np.mean([s["false_positives"] for s in per_image])
        ),
        "ref_dets_per_image": ref_total / len(per_image) if per_image else 0.0,
        "per_image": per_image,
    }


def agreement_row(result: dict) -> dict:
    """Flatten a :func:`precision_agreement` result into a CSV-friendly row."""
    return {k: v for k, v in result.items() if k != "per_image"}


AGREEMENT_FIELDS = (
    "architecture", "model", "runtime",
    "baseline_precision", "compare_precision",
    "images", "iou_thres", "recall", "mean_iou", "mean_dconf",
    "false_positives_per_image", "ref_dets_per_image",
)


def write_agreement_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(AGREEMENT_FIELDS),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_agreement_csv(path: str | Path):
    import pandas as pd

    return pd.read_csv(path)


def coco_map(*args, **kwargs):
    """True mAP — not implemented; documented so the gap is explicit.

    A real accuracy number needs a labelled dataset. The path would be
    ``ultralytics``' own validator on the *exported* artifact, e.g. building a
    ``YOLO(engine_path)`` and calling ``.val(data="coco.yaml")``, which requires
    the dataset locally and (for the ONNX/engine artifacts) their own val loader.
    That is a separate piece of work from benchmark timing, so it is called out
    rather than half-built.
    """
    raise NotImplementedError(
        "True mAP needs a labelled dataset. Use `precision_agreement` for the "
        "label-free quantization cost, or run ultralytics .val() on the exported "
        "artifact with a local dataset yaml."
    )