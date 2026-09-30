"""Configuration for the YOLO TensorRT webcam example.

Paths are resolved relative to the repository root (the parent of this package)
so the project can be moved to another machine (e.g. the RTX 3090 box) and
re-run unchanged.

Scope note
----------
This module holds *paths*, the *models under test*, and the shared *pre/post
constants*. It deliberately does NOT hold the list of runtimes or which
precisions each supports — that lives in ``runtimes/registry.py``, which is the
single source of truth. Previously both modules carried a ``RUNTIMES`` list and
they could drift apart; the aliases at the bottom of this file now derive from
the registry so there is exactly one definition.

Benchmark targets:
  * yolo11s / yolo11l / yolo26s / yolo26l  (COCO detection)
  * precisions: fp32 / fp16 (per-runtime support varies; see the registry)
  * runtimes:   see ``runtimes/registry.py``

NOTE on the GTX 960M (Maxwell, SM 5.0): TensorRT requires SM 7.5+ (and ONNX
Runtime's TensorRT EP bundles TensorRT 8.6-10.x, which needs SM 7.0+). The
960M therefore cannot run the TensorRT-backed backends at all. This project
targets RTX 3070 / 3090 (Ampere, SM 8.6) and Jetson Orin Nano (SM 8.7).
"""

from __future__ import annotations

from pathlib import Path

# --- Repository layout -------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"      # .pt / .onnx / .engine / .meta.json (git-ignored)
DATA_DIR = ROOT / "data"          # optional local media (git-ignored)

# --- Models under test -------------------------------------------------------
DEFAULT_MODEL = "yolo11s"
DEFAULT_PRECISION = "fp16"
DEFAULT_CAMERA = 0

PRECISIONS = ["fp32", "fp16"]

ARCHS = ["yolo"]

def arch_for(model: str) -> str:
    return "yolo"

# --- Pre / post processing ---------------------------------------------------
IMGSZ = 640          # model input size (square)
STRIDE = 32          # YOLO detection stride
CONF = 0.25          # confidence threshold
IOU = 0.45           # NMS IoU threshold
MAX_DET = 300        # max detections per frame
LETTERBOX_COLOR = (114, 114, 114)  # gray, matches Ultralytics default

# COCO-80 class names used for webcam annotations.
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def model_engine(name: str, precision: str) -> Path:
    return MODELS_DIR / f"{name}.{precision}.engine"


def model_meta(name: str) -> Path:
    return MODELS_DIR / f"{name}.meta.json"
