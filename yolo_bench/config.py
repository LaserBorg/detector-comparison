"""Central configuration for the YOLO benchmark harness.

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
DATA_DIR = ROOT / "data"          # sample mp4 clips (git-ignored)
RESULTS_DIR = ROOT / "results"    # csv + markdown (git-ignored)

# --- Models under test -------------------------------------------------------
MODELS = ["yolo11s", "yolo11l", "yolo26s", "yolo26l"]

# --- Defaults for a run ------------------------------------------------------
# The precision set offered by default. Individual runtimes may support less;
# the registry (and ``runtimes.plan``) handles that, reporting unsupported
# combinations as skips rather than failures.
PRECISIONS = ["fp32", "fp16"]

# Re-exported from the registry so existing callers keep working, while the
# registry stays the single source of truth. Import inside the functions below /
# at the bottom to avoid a circular import at module load.

# --- Architecture registry ---------------------------------------------------
# Maps a model (or family) to the detector architecture that knows its pre/post.
# RF-DETR is registered so it can be used with the SAME wrapper/runtimes once
# its pre/post is implemented (scaffolded in archs/rfdetr.py).
ARCHS = ["yolo", "rfdetr"]

# Default architecture for a given model name. The benchmark models are all
# classic YOLO-detect; RF-DETR models would be listed here later.
def arch_for(model: str) -> str:
    if model.startswith("rfdetr"):
        return "rfdetr"
    return "yolo"

# --- Pre / post processing ---------------------------------------------------
IMGSZ = 640          # model input size (square)
STRIDE = 32          # YOLO detection stride
CONF = 0.25          # confidence threshold
IOU = 0.45           # NMS IoU threshold
MAX_DET = 300        # max detections per frame
LETTERBOX_COLOR = (114, 114, 114)  # gray, matches Ultralytics default

# --- ONNX / TensorRT ---------------------------------------------------------
OPSET = 17           # explicit, conservatively supported opset for TRT parser
TRT_WORKSPACE_GB: float | None = None  # None -> let TRT auto-allocate

# COCO-80 class names (used for annotations / logging). For the benchmark we
# only need `nc`, which is read from the exported model, but the names make
# annotated output human-readable.
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


def model_pt(name: str) -> Path:
    return MODELS_DIR / f"{name}.pt"


def model_onnx(name: str) -> Path:
    return MODELS_DIR / f"{name}.onnx"


def model_engine(name: str, precision: str) -> Path:
    return MODELS_DIR / f"{name}.{precision}.engine"


def model_openvino(name: str) -> Path:
    """OpenVINO IR artifact for ``name``.

    OpenVINO imports the ONNX directly, so this normally points at the same
    ``.onnx`` the ORT runtimes consume (kept as a single accessor so a future
    vectorized-IR pipeline can change in one place).
    """
    return MODELS_DIR / f"{name}.onnx"


def model_meta(name: str) -> Path:
    return MODELS_DIR / f"{name}.meta.json"


# --- Registry-derived aliases ------------------------------------------------
# Kept for existing callers. The registry is authoritative; these are views of
# it. ``ALL_RUNTIMES`` is every registered backend; ``RUNTIMES`` remains the
# default *matrix* (CUDA backends first, then CPU baselines).
def _runtimes_alias() -> list[str]:
    from .runtimes.registry import ALL_KINDS

    return list(ALL_KINDS)


def __getattr__(name: str):
    """Lazily expose registry-derived names without an import cycle.

    ``config`` is imported by ``runtimes.registry``'s neighbours during package
    init, so touching the registry at module scope would be circular.
    """
    if name == "RUNTIMES":
        return _runtimes_alias()
    if name == "OPENVINO_DEVICE":
        import os

        return os.environ.get("YOLO_BENCH_OPENVINO_DEVICE", "GPU")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")