"""Shared utilities: timing, model metadata, device helpers."""

from __future__ import annotations

import json
import statistics
import subprocess
from pathlib import Path

import numpy as np

# torch is an *optional* import here so the package (and the reusable Detector
# wrapper) stays importable on a machine without PyTorch — e.g. a TensorRT-only
# Jetson image. Functions that need torch return safe defaults when it's absent.
try:
    import torch
except Exception:  # pragma: no cover - torch just isn't installed
    torch = None

from . import config


def load_meta(name: str) -> dict | None:
    """Load a model's metadata dict if present, else None."""
    path = config.model_meta(name)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def get_nc(name: str) -> int:
    """Number of classes for a model, from metadata or a safe default."""
    meta = load_meta(name)
    if meta and "nc" in meta:
        return int(meta["nc"])
    return 80


def class_names(nc: int) -> list[str]:
    if nc == 80:
        return config.COCO80
    return [f"class-{i}" for i in range(nc)]


def torch_backend() -> str:
    """Detect whether the active PyTorch install is CUDA / CPU / etc."""
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_mps  # noqa: F401

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def torch_device() -> torch.device:
    backend = torch_backend()
    if backend == "cuda":
        return torch.device("cuda:0")
    return torch.device("cpu")


def cuda_device() -> torch.device:
    assert torch is not None and torch_backend() == "cuda", \
        "CUDA not available in this environment"
    return torch.device("cuda:0")


def current_vram_gb() -> float:
    """Currently allocated+reserved VRAM in GiB, or 0.0 without CUDA."""
    if torch is None or torch_backend() != "cuda":
        return 0.0
    return torch.cuda.memory_reserved() / (1024**3)


def peek_vram_gb() -> float:
    """Peak allocated VRAM in GiB, or 0.0 without CUDA."""
    if torch is None or torch_backend() != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def reset_vram(track: bool = True) -> None:
    """Free cache and reset the peak-memory counter if tracking."""
    if torch is None or torch_backend() != "cuda":
        return
    torch.cuda.empty_cache()
    if track:
        torch.cuda.reset_peak_memory_stats()


def current_host_rss_mb() -> float:
    """Current process resident-set size in MiB (host RAM footprint).

    Together with ``peek_vram_gb`` this gives a complete picture of a runtime's
    memory footprint: weights + runtime overhead on the CPU side, and device
    buffers on the GPU side.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # "VmRSS:  123456 kB"
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def summarize(values: list[float]) -> dict[str, float]:
    """mean / median / p95 / min / max over a list of timings (seconds)."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(statistics.median(values)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def nvidia_gpu_name() -> str | None:
    """GPU name from nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip().splitlines()[0]


def detect_arch() -> str | None:
    """Compute-capability string (e.g. '8.6') for the active CUDA device."""
    if torch is None or torch_backend() != "cuda":
        return None
    major, minor = torch.cuda.get_device_capability(0)
    return f"{major}.{minor}"


def tf32_enabled() -> bool:
    """Whether CUDA matmul TF32 is currently on (affects FP32 'baseline')."""
    if torch is None or torch_backend() != "cuda":
        return False
    return bool(torch.backends.cuda.matmul.allow_tf32)


def env_summary() -> dict:
    return {
        "gpu_name": nvidia_gpu_name(),
        "arch": detect_arch(),
        "torch": torch.__version__ if torch is not None else None,
        "torch_backend": torch_backend(),
        "tf32_on": tf32_enabled(),
    }


def bgr_from_frame(frame: np.ndarray) -> tuple[int, int]:
    """Return (height, width) of a cv2 BGR frame."""
    return frame.shape[0], frame.shape[1]


def ensure_dirs() -> None:
    for d in (config.MODELS_DIR, config.DATA_DIR, config.RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def fmt(seconds: float) -> str:
    if seconds <= 0:
        return "0 ms"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"