"""Full environment fingerprint for cross-machine benchmark provenance.

Why this exists
---------------
This project is meant to be cloned onto several machines (RTX 3090, RTX 3070,
Jetson Orin Nano) and the result CSVs merged. Without hardware and library
provenance, merged rows are not comparable and cannot be attributed:

  * The RTX 3090 runs **TensorRT 11.3**, the Orin (JetPack 7.2) runs **10.16**.
    A TRT-major difference can change engine build and runtime behaviour.
  * CPU backends (`pytorch_cpu`, `ort_cpu`, `openvino_cpu`) are dominated by the
    host CPU — i7-6700K (4c/8t desktop, ~90 W) vs Orin's 6-core Cortex-A78AE at
    7-15 W is not a small difference.
  * On Jetson, the **power mode** (15 W vs MAXN) changes results more than the
    software stack does.
  * Engines are **not portable** across TensorRT majors or GPU architectures, so
    a stale `.engine` produces a deserialization error rather than a wrong number
    (good), but the CSV must record which stack built it.

Everything here is collected defensively: no function may raise, because this
runs on platforms (Jetson, arm64, containers) where the usual probes are absent.
``nvidia-smi`` in particular is unreliable (it fails on a driver/library version
mismatch), so torch is preferred and treated as the source of truth for the GPU.

Each configuration normally runs in its own subprocess, so this is re-collected
per row; it is cheap (a few ``/proc`` reads) and caching across processes would
add complexity for no gain.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import time
from pathlib import Path

# Compute-capability -> marketing architecture name. Used to report
# "Ampere" rather than just 8.6, so a report is readable without a lookup.
# Reference: https://developer.nvidia.com/cuda-gpus
GPU_ARCHITECTURES: dict[str, str] = {
    "5.0": "Maxwell", "5.2": "Maxwell", "5.3": "Maxwell",
    "6.0": "Pascal", "6.1": "Pascal", "6.2": "Pascal",
    "7.0": "Volta", "7.2": "Volta",
    "7.5": "Turing", "8.0": "Ampere", "8.6": "Ampere", "8.7": "Ampere",
    "8.9": "Ada Lovelace", "9.0": "Hopper", "10.0": "Blackwell",
    "10.3": "Blackwell", "12.0": "Blackwell",
}

# Columns produced by `fingerprint()`, in report order. Kept explicit (rather
# than derived) so the CSV schema is stable and auditable.
FINGERPRINT_FIELDS: tuple[str, ...] = (
    # run identity — so rows from different machines/runs never collide
    "run_id", "timestamp", "hostname",
    # platform
    "machine", "os_name", "kernel", "python",
    "is_jetson", "l4t", "jetpack", "nvpmodel",
    # host CPU / RAM (dominates the *_cpu runtimes)
    "cpu", "cpu_cores", "cpu_threads", "host_ram_gb",
    # GPU
    "gpu_name", "gpu_cc", "gpu_arch", "gpu_vram_gb", "gpu_driver",
    # software stack (versions matter: TRT major differs per machine)
    "cuda", "cudnn", "tensorrt", "onnxruntime", "openvino", "ultralytics",
    "torch", "torch_backend", "tf32_on",
)


def _run(cmd: list[str], timeout: float = 4.0) -> str | None:
    """Run a probe command, returning stripped stdout or None. Never raises."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    text = (out.stdout or "").strip()
    return text or None


def _read(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return None


# --- platform ---------------------------------------------------------------

def _is_jetson() -> bool:
    """Jetson detection: the Tegra release file or the device-tree model string."""
    if Path("/etc/nv_tegra_release").exists():
        return True
    model = _read("/proc/device-tree/model")
    return bool(model and "jetson" in model.lower() or model and "tegra" in model.lower())


def _l4t_version() -> str | None:
    """L4T version from /etc/nv_tegra_release, e.g. 'R39.2.1'."""
    text = _read("/etc/nv_tegra_release")
    if not text:
        # Newer Jetson Linux keeps it here instead.
        text = _read("/etc/nv_boot_control.conf")
    if not text:
        return None
    for token in text.replace(",", " ").split():
        if token.startswith("R") and token[1:2].isdigit():
            return token
    return None


def _jetpack_version() -> str | None:
    """JetPack version, derived from L4T. Returns None when not a Jetson."""
    l4t = _l4t_version()
    if not l4t:
        return None
    return f"L4T {l4t} (JetPack; see release notes for exact tag)"


def _nvpmodel() -> str | None:
    """Active Jetson power mode, e.g. 'MAXN'. Dominates Jetson results."""
    out = _run(["nvpmodel", "-q"])
    if not out:
        return None
    for line in out.splitlines():
        if "NV Power Mode" in line and ":" in line:
            return line.split(":", 1)[1].strip()
    return None


# --- host CPU / RAM ---------------------------------------------------------

def _cpu_model() -> str | None:
    """CPU model string. /proc/cpuinfo covers x86 and most arm64 kernels."""
    text = _read("/proc/cpuinfo")
    if text:
        # Only these keys hold a human-readable model. Note that a bare "model"
        # key also exists with a *numeric* value (e.g. "model : 94"), so keys are
        # matched exactly and numeric values are rejected below.
        wanted = ("model name", "hardware", "cpu model", "model")
        for line in text.splitlines():
            key, sep, value = line.partition(":")
            if not sep:
                continue
            key = key.strip().lower()
            value = value.strip()
            if key in wanted and value and not value.isdigit():
                # "model" is a weak match; prefer the strong keys by iterating
                # them in priority order.
                if key != "model":
                    return value
        for line in text.splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip().lower() == "model":
                value = value.strip()
                if value and not value.isdigit():
                    return value
    # Fallbacks: platform module, then lscpu.
    out = _run(["lscpu"])
    if out:
        for line in out.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    name = platform.processor() or None
    if name:
        return name
    model = _read("/proc/device-tree/model")
    return model.strip("\x00").strip() if model else None


def _cpu_cores() -> tuple[int | None, int | None]:
    """(physical cores, logical threads).

    Physical is parsed from /proc/cpuinfo where available; on arm64 big.LITTLE
    parts this can under-report, so callers should prefer `cpu_threads`.
    """
    threads = os.cpu_count()
    physical = None
    text = _read("/proc/cpuinfo")
    if text:
        # x86 exposes "cpu cores" per socket; multiply by distinct physical ids.
        cores = None
        for line in text.splitlines():
            key, _, value = line.partition(":")
            if key.strip().lower() == "cpu cores":
                try:
                    cores = int(value.strip())
                except ValueError:
                    pass
                break
        sockets = set()
        for line in text.splitlines():
            key, _, value = line.partition(":")
            if key.strip().lower() == "physical id":
                sockets.add(value.strip())
        if cores is not None:
            physical = cores * max(1, len(sockets))
    if physical is None:
        try:
            import psutil  # optional
            physical = psutil.cpu_count(logical=False)
        except Exception:
            physical = None
    return physical, threads


def _host_ram_gb() -> float | None:
    text = _read("/proc/meminfo")
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            try:
                kb = float(line.split()[1])
                return round(kb / (1024 ** 2), 1)
            except (IndexError, ValueError):
                return None
    return None


# --- GPU --------------------------------------------------------------------

def _gpu_from_torch() -> dict:
    """GPU facts via torch — the most reliable source here.

    Preferred over ``nvidia-smi`` because nvidia-smi can fail entirely on a
    driver/library version mismatch (observed on this host: NVML 595.91 vs the
    loaded driver), while CUDA still works fine.
    """
    out: dict = {}
    try:
        import torch

        if not torch.cuda.is_available():
            return out
        props = torch.cuda.get_device_properties(0)
        major, minor = torch.cuda.get_device_capability(0)
        cc = f"{major}.{minor}"
        out["gpu_name"] = torch.cuda.get_device_name(0)
        out["gpu_cc"] = cc
        out["gpu_arch"] = GPU_ARCHITECTURES.get(cc, "unknown")
        out["gpu_vram_gb"] = round(props.total_memory / (1024 ** 3), 1)
        if torch.version.cuda:
            out["cuda"] = torch.version.cuda
        try:
            # torch reports cuDNN as an integer like 92400 (i.e. 9.24.0);
            # format it so two machines' values are readable and comparable.
            cudnn = torch.backends.cudnn.version()
            if cudnn:
                s = str(cudnn)
                out["cudnn"] = ".".join(s[i:i + (1 if i == 0 else 2)]
                                         for i in range(0, len(s), 2)) \
                    if len(s) >= 5 else s
        except Exception:
            pass
    except Exception:
        pass
    return out


def _gpu_driver() -> str | None:
    """Driver version, from nvidia-smi or /proc/driver/nvidia/version."""
    out = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if out:
        first = out.splitlines()[0].strip()
        if first and "nvml" not in first.lower() and "failed" not in first.lower():
            return first
    # Jetson / containers without nvidia-smi.
    text = _read("/proc/driver/nvidia/version")
    if text:
        for line in text.splitlines():
            if "Kernel Module" in line:
                parts = line.split()
                for token in parts:
                    if token[:1].isdigit() and "." in token:
                        return token
    return None


# --- library versions -------------------------------------------------------

def _module_version(name: str) -> str | None:
    """Version string for an importable package, without importing it if possible.

    Uses importlib.metadata first (cheap, no side effects); falls back to a real
    import only if the distribution name doesn't match the module name.
    """
    try:
        from importlib.metadata import version as _v

        for dist in (name, name.replace("_", "-"), f"{name}-gpu"):
            try:
                return _v(dist)
            except Exception:
                continue
    except Exception:
        pass
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", None) or getattr(mod, "version", None)
    except Exception:
        return None


def _tensorrt_version() -> str | None:
    return _module_version("tensorrt")


def _torch_facts() -> dict:
    out: dict = {}
    try:
        import torch

        out["torch"] = torch.__version__
        out["torch_backend"] = "cuda" if torch.cuda.is_available() else "cpu"
        out["tf32_on"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        pass
    return out


# --- public API -------------------------------------------------------------

def _new_run_id() -> str:
    """Short, sortable-ish id: <host>-<UTC timestamp>."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{socket.gethostname().split('.')[0]}-{stamp}"


def fingerprint(run_id: str | None = None) -> dict:
    """Collect the full environment fingerprint as a flat dict of strings.

    Every value is JSON/CSV friendly (str, float, bool or None). Never raises.
    """
    physical, threads = _cpu_cores()
    is_jetson = _is_jetson()
    fp: dict = {
        "run_id": run_id or _new_run_id(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "os_name": (platform.freedesktop_os_release().get("PRETTY_NAME")
                    if hasattr(platform, "freedesktop_os_release") else None)
                   or platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "is_jetson": is_jetson,
        "l4t": _l4t_version(),
        "jetpack": _jetpack_version() if is_jetson else None,
        "nvpmodel": _nvpmodel() if is_jetson else None,
        "cpu": _cpu_model(),
        "cpu_cores": physical,
        "cpu_threads": threads,
        "host_ram_gb": _host_ram_gb(),
        "onnxruntime": _module_version("onnxruntime"),
        "openvino": _module_version("openvino"),
        "ultralytics": _module_version("ultralytics"),
        "tensorrt": _tensorrt_version(),
    }
    fp.update(_gpu_from_torch())
    fp["gpu_driver"] = _gpu_driver()
    fp.update(_torch_facts())

    # Normalise: every field present, missing ones as None so the CSV has stable
    # columns regardless of platform.
    return {k: fp.get(k) for k in FINGERPRINT_FIELDS}


def human_summary(fp: dict | None = None) -> str:
    """One-glance multi-line summary for logs and notebook headers."""
    fp = fp or fingerprint()
    cpu = fp.get("cpu") or "unknown CPU"
    cores = fp.get("cpu_cores")
    threads = fp.get("cpu_threads")
    core_txt = f"{cores}c/{threads}t" if cores else f"{threads}t"

    gpu = fp.get("gpu_name") or "no CUDA device"
    if fp.get("gpu_arch"):
        gpu += f" ({fp['gpu_arch']}, SM {fp['gpu_cc']}, {fp.get('gpu_vram_gb')} GB)"

    lines = [
        f"host      : {fp.get('hostname')}  [{fp.get('machine')}]",
        f"os        : {fp.get('os_name')} (kernel {fp.get('kernel')})",
        f"cpu       : {cpu}  [{core_txt}, {fp.get('host_ram_gb')} GB RAM]",
        f"gpu       : {gpu}",
        f"driver    : {fp.get('gpu_driver')}",
        f"cuda/cudnn: {fp.get('cuda')} / {fp.get('cudnn')}",
        f"libs      : tensorrt {fp.get('tensorrt')}, onnxruntime "
        f"{fp.get('onnxruntime')}, openvino {fp.get('openvino')}, "
        f"torch {fp.get('torch')} ({fp.get('torch_backend')})",
    ]
    if fp.get("is_jetson"):
        lines.append(f"jetson    : L4T {fp.get('l4t')}, power mode {fp.get('nvpmodel')}")
    lines.append(f"run       : {fp.get('run_id')} at {fp.get('timestamp')}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(human_summary())