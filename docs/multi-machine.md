# Multi-machine: 3070 / 3090 / Orin Nano

## Target hardware

- ✅ **RTX 3070 / 3090** (Ampere, SM 8.6) — TensorRT + FP16 tensor cores.
- ✅ **Jetson Orin Nano** (Ampere, SM 8.7) — JetPack **7.2** (Jetson Linux 39.2.1,
  Ubuntu 24.04, CUDA 13.2.2, TensorRT 10.16.2) ships `trtexec` and the TensorRT
  Python bindings. See the Orin-specific notes below.
- ⛔ **GTX 960M** (Maxwell, SM 5.0) — **cannot run TensorRT at all** (requires
  SM 7.5+; ONNX Runtime's TensorRT EP bundles TRT 8.6–10.x which needs SM 7.0+).
  It can still run `pytorch`/`ort_cuda` FP32 with pinned old PyTorch ≤ 2.7, but
  is out of scope here.

## Moving to another machine

1. `git clone` (this repo) + recreate the environment from `requirements.txt`.
2. Copy the **`.pt` checkpoints, the `.onnx` files, the `.meta.json`, and the
   sample clip** (`data/sample_1080p_h264.mp4`). **Do not copy the `.engine`
   files** — they are the only arch/TRT-version-bound artifacts and are rebuilt
   on the target. ONNX is a portable IR, so the `.onnx` (and `.pt`) travel
   freely between machines:

   ```bash
   scp models/*.pt models/*.onnx models/*.meta.json user@target:~/detector-comparison/models/
   scp data/sample_1080p_h264.mp4 user@target:~/detector-comparison/data/
   ```

   The `.fp16.onnx` files (ModelOpt AutoCast) are only consumed by **TRT ≥ 11**;
   on a TRT 10 target (e.g. JetPack 7.2) the FP16 engine is built from the plain
   `.onnx` with `BuilderFlag.FP16`, so they are optional there.
3. Re-run `export --onnx-only` **on the target device** to build the engines,
   then `check`, then `compare --out results/<tag>` (e.g. `--out
   results/orin-nano`) so the run lands in that machine's canonical CSV. Commit
   the `<tag>.csv` + `<tag>.md` pair; the notebook picks it up automatically.
4. On **Orin Nano** (JetPack 7.2, arm64): the compute stack is now unified with
   Thor — **L4T Ubuntu 24.04, kernel 6.8, CUDA 13.2.2, TensorRT 10.16.2** — so
   JetPack's own TensorRT is the supported path; you do **not** need a pip
   `tensorrt` wheel. The C++ runtime (`libnvinfer10`) ships with JetPack, but
   the **Python bindings are a separate package** and `trtexec` is not installed
   by default:

   ```bash
   sudo apt-get install python3-libnvinfer   # Python bindings for the system TRT
   ```

   This only adds the Python layer on top of the existing `libnvinfer.so.10` —
   it does not upgrade or replace the C++ runtime, so other NVInfer projects on
   the box (e.g. DeepStream apps) are unaffected. If you work in a **conda env**,
   the apt package lands in system `dist-packages`, which conda does not see —
   symlink it in (it links against the system `libnvinfer.so.10`):

   ```bash
   SP=$(python -c 'import site; print(site.getsitepackages()[0])')
   ln -s /usr/lib/python3.12/dist-packages/tensorrt $SP/tensorrt
   ln -s /usr/lib/python3.12/dist-packages/tensorrt_bindings $SP/tensorrt_bindings
   ```

   Install `onnxruntime-gpu` from the Jetson Zoo build (there is no generic
   aarch64 wheel), and note that since CUDA 13.x matches the host, a `torch`
   built for CUDA 13.x is the closest match (Jetson-specific torch wheels come
   from the Jetson Zoo / NVIDIA containers). With JetPack 7.2 there is no SD
   card image for the Orin Nano Dev Kit — flash via the unified ISO.

## Merging results across machines

Every row carries a full fingerprint (`run_id`, `hostname`, `cpu`, `cpu_cores`,
`cpu_threads`, `host_ram_gb`, `gpu_name`, `gpu_cc`, `gpu_arch`, `gpu_vram_gb`,
`gpu_driver`, `os_name`, `kernel`, `is_jetson`, `l4t`, `nvpmodel`, `cuda`,
`cudnn`, `tensorrt`, `onnxruntime`, `openvino`, `torch`, `runtime_version`, …),
so a merged CSV can be grouped by machine. Check the machine's own summary with:

```bash
python -m yolo_bench.env
```

`nvpmodel` is only populated on Jetson, and it matters more there than the
software stack: an Orin in 15 W mode will lose to one in MAXN by a wide margin.

Three caveats when comparing across machines:

- **`tensorrt` / `ort_trt` rows are not version-comparable.** This box runs
  TensorRT 11.3; JetPack 7.2 ships 10.16. A TRT-major change alters the engine
  builder and the FP16 mechanism (weak-typed `BuilderFlag.FP16` in TRT 10 vs
  graph-level ModelOpt AutoCast in TRT 11), so a cross-machine delta on those
  rows conflates GPU and SDK.
- **`ort_trt` needs an ORT build matching the installed TensorRT major.** ORT
  1.30.0's TensorRT EP links `libnvinfer.so.10`, so it cannot load against TRT 11
  — use a TRT 10 environment, or omit `ort_trt`. The registry reports it as
  skipped rather than silently falling back to another EP.
- **The `*_cpu` rows are a CPU benchmark.** On the Orin they are the headline
  number (6-core Cortex-A78AE, 7–15 W) and will differ hugely from a desktop
  CPU. That is the metric doing its job, not noise.
