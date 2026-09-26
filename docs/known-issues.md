# Known limitations & gotchas

- On the first `ort_trt` run, TensorRT builds the engine (can take minutes);
  enable `trt_engine_cache_enable` (already set) so later runs load the cache.
- Engine builds are slow: ~64 s for `yolo11s` fp32 and ~100 s for fp16 (the fp16
  path also runs ModelOpt AutoCast first). `.engine` sizes: 224 MB fp32 vs 131 MB
  fp16 for `yolo11s`.
- `onnxruntime-gpu` must stay on a **CUDA 13 build** here. Installing
  `nvidia-modelopt` downgrades it to a CUDA 12/cuDNN 9 build whose CUDA EP fails
  to load and **silently falls back to CPU** (which would invalidate every
  `ort_cuda`/`ort_trt` number). Re-assert with
  `pip install --no-deps onnxruntime-gpu==1.30.0`.
- If ORT reports `libcublasLt.so.13`/`libcudnn.so.9` missing, add the pip NVIDIA
  lib dirs to `LD_LIBRARY_PATH` (see `requirements.txt`).
- `ort_cuda` is FP32-only; FP16 on the CUDA EP is not meaningfully faster, so
  only `fp32` is meaningful there.
- **`ort_trt` requires ORT's TensorRT EP to match the installed TensorRT major.**
  The ORT TensorRT EP links against a specific `libnvinfer.so.N`, so on a host
  with TensorRT 11.x (say) but an ORT build expecting `libnvinfer.so.10`, the EP
  fails to load and ORT silently falls back to the CUDA EP — which would make the
  `ort_trt` row a plain-CUDA measurement. yolo_bench now **fails loudly** instead
  (see `RuntimeExecutor` provider assertion).
- **`peak_vram_gb` is measured device-wide** (`total - free`), sampled inside the
  frame loop. `torch.cuda.max_memory_allocated()` only sees PyTorch's own
  allocator and reports ~0.01 GB for the `tensorrt` / `ort_*` runtimes, because
  their weights live in TensorRT's / ORT's allocators. `base_vram_gb` records the
  pre-run baseline so the model's own footprint is `peak - base`.
- **Eager PyTorch FP16 can be slower than FP32** on this workload (measured: 10.5
  vs 8.7 ms for `yolo11s`), even with every parameter in half. The model is small
  (9.4 GFLOPs) and latency-bound, so kernel-launch and cast overhead dominate the
  FLOP saving. The tensor-core win appears in TensorRT FP16 (2.6 vs 4.1 ms).
  Treat eager-PyTorch FP16 as a baseline, not a speed target.
- **The ONNX frontend of OpenVINO prints hundreds of "N warnings generated"
  lines** while importing the model. Cosmetic. Do *not* try to silence it with
  `core.set_property({'LOG_LEVEL': ...})` — that suppresses GPU plugin init and
  `available_devices` silently drops `['CPU','GPU']` → `['CPU']`.
- **OpenVINO 2026.4 segfaults when a second `ov.Core()` is created**, on the
  second executor regardless of device, which breaks a multi-runtime `compare`.
  `runtimes/openvino.py` therefore uses one process-wide `ov.Core()` singleton.
- **CPU execution providers have no FP16.** `pytorch_cpu` / `ort_cpu` /
  `openvino_cpu` are FP32-only and refuse `fp16` (`RuntimeExecutor.
  supported_precisions`) rather than silently running an FP32 graph labelled FP16.
- FP16 compare on the 960M would be misleading (no tensor cores); it's excluded.
- Checkpoint download requires internet on first `export` if `.pt` files aren't
  already in `models/`.
- `host_rss_mb` is unreliable in `compare` because all configurations share one
  process and RSS accumulates; measure it with per-config `bench` runs.
- **NVENC needs a free CUDA context.** `predict --encoder nvenc` (or `auto`)
  fails with `CUDA_ERROR_OUT_OF_MEMORY` / "No capable devices found" when the
  GPU's VRAM is fully occupied by another process (e.g. a local LLM server).
  `auto` falls back to `libx264`/`mp4v` in that case; force `nvenc` to fail
  loudly instead. In production this is not an issue — NVENC coexists with
  inference as long as a few hundred MB of VRAM are free.
- **`predict` end-to-end fps ≠ `bench` fps.** `bench` times the detection
  pipeline only (CUDA-synchronized); `predict` wall time also includes video
  decode, box drawing, and output encoding. A software encoder costs ~11 ms/
  frame at 1080p, so a 170 fps detector writes only ~55 fps of mp4. The
  summary line prints both numbers; use `--encoder nvenc` to remove the
  encoder from the bottleneck.
