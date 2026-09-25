# YOLO → ONNX → TensorRT Benchmark Harness

Three goals, in priority order:

1. **Benchmark** Ultralytics YOLO checkpoints of the same complexity (e.g.
   `yolo11s` vs `yolo26s`, `yolo11l` vs `yolo26l`) across four runtimes and two
   precisions on NVIDIA GPUs, measuring **framerate**, **latency**, and
   **memory footprint**.
2. **Reusable wrapper** — a ``Detector`` that swaps runtimes transparently, to be
   imported by upcoming projects.
3. **Architecture pluggability** — pre/post is per-detector-family, so other
   architectures (e.g. RF-DETR, with different pre/post **and** a less
   problematic Apache-2.0 license vs Ultralytics AGPL) drop in later without
   touching any runtime or the wrapper.

| Runtime | Backend | Precision |
|---|---|---|
| `pytorch` | PyTorch CUDA | FP32 / FP16 (`.half()`) |
| `ort_cuda` | ONNX Runtime, CUDA EP | FP32 |
| `ort_trt` | ONNX Runtime, TensorRT EP | FP32 / FP16 (`trt_fp16_enable`) |
| `tensorrt` | **native TensorRT** (deserialize `.engine`) | FP32 / FP16 |
e
## The three metrics

| Metric | What we report | Meaning |
|---|---|---|
| **Framerate** | `fps` | 1 / end-to-end time — sustainable throughput. |
| **Latency** | `e2e_ms` (+ `e2e_med_ms`, `e2e_p95_ms`) full loop; `infer_ms` raw forward only | time a frame spends in the pipeline; median is the honest single-frame latency, p95 the tail. |
| **Memory** | `peak_vram_gb` (device) + `host_rss_mb` (process RAM) + `weights_mb` (artifact on disk) | full footprint across GPU / host / storage. |

## Target hardware

- ✅ **RTX 3070 / 3090** (Ampere, SM 8.6) — TensorRT + FP16 tensor cores.
- ✅ **Jetson Orin Nano** (Ampere, SM 8.7) — TensorRT 10.x ships with JetPack 6.
- ⛔ **GTX 960M** (Maxwell, SM 5.0) — **cannot run TensorRT at all** (requires
  SM 7.5+; ONNX Runtime's TensorRT EP bundles TRT 8.6–10.x which needs SM 7.0+).
  It can still run `pytorch`/`ort_cuda` FP32 with pinned old PyTorch ≤ 2.7, but is
  out of scope here.

> **TensorRT engines are not portable.** An `.engine` is bound to the building
> GPU architecture *and* the TensorRT/CUDA version. Build engines **on each
> target device** (see "Move to another machine" below).

## What this project does *not* do for you

- `.pt` checkpoints are FP32 and come from Ultralytics; they are never
  re-quantized *to disk* in FP16. Instead, we build/run with FP16 where the
  runtime supports it. (16-bit "checkpoints" don't exist as a first-class
  artifact; FP16 is an *execution mode*, not a stored format.)
- The FP16 path is the 16-bit variant we benchmark (≈2× on Ampere via tensor
  cores). BF16 is available implicitly via `half()` replacement but is not wired
  as a separate precision; TF32 is a *32-bit* tensor-core mode that we disable
  for a strict FP32 baseline.

## Quick start

```bash
# 1. Environment (Python 3.12 recommended)
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt       # ultralytics, onnx, onnxruntime-gpu, tensorrt

# 2. Put checkpoints and a sample clip in place
#    models/yolo11s.pt  models/yolo11l.pt  models/yolo26s.pt  models/yolo26l.pt
#    data/sample.mp4

# 3. Convert: .pt -> .onnx -> .engine (fp32 + fp16)
python -m yolo_bench export --models yolo11s yolo11l yolo26s yolo26l \
    --precisions fp32 fp16 --tf32-off

# 4. Correctness gate (backends must match ultralytics .predict())
python -m yolo_bench check --image data/bus.jpg --model yolo11s --all

# 5. Benchmark the full matrix
python -m yolo_bench compare --video data/sample.mp4 --frames 100 \
    --models yolo11s yolo11l yolo26s yolo26l \
    --runtimes pytorch ort_cuda ort_trt tensorrt --precisions fp32 fp16
```

Results land in `results/compare.csv` and `results/compare.md`.

## Commands

| Command | Purpose |
|---|---|
| `python -m yolo_bench export ...` | Convert `.pt` → `.onnx` → `.engine` (FP32 + FP16) |
| `python -m yolo_bench check ...` | Correctness gate vs `ultralytics .predict()` |
| `python -m yolo_bench bench ...` | Benchmark **one** runtime/precision/model on an mp4 |
| `python -m yolo_bench compare ...` | Run the full matrix → CSV + Markdown |

### `bench` (single config)

```bash
python -m yolo_bench bench --runtime tensorrt --precision fp16 \
    --model yolo11s --video data/sample.mp4 --frames 100 --out results/row.tsv
```

Timings are all CUDA-synchronized and reported separately:

- `e2e_ms` — full frame loop (letterbox + infer + NMS) ← the headline number
- `infer_ms` — raw forward pass only
- `pre_ms` / `post_ms` — shared letterbox / decode+NMS
- `infer_med_ms` / `infer_p95_ms` — tail latency
- `fps` — based on e2e
- `peak_vram_gb` — peak device memory

## How conversion works

1. **`.pt` → `.onnx`** via `ultralytics` (`format='onnx', imgsz=640, opset=17,
   simplify=True`). The classic `Detect` head bakes in DFL + `dist2bbox` + sigmoid,
   so the ONNX output is `(1, 4+nc, num_pred)` with boxes already in `xywh`
   (640-px space) and per-class scores already sigmoided. `--tf32-off` disables
   TF32 during trace for a strict FP32 graph.
2. **`.onnx` → `.engine`** via `trtexec` (transparent, logs tactics): `--fp16`
   for FP16, plain FP32 otherwise, `--noTF32` with `--tf32-off`.
3. Metadata (`nc`, input shape, arch, TRT version) is written to
   `models/{name}.meta.json`.
4. `--cross-check` additionally builds an FP16 engine through Ultralytics' own
   `format='engine', quantize=16` path for validation.

## Reusable wrapper + architecture pluggability

The key abstraction is the `Detector` wrapper, which ties a **detector family**
(an `Architecture`) to a **runtime backend** (a `RuntimeExecutor`):

```python
from yolo_bench import Detector

# Swap the runtime transparently:
det = Detector("yolo", "tensorrt", "fp16", "yolo11s")  # or "ort_trt", "pytorch", ...
det.load()
boxes = det(frame_bgr)                 # (M,6) [x1,y1,x2,y2,conf,cls]
det.release()
```

- **Runtime** (`runtimes/`) does exactly one thing: a raw forward pass
  `infer(blob) -> raw`. It knows *nothing* about detector families.
- **Architecture** (`archs/`) owns a family's pre/post (`prepare`/`decode`) and
  artifact resolution (`checkpoint_path`/`onnx_path`/`engine_path`). YOLO lives
  in `archs/yolo.py`; **RF-DETR** is scaffolded in `archs/rfdetr.py` (different
  pre/post, Apache-2.0 license) and only needs `prepare`/`decode`/`torch_runner`
  implemented — no runtime or wrapper changes.

This separation is deliberate: pre/post for YOLO (letterbox + decode + NMS) is a
"library" concept, not part of the runtime, so adding a second architecture is a
one-file change rather than a rewrite of every backend.

## Cross-check (native vs. Ultralytics-exported engine)

An engine built by **trtexec** should match one built by **Ultralytics
`format='engine'`** for the same precision. Use `--cross-check` to emit both and
compare detections with the `check` command.

## Moving to another machine (3070 / 3090 / Orin Nano)

1. `git clone` (this repo) + recreate the venv with `requirements.txt`.
2. Copy the **`.pt` checkpoints and `data/sample.mp4`** (not the `.engine`/`.onnx` —
   they are arch/version-bound and will be rebuilt).
3. Re-run `export` **on the target device** and then `check`/`compare`.
4. On **Orin Nano** (JetPack 6, arm64): `python` lacks `cuXXX` torch wheels; use
   the JetPack TensorRT and the Jetson Zoo `onnxruntime` build. `trtexec` lives at
   `/usr/src/tensorrt/bin/trtexec`.

## Results interpretation (Rough expectations)

- `torch FP32` < `ort_cuda FP32` — ORT CUDA kernels / graph rewrite.
- `ort_trt FP16` ≈ `tensorrt FP16` — both run the same TensorRT builder underneath.
- `tensorrt FP16` ≤ `tensorrt FP32` — FP16 tensor-core throughput on Ampere.
- `ort_trt` first run is slow (engine build); warm-up / engine-cache amorts this.

## Project layout

```
yolo_bench/
  config.py      # paths, models, precision/runtime lists, arch registry
  export.py      # .pt -> .onnx -> .engine
  detector.py    # Detector: transparent runtime + architecture swap
  bench.py       # single-config benchmark driver (fps/latency/memory)
  compare.py     # full matrix -> CSV + Markdown
  check.py       # correctness gate vs ultralytics .predict()
  utils.py       # timing, metadata, device helpers
  archs/
    base.py       # Architecture ABC (prepare/decode/artifact resolution)
    yolo.py       # YOLO Detect head: letterbox + decode + NMS
    rfdetr.py     # RF-DETR scaffold (permissive license; pre/post TODO)
  runtimes/
    base.py       # RuntimeExecutor ABC (raw forward pass)
    pytorch.py    # PyTorch CUDA
    ort.py        # ORT CUDA EP + ORT TensorRT EP
    tensorrt.py   # native TensorRT (deserialize .engine)
models/          # .pt / .onnx / .engine / .meta.json  (git-ignored)
data/            # sample mp4                          (git-ignored)
results/         # compare.csv / compare.md            (git-ignored)
```

## Known limitations / notes

- On the first `ort_trt` run, TensorRT builds the engine (can take minutes);
  enable `trt_engine_cache_enable` (already set) so later runs load the cache.
- `ort_cuda` is FP32-only; FP16 on the CUDA EP is not meaningfully faster, so
  only `fp32` is meaningful there.
- FP16 compare on the 960M would be misleading (no tensor cores); it's excluded.
- Checkpoint download requires internet on first `export` if `.pt` files aren't
  already in `models/`.