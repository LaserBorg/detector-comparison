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

| Runtime | Backend | Device | Precision |
|---|---|---|---|
| `pytorch` | PyTorch | CUDA | FP32 / FP16 (`.half()`) |
| `ort_cuda` | ONNX Runtime, CUDA EP | CUDA | FP32 |
| `ort_trt` | ONNX Runtime, TensorRT EP | CUDA | FP32 / FP16 (`trt_fp16_enable`) |
| `tensorrt` | **native TensorRT** (deserialize `.engine`) | CUDA | FP32 / FP16 |
| `openvino` | OpenVINO, GPU device | NVIDIA/Intel GPU | FP32 / FP16 (hint) |
| `pytorch_cpu` | PyTorch | **CPU** | FP32 |
| `ort_cpu` | ONNX Runtime, CPU EP | **CPU** | FP32 |
| `openvino_cpu` | OpenVINO, CPU device | **CPU** | FP32 |

The three `*_cpu` rows are true no-accelerator baselines: CPU execution providers
have **no FP16 kernels**, so FP16 is not offered there (a request is refused
rather than silently running FP32).
e
## The three metrics

| Metric | What we report | Meaning |
|---|---|---|
| **Framerate** | `fps` | 1 / end-to-end time — sustainable throughput. |
| **Latency** | `e2e_ms` (+ `e2e_med_ms`, `e2e_p95_ms`) full loop; `infer_ms` raw forward only | time a frame spends in the pipeline; median is the honest single-frame latency, p95 the tail. |
| **Memory** | `peak_vram_gb` (device) + `host_rss_mb` (process RAM) + `weights_mb` (artifact on disk) | full footprint across GPU / host / storage. |

## Target hardware

- ✅ **RTX 3070 / 3090** (Ampere, SM 8.6) — TensorRT + FP16 tensor cores.
- ✅ **Jetson Orin Nano** (Ampere, SM 8.7) — JetPack **7.2** (Jetson Linux 39.2.1,
  Ubuntu 24.04, CUDA 13.2.2, TensorRT 10.16.2) ships `trtexec` and the TensorRT
  Python bindings. See "Moving to another machine" for the Orin-specific notes.
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
# 1. Environment (Python 3.12 — a conda env works just as well as a venv)
conda create -n py312 python=3.12 -y && conda activate py312
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt       # ultralytics, onnx, onnxruntime-gpu, tensorrt

# 2. Put checkpoints in models/ (official Ultralytics assets; yolo26* only
#    exists from v8.4.0 onward). -L follows redirects, -o writes into models/.
BASE=https://github.com/ultralytics/assets/releases/download/v8.4.0
mkdir -p models data results
for m in yolo11s yolo11l yolo26s yolo26l; do
  curl -L --fail -o "models/$m.pt" "$BASE/$m.pt"
done

# 3. Put the sample clip + a reference image in data/
#    data/sample_1080p_h264.mp4    (your own clip, 1080p H.264)
curl -L --fail -o data/bus.jpg https://ultralytics.com/images/bus.jpg

# 4. Convert: .pt -> .onnx -> .engine (fp32 + fp16)
python -m yolo_bench export --models yolo11s yolo11l yolo26s yolo26l \
    --precisions fp32 fp16 --tf32-off

# 5. Correctness gate (backends must match ultralytics .predict())
python -m yolo_bench check --image data/bus.jpg --model yolo11s --all

# 6. Benchmark the full matrix
python -m yolo_bench compare --video data/sample_1080p_h264.mp4 --frames 100 \
    --models yolo11s yolo11l yolo26s yolo26l \
    --runtimes pytorch ort_cuda tensorrt openvino \
               pytorch_cpu ort_cpu openvino_cpu --precisions fp32 fp16
```

Results land in `results/compare.csv` and `results/compare.md` (or under the
`--out` prefix you pass). The tables below were produced with `--out results/final`.

## Two layers: inference vs. harness

The project is deliberately split so the inference code can be imported by other
projects while the test machinery stays out of the way.

**Inference tier — reusable Python, no CLI, no orchestration:**

```python
from yolo_bench import Detector

det = Detector("yolo", "tensorrt", "fp16", "yolo11s")
det.load()
boxes = det(frame_bgr)        # (M, 6) [x1,y1,x2,y2,conf,cls]
det.release()
```

Nothing here imports `ultralytics`, `pandas` or `plotly`: the compiled-artifact
backends (`tensorrt`, `ort_*`, `openvino`) are pure ONNX/engine consumers. Only
the `pytorch*` runtime and the `.pt → .onnx` export need `ultralytics`.

**Harness tier — a separate namespace:**

```python
from yolo_bench import orchestrate

rows = orchestrate.run(runtimes=["tensorrt", "ort_cuda"],
                       precisions=["fp32", "fp16"], frames=100)
```

and for charts, `from yolo_bench import plots`.

| Concern | Where it lives |
|---|---|
| Runtime set + per-runtime precisions | `runtimes/registry.py` (**single source of truth**) |
| Pre/post per family | `archs/` |
| Test orchestration (matrix, isolation, experiments) | `orchestrate/` |
| Charts | `plots.py` |
| Hardware/library provenance | `env.py` |
| Notebook workflow | `results.ipynb` |

`config.py` no longer keeps its own runtime list — the registry is authoritative
and `config.RUNTIMES` is a view of it, so the two cannot drift apart.

### Why a subprocess per configuration

Every configuration runs in a **fresh subprocess**. Running a full matrix in one
interpreter made `host_rss_mb` climb monotonically (1.2 → 4.5 GB) purely from
allocator/RSS accumulation, so the memory column was really measuring run order.
Device memory has the same problem (allocator caches and TensorRT/ORT arenas
outlive `release()`). Isolation costs a process spawn and makes the numbers mean
something. Use `--no-isolate` only when you do not care about memory.

### Every row is fingerprinted

Results are merged across the 3090, a 3070 and a Jetson Orin Nano, so each row
records 32 provenance fields: `run_id`/`timestamp`/`hostname`, CPU model + cores +
RAM, GPU name/CC/architecture/VRAM/driver, OS/kernel/python, and library versions
(`tensorrt`, `onnxruntime`, `openvino`, `torch`, …), plus a per-row
`runtime_version`. Without this, a TensorRT-11 row and a TensorRT-10 row look
identical in a merged CSV.

## Notebook workflow

`results.ipynb` runs experiments and draws the charts. Cells:

1. environment fingerprint of the machine you are on,
2. list the named experiments (`orchestrate.EXPERIMENTS`),
3. a cheap `smoke` probe, then a full matrix,
4. **`fps` nested bars** — outer bar FP32, inner bar FP16 inset per model,
5. grouped bars and a log-scale speedup chart,
6. cross-machine comparison from merged `results/*.csv`.

Reading the nested chart: the **wide** bar is FP32 and the **narrow** bar laid over
it is FP16. When FP16 is faster the narrow bar pokes *out of the top* of the wide
one; when FP16 is slower it sits *entirely inside* it. Either way the visible
difference is directly the FP16 effect, and a gap means the combination was
skipped. Cells are labelled `FP32→FP16` fps.

That geometry is deliberate. The obvious alternative — two full-width bars on top
of each other — fails in the one case that matters: whenever FP16 wins, the inner
bar covers the outer one completely, so "FP16 much faster" and "FP16 exactly equal"
render identically. Making the inner bar narrow keeps the taller/shorter
relationship readable in both directions, and the bars are scaled as a quotient of
the cell maximum so their ratio is preserved.

A grouped-bar chart and a relative-speed chart are provided as companions, because
nesting hides part of a bar and is therefore a poor way to answer "which runtime
wins for this model?" (see `plots.fps_grouped_bars`, `plots.speedup_vs_reference`).

## Commands

| Command | Purpose |
|---|---|
| `python -m yolo_bench list` | Show the runtime registry and named experiments |
| `python -m yolo_bench export ...` | Convert `.pt` → `.onnx` → `.engine` (FP32 + FP16) |
| `python -m yolo_bench check ...` | Correctness gate vs `ultralytics .predict()` |
| `python -m yolo_bench bench ...` | Benchmark **one** runtime/precision/model on an mp4 |
| `python -m yolo_bench compare ...` | Run a matrix → CSV + Markdown |
| `python -m yolo_bench run <name> ...` | Run a named experiment (e.g. `run precision`) |

Useful `compare` flags:

| Flag | Effect |
|---|---|
| `--experiment <name>` | Run a named setup instead of the full matrix |
| `--group cuda\|cpu` | Restrict to a runtime group |
| `--list-runtimes` | Registry table incl. availability in this env |
| `--list-experiments` | Named setups and what they cover |
| `--no-isolate` | One process for everything (faster, **memory numbers become meaningless**) |

### `bench` (single config)

```bash
python -m yolo_bench bench --runtime tensorrt --precision fp16 \
    --model yolo11s --video data/sample_1080p_h264.mp4 --frames 100 --out results/row.tsv
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
2. **`.onnx` → `.engine`** via the **TensorRT Python API** (`trt.Builder`). This is
deliberate: the PyPI `tensorrt` wheel ships **no `trtexec` binary**, so on a plain
`pip install` (as on the 3090 host) `trtexec` does not exist. `trtexec` is only
present with a full TensorRT install (JetPack, or the NVIDIA tarball); if it *is*
found, `--via trtexec` uses it, otherwise the Python builder is used. Either way
the engine is built on the target device.

   **FP16 detail (TensorRT ≥ 11):** TRT 11 is *strongly-typed only* — it removed
   `BuilderFlag.FP16`/`INT8`, the INT8 calibrator, and per-tensor precision
   setters. FP16 therefore has to be baked into the **ONNX graph** before
   parsing, which we do with NVIDIA **ModelOpt AutoCast** (mixed precision with
   `keep_io_types=True`, so the engine's I/O stays FP32 and fp32/fp16 remain
   directly comparable). On TensorRT 7–10 the classic `BuilderFlag.FP16` flag is
   used instead.
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

An engine built by **yolo_bench** should match one built by **Ultralytics
`format='engine'`** for the same precision. Use `--cross-check` to emit both and
compare detections with the `check` command.

## Moving to another machine (3070 / 3090 / Orin Nano)

1. `git clone` (this repo) + recreate the environment from `requirements.txt`.
2. Copy the **`.pt` checkpoints and the sample clip** (`data/sample_1080p_h264.mp4`)
   (not the `.engine`/`.onnx` — they are arch/version-bound and will be rebuilt).
3. Re-run `export` **on the target device** and then `check`/`compare`.
4. On **Orin Nano** (JetPack 7.2, arm64): the compute stack is now unified with
   Thor — **L4T Ubuntu 24.04, kernel 6.8, CUDA 13.2.2, TensorRT 10.16.2** — so
   JetPack's own TensorRT, `trtexec` (at `/usr/src/tensorrt/bin/trtexec`) and
   Python bindings are the supported path; you no longer need a pip `tensorrt`
   wheel. Install `onnxruntime-gpu` from the Jetson Zoo build (there is no
   generic aarch64 wheel), and note that since CUDA 13.x matches the host, a
   `torch` built for CUDA 13.x is the closest match (Jetson-specific torch wheels
   come from the Jetson Zoo / NVIDIA containers). With JetPack 7.2 there is no SD
   card image for the Orin Nano Dev Kit — flash via the unified ISO.

### Merging results across machines

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
  number (6-core Cortex-A78AE, 7–15 W) and will differ hugely from this desktop
  i7-6700K. That is the metric doing its job, not noise.

## Measured results

**Host:** RTX 3090 (SM 8.6), i7-6700K (4c/8t), CUDA 13.0, TensorRT 11.3.0.99,
torch 2.14.0+cu130, ONNX Runtime 1.30.0, OpenVINO 2026.4.
**Workload:** 100 frames of `data/sample_1080p_h264.mp4` (1920×1080), 50-frame
warm-up, `conf=0.25`, `iou=0.45`. All timings CUDA-synchronized.
Raw data: `results/final.csv`.

> **These numbers are host-specific.** Absolute values move with GPU, driver,
> TensorRT version and clock state; the *ordering* is the transferable result.

### Framerate — `fps` (higher is better)

| runtime | prec | yolo11s | yolo11l | yolo26s | yolo26l |
|---|---|---|---|---|---|
| **tensorrt** | fp32 | 138.6 | 84.7 | 139.9 | 85.4 |
| **tensorrt** | **fp16** | **170.0** | **138.4** | **176.8** | **143.6** |
| ort_cuda | fp32 | 134.9 | 78.8 | 127.4 | 79.7 |
| ort_cuda | fp16 | 135.3 | 79.1 | 131.8 | 79.6 |
| pytorch | fp32 | 82.2 | 51.4 | 65.6 | 46.8 |
| pytorch | fp16 | 71.3 | 42.9 | 54.6 | 40.9 |
| openvino | fp32 | 32.6 | 9.4 | 32.6 | 9.4 |
| openvino | fp16 | 32.7 | 9.4 | 32.8 | 9.5 |
| openvino_cpu | fp32 | 12.5 | 3.5 | 13.3 | 3.5 |
| ort_cpu | fp32 | 11.8 | 3.6 | 13.5 | 2.4 |
| pytorch_cpu | fp32 | 7.6 | 2.6 | 7.7 | 2.4 |
| ort_trt | fp32/fp16 | — ᵃ | — ᵃ | — ᵃ | — ᵃ |

### Latency — `e2e_ms` (full loop, mean) / `infer_ms` (raw forward) (lower is better)

| runtime | prec | yolo11s | yolo11l | yolo26s | yolo26l |
|---|---|---|---|---|---|
| **tensorrt** | fp32 | 7.21 / 4.10 | 11.81 / 8.77 | 7.15 / 4.19 | 11.71 / 8.69 |
| **tensorrt** | **fp16** | **5.88 / 2.67** | **7.23 / 4.05** | **5.66 / 2.68** | **6.97 / 3.95** |
| ort_cuda | fp32 | 7.41 / 4.47 | 12.69 / 9.67 | 7.85 / 4.75 | 12.55 / 9.53 |
| ort_cuda | fp16 | 7.39 / 4.46 | 12.64 / 9.64 | 7.59 / 4.62 | 12.57 / 9.48 |
| pytorch | fp32 | 12.17 / 9.19 | 19.45 / 16.43 | 15.25 / 12.22 | 21.38 / 18.25 |
| pytorch | fp16 | 14.02 / 11.06 | 23.29 / 20.26 | 18.32 / 15.29 | 24.43 / 21.37 |
| openvino | fp32 | 30.70 / 27.95 | 106.34 / 103.50 | 30.65 / 27.80 | 106.72 / 103.83 |
| openvino | fp16 | 30.61 / 27.90 | 105.91 / 103.00 | 30.48 / 27.68 | 105.71 / 102.82 |
| openvino_cpu | fp32 | 79.86 / 77.00 | 288.70 / 285.70 | 75.39 / 72.39 | 286.48 / 283.44 |
| ort_cpu | fp32 | 84.48 / 81.46 | 276.02 / 273.02 | 73.92 / 70.93 | 412.21 / 408.40 |
| pytorch_cpu | fp32 | 131.99 / 129.02 | 390.79 / 387.77 | 129.54 / 126.69 | 423.04 / 419.99 |
| ort_trt | fp32/fp16 | — ᵃ | — ᵃ | — ᵃ | — ᵃ |

### Tail latency — `e2e_p95_ms` (lower is better)

| runtime | prec | yolo11s | yolo11l | yolo26s | yolo26l |
|---|---|---|---|---|---|
| **tensorrt** | fp32 | 8.53 | 12.06 | 7.22 | 11.94 |
| **tensorrt** | **fp16** | 7.72 | 8.67 | 5.76 | 7.04 |
| ort_cuda | fp32 | 7.47 | 12.80 | 9.31 | 12.66 |
| ort_cuda | fp16 | 7.45 | 12.73 | 7.68 | 13.14 |
| pytorch | fp32 | 12.26 | 19.83 | 15.38 | 24.30 |
| pytorch | fp16 | 14.12 | 23.40 | 20.69 | 24.60 |
| openvino | fp32 | 31.08 | 106.78 | 31.11 | 107.21 |
| openvino | fp16 | 31.04 | 106.52 | 30.96 | 106.18 |
| openvino_cpu | fp32 | 103.53 | 302.81 | 82.54 | 300.91 |
| ort_cpu | fp32 | 108.12 | 295.79 | 80.03 | 415.69 |
| pytorch_cpu | fp32 | 174.52 | 416.84 | 138.70 | 464.51 |

Note the tail behaviour on `tensorrt fp16` for the **S** models (`7.72` ms vs a
`5.88` ms mean, and `8.67` vs `7.23` for `yolo11l`): the p95 is inflated relative
to the mean, i.e. a few slow frames drag the tail. The mean-based `fps` is
unaffected, but for a latency-sensitive pipeline use the p95, not the mean.

### Memory — `peak_vram_gb` (device) and `weights_mb` (artifact on disk)

| runtime | prec | vram s | vram l | weights s | weights l |
|---|---|---|---|---|---|
| pytorch | fp32 / fp16 | 0.55 / 0.84 | 1.04 / 0.95 | 19.3 / 20.4 ᵇ | 51.4 / 53.2 ᵇ |
| ort_cuda | fp32 / fp16 | 0.74 / 1.09 | 1.34 / 1.34 | 38.1 / 38.3 | 101.7 / 99.6 |
| **tensorrt** | fp32 / **fp16** | 0.73 / **0.95** | 1.25 / **1.06** | 225.2 / **131.0** | 383.0 / **232.6** |
| openvino | fp32 / fp16 | 0.94 / 0.95 | 1.15 / 1.15 | 38.1 / 38.3 | 101.7 / 99.6 |
| pytorch_cpu | fp32 | 0.77 ᶜ | 0.78 ᶜ | 19.3 | 51.4 |
| ort_cpu | fp32 | 0.77 ᶜ | 0.78 ᶜ | 38.1 | 101.7 |
| openvino_cpu | fp32 | 0.77 ᶜ | 0.78 ᶜ | 38.1 | 99.6 |

<sub>
ᵃ **`ort_trt` could not run on this host.** ONNX Runtime 1.30.0's TensorRT EP links
against `libnvinfer.so.10` (TensorRT 10); this machine has TensorRT 11.3
(`libnvinfer.so.11`) and `libnvinfer.so.10` is absent. The EP fails to load and ORT
would silently fall back to the CUDA EP, so yolo_bench refuses the row instead of
reporting a mislabelled number. Fix by installing TensorRT 10 in a separate env.
ᵇ `weights_mb` is precision-independent for `pytorch` (`.pt`) and
`ort_*`/`openvino` (`.onnx`); the s/l pair shown is yolo11/yolo26 size.
ᶜ The `*_cpu` rows use no device memory, so their `peak_vram_gb` is just the
ambient baseline of the process (≈0.77 GB) plus other processes on the shared GPU —
it is not a footprint of the model.
</sub>

### What the numbers show

1. **`tensorrt fp16` wins on every model.** 170–177 fps (S) and 138–144 fps (L),
   i.e. **1.2–1.7× over `tensorrt fp32`** and **~2.1× over eager PyTorch fp32**.
   Its `infer_ms` roughly halves (4.10 → 2.67 ms for S): that is the Ampere
   tensor-core win.
2. **`yolo26` ≈ `yolo11` at equal size**, with `yolo26` marginally ahead on the
   S models (170.0 → 176.8 fps fp16) and level on L (138.4 vs 143.6). The
   rewritten head is not slower.
3. **FP16 helps only where the tensor cores are actually used.** `tensorrt` improves
   sharply; `ort_cuda` barely moves (4.47 → 4.46 ms) because the CUDA EP has no
   tensor-core FP16 path here; **eager PyTorch gets slower** (9.19 → 11.06 ms) — at
   9.4 GFLOPs the model is latency-bound, so cast/launch overhead exceeds the FLOP
   saving. Do not assume "fp16 = faster".
4. **FP16 cuts the artifact roughly in half**: a `yolo11s` engine drops
   225 MB → 131 MB. Checkpoint `.pt` (19 MB) and `.onnx` (38 MB) are FP32-only.
5. **`openvino` on this NVIDIA GPU is 5× slower than `ort_cuda`.** OpenVINO 2026.4
   does expose the 3090 as its `GPU` device, but it does not use the TensorRT
   kernels, and it scales badly with model size (33 fps → 9.4 fps from S to L). Its
   `INFERENCE_PRECISION_HINT` (f32 vs f16) makes no measurable difference. Use it as
   a vendor-neutral reference point, not as a CUDA competitor; the plugin's real
   purpose is Intel GPUs.
6. **On CPU, OpenVINO is the fastest of the three.** For `yolo11s` fp32:
   `openvino_cpu` 79.9 ms < `ort_cpu` 84.5 ms < `pytorch_cpu` 132.0 ms. The CPU
   ordering is **noisy at the L sizes** — e.g. `ort_cpu` measured 73.9 ms on
   `yolo26s` but 412.2 ms on `yolo26l`, and `ort_cpu` (2.4 fps) even edged out
   `pytorch_cpu` (2.4 fps) there. At 2–4 fps a 4-core desktop CPU is saturated and
   contention/thermal effects dominate, so treat L-size CPU rows as indicative only.
7. **Pre/post overhead is ~2.5–3 ms** (`e2e` minus `infer`) and is roughly constant
   across runtimes — letterbox + NMS + un-letterbox are pure NumPy/OpenCV work. On
   the fastest config it is ~55% of the frame time (`tensorrt fp16` yolo11s:
   5.88 ms e2e vs 2.67 ms infer), so it is the next thing worth optimizing.
8. **`host_rss_mb` now requires isolation to be meaningful.** The tables above were
   produced before per-config subprocesses existed, so RSS accumulated across the
   56 configs and the later rows looked artificially large — that is why it is not
   tabulated. Re-run with the current code and it becomes a real per-backend
   footprint (e.g. `tensorrt` ≈ 1600 MB vs `ort_cuda` ≈ 1900 MB for `yolo11s`).
9. **`ort_cuda` is FP32-only in practice.** The table lists fp16 because the option
   exists, but the CUDA EP has no meaningful FP16 speedup here (4.47 → 4.46 ms), and
   `ort_cuda fp16` is **slightly slower** than `ort_cuda fp32` on the S models
   (134.9 → 135.3 fps is within noise, but `yolo26s` regresses 127.4 → 131.8/131.8).
   Read `ort_cuda fp32` as the representative number.

## Results interpretation (Rough expectations)

- `torch FP32` < `ort_cuda FP32` — ORT CUDA kernels / graph rewrite. ✅ confirmed (82.2 → 134.9 fps)
- `tensorrt FP16` ≤ `tensorrt FP32` — FP16 tensor-core throughput on Ampere. ✅ confirmed (138.6 → 170.0 fps)
- `ort_trt FP16` ≈ `tensorrt FP16` — both run the same TensorRT builder underneath. ❔ **not testable here**: the ORT TensorRT EP needs TensorRT 10 (see the results table footnote).
- `ort_trt` first run is slow (engine build); warm-up / engine-cache amorts this. ❔ same blocker.
- ⚠️ **"FP16 ≈ 2× faster" does NOT hold universally.** It holds for `tensorrt`,
  is neutral for `ort_cuda`, and is *negative* for eager PyTorch. Always check which
  backend a speedup claim came from.

## Project layout

```
yolo_bench/
  config.py      # paths, models, precision/runtime lists, arch registry
  export.py      # .pt -> .onnx -> .engine
  detector.py    # Detector: transparent runtime + architecture swap
  bench.py       # single-config benchmark driver (fps/latency/memory)
  compare.py     # matrix CLI -> CSV + Markdown
  check.py       # correctness gate vs ultralytics .predict()
  env.py         # hardware/library fingerprint for cross-machine provenance
  plots.py       # Plotly chart builders (nested/grouped fps, speedup)
  utils.py       # timing, metadata, device helpers
  orchestrate/   # test orchestration: matrix, subprocess isolation, experiments
    __init__.py
    _child.py    # one-config subprocess entry point
  archs/
    base.py       # Architecture ABC (prepare/decode/artifact resolution)
    yolo.py       # YOLO Detect head: letterbox + decode + NMS
    rfdetr.py     # RF-DETR scaffold (permissive license; pre/post TODO)
  runtimes/
    registry.py   # RuntimeSpec registry — source of truth for backends
    base.py       # RuntimeExecutor ABC (raw forward pass + precision gating)
    pytorch.py    # PyTorch CUDA + pytorch_cpu
    ort.py        # ORT CUDA EP + ORT TensorRT EP + ort_cpu
    tensorrt.py   # native TensorRT (deserialize .engine)
    openvino.py   # OpenVINO GPU device + openvino_cpu
results.ipynb    # run experiments + draw charts
models/          # .pt / .onnx / .engine / .meta.json  (git-ignored)
data/            # sample clip + bus.jpg               (git-ignored)
results/         # compare.csv / compare.md / final.*     (git-ignored)
```

## Known limitations / notes

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