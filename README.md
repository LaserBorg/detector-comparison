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

# 6. Benchmark the full matrix, writing the canonical per-machine CSV.
#    <tag> is the machine name (rtx3090, rtx3070, orin-nano, ...); commit the pair.
python -m yolo_bench compare --video data/sample_1080p_h264.mp4 --frames 100 \
    --models yolo11s yolo11l yolo26s yolo26l \
    --runtimes pytorch ort_cuda tensorrt openvino \
               pytorch_cpu ort_cpu openvino_cpu --precisions fp32 fp16 \
    --out results/rtx3090
```

Results land in `results/rtx3090.csv` + `results/rtx3090.md` (or under whatever
`--out` prefix you pass). The CSV is the single source of truth for that
machine; `results.ipynb` reads it to draw the charts and write `insights.md`.

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

`results.ipynb` is the reader for the per-machine CSVs. It auto-discovers every
`results/*.csv` (filename stem = machine tag), so a new machine's CSV needs no
code change. The cells, in order:

1. environment fingerprint of the machine you are on,
2. the named experiments (`orchestrate.EXPERIMENTS`) and the runtime registry,
3. a cheap `smoke` probe (throwaway — it does **not** write to `results/`),
4. **load** every `results/*.csv`, tagged by machine,
5. one machine header per source,
6. **Q1 — runtime comparison**: a slope chart + a ranking heatmap, *per machine*,
7. **Q2 — performance vs model cost**: a GFLOPs-vs-fps scatter, *per machine*,
8. **Q3 — quantization advantage**: an FP16/FP32 fps-ratio heatmap, *per machine*,
9. **Q4 — GPU comparison**: the cross-machine chart (the one question that mixes
   machines on purpose), plus a pivot table,
10. **insights**: derives a short markdown report and writes `results/insights.md`.

Q1–Q3 are deliberately **per machine** — the ranking and the FP16 gain are
properties of the hardware, so averaging two machines would produce a number that
is neither one's real measurement. Q4 is the exception: comparing GPUs *is* its
question, so it merges all machines (solid = one GPU, dashed = the other).

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
   `tensorrt` wheel. The C++ runtime (`libnvinfer10`) ships with JetPack, but the
   **Python bindings are a separate package** and `trtexec` is not installed by
   default:

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

## Results: single source of truth

There is exactly **one canonical CSV per machine** in `results/`, named by machine
tag: `results/rtx3090.csv`, `results/orin-nano.csv` (and later `results/rtx3070.csv`, …).
Each machine runs `compare --out results/<tag>` and commits its own pair
(`<tag>.csv` + the auto-generated `<tag>.md`). These CSVs are the **only** place
raw numbers live — they are fingerprinted per row (GPU, driver, library versions,
`nvpmodel`, …) so a merged set stays unambiguous.

Everything else is **derived** from those CSVs and is not committed:

- **`results.ipynb`** loads every `results/*.csv` (the filename stem is the machine
  tag), draws the charts, and writes a short `results/insights.md`. Run it to see
  the figures and the findings; it needs no code change when a new machine's CSV
  lands.
- **`results/insights.md`** — regenerated by the notebook on each run (gitignored).

> **Numbers are host-specific.** Absolute values move with GPU, driver, TensorRT
> version and clock state; the *ordering* is the transferable result. Cross-machine
> `tensorrt`/`ort_trt` rows are not version-comparable (TRT 11 vs 10) — see the
> caveats in "Moving to another machine".

To add a machine: run `compare --out results/<tag>` there, commit `<tag>.csv` +
`<tag>.md`, and un-ignore them in `.gitignore`. The notebook picks it up
automatically.

## Results interpretation (Rough expectations)

These are the *orderings* to expect; the actual numbers live in the per-machine
CSVs and the notebook charts (the fps figures below are illustrative, from the
3090 run).

- `torch FP32` < `ort_cuda FP32` — ORT CUDA kernels / graph rewrite. ✅ confirmed (~82 → ~135 fps)
- `tensorrt FP16` ≤ `tensorrt FP32` — FP16 tensor-core throughput on Ampere. ✅ confirmed (~139 → ~170 fps)
- `ort_trt FP16` ≈ `tensorrt FP16` — both run the same TensorRT builder underneath. ❔ **not testable on a TRT 11 host**: the ORT TensorRT EP needs a matching TensorRT major (see "Moving to another machine").
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
  plots.py       # Plotly chart builders (Q1-Q4: slope, heatmap, scatter, gpu)
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
results.ipynb    # reads results/*.csv, draws Q1-Q4 charts, writes insights.md
models/          # .pt / .onnx / .engine / .meta.json  (git-ignored)
data/            # sample clip + bus.jpg               (git-ignored)
results/         # <tag>.csv + <tag>.md per machine (committed); scratch ignored
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