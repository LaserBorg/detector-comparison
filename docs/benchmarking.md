# Benchmarking

The benchmark is one aspect of this project: a decision tool for choosing model
size, runtime and hardware. It is not the product — the `Detector` wrapper is.

## The three metrics

| Metric | What we report | Meaning |
|---|---|---|
| **Framerate** | `fps` | 1 / end-to-end time — sustainable throughput. |
| **Latency** | `e2e_ms` (+ `e2e_med_ms`, `e2e_p95_ms`) full loop; `infer_ms` raw forward only | time a frame spends in the pipeline; median is the honest single-frame latency, p95 the tail. |
| **Memory** | `peak_vram_gb` (device) + `host_rss_mb` (process RAM) + `weights_mb` (artifact on disk) | full footprint across GPU / host / storage. |

## Commands

| Command | Purpose |
|---|---|
| `python -m yolo_bench list` | Show the runtime registry and named experiments |
| `python -m yolo_bench export ...` | Convert `.pt` → `.onnx` → `.engine` (FP32 + FP16) |
| `python -m yolo_bench check ...` | Correctness gate vs `ultralytics .predict()` |
| `python -m yolo_bench bench ...` | Benchmark **one** runtime/precision/model on an mp4 |
| `python -m yolo_bench predict ...` | Inference only: draw detections, save annotated mp4 (`--show` to view live, `--encoder` for NVENC) |
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

### `predict` (inference only, no metrics)

```bash
python -m yolo_bench predict --runtime tensorrt --precision fp16 \
    --model yolo11l --video data/sample_1080p_h264.mp4 \
    --out out/yolo11l_trt_fp16.mp4 [--show]
```

Streams the full video frame by frame, draws boxes + labels, writes an
annotated mp4 at the source fps. `--show` displays the current frame while
processing (press `q` to stop); omit it in headless/SSH sessions.

The summary line reports **two** fps numbers:

- `detection` — the detector pipeline only (same semantics as `bench` fps)
- `end-to-end` — wall time including decode, drawing, and video encoding

They differ because the output encoder is usually the bottleneck: a software
encoder costs ~11 ms/frame at 1080p, so a 170 fps detector still only writes
~55 fps of video. `--encoder` controls the writer:

| `--encoder` | Backend | Notes |
|---|---|---|
| `auto` (default) | `h264_nvenc` → `libx264` → `mp4v` | picks the fastest available |
| `nvenc` | NVIDIA NVENC (GPU) | needs ffmpeg + a free CUDA context |
| `x264` | libx264 (CPU) | needs ffmpeg |
| `mp4v` | OpenCV built-in | no ffmpeg needed, slowest |

NVENC runs in the GPU's dedicated encoder hardware and coexists with
inference, but it needs a small CUDA context (~few hundred MB free VRAM). If
the GPU is fully occupied by another process, NVENC init fails and `auto`
falls back to CPU encoding (see `known-issues.md`).

## Why a subprocess per configuration

Every configuration runs in a **fresh subprocess**. Running a full matrix in one
interpreter made `host_rss_mb` climb monotonically (1.2 → 4.5 GB) purely from
allocator/RSS accumulation, so the memory column was really measuring run order.
Device memory has the same problem (allocator caches and TensorRT/ORT arenas
outlive `release()`). Isolation costs a process spawn and makes the numbers mean
something. Use `--no-isolate` only when you do not care about memory.

## Every row is fingerprinted

Results are merged across machines (RTX 3090, a 3070, a Jetson Orin Nano), so
each row records 32 provenance fields: `run_id`/`timestamp`/`hostname`, CPU
model + cores + RAM, GPU name/CC/architecture/VRAM/driver, OS/kernel/python, and
library versions (`tensorrt`, `onnxruntime`, `openvino`, `torch`, …), plus a
per-row `runtime_version`. Without this, a TensorRT-11 row and a TensorRT-10 row
look identical in a merged CSV.

## Results: single source of truth

There is exactly **one canonical CSV per machine** in `results/`, named by
machine tag: `results/rtx3090.csv`, `results/orin-nano.csv` (and later
`results/rtx3070.csv`, …). Each machine runs `compare --out results/<tag>` and
commits its own pair (`<tag>.csv` + the auto-generated `<tag>.md`). These CSVs
are the **only** place raw numbers live — they are fingerprinted per row (GPU,
driver, library versions, `nvpmodel`, …) so a merged set stays unambiguous.

Everything else is **derived** from those CSVs and is not committed:

- **`results.ipynb`** loads every `results/*.csv` (the filename stem is the
  machine tag), draws the charts, and writes a short `results/insights.md`. Run
  it to see the figures and the findings; it needs no code change when a new
  machine's CSV lands.
- **`results/insights.md`** — regenerated by the notebook on each run
  (gitignored).

> **Numbers are host-specific.** Absolute values move with GPU, driver,
> TensorRT version and clock state; the *ordering* is the transferable result.
> Cross-machine `tensorrt`/`ort_trt` rows are not version-comparable (TRT 11 vs
> 10) — see [multi-machine.md](multi-machine.md).

To add a machine: run `compare --out results/<tag>` there, commit `<tag>.csv` +
`<tag>.md`, and un-ignore them in `.gitignore`. The notebook picks it up
automatically.

## Notebook workflow

`results.ipynb` is the reader for the per-machine CSVs. It auto-discovers every
`results/*.csv` (filename stem = machine tag), so a new machine's CSV needs no
code change. The cells, in order:

1. environment fingerprint of the machine you are on,
2. the named experiments (`orchestrate.EXPERIMENTS`) and the runtime registry,
3. a cheap `smoke` probe (throwaway — it does **not** write to `results/`),
4. **load** every `results/*.csv`, tagged by machine,
5. one machine header per source,
6. **Q1 — runtime comparison**: grouped bars + a ranking heatmap, *per machine*,
7. **Q2 — performance vs model cost**: a GFLOPs-vs-fps scatter, *per machine*,
8. **Q3 — quantization advantage**: an FP16/FP32 fps-ratio heatmap, *per machine*,
9. **Q4 — GPU comparison**: the cross-machine chart (the one question that mixes
   machines on purpose), plus a pivot table,
10. **insights**: derives a short markdown report and writes `results/insights.md`.

Q1–Q3 are deliberately **per machine** — the ranking and the FP16 gain are
properties of the hardware, so averaging two machines would produce a number
that is neither one's real measurement. Q4 is the exception: comparing GPUs *is*
its question, so it merges all machines (first machine at full opacity, later
machines lighter).

## Results interpretation (rough expectations)

These are the *orderings* to expect; the actual numbers live in the per-machine
CSVs and the notebook charts (the fps figures below are illustrative, from the
3090 run).

- `torch FP32` < `ort_cuda FP32` — ORT CUDA kernels / graph rewrite. ✅ confirmed (~82 → ~135 fps)
- `tensorrt FP16` ≤ `tensorrt FP32` — FP16 tensor-core throughput on Ampere. ✅ confirmed (~139 → ~170 fps)
- `ort_trt FP16` ≈ `tensorrt FP16` — both run the same TensorRT builder underneath. ❔ **not testable on a TRT 11 host**: the ORT TensorRT EP needs a matching TensorRT major (see [multi-machine.md](multi-machine.md)).
- `ort_trt` first run is slow (engine build); warm-up / engine-cache amorts this. ❔ same blocker.
- ⚠️ **"FP16 ≈ 2× faster" does NOT hold universally.** It holds for `tensorrt`,
  is neutral for `ort_cuda`, and is *negative* for eager PyTorch. Always check
  which backend a speedup claim came from.
