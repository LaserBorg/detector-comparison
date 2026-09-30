# yolo_bench — detector wrapper + benchmark harness

A **functional inference wrapper** for object detectors across runtimes and
precisions — plus a benchmark harness to make informed decisions about model
size, runtime and hardware.

The primary product is the `Detector` wrapper: pick a model, a runtime and a
precision, and get standardized detections. The benchmark is one aspect of the
project — a decision tool for *which* configuration to run in production (e.g.
"yolo11l + TensorRT + FP16 on this GPU"), not the thing you ship.

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

The three `*_cpu` rows are true no-accelerator baselines: CPU execution
providers have **no FP16 kernels**, so FP16 is refused there rather than
silently running FP32.

## Usage

### 1. As a detector (production path)

```python
from yolo_bench import Detector

det = Detector("yolo", "tensorrt", "fp16", "yolo11l")   # model, runtime, precision
det.load()
boxes = det(frame_bgr)        # (M, 6) float32 [x1, y1, x2, y2, conf, cls], original pixel space
det.release()
```

Swap the runtime or the model with a constructor argument — nothing else
changes. The compiled-artifact backends (`tensorrt`, `ort_*`, `openvino`) are
pure ONNX/engine consumers and import nothing beyond their own runtime.

### 2. Annotated video (CLI)

```bash
python -m yolo_bench predict --runtime tensorrt --precision fp16 \
    --model yolo11l --video data/sample_1080p_h264.mp4 \
    --out out/yolo11l_trt_fp16.mp4 [--show]
```

Streams the full video, draws boxes + labels, writes an annotated mp4.
`--show` displays the current frame while processing (press `q` to stop);
omit it in headless/SSH sessions. `--encoder auto` (default) uses NVENC when
available, else CPU x264, else OpenCV's built-in codec — the summary line
reports detection fps and end-to-end fps separately, since the output
encoder is usually the bottleneck, not the detector.

### 3. Benchmark (decision tool)

```bash
# one config
python -m yolo_bench bench --runtime tensorrt --precision fp16 \
    --model yolo11s --video data/sample_1080p_h264.mp4 --frames 100

# full matrix -> canonical per-machine CSV (commit results/<tag>.csv + .md)
python -m yolo_bench compare --video data/sample_1080p_h264.mp4 --frames 100 \
    --models yolo11s yolo11l yolo26s yolo26l \
    --runtimes pytorch ort_cuda tensorrt openvino \
               pytorch_cpu ort_cpu openvino_cpu --precisions fp32 fp16 \
    --out results/rtx3090
```

on RTX3070m, only tensorRT ONNX 16 bit
```bash
python -m yolo_bench compare --video data/sample_1080p_h264.mp4 --frames 100 \
    --models yolo11s \
    --runtimes ort_cuda tensorrt \
    --precisions fp16 \
    --out results/rtx3070m
```

`results.ipynb` reads every `results/*.csv` (filename stem = machine tag),
draws the Q1–Q4 charts and writes `results/insights.md`.

## Setup

```bash
conda create -n py312 python=3.12 -y && conda activate py312
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# checkpoints + sample media (see docs/quickstart.md for the full steps)
python -m yolo_bench export --models yolo11s yolo11l yolo26s yolo26l \
    --precisions fp32 fp16 --tf32-off          # .pt -> .onnx -> .engine
python -m yolo_bench check --image data/bus.jpg --model yolo11s --all   # correctness gate
```

> **TensorRT engines are not portable** — an `.engine` is bound to the building
> GPU architecture *and* the TensorRT/CUDA version. Build engines on each target
> device; `.onnx`/`.pt` travel freely.

## Commands

| Command | Purpose |
|---|---|
| `python -m yolo_bench list` | Show the runtime registry and named experiments |
| `python -m yolo_bench export ...` | Convert `.pt` → `.onnx` → `.engine` (FP32 + FP16) |
| `python -m yolo_bench check ...` | Correctness gate vs `ultralytics .predict()` |
| `python -m yolo_bench bench ...` | Benchmark **one** runtime/precision/model on an mp4 |
| `python -m yolo_bench predict ...` | Inference only: draw detections, save annotated mp4 |
| `python -m yolo_bench compare ...` | Run a matrix → CSV + Markdown |
| `python -m yolo_bench run <name> ...` | Run a named experiment (e.g. `run precision`) |

## Documentation

| Doc | Contents |
|---|---|
| [docs/quickstart.md](docs/quickstart.md) | Environment, checkpoints, conversion, correctness gate — empty clone to working detector |
| [docs/architecture.md](docs/architecture.md) | The two layers (inference vs. harness), the `Detector` wrapper, runtime/architecture split, project layout |
| [docs/conversion.md](docs/conversion.md) | `.pt` → `.onnx` → `.engine` under the hood, FP16 on TRT 11, engine portability |
| [docs/benchmarking.md](docs/benchmarking.md) | Metrics, commands, subprocess isolation, per-machine CSVs, notebook workflow, expected orderings |
| [docs/multi-machine.md](docs/multi-machine.md) | Target hardware, moving to another machine (incl. Orin Nano / JetPack 7.2), merging results, cross-machine caveats |
| [docs/known-issues.md](docs/known-issues.md) | Gotchas: ORT/TRT version matching, FP16 semantics, OpenVINO quirks, memory measurement |
