# Architecture: two layers, pluggable detectors

The project is deliberately split so the inference code can be imported by other
projects while the test machinery stays out of the way.

## Inference tier — reusable Python, no CLI, no orchestration

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

## Harness tier — a separate namespace

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

## The `Detector` wrapper

The key abstraction ties a **detector family** (an `Architecture`) to a
**runtime backend** (a `RuntimeExecutor`):

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

Standardized output for every architecture/runtime combination: `(M, 6)`
float32 `[x1, y1, x2, y2, conf, cls]` in **original image pixel space**.

## Project layout

```
yolo_bench/
  config.py      # paths, models, precision/runtime lists, arch registry
  export.py      # .pt -> .onnx -> .engine
  detector.py    # Detector: transparent runtime + architecture swap
  bench.py       # single-config benchmark driver (fps/latency/memory)
  predict.py     # inference-only driver: draw detections, save annotated mp4
  compare.py     # matrix CLI -> CSV + Markdown
  check.py       # correctness gate vs ultralytics .predict()
  env.py         # hardware/library fingerprint for cross-machine provenance
  plots.py       # Plotly chart builders (Q1-Q4: bars, heatmap, scatter, gpu)
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
