# Plan

## Goal — three things, in priority order

1. **Benchmark** — compare, for checkpoints of the same complexity, the
   **framerate**, **latency**, and **memory footprint** of a YOLO detection model
   across four runtimes and two precisions on NVIDIA GPUs:
   - `pytorch` (fp32/fp16), `ort_cuda` (fp32), `ort_trt` (fp32/fp16),
     `tensorrt` (fp32/fp16).
2. **Reusable wrapper** — a `Detector` that swaps runtimes transparently, to be
   imported by upcoming projects.
3. **Architecture pluggability** — per-family pre/post so other detectors
   (e.g. RF-DETR) with different pre/post **and** a permissive license
   (Apache-2.0 vs Ultralytics AGPL) can be added in a later phase without
   touching any runtime or wrapper code.

Models: `yolo11s`, `yolo11l`, `yolo26s`, `yolo26l` (COCO detect).

## Decisions

### 1. Hardware scope (why no 960M)
The GTX 960M is **Maxwell SM 5.0**. TensorRT requires SM 7.5+; ONNX Runtime's
TensorRT EP bundles TensorRT 8.6–10.x (SM 7.0+). So the 960M cannot run TensorRT
at all. It could run `pytorch`/`ort_cuda` FP32 with a pinned PyTorch ≤ 2.7
(CUDA 12.8+ dropped Maxwell kernels), but that's a parallel, lower-value effort.
**Scope: RTX 3070/3090 (SM 8.6) and Jetson Orin Nano (SM 8.7)** — all four
backends + FP16 tensor cores.

### 2. How 16-bit works (and why there are no "FP16 .pt" checkpoints)
- The COCO checkpoints ship as **FP32**.
- **FP16** is an *execution mode*, not a stored format: the same FP32 weights run
  at half precision via `model.half()` (PyTorch), `--fp16` (TensorRT), or
  `trt_fp16_enable` (ORT). No disk artifact is "FP16" except the compiled
  `.engine` itself.
- Confusion worth mapping: **TF32** is a *32-bit* tensor-core mode (we disable it
  for a strict FP32 baseline); **BF16** is the other 16-bit float (Ampere tensor
  cores) but **FP16** is the primary 16-bit variant we benchmark.
- **FP16 speedup ≈ 2× is specific to Ampere+** (tensor cores). On Maxwell there
  are no FP16 tensor cores — FP16 is often *slower* than FP32. This is why the
  DeepStream win (your Orin Nano) is tensor-core driven, and why it is not
  portable to the 960M.

### 3. Conversion path
1. `.pt` → `.onnx` via `ultralytics` `format='onnx', imgsz=640, opset=17, simplify`.
   The classic `Detect` head (used by YOLO11 **and** YOLO26 detect) bakes in
   DFL + `dist2bbox` + sigmoid → output `(1, 4+nc, 8400)`, boxes already `xywh`.
   No per-family decode is needed.
2. `.onnx` → `.engine` via **`trtexec`** for transparency (`--fp16` for FP16,
   `--noTF32` for strict FP32). Ultralytics `format='engine', quantize=16|32` is
   kept as a **cross-check**.

### 4. Layered design (wrapper + architecture pluggability)
- **`runtimes/`** (`RuntimeExecutor`) = a *raw forward pass* only: `load(artifact)`
  + `infer(blob) -> raw`. Family-agnostic.
- **`archs/`** (`Architecture`) = a family's pre/post (`prepare`/`decode`) and
  artifact resolution (`checkpoint_path`/`onnx_path`/`engine_path`). YOLO now;
  RF-DETR scaffolded.
- **`detector.py`** (`Detector`) = the thin wrapper that binds the two, exposing
  `detect(frame) -> (M,6)`. Swap runtime via constructor; swap architecture via a
  single new `Architecture` subclass.

This is what makes the same pre/post "shared" for apples-to-apples timing, while
still allowing RF-DETR to use a completely different pre/post later. The YOLO
pre/post lives in `archs/yolo.py` (a compatibility shim `prepost.py` re-exports
it).

### 5. Engines are not portable
Each `.engine` is bound to the building GPU arch + TensorRT/CUDA version. Build
engines **on each target device**. (3090 and 3070 are both SM 8.6 but a rebuild
per device is the safest default; `--hw-compatible` is a future option.)

## Status

- [x] Scaffold + config + utils
- [x] `export.py` (`.pt` → `.onnx` → `.engine`, fp32/fp16, cross-check)
- [x] `archs/` — `Architecture` ABC + `yolo.py` + `rfdetr.py` scaffold
- [x] `runtimes/` — `RuntimeExecutor` ABC + pytorch / ort_cuda / ort_trt / tensorrt
- [x] `detector.py` — reusable wrapper (transparent runtime + arch swap)
- [x] `bench.py` (fps + latency + memory) / `compare.py` / `check.py`
- [x] `README.md`, requirements, .gitignore
- [ ] **Run on RTX 3090 host** (Python 3.12 venv) — validate conversion + correctness + matrix
- [ ] Rebuild engines + re-run on **3070** and **Orin Nano** (JetPack 6)
- [ ] Implement RF-DETR pre/post in `archs/rfdetr.py` (later phase)

## Next actions (on the 3090 box)

1. Create Python **3.12** venv (not 3.13/3.14 — avoid missing `tensorrt`/
   `onnxruntime-gpu`/`torch` wheels).
2. `pip install torch --index-url .../cu124` then `pip install -r requirements.txt`.
3. Add checkpoints to `models/` and a sample `data/sample.mp4`.
4. `python -m yolo_bench export --models ... --precisions fp32 fp16 --tf32-off`.
5. `python -m yolo_bench check --image data/bus.jpg --model yolo11s --all`.
6. `python -m yolo_bench compare --video data/sample.mp4 --frames 100 ...`.

## Open questions (revisit before the 3090 run)

1. **Host Python version** — 3.12 recommended; confirm `tensorrt`/`onnxruntime-gpu`
   wheel availability for the chosen CUDA.
2. **BF16 as a third precision** — small effort on Ampere; decide whether to add.
3. **Sample 1080p mp4** — user supplies a representative clip (recommended) vs a
   synthetic placeholder now.
4. **RF-DETR export contract** — pin down its ONNX output (end-to-end `(1,N,6)`
   vs query logits) before implementing `archs/rfdetr.py`.