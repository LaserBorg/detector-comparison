# Conversion: `.pt` → `.onnx` → `.engine`

```bash
python -m yolo_bench export --models yolo11s yolo11l yolo26s yolo26l \
    --precisions fp32 fp16 --tf32-off
```

## What happens

1. **`.pt` → `.onnx`** via `ultralytics` (`format='onnx', imgsz=640, opset=17,
   simplify=True`). The classic `Detect` head bakes in DFL + `dist2bbox` +
   sigmoid, so the ONNX output is `(1, 4+nc, num_pred)` with boxes already in
   `xywh` (640-px space) and per-class scores already sigmoided. `--tf32-off`
   disables TF32 during trace for a strict FP32 graph.
2. **`.onnx` → `.engine`** via the **TensorRT Python API** (`trt.Builder`). This
   is deliberate: the PyPI `tensorrt` wheel ships **no `trtexec` binary**, so on
   a plain `pip install` (as on the 3090 host) `trtexec` does not exist.
   `trtexec` is only present with a full TensorRT install (JetPack, or the
   NVIDIA tarball); if it *is* found, `--via trtexec` uses it, otherwise the
   Python builder is used. Either way the engine is built on the target device.

   **FP16 detail (TensorRT ≥ 11):** TRT 11 is *strongly-typed only* — it removed
   `BuilderFlag.FP16`/`INT8`, the INT8 calibrator, and per-tensor precision
   setters. FP16 therefore has to be baked into the **ONNX graph** before
   parsing, which we do with NVIDIA **ModelOpt AutoCast** (mixed precision with
   `keep_io_types=True`, so the engine's I/O stays FP32 and fp32/fp16 remain
   directly comparable). On TensorRT 7–10 the classic `BuilderFlag.FP16` flag is
   used instead.
3. Metadata (`nc`, input shape, arch, TRT version, params/GFLOPs) is written to
   `models/{name}.meta.json`.
4. `--cross-check` additionally builds an FP16 engine through Ultralytics' own
   `format='engine', quantize=16` path for validation.

## Precision semantics

- `.pt` checkpoints are FP32 and come from Ultralytics; they are never
  re-quantized *to disk* in FP16. Instead, we build/run with FP16 where the
  runtime supports it. (16-bit "checkpoints" don't exist as a first-class
  artifact; FP16 is an *execution mode*, not a stored format.)
- The FP16 path is the 16-bit variant we benchmark (≈2× on Ampere via tensor
  cores). BF16 is available implicitly via `half()` replacement but is not wired
  as a separate precision; TF32 is a *32-bit* tensor-core mode that we disable
  for a strict FP32 baseline.

## Engine portability

> **TensorRT engines are not portable.** An `.engine` is bound to the building
> GPU architecture *and* the TensorRT/CUDA version. Build engines **on each
> target device**. ONNX is a portable IR, so `.onnx` (and `.pt`) travel freely
> between machines — see [multi-machine.md](multi-machine.md).

Engine builds are slow: ~64 s for `yolo11s` fp32 and ~100 s for fp16 (the fp16
path also runs ModelOpt AutoCast first). `.engine` sizes: 224 MB fp32 vs 131 MB
fp16 for `yolo11s`.

## Cross-check (native vs. Ultralytics-exported engine)

An engine built by **yolo_bench** should match one built by **Ultralytics
`format='engine'`** for the same precision. Use `--cross-check` to emit both and
compare detections with the `check` command.
