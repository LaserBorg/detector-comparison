# Detector benchmark report

- **host**: `orin` [aarch64] — `Ubuntu 24.04.5 LTS` (kernel 6.8.12-1021-tegra)
- **cpu**: Cortex-A78AE [6c/6t, 7.3 GB RAM]
- **gpu**: Orin (Ampere, SM 8.7, 7.3 GB), driver 595.78
- **stack**: CUDA 13.2 / cuDNN 9.00.0, TensorRT 10.16.2.10, ONNX Runtime 1.23.0, torch 2.13.0+cu132 (cuda)
- **jetson**: L4T R39, power mode `MAXN_SUPER`
- run: `orin-20260926T131317Z` at 2026-09-26T15:13:17+0200

- `fps` = throughput (higher is better); `e2e_ms` = full frame loop; `infer_ms` = raw forward pass only
- `vram_gb` = device-wide peak; `weights_mb` = artifact on disk

## yolo11l

| runtime | precision | fps | e2e_ms | e2e_p95 | infer_ms | vram_gb | weights_mb |
|---|---|---|---|---|---|---|---|
| tensorrt | fp32 |   18.82 |   53.12 |   53.77 |   40.44 |    2.76 |    104.8 |
| ort_trt | fp32 | — | — | — | — | — | — |
| ort_cuda | fp32 |   14.19 |   70.49 |   71.65 |   60.49 |    3.91 |    101.7 |
| pytorch | fp32 |    8.63 |  115.87 |  116.16 |  106.97 |    3.21 |     51.4 |
| ort_cpu | fp32 |    1.25 |  801.92 |  830.95 |  794.43 |    3.08 |    101.7 |
| tensorrt | fp16 |   31.04 |   32.21 |   33.33 |   20.48 |    3.00 |     54.4 |
| ort_trt | fp16 | — | — | — | — | — | — |
| ort_cuda | fp16 |   14.09 |   70.95 |   71.95 |   60.78 |    3.93 |    101.7 |
| pytorch | fp16 |   12.22 |   81.84 |   82.27 |   72.75 |    3.36 |     51.4 |
| openvino | fp32 | — | — | — | — | — | — |
| openvino_cpu | fp32 | — | — | — | — | — | — |
| openvino | fp16 | — | — | — | — | — | — |
| openvino_cpu | fp16 | — | — | — | — | — | — |
| ort_cpu | fp16 | — | — | — | — | — | — |

Skipped in this matrix:

- `openvino` fp32: missing module(s): openvino
- `openvino_cpu` fp32: missing module(s): openvino
- `openvino` fp16: missing module(s): openvino
- `openvino_cpu` fp16: openvino_cpu supports fp32 only
- `ort_cpu` fp16: ort_cpu supports fp32 only

## yolo11s

| runtime | precision | fps | e2e_ms | e2e_p95 | infer_ms | vram_gb | weights_mb |
|---|---|---|---|---|---|---|---|
| tensorrt | fp32 |   31.58 |   31.67 |   32.26 |   19.21 |    2.86 |     40.7 |
| ort_trt | fp32 | — | — | — | — | — | — |
| ort_cuda | fp32 |   31.30 |   31.94 |   36.60 |   23.23 |    3.59 |     38.1 |
| pytorch | fp32 |   22.53 |   44.39 |   44.56 |   35.22 |    2.90 |     19.3 |
| ort_cpu | fp32 |    4.20 |  238.36 |  254.87 |  230.77 |    2.73 |     38.1 |
| tensorrt | fp16 |   52.60 |   19.01 |   23.57 |   10.26 |    2.75 |     22.6 |
| ort_trt | fp16 | — | — | — | — | — | — |
| ort_cuda | fp16 |   31.90 |   31.34 |   36.04 |   23.05 |    3.58 |     38.1 |
| pytorch | fp16 |   21.02 |   47.56 |   47.73 |   38.62 |    3.12 |     19.3 |
| openvino | fp32 | — | — | — | — | — | — |
| openvino_cpu | fp32 | — | — | — | — | — | — |
| openvino | fp16 | — | — | — | — | — | — |
| openvino_cpu | fp16 | — | — | — | — | — | — |
| ort_cpu | fp16 | — | — | — | — | — | — |

Skipped in this matrix:

- `openvino` fp32: missing module(s): openvino
- `openvino_cpu` fp32: missing module(s): openvino
- `openvino` fp16: missing module(s): openvino
- `openvino_cpu` fp16: openvino_cpu supports fp32 only
- `ort_cpu` fp16: ort_cpu supports fp32 only

## yolo26l

| runtime | precision | fps | e2e_ms | e2e_p95 | infer_ms | vram_gb | weights_mb |
|---|---|---|---|---|---|---|---|
| tensorrt | fp32 |   19.04 |   52.52 |   53.42 |   39.86 |    2.99 |    102.6 |
| ort_trt | fp32 | — | — | — | — | — | — |
| ort_cuda | fp32 |   14.08 |   71.04 |   71.86 |   60.24 |    3.88 |     99.6 |
| pytorch | fp32 |    8.08 |  123.80 |  124.94 |  116.75 |    3.27 |     53.2 |
| ort_cpu | fp32 |    1.26 |  792.56 |  820.18 |  784.40 |    3.03 |     99.6 |
| tensorrt | fp16 |   30.92 |   32.35 |   33.46 |   20.33 |    2.89 |     53.4 |
| ort_trt | fp16 | — | — | — | — | — | — |
| ort_cuda | fp16 |   14.25 |   70.16 |   71.20 |   59.95 |    3.93 |     99.6 |
| pytorch | fp16 |   11.75 |   85.12 |   85.53 |   78.15 |    3.56 |     53.2 |
| openvino | fp32 | — | — | — | — | — | — |
| openvino_cpu | fp32 | — | — | — | — | — | — |
| openvino | fp16 | — | — | — | — | — | — |
| openvino_cpu | fp16 | — | — | — | — | — | — |
| ort_cpu | fp16 | — | — | — | — | — | — |

Skipped in this matrix:

- `openvino` fp32: missing module(s): openvino
- `openvino_cpu` fp32: missing module(s): openvino
- `openvino` fp16: missing module(s): openvino
- `openvino_cpu` fp16: openvino_cpu supports fp32 only
- `ort_cpu` fp16: ort_cpu supports fp32 only

## yolo26s

| runtime | precision | fps | e2e_ms | e2e_p95 | infer_ms | vram_gb | weights_mb |
|---|---|---|---|---|---|---|---|
| tensorrt | fp32 |   31.32 |   31.93 |   32.62 |   19.54 |    2.99 |     40.9 |
| ort_trt | fp32 | — | — | — | — | — | — |
| ort_cuda | fp32 |   32.04 |   31.21 |   36.15 |   23.31 |    3.84 |     38.3 |
| pytorch | fp32 |   18.85 |   53.04 |   53.56 |   46.14 |    3.22 |     20.4 |
| ort_cpu | fp32 |    4.25 |  235.16 |  251.23 |  227.28 |    2.97 |     38.3 |
| tensorrt | fp16 |   55.12 |   18.14 |   20.48 |   10.05 |    2.98 |     23.1 |
| ort_trt | fp16 | — | — | — | — | — | — |
| ort_cuda | fp16 |   31.44 |   31.81 |   36.62 |   23.74 |    3.79 |     38.3 |
| pytorch | fp16 |   16.71 |   59.84 |   60.04 |   52.84 |    3.47 |     20.4 |
| openvino | fp32 | — | — | — | — | — | — |
| openvino_cpu | fp32 | — | — | — | — | — | — |
| openvino | fp16 | — | — | — | — | — | — |
| openvino_cpu | fp16 | — | — | — | — | — | — |
| ort_cpu | fp16 | — | — | — | — | — | — |

Skipped in this matrix:

- `openvino` fp32: missing module(s): openvino
- `openvino_cpu` fp32: missing module(s): openvino
- `openvino` fp16: missing module(s): openvino
- `openvino_cpu` fp16: openvino_cpu supports fp32 only
- `ort_cpu` fp16: ort_cpu supports fp32 only

