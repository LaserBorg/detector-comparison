# Quick start

From an empty clone to a working detector in ~10 minutes.

## 1. Environment

Python 3.12 — a conda env works just as well as a venv:

```bash
conda create -n py312 python=3.12 -y && conda activate py312
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt       # ultralytics, onnx, onnxruntime-gpu, tensorrt
```

Optional: `ffmpeg` on PATH enables faster `predict` output encoding
(NVENC on NVIDIA GPUs, else CPU x264). Without it, `predict` falls back to
OpenCV's built-in `mp4v` codec. On Ubuntu: `sudo apt-get install ffmpeg`.

## 2. Checkpoints

Official Ultralytics assets into `models/` (`yolo26*` only exists from
ultralytics v8.4.0 onward). `-L` follows redirects, `-o` writes into `models/`:

```bash
BASE=https://github.com/ultralytics/assets/releases/download/v8.4.0
mkdir -p models data results
for m in yolo11s yolo11l yolo26s yolo26l; do
  curl -L --fail -o "models/$m.pt" "$BASE/$m.pt"
done
```

## 3. Sample media

```bash
# data/sample_1080p_h264.mp4    (your own clip, 1080p H.264)
curl -L --fail -o data/bus.jpg https://ultralytics.com/images/bus.jpg
```

## 4. Convert: `.pt` → `.onnx` → `.engine` (fp32 + fp16)

```bash
python -m yolo_bench export --models yolo11s yolo11l yolo26s yolo26l \
    --precisions fp32 fp16 --tf32-off
```

See [conversion.md](conversion.md) for what this does under the hood (and the
TensorRT 11 FP16 detail).

## 5. Correctness gate

Backends must match `ultralytics .predict()` before any number is trusted:

```bash
python -m yolo_bench check --image data/bus.jpg --model yolo11s --all
```

## 6. Use it

Now you can run inference (see the README for the full examples):

```bash
# single frame via Python
python -c "from yolo_bench import Detector; ..."

# annotated video via CLI
python -m yolo_bench predict --runtime tensorrt --precision fp16 \
    --model yolo11l --video data/sample_1080p_h264.mp4 --out out/yolo11l.mp4
```

…or benchmark the full matrix:

```bash
python -m yolo_bench compare --video data/sample_1080p_h264.mp4 --frames 100 \
    --models yolo11s yolo11l yolo26s yolo26l \
    --runtimes pytorch ort_cuda tensorrt openvino \
               pytorch_cpu ort_cpu openvino_cpu --precisions fp32 fp16 \
    --out results/rtx3090
```

See [benchmarking.md](benchmarking.md).
