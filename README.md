# YOLO TensorRT webcam inference

Minimal native TensorRT inference for YOLO11 detection models. The package
handles frame preprocessing, TensorRT execution, postprocessing, NMS, and
webcam annotation.

## Requirements

- NVIDIA GPU with a working CUDA and TensorRT installation
- Python 3.10+
- A TensorRT engine built for the current GPU
- A display for the OpenCV webcam window

Install the Python dependencies after installing the CUDA-enabled PyTorch build
that matches the machine:

```bash
pip install -r requirements.txt
```

## Download And Build The Engine

Install the conversion dependency from `requirements.txt`, then run this on
the target GPU. Ultralytics downloads the standard COCO-trained `yolo11s.pt`
checkpoint if it is not already present and exports the FP16 TensorRT engine:

```bash
python -m yolo_bench.export --model yolo11s --precision fp16 --force
```

This creates `models/yolo11s.pt`, `models/yolo11s.onnx`,
`models/yolo11s.fp16.onnx`, and `models/yolo11s.fp16.engine`.
The export uses a static `1x3x640x640` input and produces the
usual YOLO output `(1, 84, 8400)`. TensorRT engines are not portable:
build one on the RTX 3070, then build another on the Orin Nano.

## Run

```bash
python -m yolo_bench
```

Defaults are YOLO11s, FP16, and camera `0`. Press `q` to quit.
The window shows preprocessing, TensorRT plus transfer, postprocessing, total
latency, and completed display-cycle FPS so the active bottleneck is visible.

Options:

```bash
python -m yolo_bench \
  --source data/sample_1080p_h264.mp4 \
  # --camera 0 \
  --model yolo11s \
  --precision fp16 \
  --conf 0.25 \
  --iou 0.45
```

For direct webcam access, run the command in Windows or native Linux, where
OpenCV can see the attached camera.

The same detector can be embedded in another project:

```python
from yolo_bench import Detector

detector = Detector("yolo11s", "fp16")
detector.load()
boxes = detector(frame_bgr)
# boxes: float32 [x1, y1, x2, y2, confidence, class_id]
detector.release()
```

The inference path uses PyTorch CUDA tensors only for TensorRT device buffers;
there is no PyTorch model execution.
