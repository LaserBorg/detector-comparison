"""YOLO (Detect head) architecture: the classic Ultralytics detection contract.

Covers YOLO11 and YOLO26 (and v8/v9/v5) detection models. The exported graph
bakes in DFL + ``dist2bbox`` + sigmoid, so the output of PyTorch / ONNX /
TensorRT is the same ``(1, 4 + nc, num_pred)`` tensor:

  * channels ``[0:4]``   = decoded box in **xywh** (IMGSZ space)
  * channels ``[4:4+nc]`` = **already-sigmoid** per-class scores

``num_pred`` = 8400 at 640x640 (80x80 + 40x40 + 20x20 anchors).

Pre : letterbox + BGR->RGB + /255 -> ``(1, 3, 640, 640)`` fp32.
Post: threshold -> xywh2xyxy -> un-letterbox -> class-aware NMS.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .. import config
from .base import Architecture


@dataclass
class LetterboxInfo:
    """Geometry to map IMGSZ-px boxes back to the original frame."""

    r: float          # resize gain (old / new)
    dw: float         # x padding (each side), Ultralytics convention
    dh: float         # y padding (each side)
    orig_h: int       # original frame height
    orig_w: int       # original frame width

    def scale(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        out = boxes_xyxy.astype(np.float32, copy=True)
        out[:, [0, 2]] -= self.dw
        out[:, [1, 3]] -= self.dh
        out /= self.r
        out[:, [0, 2]] = out[:, [0, 2]].clip(0, self.orig_w)
        out[:, [1, 3]] = out[:, [1, 3]].clip(0, self.orig_h)
        return out


def letterbox(frame_bgr: np.ndarray) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize + pad a BGR frame to IMGSZ x IMGSZ, matching Ultralytics."""
    h, w = frame_bgr.shape[:2]
    r = min(config.IMGSZ / h, config.IMGSZ / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))  # (w, h)
    dw = (config.IMGSZ - new_unpad[0]) / 2.0
    dh = (config.IMGSZ - new_unpad[1]) / 2.0

    if (w, h) != new_unpad:
        frame_bgr = cv2.resize(frame_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        frame_bgr, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=config.LETTERBOX_COLOR,
    )
    return padded, LetterboxInfo(r=r, dw=dw, dh=dh, orig_h=h, orig_w=w)


def blob_numpy(padded: np.ndarray) -> np.ndarray:
    """Letterboxed BGR -> (1, 3, IMGSZ, IMGSZ) float32 in [0, 1], CHW, RGB.

    Guaranteed C-contiguous (the newaxis would otherwise leave a stride-0
    leading dim that ``torch.from_numpy`` / ONNX Runtime reject).
    """
    x = padded[:, :, ::-1]  # BGR -> RGB
    x = np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)
    x /= 255.0
    return np.ascontiguousarray(x[None])


def blob_torch(padded: np.ndarray, device) -> "torch.Tensor":
    """Letterboxed BGR -> (1, 3, IMGSZ, IMGSZ) float32 on `device`, CHW, RGB."""
    import torch  # lazy: only used by callers that pass a torch device

    return torch.from_numpy(blob_numpy(padded)).to(device)


def _xywh2xyxy(xywh: np.ndarray) -> np.ndarray:
    out = np.empty_like(xywh, dtype=np.float32)
    out[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    out[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    out[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    out[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return out


def _nms_class(dets: np.ndarray, iou_thres: float) -> np.ndarray:
    """Class-aware NMS on (N,6) [x1,y1,x2,y2,conf,cls]; returns keep-mask."""
    if dets.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    scores, classes = dets[:, 4], dets[:, 5]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)
        # suppress only same-class boxes (class-aware NMS, Ultralytics default)
        same = classes[order[1:]] == classes[i]
        order = order[1:][~((iou > iou_thres) & same)]
    return np.asarray(keep, dtype=np.int64)


def postprocess(
    pred: np.ndarray,
    nc: int,
    info: LetterboxInfo,
    conf: float = config.CONF,
    iou: float = config.IOU,
    max_det: int = config.MAX_DET,
) -> np.ndarray:
    """Decode an exported (1, 4+nc, N) output into (M, 6) detections.

    Returns ``[x1, y1, x2, y2, conf, cls]`` in ORIGINAL image space.
    """
    pred = pred[0]                      # (4+nc, num_pred)
    boxes = pred[:4]                    # xywh, IMGSZ-px
    scores = pred[4:4 + nc]             # already sigmoided

    cls_scores = scores.max(axis=0)
    cls_ids = scores.argmax(axis=0)

    mask = cls_scores > conf
    if not mask.any():
        return np.zeros((0, 6), dtype=np.float32)

    xywh = boxes[:, mask].T
    xyxy = _xywh2xyxy(xywh)
    confs = cls_scores[mask]
    ids = cls_ids[mask].astype(np.float32)

    dets = np.concatenate([xyxy, confs[:, None], ids[:, None]], axis=1)

    if dets.shape[0] > max_det:
        # keep highest-scoring candidates first to bound the O(n^2) NMS loop
        dets = dets[dets[:, 4].argsort()[::-1][:max_det]]

    dets = dets[_nms_class(dets, iou)]

    xyxy = info.scale(dets[:, :4])
    return np.concatenate([xyxy, dets[:, 4:6]], axis=1).astype(np.float32)


class YoloArchitecture(Architecture):
    name = "yolo"
    default_nc = 80
    imgsz = config.IMGSZ

    def prepare(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, LetterboxInfo]:
        padded, info = letterbox(frame_bgr)
        return blob_numpy(padded), info

    def decode(self, raw, ctx, conf, iou, max_det) -> np.ndarray:
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        return postprocess(np.asarray(raw), self.nc, ctx,
                           conf=conf, iou=iou, max_det=max_det)

    def torch_runner(self, model: str, device, precision: str):
        import torch  # lazy: only the 'pytorch' runtime needs it
        from ultralytics import YOLO

        # Strict FP32: disable TF32 so "fp32" is genuinely 32-bit math, matching
        # what the ONNX graph / --noTF32 engine builds produce for a fair compare.
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        dtype = torch.float16 if precision == "fp16" else torch.float32

        m = YOLO(str(self.checkpoint_path(model)))
        net = m.model.eval().to(device)
        if precision == "fp16" and device.type == "cuda":
            net = net.half()

        def run(blob: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(blob).to(device).to(dtype)
            with torch.no_grad():
                y = net(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            return y.float().cpu().numpy()

        return run