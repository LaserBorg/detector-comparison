"""RF-DETR architecture (scaffold — implemented in a later project phase).

RF-DETR (Real-Time Detection Transformer) has a **completely different** contract
from the classic YOLO Detect head, so it needs its own pre/post:

  * Pre: resize (no aspect-preserving letterbox), ImageNet mean/std
    normalization, channel order, and stride-multiple padding.
  * Post: its head emits either an end-to-end ``(1, N, 6)``
    ``[x1,y1,x2,y2,conf,cls]`` tensor (NMS baked in) or, when exported as
    logits, query boxes + class scores requiring their own softmax + NMS.

The ``Detector`` wrapper already supports this shape of architecture: only
``prepare`` / ``decode`` / ``torch_runner`` need to be implemented — no runtime
or wrapper changes are required. This class is a labelled scaffold so the
codebase is ready; it intentionally raises ``NotImplementedError`` until the
real RF-DETR export contract is pinned down.

Licensing note: RF-DETR is Apache-2.0 (vs Ultralytics AGPL-3.0). Layering
architectures like this is what lets a permissive-license detector drop in
without touching any runtime.
"""

from __future__ import annotations

from .base import Architecture


class RfDetrArchitecture(Architecture):
    name = "rfdetr"
    default_nc = 80

    def prepare(self, frame_bgr):
        raise NotImplementedError(
            "RF-DETR preprocessing is not implemented yet (expected: resize + "
            "ImageNet mean/std normalization, no aspect-preserving letterbox)."
        )

    def decode(self, raw, ctx, conf, iou, max_det):
        raise NotImplementedError(
            "RF-DETR postprocessing is not implemented yet (expected: query-box "
            "logits + softmax, or end-to-end (1, N, 6) output)."
        )

    def torch_runner(self, model, device, precision):
        raise NotImplementedError(
            "RF-DETR has no PyTorch runner wired up yet (used by the 'pytorch' "
            "runtime only — the ONNX/TensorRT paths do not need it)."
        )