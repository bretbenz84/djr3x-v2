"""
perception/place_embedder.py — MobileCLIP-S2 image encoder for place recognition.

Loads ONE MobileCLIP-S2 model (open_clip) and exposes ``encode_image(frame)`` returning
the RAW (unnormalized) 512-d image embedding that ``PlaceRecognizer`` L2-normalizes. This
is the ONLY place torch/open_clip is imported for the place feature — the recognizer
itself stays model-agnostic (its ``embed_fn`` is dependency-injected), so this file is the
single seam where the concrete encoder lives.

Design notes:
  * Heavy imports (torch, open_clip) happen inside ``load_place_embedder`` so importing
    this module is cheap and a missing/broken encoder degrades to "feature off" (returns
    None) instead of crashing startup.
  * Camera frames are OpenCV BGR; MobileCLIP wants RGB. When ``bgr_input`` (config
    PLACE_FRAME_IS_BGR) is True the channels are swapped so LIVE query frames match the
    RGB representation used when a room was enrolled — otherwise every query would be
    channel-swapped relative to its gallery.
  * The torch model is not guaranteed thread-safe, so ``encode_image`` serializes on an
    internal lock; the recognizer already calls it off its own lock.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import numpy as np

import config

_log = logging.getLogger(__name__)


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, path)


class PlaceEmbedder:
    """Callable-ish wrapper around a loaded MobileCLIP model. Pass ``encode_image`` as the
    recognizer's ``embed_fn``."""

    def __init__(self, model, preprocess, device, *, model_tag: str, name: str,
                 bgr_input: bool):
        self._model = model
        self._pre = preprocess
        self._device = device
        self.model_tag = model_tag
        self.name = name
        self._bgr = bool(bgr_input)
        self._lock = threading.Lock()

    def encode_image(self, frame) -> np.ndarray:
        import torch
        from PIL import Image

        arr = np.asarray(frame)
        if arr.ndim == 2:                       # grayscale -> 3-channel
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.shape[2] == 4:                   # drop alpha
            arr = arr[:, :, :3]
        if self._bgr:                           # OpenCV BGR -> RGB
            arr = arr[:, :, ::-1]
        img = Image.fromarray(np.ascontiguousarray(arr.astype("uint8")), "RGB")
        with self._lock, torch.no_grad():
            tensor = self._pre(img).unsqueeze(0).to(self._device)
            feats = self._model.encode_image(tensor)
        return feats[0].detach().float().cpu().numpy()


def load_place_embedder(
    *,
    model_name: Optional[str] = None,
    pretrained: Optional[str] = None,
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    model_tag: Optional[str] = None,
    bgr_input: Optional[bool] = None,
) -> Optional[PlaceEmbedder]:
    """Load MobileCLIP-S2 (or whatever config points at) and return a PlaceEmbedder, or
    None if torch/open_clip or the weights are unavailable. Never raises — a failure just
    disables the feature."""
    model_name = model_name or getattr(config, "PLACE_OPEN_CLIP_MODEL", "MobileCLIP-S2")
    pretrained = pretrained or getattr(config, "PLACE_OPEN_CLIP_PRETRAINED", "datacompdr")
    cache_dir = cache_dir or _resolve(getattr(config, "PLACE_MODEL_DIR", "assets/models/mobileclip"))
    model_tag = model_tag or getattr(config, "PLACE_MODEL_TAG", "mobileclip_s2_v1")
    if bgr_input is None:
        bgr_input = bool(getattr(config, "PLACE_FRAME_IS_BGR", True))
    try:
        import torch
        import open_clip

        if device is None:
            device = getattr(config, "PLACE_EMBED_DEVICE", None) or "cpu"
        os.makedirs(cache_dir, exist_ok=True)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, cache_dir=cache_dir
        )
        model.eval().to(device)
        _log.info("place encoder loaded: %s/%s on %s (tag=%s)",
                  model_name, pretrained, device, model_tag)
        return PlaceEmbedder(
            model, preprocess, device,
            model_tag=model_tag, name=f"{model_name}/{pretrained}", bgr_input=bgr_input,
        )
    except Exception as exc:  # noqa: BLE001 — any failure disables the feature cleanly
        _log.warning(
            "place encoder unavailable (%s); place recognition disabled. "
            "Run setup_assets.py / `pip install open_clip_torch` to enable it.", exc,
        )
        return None
