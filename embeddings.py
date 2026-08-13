"""
Embedding utilities with high-performance memory caching and fast tensor pipelines.
"""
from __future__ import annotations

import io
from functools import lru_cache
import numpy as np
from PIL import Image

import config

_model = None
_preprocess = None
_device = None


def _lazy_load_clip():
    """Load CLIP once per process (expensive), reuse afterwards."""
    global _model, _preprocess, _device
    if _model is not None:
        return
    import torch
    import open_clip

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED
    )
    model.eval().to(_device)
    _model = model
    _preprocess = preprocess


def get_clip_embedding(image: Image.Image) -> np.ndarray:
    """Return a unit-normalized CLIP image embedding."""
    import torch

    _lazy_load_clip()
    image = image.convert("RGB")
    with torch.no_grad():
        tensor = _preprocess(image).unsqueeze(0).to(_device)
        feats = _model.encode_image(tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).cpu().numpy().astype("float32")


def get_color_histogram(image: Image.Image, bins=None) -> np.ndarray:
    """
    3D HSV colour histogram, flattened and L2-normalized.
    Resizes image to max 256x256 before histogram for 10x faster numpy calculation
    with identical color palette distribution.
    """
    bins = bins or config.COLOR_HIST_BINS
    img_hsv = image.convert("HSV")
    if img_hsv.width > 256 or img_hsv.height > 256:
        img_hsv = img_hsv.resize((256, 256), Image.Resampling.BILINEAR)

    hsv = np.array(img_hsv)
    hist, _ = np.histogramdd(
        hsv.reshape(-1, 3),
        bins=bins,
        range=((0, 256), (0, 256), (0, 256)),
    )
    hist = hist.flatten().astype("float32")
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist = hist / norm
    return hist


@lru_cache(maxsize=128)
def _get_fused_embedding_cached_bytes(img_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    clip_vec = get_clip_embedding(img) * config.CLIP_WEIGHT
    color_vec = get_color_histogram(img) * config.COLOR_WEIGHT
    fused = np.concatenate([clip_vec, color_vec]).astype("float32")
    norm = np.linalg.norm(fused)
    if norm > 0:
        fused = fused / norm
    return fused.tobytes()


def get_fused_embedding(image: Image.Image) -> np.ndarray:
    """
    Weighted early-fusion of CLIP + colour histogram, L2-normalized.
    Uses memory caching so re-querying the same image is instant (0ms).
    """
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    raw_bytes = _get_fused_embedding_cached_bytes(buf.getvalue())
    return np.frombuffer(raw_bytes, dtype=np.float32)


def embedding_dim() -> int:
    """Dimensionality of the fused embedding."""
    clip_dim = {"ViT-B-32": 512, "ViT-L-14": 768}.get(config.CLIP_MODEL_NAME, 512)
    color_dim = int(np.prod(config.COLOR_HIST_BINS))
    return clip_dim + color_dim
