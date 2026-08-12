"""
Embedding utilities.

Why fusion instead of plain CLIP?
----------------------------------
Every image in this catalogue is a saree, so CLIP's dominant learned signal
("this is a saree, draped on a mannequin/model, photographed on white
background") is nearly IDENTICAL across the whole dataset. That collapses
much of the embedding space and is exactly why the assignment brief warns
that "a basic embedding search will return loose, generic results."

What actually differs between two sarees is fine-grained: the colour
combination, the print/motif, the border and pallu work, and the fabric's
visual texture. We address this with a fused embedding:

    1. CLIP ViT-B-32 embedding  -> overall visual/semantic similarity
    2. HSV colour histogram     -> precise colour-combination similarity
       (deliberately colour-space HSV, not RGB, since Hue is far more
       robust to the lighting/exposure differences between product photos)

The two are L2-normalized independently, weighted, concatenated, and
L2-normalized again. Cosine similarity (equivalently, inner product on the
normalized fused vector) then implicitly balances "looks like the same kind
of garment" against "is actually the same colour and pattern" -- which is
what a human would judge saree similarity by.
"""
from __future__ import annotations

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
    Captures the colour palette of the saree independent of the exact
    weave/print CLIP focuses on.
    """
    bins = bins or config.COLOR_HIST_BINS
    hsv = np.array(image.convert("HSV"))
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


def get_fused_embedding(image: Image.Image) -> np.ndarray:
    """
    Weighted early-fusion of CLIP + colour histogram, L2-normalized so
    inner product == cosine similarity in the fused space.
    """
    clip_vec = get_clip_embedding(image) * config.CLIP_WEIGHT
    color_vec = get_color_histogram(image) * config.COLOR_WEIGHT
    fused = np.concatenate([clip_vec, color_vec]).astype("float32")
    norm = np.linalg.norm(fused)
    if norm > 0:
        fused = fused / norm
    return fused


def embedding_dim() -> int:
    """Dimensionality of the fused embedding (needed to init the FAISS index)."""
    clip_dim = {"ViT-B-32": 512, "ViT-L-14": 768}.get(config.CLIP_MODEL_NAME, 512)
    color_dim = int(np.prod(config.COLOR_HIST_BINS))
    return clip_dim + color_dim
