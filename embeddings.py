"""
Embedding utilities with 3-Way Fusion (Whole-Image CLIP + Color Histogram + Border/Pallu Region-Crop CLIP).

Why 3-Way Region-Aware Fusion?
-------------------------------
Every image in this catalogue is a saree, so CLIP's dominant learned signal
is "saree draped on a model on a white background". What distinguishes authentic
handloom sarees are fine-grained elements:
1. Whole-body silhouette, drapery & motif layout (CLIP whole-image).
2. Exact multi-color HSV palette distribution (3D HSV Histogram).
3. Fine-grained border and pallu weave work (Zari jaal, temple borders, kadiyal borders,
   scalloped borders, contrast hems) extracted via bottom-third and right-edge region crops.
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
    Resizes image to max 256x256 before histogram for fast computation
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


def get_border_pallu_crop_embedding(image: Image.Image) -> np.ndarray:
    """
    Extracts and embeds the border (bottom third) and pallu (right edge) regions.
    Combines bottom border drape and right-side pallu fall to capture fine-grained weave motifs.
    """
    W, H = image.size
    bottom_h = int(H * (1.0 - getattr(config, "BORDER_BOTTOM_RATIO", 0.35)))
    right_w = int(W * (1.0 - getattr(config, "BORDER_RIGHT_RATIO", 0.35)))

    bottom_crop = image.crop((0, bottom_h, W, H))
    right_crop = image.crop((right_w, 0, W, H))

    emb_bottom = get_clip_embedding(bottom_crop)
    emb_right = get_clip_embedding(right_crop)

    emb_border_raw = emb_bottom + emb_right
    norm = np.linalg.norm(emb_border_raw)
    if norm > 0:
        return (emb_border_raw / norm).astype("float32")
    return emb_bottom.astype("float32")


@lru_cache(maxsize=128)
def _get_fused_embedding_cached_bytes(img_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # 1. Whole-Image CLIP (0.55)
    clip_whole = get_clip_embedding(img) * config.CLIP_WEIGHT
    
    # 2. 3D HSV Color Histogram (0.25)
    color_vec = get_color_histogram(img) * config.COLOR_WEIGHT
    
    # 3. Border/Pallu Region-Crop CLIP (0.20)
    clip_border = get_border_pallu_crop_embedding(img) * config.BORDER_WEIGHT
    
    fused = np.concatenate([clip_whole, color_vec, clip_border]).astype("float32")
    norm = np.linalg.norm(fused)
    if norm > 0:
        fused = fused / norm
    return fused.tobytes()


def get_fused_embedding(image: Image.Image) -> np.ndarray:
    """
    Weighted 3-way early-fusion (Whole CLIP + HSV Color + Border Crop CLIP), L2-normalized.
    Uses in-memory LRU caching for instant re-queries.
    """
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    raw_bytes = _get_fused_embedding_cached_bytes(buf.getvalue())
    return np.frombuffer(raw_bytes, dtype=np.float32)


def embedding_dim() -> int:
    """Dimensionality of the 3-way fused embedding (512 + 512 + 512 = 1536)."""
    clip_dim = {"ViT-B-32": 512, "ViT-L-14": 768}.get(config.CLIP_MODEL_NAME, 512)
    color_dim = int(np.prod(config.COLOR_HIST_BINS))
    border_dim = clip_dim
    return clip_dim + color_dim + border_dim
