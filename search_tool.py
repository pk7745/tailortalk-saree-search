"""
Enhanced Search Tool for TailorTalk Saree Search.

Supports:
1. Hybrid Visual Similarity Search (FAISS + CLIP + Color Histogram)
2. Fine-grained Multi-Attribute Filtering (Color, Fabric, Pattern, Price range, Keywords)
3. Text & Attribute Catalogue Discovery (when no image is uploaded)
"""
from __future__ import annotations

import io
import re
from functools import lru_cache
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import requests
from PIL import Image

import config
from embeddings import get_fused_embedding

# Comprehensive color, fabric, and pattern taxonomies
COLORS = [
    'baby pink', 'rani pink', 'peach pink', 'dusty pink', 'pink',
    'navy blue', 'sky blue', 'royal blue', 'dark blue', 'blue',
    'mint green', 'bottle green', 'pista green', 'dark green', 'olive green', 'green',
    'dark yellow', 'mustard', 'lemon yellow', 'yellow',
    'dusty purple', 'lavender', 'magenta', 'purple', 'violet',
    'rust orange', 'orange', 'peach',
    'maroon', 'red', 'crimson',
    'white', 'cream', 'off white', 'beige',
    'black', 'grey', 'silver', 'gold', 'golden', 'copper', 'brown'
]

FABRICS = [
    'pashmina banarasi', 'banarasi satin', 'banarasi',
    'organza tissue', 'pure organza', 'organza',
    'ajrakh printed', 'ajrakh',
    'pashmina',
    'linen silk', 'fancy linen', 'linen',
    'satin printed', 'satin',
    'munga crape', 'munga silk', 'munga', 'crape', 'crepe',
    'mysore silk', 'pure mysore silk', 'kanchipuram', 'chanderi',
    'tussar silk', 'tussar', 'kadiyal', 'georgette', 'pure silk', 'semi silk',
    'cotton silk', 'cotton', 'tissue', 'chiffon', 'patola', 'bandhani', 'habutai', 'chikankari'
]

PATTERNS = [
    'zari border', 'golden zari', 'contrast border', 'kadiyal border',
    'madhubani print', 'ajrakh print', 'lotus print', 'lotus printed', 'floral', 'printed',
    'aplic work', 'applique work', 'geometric zari', 'checks', 'embroidery',
    'traditional art', 'butti', 'all-over', 'mirror work', 'chikankari'
]


@lru_cache(maxsize=1)
def _load_index_and_meta():
    index = faiss.read_index(config.INDEX_PATH)
    meta = pd.read_parquet(config.METADATA_PATH)
    return index, meta


def load_image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=config.DOWNLOAD_TIMEOUT, headers=config.REQUEST_HEADERS)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def extract_attributes(name: str) -> dict[str, str]:
    """Extract structured color, fabric, and pattern attributes from saree title."""
    t = name.lower()
    found_color = None
    for c in sorted(COLORS, key=len, reverse=True):
        if re.search(r'\b' + re.escape(c) + r'\b', t):
            found_color = c.title()
            break

    found_fabric = None
    for f in sorted(FABRICS, key=len, reverse=True):
        if re.search(r'\b' + re.escape(f) + r'\b', t):
            found_fabric = f.title()
            break

    found_pattern = None
    for p in sorted(PATTERNS, key=len, reverse=True):
        if re.search(r'\b' + re.escape(p) + r'\b', t):
            found_pattern = p.title()
            break

    return {
        "color": found_color or "Multicolor",
        "fabric": found_fabric or "Silk Blend",
        "pattern": found_pattern or "Traditional Art",
    }


def search_sarees(
    query_image: Optional[Image.Image] = None,
    color: Optional[str] = None,
    fabric: Optional[str] = None,
    pattern: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    keyword: Optional[str] = None,
    top_k: int = config.DEFAULT_TOP_K,
) -> list[dict]:
    """
    Advanced multi-modal & multi-attribute saree search.
    - If query_image is given: FAISS similarity search + optional metadata filtering.
    - If query_image is None: Metadata-driven catalogue filtering & retrieval.
    """
    top_k = max(1, min(top_k, 20))
    index, meta = _load_index_and_meta()

    if query_image is not None:
        query_vec = get_fused_embedding(query_image).reshape(1, -1).astype("float32")
        scores, indices = index.search(query_vec, index.ntotal)
        scores_arr = scores[0]
        indices_arr = indices[0]
    else:
        indices_arr = np.arange(len(meta))
        scores_arr = np.ones(len(meta))

    results = []
    for score, idx in zip(scores_arr, indices_arr):
        if idx == -1 or idx >= len(meta):
            continue
        row = meta.iloc[idx]
        name = str(row["name"])
        name_lower = name.lower()

        try:
            p_val = float(re.sub(r"[^\d.]", "", str(row["price"])))
        except (ValueError, TypeError):
            p_val = 0.0

        attrs = extract_attributes(name)

        if color and color.lower() not in name_lower and color.lower() not in attrs["color"].lower():
            continue
        if fabric and fabric.lower() not in name_lower and fabric.lower() not in attrs["fabric"].lower():
            continue
        if pattern and pattern.lower() not in name_lower and pattern.lower() not in attrs["pattern"].lower():
            continue
        if min_price is not None and p_val < min_price:
            continue
        if max_price is not None and p_val > max_price:
            continue
        if keyword and keyword.lower() not in name_lower:
            continue

        results.append(
            {
                "name": name,
                "sku": row.get("sku", ""),
                "price": row.get("price", ""),
                "image_url": row["image_url"],
                "product_link": row.get("product_link", ""),
                "score": round(float(score), 4) if query_image is not None else 1.0,
                "color": attrs["color"],
                "fabric": attrs["fabric"],
                "pattern": attrs["pattern"],
            }
        )
        if len(results) >= top_k:
            break

    return results


def search_similar_sarees(query_image: Image.Image, top_k: int = config.DEFAULT_TOP_K) -> list[dict]:
    """Compatibility wrapper for selftest and standard similarity search."""
    return search_sarees(query_image=query_image, top_k=top_k)


def index_size() -> int:
    index, _ = _load_index_and_meta()
    return index.ntotal
