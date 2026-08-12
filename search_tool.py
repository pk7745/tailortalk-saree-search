"""
Search tool for TailorTalk Saree Search.

Implements hybrid search:
1. First retrieves vector similarity results from FAISS (over-fetching candidates).
2. Then applies deterministic metadata filters (price, color, fabric).
3. Returns the top_k items that satisfy all criteria, strictly ranked by cosine similarity.
4. Includes enriched specifications (material, length, blouse, wash care, etc.) when present.
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

# Fixed vocabulary list for deterministic string matching
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
    'pashmina banarasi', 'banarasi satin', 'banarasi', 'banaras',
    'organza tissue', 'pure organza', 'organza',
    'ajrakh printed', 'ajrakh',
    'pashmina',
    'linen silk', 'fancy linen', 'linen',
    'satin printed', 'satin',
    'munga crape', 'munga silk', 'munga', 'crape', 'crepe',
    'mysore silk', 'pure mysore silk', 'kanchipuram', 'chanderi',
    'tussar silk', 'tussar', 'kadiyal', 'georgette', 'pure silk', 'semi silk',
    'cotton silk', 'cotton', 'tissue', 'chiffon', 'patola', 'bandhani', 'habutai', 'chikankari', 'silk'
]

PATTERNS = [
    'zari border', 'golden zari', 'contrast border', 'kadiyal border',
    'madhubani print', 'ajrakh print', 'lotus print', 'lotus printed', 'floral', 'printed',
    'aplic work', 'applique work', 'geometric zari', 'checks', 'embroidery',
    'traditional art', 'butti', 'all-over', 'mirror work', 'chikankari', 'zari', 'embroidered'
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


def parse_query_intent(text: str) -> dict:
    """Extract colors, fabrics, patterns, budget constraints, and count from user prompt."""
    t = text.lower()

    max_p = None
    min_p = None

    m_between = re.search(r'between\s*(?:rs\.?|inr|₹)?\s*(\d+)\s*(?:and|to|-)\s*(?:rs\.?|inr|₹)?\s*(\d+)', t)
    if m_between:
        min_p = float(m_between.group(1))
        max_p = float(m_between.group(2))
    else:
        m_under = re.search(r'(?:under|below|less\s+than|upto|up\s+to|max|budget|within|<=?)\s*(?:rs\.?|inr|₹)?\s*(\d+)', t)
        if m_under:
            max_p = float(m_under.group(1))

        m_above = re.search(r'(?:above|over|more\s+than|min|>=?)\s*(?:rs\.?|inr|₹)?\s*(\d+)', t)
        if m_above:
            min_p = float(m_above.group(1))

    color_found = None
    for c in sorted(COLORS, key=len, reverse=True):
        if re.search(r'\b' + re.escape(c) + r'\b', t):
            color_found = c
            break

    fabric_found = None
    for f in sorted(FABRICS, key=len, reverse=True):
        if re.search(r'\b' + re.escape(f) + r'\b', t):
            fabric_found = f
            break

    pattern_found = None
    for p in sorted(PATTERNS, key=len, reverse=True):
        if re.search(r'\b' + re.escape(p) + r'\b', t):
            pattern_found = p
            break

    top_k = config.DEFAULT_TOP_K
    m_count = re.search(r'\b(?:show|get|find|top)?\s*(\d+)\s*(?:sarees|items|matches|results)\b', t)
    if m_count and m_count.group(1):
        val = int(m_count.group(1))
        if 1 <= val <= 20 and val != int(max_p or 0) and val != int(min_p or 0):
            top_k = val

    return {
        "color": color_found,
        "fabric": fabric_found,
        "pattern": pattern_found,
        "min_price": min_p,
        "max_price": max_p,
        "top_k": top_k,
    }


def search_similar_sarees(
    query_image: Optional[Image.Image] = None,
    top_k: int = config.DEFAULT_TOP_K,
    max_price: Optional[float] = None,
    min_price: Optional[float] = None,
    color: Optional[str] = None,
    fabric: Optional[str] = None,
) -> list[dict]:
    """
    Hybrid similarity search:
    1. Over-fetch vector similarity scores from FAISS (e.g. all or top candidates).
    2. Filter candidates on max_price, min_price, color, fabric metadata.
    3. Return the top_k filtered items, ranked strictly by cosine similarity score.
    """
    top_k = max(1, min(top_k, config.MAX_TOP_K if max_price is None and color is None and fabric is None else 20))
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
        name_str = str(row["name"]).lower()
        color_str = str(row.get("color", "")).lower()
        fabric_str = str(row.get("fabric", "")).lower()

        try:
            p_val = float(re.sub(r"[^\d.]", "", str(row["price"])))
        except (ValueError, TypeError):
            p_val = 0.0

        if color and color.lower() not in name_str and color.lower() not in color_str:
            continue
        if fabric and fabric.lower() not in name_str and fabric.lower() not in fabric_str:
            continue
        if max_price is not None and p_val > max_price:
            continue
        if min_price is not None and p_val < min_price:
            continue

        item_dict = {
            "name": row["name"],
            "sku": row.get("sku", ""),
            "price": row.get("price", ""),
            "image_url": row["image_url"],
            "product_link": row.get("product_link", ""),
            "score": round(float(score), 4) if query_image is not None else 1.0,
            "color": row.get("color", "").title() if row.get("color") else "Multicolor",
            "fabric": row.get("fabric", "").title() if row.get("fabric") else "Silk Blend",
            "pattern": row.get("pattern", "").title() if row.get("pattern") else "Classic",
        }

        # Include enriched on-page specs when present
        for field in [
            "material",
            "blouse_included",
            "blouse_length",
            "saree_length",
            "saree_weight",
            "wash_care",
            "net_quantity",
            "occasion",
            "work_type",
            "stock_status",
        ]:
            val = row.get(field)
            if pd.notnull(val) and val != "" and val is not None:
                item_dict[field] = str(val)

        results.append(item_dict)
        if len(results) >= top_k:
            break

    return results


def index_size() -> int:
    index, _ = _load_index_and_meta()
    return index.ntotal
