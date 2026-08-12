"""
Search tool for TailorTalk Saree Search.

Implements hybrid search:
1. First retrieves vector similarity results from FAISS (over-fetching candidates).
2. Then applies deterministic metadata filters (price, color, fabric, pattern).
3. Returns the top_k items that satisfy all criteria, strictly ranked by cosine similarity.
4. Includes enriched 4-tier specifications (specs_source, sibling_sku, material, length, blouse, wash care, etc.).
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
    'baby pink', 'rani pink', 'peach pink', 'dusty pink', 'onion pink', 'pink',
    'navy blue', 'sky blue', 'royal blue', 'dark blue', 'peacock blue', 'rama blue', 'blue',
    'mint green', 'bottle green', 'pista green', 'dark green', 'olive green', 'pastel green', 'sea green', 'green',
    'dark yellow', 'mustard yellow', 'mustard', 'lemon yellow', 'yellow',
    'dusty purple', 'lavender', 'magenta', 'purple', 'violet', 'lilac', 'mauve',
    'rust orange', 'orange', 'peach', 'coral',
    'maroon', 'red', 'crimson', 'wine', 'burgundy',
    'white', 'cream', 'off white', 'ivory', 'beige',
    'black', 'grey', 'silver', 'gold', 'golden', 'copper', 'brown', 'coffee'
]

FABRICS = [
    'pashmina banarasi', 'banarasi satin', 'banarasi crape', 'kora banarasi', 'banarasi', 'banaras',
    'organza tissue', 'pure organza', 'semi organza', 'tissue organza', 'organza',
    'ajrakh printed', 'ajrakh',
    'pashmina',
    'linen silk', 'fancy linen', 'linen cotton', 'linen',
    'satin printed', 'satin',
    'munga crape', 'munga silk', 'munga', 'crape', 'crepe',
    'mysore silk', 'pure mysore silk', 'kanchipuram', 'semi kanchipuram', 'chanderi', 'tissue chanderi',
    'tussar silk', 'pure tussar', 'semi tussar', 'tussar', 'kadiyal', 'georgette', 'semi khaddhi georgette', 'pure silk', 'semi silk',
    'cotton silk', 'mul cotton', 'maheshwari cotton', 'kota cotton', 'cotton', 'tissue', 'chiffon', 'patola', 'bandhani', 'habutai', 'chikankari', 'silk'
]

PATTERNS = [
    'zari border', 'golden zari', 'contrast border', 'kadiyal border', 'temple border', 'rising border',
    'madhubani print', 'ajrakh print', 'lotus print', 'lotus printed', 'floral print', 'floral', 'printed',
    'aplic work', 'applique work', 'geometric zari', 'checks', 'embroidery', 'embroidered',
    'traditional art', 'butti', 'all-over', 'mirror work', 'chikankari', 'zari', 'stripes', 'circle print'
]

WORD_TO_NUM = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'thousand': 1000, 'k': 1000
}


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

    # Handle text numbers like "three thousand" -> 3000, "3k" -> 3000
    t_norm = t
    t_norm = re.sub(r'\bthree\s+thousand\b', '3000', t_norm)
    t_norm = re.sub(r'\btwo\s+thousand\b', '2000', t_norm)
    t_norm = re.sub(r'\bfour\s+thousand\b', '4000', t_norm)
    t_norm = re.sub(r'\bfive\s+thousand\b', '5000', t_norm)
    t_norm = re.sub(r'\bten\s+thousand\b', '10000', t_norm)
    t_norm = re.sub(r'(\d+)\s*k\b', lambda m: str(int(m.group(1)) * 1000), t_norm)

    # 1. Range: between X and Y
    m_between = re.search(r'between\s*(?:rs\.?|inr|₹)?\s*(\d+)\s*(?:and|to|-)\s*(?:rs\.?|inr|₹)?\s*(\d+)', t_norm)
    if m_between:
        min_p = float(m_between.group(1))
        max_p = float(m_between.group(2))
    else:
        # 2. Max price variations
        m_under = re.search(
            r'(?:under|below|cheaper\s+than|less\s+than|budget\s+below|within|not\s+exceeding|upto|up\s+to|max|budget|<=?)\s*(?:rs\.?|inr|₹)?\s*(\d+)',
            t_norm
        )
        if not m_under:
            m_under = re.search(r'(?:rs\.?|inr|₹)?\s*(\d+)\s*(?:max|or\s+less|or\s+under|budget)', t_norm)

        if m_under:
            max_p = float(m_under.group(1))

        # 3. Min price variations
        m_above = re.search(
            r'(?:above|over|more\s+than|costlier\s+than|higher\s+than|min|starting\s+from|>=?)\s*(?:rs\.?|inr|₹)?\s*(\d+)',
            t_norm
        )
        if m_above:
            min_p = float(m_above.group(1))

    color_found = None
    for c in sorted(COLORS, key=len, reverse=True):
        if re.search(r'\b' + re.escape(c) + r'\b', t_norm):
            color_found = c
            break

    fabric_found = None
    for f in sorted(FABRICS, key=len, reverse=True):
        if re.search(r'\b' + re.escape(f) + r'\b', t_norm):
            fabric_found = f
            break

    pattern_found = None
    for p in sorted(PATTERNS, key=len, reverse=True):
        if re.search(r'\b' + re.escape(p) + r'\b', t_norm):
            pattern_found = p
            break

    top_k = config.DEFAULT_TOP_K
    m_count = re.search(r'\b(?:show|get|find|top)?\s*(\d+)\s*(?:sarees|items|matches|results)\b', t_norm)
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
    1. Over-fetch vector similarity scores from FAISS (all 1,070 candidates).
    2. Filter candidates on max_price, min_price, color, fabric metadata.
    3. Return the top_k filtered items, ranked strictly by cosine similarity score.
    4. Includes 4-tier provenance tag (specs_source) and sibling_sku if applicable.
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

        raw_score = float(score) if query_image is not None else 1.0
        specs_src = row.get("specs_source", "own_page")

        item_dict = {
            "name": row["name"],
            "sku": row.get("sku", ""),
            "price": row.get("price", ""),
            "image_url": row["image_url"],
            "product_link": row.get("product_link", ""),
            "score": round(raw_score, 4),
            "is_weak_match": raw_score < 0.60,
            "specs_source": specs_src,
            "sibling_sku": row.get("sibling_sku"),
            "color": row.get("color", "").title() if row.get("color") else "Multicolor",
            "fabric": row.get("fabric", "").title() if row.get("fabric") else "Silk Blend",
            "pattern": row.get("pattern", "").title() if row.get("pattern") else "Classic",
        }

        # Include 4-tier specifications
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
