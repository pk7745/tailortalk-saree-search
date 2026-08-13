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
import os
import re
from functools import lru_cache
from typing import Optional

import config
import faiss
import numpy as np
import pandas as pd
import requests
from PIL import Image

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
    'zari border', 'golden zari border', 'golden zari', 'gold zari border', 'gold zari',
    'silver zari border', 'silver zari', 'contrast border', 'kadiyal border', 'kadiyal',
    'temple border', 'temple motif', 'temple design', 'rising border', 'kanchi border',
    'madhubani print', 'ajrakh print', 'ajrakh printed', 'lotus print', 'lotus printed',
    'floral print', 'floral work', 'floral', 'printed',
    'aplic work', 'applique work', 'applique',
    'geometric zari', 'geometric print', 'checks',
    'embroidery', 'embroidered', 'traditional art', 'butti', 'all-over', 'mirror work',
    'chikankari', 'zari work', 'zari', 'stripes', 'circle print',
    'kutch work', 'brocade', 'ikkat', 'jacquard',
    'parrot pallu', 'zari pallu', 'floral pallu', 'pallu', 'border'
]

WORD_TO_NUM = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'thousand': 1000, 'k': 1000
}


def _matches_pattern(pattern_query: str, row: pd.Series) -> bool:
    """Check if a saree row matches the requested pattern, border, pallu, or work type."""
    if not pattern_query:
        return True

    pq = pattern_query.lower().strip()
    name_str = str(row.get("name", "")).lower()
    pattern_str = str(row.get("pattern", "")).lower()
    material_str = str(row.get("material", "")).lower()
    work_str = str(row.get("work_type", "")).lower()
    border_str = str(row.get("border", "")).lower()
    pallu_str = str(row.get("pallu", "")).lower()
    nb_str = str(row.get("name_border", "")).lower()
    np_str = str(row.get("name_pallu", "")).lower()
    nw_str = str(row.get("name_work", "")).lower()
    vb_str = str(row.get("visual_border_detected", "")).lower()
    vz_str = str(row.get("visual_zari_detected", "")).lower()
    vc_str = str(row.get("visual_contrast_border", "")).lower()

    combined_text = f"{name_str} {pattern_str} {material_str} {work_str} {border_str} {pallu_str} {nb_str} {np_str} {nw_str} {vb_str} {vz_str} {vc_str}"

    if pq in combined_text:
        return True

    synonyms = {
        'zari border': ['zari border', 'golden zari', 'gold zari', 'zari'],
        'golden zari': ['golden zari', 'gold zari', 'zari border', 'zari'],
        'gold zari': ['gold zari', 'golden zari', 'zari border', 'zari'],
        'zari work': ['zari border', 'golden zari', 'gold zari', 'zari'],
        'zari': ['zari border', 'golden zari', 'gold zari', 'zari'],
        'temple border': ['temple border', 'temple', 'temple motif'],
        'kadiyal border': ['kadiyal border', 'kadiyal'],
        'kadiyal': ['kadiyal border', 'kadiyal'],
        'contrast border': ['contrast border', 'contrast'],
        'rising border': ['rising border'],
        'kanchi border': ['kanchi border', 'kanchi'],
        'floral work': ['floral', 'floral print', 'floral embroidery'],
        'floral print': ['floral', 'floral print'],
        'floral': ['floral', 'floral print'],
        'embroidery': ['embroidery', 'embroidered', 'chikankari', 'kutch work'],
        'embroidered': ['embroidery', 'embroidered', 'chikankari', 'kutch work'],
        'applique': ['aplic work', 'applique work', 'aplic', 'applique', 'turkish aplique'],
        'applique work': ['aplic work', 'applique work', 'aplic', 'applique', 'turkish aplique'],
        'aplic work': ['aplic work', 'applique work', 'aplic', 'applique', 'turkish aplique'],
        'geometric zari': ['geometric zari', 'geometric print', 'geometric'],
        'checks': ['checks', 'check'],
        'stripes': ['stripes', 'stripe'],
        'pallu': ['pallu', 'zari pallu', 'parrot pallu', 'floral pallu'],
        'border': ['border', 'zari border', 'temple border', 'kadiyal border', 'contrast border', 'kanchi border', 'rising border'],
    }

    cand_terms = synonyms.get(pq, [pq])
    for term in cand_terms:
        if term in combined_text:
            return True

    for word in pq.split():
        if word in ['border', 'saree', 'work', 'design', 'with', 'and', 'the']:
            continue
        if word in combined_text:
            return True

    return False


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


def _compute_attribute_match_score(row: pd.Series, color: Optional[str] = None, fabric: Optional[str] = None, pattern: Optional[str] = None) -> tuple[float, bool]:
    """
    Computes an attribute match score S_attr in [0.0, 1.0] for requested metadata attributes.
    Scoring:
    1.0 = Exact match in primary catalogue/scraped metadata
    0.8 = Synonym / family match
    0.5 = Secondary visual detection signal
    0.0 = No match
    """
    scores = []

    if pattern:
        pq = pattern.lower().strip()
        name_str = str(row.get("name", "")).lower()
        pat_str = str(row.get("pattern", "")).lower()
        border_str = str(row.get("border", "")).lower()
        pallu_str = str(row.get("pallu", "")).lower()
        work_str = str(row.get("work_type", "")).lower()
        nb_str = str(row.get("name_border", "")).lower()
        np_str = str(row.get("name_pallu", "")).lower()
        nw_str = str(row.get("name_work", "")).lower()

        primary_text = f"{name_str} {pat_str} {border_str} {pallu_str} {work_str} {nb_str} {np_str} {nw_str}"

        synonyms = {
            'zari border': ['zari border', 'golden zari border', 'gold zari border', 'golden zari', 'gold zari'],
            'golden zari': ['golden zari', 'gold zari', 'zari border', 'zari'],
            'temple border': ['temple border', 'temple motif', 'temple'],
            'kadiyal border': ['kadiyal border', 'kadiyal'],
            'contrast border': ['contrast border', 'contrast'],
            'floral': ['floral print', 'floral work', 'floral embroidery', 'floral'],
            'embroidery': ['embroidery', 'embroidered', 'chikankari', 'kutch work'],
            'pallu': ['parrot pallu', 'zari pallu', 'floral pallu', 'pallu'],
        }

        if pq in primary_text:
            scores.append(1.0)
        else:
            cand_syns = synonyms.get(pq, [pq])
            if any(s in primary_text for s in cand_syns):
                scores.append(0.8)
            else:
                vb_str = str(row.get("visual_border_detected", "")).lower()
                vz_str = str(row.get("visual_zari_detected", "")).lower()
                vc_str = str(row.get("visual_contrast_border", "")).lower()
                if ("zari" in pq and "detected" in vz_str) or ("border" in pq and "detected" in vb_str):
                    scores.append(0.5)
                else:
                    scores.append(0.0)

    if color:
        cq = color.lower().strip()
        c_str = str(row.get("color", "")).lower()
        n_str = str(row.get("name", "")).lower()
        sc_str = str(row.get("scraped_color", "")).lower()
        combined_c = f"{c_str} {n_str} {sc_str}"
        if cq in combined_c:
            scores.append(1.0)
        elif any(part in combined_c for part in cq.split()):
            scores.append(0.8)
        else:
            scores.append(0.0)

    if fabric:
        fq = fabric.lower().strip()
        f_str = str(row.get("fabric", "")).lower()
        m_str = str(row.get("material", "")).lower()
        n_str = str(row.get("name", "")).lower()
        combined_f = f"{f_str} {m_str} {n_str}"
        if fq in combined_f:
            scores.append(1.0)
        elif any(part in combined_f for part in fq.split()):
            scores.append(0.8)
        else:
            scores.append(0.0)

    if not scores:
        return 1.0, True

    attr_score = float(np.mean(scores))
    passes = attr_score > 0.0
    return attr_score, passes


try:
    import qdrant_store
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


def search_similar_sarees(
    query_image: Optional[Image.Image] = None,
    top_k: int = config.DEFAULT_TOP_K,
    max_price: Optional[float] = None,
    min_price: Optional[float] = None,
    color: Optional[str] = None,
    fabric: Optional[str] = None,
    pattern: Optional[str] = None,
) -> list[dict]:
    """
    Hybrid similarity search with Qdrant Vector Engine & FAISS Fallback:
    1. Primary: Vector search on Qdrant Cloud / Local storage using Qdrant similarity search.
    2. Fallback: If Qdrant is disabled or unavailable, automatically falls back to local FAISS index.
    3. Price filter: Enforces min_price and max_price budget bounds.
    4. Compute attribute relevance score S_attr for requested color, fabric, and pattern filters.
    5. Compute final combined score:
       - Image-Only: FinalScore = S_visual (100% visual dominant)
       - Image + Attributes: FinalScore = 0.75 * S_visual + 0.25 * S_attr (75% visual, 25% attribute boost)
       - Text-Only: FinalScore = S_attr
    6. Sort candidates by FinalScore descending and return top_k.
    """
    top_k = max(1, min(top_k, 20))
    has_attr_queries = bool(color or fabric or pattern)
    candidates = []

    # Attempt Primary Vector Search via Qdrant
    used_qdrant = False
    if config.USE_QDRANT and QDRANT_AVAILABLE:
        try:
            q_client = qdrant_store.get_qdrant_client()
            if qdrant_store.health_check(q_client) and qdrant_store.collection_exists(q_client, config.QDRANT_COLLECTION_NAME):
                if query_image is not None:
                    query_vec = get_fused_embedding(query_image).reshape(1, -1).astype("float32")
                else:
                    query_vec = np.ones((1, 1024), dtype="float32") / np.sqrt(1024)

                q_results = qdrant_store.search_sarees(
                    client=q_client,
                    query_vector=query_vec,
                    limit=min(200, top_k * 10),
                    min_price=min_price,
                    max_price=max_price,
                    collection_name=config.QDRANT_COLLECTION_NAME,
                )

                for hit in q_results:
                    payload = hit["payload"]
                    row = pd.Series(payload)

                    try:
                        p_val = float(payload.get("price_numeric", 0.0))
                    except (ValueError, TypeError):
                        p_val = 0.0

                    if max_price is not None and p_val > max_price:
                        continue
                    if min_price is not None and p_val < min_price:
                        continue

                    attr_score, passes = _compute_attribute_match_score(row, color=color, fabric=fabric, pattern=pattern)
                    if has_attr_queries and not passes:
                        continue

                    raw_score = float(hit["score"]) if query_image is not None else 1.0

                    if query_image is not None and has_attr_queries:
                        final_score = 0.75 * raw_score + 0.25 * attr_score
                    elif query_image is not None:
                        final_score = raw_score
                    else:
                        final_score = attr_score

                    specs_src = payload.get("specs_source", "own_page")

                    item_dict = {
                        "name": payload.get("name", ""),
                        "sku": payload.get("sku", ""),
                        "price": payload.get("price", ""),
                        "image_url": payload.get("image_url", ""),
                        "product_link": payload.get("product_link", ""),
                        "score": round(raw_score, 4),
                        "final_score": round(final_score, 4),
                        "attribute_score": round(attr_score, 4),
                        "is_weak_match": raw_score < 0.60,
                        "specs_source": specs_src,
                        "sibling_sku": payload.get("sibling_sku"),
                        "color": payload.get("color", "").title() if payload.get("color") else "Multicolor",
                        "fabric": payload.get("fabric", "").title() if payload.get("fabric") else "Silk Blend",
                        "pattern": payload.get("pattern", "").title() if payload.get("pattern") else "Classic",
                        "vector_engine": "Qdrant",
                    }

                    for field in [
                        "material", "blouse_included", "blouse_length", "saree_length",
                        "saree_weight", "wash_care", "net_quantity", "occasion",
                        "work_type", "stock_status",
                    ]:
                        val = payload.get(field)
                        if val is not None and str(val) != "" and str(val) != "None":
                            item_dict[field] = str(val)

                    candidates.append(item_dict)

                used_qdrant = True
        except Exception as q_err:
            print(f"Warning: Qdrant search encountered an error: {q_err}. Falling back to FAISS.")
            used_qdrant = False

    # Fallback to Local FAISS if Qdrant was not used or failed
    if not used_qdrant:
        index, meta = _load_index_and_meta()

        if query_image is not None:
            query_vec = get_fused_embedding(query_image).reshape(1, -1).astype("float32")
            scores, indices = index.search(query_vec, index.ntotal)
            scores_arr = scores[0]
            indices_arr = indices[0]
        else:
            indices_arr = np.arange(len(meta))
            scores_arr = np.ones(len(meta))

        for score, idx in zip(scores_arr, indices_arr):
            if idx == -1 or idx >= len(meta):
                continue
            row = meta.iloc[idx]

            try:
                p_val = float(re.sub(r"[^\d.]", "", str(row["price"])))
            except (ValueError, TypeError):
                p_val = 0.0

            if max_price is not None and p_val > max_price:
                continue
            if min_price is not None and p_val < min_price:
                continue

            attr_score, passes = _compute_attribute_match_score(row, color=color, fabric=fabric, pattern=pattern)
            if has_attr_queries and not passes:
                continue

            raw_score = float(score) if query_image is not None else 1.0

            if query_image is not None and has_attr_queries:
                final_score = 0.75 * raw_score + 0.25 * attr_score
            elif query_image is not None:
                final_score = raw_score
            else:
                final_score = attr_score

            specs_src = row.get("specs_source", "own_page")

            item_dict = {
                "name": row["name"],
                "sku": row.get("sku", ""),
                "price": row.get("price", ""),
                "image_url": row["image_url"],
                "product_link": row.get("product_link", ""),
                "score": round(raw_score, 4),
                "final_score": round(final_score, 4),
                "attribute_score": round(attr_score, 4),
                "is_weak_match": raw_score < 0.60,
                "specs_source": specs_src,
                "sibling_sku": row.get("sibling_sku"),
                "color": row.get("color", "").title() if row.get("color") else "Multicolor",
                "fabric": row.get("fabric", "").title() if row.get("fabric") else "Silk Blend",
                "pattern": row.get("pattern", "").title() if row.get("pattern") else "Classic",
                "vector_engine": "FAISS (Fallback)",
            }

            for field in [
                "material", "blouse_included", "blouse_length", "saree_length",
                "saree_weight", "wash_care", "net_quantity", "occasion",
                "work_type", "stock_status",
            ]:
                val = row.get(field)
                if pd.notnull(val) and val != "" and val is not None:
                    item_dict[field] = str(val)

            candidates.append(item_dict)

    # Deduplicate candidates using stable product identifier (product_link, image_url, or sku)
    # preserving the item with the highest final_score when duplicates occur
    unique_map = {}
    for item in candidates:
        pid = item.get("product_link") or item.get("image_url") or item.get("sku") or item.get("name")
        if pid not in unique_map or item["final_score"] > unique_map[pid]["final_score"]:
            unique_map[pid] = item

    unique_candidates = list(unique_map.values())
    unique_candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return unique_candidates[:top_k]



def index_size() -> int:
    if config.USE_QDRANT and QDRANT_AVAILABLE:
        try:
            q_client = qdrant_store.get_qdrant_client()
            info = qdrant_store.get_collection_info(q_client, config.QDRANT_COLLECTION_NAME)
            if "points_count" in info and info["points_count"] is not None:
                return info["points_count"]
        except Exception:
            pass
    index, _ = _load_index_and_meta()
    return index.ntotal
