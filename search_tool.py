"""
Loads the pre-built FAISS index + metadata once, and exposes the single
function that both the Streamlit app and the LangChain agent tool call
into: search_similar_sarees().

This is intentionally decoupled from LangChain so it also works from a
plain unit test / notebook, and so the "tool" the LLM calls is a thin
schema-validated wrapper around one well-tested function -- not business
logic tangled inside agent code.
"""
from __future__ import annotations

import io
from functools import lru_cache

import faiss
import numpy as np
import pandas as pd
import requests
from PIL import Image

import config
from embeddings import get_fused_embedding


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


def search_similar_sarees(query_image: Image.Image, top_k: int = config.DEFAULT_TOP_K) -> list[dict]:
    """
    Core similarity search. Returns a ranked list of dicts:
        {name, sku, price, image_url, product_link, score}
    `score` is cosine similarity in [-1, 1] (in practice ~[0.3, 1.0] for
    this fused embedding space on this catalogue), higher = more similar.
    """
    top_k = max(1, min(top_k, config.MAX_TOP_K))
    index, meta = _load_index_and_meta()

    query_vec = get_fused_embedding(query_image).reshape(1, -1).astype("float32")
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        row = meta.iloc[idx]
        results.append(
            {
                "name": row["name"],
                "sku": row["sku"],
                "price": row["price"],
                "image_url": row["image_url"],
                "product_link": row["product_link"],
                "score": round(float(score), 4),
            }
        )
    return results


def index_size() -> int:
    index, _ = _load_index_and_meta()
    return index.ntotal
