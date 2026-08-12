"""
One-time (or re-run-when-catalogue-changes) offline job:

    data/products.csv  -->  download each image  -->  fuse embedding
                        -->  data/saree_index.faiss (vectors)
                        -->  data/metadata.parquet   (name, url, price, link)

Run this LOCALLY (or on Colab) BEFORE deploying -- Streamlit Community
Cloud's free tier is not meant to download + embed 1000+ images on every
boot. We commit the small resulting index + metadata files to the repo and
the deployed app just loads them (see search_tool.py).

Usage:
    python ingest.py                # embed everything not yet indexed
    python ingest.py --limit 50     # quick smoke test on first 50 rows
    python ingest.py --rebuild      # wipe and start over
"""
from __future__ import annotations

import argparse
import io
import os
import time

import faiss
import numpy as np
import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import config
from embeddings import get_fused_embedding, embedding_dim


def download_image(url: str) -> Image.Image | None:
    for attempt in range(config.DOWNLOAD_RETRIES + 1):
        try:
            resp = requests.get(
                url, timeout=config.DOWNLOAD_TIMEOUT, headers=config.REQUEST_HEADERS
            )
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
            img.load()
            return img
        except (requests.RequestException, UnidentifiedImageError, OSError):
            if attempt < config.DOWNLOAD_RETRIES:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process first N rows")
    parser.add_argument("--rebuild", action="store_true", help="ignore any existing index")
    args = parser.parse_args()

    os.makedirs(config.DATA_DIR, exist_ok=True)
    df = pd.read_csv(config.PRODUCTS_CSV)
    df = df.dropna(subset=["image_url"]).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    if args.rebuild or not os.path.exists(config.METADATA_PATH):
        done_urls = set()
        records = []
        vectors = []
    else:
        existing_meta = pd.read_parquet(config.METADATA_PATH)
        existing_index = faiss.read_index(config.INDEX_PATH)
        done_urls = set(existing_meta["image_url"])
        records = existing_meta.to_dict("records")
        vectors = [existing_index.reconstruct(i) for i in range(existing_index.ntotal)]
        print(f"Resuming: {len(done_urls)} images already indexed.")

    failed = []
    todo = df[~df["image_url"].isin(done_urls)]
    print(f"Embedding {len(todo)} new images (of {len(df)} total rows)...")

    for _, row in tqdm(todo.iterrows(), total=len(todo)):
        url = row["image_url"]
        img = download_image(url)
        if img is None:
            failed.append({"name": row["Name"], "image_url": url, "reason": "download_failed"})
            continue
        try:
            vec = get_fused_embedding(img)
        except Exception as e:  # noqa: BLE001 - log and keep going, don't kill a 1000-image run
            failed.append({"name": row["Name"], "image_url": url, "reason": f"embed_failed: {e}"})
            continue

        vectors.append(vec)
        records.append(
            {
                "name": row["Name"],
                "sku": row.get("SKU", ""),
                "price": row.get("Discounted Price", row.get("Retail Price", "")),
                "image_url": url,
                "product_link": row.get("Website Link", ""),
            }
        )

    if not vectors:
        print("No vectors embedded -- nothing to save. Check network access to image host.")
        return

    dim = embedding_dim()
    mat = np.vstack(vectors).astype("float32")
    index = faiss.IndexFlatIP(dim)  # inner product on L2-normalized vecs == cosine similarity
    index.add(mat)
    faiss.write_index(index, config.INDEX_PATH)

    meta_df = pd.DataFrame(records)
    meta_df.to_parquet(config.METADATA_PATH, index=False)

    if failed:
        pd.DataFrame(failed).to_csv(config.FAILED_LOG, index=False)

    print(f"Done. Indexed {index.ntotal} sarees -> {config.INDEX_PATH}")
    print(f"Metadata -> {config.METADATA_PATH}")
    if failed:
        print(f"{len(failed)} images failed to download/embed -> {config.FAILED_LOG}")


if __name__ == "__main__":
    main()
