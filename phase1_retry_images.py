"""
Phase 1: Retry downloading the 4 failed images and add them to the FAISS index.
SKUs: QS264566, QS270932, QS282590, QS282741
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import time
import faiss
import numpy as np
import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError

import config
from embeddings import get_fused_embedding, embedding_dim

MISSING_SKUS = ["QS264566", "QS270932", "QS282590", "QS282741"]

HEADERS_LIST = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"},
    {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"},
]

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
    'ajrakh printed', 'ajrakh', 'pashmina',
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

import re

def extract_attributes(name):
    t = name.lower()
    c = next((c for c in sorted(COLORS, key=len, reverse=True) if re.search(r'\b' + re.escape(c) + r'\b', t)), '')
    f = next((f for f in sorted(FABRICS, key=len, reverse=True) if re.search(r'\b' + re.escape(f) + r'\b', t)), '')
    p = next((p for p in sorted(PATTERNS, key=len, reverse=True) if re.search(r'\b' + re.escape(p) + r'\b', t)), '')
    return c, f, p


def download_image(url, max_retries=5, timeout=30):
    for attempt in range(max_retries):
        headers = HEADERS_LIST[attempt % len(HEADERS_LIST)]
        try:
            resp = requests.get(url, timeout=timeout, headers=headers)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
            img.load()
            return img
        except Exception as e:
            print(f"  Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
    return None


def main():
    csv_df = pd.read_csv(config.PRODUCTS_CSV)
    meta = pd.read_parquet(config.METADATA_PATH)
    index = faiss.read_index(config.INDEX_PATH)

    print(f"Current state: {len(meta)} metadata rows, {index.ntotal} FAISS vectors")
    print(f"Retrying {len(MISSING_SKUS)} failed image downloads...")

    existing_urls = set(meta["image_url"].values)
    records = meta.to_dict("records")
    vectors = [index.reconstruct(i) for i in range(index.ntotal)]

    succeeded = []
    still_failed = []

    for sku in MISSING_SKUS:
        rows = csv_df[csv_df["SKU"] == sku]
        if len(rows) == 0:
            print(f"  SKU {sku}: NOT FOUND in CSV")
            still_failed.append({"sku": sku, "reason": "not_in_csv"})
            continue

        for _, row in rows.iterrows():
            url = row["image_url"]
            if url in existing_urls:
                print(f"  SKU {sku} image {url[:60]}... already indexed, skipping")
                continue

            print(f"  SKU {sku}: downloading {url[:60]}...")
            img = download_image(url)

            if img is None:
                print(f"  SKU {sku}: FAILED after all retries")
                still_failed.append({
                    "sku": sku,
                    "name": row["Name"],
                    "image_url": url,
                    "reason": "download_failed_after_5_retries"
                })
                continue

            print(f"  SKU {sku}: Image downloaded OK ({img.size}), embedding...")
            try:
                vec = get_fused_embedding(img)
            except Exception as e:
                print(f"  SKU {sku}: Embedding FAILED: {e}")
                still_failed.append({
                    "sku": sku,
                    "name": row["Name"],
                    "image_url": url,
                    "reason": f"embed_failed: {e}"
                })
                continue

            c, f, p = extract_attributes(row["Name"])
            new_record = {
                "name": row["Name"],
                "sku": row["SKU"],
                "price": row.get("Discounted Price", row.get("Retail Price", "")),
                "image_url": url,
                "product_link": row.get("Website Link", ""),
                "color": c,
                "fabric": f,
                "pattern": p,
            }
            # Add empty columns for existing metadata fields
            for col in meta.columns:
                if col not in new_record:
                    new_record[col] = None

            records.append(new_record)
            vectors.append(vec)
            existing_urls.add(url)
            succeeded.append(sku)
            print(f"  SKU {sku}: SUCCESS - embedded and added")

    if succeeded:
        # Rebuild FAISS index with all vectors
        dim = embedding_dim()
        mat = np.vstack(vectors).astype("float32")
        new_index = faiss.IndexFlatIP(dim)
        new_index.add(mat)
        faiss.write_index(new_index, config.INDEX_PATH)

        new_meta = pd.DataFrame(records)
        new_meta.to_parquet(config.METADATA_PATH, index=False)

        print(f"\nUpdated: {new_index.ntotal} FAISS vectors, {len(new_meta)} metadata rows")
    else:
        print("\nNo new images recovered.")

    print(f"\n=== PHASE 1 RESULTS ===")
    print(f"Succeeded: {len(succeeded)} - {succeeded}")
    print(f"Still failed: {len(still_failed)}")
    for f in still_failed:
        print(f"  {f}")

    # Update failed_downloads.csv
    if still_failed:
        pd.DataFrame(still_failed).to_csv(config.FAILED_LOG, index=False)
        print(f"Updated {config.FAILED_LOG}")

    # Final verification
    meta_final = pd.read_parquet(config.METADATA_PATH)
    idx_final = faiss.read_index(config.INDEX_PATH)
    print(f"\nFinal: {len(meta_final)} metadata rows, {idx_final.ntotal} FAISS vectors")
    print(f"Match: {len(meta_final) == idx_final.ntotal}")


if __name__ == "__main__":
    main()
