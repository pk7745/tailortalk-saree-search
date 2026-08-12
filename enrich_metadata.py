"""
Script: enrich_metadata.py
Enriches data/metadata.parquet with real scraped product specifications from product_link.

Fields extracted when present on the live page:
- material (fabric composition)
- blouse_included (e.g. 'Attached', 'Unstitched')
- blouse_length (e.g. '0.9m', '1m')
- saree_length (e.g. '5.5m', '5.6m')
- saree_weight (e.g. '0.55', '0.7kg')
- wash_care (e.g. 'Dry Clean')
- net_quantity (e.g. '1N')
- occasion
- work_type
- stock_status (e.g. 'In Stock', 'Out of Stock', 'Available for Preorder')

Resumable, rate-limited, non-blocking, and logs failures to data/enrich_failed.csv.
"""
from __future__ import annotations

import concurrent.futures
import os
import re
import sys
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

import config

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
ENRICH_FAILED_CSV = os.path.join(config.DATA_DIR, "enrich_failed.csv")
TIMEOUT = 15


def parse_product_html(html_text: str) -> dict:
    soup = BeautifulSoup(html_text, "html.parser")

    h1 = soup.find(["h1", "h2"])
    if h1 and "404" in h1.get_text():
        return {}

    extracted = {
        "material": None,
        "blouse_included": None,
        "blouse_length": None,
        "saree_length": None,
        "saree_weight": None,
        "wash_care": None,
        "net_quantity": None,
        "occasion": None,
        "work_type": None,
        "stock_status": None,
    }

    raw_pairs = []
    # 1. Parse table rows
    for tr in soup.find_all("tr"):
        t = tr.get_text(" ", strip=True)
        if ":-" in t:
            parts = t.split(":-", 1)
            raw_pairs.append((parts[0].strip(), parts[1].strip()))
        elif ":" in t and not any(k in t.lower() for k in ["http", "sku", "note"]):
            parts = t.split(":", 1)
            raw_pairs.append((parts[0].strip(), parts[1].strip()))

    # 2. Parse text blocks (p, li, div)
    for tag in soup.find_all(["p", "li", "div"]):
        lines = tag.get_text("\n", strip=True).split("\n")
        for line in lines:
            if ":-" in line:
                parts = line.split(":-", 1)
                raw_pairs.append((parts[0].strip(), parts[1].strip()))
            elif ":" in line and len(line) < 100:
                parts = line.split(":", 1)
                k, v = parts[0].strip(), parts[1].strip()
                if any(
                    target in k.lower()
                    for target in [
                        "material",
                        "colour",
                        "color",
                        "blouse",
                        "length",
                        "weight",
                        "washing",
                        "wash",
                        "occasion",
                        "work",
                        "quantity",
                    ]
                ):
                    raw_pairs.append((k, v))

    for k_raw, v_raw in raw_pairs:
        k = k_raw.lower().replace(" ", "").replace("_", "")
        v = re.split(
            r"(?:Colou?r|Blouse|Saree|Washing|Net|Note|Quantity|Material|SKU)",
            v_raw,
            flags=re.IGNORECASE,
        )[0].strip()
        v = v.rstrip(".,-").strip()
        if not v:
            continue

        if "material" in k or "fabric" in k:
            if not extracted["material"]:
                extracted["material"] = v
        elif "blouselength" in k or "blousepiecelength" in k:
            if not extracted["blouse_length"]:
                extracted["blouse_length"] = v
        elif "blouse" in k:
            if not extracted["blouse_included"]:
                extracted["blouse_included"] = v
        elif "sareelength" in k or "length" in k:
            if not extracted["saree_length"]:
                extracted["saree_length"] = v
        elif "sareeweight" in k or "weight" in k:
            if not extracted["saree_weight"]:
                extracted["saree_weight"] = v
        elif "washing" in k or "wash" in k:
            if not extracted["wash_care"]:
                extracted["wash_care"] = v
        elif "netquantity" in k or "quantity" in k:
            if not extracted["net_quantity"] and len(v) <= 10:
                extracted["net_quantity"] = v
        elif "occasion" in k:
            if not extracted["occasion"]:
                extracted["occasion"] = v
        elif "work" in k or "craft" in k:
            if not extracted["work_type"]:
                extracted["work_type"] = v

    full_text = soup.get_text(" ", strip=True).lower()
    if "out of stock" in full_text:
        extracted["stock_status"] = "Out of Stock"
    elif "preorder" in full_text:
        extracted["stock_status"] = "Available for Preorder"
    elif "add to cart" in full_text or "buy now" in full_text:
        extracted["stock_status"] = "In Stock"

    return extracted


def fetch_and_enrich(row_dict: dict) -> tuple[dict, dict | None]:
    url = row_dict.get("product_link", "")
    sku = row_dict.get("sku", "")
    name = row_dict.get("name", "")

    empty_res = {
        "sku": sku,
        "material": None,
        "blouse_included": None,
        "blouse_length": None,
        "saree_length": None,
        "saree_weight": None,
        "wash_care": None,
        "net_quantity": None,
        "occasion": None,
        "work_type": None,
        "stock_status": None,
    }

    if not url or not str(url).startswith("http"):
        return empty_res, {"sku": sku, "name": name, "url": url, "reason": "missing_or_invalid_url"}

    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code == 200:
            parsed = parse_product_html(resp.text)
            parsed["sku"] = sku
            return parsed, None
        else:
            return empty_res, {"sku": sku, "name": name, "url": url, "reason": f"http_{resp.status_code}"}
    except Exception as e:
        return empty_res, {"sku": sku, "name": name, "url": url, "reason": str(e)}


def main():
    if not os.path.exists(config.METADATA_PATH):
        print(f"Metadata file {config.METADATA_PATH} not found.")
        sys.exit(1)

    meta = pd.read_parquet(config.METADATA_PATH)
    print(f"Loaded metadata.parquet with {len(meta)} records.")

    enriched_cache_path = os.path.join(config.DATA_DIR, "enriched_details.parquet")
    existing_enriched = {}
    if os.path.exists(enriched_cache_path):
        prev_df = pd.read_parquet(enriched_cache_path)
        existing_enriched = {r["sku"]: r for r in prev_df.to_dict("records")}
        print(f"Resuming: {len(existing_enriched)} records already enriched.")

    rows_to_do = [r for r in meta.to_dict("records") if r["sku"] not in existing_enriched]
    print(f"Fetching details for {len(rows_to_do)} products...")

    enriched_results = list(existing_enriched.values())
    failures = []

    # Multi-threaded polite fetcher with 8 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_and_enrich, r): r for r in rows_to_do}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            parsed, fail = future.result()
            enriched_results.append(parsed)
            if fail:
                failures.append(fail)
            time.sleep(0.05)

    enriched_df = pd.DataFrame(enriched_results).drop_duplicates(subset=["sku"])
    enriched_df.to_parquet(enriched_cache_path, index=False)

    # Join enriched fields directly into metadata.parquet
    detail_cols = [
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
    ]
    for col in detail_cols:
        if col in meta.columns:
            meta = meta.drop(columns=[col])

    merged_meta = meta.merge(enriched_df[["sku"] + detail_cols], on="sku", how="left")
    merged_meta.to_parquet(config.METADATA_PATH, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(ENRICH_FAILED_CSV, index=False)

    print("\n" + "=" * 60)
    print(f"Enrichment Complete!")
    print(f"Total rows in metadata: {len(merged_meta)}")
    successful_count = sum(1 for r in enriched_results if r.get("material") or r.get("blouse_included") or r.get("saree_length"))
    print(f"Successfully enriched with on-page specs: {successful_count}")
    print(f"Failed / Unavailable pages logged: {len(failures)} -> {ENRICH_FAILED_CSV}")


if __name__ == "__main__":
    main()
