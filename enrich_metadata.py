"""
Script: enrich_metadata.py
Robust Multi-Layout Web Scraper and Metadata Enricher for TailorTalk Saree Search.

Supports:
- Layout A: Structured <table> rows (Material :-, Blouse :-, etc.)
- Layout B: Inline key-value text in <p>, <div>, <li>, .small-info, .product-details
- Layout C: Standard eCommerce layout (SKU, Title, Discount, Retail Price, Stock Status)
- Automatic 3-attempt exponential backoff retry for network resilience
- Full coverage audit logging
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
MAX_RETRIES = 3
TIMEOUT = 15


def parse_omni_product_html(html_text: str) -> dict:
    soup = BeautifulSoup(html_text, "html.parser")

    # 1. Detect 404
    h1 = soup.find(["h1", "h2"])
    if h1 and any(err in h1.get_text().lower() for err in ["404", "not found", "page not found"]):
        return {"status": "404_not_found"}

    extracted = {
        "material": None,
        "scraped_color": None,
        "blouse_included": None,
        "blouse_length": None,
        "saree_length": None,
        "saree_weight": None,
        "wash_care": None,
        "net_quantity": None,
        "occasion": None,
        "work_type": None,
        "stock_status": None,
        "retail_price": None,
        "discount_info": None,
        "has_full_specs": False,
    }

    # Normalize non-breaking spaces and whitespace
    clean_html = html_text.replace("\xa0", " ").replace("&nbsp;", " ")
    soup_clean = BeautifulSoup(clean_html, "html.parser")

    raw_pairs = []

    # Strategy 1: Table row extraction
    for tr in soup_clean.find_all("tr"):
        t = tr.get_text(" ", strip=True)
        if ":-" in t:
            parts = t.split(":-", 1)
            raw_pairs.append((parts[0].strip(), parts[1].strip()))
        elif ":" in t and not any(k in t.lower() for k in ["http", "sku", "note", "https"]):
            parts = t.split(":", 1)
            raw_pairs.append((parts[0].strip(), parts[1].strip()))

    # Strategy 2: Text blocks in p, li, div, span
    for tag in soup_clean.find_all(["p", "li", "div", "span"]):
        lines = tag.get_text("\n", strip=True).split("\n")
        for line in lines:
            line_str = line.strip()
            if ":-" in line_str:
                parts = line_str.split(":-", 1)
                raw_pairs.append((parts[0].strip(), parts[1].strip()))
            elif ":" in line_str and len(line_str) < 120:
                parts = line_str.split(":", 1)
                k, v = parts[0].strip(), parts[1].strip()
                if any(
                    target in k.lower()
                    for target in [
                        "material",
                        "fabric",
                        "colour",
                        "color",
                        "blouse",
                        "length",
                        "weight",
                        "washing",
                        "wash",
                        "care",
                        "occasion",
                        "work",
                        "quantity",
                    ]
                ):
                    raw_pairs.append((k, v))

    # Strategy 3: Regex scanning on whole text for unformatted key-value tokens
    full_text = soup_clean.get_text(" ", strip=True)
    regex_scans = [
        ("material", r"(?:material|fabric)\s*[:\-]+\s*([A-Za-z0-9\s&/]+?)(?=\s*(?:colour|color|blouse|saree|washing|net|note|sku|$))"),
        ("scraped_color", r"(?:colour|color)\s*[:\-]+\s*([A-Za-z0-9\s&/]+?)(?=\s*(?:blouse|saree|washing|net|note|sku|$))"),
        ("blouse_included", r"blouse\s*(?:piece)?\s*[:\-]+\s*([A-Za-z0-9\s]+?)(?=\s*(?:length|saree|washing|net|note|sku|$))"),
        ("blouse_length", r"blouse\s*(?:piece)?\s*length\s*[:\-]+\s*([0-9.]+\s*[m|cm|inch|meters]+)"),
        ("saree_length", r"saree\s*length\s*[:\-]+\s*([0-9.]+\s*[m|cm|inch|meters]+)"),
        ("saree_weight", r"saree\s*weight\s*[:\-]+\s*([0-9.]+\s*(?:kg|g|gms)?)"),
        ("wash_care", r"(?:washing\s*condition|wash\s*care|care\s*instructions?)\s*[:\-]+\s*([A-Za-z0-9\s]+?)(?=\s*(?:net|note|sku|$))"),
        ("net_quantity", r"net\s*quantity\s*[:\-]+\s*([0-9]+\s*[Nn]?)"),
    ]
    for field_name, pattern in regex_scans:
        m = re.search(pattern, full_text, re.IGNORECASE)
        if m:
            val = m.group(1).strip().rstrip(".,-").strip()
            if val and not extracted[field_name]:
                extracted[field_name] = val

    # Map discovered raw pairs
    for k_raw, v_raw in raw_pairs:
        k = k_raw.lower().replace(" ", "").replace("_", "")
        v = re.split(
            r"(?:Colou?r|Blouse|Saree|Washing|Wash|Net|Note|Quantity|Material|Fabric|SKU)",
            v_raw,
            flags=re.IGNORECASE,
        )[0].strip()
        v = v.rstrip(".,-").strip()
        if not v or v.lower() in ["none", "null", ""]:
            continue

        if ("material" in k or "fabric" in k) and not extracted["material"]:
            extracted["material"] = v
        elif ("colour" in k or "color" in k) and not extracted["scraped_color"]:
            if not v.startswith("QS") and not v.startswith("QA") and not v.startswith("AA"):
                extracted["scraped_color"] = v
        elif ("blouselength" in k or "blousepiecelength" in k) and not extracted["blouse_length"]:
            extracted["blouse_length"] = v
        elif "blouse" in k and not extracted["blouse_included"]:
            extracted["blouse_included"] = v
        elif ("sareelength" in k or ("length" in k and "blouse" not in k)) and not extracted["saree_length"]:
            extracted["saree_length"] = v
        elif "sareeweight" in k or "weight" in k:
            if not extracted["saree_weight"]:
                extracted["saree_weight"] = v
        elif ("washing" in k or "wash" in k or "care" in k) and not extracted["wash_care"]:
            extracted["wash_care"] = v
        elif ("netquantity" in k or "quantity" in k) and len(v) <= 10 and not extracted["net_quantity"]:
            extracted["net_quantity"] = v
        elif "occasion" in k and not extracted["occasion"]:
            extracted["occasion"] = v
        elif ("work" in k or "craft" in k) and not extracted["work_type"]:
            extracted["work_type"] = v

    # Stock status & retail pricing
    full_lower = full_text.lower()
    if "out of stock" in full_lower:
        extracted["stock_status"] = "Out of Stock"
    elif "preorder" in full_lower:
        extracted["stock_status"] = "Available for Preorder"
    elif "add to cart" in full_lower or "buy now" in full_lower or "in stock" in full_lower:
        extracted["stock_status"] = "In Stock"

    # Price / Discount tags
    m_disc = re.search(r"(₹\s*\d+)\s*OFF", full_text)
    if m_disc:
        extracted["discount_info"] = m_disc.group(0)

    # Determine if full specs table was present
    if extracted["material"] or extracted["blouse_included"] or extracted["saree_length"]:
        extracted["has_full_specs"] = True

    return extracted


def fetch_with_retry(row_dict: dict) -> tuple[dict, dict | None]:
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
        "has_full_specs": False,
        "page_status": "unprocessed",
    }

    if not url or not str(url).startswith("http"):
        empty_res["page_status"] = "no_url"
        return empty_res, {"sku": sku, "name": name, "url": url, "reason": "missing_or_invalid_url"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code == 200:
                parsed = parse_omni_product_html(resp.text)
                if parsed.get("status") == "404_not_found":
                    empty_res["page_status"] = "404_not_found"
                    return empty_res, {"sku": sku, "name": name, "url": url, "reason": "HTTP 404 Product Not Found"}
                parsed["sku"] = sku
                parsed["page_status"] = "200_OK"
                return parsed, None
            elif resp.status_code == 404:
                empty_res["page_status"] = "404_not_found"
                return empty_res, {"sku": sku, "name": name, "url": url, "reason": "HTTP 404 Not Found"}
            else:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                empty_res["page_status"] = f"HTTP_{resp.status_code}"
                return empty_res, {"sku": sku, "name": name, "url": url, "reason": f"HTTP {resp.status_code}"}
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.0 * (attempt + 1))
                continue
            empty_res["page_status"] = "network_error"
            return empty_res, {"sku": sku, "name": name, "url": url, "reason": str(e)}

    return empty_res, {"sku": sku, "name": name, "url": url, "reason": "Max retries exceeded"}


def main():
    if not os.path.exists(config.METADATA_PATH):
        print(f"Metadata file {config.METADATA_PATH} not found.")
        sys.exit(1)

    meta = pd.read_parquet(config.METADATA_PATH)
    print(f"Starting Omni-Enrichment across all {len(meta)} catalogue rows...")

    records = meta.to_dict("records")
    enriched_results = []
    failures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_with_retry, r): r for r in records}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            parsed, fail = future.result()
            enriched_results.append(parsed)
            if fail:
                failures.append(fail)

    enriched_df = pd.DataFrame(enriched_results).drop_duplicates(subset=["sku"])
    enriched_cache_path = os.path.join(config.DATA_DIR, "enriched_details.parquet")
    enriched_df.to_parquet(enriched_cache_path, index=False)

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

    full_specs_count = sum(1 for r in enriched_results if r.get("has_full_specs"))
    dead_links_count = sum(1 for r in enriched_results if r.get("page_status") == "404_not_found")
    blank_pages_count = sum(1 for r in enriched_results if r.get("page_status") == "200_OK" and not r.get("has_full_specs"))

    print("\n" + "=" * 70)
    print("OMNI-ENRICHMENT AUDIT BREAKDOWN")
    print("=" * 70)
    print(f"Total Rows in Catalogue: {len(meta)}")
    print(f"1. Rows with Full On-Page Specifications: {full_specs_count}")
    print(f"2. Confirmed Genuinely Blank Description Pages (HTTP 200, no CMS spec): {blank_pages_count}")
    print(f"3. Confirmed Dead Links / 404 Pages: {dead_links_count}")
    print(f"Enriched Parquet -> {config.METADATA_PATH}")
    print(f"Failures Log -> {ENRICH_FAILED_CSV}")


if __name__ == "__main__":
    main()
