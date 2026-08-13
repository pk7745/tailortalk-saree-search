"""
Inspect product webpages for the 4 missing SKUs to find alternative valid image URLs.
SKUs: QS264566, QS270932, QS282590, QS282741
"""
import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
import config

MISSING_SKUS = ["QS264566", "QS270932", "QS282590", "QS282741"]
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def check_url_working(url):
    try:
        r = requests.head(url, timeout=5, headers=HEADERS)
        if r.status_code == 200:
            return True
        r = requests.get(url, timeout=5, headers=HEADERS, stream=True)
        return r.status_code == 200
    except Exception:
        return False


def main():
    csv_df = pd.read_csv(config.PRODUCTS_CSV)

    print("=" * 80)
    print("INSPECTING PRODUCT WEBPAGES FOR 4 MISSING SKUS")
    print("=" * 80)

    recovered = {}

    for sku in MISSING_SKUS:
        rows = csv_df[csv_df["SKU"] == sku]
        print(f"\n--- SKU: {sku} ---")
        if len(rows) == 0:
            print("  Not in CSV!")
            continue

        for _, r in rows.iterrows():
            name = r["Name"]
            page_url = r["Website Link"]
            csv_img_url = r["image_url"]

            print(f"  Name:     {name}")
            print(f"  Page URL: {page_url}")
            print(f"  CSV Img:  {csv_img_url}")

            try:
                resp = requests.get(page_url, timeout=15, headers=HEADERS)
                print(f"  Page HTTP Status: {resp.status_code}")
                if resp.status_code != 200:
                    print("  Page unavailable.")
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")
                candidates = []

                # 1. OpenGraph meta
                for og in soup.find_all("meta", property="og:image"):
                    content = og.get("content")
                    if content:
                        candidates.append(("og:image", content))

                for og in soup.find_all("meta", attrs={"name": "og:image"}):
                    content = og.get("content")
                    if content:
                        candidates.append(("meta og:image", content))

                # 2. JSON-LD structured data
                for script in soup.find_all("script", type="application/ld+json"):
                    if script.string:
                        matches = re.findall(
                            r"https?://[^\s\"']+\.(?:jpg|png|webp|jpeg)",
                            script.string,
                            re.IGNORECASE,
                        )
                        for m in matches:
                            candidates.append(("json-ld", m))

                # 3. img tags, data-src, srcset
                for img in soup.find_all("img"):
                    src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
                    if src and ("storage" in src or "upload" in src or "product" in src or "media" in src or "byrappa" in src):
                        if not src.startswith("http"):
                            if src.startswith("//"):
                                src = "https:" + src
                            elif src.startswith("/"):
                                src = "https://byrappasilks.in" + src
                        candidates.append(("img tag", src))

                print(f"  Found {len(candidates)} candidate image URLs:")
                valid_alt = None
                for src_type, cand_url in candidates:
                    is_ok = check_url_working(cand_url)
                    status_str = "200 OK (VALID!)" if is_ok else "FAILED/404"
                    print(f"    - [{src_type}] {cand_url[:80]}... --> {status_str}")
                    if is_ok and valid_alt is None and cand_url != csv_img_url:
                        valid_alt = cand_url

                if valid_alt:
                    print(f"  ✅ RECOVERED ALTERNATIVE VALID IMAGE: {valid_alt}")
                    recovered[sku] = {
                        "name": name,
                        "sku": sku,
                        "page_url": page_url,
                        "csv_img_url": csv_img_url,
                        "recovered_img_url": valid_alt,
                    }
                else:
                    print("  ❌ No alternative valid image found on product page.")

            except Exception as e:
                print(f"  Error inspecting page: {e}")

    print("\n" + "=" * 80)
    print(f"SUMMARY OF RECOVERY INVESTIGATION:")
    print(f"Total Missing SKUs Investigated: {len(MISSING_SKUS)}")
    print(f"Recovered Missing Images:        {len(recovered)}")
    print("=" * 80)
    for k, v in recovered.items():
        print(f"  - {k}: {v['recovered_img_url']}")


if __name__ == "__main__":
    main()
