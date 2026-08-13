import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

logf = open("eval_output.log", "w", encoding="utf-8")
sys.stdout = logf
sys.stderr = logf

try:
    import io
    import requests
    import pandas as pd
    import numpy as np
    import faiss
    from PIL import Image
    import config
    from search_tool import search_similar_sarees

    meta = pd.read_parquet(config.METADATA_PATH)
    SAMPLE_INDICES = [0, 5, 12, 25, 42]
    HEADERS = {"User-Agent": "Mozilla/5.0"}

    def download_img(url):
        try:
            r = requests.get(url, timeout=10, headers=HEADERS)
            if r.status_code == 200:
                return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
        return None

    print("=" * 90)
    print("FINAL END-TO-END QUALITY EVALUATION WITH 5 REAL CATALOGUE IMAGES")
    print("=" * 90)

    for s_idx in SAMPLE_INDICES:
        row = meta.iloc[s_idx]
        name = row["name"]
        sku = row["sku"]
        img_url = row["image_url"]
        c_color = row.get("color", "")
        c_fabric = row.get("fabric", "")

        print("\n" + "-" * 90)
        print(f"QUERY SAREE #{s_idx+1}: {name}")
        print(f"SKU: {sku} | Color: {c_color} | Fabric: {c_fabric}")

        img = download_img(img_url)
        if img is None:
            print("Could not download real image, skipping...")
            continue

        # Test 1: Image Only
        r1 = search_similar_sarees(query_image=img, top_k=5)
        print("\n  [1. Image-Only Similarity]")
        for i, item in enumerate(r1, 1):
            print(f"     #{i} [{item['sku']}] {item['name'][:55]} | Final: {item['final_score']} (Vis: {item['score']})")

        # Test 2: Image + Golden Zari Border
        r2 = search_similar_sarees(query_image=img, pattern="golden zari", top_k=5)
        print("\n  [2. Image + Golden Zari Border]")
        for i, item in enumerate(r2, 1):
            print(f"     #{i} [{item['sku']}] {item['name'][:55]} | Final: {item['final_score']} (Vis: {item['score']}, Attr: {item['attribute_score']})")

        # Test 3: Image + Color
        target_color = "pink" if "pink" in name.lower() or "pink" in c_color.lower() else "blue"
        r3 = search_similar_sarees(query_image=img, color=target_color, top_k=5)
        print(f"\n  [3. Image + Colour '{target_color}']")
        for i, item in enumerate(r3, 1):
            print(f"     #{i} [{item['sku']}] {item['name'][:55]} | Final: {item['final_score']} (Vis: {item['score']}, Attr: {item['attribute_score']})")

        # Test 4: Image + Fabric
        target_fabric = "banarasi" if "banaras" in name.lower() or "banarasi" in c_fabric.lower() else "organza"
        r4 = search_similar_sarees(query_image=img, fabric=target_fabric, top_k=5)
        print(f"\n  [4. Image + Fabric '{target_fabric}']")
        for i, item in enumerate(r4, 1):
            print(f"     #{i} [{item['sku']}] {item['name'][:55]} | Final: {item['final_score']} (Vis: {item['score']}, Attr: {item['attribute_score']})")

        # Test 5: Image + Pattern / Pallu
        r5 = search_similar_sarees(query_image=img, pattern="temple border", top_k=5)
        print("\n  [5. Image + Pattern 'temple border']")
        for i, item in enumerate(r5, 1):
            print(f"     #{i} [{item['sku']}] {item['name'][:55]} | Final: {item['final_score']} (Vis: {item['score']}, Attr: {item['attribute_score']})")

    print("\nEVALUATION COMPLETE!")
except Exception as e:
    import traceback
    print("ERROR:", e)
    traceback.print_exc()
finally:
    logf.flush()
    logf.close()
