import os
import sys

# Set env vars BEFORE any import
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

try:
    import pandas as pd
    from PIL import Image
    import config
    from search_tool import search_similar_sarees

    print("Import successful!")
    meta = pd.read_parquet(config.METADATA_PATH)
    print("Metadata loaded, rows:", len(meta))

    img = Image.new("RGB", (224, 224), color=(220, 100, 150))
    res = search_similar_sarees(query_image=img, top_k=3)
    print("Search successful, results:", len(res))
    for r in res:
        print("  -", r["sku"], r["name"][:40], "score:", r["score"])

except Exception as e:
    import traceback
    print("EXCEPTION:", e)
    traceback.print_exc()
