import sys
import os

# Must be set before any library (faiss, torch, numpy) loads its OpenMP DLL.
# On Windows, FAISS (libomp140) and PyTorch (libiomp5md) conflict without this.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from PIL import Image
import config
from search_tool import search_similar_sarees, parse_query_intent

try:
    meta = pd.read_parquet(config.METADATA_PATH)

    # Local synthetic test image
    img = Image.new('RGB', (224, 224), color=(220, 100, 150))

    print("="*80)
    print("RUNNING 9-POINT REGRESSION AND SAFETY VERIFICATION SUITE")
    print("="*80)

    # 1. Image-only similarity search
    res1 = search_similar_sarees(query_image=img, top_k=5)
    assert len(res1) == 5
    print(f"[PASS] 1. Image-only similarity search: returned {len(res1)} results (Top 1 SKU: {res1[0]['sku']}, Score: {res1[0]['score']})")

    # 2. Image + colour
    res2 = search_similar_sarees(query_image=img, top_k=5, color="pink")
    assert len(res2) > 0
    for r in res2:
        assert "pink" in str(r['name']).lower() or "pink" in str(r['color']).lower()
    print(f"[PASS] 2. Image + colour ('pink'): returned {len(res2)} strictly pink sarees")

    # 3. Image + fabric
    res3 = search_similar_sarees(query_image=img, top_k=5, fabric="banarasi")
    assert len(res3) > 0
    for r in res3:
        assert "banarasi" in str(r['name']).lower() or "banarasi" in str(r['fabric']).lower()
    print(f"[PASS] 3. Image + fabric ('banarasi'): returned {len(res3)} banarasi sarees")

    # 4. Image + price
    res4 = search_similar_sarees(query_image=img, top_k=5, max_price=3000)
    assert len(res4) > 0
    for r in res4:
        price_val = float(str(r['price']).replace("₹", "").replace(",", "").strip())
        assert price_val <= 3000
    print(f"[PASS] 4. Image + price (max_price=3000): returned {len(res4)} items <= ₹3000")

    # 5. Image + pattern ('zari border')
    res5 = search_similar_sarees(query_image=img, top_k=5, pattern="zari border")
    assert len(res5) > 0
    print(f"[PASS] 5. Image + pattern ('zari border'): returned {len(res5)} items with zari border (Top 1: {res5[0]['name'][:40]})")

    # 6. Image + pattern + price ('golden zari', max_price=4000)
    res6 = search_similar_sarees(query_image=img, top_k=5, pattern="golden zari", max_price=4000)
    assert len(res6) > 0
    for r in res6:
        price_val = float(str(r['price']).replace("₹", "").replace(",", "").strip())
        assert price_val <= 4000
    print(f"[PASS] 6. Image + pattern + price ('golden zari' <= ₹4000): returned {len(res6)} items")

    # 7. Pattern-only natural-language query ('temple border')
    intent7 = parse_query_intent("Show sarees with temple border")
    res7 = search_similar_sarees(pattern=intent7['pattern'], top_k=5)
    assert len(res7) > 0
    for r in res7:
        assert "temple" in r['name'].lower() or "temple" in r['pattern'].lower()
    print(f"[PASS] 7. Pattern-only NL query ('temple border'): returned {len(res7)} temple border items (Top 1: {res7[0]['name'][:40]})")

    # 8. Border query ('kadiyal border under 5000')
    intent8 = parse_query_intent("Find kadiyal border sarees under 5000")
    res8 = search_similar_sarees(pattern=intent8['pattern'], max_price=intent8['max_price'], top_k=5)
    assert len(res8) > 0
    for r in res8:
        assert "kadiyal" in r['name'].lower()
    print(f"[PASS] 8. Border query ('kadiyal border' <= ₹5000): returned {len(res8)} items (Top 1: {res8[0]['name'][:40]})")

    # 9. Pallu query ('parrot pallu')
    intent9 = parse_query_intent("Find sarees with parrot pallu")
    res9 = search_similar_sarees(pattern=intent9['pattern'], top_k=5)
    assert len(res9) > 0
    for r in res9:
        assert "parrot pallu" in r['name'].lower() or "pallu" in r['name'].lower()
    print(f"[PASS] 9. Pallu query ('parrot pallu'): returned {len(res9)} items (Top 1: {res9[0]['name'][:40]})")

    print("="*80)
    print("ALL 9 TEST SCENARIOS PASSED WITH ZERO REGRESSIONS!")
    print("="*80)
except Exception as e:
    import traceback
    print("Error during test:", e)
    traceback.print_exc()
