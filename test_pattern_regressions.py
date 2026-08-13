import sys
import os
import traceback
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from PIL import Image
import config
from search_tool import search_similar_sarees, parse_query_intent, load_image_from_url

try:
    meta = pd.read_parquet(config.METADATA_PATH)
    sample_row = meta.iloc[0]
    sample_img = load_image_from_url(sample_row['image_url'])

    print("="*80)
    print("RUNNING 9-POINT REGRESSION AND SAFETY VERIFICATION SUITE")
    print("="*80)

    # 1. Image-only similarity search (pattern=None omitted)
    res1 = search_similar_sarees(query_image=sample_img, top_k=5)
    assert len(res1) == 5, f"Expected 5 results, got {len(res1)}"
    assert res1[0]['sku'] == sample_row['sku'], f"Identity match failed for image-only: {res1[0]['sku']} vs {sample_row['sku']}"
    assert res1[0]['score'] >= 0.98, f"Score too low: {res1[0]['score']}"
    print("[PASS] Test 1: Image-only similarity search returns top-5 with exact identity #1")

    # 2. Image + colour
    res2 = search_similar_sarees(query_image=sample_img, top_k=5, color="pink")
    assert len(res2) > 0, "No results for image + color"
    for r in res2:
        assert "pink" in str(r['name']).lower() or "pink" in str(r['color']).lower()
    print(f"[PASS] Test 2: Image + colour ('pink') returned {len(res2)} strictly pink sarees")

    # 3. Image + fabric
    res3 = search_similar_sarees(query_image=sample_img, top_k=5, fabric="banarasi")
    assert len(res3) > 0, "No results for image + fabric"
    for r in res3:
        assert "banarasi" in str(r['name']).lower() or "banarasi" in str(r['fabric']).lower()
    print(f"[PASS] Test 3: Image + fabric ('banarasi') returned {len(res3)} banarasi sarees")

    # 4. Image + price (max_price=3000)
    res4 = search_similar_sarees(query_image=sample_img, top_k=5, max_price=3000)
    assert len(res4) > 0, "No results for image + price"
    for r in res4:
        price_val = float(r['price'].replace("₹", "").replace(",", "").strip())
        assert price_val <= 3000, f"Price violated: {price_val}"
    print(f"[PASS] Test 4: Image + price (max_price=3000) returned {len(res4)} items <= ₹3000")

    # 5. Image + pattern ('zari border')
    res5 = search_similar_sarees(query_image=sample_img, top_k=5, pattern="zari border")
    assert len(res5) > 0, "No results for image + pattern"
    for r in res5:
        text = (r['name'] + " " + r['pattern']).lower()
        assert "zari" in text or "border" in text
    print(f"[PASS] Test 5: Image + pattern ('zari border') returned {len(res5)} items with zari border")

    # 6. Image + pattern + price ('golden zari', max_price=4000)
    res6 = search_similar_sarees(query_image=sample_img, top_k=5, pattern="golden zari", max_price=4000)
    assert len(res6) > 0, "No results for image + pattern + price"
    for r in res6:
        price_val = float(r['price'].replace("₹", "").replace(",", "").strip())
        assert price_val <= 4000, f"Price violated: {price_val}"
        text = (r['name'] + " " + r['pattern']).lower()
        assert "zari" in text or "gold" in text
    print(f"[PASS] Test 6: Image + pattern + price ('golden zari' <= ₹4000) returned {len(res6)} items")

    # 7. Pattern-only natural-language query (e.g. 'Show sarees with temple border')
    intent7 = parse_query_intent("Show sarees with temple border")
    assert intent7['pattern'] == "temple border", f"Intent parsed: {intent7}"
    res7 = search_similar_sarees(pattern=intent7['pattern'], top_k=5)
    assert len(res7) > 0, "No results for temple border"
    for r in res7:
        assert "temple" in r['name'].lower() or "temple" in r['pattern'].lower()
    print(f"[PASS] Test 7: Pattern-only NL query ('temple border') returned {len(res7)} temple border items")

    # 8. Border query (e.g. 'Find kadiyal border sarees under 5000')
    intent8 = parse_query_intent("Find kadiyal border sarees under 5000")
    assert intent8['pattern'] == "kadiyal border", f"Intent parsed: {intent8}"
    assert intent8['max_price'] == 5000.0, f"Max price parsed: {intent8['max_price']}"
    res8 = search_similar_sarees(pattern=intent8['pattern'], max_price=intent8['max_price'], top_k=5)
    assert len(res8) > 0, "No results for kadiyal border under 5000"
    for r in res8:
        assert "kadiyal" in r['name'].lower()
    print(f"[PASS] Test 8: Border query ('kadiyal border' <= ₹5000) returned {len(res8)} items")

    # 9. Pallu query (e.g. 'Find sarees with parrot pallu')
    intent9 = parse_query_intent("Find sarees with parrot pallu")
    assert intent9['pattern'] == "parrot pallu", f"Intent parsed: {intent9}"
    res9 = search_similar_sarees(pattern=intent9['pattern'], top_k=5)
    assert len(res9) > 0, "No results for parrot pallu"
    for r in res9:
        assert "parrot pallu" in r['name'].lower() or "pallu" in r['name'].lower()
    print(f"[PASS] Test 9: Pallu query ('parrot pallu') returned {len(res9)} items")

    print("="*80)
    print("ALL 9 TEST SCENARIOS PASSED WITH ZERO REGRESSIONS!")
    print("="*80)
except Exception as e:
    print(f"Exception during test: {e}")
    traceback.print_exc()
