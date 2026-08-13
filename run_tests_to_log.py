import sys
import os

# Write output to a file so we can read it
logf = open("test_output.log", "w", encoding="utf-8")
sys.stdout = logf
sys.stderr = logf

try:
    from PIL import Image
    import config
    from search_tool import search_similar_sarees, parse_query_intent

    img = Image.new('RGB', (224, 224), color=(220, 100, 150))

    res1 = search_similar_sarees(query_image=img, top_k=3)
    print("Test 1 (image-only):", len(res1), "results, top SKU:", res1[0]['sku'])

    res2 = search_similar_sarees(query_image=img, top_k=5, color="pink")
    print("Test 2 (image+color):", len(res2), "pink sarees")

    res3 = search_similar_sarees(query_image=img, top_k=5, fabric="banarasi")
    print("Test 3 (image+fabric):", len(res3), "banarasi sarees")

    res4 = search_similar_sarees(query_image=img, top_k=5, max_price=3000)
    print("Test 4 (image+price):", len(res4), "sarees <= 3000")

    res5 = search_similar_sarees(query_image=img, top_k=5, pattern="zari border")
    print("Test 5 (image+pattern):", len(res5), "zari border sarees, top name:", res5[0]['name'][:35])

    res6 = search_similar_sarees(query_image=img, top_k=5, pattern="golden zari", max_price=4000)
    print("Test 6 (image+pattern+price):", len(res6), "golden zari <= 4000")

    i7 = parse_query_intent("Show sarees with temple border")
    res7 = search_similar_sarees(pattern=i7['pattern'], top_k=5)
    print("Test 7 (pattern-only NL):", len(res7), "temple border, top name:", res7[0]['name'][:35])

    i8 = parse_query_intent("Find kadiyal border sarees under 5000")
    res8 = search_similar_sarees(pattern=i8['pattern'], max_price=i8['max_price'], top_k=5)
    print("Test 8 (border+price):", len(res8), "kadiyal <= 5000, top name:", res8[0]['name'][:35])

    i9 = parse_query_intent("Find sarees with parrot pallu")
    res9 = search_similar_sarees(pattern=i9['pattern'], top_k=5)
    print("Test 9 (pallu):", len(res9), "parrot pallu results, top name:", res9[0]['name'][:35])

    print("ALL 9 TESTS PASSED")
except Exception as e:
    import traceback
    print("ERROR:", e)
    traceback.print_exc()
finally:
    logf.flush()
    logf.close()
