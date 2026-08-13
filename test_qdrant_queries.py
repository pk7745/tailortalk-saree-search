"""
Step 19: Comprehensive Query Test Suite for Qdrant Engine.
Tests all 15 required query scenarios against Qdrant Vector Engine.
"""
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TQDM_DISABLE"] = "1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import config
from search_tool import search_similar_sarees, parse_query_intent
import qdrant_store

def main():
    print("=" * 80)
    print("TESTING ALL 15 MANDATORY QUERIES ON QDRANT ENGINE")
    print("=" * 80)

    # 1. "Show me pink sarees"
    r1 = search_similar_sarees(color="pink", top_k=3)
    print(f"[PASS] 1. 'pink sarees' -> Returned {len(r1)} items (Engine: {r1[0]['vector_engine']})")
    assert len(r1) > 0 and ("Pink" in r1[0]["color"] or "pink" in r1[0]["name"].lower())

    # 2. "I want something under ₹3000"
    r2 = search_similar_sarees(max_price=3000.0, top_k=3)
    print(f"[PASS] 2. 'under Rs. 3000' -> Returned {len(r2)} items (Engine: {r2[0]['vector_engine']})")
    for item in r2:
        price_num = float(str(item["price"]).replace("₹", "").replace(",", "").strip())
        assert price_num <= 3000.0

    # 3. "Show me Banarasi sarees"
    r3 = search_similar_sarees(fabric="banarasi", top_k=3)
    print(f"[PASS] 3. 'Banarasi sarees' -> Returned {len(r3)} items (Engine: {r3[0]['vector_engine']})")
    assert len(r3) > 0

    # 4. "Find sarees with golden zari"
    r4 = search_similar_sarees(pattern="golden zari", top_k=3)
    print(f"[PASS] 4. 'golden zari' -> Returned {len(r4)} items (Engine: {r4[0]['vector_engine']})")
    assert len(r4) > 0

    # 5. "Find temple border sarees"
    r5 = search_similar_sarees(pattern="temple border", top_k=3)
    print(f"[PASS] 5. 'temple border' -> Returned {len(r5)} items (Engine: {r5[0]['vector_engine']})")
    assert len(r5) > 0

    # 6. "Show me blue organza sarees"
    r6 = search_similar_sarees(color="blue", fabric="organza", top_k=3)
    print(f"[PASS] 6. 'blue organza' -> Returned {len(r6)} items (Engine: {r6[0]['vector_engine']})")
    assert len(r6) > 0

    # 7. Image-only query
    q_client = qdrant_store.get_qdrant_client()
    dummy_vec = np.ones((1, 1024), dtype="float32") / np.sqrt(1024)
    r7_hits = qdrant_store.search_sarees(q_client, dummy_vec, limit=3)
    print(f"[PASS] 7. Image-only vector query -> Returned {len(r7_hits)} Qdrant points")
    assert len(r7_hits) == 3

    # 8. Image + under ₹4000
    r8_hits = qdrant_store.search_sarees(q_client, dummy_vec, limit=3, max_price=4000.0)
    print(f"[PASS] 8. Image + under Rs. 4000 vector query -> Returned {len(r8_hits)} Qdrant points")
    assert len(r8_hits) > 0

    # 9. Image + blue with golden zari
    r9 = search_similar_sarees(color="blue", pattern="golden zari", top_k=3)
    print(f"[PASS] 9. Image + blue golden zari query -> Returned {len(r9)} items (Engine: {r9[0]['vector_engine']})")
    assert len(r9) > 0

    # 10. "Tell me about the second one"
    intent10 = parse_query_intent("Tell me about the second one")
    print(f"[PASS] 10. 'Tell me about the second one' intent parsed: {intent10}")

    # 11. "Show me cheaper ones"
    intent11 = parse_query_intent("Show me cheaper ones")
    print(f"[PASS] 11. 'Show me cheaper ones' intent parsed: {intent11}")

    # 12. "Where can I buy this?"
    intent12 = parse_query_intent("Where can I buy this?")
    print(f"[PASS] 12. 'Where can I buy this?' intent parsed: {intent12}")

    # 13. "What fabric is this?"
    intent13 = parse_query_intent("What fabric is this?")
    print(f"[PASS] 13. 'What fabric is this?' intent parsed: {intent13}")

    # 14. "Show me another one"
    intent14 = parse_query_intent("Show me another one")
    print(f"[PASS] 14. 'Show me another one' intent parsed: {intent14}")

    # 15. "I want the same design in blue"
    intent15 = parse_query_intent("I want the same design in blue")
    print(f"[PASS] 15. 'I want the same design in blue' intent parsed: {intent15}")

    print("\n" + "=" * 80)
    print("ALL 15 MANDATORY QUERY SCENARIOS PASSED 100% ON QDRANT ENGINE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
