"""
Comprehensive Verification Script for Duplicate Results Fix.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TQDM_DISABLE"] = "1"

import config
import qdrant_store
from search_tool import search_similar_sarees, parse_query_intent

def _get_product_id(r: dict) -> str:
    return r.get("product_link") or r.get("image_url") or r.get("sku") or r.get("name", "")

def verify_uniqueness(results, query_name):
    pids = [_get_product_id(r) for r in results]
    unique_pids = set(pids)
    print(f"   Query '{query_name}': Returned {len(results)} items | Unique Product IDs: {len(unique_pids)}/{len(pids)}")
    assert len(results) == len(unique_pids), f"Duplicates found in query '{query_name}'!"
    for item in results:
        print(f"      - ID: {item.get('product_link')[-25:]} | SKU: {item.get('sku')} | Name: {item.get('name')[:35]} | Score: {item.get('final_score')}")

def main():
    print("=" * 80)
    print("VERIFYING DUPLICATE SEARCH RESULTS FIX")
    print("=" * 80)

    # Test A: "show similar sarees under 5000 rupees"
    print("\n--- Test A: 'show similar sarees under 5000 rupees' ---")
    intent_a = parse_query_intent("show similar sarees under 5000 rupees")
    res_a = search_similar_sarees(
        max_price=intent_a["max_price"],
        top_k=intent_a["top_k"]
    )
    verify_uniqueness(res_a, "under 5000 rupees")

    # Test B: "show red sarees"
    print("\n--- Test B: 'show red sarees' ---")
    intent_b = parse_query_intent("show red sarees")
    res_b = search_similar_sarees(
        color=intent_b["color"],
        top_k=intent_b["top_k"]
    )
    verify_uniqueness(res_b, "red sarees")

    # Test C: "show silk sarees under 5000"
    print("\n--- Test C: 'show silk sarees under 5000' ---")
    intent_c = parse_query_intent("show silk sarees under 5000")
    res_c = search_similar_sarees(
        fabric=intent_c["fabric"],
        max_price=intent_c["max_price"],
        top_k=intent_c["top_k"]
    )
    verify_uniqueness(res_c, "silk sarees under 5000")

    # Test E: Verify Qdrant still has exactly 1,070 points
    print("\n--- Test E: Qdrant Collection Point Count ---")
    client = qdrant_store.get_qdrant_client()
    info = qdrant_store.get_collection_info(client, config.QDRANT_COLLECTION_NAME)
    print(f"   Qdrant collection '{config.QDRANT_COLLECTION_NAME}' points count: {info.get('points_count')}")
    assert info.get("points_count") == 1070, f"Expected 1070 points in Qdrant, got {info.get('points_count')}"

    print("\n" + "=" * 80)
    print("ALL DUPLICATE FIX VERIFICATION AUDITS PASSED 100%!")
    print("=" * 80)

if __name__ == "__main__":
    main()
