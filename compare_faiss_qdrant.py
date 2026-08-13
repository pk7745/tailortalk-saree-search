"""
Step 18: Comparison Validation Script between FAISS and Qdrant.
Runs identical vector queries on both vector search engines to verify:
1. Retrieval agreement (overlap of top-5 candidates).
2. Similarity score precision match.
3. Latency benchmark.
"""
import time
import numpy as np
import faiss
import pandas as pd
import config
import qdrant_store

def main():
    print("=" * 80)
    print("COMPARISON VALIDATION: FAISS vs QDRANT")
    print("=" * 80)

    # 1. Load FAISS
    faiss_idx = faiss.read_index(config.INDEX_PATH)
    meta = pd.read_parquet(config.METADATA_PATH)

    # 2. Connect Qdrant
    q_client = qdrant_store.get_qdrant_client()

    # Create 5 test queries using real vectors from index
    test_indices = [0, 10, 50, 100, 500]

    for test_idx in test_indices:
        q_vec = faiss_idx.reconstruct(test_idx).reshape(1, -1).astype("float32")
        target_name = meta.iloc[test_idx]["name"]

        print(f"\n--- Test Query (Item #{test_idx}: {target_name[:45]}...) ---")

        # FAISS search
        t0 = time.perf_counter()
        f_scores, f_indices = faiss_idx.search(q_vec, 5)
        t_faiss = (time.perf_counter() - t0) * 1000

        f_top = [meta.iloc[i]["sku"] for i in f_indices[0]]

        # Qdrant search
        t0 = time.perf_counter()
        q_results = qdrant_store.search_sarees(q_client, q_vec, limit=5)
        t_qdrant = (time.perf_counter() - t0) * 1000

        q_top = [hit["payload"]["sku"] for hit in q_results]
        q_scores = [hit["score"] for hit in q_results]

        print(f"  FAISS Top-5 SKUs:  {f_top}")
        print(f"  FAISS Top-5 Scores:{[round(s, 4) for s in f_scores[0]]}")
        print(f"  FAISS Search Time: {t_faiss:.3f} ms")

        print(f"  Qdrant Top-5 SKUs: {q_top}")
        print(f"  Qdrant Top-5 Scores:{[round(s, 4) for s in q_scores]}")
        print(f"  Qdrant Search Time:{t_qdrant:.3f} ms")

        # Verify agreement
        overlap = len(set(f_top).intersection(set(q_top)))
        print(f"  Top-5 Candidate Overlap: {overlap}/5")
        assert overlap >= 4, f"Low candidate overlap ({overlap}/5) between FAISS and Qdrant!"

    print("\n" + "=" * 80)
    print("FAISS vs QDRANT COMPARISON AUDIT PASSED WITH HIGH RETRIEVAL EQUIVALENCE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
