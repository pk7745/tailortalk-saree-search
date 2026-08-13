"""
Migration Script: Populates Qdrant Vector Database from existing FAISS vectors and Parquet metadata.
Idempotent and deterministic.
"""
import os
import faiss
import numpy as np
import pandas as pd
import config
import qdrant_store

def main():
    print("=" * 80)
    print("MIGRATING TAILORTALK CATALOGUE TO QDRANT VECTOR DATABASE")
    print("=" * 80)

    # 1. Connect to Qdrant
    client = qdrant_store.get_qdrant_client()
    if not qdrant_store.health_check(client):
        print("Error: Could not connect to Qdrant.")
        return

    print(f"Target Collection: '{config.QDRANT_COLLECTION_NAME}'")

    # 2. Load FAISS Vectors and Parquet Metadata
    if not os.path.exists(config.INDEX_PATH) or not os.path.exists(config.METADATA_PATH):
        print("Error: FAISS index or metadata.parquet missing.")
        return

    faiss_idx = faiss.read_index(config.INDEX_PATH)
    meta_df = pd.read_parquet(config.METADATA_PATH)

    print(f"Loaded FAISS Vectors: {faiss_idx.ntotal} (Dimension: {faiss_idx.d})")
    print(f"Loaded Metadata Rows: {len(meta_df)} (Columns: {len(meta_df.columns)})")

    assert faiss_idx.ntotal == len(meta_df), "FAISS ntotal does not match metadata length!"

    # Reconstruct vectors from FAISS
    vectors = np.array([faiss_idx.reconstruct(i) for i in range(faiss_idx.ntotal)])

    # 3. Upsert to Qdrant
    upserted_count = qdrant_store.upsert_sarees(
        client=client,
        vectors=vectors,
        meta_df=meta_df,
        collection_name=config.QDRANT_COLLECTION_NAME,
        batch_size=100,
    )

    # 4. Verify Collection Info
    info = qdrant_store.get_collection_info(client, config.QDRANT_COLLECTION_NAME)
    print("\n" + "=" * 80)
    print("QDRANT MIGRATION SUMMARY:")
    print(f"Points Upserted:    {upserted_count}")
    print(f"Collection Status:  {info.get('status')}")
    print(f"Collection Vectors: {info.get('vectors_count')}")
    print(f"Collection Points:  {info.get('points_count')}")
    print("=" * 80)
    print("SUCCESS: Qdrant migration completed successfully!")

if __name__ == "__main__":
    main()
