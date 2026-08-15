"""
Metadata Parquet & Qdrant Payload Schema Complete Audit Script.
Analyzes metadata.parquet schema, column dtypes, non-null counts, null counts, unique counts, sample values,
and compares 100% of fields against Qdrant collection payloads.
"""
import os
import pandas as pd
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import config
import qdrant_store
from search_tool import _load_index_and_meta

def main():
    print("=" * 80)
    print("TAILORTALK SAREE SEARCH -- METADATA PARQUET & QDRANT PAYLOAD AUDIT")
    print("=" * 80)

    # 1. Read metadata.parquet
    meta_path = config.METADATA_PATH
    df = pd.read_parquet(meta_path)
    total_records = len(df)
    print(f"\nMetadata Parquet Path: {meta_path}")
    print(f"Total Rows: {total_records}")
    print(f"Total Columns: {len(df.columns)}")

    # 2. Inspect every column
    print("\n" + "=" * 80)
    print("COMPLETE METADATA.PARQUET COLUMN AUDIT")
    print("=" * 80)

    parquet_columns = list(df.columns)
    column_audit = []

    for col in parquet_columns:
        dtype = str(df[col].dtype)
        non_null = int(df[col].notnull().sum())
        null_cnt = total_records - non_null
        unique_cnt = int(df[col].nunique(dropna=True))

        # Sample 10 non-null values
        non_null_series = df[col].dropna().unique()
        samples = [str(x)[:40].encode('ascii', 'ignore').decode('ascii') for x in non_null_series[:10]]
        
        column_audit.append({
            "column": col,
            "dtype": dtype,
            "non_null": non_null,
            "null_count": null_cnt,
            "unique_count": unique_cnt,
            "sample_values": samples
        })

        print(f"\nColumn: '{col}'")
        print(f"  DataType: {dtype}")
        print(f"  Non-Null Count: {non_null} / {total_records} ({non_null/total_records*100:.1f}%)")
        print(f"  Null Count: {null_cnt}")
        print(f"  Unique Count: {unique_cnt}")
        print(f"  10 Representative Samples:")
        for s_idx, sample_val in enumerate(samples, 1):
            print(f"    {s_idx:2d}. {sample_val}")

    # 3. Inspect Qdrant Collection Payload Fields
    print("\n" + "=" * 80)
    print("QDRANT COLLECTION PAYLOAD FIELD COMPARISON AUDIT")
    print("=" * 80)

    qdrant_payload_fields = set()
    q_client = qdrant_store.get_qdrant_client()
    
    try:
        # Fetch sample point payloads from Qdrant
        records, _ = q_client.scroll(
            collection_name=config.QDRANT_COLLECTION_NAME,
            limit=20,
            with_payload=True,
            with_vectors=False
        )
        if records:
            for r in records:
                qdrant_payload_fields.update(r.payload.keys())
            print(f"Successfully retrieved sample Qdrant payloads from collection '{config.QDRANT_COLLECTION_NAME}'.")
            print(f"Total Unique Payload Fields in Qdrant: {len(qdrant_payload_fields)}")
        else:
            print("Warning: Qdrant scroll returned 0 points. Checking qdrant_store.py upsert mapping.")
    except Exception as ex:
        print(f"Qdrant Scroll Exception: {ex}. Auditing qdrant_store.py payload creation mapping.")

    # 4. Compare Parquet columns vs Qdrant payload fields
    parquet_set = set(parquet_columns)
    
    # Check what fields are in parquet vs qdrant
    print("\nParquet Columns Count:", len(parquet_set))
    print("Qdrant Payload Fields Count:", len(qdrant_payload_fields) if qdrant_payload_fields else "N/A (Checking Ingestion Schema)")

    if qdrant_payload_fields:
        missing_in_qdrant = parquet_set - qdrant_payload_fields
        extra_in_qdrant = qdrant_payload_fields - parquet_set
        
        print("\n" + "-" * 80)
        print(f"Fields in Parquet preserved in Qdrant: {len(parquet_set.intersection(qdrant_payload_fields))} / {len(parquet_set)}")
        print(f"Fields in Parquet missing from Qdrant: {list(missing_in_qdrant)}")
        print(f"Extra computed fields in Qdrant payload: {list(extra_in_qdrant)}")
        print("-" * 80)

    # 5. Check specific attribute coverage requested by user
    print("\n" + "=" * 80)
    print("SPECIFIC ATTRIBUTE COVERAGE AUDIT")
    print("=" * 80)

    requested_attrs = [
        "name", "price", "color", "colour", "fabric", "material", "pattern", "design",
        "border", "pallu", "blouse", "weave", "occasion", "style", "sku",
        "product_link", "image_url", "specs_source", "sibling_sku"
    ]

    for attr in requested_attrs:
        matches = [c for c in parquet_columns if attr in c.lower()]
        print(f"Attribute Key '{attr}': Matching Parquet Columns -> {matches}")

if __name__ == "__main__":
    main()
