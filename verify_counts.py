"""
Verify dataset and index counts strictly (no ML model loading).
"""
import os
import pandas as pd
import faiss
import config

def main():
    csv_df = pd.read_csv(config.PRODUCTS_CSV)
    meta = pd.read_parquet(config.METADATA_PATH)
    idx = faiss.read_index(config.INDEX_PATH)
    failed_df = pd.read_csv(config.FAILED_LOG)

    print("=" * 80)
    print("DATASET & INDEX COUNT AUDIT")
    print("=" * 80)
    print(f"Total CSV Rows:                 {len(csv_df)}")
    print(f"Successfully Downloaded Images: {len(meta)}")
    print(f"Documented Image Failures:      {len(failed_df)}")
    print(f"FAISS Index Vector Count:       {idx.ntotal}")
    print(f"Metadata Row Count:             {len(meta)}")
    print(f"Unique Image URLs in Metadata:  {meta['image_url'].nunique()}")
    print(f"FAISS Dimension:                {idx.d}")

    assert len(csv_df) == 1074
    assert len(meta) == 1070
    assert idx.ntotal == 1070
    assert meta["image_url"].nunique() == 1070
    assert idx.ntotal == len(meta)

    print("\nENRICHED METADATA COLUMNS (Total 37):")
    req_cols = ["border", "pallu", "work_type", "visual_border_detected", "visual_zari_detected", "visual_contrast_border", "name_border", "name_pallu", "name_work"]
    for c in req_cols:
        assert c in meta.columns
        nn = meta[c].notna().sum()
        print(f"  - Column '{c}': non-null = {nn}/{len(meta)}")

    print("\n[SUCCESS] COUNT & ENRICHMENT AUDIT PASSED 100%!")

if __name__ == "__main__":
    main()
