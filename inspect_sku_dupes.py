import pandas as pd
import faiss
import config

def main():
    csv_df = pd.read_csv(config.PRODUCTS_CSV)
    meta = pd.read_parquet(config.METADATA_PATH)
    idx = faiss.read_index(config.INDEX_PATH)

    print("=" * 80)
    print("1. DATASET & IDENTIFIER OVERVIEW")
    print("=" * 80)
    print(f"CSV Total Rows:                {len(csv_df)}")
    print(f"CSV Unique SKUs:               {csv_df['SKU'].nunique()}")
    print(f"Metadata Total Rows:           {len(meta)}")
    print(f"Metadata Unique SKUs:          {meta['sku'].nunique()}")
    print(f"Metadata Unique Image URLs:    {meta['image_url'].nunique()}")
    print(f"Metadata Unique Product Links: {meta['product_link'].nunique()}")
    print(f"FAISS Vector Count:            {idx.ntotal}")

    print("\n" + "=" * 80)
    print("2. FAISS INDEX POSITION TO METADATA ROW MAPPING VERIFICATION")
    print("=" * 80)
    print(f"FAISS ntotal ({idx.ntotal}) == Metadata length ({len(meta)})? {idx.ntotal == len(meta)}")
    for row_idx in [0, 50, 100, 500, 1000]:
        vec = idx.reconstruct(row_idx)
        m_row = meta.iloc[row_idx]
        print(f"  Index position {row_idx:4d} -> SKU: {m_row['sku']:10s} | Name: {m_row['name'][:50]}")

    print("\n" + "=" * 80)
    print("3. DEEP INSPECTION OF SKU: QS239312")
    print("=" * 80)
    rows_239312 = meta[meta['sku'] == 'QS239312']
    print(f"Found {len(rows_239312)} distinct product listings sharing SKU 'QS239312':")
    for idx_pos, (_, r) in enumerate(rows_239312.iterrows(), 1):
        print(f"\n  Listing #{idx_pos}:")
        print(f"    Name:        {r['name']}")
        print(f"    Price:       Rs. {r['price']}")
        print(f"    Image URL:   {r['image_url']}")
        print(f"    Product Link:{r['product_link']}")

    print("\n" + "=" * 80)
    print("4. DEEP INSPECTION OF SKU: QS280547")
    print("=" * 80)
    rows_280547 = meta[meta['sku'] == 'QS280547']
    print(f"Found {len(rows_280547)} distinct product listings sharing SKU 'QS280547':")
    for idx_pos, (_, r) in enumerate(rows_280547.iterrows(), 1):
        print(f"\n  Listing #{idx_pos}:")
        print(f"    Name:        {r['name']}")
        print(f"    Price:       Rs. {r['price']}")
        print(f"    Image URL:   {r['image_url']}")
        print(f"    Product Link:{r['product_link']}")

    print("\n" + "=" * 80)
    print("5. TOP 5 MOST COMMON SHARED SKUs IN CATALOGUE")
    print("=" * 80)
    top_shared = meta['sku'].value_counts().head(5)
    for sku, count in top_shared.items():
        print(f"  SKU '{sku}': shared across {count} distinct product rows/variants")

if __name__ == '__main__':
    main()
