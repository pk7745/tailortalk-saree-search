"""
Phase 4 & 5: Complete 3-Source Metadata Enrichment & Fusion.
Combines:
1. CSV source metadata
2. Website scraped metadata (border, pallu, material, blouse, work)
3. OpenCV visual analysis signals
4. Product Name keyword extraction (name_border, name_pallu, name_work)

Updates data/metadata.parquet cleanly without breaking existing structure.
"""
import os
import re
import pandas as pd
import config

BORDER_PATTERNS = [
    'golden zari border', 'gold zari border', 'silver zari border', 'zari border',
    'golden zari', 'gold zari', 'silver zari', 'temple border', 'kadiyal border',
    'contrast border', 'rising border', 'kanchi border', 'broad border', 'border'
]

PALLU_PATTERNS = [
    'parrot pallu', 'zari pallu', 'floral pallu', 'contrast pallu', 'pallu'
]

WORK_PATTERNS = [
    'zari work', 'applique work', 'aplic work', 'applique', 'chikankari',
    'embroidery', 'embroidered', 'kutch work', 'mirror work', 'traditional art',
    'butti', 'all-over', 'madhubani print', 'ajrakh print', 'lotus print',
    'floral print', 'geometric zari', 'checks', 'stripes', 'brocade', 'ikkat', 'jacquard'
]


def extract_name_signals(name: str) -> tuple[str, str, str]:
    """Extract border, pallu, and work_type keywords from product name string."""
    t = str(name).lower()

    b_found = next((b for b in BORDER_PATTERNS if re.search(r'\b' + re.escape(b) + r'\b', t)), '')
    p_found = next((p for p in PALLU_PATTERNS if re.search(r'\b' + re.escape(p) + r'\b', t)), '')
    w_found = next((w for w in WORK_PATTERNS if re.search(r'\b' + re.escape(w) + r'\b', t)), '')

    return b_found, p_found, w_found


def main():
    meta_path = config.METADATA_PATH
    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found.")
        return

    meta = pd.read_parquet(meta_path)
    print(f"Loaded base metadata: {len(meta)} rows")

    # 1. Extract signals from Name
    name_borders, name_pallus, name_works = [], [], []
    for _, r in meta.iterrows():
        b, p, w = extract_name_signals(r["name"])
        name_borders.append(b)
        name_pallus.append(p)
        name_works.append(w)

    meta["name_border"] = name_borders
    meta["name_pallu"] = name_pallus
    meta["name_work"] = name_works

    # 2. Merge Website Enriched Details if present
    enriched_cache = os.path.join(config.DATA_DIR, "enriched_details.parquet")
    if os.path.exists(enriched_cache):
        e_df = pd.read_parquet(enriched_cache)
        # Drop existing overlapping cols if needed
        cols_to_merge = [c for c in e_df.columns if c != "sku" and c not in meta.columns]
        if "sku" in e_df.columns:
            # Deduplicate e_df by sku
            e_df_uniq = e_df.drop_duplicates(subset=["sku"])
            meta = meta.merge(e_df_uniq[["sku"] + cols_to_merge], on="sku", how="left")
            print(f"Merged website details: added {cols_to_merge}")

    # Ensure border, pallu, work_type, border_color exist
    if "border" not in meta.columns:
        meta["border"] = meta["name_border"]
    else:
        meta["border"] = meta["border"].fillna(meta["name_border"])

    if "pallu" not in meta.columns:
        meta["pallu"] = meta["name_pallu"]
    else:
        meta["pallu"] = meta["pallu"].fillna(meta["name_pallu"])

    if "work_type" not in meta.columns or meta["work_type"].isnull().all():
        meta["work_type"] = meta["name_work"]
    else:
        meta["work_type"] = meta["work_type"].fillna(meta["name_work"])

    # 3. Merge Visual Features if present
    vis_cache = os.path.join(config.DATA_DIR, "visual_features.parquet")
    if os.path.exists(vis_cache):
        v_df = pd.read_parquet(vis_cache)
        v_cols = [c for c in v_df.columns if c not in ["sku", "image_url"] and c not in meta.columns]
        if "image_url" in v_df.columns:
            v_df_uniq = v_df.drop_duplicates(subset=["image_url"])
            meta = meta.merge(v_df_uniq[["image_url"] + v_cols], on="image_url", how="left")
            print(f"Merged visual features: added {v_cols}")

    # Set default values for missing visual fields
    for col in ["visual_border_detected", "visual_contrast_border", "visual_zari_detected", "visual_decorative_work"]:
        if col not in meta.columns:
            meta[col] = "unknown"
        else:
            meta[col] = meta[col].fillna("unknown")

    meta["visual_border"] = meta["visual_border_detected"]
    meta["visual_zari"] = meta["visual_zari_detected"]

    # Save final enriched metadata
    meta.to_parquet(meta_path, index=False)
    print(f"\nSuccessfully written enriched metadata to {meta_path}")
    print(f"Total Rows: {len(meta)}")
    print(f"Total Columns ({len(meta.columns)}): {list(meta.columns)}")
    print("\nNon-null count summary:")
    for c in meta.columns:
        nn = meta[c].notna().sum()
        print(f"  {c}: {nn}/{len(meta)}")


if __name__ == "__main__":
    main()
