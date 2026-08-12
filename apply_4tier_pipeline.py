"""
Apply the 4-Tier Data Pipeline across all 1,070 catalogue records.
"""
from __future__ import annotations

import os
import re
import faiss
import numpy as np
import pandas as pd
import config

COLOR_TERMS = [
    r'\b(?:pink|light pink|dark pink|rani pink|baby pink|fuchsia|fushia|rose|magenta|onion pink)\b',
    r'\b(?:red|deep red|crimson|ruby|maroon|wine|burgundy|rust)\b',
    r'\b(?:blue|navy blue|royal blue|sky blue|teal|cyan|indigo|peacock blue|rama blue)\b',
    r'\b(?:green|olive green|parrot green|emerald|mint|pastel green|sea green|sage green|rama green)\b',
    r'\b(?:yellow|mustard|lemon|gold|golden|amber)\b',
    r'\b(?:black|jet black|charcoal)\b',
    r'\b(?:white|off white|pure white|cream|ivory|pearl)\b',
    r'\b(?:purple|violet|lavender|lilac|mauve)\b',
    r'\b(?:orange|peach|coral|copper|tangerine)\b',
    r'\b(?:brown|coffee|tan|beige|khaki|bronze)\b',
    r'\b(?:grey|gray|silver)\b',
    r'\b(?:colour|color|shade|dual shade|multi colour|multicolour|with|and|&)\b'
]

def clean_design_base(name: str) -> str:
    s = str(name).lower()
    s = re.sub(r'\b[qqaawwi]\w*\d+\w*\b', '', s)
    for ct in COLOR_TERMS:
        s = re.sub(ct, ' ', s, flags=re.IGNORECASE)
    s = re.sub(r'[^a-z\s]', ' ', s)
    tokens = [t for t in s.split() if len(t) > 2]
    return ' '.join(tokens)

def main():
    meta = pd.read_parquet(config.METADATA_PATH)
    index = faiss.read_index(config.INDEX_PATH)
    vectors = np.array([index.reconstruct(i) for i in range(index.ntotal)])

    meta['design_base'] = meta['name'].apply(clean_design_base)

    # 1. Tier 1: Own page
    tier1_mask = meta['material'].notnull() & (meta['material'] != '') & meta['blouse_included'].notnull()
    tier1_indices = meta[tier1_mask].index.tolist()

    meta['specs_source'] = 'unavailable'
    meta['sibling_sku'] = None

    for idx in tier1_indices:
        meta.at[idx, 'specs_source'] = 'own_page'

    # 2. Tier 2: Sibling inference
    tier2_count = 0
    for i in range(len(meta)):
        if meta.at[i, 'specs_source'] == 'own_page':
            continue
        
        base_i = meta.at[i, 'design_base']
        vec_i = vectors[i]
        
        best_sibling_idx = None
        best_sim = -1.0
        
        for j in tier1_indices:
            base_j = meta.at[j, 'design_base']
            tokens_i = set(base_i.split())
            tokens_j = set(base_j.split())
            overlap = len(tokens_i.intersection(tokens_j))
            
            if overlap >= 2 or (len(tokens_i) >= 1 and tokens_i.issubset(tokens_j)):
                sim = float(np.dot(vec_i, vectors[j]))
                if sim > best_sim:
                    best_sim = sim
                    best_sibling_idx = j
                    
        if best_sibling_idx is not None and best_sim >= 0.70:
            sib = meta.iloc[best_sibling_idx]
            meta.at[i, 'specs_source'] = 'inferred_from_sibling'
            meta.at[i, 'sibling_sku'] = sib['sku']
            meta.at[i, 'material'] = sib['material']
            meta.at[i, 'blouse_included'] = sib['blouse_included']
            meta.at[i, 'blouse_length'] = sib['blouse_length']
            meta.at[i, 'saree_length'] = sib['saree_length']
            meta.at[i, 'saree_weight'] = sib['saree_weight']
            meta.at[i, 'wash_care'] = sib['wash_care']
            meta.at[i, 'net_quantity'] = sib['net_quantity']
            tier2_count += 1

    # 3. Tier 3: Visual inference
    tier3_indices = meta[meta['specs_source'] == 'unavailable'].index.tolist()
    for idx in tier3_indices:
        meta.at[idx, 'specs_source'] = 'visual_inference'
        meta.at[idx, 'blouse_length'] = None
        meta.at[idx, 'saree_length'] = None
        meta.at[idx, 'saree_weight'] = None
        meta.at[idx, 'wash_care'] = None

    # Drop intermediate column
    meta = meta.drop(columns=['design_base'])
    meta.to_parquet(config.METADATA_PATH, index=False)

    print("Successfully applied 4-tier pipeline to data/metadata.parquet:")
    print(meta['specs_source'].value_counts())

if __name__ == '__main__':
    main()
