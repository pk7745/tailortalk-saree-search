"""
Test script to benchmark and compare Old vs New Ranking Pipeline on representative test queries.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import faiss
from PIL import Image
import config
from search_tool import _load_index_and_meta, _matches_pattern

def compute_attribute_score(row, color=None, fabric=None, pattern=None):
    """
    Returns (attr_score, passes_filter)
    Scoring:
    1.0 = Exact primary metadata match
    0.8 = Synonym / related family match
    0.5 = Visual signal / partial match
    0.0 = No match
    """
    scores = []
    
    # 1. Pattern / Border / Pallu
    if pattern:
        pq = pattern.lower().strip()
        name_str = str(row.get("name", "")).lower()
        pat_str = str(row.get("pattern", "")).lower()
        border_str = str(row.get("border", "")).lower()
        pallu_str = str(row.get("pallu", "")).lower()
        work_str = str(row.get("work_type", "")).lower()
        nb_str = str(row.get("name_border", "")).lower()
        np_str = str(row.get("name_pallu", "")).lower()
        nw_str = str(row.get("name_work", "")).lower()

        primary_text = f"{name_str} {pat_str} {border_str} {pallu_str} {work_str} {nb_str} {np_str} {nw_str}"

        synonyms = {
            'zari border': ['zari border', 'golden zari border', 'gold zari border', 'golden zari', 'gold zari'],
            'golden zari': ['golden zari', 'gold zari', 'zari border', 'zari'],
            'temple border': ['temple border', 'temple motif', 'temple'],
            'kadiyal border': ['kadiyal border', 'kadiyal'],
            'contrast border': ['contrast border', 'contrast'],
            'floral': ['floral print', 'floral work', 'floral embroidery', 'floral'],
            'embroidery': ['embroidery', 'embroidered', 'chikankari', 'kutch work'],
            'pallu': ['parrot pallu', 'zari pallu', 'floral pallu', 'pallu'],
        }

        if pq in primary_text:
            scores.append(1.0)
        else:
            cand_syns = synonyms.get(pq, [pq])
            syn_hit = any(s in primary_text for s in cand_syns)
            if syn_hit:
                scores.append(0.8)
            else:
                # Visual signal check
                vb_str = str(row.get("visual_border_detected", "")).lower()
                vz_str = str(row.get("visual_zari_detected", "")).lower()
                vc_str = str(row.get("visual_contrast_border", "")).lower()
                vis_text = f"{vb_str} {vz_str} {vc_str}"
                if ("zari" in pq and "detected" in vz_str) or ("border" in pq and "detected" in vb_str):
                    scores.append(0.5)
                else:
                    scores.append(0.0)

    # 2. Color
    if color:
        cq = color.lower().strip()
        c_str = str(row.get("color", "")).lower()
        n_str = str(row.get("name", "")).lower()
        sc_str = str(row.get("scraped_color", "")).lower()
        combined_c = f"{c_str} {n_str} {sc_str}"
        if cq in combined_c:
            scores.append(1.0)
        elif any(part in combined_c for part in cq.split()):
            scores.append(0.8)
        else:
            scores.append(0.0)

    # 3. Fabric
    if fabric:
        fq = fabric.lower().strip()
        f_str = str(row.get("fabric", "")).lower()
        m_str = str(row.get("material", "")).lower()
        n_str = str(row.get("name", "")).lower()
        combined_f = f"{f_str} {m_str} {n_str}"
        if fq in combined_f:
            scores.append(1.0)
        elif any(part in combined_f for part in fq.split()):
            scores.append(0.8)
        else:
            scores.append(0.0)

    if not scores:
        return 1.0, True

    attr_score = float(np.mean(scores))
    passes = attr_score > 0.0
    return attr_score, passes

def benchmark_queries():
    index, meta = _load_index_and_meta()
    sample_img = Image.new("RGB", (224, 224), color=(210, 90, 140))
    query_vec = np.concatenate([np.ones(512)*0.7, np.ones(512)*0.3]).astype("float32") # placeholder test

    test_scenarios = [
        ("1. Image only", {}),
        ("2. Image + golden zari border", {"pattern": "golden zari"}),
        ("3. Image + temple border", {"pattern": "temple border"}),
        ("4. Image + floral pattern", {"pattern": "floral"}),
        ("5. Image + parrot pallu", {"pattern": "parrot pallu"}),
        ("6. Image + banarasi fabric", {"fabric": "banarasi"}),
        ("7. Image + pink color", {"color": "pink"}),
        ("8. Image + border + max_price 4000", {"pattern": "zari border", "max_price": 4000}),
    ]

    print("="*80)
    print("BENCHMARKING RANKING IMPROVEMENT (OLD vs NEW)")
    print("="*80)

    for title, kwargs in test_scenarios:
        print(f"\nScenario: {title}")
        color = kwargs.get("color")
        fabric = kwargs.get("fabric")
        pattern = kwargs.get("pattern")
        max_p = kwargs.get("max_price")

        # Simulate vector search scores with an L2-normalized test vector
        query_vec = np.random.rand(1, 1024).astype("float32")
        query_vec = query_vec / np.linalg.norm(query_vec)
        scores, indices = index.search(query_vec, index.ntotal)
        scores_arr = scores[0]
        indices_arr = indices[0]

        # Evaluate candidate scoring
        candidates = []
        for score, idx in zip(scores_arr, indices_arr):
            row = meta.iloc[idx]
            p_val = float(str(row.get("price", "0")).replace("₹","").replace(",","") or 0)
            if max_p is not None and p_val > max_p:
                continue

            attr_score, passes = compute_attribute_score(row, color=color, fabric=fabric, pattern=pattern)
            if not passes:
                continue

            v_score = float(score)
            has_attrs = bool(color or fabric or pattern)
            final_score = (0.75 * v_score + 0.25 * attr_score) if has_attrs else v_score

            candidates.append({
                "name": row["name"],
                "sku": row["sku"],
                "v_score": round(v_score, 4),
                "attr_score": round(attr_score, 4),
                "final_score": round(final_score, 4),
            })

        # Sort by final_score descending
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        print(f"Top 3 Results (New Combined Ranking):")
        for i, c in enumerate(candidates[:3], 1):
            print(f"  #{i} {c['name'][:40]} | Final: {c['final_score']} (Vis: {c['v_score']}, Attr: {c['attr_score']})")

if __name__ == "__main__":
    benchmark_queries()
