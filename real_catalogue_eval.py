"""
Real Catalogue Image Quality Evaluation.
Uses the exact L2-normalized 1024-dimensional fused embeddings stored in saree_index.faiss
for 5 real catalogue items to evaluate real visual search and attribute re-ranking.
"""
import faiss
import numpy as np
import pandas as pd
import config
from search_tool import _compute_attribute_match_score, _load_index_and_meta

SAMPLE_INDICES = [0, 5, 12, 25, 42]

def run_query_eval(idx, meta, query_vec, color=None, fabric=None, pattern=None, max_price=None, top_k=5):
    scores, indices = idx.search(query_vec.reshape(1, -1).astype("float32"), idx.ntotal)
    scores_arr = scores[0]
    indices_arr = indices[0]

    has_attr = bool(color or fabric or pattern)
    candidates = []

    for sc, i in zip(scores_arr, indices_arr):
        if i == -1 or i >= len(meta):
            continue
        row = meta.iloc[i]

        p_str = str(row.get("price", "0")).replace("₹", "").replace(",", "").strip()
        try:
            p_val = float(p_str)
        except ValueError:
            p_val = 0.0

        if max_price is not None and p_val > max_price:
            continue

        attr_score, passes = _compute_attribute_match_score(row, color=color, fabric=fabric, pattern=pattern)
        if has_attr and not passes:
            continue

        v_score = float(sc)
        if has_attr:
            final_score = 0.75 * v_score + 0.25 * attr_score
        else:
            final_score = v_score

        candidates.append({
            "sku": row["sku"],
            "name": row["name"],
            "v_score": round(v_score, 4),
            "attr_score": round(attr_score, 4),
            "final_score": round(final_score, 4),
        })

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates[:top_k]


def main():
    print("=" * 95)
    print("REAL CATALOGUE IMAGE QUALITY EVALUATION (5 REAL PRODUCT EMBEDDINGS)")
    print("=" * 95)

    idx, meta = _load_index_and_meta()

    for s_idx in SAMPLE_INDICES:
        row = meta.iloc[s_idx]
        name = row["name"]
        sku = row["sku"]
        c_color = row.get("color", "")
        c_fabric = row.get("fabric", "")

        # Extract the real 1024-d fused CLIP+HSV embedding directly from FAISS
        real_q_vec = idx.reconstruct(s_idx)

        print("\n" + "-" * 95)
        print(f"QUERY SAREE #{s_idx+1}: {name}")
        print(f"SKU: {sku} | Color: {c_color} | Fabric: {c_fabric}")

        # 1. Image Only Similarity
        r1 = run_query_eval(idx, meta, real_q_vec, top_k=5)
        print("\n  [1. Image-Only Similarity]")
        for i, item in enumerate(r1, 1):
            print(f"     #{i} [{item['sku']}] {item['name'][:60]} | VisScore: {item['v_score']}")

        # 2. Image + Golden Zari Border
        r2 = run_query_eval(idx, meta, real_q_vec, pattern="golden zari", top_k=5)
        print("\n  [2. Image + Golden Zari Border]")
        for i, item in enumerate(r2, 1):
            print(f"     #{i} [{item['sku']}] {item['name'][:60]} | Final: {item['final_score']} (Vis: {item['v_score']}, Attr: {item['attr_score']})")

        # 3. Image + Colour
        target_color = "pink" if "pink" in name.lower() or "pink" in c_color.lower() else "blue"
        r3 = run_query_eval(idx, meta, real_q_vec, color=target_color, top_k=5)
        print(f"\n  [3. Image + Colour '{target_color}']")
        for i, item in enumerate(r3, 1):
            print(f"     #{i} [{item['sku']}] {item['name'][:60]} | Final: {item['final_score']} (Vis: {item['v_score']}, Attr: {item['attr_score']})")

        # 4. Image + Fabric
        target_fabric = "banarasi" if "banaras" in name.lower() or "banarasi" in c_fabric.lower() else "organza"
        r4 = run_query_eval(idx, meta, real_q_vec, fabric=target_fabric, top_k=5)
        print(f"\n  [4. Image + Fabric '{target_fabric}']")
        for i, item in enumerate(r4, 1):
            print(f"     #{i} [{item['sku']}] {item['name'][:60]} | Final: {item['final_score']} (Vis: {item['v_score']}, Attr: {item['attr_score']})")

        # 5. Image + Temple Border
        r5 = run_query_eval(idx, meta, real_q_vec, pattern="temple border", top_k=5)
        print("\n  [5. Image + Pattern 'temple border']")
        for i, item in enumerate(r5, 1):
            print(f"     #{i} [{item['sku']}] {item['name'][:60]} | Final: {item['final_score']} (Vis: {item['v_score']}, Attr: {item['attr_score']})")

    print("\n" + "=" * 95)
    print("EVALUATION COMPLETE - ALL 5 REAL CATALOGUE ITEMS TESTED ACCROSS 5 QUERY TYPES!")
    print("=" * 95)

if __name__ == "__main__":
    main()
