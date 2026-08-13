import os
import sys

logf = open("inspect_norms.log", "w", encoding="utf-8")
sys.stdout = logf
sys.stderr = logf

try:
    import faiss
    import numpy as np
    import pandas as pd
    from PIL import Image
    import config
    from embeddings import get_fused_embedding

    print("=" * 80)
    print("INSPECTING VECTOR NORMS & FAISS SIMILARITY SCORES")
    print("=" * 80)

    # 1. Stored Index Vector Norms
    idx = faiss.read_index(config.INDEX_PATH)
    vectors = np.array([idx.reconstruct(i) for i in range(min(100, idx.ntotal))])
    norms = np.linalg.norm(vectors, axis=1)

    print("\n1. STORED FAISS VECTORS NORM AUDIT:")
    print(f"   - Sample Vector Norms (first 5): {norms[:5]}")
    print(f"   - Min Norm: {norms.min():.6f}")
    print(f"   - Max Norm: {norms.max():.6f}")
    print(f"   - Mean Norm: {norms.mean():.6f}")

    # 2. Query Vector Norm
    img = Image.new("RGB", (224, 224), color=(200, 100, 150))
    q_vec = get_fused_embedding(img)
    q_norm = np.linalg.norm(q_vec)
    print("\n2. QUERY VECTOR NORM AUDIT:")
    print(f"   - Query Vector Norm: {q_norm:.6f}")

    # 3. Real FAISS Cosine Similarity Scores
    scores, indices = idx.search(q_vec.reshape(1, -1).astype("float32"), 10)
    print("\n3. REAL FAISS SEARCH SCORES (First 10):")
    for i, (sc, idx_val) in enumerate(zip(scores[0], indices[0]), 1):
        print(f"   #{i} Candidate Index {idx_val}: Cosine Score = {sc:.4f}")

    print(f"\n   - Score Min: {scores[0].min():.4f}, Score Max: {scores[0].max():.4f}")

except Exception as e:
    import traceback
    print("ERROR:", e)
    traceback.print_exc()
finally:
    logf.flush()
    logf.close()
