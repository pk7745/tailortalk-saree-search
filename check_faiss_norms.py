import os
import faiss
import numpy as np
import config

print("=" * 80)
print("STORED FAISS INDEX VECTOR NORM AUDIT")
print("=" * 80)

idx = faiss.read_index(config.INDEX_PATH)
print(f"Index Total Vectors: {idx.ntotal}")
print(f"Index Dimension:     {idx.d}")

vectors = np.array([idx.reconstruct(i) for i in range(idx.ntotal)])
norms = np.linalg.norm(vectors, axis=1)

print(f"Norm Sample (first 10): {norms[:10]}")
print(f"Min Norm:   {norms.min():.6f}")
print(f"Max Norm:   {norms.max():.6f}")
print(f"Mean Norm:  {norms.mean():.6f}")
print(f"Std Dev:    {norms.std():.6f}")
