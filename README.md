# TailorTalk — Saree Visual & Attribute Similarity Search Agent

A conversational shopping assistant that finds visually and stylistically matching sarees from an authentic ~1,070-item catalogue.
Upload or link a saree photo, chat naturally (*"find me something like this in pink under ₹4000"*, *"is the blouse included?"*), and the agent calls a hybrid multi-modal search engine combining FAISS vector search, deterministic metadata filters, and real scraped product specifications.

- **Live Deployed App:** [https://tailortalk-saree-search-27xcws5eqjfxzkdpbmmpy6.streamlit.app/](https://tailortalk-saree-search-27xcws5eqjfxzkdpbmmpy6.streamlit.app/)
- **GitHub Repository:** [https://github.com/pk7745/tailortalk-saree-search](https://github.com/pk7745/tailortalk-saree-search)

---

## 1. Architecture & Multi-Modal Search

```
data/products.csv (name, sku, price, image_url, product_link)
        │
        ├──► ingest.py (offline indexing)
        │       └─ Dual-Representation: CLIP Whole (512d) + 3D HSV (512d) → FAISS IndexFlatIP (1024d)
        │
        ├──► enrich_metadata.py (offline web scraper)
        │       └─ fetch product_link → extract material, blouse, saree length, wash care, stock
        │
        └──► apply_4tier_pipeline.py (4-Tier provenance pipeline)
                ├─ Tier 1: Own product page (546 records, 51.0%)
                ├─ Tier 2: Design-family sibling inference (484 records, 45.2%)
                ├─ Tier 3: Visual inference from photo (40 records, 3.7%)
                └─ Tier 4: Confirmed unavailable (0 records, 0.0%)
                        │
                        ▼
        data/saree_index.faiss (1,070 vectors, 1024d) + data/metadata.parquet
                        │
                        ▼ committed to GitHub
┌─────────────────────────── app.py (Streamlit Community Cloud) ──────────────────────────┐
│  Image Upload / URL   →  session_state.current_image (LRU cached 3.0ms embeddings)      │
│  Chat Input & Memory  →  agent.py (LangChain + Gemini Tool-Calling)                     │
│                             └─ find_similar_sarees(top_k, max_price, color, fabric)     │
│                                      └─ search_tool.py (Over-fetch FAISS + Filter)      │
│  Results Grid & Cards →  Persistent in-chat cards with images, prices, badges & links   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Key Capabilities & Technical Design

### A. Dual-Representation Early-Fused Embeddings (`embeddings.py`)
Because all 1,070 items share the same basic category silhouette, standard embeddings collapse. We combine two complementary representations into a 1,024-dimensional L2-normalized vector:
1. **Whole-Image OpenCLIP ViT-B/32 (70% weight)**: Captures global silhouette, drapery structure, and semantic motif textures.
2. **3D HSV Color Histogram (30% weight)**: Captures precise multi-tone color distributions across 512 bins (independent of lighting exposure variances).
Because every catalogue image belongs to the same category (*sarees*), plain CLIP often over-indexes on general "saree-ness". We solve this via **early-fusion**:
1. **OpenCLIP ViT-B/32** (512-d, L2 normalized): Garment silhouette, drapery, and semantic patterns.
2. **3D HSV Color Histogram** (8×8×8 = 512-d, L2 normalized): Hue, saturation, and multi-color distribution.
3. **Fused Vector**: $\text{normalize}(0.7 \cdot \text{CLIP} + 0.3 \cdot \text{HSV}) \rightarrow \mathbf{1,024}\text{-dimensional vector}$.

### B. Hybrid Search & Filtering (`search_tool.py`)
- **Over-fetch + Filter**: Queries vector similarity from FAISS first, then strictly enforces user constraints (`max_price`, `min_price`, `color`, `fabric`), returning the `top_k` results ranked by true cosine similarity.
- **Empirical Thresholding**: Matches with cosine similarity $\ge 0.60$ are confirmed visual matches; below $0.60$ are explicitly tagged as stylistic alternatives.
- **Natural Language Parsing**: `parse_query_intent()` extracts budget constraints across 8+ phrasings (*"under 3000"*, *"cheaper than 3k"*, *"budget below three thousand"*), 40+ colors, and 30+ fabric weaves.

### C. Strict 4-Tier Provenance Pipeline (`apply_4tier_pipeline.py`)
Every record is tagged with its authentic `specs_source`:
1. **Tier 1 (Own Product Page — 546 records, 51.0%)**: Verified specification table extracted from the live merchant page.
2. **Tier 2 (Design-Family Sibling — 484 records, 45.2%)**: Inferred from identical design siblings with matching base names and high visual similarity ($\ge 0.70$). Sibling SKU is stored alongside.
3. **Tier 3 (Visual Inference — 40 records, 3.7%)**: Visual attributes (color, pattern, weave appearance) observed from photo. Measurements, lengths, and wash-care remain strictly `None`.
4. **Tier 4 (Unavailable — 0 records, 0.0%)**: All 1,070 indexed catalogue images are resolved.

### D. Grounded Conversational Memory (`agent.py`)
- Multi-turn conversational memory allows natural follow-ups (*"what's the price of the second one?"*, *"is that one dry clean only?"*).
- Strict groundedness: The agent states facts as confirmed for Tier 1, explicitly discloses sibling derivation for Tier 2, and refuses to guess measurements for Tier 3.

### E. Border/Pallu Region Matching: Chronicle of 3 Investigations
To address fine-grained border and pallu differentiation, we conducted three rigorous empirical investigations across all 1,070 catalogue records:

1. **Attempt 1: Fixed Geometric Cropping (Bottom 35% / Right 35%)**:
   - *Hypothesis*: Saree borders predominantly fall along the lower skirt and right-side pallu fall.
   - *Outcome*: Varied mannequin draping angles and folds caused arbitrary geometric cuts to slice through plain pleats or background, degrading 3 of 10 test pairs and introducing spatial noise that reduced self-identity confidence.

2. **Attempt 2: Pretrained Deep Segmentation (`sayeed99/segformer-b3-fashion`)**:
   - *Hypothesis*: Semantic segmentation would adaptively detect ornamentation (`applique`, `bead`, `fringe`, `sequin`, `tassel`).
   - *Outcome*: Revealed a fundamental domain mismatch. SegFormer is trained on Western street-wear with sewn-on trims, whereas authentic Indian handloom sarees feature **jacquard-woven gold Zari and brocade wefts** integrated directly into the fabric's warp and weft (0.00% detection). The mandatory hard fallback safely defaulted 100% of images to whole-image representations without degradation.

3. **Attempt 3: Classical Computer Vision Heuristic (OpenCV Edge Density & Texture Variance)**:
   - *Hypothesis*: Multi-strip Canny edge density, Laplacian variance, and HSV saturation gradients would locate ornate high-frequency border bands.
   - *Outcome*: Successfully detected ornate regions across 93.6% (1,001/1,070) of catalogue images. However, feeding non-standard rectangular bounding boxes into CLIP's fixed 224×224 input tensor caused aspect-ratio warping and high-frequency edge artifacts. On a 22-pair full-catalogue rank test, 10 pairs degraded (distractors moved closer, with diff-border median rank worsening from #182 to #168).

4. **Final Proven Architecture**:
   - By unanimous empirical evidence, the **Dual-Representation Early-Fusion Model (Whole-Image CLIP 0.70 + 3D HSV Color 0.30)** with lossless in-memory caching provides the highest retrieval precision and stability:
     - **Same-Border / Matching Design Targets**: Median Rank **#10** (top matches at #3, #7, #8, #10, #16).
     - **Different-Border Distractors**: Median Rank **#182** (pushed down to #305, #516, #879, #965).
     - **Separation**: **172 rank positions** of natural discrimination without spatial distortion, retaining **100% exact self-identity (`score >= 0.98`)** and **3.0ms re-query latency**. Specific border weave names (*Temple Border*, *Kadiyal*, *Zari*) are grounded via scraped product metadata.

---

## 3. Tech Stack & Decisions

| Layer | Component | Rationale |
|---|---|---|
| **Vector DB** | **Meta FAISS (`IndexFlatIP`)** | In-process exact cosine similarity for 1,070 vectors; 0ms network latency, zero cloud hosting costs. |
| **Embeddings** | **OpenCLIP ViT-B/32 + HSV** | Dual-representation early-fusion with lossless in-memory caching (~3ms re-queries). |
| **Agent / LLM** | **LangChain + Google Gemini Flash** | Function calling with typed `Pydantic` filter schemas. |
| **Frontend** | **Streamlit** | Interactive chat interface, persistent product cards, image previews, and luxury badges. |

---

## 4. Verification & Testing

Run the automated self-test suite:
```bash
python selftest.py
```

### Self-Test Results (7/7 Checks Passing):
```text
[PASS] Coverage >= 90% of catalogue indexed (1,070 / 1,074 rows, 99.6%)
[PASS] Identity: querying a catalogue image returns itself as #1 (score >= 0.98)
[PASS] Discrimination: similarity scores have real spread across catalogue (> 0.15)
[PASS] Schema: results contain documented fields, top_k respected
[PASS] Metadata Filtering: filtered query (max_price=3000) strictly enforces price <= 3000
[PASS] Agent calls the tool when asked for similar items
[PASS] Agent does NOT call the tool for unrelated chit-chat
```

---

## 5. Local Setup Instructions

```bash
git clone https://github.com/pk7745/tailortalk-saree-search.git
cd tailortalk-saree-search

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt

# Run the app locally
streamlit run app.py
```
