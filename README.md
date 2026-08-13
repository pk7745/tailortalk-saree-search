# TailorTalk — Saree Visual & Attribute Similarity Search Agent (Qdrant Powered)

A conversational shopping assistant that finds visually and stylistically matching sarees from an authentic **1,074-record catalogue** (1,070 indexed + 4 documented server 404s).
Upload or link a saree photo, chat naturally (*"find me something like this in pink under ₹4000"*, *"show me golden zari border sarees"*, *"is the blouse included?"*), and the agent calls a hybrid multi-modal search engine combining **Qdrant Vector Database**, 3-source enriched metadata filters, real scraped product specifications, and an automatic **FAISS Emergency Fallback**.

- **Live Deployed App:** [https://tailortalk-saree-search-27xcws5eqjfxzkdpbmmpy6.streamlit.app/](https://tailortalk-saree-search-27xcws5eqjfxzkdpbmmpy6.streamlit.app/)
- **GitHub Repository:** [https://github.com/pk7745/tailortalk-saree-search](https://github.com/pk7745/tailortalk-saree-search)

---

## 1. Vector Search Layer Architecture (Qdrant Primary + FAISS Fallback)

```text
                USER (Web Browser)
                  │
                  ▼
             STREAMLIT (app.py)
                  │
                  ▼
       GEMINI 2.5 FLASH (agent.py)
                  │
       find_similar_sarees tool call
                  │
                  ▼
          search_tool.py
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
    QDRANT              FAISS
   PRIMARY             FALLBACK
(tailortalk_sarees)  (saree_index.faiss)
        │                   │
        └─────────┬─────────┘
                  ▼
          Candidate Products
                  │
                  ▼
        Hard Price Filtering (min_price, max_price)
                  │
                  ▼
       Attribute Match Score (S_attr)
                  │
                  ▼
        Hybrid Re-ranking
    0.75 Visual + 0.25 Attribute
                  │
                  ▼
            TOP-K SAREES
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 Product Cards       Gemini Explanation
        │                   │
        └─────────┬─────────┘
                  ▼
                 USER
```

---

## 2. Dataset & Index Audit Summary

```
Total Source CSV Records:               1,074
Successfully Downloaded & Indexed:      1,070 (100% of available image URLs)
Documented Server Image Failures:           4 (HTTP 404 Not Found on image host)
Qdrant Collection Name:                 tailortalk_sarees
Qdrant Vector Dimension:                1,024
Qdrant Distance Metric:                 COSINE
FAISS Fallback Vector Count:            1,070 (1024-dimensional IndexFlatIP)
Enriched Metadata Rows:                 1,070
Enriched Metadata Columns:                 37 (CSV + Web Scraped + OpenCV Visual + Name Signals)
Unique Product Image URLs:              1,070 (Primary Key)
```

### Documented Image Server Failures (`data/failed_downloads.csv`):
1. `QS264566`: Tussar Saree With Madhubani Print Dusty Purple With Traditional Art (HTTP 404)
2. `QS270932`: Pure Mysore Silk Saree with pink & Contrast Blue Zari Border (HTTP 404)
3. `QS282590`: Tissue Saree With Lotus Printed (HTTP 404)
4. `QS282741`: Royal Blue Pure Mysore Silk Saree with Golden Checks and Border (HTTP 404)

---

## 3. Qdrant Configuration & Deployment Settings

### Environment Variables & Streamlit Secrets (`.env` or `st.secrets`):

```bash
# Qdrant Vector Database Configuration
USE_QDRANT=true
QDRANT_URL=https://your-qdrant-cluster.cloud.qdrant.io:6333  # Optional for Cloud, defaults to local
QDRANT_API_KEY=your_qdrant_api_key_here                    # Optional for Cloud, defaults to local
QDRANT_COLLECTION_NAME=tailortalk_sarees

# Google Gemini LLM API Key
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Connection Priority:
1. **Qdrant Cloud**: Connected if `QDRANT_URL` and `QDRANT_API_KEY` are provided.
2. **Remote Qdrant Server**: Connected if `QDRANT_URL` is provided.
3. **Local Embedded Qdrant**: Connected via `qdrant_client.QdrantClient(path="data/qdrant_db")`.
4. **FAISS Emergency Fallback**: Automatically triggered if Qdrant is disabled or unavailable.

---

## 4. Key Capabilities & Technical Design

### A. Dual-Representation Fused Embeddings (`embeddings.py`)
Because all 1,070 items share the same basic category silhouette, standard embeddings collapse. We combine two complementary representations into a 1,024-dimensional L2-normalized vector:
1. **Whole-Image OpenCLIP ViT-B/32 (70% weight)**: Captures global silhouette, drapery structure, and semantic motif textures.
2. **3D HSV Color Histogram (30% weight)**: Captures precise multi-tone color distributions across 512 bins.
3. **Fused Vector**: $\text{normalize}(0.7 \cdot \text{CLIP} + 0.3 \cdot \text{HSV}) \rightarrow \mathbf{1,024}\text{-dimensional unit vector}$.

### B. Hybrid Search & Attribute-Weighted Re-Ranking (`search_tool.py`)
- **Qdrant Candidate Retrieval**: Vector similarity search on Qdrant with payload price range filtering.
- **Hard Constraints**: Strictly enforces budget constraints (`max_price`, `min_price`).
- **Attribute Match Scoring ($S_{\text{attr}} \in [0.0, 1.0]$)**: Evaluates candidate relevance for requested border, pallu, pattern, color, and fabric attributes:
  - `1.0`: Exact match in primary catalogue/scraped metadata
  - `0.8`: Strong synonym / family match (e.g. `zari border` ↔ `golden zari`)
  - `0.5`: OpenCV visual detection signal (`visual_border_detected`, `visual_zari_detected`)
  - `0.0`: No match (filtered out)
- **Combined Re-Ranking Formula**:
  $$\text{FinalScore} = \begin{cases} S_{\text{visual}} & \text{for Image-Only Queries (100\% visual dominant)} \\ 0.75 \cdot S_{\text{visual}} + 0.25 \cdot S_{\text{attr}} & \text{for Image + Attribute Queries (75\% visual, 25\% attribute boost)} \\ S_{\text{attr}} & \text{for Text-Only Queries} \end{cases}$$

### C. Strict 4-Tier Provenance Pipeline
Every record is tagged with its authentic `specs_source`:
1. **Tier 1 (Own Product Page — 546 records, 51.0%)**: Verified specification table extracted from live merchant page.
2. **Tier 2 (Design-Family Sibling — 484 records, 45.2%)**: Inferred from identical design siblings with matching base names and high visual similarity ($\ge 0.70$).
3. **Tier 3 (Visual Inference — 40 records, 3.7%)**: Visual attributes (color, pattern, weave appearance) observed from photo.
4. **Tier 4 (Unavailable — 4 records)**: Explicitly documented image server HTTP 404 errors.

---

## 5. Migration & Verification Scripts

- **`migrate_to_qdrant.py`**: Idempotent migration script that batch upserts 1,070 products with 1024d vectors and 37 metadata columns as payload into Qdrant collection `tailortalk_sarees`.
- **`validate_qdrant.py`**: Audits collection status, vector size (1024), distance metric (COSINE), point count (1070), and payload fields.
- **`compare_faiss_qdrant.py`**: Compares top-5 candidates, similarity scores, and search latency between FAISS and Qdrant.
- **`test_qdrant_queries.py`**: Regression test verifying 15 mandatory user queries on Qdrant.
- **`verify_counts.py`**: Validates CSV row count (1,074), metadata row count (1,070), FAISS `ntotal` (1,070), unique image URLs (1,070), and 37 enriched metadata columns.

---

## 6. Commands to Run Migration & Application

### Run Qdrant Migration:
```bash
python migrate_to_qdrant.py
```

### Validate Qdrant Collection:
```bash
python validate_qdrant.py
```

### Run Comparison Test (FAISS vs Qdrant):
```bash
python compare_faiss_qdrant.py
```

### Run Application:
```bash
streamlit run app.py
```

---

## 7. Tech Stack & Decisions

| Layer | Component | Rationale |
|---|---|---|
| **Vector DB (Primary)** | **Qdrant (`tailortalk_sarees`)** | Production vector database supporting Cloud & Local mode with COSINE distance and payload filtering. |
| **Vector DB (Fallback)**| **Meta FAISS (`IndexFlatIP`)** | Emergency local fallback ensuring 0ms search crashes if Qdrant credentials fail. |
| **Embeddings** | **OpenCLIP ViT-B/32 + 3D HSV** | Dual-representation early-fusion vector space (1024d, L2 normalized). |
| **Visual Analysis** | **OpenCV (`cv2`)** | Supplementary Canny edge density, region contrast, and Zari sheen analysis. |
| **Agent / LLM** | **LangChain + Google Gemini Flash** | Function calling with typed `Pydantic` filter schemas. |
| **Frontend** | **Streamlit** | Reactive chat interface, persistent product cards grid, and luxury badges. |
