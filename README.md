# TailorTalk — Saree Visual & Attribute Similarity Search Agent

A conversational shopping assistant that finds visually and stylistically matching sarees from an authentic ~1,070-item catalogue.
Upload or link a saree photo, chat naturally (*"find me something like this in pink under ₹4000"*, *"is the blouse included?"*), and the agent calls a hybrid multi-modal search engine combining FAISS vector search, deterministic metadata filters, and real scraped product specifications.

- **Live Deployed App:** [https://tailortalk-saree-search.streamlit.app](https://tailortalk-saree-search.streamlit.app)
- **GitHub Repository:** [https://github.com/pk7745/tailortalk-saree-search](https://github.com/pk7745/tailortalk-saree-search)

---

## 1. Architecture & Multi-Modal Search

```
data/products.csv (name, sku, price, image_url, product_link)
        │
        ├──► ingest.py (offline indexing)
        │       └─ download images → fused CLIP (512d) + HSV (512d) → FAISS IndexFlatIP (1024d)
        │
        └──► enrich_metadata.py (offline web scraper)
                └─ fetch product_link → extract material, blouse, saree length, wash care, stock
                        │
                        ▼
        data/saree_index.faiss (1,070 vectors) + data/metadata.parquet
                        │
                        ▼ committed to GitHub
┌─────────────────────────── app.py (Streamlit Community Cloud) ──────────────────────────┐
│  Image Upload / URL   →  session_state.current_image                                    │
│  Chat Input & Memory  →  agent.py (LangChain + Gemini Tool-Calling)                     │
│                             └─ find_similar_sarees(top_k, max_price, color, fabric)     │
│                                      └─ search_tool.py (Over-fetch FAISS + Filter)      │
│  Results Grid & Cards →  Renders image, title, price, badges (color, fabric), link      │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Key Capabilities & Technical Design

### A. Fused Dual-Representation Embeddings (`embeddings.py`)
Because every catalogue image belongs to the same category (*sarees*), plain CLIP often over-indexes on general "saree-ness". We solve this via **early-fusion**:
1. **OpenCLIP ViT-B/32** (512-d, L2 normalized): Garment silhouette, drapery, and semantic patterns.
2. **3D HSV Color Histogram** (8×8×8 = 512-d, L2 normalized): Hue, saturation, and multi-color distribution.
3. **Fused Vector**: $\text{normalize}(0.7 \cdot \text{CLIP} + 0.3 \cdot \text{HSV}) \rightarrow \mathbf{1,024}\text{-dimensional vector}$.

### B. Hybrid Search & Filtering (`search_tool.py`)
- **Over-fetch + Filter**: Queries vector similarity from FAISS first, then strictly enforces user constraints (`max_price`, `min_price`, `color`, `fabric`), returning the `top_k` results ranked by true cosine similarity.
- **Natural Language Parsing**: `parse_query_intent()` extracts budget constraints (*"under 3000"*), 40+ color tones (*"rani pink"*, *"navy blue"*), and authentic weave taxonomies (*"banarasi"*, *"organza"*, *"pashmina"*, *"linen"*, *"satin"*).

### C. On-Page Metadata Enrichment (`enrich_metadata.py`)
- Live scraping of `byrappasilks.in` product links extracts verified specifications: `material`, `blouse_included`, `blouse_length`, `saree_length`, `saree_weight`, `wash_care`, `net_quantity`, and `stock_status`.
- **Enrichment Coverage & Honest Limitation**: **411 out of 1,070 items (38.4%)** contain full structured specification tables on the source site. For items where a specific field is not rendered on the merchant site, an explicit `None` is preserved to guarantee zero LLM hallucination.

### D. Grounded Conversational Memory (`agent.py`)
- Multi-turn conversational memory allows natural follow-ups (*"what's the price of the second one?"*, *"is that one dry clean only?"*).
- Strict groundedness: The agent answers product questions strictly from verified tool metadata, stating when a field is unrecorded rather than inventing details.

---

## 3. Tech Stack & Decisions

| Layer | Component | Rationale |
|---|---|---|
| **Vector DB** | **Meta FAISS (`IndexFlatIP`)** | In-process exact cosine similarity for 1,070 vectors; 0ms network latency, zero cloud hosting costs. |
| **Embeddings** | **OpenCLIP ViT-B/32 + HSV** | Lightweight CPU inference (~1s per query) on free cloud hosting. |
| **Agent / LLM** | **LangChain + Google Gemini Flash** | Function calling with typed `Pydantic` filter schemas. |
| **Frontend** | **Streamlit** | Interactive chat interface, image previews, and luxury card badges. |

---

## 4. Verification & Testing

Run the automated self-test suite:
```bash
python selftest.py
```

### Self-Test Results (6/6 Checks Passing):
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
