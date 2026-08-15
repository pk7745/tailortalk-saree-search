# 🥻 TailorTalk Saree Search — Evidence-First Multimodal AI Shopping Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tailortalk-saree-search-27xcws5eqjfxzkdpbmmpy6.streamlit.app/)
[![Qdrant Vector DB](https://img.shields.io/badge/VectorDB-Qdrant%201024d-FF4A00.svg)](https://qdrant.tech/)
[![Google Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-4285F4.svg)](https://aistudio.google.com/)
[![OpenCLIP Vision](https://img.shields.io/badge/Vision-OpenCLIP%20ViT--B%2F32-2B2D42.svg)](https://github.com/mlfoundations/open_clip)
[![Python 3.10](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://www.python.org/)

**TailorTalk Saree Search** is an evidence-first, grounded, multimodal saree discovery engine and conversational shopping assistant. It integrates **1,024-dimensional fused visual vector search** via **Qdrant**, deterministic attribute filtering, **SSRF-protected live webpage verification**, and **Google Gemini 2.5 Flash** tool-calling orchestration to deliver precise product recommendations from a catalogue of **1,070 authentic saree products**.

🌐 **Live Deployment**: [tailortalk-saree-search-27xcws5eqjfxzkdpbmmpy6.streamlit.app](https://tailortalk-saree-search-27xcws5eqjfxzkdpbmmpy6.streamlit.app/)

---

## 📑 Table of Contents
- [✨ Key Features](#-key-features)
- [🏛️ System Architecture & Data Flow](#️-system-architecture--data-flow)
- [🧬 Fused Multimodal Embedding Pipeline](#-fused-multimodal-embedding-pipeline)
- [⚡ Multi-Stage Qdrant Retrieval Engine](#-multi-stage-qdrant-retrieval-engine)
- [🛡️ Web Evidence Verification & SSRF Security](#️-web-evidence-verification--ssrf-security)
- [🎯 Source-Priority Evidence Routing](#-source-priority-evidence-routing)
- [📊 Catalogue Schema & Qdrant Payload Parity](#-catalogue-schema--qdrant-payload-parity)
- [🧪 Automated Test & Benchmark Results](#-automated-test--benchmark-results)
- [🚀 Quick Start & Installation](#-quick-start--installation)
- [📂 Repository Structure](#-repository-structure)

---

## ✨ Key Features

- **1,024d Fused Vector Search**: Fuses **OpenCLIP ViT-B/32** (512d) semantic vision embeddings with **3D HSV Color Histograms** (512d) for high-fidelity visual and color similarity search.
- **Qdrant Vector Database**: Primary vector search engine running on Qdrant Cloud / Local Embedded with payload indexing on `price_numeric`. Includes automatic fallback to local **FAISS**.
- **SSRF-Protected Web Verification (`web_verifier.py`)**: Live HTTP fetching tool that extracts JSON-LD structured schemas (`Product`, `Offer`) and HTML meta specs from official merchant product pages. Includes private IP blocking and domain allowlisting.
- **Source-Priority Answering**: 4-tier evidence hierarchy (Official Webpage > Catalogue Metadata > Search Metadata > Visual Observation). Never fabricates unverified product facts.
- **Multi-Turn Reference Resolution**: Formats active displayed candidates into prompt context to deterministically resolve pronoun references (*"the second saree"*, *"how much is it?"*, *"is it available?"*).
- **Conflict Resolution**: Explicitly reports price or availability discrepancies if catalogue metadata and live official webpage differ (*"The catalogue lists ₹3,150, while the official product page currently lists ₹3,499."*).
- **0.0% Duplicate Card Guarantee**: 3-layer deduplication by stable product ID (`product_link` $\rightarrow$ `image_url` $\rightarrow$ `sku`).
- **Minimalist Fashion-Oriented UI**: Streamlit interface designed with custom CSS styling (`Playfair Display` + `Plus Jakarta Sans` typography).

---

## 🏛️ System Architecture & Data Flow

```text
                                 ┌─────────────────────────┐
                                 │       USER INPUT        │
                                 │  (Text Query / Image)   │
                                 └────────────┬────────────┘
                                              │
                                              ▼
                                 ┌─────────────────────────┐
                                 │   STREAMLIT UI (app.py) │
                                 │ (Session State / Cards) │
                                 └────────────┬────────────┘
                                              │
                                              ▼
                                 ┌─────────────────────────┐
                                 │  GEMINI ORCHESTRATOR    │
                                 │ (agent.py / Tools Enum) │
                                 └──────┬────────────┬─────┘
                                        │            │
                 ┌──────────────────────┘            └──────────────────────┐
                 ▼                                                          ▼
  ┌──────────────────────────────┐                           ┌──────────────────────────────┐
  │   find_similar_sarees        │                           │ fetch_official_product_details│
  │  (search_tool.py / Qdrant)   │                           │ (web_verifier.py / JSON-LD)  │
  └──────────────┬───────────────┘                           └──────────────┬───────────────┘
                 │                                                          │
  ┌──────────────┴───────────────┐                                          │
  │ 1. 1024d Fused Embedding     │                                          │
  │ 2. Over-Fetch Pool (N=50..100)│                                          │
  │ 3. Hard Budget Constraints   │                                          │
  │ 4. 75/25 Hybrid Reranking    │                                          │
  │ 5. Stable ID Deduplication   │                                          │
  └──────────────┬───────────────┘                                          │
                 │                                                          │
                 └──────────────────────────┬───────────────────────────────┘
                                            ▼
                             ┌──────────────────────────────┐
                             │ 4-Tier Grounded Answer Engine│
                             │ (Source-Priority Reasoning)  │
                             └──────────────┬───────────────┘
                                            │
                                            ▼
                             ┌──────────────────────────────┐
                             │  Product Cards & Chat UI     │
                             │ ("✓ Verified from page")     │
                             └──────────────────────────────┘
```

---

## 🧬 Fused Multimodal Embedding Pipeline

Single vision models under-weight precise color combinations in apparel catalogues. TailorTalk computes a **fused 1,024-dimensional L2-normalized vector**:

$$\mathbf{v}_{\text{fused}} = \text{L2Norm}\left( \text{Concat}\left( 0.70 \cdot \text{L2Norm}(\mathbf{v}_{\text{CLIP}}), \; 0.30 \cdot \text{L2Norm}(\mathbf{v}_{\text{HSV}}) \right) \right)$$

```text
Saree Image ───┬───> OpenCLIP ViT-B/32 ───> 512-dim Vector ───┐
               │                                              ├─> Normalize ───> 1024-dim Fused Vector
               └───> 3D HSV Histogram  ───> 512-dim Vector ───┘
                     (8x8x8 Bins)
```

- **OpenCLIP ViT-B/32 (512d, 70% Weight)**: Captures weave pattern, motif density, border texture, and garment silhouette.
- **3D HSV Histogram (512d, 30% Weight)**: Captures exact color palette distributions ($8 \times 8 \times 8 = 512$ bins) in Hue-Saturation-Value space.

---

## ⚡ Multi-Stage Qdrant Retrieval Engine

Candidates pass through a 6-stage candidate processing pipeline:

1. **Stage A (Embedding)**: Generate fused 1,024d embedding vector.
2. **Stage B (Over-Fetching)**: Retrieve candidate pool ($N = 50..100$ points) from Qdrant Cloud/Local embedded.
3. **Stage C (Hard Filtering)**: Filter by exact numeric constraints (`price <= max_price`, `color`, `fabric`, `pattern`).
4. **Stage D (Hybrid Reranking)**: Rank candidates using $S_{\text{final}} = 0.75 \cdot S_{\text{visual}} + 0.25 \cdot S_{\text{attribute}}$.
5. **Stage E (Deduplication)**: Deduplicate candidates by stable product ID (`product_link` $\rightarrow$ `image_url` $\rightarrow$ `sku`).
6. **Stage F (Delivery)**: Deliver unique top-$K$ candidates to the UI and LLM agent.

---

## 🛡️ Web Evidence Verification & SSRF Security

When live product facts are requested, Gemini invokes `fetch_official_product_details(url: str)`:

```text
Agent Calls fetch_official_product_details(url)
   │
   ▼
SSRF Security Inspection (_is_safe_url)
 ├── Scheme Check (Must be http/https)
 ├── DNS Resolution Check (socket.getaddrinfo)
 ├── Block Private IP Networks (127.0.0.0/8, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 169.254.169.254)
 └── Domain Allowlisting (houseofbyrappa.com, byrappa.com, byrappasilks.in, tailortalk.app)
   │
   ▼ (If Safe)
HTTP Stream Request (Timeout: 5s, Max Body Size: 1MB)
   │
   ▼
BeautifulSoup HTML & JSON-LD Parser
 ├── Extract <script type="application/ld+json"> (Product / Offer Schema)
 ├── Extract OpenGraph Meta Tags (og:price:amount, og:description)
 └── Extract Product Specification Text Snippets
   │
   ▼
LRU In-Memory Cache (@lru_cache(maxsize=128))
```

---

## 🎯 Source-Priority Evidence Routing

Gemini acts strictly as an **orchestration layer**, deriving facts from retrieved evidence:

| Product Attribute | 1st Priority Source | 2nd Priority Source | Fallback / Grounding |
|---|---|---|---|
| **Current Price & Stock** | Official Product Webpage (`web_verifier`) | Catalogue Metadata (`metadata.parquet`) | Never infer visually |
| **Fabric & Specifications** | Official Product Webpage (`web_verifier`) | Catalogue Metadata (`metadata.parquet`) | Never infer exact weave visually |
| **Product SKU / Identity** | Official Webpage | Catalogue Metadata | — |
| **Visual Similarity** | OpenCLIP + 3D HSV Fused Vector | Qdrant Engine | — |
| **Observable Color / Pattern** | Image & Visual Histogram | Catalogue Metadata | State visual observation |

---

## 📊 Catalogue Schema & Qdrant Payload Parity

The catalogue contains **1,070 authentic saree products** and **37 metadata columns** in `data/metadata.parquet`:

| Key Metadata Columns | Data Type | Non-Null Count (%) | Description |
|---|---|---|---|
| `sku` | `object` | 1,070 (100.0%) | Stock keeping unit identifier |
| `name` | `object` | 1,070 (100.0%) | Full product title |
| `price` & `price_numeric` | `object` / `float64` | 1,038 (97.0%) | Price in INR for numeric range filtering |
| `color` | `object` | 1,070 (100.0%) | Color palette filter (54 unique values) |
| `fabric` | `object` | 1,070 (100.0%) | Fabric weave (24 unique values) |
| `pattern` | `object` | 1,070 (100.0%) | Border, pallu, & work type (27 unique values) |
| `product_link` | `object` | 1,070 (100.0%) | Original merchant store URL |
| `image_url` | `object` | 1,070 (100.0%) | Product image URL |
| `specs_source` | `object` | 1,070 (100.0%) | Provenance tier (`own_page`, `inferred_from_sibling`, `visual_inference`) |

> **100% Qdrant Payload Parity**: All 37 parquet columns are preserved in Qdrant point payloads via `row.to_dict()`, ensuring zero data loss during vector ingestion.

---

## 🧪 Automated Test & Benchmark Results

### 1. Adversarial End-to-End Audit Suite (`run_adversarial_audit.py`)
```text
================================================================================
FINAL ADVERSARIAL AUDIT SUMMARY REPORT
================================================================================
  Group 1: Visual Search             : PASS
  Group 2: Hard Constraints          : PASS
  Group 3: Reference Resolution      : PASS
  Group 4: Web Verification          : PASS
  Group 5: Source Conflict           : PASS
  Group 6: Exact Answering           : PASS
  Group 7: Grounding Separation      : PASS
  Group 8: No-Hallucination          : PASS
  Group 9: Ambiguous References      : PASS
  Group 10: Web Failure Fallback     : PASS
  Group 11: FAISS Fallback           : PASS
  Group 12: Security Protections     : PASS
  Group 13: UI Integrity             : PASS
  Group 14: Real Image Matching      : PASS (Manual Visual Relevance Confirmed)

--------------------------------------------------------------------------------
  Total Audited Queries:           4
  Average Retrieval Latency:       174.86 ms
  Duplicate Result Rate:           0.0%
  Hard Constraint Satisfaction:    100.0%
  SSRF Security Protection Rate:   100.0%
  Qdrant Point Index Integrity:   1,070 Points Intact
--------------------------------------------------------------------------------
OVERALL AUDIT STATUS: ALL 14 TEST GROUPS PASSED 100%
================================================================================
```

### 2. Visual Search Benchmark (`benchmark_visual_search.py`)
```text
BENCHMARK SUMMARY:
  Average Query Latency:        174.86 ms
  Constraint Satisfaction Rate: 100.0%
  Duplicate Candidate Rate:     0.0%
  Qdrant Multi-Stage Pipeline:  Fully Operational
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/pk7745/tailortalk-saree-search.git
cd tailortalk-saree-search

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Variables Setup (`.env`)
Create a `.env` file in the project root:
```ini
GOOGLE_API_KEY="your_google_gemini_api_key"
USE_QDRANT="true"
QDRANT_URL="your_qdrant_cluster_url"
QDRANT_API_KEY="your_qdrant_api_key"
QDRANT_COLLECTION_NAME="tailortalk_sarees"
```

### 4. Run Application
```bash
streamlit run app.py
```

### 5. Run Verification Battery
```bash
# Run Security & Evidence Pipeline Battery
python test_evidence_pipeline.py

# Run Visual Search Benchmark
python benchmark_visual_search.py

# Run Full 14-Group End-to-End Adversarial Audit
python run_adversarial_audit.py
```

---

## 📂 Repository Structure

```text
tailortalk-saree-search/
├── app.py                         # Streamlit UI, chat loop, & responsive card grid
├── agent.py                       # Gemini agent executor, prompt, & tool definitions
├── search_tool.py                 # Multi-stage Qdrant retrieval & hybrid reranking
├── web_verifier.py                # SSRF-protected live webpage evidence verification
├── qdrant_store.py                # Qdrant vector database connection & collection manager
├── embeddings.py                  # Fused 1024d OpenCLIP + 3D HSV embedding pipeline
├── config.py                      # Central configuration, paths, & env variables
├── ingest.py                      # Data ingestion & parquet dataset builder
├── migrate_to_qdrant.py           # Qdrant collection creator & point upsert script
├── enrich_metadata.py             # Merchant metadata scraping & enrichment pipeline
├── data/
│   ├── metadata.parquet           # 1,070-item product metadata (37 columns)
│   ├── saree_index.faiss          # Local FAISS index fallback (1024d)
│   ├── products.csv               # Master catalogue CSV
│   └── images/                    # Local catalogue images
├── test_evidence_pipeline.py      # Security & evidence test battery
├── benchmark_visual_search.py     # Visual search benchmark script
├── run_adversarial_audit.py       # 14-group end-to-end audit suite
├── audit_metadata_qdrant_schema.py# Parquet & Qdrant payload parity audit
├── test_dupe_fixes.py             # Deduplication regression suite
├── test_qdrant_queries.py         # Qdrant query regression suite
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 📄 License & Acknowledgments

Developed as part of the **TailorTalk Saree Search** project. Powered by **Google Gemini**, **Qdrant Vector Database**, **OpenCLIP**, and **Streamlit**.
