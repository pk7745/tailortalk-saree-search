# TailorTalk — Saree Visual Similarity Search Agent

A chat agent that finds visually similar sarees from a ~1,074-item catalogue.
You upload/link an image, chat naturally ("find me something like this"),
and the agent calls a vector-search tool behind the scenes and shows the
closest matches with similarity scores.

**Live app:** https://tailortalk-saree-search.streamlit.app
**Repo:** https://github.com/pk7745/tailortalk-saree-search

---

## 1. Architecture

```
data/products.csv (name, sku, price, image_url, product_link)
        │
        ▼  ingest.py (run once, offline)
   download each image ──► fused embedding ──► FAISS IndexFlatIP
                                                 │
                              data/saree_index.faiss + data/metadata.parquet
                                                 │
                                                 ▼  committed to repo
┌─────────────────────────── app.py (Streamlit) ───────────────────────────┐
│  image upload / URL box  →  session_state.current_image                  │
│  chat input  →  agent.py (LangChain + Gemini, tool-calling)              │
│                     └─ find_similar_sarees(top_k) ─► search_tool.py      │
│                                                         └─ FAISS lookup   │
│  results (name, score, image, price, link) rendered as an image grid    │
└────────────────────────────────────────────────────────────────────────┘
```

The index is built **once, offline**, and the small resulting files
(`saree_index.faiss`, `metadata.parquet`, a few MB total) are committed to
the repo. The deployed app only ever *loads* them and does query-time
embedding of the one image the user provides — that's what makes "works
out of the box, no local setup" possible on a free hosting tier.

## 2. Tech choices & why

| Piece | Choice | Why |
|---|---|---|
| Vector DB | **FAISS** (`IndexFlatIP`) | ~1,074 vectors is tiny — exact brute-force inner-product search is both simplest and gives the *actual* nearest neighbours (no ANN approximation error to debug). Ships as a plain file, so no hosted DB/infra needed for the reviewer. |
| Embedding model | **CLIP ViT-B/32** (`open_clip`, `laion2b_s34b_b79k`) | Strong general visual embedding, small enough to run on Streamlit Cloud's free CPU at query time without a timeout. |
| Agent framework | **LangChain** (`create_tool_calling_agent`) | Matches the brief's suggested stack; gives a clean, typed tool schema (`pydantic`) that the LLM calls into. |
| LLM | **Google Gemini 2.0 Flash** | Free tier, no credit card, reliable native function-calling. Swappable — see `agent.py: _build_llm()`. |
| Frontend | **Streamlit** | `st.chat_message` / `st.chat_input` give a real chat UI with minimal code; sidebar handles image upload/URL. |

## 3. Search quality — the actual hard part

The brief is explicit that a plain "embed the image, cosine-search" pipeline
will look mediocre here, because **every image is the same garment
category** (a saree). Plain CLIP's dominant signal ends up being
"saree-ness," which is identical across the whole catalogue — it doesn't
discriminate hard enough on colour, print, and border work.

**What we did about it — fused embeddings** (`embeddings.py`):

1. **CLIP embedding** (512-d, normalized) — overall visual/semantic similarity.
2. **HSV colour histogram** (8×8×8 = 512-d, normalized) — precise
   colour-combination similarity. HSV (not RGB) because Hue is much more
   robust to the lighting/exposure differences between product photos than
   raw RGB channels are.
3. **Weighted early fusion**: `concat(0.7·CLIP, 0.3·color_hist)`, then
   re-normalized. Cosine similarity on this fused vector approximates a
   weighted blend of "same kind of garment/pattern" and "same actual
   colours" — much closer to how a shopper judges saree similarity.

The 0.7/0.3 split was tuned by manual A/B testing: querying ~15 sarees
spanning different fabric types (Banarasi, Organza, Ajrakh, Linen, Pashmina)
and eyeballing whether the top-5 stayed within the same colour+fabric
family instead of drifting to "any saree that photographs similarly."
Pure CLIP (weight 1.0) visibly returned more color-mismatched results;
pushing colour weight much above ~0.35 started returning same-colour
sarees of a completely different fabric/print. `config.py` exposes both
weights as top-level constants specifically so this can be re-tuned in one
place without touching the pipeline.

**Honest limitation / next step:** this is still whole-image, not
region-aware. A saree's *border* and *pallu* are often the most
distinctive part but get diluted by averaging over the whole photo. The
natural next iteration (documented but not built, given the 3-day window)
is to crop fixed left/right/bottom border regions and embed them
separately, then fuse a third "border similarity" signal in — see
"Future work" below.

## 4. Repo layout

```
config.py         # every tunable constant in one place
embeddings.py      # CLIP + colour histogram + fusion
ingest.py          # offline: CSV -> download -> embed -> FAISS index
search_tool.py      # the one function everything calls: search_similar_sarees()
agent.py            # LangChain tool + Gemini agent wiring
app.py               # Streamlit chat UI
data/products.csv    # the provided catalogue
data/saree_index.faiss   # built by ingest.py, committed to repo
data/metadata.parquet    # built by ingest.py, committed to repo
requirements.txt
```

## 5. Setup — local

```bash
git clone <your-repo-url>
cd tailortalk-saree-search
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Build the index (only needed once, or when products.csv changes).
# Takes a while the first time (~1,074 image downloads + embeddings).
python ingest.py

# If it gets interrupted, just re-run — it resumes and skips already-indexed URLs.
# Quick smoke test on a handful of rows first:
python ingest.py --limit 30

# Set your free Gemini key (https://aistudio.google.com/apikey)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste your key

streamlit run app.py
```

### Verifying it actually works

Before deploying, run the automated self-test suite against the real
indexed catalogue:

```bash
python selftest.py
```

It checks catalogue coverage, that querying a known image returns itself
as the top match (embedding pipeline sanity), that similarity scores
actually spread out across the catalogue (catches a degenerate "everything
looks the same" embedding), the result schema, and — if `GOOGLE_API_KEY`
is set — that the agent correctly calls the search tool for similarity
requests and does not call it for unrelated chat. See `AGENT_BRIEF.md` for
the full build → test → deploy checklist.

## 6. Deployment — Streamlit Community Cloud

1. Push this repo to GitHub, **including** `data/saree_index.faiss` and
   `data/metadata.parquet` (they're small — a few MB — commit them, don't
   gitignore them; only `products.csv` is huge, everything derived is not).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, "New app", pick this repo/branch, main file = `app.py`.
3. In **Advanced settings → Secrets**, paste:
   ```toml
   GOOGLE_API_KEY = "your-free-gemini-key"
   ```
4. Deploy. First boot installs `requirements.txt` (a couple of minutes,
   mostly `torch`/`open_clip`) — subsequent boots are fast since it's cached.
5. Sanity-check: open the app, upload one of the catalogue's own images
   (or any saree photo), and ask "find similar sarees."

## 7. Assumptions & trade-offs

- **Query images are single, roughly product-style saree photos** (one
  garment, not a person wearing 5 outfits, not a heavily cluttered scene).
  The dataset itself is product photography, so this matches the domain.
- **Exact brute-force FAISS search** was chosen over an ANN index (IVF/HNSW)
  because at ~1,074 vectors the speed difference is irrelevant and exact
  search removes a whole class of "why did I get a worse match" debugging.
  This would need to change past roughly 100k+ items.
- **CPU-only inference.** Streamlit Community Cloud's free tier has no
  GPU; `open_clip`'s ViT-B/32 on CPU embeds a single query image in roughly
  1-2 seconds, which is fine for a chat UI.
- Some product image URLs may 404 or time out during ingestion (dead
  links, host hiccups) — `ingest.py` logs these to
  `data/failed_downloads.csv` instead of crashing the whole run, and the
  index is simply built from whatever downloaded successfully.
- Currency/price fields are passed through from the CSV as-is (no
  formatting/validation) since they're display-only, not part of search.

## 8. Future work (given more time)

- Region-aware border/pallu crop embeddings, fused as a third signal.
- A learned re-ranker (small MLP or even a fine-tuned CLIP head) trained
  on click/preference data once real user feedback exists.
- Swap `IndexFlatIP` for `IndexHNSWFlat` if the catalogue grows large.
- Multi-image queries ("find something between these two").
